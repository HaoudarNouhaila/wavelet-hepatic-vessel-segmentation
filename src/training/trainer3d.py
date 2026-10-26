"""Entraînement des réseaux 3D (MONAI) par patches, mêmes données et même perte qu'en 2D.

Options utiles sur GPU : précision mixte (`amp`), écrêtage des gradients (`grad_clip`) et
décroissance cosinus du taux d'apprentissage en fonction du temps écoulé
(`scheduler: cosine_time`), qui garantit une décroissance complète dans la limite `max_minutes`.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.evaluation.metrics import overlap_metrics
from src.models.unet3d import PatchDataset3D, build_model_3d, predict_volume_3d
from src.preprocessing.intensity import background_value
from src.preprocessing.pipeline import load_processed
from src.training.losses import build_loss
from src.training.trainer import EarlyStopping
from src.utils import get_device, resolve, set_seed


def save_checkpoint_3d(path: Path, model, epoch: int, best: float, cfg: dict) -> None:
    torch.save({"model": model.state_dict(), "epoch": epoch, "best_metric": best, "config": cfg}, path)


def load_model_3d(path: Path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model_3d(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


load_unet3d = load_model_3d  # nom utilisé par les premières versions


def cosine_time_lr(lr0: float, lr_min: float, elapsed_min: float, total_min: float) -> float:
    """Taux d'apprentissage cosinus en fonction de la fraction du temps écoulé."""
    progress = min(max(elapsed_min / total_min, 0.0), 1.0)
    return lr_min + 0.5 * (lr0 - lr_min) * (1 + math.cos(math.pi * progress))


def train_3d(cfg: dict, run_dir: Path, log=print) -> dict:
    exp, data_cfg, model_cfg, tr = cfg["experiment"], cfg["data"], cfg["model"], cfg["training"]
    set_seed(exp["seed"])
    torch.set_num_threads(tr.get("num_threads", torch.get_num_threads()))
    device = get_device(tr.get("device", "auto"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    split = json.loads(resolve(data_cfg["split_file"]).read_text(encoding="utf-8"))[data_cfg.get("split_key", "subset")]
    stats = json.loads(resolve(data_cfg["stats_file"]).read_text(encoding="utf-8"))
    fill = background_value(stats)
    processed = resolve(data_cfg["processed_dir"])
    train_ids = split["train"][: tr.get("max_train_cases")]
    val_ids = split["val"][: tr.get("max_val_cases")]

    dataset = PatchDataset3D(processed, train_ids, tr["patch_size"], tr["samples_per_epoch"], tr["fg_prob"], fill,
                             True, exp["seed"], tr.get("noise_prob", 0.0), tr.get("noise_std", 0.05))
    loader = DataLoader(dataset, batch_size=tr["batch_size"], shuffle=False, num_workers=tr.get("num_workers", 0),
                        pin_memory=device.type == "cuda", drop_last=True)
    val_cases = [load_processed(processed / cid) for cid in val_ids]
    model = build_model_3d(model_cfg).to(device)
    criterion = build_loss(cfg.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    lr_min = tr["lr"] * tr.get("lr_min_factor", 0.01)
    time_schedule = tr.get("scheduler", "cosine") == "cosine_time"
    if time_schedule and not tr.get("max_minutes"):
        raise ValueError("scheduler cosine_time demande training.max_minutes")
    scheduler = None if time_schedule else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tr["max_epochs"], eta_min=lr_min)
    amp = bool(tr.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    stopper = EarlyStopping(tr["early_stopping_patience"], min_epochs=tr.get("min_epochs", 0))
    n_params = sum(p.numel() for p in model.parameters())
    log(f"{model_cfg.get('type', 'unet3d')} : {n_params:,} paramètres, {len(train_ids)} volumes d'entraînement, "
        f"device={device}, amp={amp}")

    history, best_epoch, t_start = [], 0, time.time()
    stopped_by_time = False
    for epoch in range(1, tr["max_epochs"] + 1):
        t0 = time.time()
        if time_schedule:
            for group in optimizer.param_groups:
                group["lr"] = cosine_time_lr(tr["lr"], lr_min, (t0 - t_start) / 60, tr["max_minutes"])
        model.train()
        dataset.set_epoch(epoch)
        running = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = model(batch["image"].to(device, non_blocking=True))
            loss = criterion(logits.float(), batch["label"].to(device, non_blocking=True))
            scaler.scale(loss).backward()
            if tr.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        row = {"epoch": epoch, "lr": lr, "train_loss": running / len(loader), "train_time_s": time.time() - t0}
        if epoch % tr.get("val_every", 1) == 0 or epoch == tr["max_epochs"]:
            dice_v, dice_t = [], []
            for case in val_cases:
                probs = predict_volume_3d(model, case["image"], tr["roi_size"], overlap=tr.get("val_overlap", 0.25),
                                          device=device, amp=amp)
                pred = probs.argmax(0)
                lab = np.asarray(case["label"])
                dice_v.append(overlap_metrics(pred == 1, lab == 1)["dice"])
                dice_t.append(overlap_metrics(pred == 2, lab == 2)["dice"])
            row.update({"val_dice_vessel": float(np.nanmean(dice_v)), "val_dice_tumour": float(np.nanmean(dice_t))})
            if stopper.step(row["val_dice_vessel"]):
                best_epoch = epoch
                save_checkpoint_3d(run_dir / "best.pt", model, epoch, stopper.best, cfg)
        row["epoch_time_s"] = time.time() - t0
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        save_checkpoint_3d(run_dir / "last.pt", model, epoch, stopper.best, cfg)
        log(f"époque {epoch:3d} | loss {row['train_loss']:.4f} | Dice vaisseaux val "
            f"{row.get('val_dice_vessel', float('nan')):.4f} | {row['epoch_time_s']:.0f} s")
        if stopper.should_stop:
            log(f"Early stopping à l'époque {epoch}")
            break
        if tr.get("max_minutes") and (time.time() - t_start) / 60 > tr["max_minutes"] and epoch < tr["max_epochs"]:
            log(f"Limite de temps atteinte à l'époque {epoch}")
            stopped_by_time = True
            break
    summary = {"run": run_dir.name, "best_epoch": best_epoch, "best_val_dice_vessel": stopper.best,
               "epochs_run": len(history), "total_time_min": (time.time() - t_start) / 60, "parameters": n_params,
               "stopped_by_time_limit": stopped_by_time}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
