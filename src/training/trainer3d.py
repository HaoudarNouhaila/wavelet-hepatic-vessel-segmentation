"""Entraînement du U-Net 3D (MONAI) par patches, mêmes données et même perte qu'en 2D."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.evaluation.metrics import overlap_metrics
from src.models.unet3d import PatchDataset3D, build_unet3d, predict_volume_3d
from src.preprocessing.intensity import background_value
from src.preprocessing.pipeline import load_processed
from src.training.losses import build_loss
from src.training.trainer import EarlyStopping
from src.utils import get_device, resolve, set_seed


def save_checkpoint_3d(path: Path, model, epoch: int, best: float, cfg: dict) -> None:
    torch.save({"model": model.state_dict(), "epoch": epoch, "best_metric": best, "config": cfg}, path)


def load_unet3d(path: Path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_unet3d(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


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
                             True, exp["seed"])
    loader = DataLoader(dataset, batch_size=tr["batch_size"], shuffle=False, num_workers=0, drop_last=True)
    val_cases = [load_processed(processed / cid) for cid in val_ids]
    model = build_unet3d(model_cfg).to(device)
    criterion = build_loss(cfg.get("loss", {}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr["max_epochs"], eta_min=tr["lr"] * 0.01)
    stopper = EarlyStopping(tr["early_stopping_patience"], min_epochs=tr.get("min_epochs", 0))
    n_params = sum(p.numel() for p in model.parameters())
    log(f"U-Net 3D : {n_params:,} paramètres, {len(train_ids)} volumes d'entraînement, device={device}")

    history, best_epoch, t_start = [], 0, time.time()
    for epoch in range(1, tr["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        dataset.set_epoch(epoch)
        running = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch["image"].to(device)), batch["label"].to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        row = {"epoch": epoch, "lr": lr, "train_loss": running / len(loader), "train_time_s": time.time() - t0}
        if epoch % tr.get("val_every", 1) == 0 or epoch == tr["max_epochs"]:
            dice_v, dice_t = [], []
            for case in val_cases:
                probs = predict_volume_3d(model, case["image"], tr["roi_size"], device=device)
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
    summary = {"run": run_dir.name, "best_epoch": best_epoch, "best_val_dice_vessel": stopper.best,
               "epochs_run": len(history), "total_time_min": (time.time() - t_start) / 60, "parameters": n_params}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
