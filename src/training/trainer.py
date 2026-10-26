"""Boucle d'entraînement des modèles 2D.

Choix principaux :
- AdamW + décroissance cosinus du taux d'apprentissage ;
- validation à chaque époque sur les volumes complets de validation ;
- sélection du meilleur modèle et early stopping sur le Dice moyen des
  vaisseaux en validation (l'ensemble de test n'est jamais utilisé ici) ;
- historique enregistré dans history.csv, checkpoints best.pt et last.pt.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset2d import PatchDataset2D, injections_from_batch
from src.evaluation.inference import iterate_volume_logits
from src.evaluation.metrics import overlap_metrics
from src.models.unet2d import build_model_2d, count_parameters
from src.preprocessing.intensity import background_value
from src.preprocessing.pipeline import load_processed
from src.training.losses import build_loss, one_hot
from src.utils import get_device, resolve, set_seed
from src.wavelets.transforms import WaveletFeatures


class EarlyStopping:
    """Arrête l'entraînement si la métrique ne s'améliore plus pendant `patience` époques.

    L'arrêt n'est jamais déclenché avant `min_epochs` évaluations, pour ne pas
    interrompre un modèle qui démarre plus lentement que les autres.
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, min_epochs: int = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best = -np.inf
        self.bad_epochs = 0
        self.steps = 0

    def step(self, value: float) -> bool:
        """Retourne True si la valeur est un nouveau meilleur score."""
        self.steps += 1
        if value > self.best + self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.steps >= self.min_epochs and self.bad_epochs >= self.patience


def build_wavelet(model_cfg: dict, stats_file: Path | None) -> WaveletFeatures | None:
    mode = model_cfg.get("wavelet_mode", "none")
    if mode == "none":
        return None
    wavelet = model_cfg.get("wavelet", "haar")
    levels = model_cfg.get("wavelet_levels", 2)
    key = f"{mode}_{wavelet}_L{levels if mode == 'dwt_multilevel' else 1}"
    all_stats = json.loads(Path(stats_file).read_text(encoding="utf-8"))
    if key not in all_stats:
        raise KeyError(f"Statistiques ondelettes '{key}' absentes de {stats_file} (relancer prepare_data.py)")
    return WaveletFeatures(mode, wavelet, levels, stats=all_stats[key])


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_metric: float, cfg: dict) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "best_metric": best_metric,
        "config": cfg,
    }, path)


def load_model_from_checkpoint(path: Path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model_2d(ckpt["config"]["model"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.inference_mode()
def validate(model, val_cases: list[dict], wavelet, fill_value: float, loss_cfg: dict, num_classes: int,
             device, batch_size: int = 8, slice_stride: int = 1) -> dict:
    """Perte et Dice par volume sur les volumes de validation.

    La Dice loss est calculée sur le volume entier (sommes accumulées coupe
    par coupe), ce qui évite qu'elle soit dominée par les coupes sans
    structure annotée. Avec `slice_stride` > 1, une coupe sur `slice_stride`
    est utilisée (sous-échantillonnage régulier, indépendant des labels),
    pour réduire le coût de la validation sur CPU.
    """
    losses, dice_vessel, dice_tumour = [], [], []
    for case in val_cases:
        volume = case["image"][::slice_stride]
        label = np.asarray(case["label"][::slice_stride])
        inter = torch.zeros(num_classes - 1)
        denom = torch.zeros(num_classes - 1)
        ce_sum, n_vox = 0.0, 0
        pred = np.zeros(label.shape, dtype=np.uint8)
        for idx, logits in iterate_volume_logits(model, volume, wavelet, fill_value, batch_size, device):
            target = torch.from_numpy(label[idx.start : idx.stop].astype(np.int64)).to(device)
            ce_sum += F.cross_entropy(logits, target, reduction="sum").item()
            n_vox += target.numel()
            probs = logits.softmax(1)[:, 1:]
            tgt = one_hot(target, num_classes)[:, 1:]
            inter += (probs * tgt).sum((0, 2, 3)).cpu()
            denom += (probs.sum((0, 2, 3)) + tgt.sum((0, 2, 3))).cpu()
            pred[idx.start : idx.stop] = logits.argmax(1).cpu().numpy()
        soft_dice = ((2 * inter + 1e-5) / (denom + 1e-5)).mean().item()
        losses.append(loss_cfg.get("dice_weight", 1.0) * (1 - soft_dice) + loss_cfg.get("ce_weight", 1.0) * ce_sum / n_vox)
        dice_vessel.append(overlap_metrics(pred == 1, label == 1)["dice"])
        dice_tumour.append(overlap_metrics(pred == 2, label == 2)["dice"])
    return {
        "val_loss": float(np.mean(losses)),
        "val_dice_vessel": float(np.nanmean(dice_vessel)),
        "val_dice_tumour": float(np.nanmean(dice_tumour)) if not np.all(np.isnan(dice_tumour)) else float("nan"),
    }


def train(cfg: dict, run_dir: Path, log=print) -> dict:
    """Entraîne un modèle 2D selon la configuration et retourne un résumé."""
    exp, data_cfg, model_cfg, tr = cfg["experiment"], cfg["data"], cfg["model"], cfg["training"]
    seed = exp["seed"]
    set_seed(seed)
    torch.set_num_threads(tr.get("num_threads", torch.get_num_threads()))
    device = get_device(tr.get("device", "auto"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    split = json.loads(resolve(data_cfg["split_file"]).read_text(encoding="utf-8"))[data_cfg.get("split_key", "subset")]
    stats = json.loads(resolve(data_cfg["stats_file"]).read_text(encoding="utf-8"))
    fill_value = background_value(stats)
    wavelet = build_wavelet(model_cfg, resolve(data_cfg["wavelet_stats_file"]))
    processed = resolve(data_cfg["processed_dir"])
    train_ids = split["train"][: tr.get("max_train_cases")]
    val_ids = split["val"][: tr.get("max_val_cases")]

    train_set = PatchDataset2D(processed, train_ids, tr["patch_size"], tr["samples_per_epoch"], tr["fg_prob"],
                               fill_value, wavelet, tr.get("augmentation"), seed)
    # Pas de persistent_workers : les processus de chargement sont recréés à chaque époque et
    # reçoivent ainsi le numéro d'époque à jour (sinon ils tireraient toujours les mêmes patches).
    loader = DataLoader(train_set, batch_size=tr["batch_size"], shuffle=False, num_workers=tr.get("num_workers", 0),
                        pin_memory=device.type == "cuda", drop_last=True)
    val_cases = [load_processed(processed / cid) for cid in val_ids]

    model = build_model_2d(model_cfg).to(device)
    criterion = build_loss(cfg.get("loss", {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tr["max_epochs"], eta_min=tr["lr"] * 0.01)
    stopper = EarlyStopping(tr["early_stopping_patience"], min_epochs=tr.get("min_epochs", 0))
    log(f"Modèle {model_cfg.get('wavelet_mode', 'none')} : {count_parameters(model):,} paramètres, "
        f"{len(train_ids)} volumes d'entraînement, {len(val_ids)} de validation, device={device}")

    history = []
    best_epoch = 0
    stopped_by_time = False
    start_time = time.time()
    for epoch in range(1, tr["max_epochs"] + 1):
        t0 = time.time()
        model.train()
        train_set.set_epoch(epoch)
        running, n_batches = 0.0, 0
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, injections_from_batch(batch, device)), y)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        t_train = time.time() - t0
        val = validate(model, val_cases, wavelet, fill_value, cfg.get("loss", {}), model_cfg.get("num_classes", 3),
                       device, tr.get("val_batch_size", 8), tr.get("val_slice_stride", 1))
        improved = stopper.step(val["val_dice_vessel"])
        row = {"epoch": epoch, "lr": lr, "train_loss": running / max(n_batches, 1), **val,
               "train_time_s": t_train, "epoch_time_s": time.time() - t0}
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if improved:
            best_epoch = epoch
            save_checkpoint(run_dir / "best.pt", model, None, None, epoch, stopper.best, cfg)
        save_checkpoint(run_dir / "last.pt", model, optimizer, scheduler, epoch, stopper.best, cfg)
        log(f"époque {epoch:3d} | loss {row['train_loss']:.4f} | val loss {val['val_loss']:.4f} | "
            f"Dice vaisseaux {val['val_dice_vessel']:.4f} | Dice tumeurs {val['val_dice_tumour']:.4f} | "
            f"{row['epoch_time_s']:.0f} s{' *' if improved else ''}")
        if stopper.should_stop:
            log(f"Early stopping à l'époque {epoch} (meilleure époque : {best_epoch})")
            break
        if tr.get("max_minutes") and (time.time() - start_time) / 60 > tr["max_minutes"] \
                and epoch < tr["max_epochs"]:
            log(f"Limite de temps atteinte à l'époque {epoch} (meilleure époque : {best_epoch})")
            stopped_by_time = True
            break

    summary = {
        "run": run_dir.name,
        "best_epoch": best_epoch,
        "best_val_dice_vessel": stopper.best,
        "epochs_run": len(history),
        "total_time_min": (time.time() - start_time) / 60,
        "parameters": count_parameters(model),
        "stopped_by_time_limit": stopped_by_time,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
