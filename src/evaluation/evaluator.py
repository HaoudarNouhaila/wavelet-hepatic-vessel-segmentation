"""Évaluation de plusieurs modèles sur les mêmes volumes et conditions.

Chaque volume est rechargé depuis le NIfTI d'origine, éventuellement
dégradé (étude de la résolution), puis normalisé avec les statistiques de
l'entraînement. Tous les modèles reçoivent exactement la même entrée. Les
métriques sont calculées par volume, sur la grille d'origine, avec
l'espacement physique des voxels.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.msd import load_case
from src.evaluation.inference import TTA_FLIPS_2D, predict_volume, probs_to_labels, probs_to_original_grid
from src.evaluation.metrics import all_metrics
from src.evaluation.vessel_analysis import centerline_recall_by_radius, component_analysis
from src.preprocessing.degradation import degrade
from src.preprocessing.intensity import background_value, normalize_ct
from src.preprocessing.pipeline import resample_inplane
from src.training.trainer import build_wavelet, load_model_from_checkpoint
from src.utils import resolve

CLASS_NAMES = {1: "vessel", 2: "tumour"}


def load_run(run_dir: Path, device="cpu", checkpoint: str = "best.pt", tta_flips: int = 1,
             overlap: float = 0.25, amp: bool = False) -> dict:
    """Charge un modèle entraîné (2D ou 3D) et sa fonction de prédiction volume -> probabilités.

    `tta_flips` (1, 2 ou 4 orientations), `overlap` et `amp` ne servent qu'à l'étape
    d'optimisation ; leurs valeurs par défaut reproduisent l'inférence des premières expériences.
    """
    from src.models.unet3d import MODEL_TYPES_3D, TTA_FLIPS_3D, predict_volume_3d

    ckpt = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    stats = json.loads(resolve(cfg["data"]["stats_file"]).read_text(encoding="utf-8"))
    fill = background_value(stats)
    if cfg["model"].get("type") in MODEL_TYPES_3D:
        from src.training.trainer3d import load_model_3d
        model, _ = load_model_3d(run_dir / checkpoint, device)
        roi = cfg["training"]["roi_size"]
        wavelet = None
        flips = TTA_FLIPS_3D[: max(1, tta_flips)]
        predict = lambda volume: predict_volume_3d(model, volume, roi, overlap=overlap, device=device,  # noqa: E731
                                                   amp=amp, flips=flips)
    else:
        model, _ = load_model_from_checkpoint(run_dir / checkpoint, device)
        wavelet = build_wavelet(cfg["model"], resolve(cfg["data"]["wavelet_stats_file"]))
        flips = TTA_FLIPS_2D[: max(1, tta_flips)]
        predict = lambda volume: predict_volume(model, volume, wavelet, fill, device=device,  # noqa: E731
                                                flips=flips)
    return {
        "name": run_dir.name, "model": model, "cfg": cfg, "stats": stats, "wavelet": wavelet, "fill": fill,
        "epoch": ckpt["epoch"], "model_type": cfg["experiment"]["name"], "seed": cfg["experiment"]["seed"],
        "predict": predict,
    }


def prepare_volume(image_hu: np.ndarray, spacing, stats: dict, condition: dict, target_inplane, seed: int) -> np.ndarray:
    degraded = degrade(image_hu, spacing, condition, seed=seed)
    if target_inplane:
        degraded, _ = resample_inplane(degraded, spacing, target_inplane, order=1)
    return normalize_ct(degraded, stats), degraded


def case_metrics(pred: np.ndarray, label: np.ndarray, spacing) -> list[dict]:
    rows = []
    for cls, name in CLASS_NAMES.items():
        p, g = pred == cls, label == cls
        row = {"class": name, **all_metrics(p, g, spacing, with_cldice=(cls == 1))}
        row["gt_voxels"], row["pred_voxels"] = int(g.sum()), int(p.sum())
        if cls == 1:
            row.update(centerline_recall_by_radius(p, g, spacing))
            row.update(component_analysis(p, g))
        rows.append(row)
    return rows


def evaluate(runs: list[dict], raw_dir: Path, case_ids: list[str], conditions: dict[str, dict],
             target_inplane=None, device="cpu", predictions_dir: Path | None = None,
             existing: pd.DataFrame | None = None, out_csv: Path | None = None, log=print) -> pd.DataFrame:
    """Évalue tous les modèles sur tous les cas et conditions.

    Les combinaisons déjà présentes dans `existing` sont sautées, ce qui
    permet de reprendre une évaluation interrompue. Avec `out_csv`, les
    lignes sont ajoutées au fichier après chaque volume.
    """
    done = set()
    if existing is not None and len(existing):
        done = set(zip(existing["run"], existing["condition"], existing["case_id"]))
    rows = []
    stats = runs[0]["stats"]
    if any(r["stats"] != stats for r in runs):
        raise ValueError("Les modèles comparés doivent partager la même normalisation")
    for cond_name, condition in conditions.items():
        for case_index, cid in enumerate(case_ids):
            todo = [r for r in runs if (r["name"], cond_name, cid) not in done]
            if not todo:
                continue
            case = load_case(raw_dir, cid)
            volume, _ = prepare_volume(case.image, case.spacing, stats, condition, target_inplane, seed=case_index)
            first_row = len(rows)
            for run in todo:
                t0 = time.time()
                probs = run["predict"](volume)
                pred = probs_to_labels(probs_to_original_grid(probs, case.label.shape))
                t_pred = time.time() - t0
                for row in case_metrics(pred, case.label, case.spacing):
                    rows.append({"run": run["name"], "model": run["model_type"], "seed": run["seed"],
                                 "condition": cond_name, "case_id": cid, "spacing_z": case.spacing[0],
                                 "spacing_xy": case.spacing[1], "n_slices": case.label.shape[0],
                                 "inference_s": t_pred, **row})
                if predictions_dir is not None:
                    out = predictions_dir / run["name"] / cond_name
                    out.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(out / f"{cid}.npz", pred=pred)
                vessel = rows[-2]
                log(f"[{cond_name}] {cid} {run['name']}: Dice vaisseaux {vessel['dice']:.3f}, "
                    f"clDice {vessel['cldice']:.3f}, HD95 {vessel['hd95_mm']:.1f} mm ({t_pred:.0f} s)")
            if out_csv is not None:
                out_csv = Path(out_csv)
                out_csv.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows[first_row:]).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
    return pd.DataFrame(rows)
