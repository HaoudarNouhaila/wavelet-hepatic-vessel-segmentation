"""Optimisation des métriques sur la validation, puis évaluation finale sur le test.

Étapes (--step) :
  probs   probabilités de chaque modèle (avec augmentation au test), mises en cache par volume ;
  search  choix, sur les 45 volumes de validation, de l'ensemble de modèles, du seuil de
          probabilité des vaisseaux et de la taille minimale des composantes conservées ;
  final   évaluation complète sur le test de la configuration retenue et de références,
          puis robustesse de la configuration retenue aux images dégradées.

L'ensemble de test n'intervient qu'à l'étape final, une fois les choix figés sur la validation.

Exemples :
    python scripts/optimize.py --step probs --split val
    python scripts/optimize.py --step search
    python scripts/optimize.py --step probs --split test --runs dynunet unet3d
    python scripts/optimize.py --step final
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.msd import load_case
from src.data.splits import load_split
from src.utils import get_device, load_config, resolve

CFG_DEFAULT = "configs/kaggle/optimisation.yaml"


def to_uint8(p: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(p, dtype=np.float32) * 255), 0, 255).astype(np.uint8)


def cache_file(cfg: dict, split: str, run: str, case_id: str) -> Path:
    return resolve(cfg["cache_dir"]) / split / run / f"{case_id}.npz"


def load_member_probs(cfg: dict, split: str, run: str, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(cache_file(cfg, split, run, case_id))
    return data["vessel"].astype(np.float32) / 255, data["tumour"].astype(np.float32) / 255


def ensemble_probs(cfg: dict, split: str, members: dict[str, float], case_id: str):
    """Moyenne pondérée des probabilités des membres, accumulée membre par membre (mémoire)."""
    total = float(sum(members.values()))
    p_vessel = p_tumour = None
    for run, weight in members.items():
        v, t = load_member_probs(cfg, split, run, case_id)
        if p_vessel is None:
            p_vessel, p_tumour = np.zeros_like(v), np.zeros_like(t)
        p_vessel += v * (weight / total)
        p_tumour += t * (weight / total)
    return p_vessel, p_tumour


def load_gt(cfg: dict, split: str, case_id: str):
    data = np.load(resolve(cfg["cache_dir"]) / split / "_gt" / f"{case_id}.npz")
    return data["label"], tuple(float(s) for s in data["spacing"])


# Étape probs -----------------------------------------------------------------------------------

def step_probs(cfg: dict, data_cfg: dict, split: str, run_names: list[str] | None) -> None:
    from src.evaluation.evaluator import load_run, prepare_volume
    from src.evaluation.inference import probs_to_original_grid

    device = get_device(cfg.get("device", "auto"))
    case_ids = load_split(resolve(data_cfg["split"]["file"]), "subset")[split]
    raw_dir = resolve(data_cfg["dataset"]["raw_dir"])
    target = data_cfg["preprocessing"].get("target_inplane_spacing")
    gt_dir = resolve(cfg["cache_dir"]) / split / "_gt"
    gt_dir.mkdir(parents=True, exist_ok=True)
    names = run_names or list(cfg["runs"])
    for name in names:
        spec = cfg["runs"][name]
        run_dir = resolve(spec["dir"])
        if not (run_dir / "best.pt").exists():
            print(f"{name} : pas de checkpoint dans {run_dir}, ignoré", flush=True)
            continue
        run = load_run(run_dir, device, tta_flips=spec.get("tta", 1), overlap=spec.get("overlap", 0.25),
                       amp=spec.get("amp", False))
        t0 = time.time()
        for cid in case_ids:
            out = cache_file(cfg, split, name, cid)
            if out.exists():
                continue
            case = load_case(raw_dir, cid)
            gt_file = gt_dir / f"{cid}.npz"
            if not gt_file.exists():
                np.savez_compressed(gt_file, label=case.label, spacing=np.asarray(case.spacing))
            volume, _ = prepare_volume(case.image, case.spacing, run["stats"], {}, target, seed=0)
            probs = probs_to_original_grid(run["predict"](volume), case.label.shape)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, vessel=to_uint8(probs[1]), tumour=to_uint8(probs[2]))
        print(f"{name} : probabilités {split} en {(time.time() - t0) / 60:.1f} min", flush=True)


# Étape search ----------------------------------------------------------------------------------

def _search_case(args) -> list[dict]:
    from src.evaluation.postprocess import dice_grid

    cfg, case_id, ensembles = args
    gt, _ = load_gt(cfg, "val", case_id)
    rows = []
    for name, members in ensembles.items():
        p_vessel, _ = ensemble_probs(cfg, "val", members, case_id)
        grid = dice_grid(p_vessel, gt == 1, cfg["thresholds"], cfg["min_sizes"])
        rows += [{"ensemble": name, "threshold": t, "min_voxels": m, "case_id": case_id, "dice": d}
                 for (t, m), d in grid.items()]
    return rows


def available_ensembles(cfg: dict, split: str, case_ids: list[str]) -> dict[str, dict]:
    out = {}
    for name, members in cfg["ensembles"].items():
        missing = [r for r in members if not all(cache_file(cfg, split, r, c).exists() for c in case_ids)]
        if missing:
            print(f"ensemble {name} ignoré : probabilités manquantes pour {missing}", flush=True)
        else:
            out[name] = members
    return out


def step_search(cfg: dict, data_cfg: dict, workers: int) -> None:
    case_ids = load_split(resolve(data_cfg["split"]["file"]), "subset")["val"]
    ensembles = available_ensembles(cfg, "val", case_ids)
    jobs = [(cfg, cid, ensembles) for cid in case_ids]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_search_case, jobs))
    else:
        parts = [_search_case(j) for j in jobs]
    per_case = pd.DataFrame([r for part in parts for r in part])
    table = per_case.groupby(["ensemble", "threshold", "min_voxels"])["dice"].agg(["mean", "median", "std"])
    table = table.reset_index().sort_values("mean", ascending=False)
    metrics_dir = resolve(cfg["results_dir"]) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / "val_search.csv", index=False)
    best = table.iloc[0]
    per_ensemble = table.groupby("ensemble").head(1)
    default = table[(table["threshold"] == 0.5) & (table["min_voxels"] == 0)]
    selection = {
        "critere": "Dice moyen des vaisseaux sur les 45 volumes de validation",
        "retenu": {"ensemble": best["ensemble"], "threshold": float(best["threshold"]),
                   "min_voxels": int(best["min_voxels"]), "val_dice": float(best["mean"])},
        "meilleur_par_ensemble": per_ensemble[["ensemble", "threshold", "min_voxels", "mean"]].to_dict("records"),
        "sans_post_traitement": default[["ensemble", "mean"]].to_dict("records"),
        "ensembles": ensembles,
    }
    (metrics_dir / "selection.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8")
    print(per_ensemble.round(4).to_string(index=False))
    print("Retenu :", selection["retenu"], flush=True)


# Étape final -----------------------------------------------------------------------------------

def final_configs(cfg: dict, selection: dict) -> list[dict]:
    """Configurations évaluées sur le test : la configuration retenue et, pour chaque référence,
    la version brute (seuil 0,5, sans filtrage) et la version post-traitée réglée sur la validation."""
    best_pp = {r["ensemble"]: r for r in selection["meilleur_par_ensemble"]}
    chosen = selection["retenu"]["ensemble"]
    names = [chosen] + [r for r in cfg["final"]["references"] if r != chosen and r in best_pp]
    out = []
    for name in names:
        members = selection["ensembles"][name]
        out.append({"config": f"{name} (brut)", "ensemble": name, "members": members,
                    "threshold": 0.5, "min_voxels": 0})
        out.append({"config": f"{name} (post-traité)", "ensemble": name, "members": members,
                    "threshold": float(best_pp[name]["threshold"]), "min_voxels": int(best_pp[name]["min_voxels"])})
    return out


def _final_case(args) -> list[dict]:
    from src.evaluation.evaluator import case_metrics
    from src.evaluation.postprocess import labels_from_probs

    cfg, case_id, configs = args
    gt, spacing = load_gt(cfg, "test", case_id)
    rows = []
    cache: dict[str, tuple] = {}
    for c in configs:
        if c["ensemble"] not in cache:
            cache[c["ensemble"]] = ensemble_probs(cfg, "test", c["members"], case_id)
        p_vessel, p_tumour = cache[c["ensemble"]]
        pred = labels_from_probs(p_vessel, p_tumour, c["threshold"], c["min_voxels"])
        for row in case_metrics(pred, gt, spacing):
            rows.append({"config": c["config"], "ensemble": c["ensemble"], "threshold": c["threshold"],
                         "min_voxels": c["min_voxels"], "condition": "original", "case_id": case_id, **row})
    return rows


def step_final(cfg: dict, data_cfg: dict, workers: int, degradations: bool) -> None:
    metrics_dir = resolve(cfg["results_dir"]) / "metrics"
    selection = json.loads((metrics_dir / "selection.json").read_text(encoding="utf-8"))
    case_ids = load_split(resolve(data_cfg["split"]["file"]), "subset")["test"]
    configs = final_configs(cfg, selection)
    needed = sorted({r for c in configs for r in c["members"]})
    missing = [r for r in needed if not all(cache_file(cfg, "test", r, c).exists() for c in case_ids)]
    if missing:
        raise RuntimeError(f"Probabilités de test manquantes pour {missing} : lancer --step probs --split test")
    jobs = [(cfg, cid, configs) for cid in case_ids]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_final_case, jobs))
    else:
        parts = [_final_case(j) for j in jobs]
    table = pd.DataFrame([r for part in parts for r in part])
    table.to_csv(metrics_dir / "test_final.csv", index=False)
    summary = table.groupby(["class", "config"])[["dice", "cldice", "hd95_mm", "precision", "recall"]].mean()
    summary.to_csv(metrics_dir / "test_final_summary.csv")
    print(summary.round(4).to_string(), flush=True)
    if degradations:
        step_degradations(cfg, data_cfg, selection, case_ids)


def step_degradations(cfg: dict, data_cfg: dict, selection: dict, case_ids: list[str]) -> None:
    """Robustesse de la configuration retenue (probabilités recalculées sur les images dégradées)."""
    from src.evaluation.evaluator import case_metrics, load_run, prepare_volume
    from src.evaluation.inference import probs_to_original_grid
    from src.evaluation.postprocess import labels_from_probs

    eval_cfg = load_config(resolve(cfg["evaluation_config"]))
    best = selection["retenu"]
    members = selection["ensembles"][best["ensemble"]]
    device = get_device(cfg.get("device", "auto"))
    runs = {}
    for name in members:
        spec = cfg["runs"][name]
        runs[name] = load_run(resolve(spec["dir"]), device, tta_flips=spec.get("tta", 1),
                              overlap=spec.get("overlap", 0.25), amp=spec.get("amp", False))
    raw_dir = resolve(data_cfg["dataset"]["raw_dir"])
    target = data_cfg["preprocessing"].get("target_inplane_spacing")
    out_csv = resolve(cfg["results_dir"]) / "metrics" / "test_final_degradations.csv"
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = set(zip(prev["condition"], prev["case_id"]))
    for cond in cfg["final"]["degradations"]:
        condition = eval_cfg["conditions"][cond]
        for index, cid in enumerate(case_ids):
            if (cond, cid) in done:
                continue
            case = load_case(raw_dir, cid)
            total = float(sum(members.values()))
            p_vessel = np.zeros(case.label.shape, np.float32)
            p_tumour = np.zeros(case.label.shape, np.float32)
            for name, run in runs.items():
                volume, _ = prepare_volume(case.image, case.spacing, run["stats"], condition, target, seed=index)
                probs = probs_to_original_grid(run["predict"](volume), case.label.shape)
                p_vessel += probs[1].astype(np.float32) * (members[name] / total)
                p_tumour += probs[2].astype(np.float32) * (members[name] / total)
            pred = labels_from_probs(p_vessel, p_tumour, best["threshold"], best["min_voxels"])
            rows = [{"config": f"{best['ensemble']} (post-traité)", "condition": cond, "case_id": cid, **row}
                    for row in case_metrics(pred, case.label, case.spacing)]
            pd.DataFrame(rows).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"dégradation {cond} terminée", flush=True)


def step_figures(cfg: dict, data_cfg: dict) -> None:
    """Carte du Dice de validation (seuil x taille minimale), comparaison sur le test et exemple."""
    import matplotlib.pyplot as plt

    from src.evaluation.postprocess import labels_from_probs
    from src.visualization import plots

    metrics_dir = resolve(cfg["results_dir"]) / "metrics"
    fig_dir = resolve(cfg["results_dir"]) / "figures"
    selection = json.loads((metrics_dir / "selection.json").read_text(encoding="utf-8"))
    best = selection["retenu"]
    search = pd.read_csv(metrics_dir / "val_search.csv")
    grid = search[search["ensemble"] == best["ensemble"]].pivot(index="min_voxels", columns="threshold", values="mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(grid.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(grid.shape[1]), [f"{t:.2f}" for t in grid.columns])
    ax.set_yticks(range(grid.shape[0]), [str(m) for m in grid.index])
    for (i, j), value in np.ndenumerate(grid.to_numpy()):
        ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7,
                color="white" if value > np.nanmean(grid.to_numpy()) else plots.INK)
    ax.set(xlabel="seuil de probabilité des vaisseaux", ylabel="taille minimale des composantes (voxels)",
           title=f"Dice moyen en validation, ensemble « {best['ensemble']} »")
    ax.grid(False)
    fig.colorbar(image, ax=ax, label="Dice")
    plots.save(fig, fig_dir / "recherche_validation.png")

    final = pd.read_csv(metrics_dir / "test_final.csv")
    vessel = final[final["class"] == "vessel"]
    configs = list(dict.fromkeys(vessel["config"]))
    metrics = [("dice", "Dice"), ("cldice", "clDice"), ("hd95_mm", "HD95 (mm)")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 0.45 * len(configs) + 2), squeeze=False)
    for ax, (col, name) in zip(axes[0], metrics):
        means = [vessel.loc[vessel["config"] == c, col].mean() for c in configs]
        colors = [plots.ORANGE if "post-traité" in c else plots.BLUE for c in configs]
        ax.barh(range(len(configs)), means, color=colors, height=0.6)
        for k, m in enumerate(means):
            ax.text(m, k, f" {m:.3f}" if col != "hd95_mm" else f" {m:.1f}", va="center", fontsize=8, color=plots.INK)
        ax.set_yticks(range(len(configs)), configs if ax is axes[0][0] else [""] * len(configs), fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{name} vaisseaux (test, moyenne sur les volumes)")
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=plots.BLUE, label="brut (seuil 0,5)"),
                        plt.Rectangle((0, 0), 1, 1, color=plots.ORANGE, label="post-traité (réglé sur la validation)")],
               loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.04))
    plots.save(fig, fig_dir / "comparaison_test.png")

    chosen = f"{best['ensemble']} (post-traité)"
    per_case = vessel[vessel["config"] == chosen].sort_values("dice")
    cid = per_case.iloc[len(per_case) // 2]["case_id"]
    raw_dir = resolve(data_cfg["dataset"]["raw_dir"])
    case = load_case(raw_dir, cid)
    preds = {}
    for c in [f"{best['ensemble']} (brut)", chosen]:
        row = final[final["config"] == c].iloc[0]
        members = selection["ensembles"][row["ensemble"]]
        p_vessel, p_tumour = ensemble_probs(cfg, "test", members, cid)
        preds[c] = labels_from_probs(p_vessel, p_tumour, float(row["threshold"]), int(row["min_voxels"]))
    z = plots.best_slices(case.label, 1, 1)[0]
    plots.plot_model_comparison_slice(case.image, case.label, preds, z, fig_dir / f"exemple_{cid}.png",
                                      f"{cid} (volume de test au Dice médian pour la configuration retenue)")
    print(f"Figures dans {fig_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CFG_DEFAULT)
    parser.add_argument("--step", required=True, choices=["probs", "search", "final", "figures"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--runs", nargs="*", help="modèles concernés par --step probs (par défaut : tous)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-degradations", action="store_true")
    args = parser.parse_args()
    cfg = load_config(resolve(args.config))
    data_cfg = load_config(resolve(cfg["data_config"]))
    if args.step == "probs":
        step_probs(cfg, data_cfg, args.split, args.runs)
    elif args.step == "search":
        step_search(cfg, data_cfg, args.workers)
    elif args.step == "final":
        step_final(cfg, data_cfg, args.workers, not args.no_degradations)
    else:
        step_figures(cfg, data_cfg)


if __name__ == "__main__":
    main()
