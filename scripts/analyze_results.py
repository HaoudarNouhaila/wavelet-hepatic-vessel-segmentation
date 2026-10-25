"""Synthèse des résultats : tableaux récapitulatifs, tests statistiques et figures.

Entrées : results/metrics/<split>_per_volume.csv, experiments/*/history.csv
et les prédictions enregistrées par evaluate.py.

Exemple :
    python scripts/analyze_results.py --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.data.msd import load_case
from src.evaluation.vessel_analysis import radius_bin_names
from src.utils import load_config, resolve
from src.visualization import plots

CONDITION_LABELS = {"original": "originale", "blur_1mm": "flou 1 mm", "down_x2": "sous-éch. x2",
                    "down_x3": "sous-éch. x3", "down_x4": "sous-éch. x4", "down_x2_noise20": "x2 + bruit 20 HU"}
MAIN_METRICS = ["dice", "iou", "precision", "recall", "f1", "hd95_mm", "cldice"]
BASELINE = "unet2d"


def seed_averaged(df: pd.DataFrame) -> pd.DataFrame:
    """Moyenne des métriques sur les seeds, par modèle, condition et volume."""
    numeric = [c for c in df.columns if df[c].dtype.kind in "fi" and c not in ("seed",)]
    return df.groupby(["model", "condition", "case_id"], as_index=False)[numeric].mean()


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = MAIN_METRICS + [c for c in df.columns if c.startswith("cl_recall_")] + \
        ["gt_components_detected", "pred_components_fp", "largest_component_fraction"]
    cols = [c for c in cols if c in df.columns]
    out = df.groupby(["class", "model", "condition"])[cols].agg(["mean", "std", "median"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    n = df.groupby(["class", "model", "condition"])["case_id"].nunique().rename("n_volumes")
    seeds = df.groupby(["class", "model", "condition"])["seed"].nunique().rename("n_seeds")
    return pd.concat([n, seeds, out], axis=1).reset_index()


def seed_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Moyenne sur les volumes pour chaque seed, puis moyenne et écart-type entre seeds."""
    per_seed = df.groupby(["class", "model", "condition", "seed"])[["dice", "cldice", "hd95_mm"]].mean()
    out = per_seed.groupby(["class", "model", "condition"]).agg(["mean", "std", "count"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.reset_index()


def _paired_rows(vessel: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    for cond in vessel["condition"].unique():
        base = vessel[(vessel["model"] == BASELINE) & (vessel["condition"] == cond)].set_index("case_id")
        for model in vessel["model"].unique():
            if model == BASELINE:
                continue
            other = vessel[(vessel["model"] == model) & (vessel["condition"] == cond)].set_index("case_id")
            common = base.index.intersection(other.index)
            if len(common) == 0:
                continue
            for metric in ("dice", "cldice", "hd95_mm", "recall", "precision"):
                a, b = base.loc[common, metric], other.loc[common, metric]
                ok = a.notna() & b.notna()
                diff = (b - a)[ok]
                p = wilcoxon(b[ok], a[ok]).pvalue if ok.sum() >= 6 and np.any(diff != 0) else np.nan
                gain = -diff if metric == "hd95_mm" else diff  # pour HD95, une valeur plus faible est meilleure
                rows.append({"seeds": label, "condition": cond, "model": model, "metric": metric,
                             "n_volumes": int(ok.sum()), "mean_diff_vs_unet2d": diff.mean(),
                             "median_diff_vs_unet2d": diff.median(), "volumes_better": int((gain > 0).sum()),
                             "volumes_worse": int((gain < 0).sum()), "wilcoxon_p": p})
    return rows


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Test de Wilcoxon apparié (par volume) de chaque modèle contre le U-Net de référence.

    Le test est fait pour chaque seed séparément (comparaison des modèles de même seed), puis
    sur les valeurs moyennées sur les seeds. Il ne tient compte que de la variabilité entre
    volumes, pas de la variabilité entre entraînements.
    """
    vessel = df[df["class"] == "vessel"]
    rows = []
    for seed in sorted(vessel["seed"].unique()):
        rows += _paired_rows(vessel[vessel["seed"] == seed], f"seed {seed}")
    rows += _paired_rows(seed_averaged(vessel), "moyenne des seeds")
    return pd.DataFrame(rows)


def per_seed_table(df: pd.DataFrame) -> pd.DataFrame:
    """Moyenne sur les volumes, pour chaque modèle, seed et condition."""
    cols = ["dice", "iou", "precision", "recall", "f1", "hd95_mm", "cldice"] + \
        [c for c in df.columns if c.startswith("cl_recall_")]
    return df.groupby(["class", "condition", "model", "seed"])[cols].mean().reset_index()


def load_histories(exp_dir: Path) -> dict[str, list[pd.DataFrame]]:
    histories: dict[str, list[pd.DataFrame]] = {}
    for run in sorted(exp_dir.glob("*_seed*")):
        if (run / "history.csv").exists() and (run / "config.yaml").exists():
            import yaml
            name = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))["experiment"]["name"]
            if name == "unet3d":
                continue
            histories.setdefault(name, []).append(pd.read_csv(run / "history.csv"))
    return histories


def load_pred(pred_dir: Path, run: str, condition: str, case_id: str):
    path = pred_dir / run / condition / f"{case_id}.npz"
    return np.load(path)["pred"] if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--eval-config", default="configs/evaluation.yaml")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    eval_cfg = load_config(resolve(args.eval_config))
    data_cfg = load_config(resolve(eval_cfg["data_config"]))
    metrics_dir, fig_dir = resolve("results/metrics"), resolve("results/figures")
    df = pd.read_csv(metrics_dir / f"{args.split}_per_volume.csv")

    summary = summary_table(df)
    summary.to_csv(metrics_dir / f"{args.split}_summary.csv", index=False)
    seed_spread(df).to_csv(metrics_dir / f"{args.split}_seed_spread.csv", index=False)
    per_seed = per_seed_table(df)
    per_seed.to_csv(metrics_dir / f"{args.split}_per_seed.csv", index=False)
    tests = paired_tests(df)
    tests.to_csv(metrics_dir / f"{args.split}_wilcoxon_vs_unet2d.csv", index=False)
    vessel = df[df["class"] == "vessel"]
    view = per_seed[per_seed["class"] == "vessel"].set_index(["condition", "model", "seed"])
    print(view[["dice", "cldice", "hd95_mm", "recall", "precision"]].round(4).to_string())
    print(tests[tests["metric"].isin(["dice", "cldice"])].round(4).to_string(index=False))
    if args.no_figures:
        return

    histories = load_histories(resolve("experiments"))
    if histories:
        plots.plot_training_curves(histories, fig_dir / "training" / "courbes_apprentissage.png",
                                   "Apprentissage des modèles 2D (traits clairs : autres seeds)")

    avg = seed_averaged(vessel)
    original = avg[avg["condition"] == "original"]
    plots.plot_metric_by_model(original, [("dice", "Dice vaisseaux"), ("cldice", "clDice vaisseaux"),
                                          ("hd95_mm", "HD95 vaisseaux (mm)"), ("recall", "Rappel vaisseaux")],
                               fig_dir / "comparison" / "metriques_par_modele.png",
                               f"Métriques par volume ({args.split}, images originales, moyenne des seeds)")
    tumour = seed_averaged(df[(df["class"] == "tumour") & (df["condition"] == "original")])
    plots.plot_metric_by_model(tumour, [("dice", "Dice tumeurs"), ("hd95_mm", "HD95 tumeurs (mm)")],
                               fig_dir / "comparison" / "metriques_tumeurs.png",
                               f"Classe tumeur ({args.split}, images originales)")

    # Étude de la résolution : seed 0 uniquement, évaluée dans toutes les conditions, pour que
    # chaque point des courbes repose sur les mêmes modèles.
    conditions = [c for c in CONDITION_LABELS if c in set(vessel["condition"])]
    bins = [f"cl_recall_{n}" for n in radius_bin_names()]
    seed0 = vessel[vessel["seed"] == 0]
    per_case_cond = seed0.groupby(["model", "condition"])[["dice", "cldice", "hd95_mm"] + bins].agg(["mean", "std"])
    relative = seed0.pivot_table(index=["model", "case_id"], columns="condition", values="dice")
    relative = relative.div(relative["original"], axis=0).groupby("model").median()
    relative.round(3).to_csv(metrics_dir / f"{args.split}_dice_relatif_resolution.csv")
    print("Dice relatif à l'image originale (médiane sur les volumes, seed 0) :")
    print(relative.round(3).to_string())
    plots.plot_degradation(per_case_cond, [("dice", "Dice vaisseaux"), ("cldice", "clDice vaisseaux"),
                                           ("hd95_mm", "HD95 vaisseaux (mm)")], conditions, CONDITION_LABELS,
                           fig_dir / "resolution" / "metriques_selon_resolution.png",
                           "Effet de la dégradation de la résolution (seed 0, moyenne ± écart-type sur 20 volumes)")
    plots.plot_degradation(per_case_cond, [(b, f"Rappel ligne centrale, {b.replace('cl_recall_', '')}") for b in bins],
                           conditions, CONDITION_LABELS, fig_dir / "resolution" / "rappel_par_calibre.png",
                           "Rappel de la ligne centrale de référence selon le rayon local du vaisseau (seed 0)")
    vessel_seeds = per_seed[per_seed["class"] == "vessel"]
    plots.plot_degradation_by_seed(vessel_seeds, [("dice", "Dice vaisseaux"), ("cldice", "clDice vaisseaux"),
                                                  ("hd95_mm", "HD95 vaisseaux (mm)")], conditions, CONDITION_LABELS,
                                   fig_dir / "resolution" / "metriques_selon_resolution_par_seed.png",
                                   "Effet de la résolution pour chaque entraînement (moyenne sur 20 volumes)")
    plots.plot_seed_comparison(vessel_seeds[vessel_seeds["condition"] == "original"],
                               [("dice", "Dice vaisseaux"), ("cldice", "clDice vaisseaux"),
                                ("hd95_mm", "HD95 vaisseaux (mm)"), ("precision", "Précision vaisseaux")],
                               fig_dir / "comparison" / "metriques_par_seed.png",
                               f"Moyenne sur les volumes de {args.split} pour chaque seed (images originales)")

    pred_dir = resolve(eval_cfg["predictions_dir"].format(split=args.split))
    raw_dir = resolve(data_cfg["dataset"]["raw_dir"])
    runs = {}
    for run in sorted(set(df["run"])):
        model = df.loc[df["run"] == run, "model"].iloc[0]
        runs.setdefault(model, run)  # première seed de chaque modèle pour les figures
    models = [m for m in plots.MODEL_LABELS if m in runs and m != "unet3d"]
    base_orig = original[original["model"] == BASELINE].sort_values("dice")
    picks = {"median": base_orig.iloc[len(base_orig) // 2]["case_id"], "worst": base_orig.iloc[0]["case_id"],
             "best": base_orig.iloc[-1]["case_id"]}
    tag_labels = {"median": "Dice médian", "worst": "Dice le plus faible", "best": "Dice le plus élevé"}
    for tag, cid in picks.items():
        case = load_case(raw_dir, cid)
        preds = {m: load_pred(pred_dir, runs[m], "original", cid) for m in models}
        preds = {m: p for m, p in preds.items() if p is not None}
        if not preds:
            continue
        z = plots.best_slices(case.label, 1, 1)[0]
        plots.plot_model_comparison_slice(case.image, case.label, preds, z,
                                          fig_dir / "predictions" / f"comparaison_{tag}_{cid}.png",
                                          f"{cid} (volume de test au {tag_labels[tag]} pour le U-Net 2D, seed 0)")
        if tag == "median":
            plots.plot_vessels_3d(case.label == 1, case.spacing, fig_dir / "predictions" / f"rendu3d_{cid}.png",
                                  f"Vaisseaux en 3D, {cid} : référence et U-Net 2D", other=preds[models[0]] == 1)
            for cond in conditions:
                cp = {m: load_pred(pred_dir, runs[m], cond, cid) for m in models}
                cp = {m: p for m, p in cp.items() if p is not None}
                if cp:
                    plots.plot_model_comparison_slice(case.image, case.label, cp, z,
                                                      fig_dir / "resolution" / f"predictions_{cond}_{cid}.png",
                                                      f"{cid}, condition : {CONDITION_LABELS[cond]}")
    (metrics_dir / f"{args.split}_figure_cases.json").write_text(json.dumps(picks, indent=2), encoding="utf-8")
    print(f"Tableaux dans {metrics_dir}, figures dans {fig_dir}")


if __name__ == "__main__":
    main()
