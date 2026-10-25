"""Tests des fonctions de synthèse des résultats sur un tableau synthétique."""

import importlib.util

import numpy as np
import pandas as pd
import pytest

from src.utils import PROJECT_ROOT
from src.visualization import plots

spec = importlib.util.spec_from_file_location("analyze_results", PROJECT_ROOT / "scripts" / "analyze_results.py")
analyze = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


@pytest.fixture
def table():
    rng = np.random.default_rng(0)
    case_effect = rng.normal(0, 0.05, 8)  # difficulté propre à chaque volume, commune aux modèles
    rows = []
    for model, shift in (("unet2d", 0.0), ("wavelet_unet_swt_haar", 0.02)):
        for seed in (0, 1):
            for cond in ("original", "down_x2"):
                for i in range(8):
                    for cls in ("vessel", "tumour"):
                        dice = 0.6 + shift - (0.1 if cond == "down_x2" else 0) + case_effect[i] + rng.normal(0, 0.003)
                        rows.append({"run": f"{model}_seed{seed}", "model": model, "seed": seed, "condition": cond,
                                     "case_id": f"c{i}", "class": cls, "dice": dice, "iou": dice / (2 - dice),
                                     "precision": dice, "recall": dice, "f1": dice, "hd95_mm": 10 - 5 * dice,
                                     "cldice": dice + 0.05, "cl_recall_r0-1.5mm": dice - 0.2,
                                     "cl_recall_r1.5-3mm": dice, "cl_recall_r>3mm": dice + 0.1,
                                     "gt_components_detected": 0.5, "pred_components_fp": 3,
                                     "largest_component_fraction": 0.7})
    return pd.DataFrame(rows)


def test_summary_and_seed_spread(table):
    summary = analyze.summary_table(table)
    row = summary[(summary["class"] == "vessel") & (summary["model"] == "unet2d") & (summary["condition"] == "original")]
    assert row["n_volumes"].item() == 8 and row["n_seeds"].item() == 2
    assert row["dice_mean"].item() == pytest.approx(0.6, abs=0.05)
    spread = analyze.seed_spread(table)
    assert (spread["dice_count"] == 2).all()


def test_paired_tests_detect_consistent_shift(table):
    tests = analyze.paired_tests(table)
    assert set(tests["seeds"]) == {"seed 0", "seed 1", "moyenne des seeds"}
    dice = tests[(tests["metric"] == "dice") & (tests["condition"] == "original")]
    for _, row in dice.iterrows():
        assert row["mean_diff_vs_unet2d"] == pytest.approx(0.02, abs=0.015)
        assert row["volumes_better"] == 8
        assert row["wilcoxon_p"] < 0.05


def test_per_seed_table_and_plots(table, tmp_path):
    per_seed = analyze.per_seed_table(table)
    vessel = per_seed[per_seed["class"] == "vessel"]
    assert len(vessel) == 2 * 2 * 2  # modèles x seeds x conditions
    plots.plot_degradation_by_seed(vessel, [("dice", "Dice")], ["original", "down_x2"], analyze.CONDITION_LABELS,
                                   tmp_path / "d.png")
    plots.plot_seed_comparison(vessel[vessel["condition"] == "original"], [("dice", "Dice")], tmp_path / "e.png")
    assert (tmp_path / "d.png").exists() and (tmp_path / "e.png").exists()


def test_plots_run(table, tmp_path):
    avg = analyze.seed_averaged(table[table["class"] == "vessel"])
    plots.plot_metric_by_model(avg[avg["condition"] == "original"], [("dice", "Dice")], tmp_path / "a.png")
    stats = avg.groupby(["model", "condition"])[["dice", "cldice"]].agg(["mean", "std"])
    plots.plot_degradation(stats, [("dice", "Dice"), ("cldice", "clDice")], ["original", "down_x2"],
                           analyze.CONDITION_LABELS, tmp_path / "b.png")
    hist = pd.DataFrame({"epoch": [1, 2], "train_loss": [1.0, 0.8], "val_loss": [1.1, 0.9], "val_dice_vessel": [0.2, 0.3]})
    plots.plot_training_curves({"unet2d": [hist, hist]}, tmp_path / "c.png")
    assert all((tmp_path / f).exists() for f in ("a.png", "b.png", "c.png"))
