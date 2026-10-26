"""Tests de l'étape d'optimisation : post-traitement, DynUNet, augmentation au test, recherche
sur la validation et évaluation finale."""

import importlib.util
import json

import numpy as np
import pytest
import torch

from src.evaluation.metrics import overlap_metrics
from src.evaluation.postprocess import dice_grid, labels_from_probs, remove_small_components, weighted_average
from src.models.unet3d import TTA_FLIPS_3D, build_model_3d, predict_volume_3d
from src.training.trainer3d import cosine_time_lr
from src.utils import PROJECT_ROOT, load_config, resolve
from tests.test_integration import prepared  # noqa: F401 (fixture partagée)

TINY_DYNUNET = {"type": "dynunet", "kernels": [[1, 3, 3], [3, 3, 3], [3, 3, 3]],
                "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2]], "filters": [4, 8, 16], "res_block": True}


def test_remove_small_components():
    mask = np.zeros((5, 20, 20), bool)
    mask[1:4, 2:8, 2:8] = True  # 108 voxels
    mask[2, 15, 15] = True  # 1 voxel isolé
    out = remove_small_components(mask, 10)
    assert out.sum() == 108 and not out[2, 15, 15]
    np.testing.assert_array_equal(remove_small_components(mask, 0), mask)


def test_labels_from_probs_matches_argmax_at_default():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3, 4, 16, 16))
    probs = np.exp(logits) / np.exp(logits).sum(0)
    labels = labels_from_probs(probs[1], probs[2], threshold=0.5, min_voxels=0)
    argmax = probs.argmax(0)
    # un vaisseau avec p >= 0,5 est forcément la classe majoritaire
    assert np.all(argmax[labels == 1] == 1)
    assert np.all(labels[argmax == 2] == 2)
    low = labels_from_probs(probs[1], probs[2], threshold=0.2)
    assert (low == 1).sum() >= (labels == 1).sum()


def test_dice_grid_matches_direct_computation():
    rng = np.random.default_rng(1)
    p = rng.random((6, 24, 24)).astype(np.float32)
    gt = rng.random((6, 24, 24)) > 0.7
    grid = dice_grid(p, gt, [0.4, 0.6], [0, 5, 50])
    for (t, m), d in grid.items():
        pred = remove_small_components(p >= t, m)
        assert d == pytest.approx(overlap_metrics(pred, gt)["dice"], abs=1e-9)


def test_weighted_average():
    a, b = np.zeros((2, 2)), np.ones((2, 2))
    np.testing.assert_allclose(weighted_average([a, b], [1, 3]), 0.75)
    with pytest.raises(ValueError):
        weighted_average([a], [1, 2])


def test_cosine_time_lr():
    assert cosine_time_lr(1e-3, 1e-5, 0, 100) == pytest.approx(1e-3)
    assert cosine_time_lr(1e-3, 1e-5, 50, 100) == pytest.approx((1e-3 + 1e-5) / 2)
    assert cosine_time_lr(1e-3, 1e-5, 150, 100) == pytest.approx(1e-5)


def test_dynunet_and_tta_shapes():
    torch.manual_seed(0)
    model = build_model_3d(TINY_DYNUNET)
    out = model(torch.randn(1, 1, 8, 32, 32))
    assert out.shape == (1, 3, 8, 32, 32)
    volume = np.random.default_rng(0).normal(size=(10, 40, 40)).astype(np.float32)
    plain = predict_volume_3d(model, volume, (8, 32, 32))
    tta = predict_volume_3d(model, volume, (8, 32, 32), flips=TTA_FLIPS_3D)
    assert plain.shape == tta.shape == (3, 10, 40, 40)
    np.testing.assert_allclose(tta.sum(0), 1.0, atol=1e-4)
    assert not np.allclose(plain, tta)


def test_2d_tta_shapes():
    from src.evaluation.inference import TTA_FLIPS_2D, predict_volume
    from src.models.unet2d import UNet2D

    torch.manual_seed(0)
    model = UNet2D(1, 3, (4, 8, 16, 32, 64))
    volume = np.random.default_rng(0).normal(size=(3, 48, 40)).astype(np.float32)
    plain = predict_volume(model, volume, None, -3.0)
    tta = predict_volume(model, volume, None, -3.0, flips=TTA_FLIPS_2D)
    assert plain.shape == tta.shape == (3, 3, 48, 40)
    np.testing.assert_allclose(tta.astype(np.float32).sum(0), 1.0, atol=1e-2)


def test_optimization_pipeline_end_to_end(prepared, tmp_path):  # noqa: F811
    from src.training.trainer import train
    from src.training.trainer3d import train_3d

    root = prepared["root"]
    data_paths = {"processed_dir": str(prepared["processed"]), "split_file": str(root / "split.json"),
                  "stats_file": str(root / "stats.json"), "wavelet_stats_file": str(root / "wstats.json")}
    cfg2d = load_config(resolve("configs/base_2d.yaml"))
    cfg2d["experiment"].update({"name": "unet2d", "seed": 0, "output_dir": str(root / "exp")})
    cfg2d["data"].update(data_paths)
    cfg2d["model"].update({"features": [4, 8, 16, 32, 64]})
    cfg2d["training"].update({"patch_size": 32, "batch_size": 4, "samples_per_epoch": 8, "max_epochs": 1,
                              "device": "cpu"})
    train(cfg2d, root / "exp" / "u2d", log=lambda *_: None)

    cfg3d = load_config(resolve("configs/kaggle/dynunet.yaml"))
    cfg3d["experiment"].update({"output_dir": str(root / "exp")})
    cfg3d["data"].update(data_paths)
    cfg3d["model"] = TINY_DYNUNET | {"num_classes": 3}
    cfg3d["training"].update({"patch_size": [8, 32, 32], "roi_size": [8, 32, 32], "batch_size": 2,
                              "samples_per_epoch": 4, "max_epochs": 2, "val_every": 1, "device": "cpu",
                              "num_workers": 0, "max_minutes": 5})
    summary = train_3d(cfg3d, root / "exp" / "dyn", log=lambda *_: None)
    assert summary["epochs_run"] == 2

    spec = importlib.util.spec_from_file_location("optimize", PROJECT_ROOT / "scripts" / "optimize.py")
    optimize = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optimize)
    data_cfg = {"dataset": {"raw_dir": str(prepared["raw"])}, "split": {"file": str(root / "split.json")},
                "preprocessing": {"target_inplane_spacing": None}}
    eval_cfg = tmp_path / "eval.yaml"
    eval_cfg.write_text("conditions:\n  down_x2: {downsample: 2}\n", encoding="utf-8")
    cfg = {"cache_dir": str(tmp_path / "cache"), "results_dir": str(tmp_path / "res"), "device": "cpu",
           "evaluation_config": str(eval_cfg),
           "runs": {"u2d": {"dir": str(root / "exp" / "u2d"), "tta": 2},
                    "dyn": {"dir": str(root / "exp" / "dyn"), "tta": 4, "overlap": 0.5}},
           "ensembles": {"dyn": {"dyn": 1}, "u2d": {"u2d": 1}, "both": {"dyn": 1, "u2d": 1}},
           "thresholds": [0.3, 0.5], "min_sizes": [0, 10],
           "final": {"references": ["u2d", "dyn"], "degradations": ["down_x2"]}}
    for split in ("val", "test"):
        optimize.step_probs(cfg, data_cfg, split, None)
    optimize.step_search(cfg, data_cfg, workers=1)
    selection = json.loads((tmp_path / "res" / "metrics" / "selection.json").read_text(encoding="utf-8"))
    assert selection["retenu"]["ensemble"] in {"dyn", "u2d", "both"}
    optimize.step_final(cfg, data_cfg, workers=1, degradations=True)
    import pandas as pd
    final = pd.read_csv(tmp_path / "res" / "metrics" / "test_final.csv")
    assert {"dice", "cldice", "hd95_mm"} <= set(final.columns)
    assert final["config"].str.contains("post-traité").any() and final["config"].str.contains("brut").any()
    degr = pd.read_csv(tmp_path / "res" / "metrics" / "test_final_degradations.csv")
    assert set(degr["condition"]) == {"down_x2"}
    optimize.step_figures(cfg, data_cfg)
    figures = {p.name for p in (tmp_path / "res" / "figures").iterdir()}
    assert {"recherche_validation.png", "comparaison_test.png"} <= figures
