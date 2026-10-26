"""Tests d'intégration : prétraitement -> DataLoader -> entraînement -> évaluation,
sur un petit jeu de données synthétique au format Task08."""

import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset2d import PatchDataset2D, injections_from_batch
from src.data.msd import load_case
from src.data.splits import make_split, save_split
from src.evaluation.evaluator import evaluate, load_run
from src.evaluation.inference import predict_volume
from src.models.unet2d import build_model_2d
from src.preprocessing.intensity import fit_intensity_stats, sample_foreground
from src.preprocessing.pipeline import preprocess_case, save_processed
from src.training.trainer import load_model_from_checkpoint, save_checkpoint, train
from src.utils import load_config, resolve
from src.wavelets.transforms import WaveletFeatures
from tests.conftest import write_fake_msd

SMALL = [4, 8, 16, 32, 64]


@pytest.fixture
def prepared(tmp_path):
    raw = tmp_path / "raw"
    ids = write_fake_msd(raw, n_cases=6, shape=(64, 64, 10))
    split = make_split(ids, seed=0, fractions=(0.5, 0.25, 0.25))
    save_split(tmp_path / "split.json", split, split, 0, (0.5, 0.25, 0.25))
    rng = np.random.default_rng(0)
    cases = {cid: load_case(raw, cid) for cid in ids}
    stats = fit_intensity_stats([sample_foreground(cases[c].image, cases[c].label, 5000, rng) for c in split["train"]])
    (tmp_path / "stats.json").write_text(json.dumps(stats))
    processed = tmp_path / "processed"
    for cid, case in cases.items():
        save_processed(processed / cid, preprocess_case(case, stats))
    wstats = {}
    slices = [np.asarray(np.load(processed / c / "image.npy")[k], np.float32) for c in split["train"] for k in range(10)]
    for mode, wavelet, levels, key in [("swt", "haar", 1, "swt_haar_L1"), ("dwt_multilevel", "haar", 2, "dwt_multilevel_haar_L2")]:
        wstats[key] = WaveletFeatures(mode, wavelet, levels).fit(slices)
    (tmp_path / "wstats.json").write_text(json.dumps(wstats))
    return {"root": tmp_path, "raw": raw, "split": split, "stats": stats, "processed": processed, "wstats": wstats}


def make_cfg(prepared, mode: str, run_name: str) -> dict:
    cfg = load_config(resolve("configs/base_2d.yaml"))
    root = prepared["root"]
    cfg["experiment"].update({"name": run_name, "seed": 0, "output_dir": str(root / "exp")})
    cfg["data"].update({"processed_dir": str(prepared["processed"]), "split_file": str(root / "split.json"),
                        "stats_file": str(root / "stats.json"), "wavelet_stats_file": str(root / "wstats.json")})
    cfg["model"].update({"wavelet_mode": mode, "wavelet": "haar", "wavelet_levels": 2, "features": SMALL})
    cfg["training"].update({"patch_size": 32, "batch_size": 4, "samples_per_epoch": 8, "max_epochs": 2,
                            "early_stopping_patience": 5, "device": "cpu"})
    return cfg


@pytest.mark.parametrize("mode", ["none", "swt", "dwt_multilevel"])
def test_patch_dataset_shapes(prepared, mode):
    wavelet = None
    if mode != "none":
        key = "swt_haar_L1" if mode == "swt" else "dwt_multilevel_haar_L2"
        wavelet = WaveletFeatures(mode, "haar", 2, prepared["wstats"][key])
    ds = PatchDataset2D(prepared["processed"], prepared["split"]["train"], 32, 6, 0.7, -3.0, wavelet, {}, seed=1)
    batch = next(iter(DataLoader(ds, batch_size=3)))
    assert batch["image"].shape == (3, 5 if mode == "swt" else 1, 32, 32)
    assert batch["label"].shape == (3, 32, 32) and batch["label"].dtype == torch.int64
    assert set(torch.unique(batch["label"]).tolist()) <= {0, 1, 2}
    inj = injections_from_batch(batch, "cpu")
    if mode == "dwt_multilevel":
        assert [t.shape for t in inj] == [(3, 4, 16, 16), (3, 4, 8, 8)]
    else:
        assert inj is None


def test_same_patches_for_all_models(prepared):
    """Pour une même seed, les patches (CT et masque) sont identiques quel que soit le modèle."""
    w = WaveletFeatures("swt", "haar", 1, prepared["wstats"]["swt_haar_L1"])
    aug = {"p_affine": 0.5}
    a = PatchDataset2D(prepared["processed"], prepared["split"]["train"], 32, 5, 0.7, -3.0, None, aug, seed=7)
    b = PatchDataset2D(prepared["processed"], prepared["split"]["train"], 32, 5, 0.7, -3.0, w, aug, seed=7)
    for i in range(5):
        sa, sb = a[i], b[i]
        torch.testing.assert_close(sa["image"][0], sb["image"][0])
        torch.testing.assert_close(sa["label"], sb["label"])


def test_workers_see_new_patches_each_epoch(prepared):
    """Avec des processus de chargement, chaque époque doit tirer de nouveaux patches."""
    ds = PatchDataset2D(prepared["processed"], prepared["split"]["train"], 32, 4, 0.7, -3.0, seed=3)
    loader = DataLoader(ds, batch_size=4, num_workers=1)
    batches = []
    for epoch in (1, 2):
        ds.set_epoch(epoch)
        batches.append(next(iter(loader))["image"])
    assert not torch.equal(batches[0], batches[1])
    ds.set_epoch(1)
    torch.testing.assert_close(next(iter(loader))["image"], batches[0])  # reproductible
    import pickle
    assert len(pickle.dumps(ds)) < 100_000  # seuls les chemins sont transmis aux processus


def test_split_has_no_leakage_into_training(prepared):
    ds = PatchDataset2D(prepared["processed"], prepared["split"]["train"], 32, 4, 0.7, -3.0)
    assert set(ds.case_ids).isdisjoint(prepared["split"]["val"] + prepared["split"]["test"])


def test_checkpoint_roundtrip(tmp_path):
    cfg = {"model": {"wavelet_mode": "swt", "features": SMALL}}
    model = build_model_2d(cfg["model"])
    save_checkpoint(tmp_path / "m.pt", model, None, None, 3, 0.5, cfg)
    loaded, ckpt = load_model_from_checkpoint(tmp_path / "m.pt")
    x = torch.randn(1, 5, 32, 32)
    model.eval()
    torch.testing.assert_close(model(x), loaded(x))
    assert ckpt["epoch"] == 3


@pytest.mark.parametrize("mode", ["none", "dwt_multilevel"])
def test_train_and_evaluate_end_to_end(prepared, mode):
    cfg = make_cfg(prepared, mode, f"run_{mode}")
    run_dir = prepared["root"] / "exp" / f"run_{mode}"
    summary = train(cfg, run_dir, log=lambda *_: None)
    assert (run_dir / "best.pt").exists() and (run_dir / "last.pt").exists()
    history = (run_dir / "history.csv").read_text().splitlines()
    assert len(history) == 1 + summary["epochs_run"]
    run = load_run(run_dir)
    case = load_case(prepared["raw"], prepared["split"]["test"][0])
    from src.preprocessing.intensity import normalize_ct
    probs = predict_volume(run["model"], normalize_ct(case.image, run["stats"]), run["wavelet"], run["fill"])
    assert probs.shape == (3,) + case.image.shape
    np.testing.assert_allclose(probs.astype(np.float32).sum(0), 1.0, atol=1e-2)
    out_csv = prepared["root"] / f"eval_{mode}.csv"
    table = evaluate([run], prepared["raw"], prepared["split"]["test"], {"original": {}, "down_x2": {"downsample": 2}},
                     predictions_dir=prepared["root"] / "pred", out_csv=out_csv, log=lambda *_: None)
    assert len(table) == 2 * 2 * len(prepared["split"]["test"])  # conditions x classes x volumes
    import pandas as pd
    saved = pd.read_csv(out_csv)
    assert len(saved) == len(table) and list(saved.columns) == list(table.columns)
    assert {"dice", "iou", "precision", "recall", "f1", "hd95_mm", "cldice"} <= set(table.columns)
    again = evaluate([run], prepared["raw"], prepared["split"]["test"], {"original": {}}, existing=table,
                     log=lambda *_: None)
    assert len(again) == 0  # reprise : rien à recalculer


def test_unet3d_train_and_evaluate(prepared):
    from src.training.trainer3d import train_3d

    cfg = load_config(resolve("configs/unet_3d.yaml"))
    root = prepared["root"]
    cfg["experiment"].update({"output_dir": str(root / "exp")})
    cfg["data"].update({"processed_dir": str(prepared["processed"]), "split_file": str(root / "split.json"),
                        "stats_file": str(root / "stats.json")})
    cfg["model"].update({"channels": [4, 8, 16, 32]})
    cfg["training"].update({"patch_size": [8, 32, 32], "roi_size": [8, 32, 32], "batch_size": 2,
                            "samples_per_epoch": 4, "max_epochs": 2, "val_every": 1, "device": "cpu"})
    run_dir = root / "exp" / "unet3d_seed0"
    summary = train_3d(cfg, run_dir, log=lambda *_: None)
    assert summary["epochs_run"] == 2 and (run_dir / "best.pt").exists()
    run = load_run(run_dir)
    table = evaluate([run], prepared["raw"], prepared["split"]["test"][:1], {"original": {}}, log=lambda *_: None)
    assert set(table["class"]) == {"vessel", "tumour"} and table["dice"].between(0, 1).all()
