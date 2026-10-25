import numpy as np
import pytest
import torch
from monai.metrics import compute_hausdorff_distance

from src.evaluation.metrics import cldice, hd95, overlap_metrics
from src.evaluation.vessel_analysis import centerline_recall_by_radius, component_analysis, error_map
from tests.conftest import tube_mask


def test_overlap_metrics_known_values():
    gt = np.zeros((10, 10), bool)
    gt[:, :5] = True  # 50 voxels
    pred = np.zeros_like(gt)
    pred[:, 3:8] = True  # 50 voxels dont 20 corrects
    m = overlap_metrics(pred, gt)
    assert m["dice"] == pytest.approx(2 * 20 / 100)
    assert m["iou"] == pytest.approx(20 / 80)
    assert m["precision"] == pytest.approx(0.4) and m["recall"] == pytest.approx(0.4)
    assert m["f1"] == pytest.approx(m["dice"])


def test_overlap_metrics_empty_cases():
    empty = np.zeros((5, 5), bool)
    full = np.ones((5, 5), bool)
    assert np.isnan(overlap_metrics(empty, empty)["dice"])
    m = overlap_metrics(empty, full)
    assert m["dice"] == 0 and m["recall"] == 0 and np.isnan(m["precision"]) and m["f1"] == 0


def test_hd95_matches_monai_with_spacing():
    rng = np.random.default_rng(0)
    gt = np.zeros((20, 40, 40), bool)
    gt[5:15, 10:30, 12:28] = True
    pred = np.roll(gt, 3, axis=2)
    pred[8:10, 2:6, 2:6] = True  # îlot de faux positifs
    pred &= rng.random(pred.shape) > 0.02
    spacing = (2.5, 0.7, 0.7)
    ours = hd95(pred, gt, spacing)
    ref = compute_hausdorff_distance(torch.from_numpy(pred[None, None]), torch.from_numpy(gt[None, None]),
                                     percentile=95, spacing=spacing).item()
    assert ours == pytest.approx(ref, rel=1e-4)
    assert hd95(gt, gt, spacing) == 0.0
    assert np.isnan(hd95(np.zeros_like(gt), gt, spacing))


def test_hd95_translation_in_mm():
    gt = np.zeros((1, 50, 50), bool)
    gt[0, 10:40, 10:20] = True
    pred = np.roll(gt, 4, axis=2)
    assert hd95(pred, gt, (5.0, 1.0, 0.5)) == pytest.approx(4 * 0.5)


def test_cldice_on_synthetic_tubes():
    gt = tube_mask((40, 32, 32), radius=3)
    assert cldice(gt, gt) == pytest.approx(1.0)
    # vaisseau prédit trop épais : le Dice chute mais la topologie est conservée
    thick = tube_mask((40, 32, 32), radius=6)
    assert overlap_metrics(thick, gt)["dice"] < 0.5
    assert cldice(thick, gt) > 0.95
    # vaisseau interrompu : le Dice reste élevé, le clDice baisse davantage
    broken = gt.copy()
    broken[18:24] = False
    d, c = overlap_metrics(broken, gt)["dice"], cldice(broken, gt)
    assert c < 1.0 and c < d + 0.05
    assert cldice(np.zeros_like(gt), gt) == 0.0
    assert np.isnan(cldice(np.zeros_like(gt), np.zeros_like(gt)))


def test_cldice_penalises_missing_thin_branch_more_than_dice():
    trunk = tube_mask((48, 48, 48), radius=6, axis=0)
    branch = tube_mask((48, 48, 48), radius=1.5, axis=1, center=(10, 10))
    gt = trunk | branch
    pred = trunk.copy()  # la petite branche est manquée
    dice = overlap_metrics(pred, gt)["dice"]
    assert dice > 0.95
    assert cldice(pred, gt) < dice - 0.05


def test_centerline_recall_by_radius():
    shape = (30, 64, 64)
    thin = tube_mask(shape, radius=1.2, axis=0, center=(16, 16))
    thick = tube_mask(shape, radius=5, axis=0, center=(44, 44))
    gt = thin | thick
    out = centerline_recall_by_radius(thick, gt, spacing=(1.0, 1.0, 1.0), bins=(0.0, 2.0, np.inf))
    assert out["cl_recall_r0-2mm"] == pytest.approx(0.0)
    assert out["cl_recall_r>2mm"] == pytest.approx(1.0)
    assert out["cl_points_r0-2mm"] > 0


def test_component_analysis():
    gt = np.zeros((10, 30, 30), bool)
    gt[:, 2:6, 2:6] = True
    gt[:, 20:25, 20:25] = True
    pred = np.zeros_like(gt)
    pred[:, 2:6, 2:6] = True
    pred[:, 12:14, 12:14] = True  # faux positif isolé
    out = component_analysis(pred, gt)
    assert out["gt_components"] == 2 and out["pred_components"] == 2
    assert out["gt_components_detected"] == pytest.approx(0.5)
    assert out["pred_components_fp"] == 1


def test_error_map_codes():
    gt = np.array([0, 1, 1, 0], bool)
    pred = np.array([0, 1, 0, 1], bool)
    np.testing.assert_array_equal(error_map(pred, gt), [0, 1, 3, 2])
