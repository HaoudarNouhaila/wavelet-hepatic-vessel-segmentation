import numpy as np
import pytest

from src.preprocessing.augment import Augmenter2D, affine_crop, crop_with_padding
from src.preprocessing.degradation import degrade, downsample_upsample, gaussian_blur
from src.preprocessing.intensity import background_value, fit_intensity_stats, normalize_ct
from src.preprocessing.pipeline import load_processed, preprocess_case, resample_inplane, save_processed
from src.data.msd import load_case


def test_intensity_normalisation():
    rng = np.random.default_rng(0)
    stats = fit_intensity_stats([rng.normal(150, 40, 10000), rng.normal(100, 30, 5000)])
    assert stats["clip_low"] < stats["mean"] < stats["clip_high"]
    out = normalize_ct(np.array([-1000.0, stats["mean"], 5000.0]), stats)
    assert out[1] == pytest.approx(0, abs=1e-6)
    assert out[0] == pytest.approx(background_value(stats))
    assert out.dtype == np.float32


def test_resample_inplane_labels_stay_discrete():
    label = np.zeros((3, 40, 40), np.uint8)
    label[:, 10:20, 10:30] = 1
    label[:, 25:30, 5:8] = 2
    out, spacing = resample_inplane(label, (5.0, 0.6, 0.6), 0.8, order=0)
    assert out.shape == (3, 30, 30)
    assert set(np.unique(out)) == {0, 1, 2}
    assert spacing == pytest.approx((5.0, 0.8, 0.8))
    same, sp = resample_inplane(label, (5.0, 0.8, 0.8), 0.8, order=0)
    assert same is label and sp == (5.0, 0.8, 0.8)


def test_degradations_keep_grid_and_remove_detail():
    rng = np.random.default_rng(1)
    vol = rng.normal(0, 100, (2, 64, 64)).astype(np.float32)
    for cond in ({"blur_sigma_mm": 1.0}, {"downsample": 2}, {"downsample": 4}, {"downsample": 2, "noise_hu": 20}):
        assert degrade(vol, (5, 0.8, 0.8), cond).shape == vol.shape
    hf = lambda v: np.abs(np.diff(v, axis=2)).mean()
    assert hf(downsample_upsample(vol, 4)) < hf(downsample_upsample(vol, 2)) < hf(vol)
    assert gaussian_blur(vol, (5, 0.8, 0.8), 1.0).mean() == pytest.approx(vol.mean(), abs=1.0)
    np.testing.assert_array_equal(degrade(vol, (5, 1, 1), {}), vol)
    a = degrade(vol, (5, 1, 1), {"noise_hu": 10}, seed=3)
    b = degrade(vol, (5, 1, 1), {"noise_hu": 10}, seed=3)
    np.testing.assert_array_equal(a, b)


def test_crop_with_padding():
    arr = np.arange(16, dtype=float).reshape(4, 4)
    out = crop_with_padding(arr, -1, 2, 3, fill=-5)
    assert out.shape == (3, 3)
    assert out[0, 0] == -5 and out[1, 0] == arr[0, 2] and out[1, 2] == -5


def test_spatial_augmentation_keeps_image_label_alignment():
    """L'image est une copie du masque : après augmentation, ils doivent coïncider."""
    rng = np.random.default_rng(0)
    label = np.zeros((96, 96), np.uint8)
    label[30:60, 20:40] = 1
    label[50:70, 60:75] = 2
    image = label.astype(np.float32)
    aug = Augmenter2D({"p_affine": 1.0, "p_flip": 0.5})
    for _ in range(10):
        img, lab = aug.spatial(image, label, 64, 0.0, rng)
        assert img.shape == lab.shape == (64, 64)
        assert set(np.unique(lab)) <= {0, 1, 2}
        agree = np.mean(np.rint(img) == lab)
        assert agree > 0.97  # seuls quelques pixels de bord diffèrent (interpolation)


def test_affine_crop_identity():
    img = np.random.default_rng(0).normal(size=(40, 40)).astype(np.float32)
    lab = (img > 0).astype(np.uint8)
    out_img, out_lab = affine_crop(img, lab, (40 - 1) / 2, 20, 0.0, 1.0, 0.0)
    np.testing.assert_allclose(out_img, img[10:30, 10:30], atol=1e-5)
    np.testing.assert_array_equal(out_lab, lab[10:30, 10:30])


def test_preprocess_and_save_roundtrip(fake_msd, tmp_path):
    root, ids = fake_msd
    case = load_case(root, ids[0])
    stats = {"clip_low": -50.0, "clip_high": 300.0, "mean": 150.0, "std": 60.0}
    processed = preprocess_case(case, stats)
    assert processed["image"].dtype == np.float16 and processed["image"].shape == case.image.shape
    np.testing.assert_array_equal(processed["label"], case.label)
    coords = processed["fg_coords"]
    assert np.all(case.label[tuple(coords.T)] > 0)
    save_processed(tmp_path / ids[0], processed)
    loaded = load_processed(tmp_path / ids[0])
    np.testing.assert_array_equal(loaded["label"], case.label)
    assert loaded["meta"]["spacing_zhw"] == pytest.approx([5.0, 0.8, 0.8])
