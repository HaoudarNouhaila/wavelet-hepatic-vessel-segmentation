import numpy as np
import pytest

from src.wavelets.transforms import (
    WaveletFeatures,
    dwt2_subbands,
    idwt2_reconstruct,
    iswt2_reconstruct,
    multilevel_dwt,
    swt2_subbands,
)

RNG = np.random.default_rng(0)
IMAGE = RNG.normal(size=(64, 48)).astype(np.float64)


@pytest.mark.parametrize("wavelet", ["haar", "db2"])
def test_dwt_shapes_and_perfect_reconstruction(wavelet):
    bands = dwt2_subbands(IMAGE, wavelet)
    assert set(bands) == {"LL", "LH", "HL", "HH"}
    assert all(b.shape == (32, 24) for b in bands.values())
    np.testing.assert_allclose(idwt2_reconstruct(bands, wavelet), IMAGE, atol=1e-10)


@pytest.mark.parametrize("wavelet", ["haar", "db2"])
def test_swt_keeps_resolution_and_reconstructs(wavelet):
    bands = swt2_subbands(IMAGE, wavelet)
    assert all(b.shape == IMAGE.shape for b in bands.values())
    np.testing.assert_allclose(iswt2_reconstruct(bands, wavelet), IMAGE, atol=1e-10)


def test_haar_subbands_meaning():
    """LL = somme des blocs 2x2 / 2 ; une image constante n'a aucun détail."""
    bands = dwt2_subbands(IMAGE, "haar")
    blocks = IMAGE.reshape(32, 2, 24, 2).sum(axis=(1, 3)) / 2
    np.testing.assert_allclose(bands["LL"], blocks, atol=1e-12)
    flat = dwt2_subbands(np.full((16, 16), 3.0), "haar")
    for name in ("LH", "HL", "HH"):
        assert np.abs(flat[name]).max() < 1e-12


def test_detail_orientation():
    """Un contour horizontal (variation selon les lignes) apparaît dans LH, pas dans HL."""
    img = np.zeros((32, 32))
    img[15:, :] = 1.0  # contour entre les lignes 14 et 15 (au milieu d'un bloc 2x2)
    bands = dwt2_subbands(img, "haar")
    assert np.abs(bands["LH"]).sum() > 0
    assert np.abs(bands["HL"]).sum() == pytest.approx(0)


def test_multilevel_sizes_match_unet_stages():
    levels = multilevel_dwt(np.zeros((192, 192)), levels=3)
    assert [lvl["LL"].shape for lvl in levels] == [(96, 96), (48, 48), (24, 24)]


def test_haar_dwt_on_aligned_crop_equals_full_image():
    """Avec Haar, la DWT d'un recadrage aligné sur la grille 2x2 est exactement
    le recadrage de la DWT de l'image : pas d'effet de bord."""
    full = dwt2_subbands(IMAGE, "haar")
    crop = dwt2_subbands(IMAGE[8:40, 4:36], "haar")
    for k in full:
        np.testing.assert_allclose(crop[k], full[k][4:20, 2:18])


def test_features_normalisation_and_errors():
    images = [RNG.normal(2.0, 3.0, size=(32, 32)) for _ in range(10)]
    for mode in ("swt", "dwt_multilevel"):
        feats = WaveletFeatures(mode, "haar", levels=2)
        with pytest.raises(RuntimeError):
            feats(images[0])
        feats.fit(images)
        out = feats(images[0])
        assert len(out) == (1 if mode == "swt" else 2)
        assert out[0].shape[0] == 4 and out[0].dtype == np.float32
        stacked = np.stack([feats(im)[0] for im in images])
        np.testing.assert_allclose(stacked.mean(axis=(0, 2, 3)), 0, atol=1e-5)
        np.testing.assert_allclose(stacked.std(axis=(0, 2, 3)), 1, atol=1e-4)
    with pytest.raises(ValueError):
        WaveletFeatures("dwt_multilevel", "haar", levels=2, stats={}).raw(np.zeros((30, 32)))
    with pytest.raises(ValueError):
        WaveletFeatures("unknown")
