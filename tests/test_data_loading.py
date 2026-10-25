import nibabel as nib
import numpy as np
import pytest

from src.data.msd import check_alignment, check_labels, list_cases, load_case, load_nifti_zhw, to_zhw
from tests.conftest import write_fake_msd


def test_list_cases_ignores_hidden_files(fake_msd):
    root, ids = fake_msd
    assert list_cases(root) == ids


def test_load_case_shapes_and_labels(fake_msd):
    root, ids = fake_msd
    case = load_case(root, ids[0])
    assert case.image.shape == (12, 64, 64)  # (Z, H, W)
    assert case.label.shape == case.image.shape
    assert case.label.dtype == np.uint8
    assert set(np.unique(case.label)) <= {0, 1, 2}
    assert case.spacing == pytest.approx((5.0, 0.8, 0.8))
    assert (case.label == 1).sum() > 0 and (case.label == 2).sum() > 0


def test_orientation_is_canonicalised(tmp_path):
    """Un même volume enregistré en RAS et en LPS doit donner le même tableau ZHW."""
    data = np.zeros((8, 6, 4), np.float32)
    data[1, 2, 3] = 1.0  # repère asymétrique
    ras_affine = np.diag([0.7, 0.9, 2.0, 1.0])
    # En LPS, les axes x et y sont inversés ; la translation garde chaque voxel
    # à la même position dans l'espace monde.
    lps_affine = np.diag([-0.7, -0.9, 2.0, 1.0])
    lps_affine[:3, 3] = [0.7 * 7, 0.9 * 5, 0]
    nib.save(nib.Nifti1Image(data, ras_affine), str(tmp_path / "ras.nii.gz"))
    nib.save(nib.Nifti1Image(data[::-1, ::-1, :].copy(), lps_affine), str(tmp_path / "lps.nii.gz"))
    a, sa, _, codes_a = load_nifti_zhw(tmp_path / "ras.nii.gz")
    b, sb, _, codes_b = load_nifti_zhw(tmp_path / "lps.nii.gz")
    assert codes_a == ("R", "A", "S") and codes_b == ("L", "P", "S")
    np.testing.assert_array_equal(a, b)
    assert sa == pytest.approx(sb)
    assert sa == pytest.approx((2.0, 0.9, 0.7))


def test_to_zhw_display_convention():
    vol = np.zeros((5, 4, 3))  # (x, y, z) en RAS+
    vol[4, 0, 2] = 1  # x maximal = droite du patient, y minimal = postérieur
    zhw = to_zhw(vol)
    assert zhw.shape == (3, 4, 5)
    # droite du patient à gauche de l'image (colonne 0), postérieur en bas (dernière ligne)
    assert zhw[2, 3, 0] == 1


def test_prediction_written_in_original_world_space(tmp_path):
    """predict.py repasse de (Z, H, W) au repère RAS+ : le masque enregistré doit se
    superposer au volume d'origine dans l'espace monde."""
    import importlib.util
    from src.utils import PROJECT_ROOT

    spec = importlib.util.spec_from_file_location("predict", PROJECT_ROOT / "scripts" / "predict.py")
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)
    data = np.random.default_rng(0).random((6, 5, 4)).astype(np.float32)
    affine = np.diag([-0.7, -0.9, 2.0, 1.0])  # LPS
    nib.save(nib.Nifti1Image(data, affine), str(tmp_path / "ct.nii.gz"))
    zhw, _, canonical_affine, _ = load_nifti_zhw(tmp_path / "ct.nii.gz")
    np.testing.assert_array_equal(predict.zhw_to_ras(to_zhw(data)), data)
    written = nib.Nifti1Image(predict.zhw_to_ras(zhw), canonical_affine)
    original = nib.load(str(tmp_path / "ct.nii.gz"))
    # même valeur au même point du monde
    ijk = np.array([1, 2, 3, 1])
    world = original.affine @ ijk
    ijk_written = np.rint(np.linalg.inv(written.affine) @ world).astype(int)
    assert written.get_fdata()[tuple(ijk_written[:3])] == pytest.approx(data[1, 2, 3])


def test_misaligned_label_is_rejected():
    img = np.zeros((4, 4, 4))
    with pytest.raises(ValueError):
        check_alignment(img, np.eye(4), np.zeros((4, 4, 5)), np.eye(4), "x")
    shifted = np.eye(4)
    shifted[0, 3] = 2.0
    with pytest.raises(ValueError):
        check_alignment(img, np.eye(4), img, shifted, "x")


def test_unknown_labels_are_rejected():
    with pytest.raises(ValueError):
        check_labels(np.array([0, 1, 3], dtype=float), "x")
    with pytest.raises(ValueError):
        check_labels(np.array([0, 0.5]), "x")
    assert check_labels(np.array([0.0, 1.0, 2.0]), "x").dtype == np.uint8


def test_non_canonical_dataset_loads_identically(tmp_path):
    affine = np.diag([-0.8, -0.8, 5.0, 1.0])
    ids_lps = write_fake_msd(tmp_path / "lps", n_cases=1, affine=affine)
    ids_ras = write_fake_msd(tmp_path / "ras", n_cases=1)
    lps = load_case(tmp_path / "lps", ids_lps[0])
    ras = load_case(tmp_path / "ras", ids_ras[0])
    assert lps.original_axcodes == ("L", "P", "S")
    assert lps.image.shape == ras.image.shape == (12, 64, 64)
    # les tableaux d'origine sont identiques mais l'affine diffère : après
    # réorientation, le contenu doit être retourné dans le plan
    np.testing.assert_array_equal(lps.label, ras.label[:, ::-1, ::-1])
