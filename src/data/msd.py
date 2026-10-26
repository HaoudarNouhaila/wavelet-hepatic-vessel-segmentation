"""Chargement des cas du Medical Segmentation Decathlon (Task08_HepaticVessel).

Conventions utilisées dans tout le projet :
- les volumes sont réorientés en RAS+ (nibabel.as_closest_canonical), ce qui
  ne fait que permuter ou inverser des axes, sans interpolation ;
- ils sont ensuite stockés en tableau (Z, H, W) dans la convention
  radiologique d'affichage : lignes de l'antérieur vers le postérieur,
  colonnes de la droite du patient vers sa gauche ;
- l'espacement associé est donné dans le même ordre (sz, sh, sw), en mm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

LABELS = {0: "background", 1: "vessel", 2: "tumour"}


@dataclass
class Case:
    case_id: str
    image: np.ndarray  # (Z, H, W), float32, unités Hounsfield
    label: np.ndarray | None  # (Z, H, W), uint8
    spacing: tuple[float, float, float]  # (sz, sh, sw) en mm
    original_shape: tuple[int, int, int]
    original_axcodes: tuple[str, str, str]


def task_dir(root: str | Path) -> Path:
    """Retourne le dossier Task08_HepaticVessel à partir d'un dossier racine."""
    root = Path(root)
    return root if root.name == "Task08_HepaticVessel" else root / "Task08_HepaticVessel"


def list_cases(root: str | Path, subset: str = "imagesTr") -> list[str]:
    """Liste les identifiants de cas disponibles localement.

    Les fichiers cachés macOS ("._hepaticvessel_001.nii.gz") présents dans
    l'archive officielle sont ignorés.
    """
    folder = task_dir(root) / subset
    names = [
        p.name[: -len(".nii.gz")]
        for p in folder.glob("*.nii.gz")
        if not p.name.startswith(".")
    ]
    return sorted(names)


def case_paths(root: str | Path, case_id: str) -> tuple[Path, Path]:
    base = task_dir(root)
    return base / "imagesTr" / f"{case_id}.nii.gz", base / "labelsTr" / f"{case_id}.nii.gz"


def to_zhw(volume: np.ndarray) -> np.ndarray:
    """Passe d'un volume RAS+ (x, y, z) au tableau (Z, H, W) d'affichage."""
    return np.ascontiguousarray(np.flip(volume, axis=(0, 1)).transpose(2, 1, 0))


def zhw_spacing(zooms) -> tuple[float, float, float]:
    sx, sy, sz = (float(z) for z in zooms[:3])
    return sz, sy, sx


def load_nifti_zhw(path: Path, dtype=np.float32) -> tuple[np.ndarray, tuple, np.ndarray, tuple]:
    """Charge un NIfTI, le réoriente en RAS+ et retourne (tableau ZHW, espacement, affine, axcodes d'origine)."""
    img = nib.load(str(path))
    axcodes = nib.aff2axcodes(img.affine)
    canonical = nib.as_closest_canonical(img)
    data = np.asanyarray(canonical.dataobj).astype(dtype, copy=False)
    if data.ndim != 3:
        raise ValueError(f"{path.name} : volume 3D attendu, dimension {data.shape}")
    return to_zhw(data), zhw_spacing(canonical.header.get_zooms()), canonical.affine, axcodes


def load_case(root: str | Path, case_id: str, with_label: bool = True) -> Case:
    """Charge un cas et vérifie la cohérence image / masque."""
    image_path, label_path = case_paths(root, case_id)
    image, spacing, affine, axcodes = load_nifti_zhw(image_path, np.float32)
    label = None
    if with_label:
        label_raw, label_spacing, label_affine, _ = load_nifti_zhw(label_path, np.float32)
        check_alignment(image, affine, label_raw, label_affine, case_id)
        label = check_labels(label_raw, case_id)
    return Case(case_id, image, label, spacing, image.shape, axcodes)


def check_alignment(image, image_affine, label, label_affine, case_id: str, atol: float = 1e-3) -> None:
    """Vérifie que l'image et le masque partagent la même grille de voxels."""
    if image.shape != label.shape:
        raise ValueError(f"{case_id} : dimensions image {image.shape} et masque {label.shape} différentes")
    if not np.allclose(image_affine, label_affine, atol=atol):
        raise ValueError(f"{case_id} : matrices affines image et masque différentes")


def check_labels(label: np.ndarray, case_id: str) -> np.ndarray:
    """Vérifie que le masque ne contient que des labels entiers connus."""
    rounded = np.rint(label)
    if not np.allclose(rounded, label):
        raise ValueError(f"{case_id} : valeurs non entières dans le masque")
    values = set(np.unique(rounded).astype(int).tolist())
    unknown = values - set(LABELS)
    if unknown:
        raise ValueError(f"{case_id} : labels inattendus {sorted(unknown)}")
    return rounded.astype(np.uint8)


def read_dataset_json(root: str | Path) -> dict:
    path = task_dir(root) / "dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))
