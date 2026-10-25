"""Données synthétiques partagées par les tests."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np
import pytest


def tube_mask(shape, radius: float, axis: int = 0, center=None) -> np.ndarray:
    """Cylindre plein le long d'un axe."""
    grids = np.indices(shape)
    others = [a for a in range(len(shape)) if a != axis]
    center = center or [shape[a] / 2 for a in others]
    dist2 = sum((grids[a] - c) ** 2 for a, c in zip(others, center))
    return dist2 <= radius ** 2


def write_fake_msd(root, n_cases: int = 4, shape=(64, 64, 12), seed: int = 0, affine=None) -> list[str]:
    """Crée une arborescence Task08 minimale avec des volumes synthétiques.

    Le CT contient un "foie" homogène, un vaisseau tubulaire brillant (label 1)
    et une petite tumeur hypodense (label 2).
    """
    rng = np.random.default_rng(seed)
    base = root / "Task08_HepaticVessel"
    (base / "imagesTr").mkdir(parents=True)
    (base / "labelsTr").mkdir(parents=True)
    affine = np.diag([0.8, 0.8, 5.0, 1.0]) if affine is None else affine
    ids = []
    for i in range(n_cases):
        cid = f"hepaticvessel_{i:03d}"
        x, y, z = np.indices(shape)
        liver = ((x - 32) ** 2 + (y - 32) ** 2) < 26 ** 2
        vessel = ((x - 30 - i) ** 2 + (y - 34) ** 2) < 4 ** 2
        tumour = ((x - 42) ** 2 + (y - 22) ** 2 + ((z - 6) * 4) ** 2) < 5 ** 2
        image = np.full(shape, -900.0)
        image[liver] = 100
        image[vessel] = 220
        image[tumour & liver] = 40
        image += rng.normal(0, 10, shape)
        label = np.zeros(shape, np.uint8)
        label[vessel] = 1
        label[tumour & liver & ~vessel] = 2
        nib.save(nib.Nifti1Image(image.astype(np.int16), affine), str(base / "imagesTr" / f"{cid}.nii.gz"))
        nib.save(nib.Nifti1Image(label, affine), str(base / "labelsTr" / f"{cid}.nii.gz"))
        ids.append(cid)
    (base / "imagesTr" / "._hepaticvessel_000.nii.gz").write_bytes(b"\x00\x05\x16\x07")
    (base / "dataset.json").write_text(json.dumps({"labels": {"0": "background", "1": "Vessel", "2": "Tumour"}}))
    return ids


@pytest.fixture
def fake_msd(tmp_path):
    ids = write_fake_msd(tmp_path)
    return tmp_path, ids
