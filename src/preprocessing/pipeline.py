"""Prétraitement d'un cas et stockage sur disque.

Format de sortie, un dossier par cas dans `data/processed/` :
- image.npy : CT normalisé, float16, (Z, H, W) ;
- label.npy : masque, uint8, (Z, H, W), valeurs {0, 1, 2} ;
- fg_coords.npy : coordonnées (z, y, x) d'un échantillon de voxels annotés,
  utilisées pour tirer des patches centrés sur les structures ;
- meta.json : espacement, dimensions d'origine, orientation, etc.
Les fichiers NIfTI d'origine ne sont jamais modifiés.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from src.data.msd import Case
from src.preprocessing.intensity import normalize_ct


def resample_inplane(volume: np.ndarray, spacing, target: float, order: int) -> tuple[np.ndarray, tuple]:
    """Rééchantillonne les axes H et W vers un espacement cible (mm).

    `order=1` (linéaire) pour les images, `order=0` (plus proche voisin)
    pour les masques. L'axe Z n'est pas modifié : les modèles 2D traitent
    les coupes indépendamment.
    """
    sz, sh, sw = spacing
    zoom = (1.0, sh / target, sw / target)
    if np.allclose(zoom, 1.0, atol=1e-3):
        return volume, tuple(spacing)
    out = ndimage.zoom(volume, zoom, order=order, mode="nearest", grid_mode=True)
    new_spacing = (sz, sh * volume.shape[1] / out.shape[1], sw * volume.shape[2] / out.shape[2])
    return out, new_spacing


def preprocess_case(case: Case, intensity_stats: dict, target_inplane: float | None = None,
                    max_fg_coords: int = 20000, seed: int = 0) -> dict:
    image, label, spacing = case.image, case.label, case.spacing
    if target_inplane:
        image, new_spacing = resample_inplane(image, spacing, target_inplane, order=1)
        label, _ = resample_inplane(label, spacing, target_inplane, order=0)
        spacing = new_spacing
    image = normalize_ct(image, intensity_stats)
    coords = np.argwhere(label > 0).astype(np.int32)
    if len(coords) > max_fg_coords:
        rng = np.random.default_rng(seed)
        coords = coords[np.sort(rng.choice(len(coords), max_fg_coords, replace=False))]
    meta = {
        "case_id": case.case_id,
        "spacing_zhw": [float(s) for s in spacing],
        "original_spacing_zhw": [float(s) for s in case.spacing],
        "shape_zhw": list(image.shape),
        "original_shape_zhw": list(case.original_shape),
        "original_axcodes": list(case.original_axcodes),
        "label_voxels": {str(k): int((label == k).sum()) for k in (1, 2)},
    }
    return {"image": image.astype(np.float16), "label": label.astype(np.uint8), "fg_coords": coords, "meta": meta}


def save_processed(out_dir: Path, processed: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "image.npy", processed["image"])
    np.save(out_dir / "label.npy", processed["label"])
    np.save(out_dir / "fg_coords.npy", processed["fg_coords"])
    (out_dir / "meta.json").write_text(json.dumps(processed["meta"], indent=2), encoding="utf-8")


def load_processed(case_dir: Path, mmap: bool = True) -> dict:
    mode = "r" if mmap else None
    return {
        "image": np.load(case_dir / "image.npy", mmap_mode=mode),
        "label": np.load(case_dir / "label.npy", mmap_mode=mode),
        "fg_coords": np.load(case_dir / "fg_coords.npy"),
        "meta": json.loads((case_dir / "meta.json").read_text(encoding="utf-8")),
    }
