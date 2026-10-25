"""Normalisation des intensités CT.

Schéma inspiré de nnU-Net pour le CT : les valeurs Hounsfield sont
tronquées aux percentiles 0,5 et 99,5 des voxels annotés (vaisseaux et
tumeurs) puis centrées-réduites avec la moyenne et l'écart-type de ces
mêmes voxels. Les statistiques sont estimées uniquement sur les volumes
d'entraînement.
"""

from __future__ import annotations

import numpy as np


def sample_foreground(image: np.ndarray, label: np.ndarray, max_voxels: int, rng: np.random.Generator) -> np.ndarray:
    values = image[label > 0]
    if values.size > max_voxels:
        values = rng.choice(values, max_voxels, replace=False)
    return values.astype(np.float64)


def fit_intensity_stats(samples: list[np.ndarray], lower: float = 0.5, upper: float = 99.5) -> dict[str, float]:
    """Calcule les bornes de troncature et les paramètres de centrage-réduction."""
    values = np.concatenate(samples)
    lo, hi = np.percentile(values, [lower, upper])
    clipped = np.clip(values, lo, hi)
    return {
        "clip_low": float(lo),
        "clip_high": float(hi),
        "mean": float(clipped.mean()),
        "std": float(clipped.std()),
        "n_voxels": int(values.size),
    }


def normalize_ct(image: np.ndarray, stats: dict) -> np.ndarray:
    clipped = np.clip(image, stats["clip_low"], stats["clip_high"])
    return ((clipped - stats["mean"]) / stats["std"]).astype(np.float32)


def background_value(stats: dict) -> float:
    """Valeur normalisée de l'air, utilisée pour le padding."""
    return float((stats["clip_low"] - stats["mean"]) / stats["std"])
