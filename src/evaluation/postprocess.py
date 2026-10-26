"""Post-traitement des probabilités et ensembles de modèles.

Toutes les valeurs (seuil, taille minimale des composantes, poids des modèles) sont choisies sur
l'ensemble de validation (voir scripts/optimize.py), jamais sur l'ensemble de test.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

STRUCTURE_26 = ndimage.generate_binary_structure(3, 3)


def component_sizes(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Étiquettes des composantes connexes (connexité 26) et taille de chacune (indice 0 = fond)."""
    labels, _ = ndimage.label(mask, STRUCTURE_26)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels, sizes


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Supprime les composantes connexes de moins de `min_voxels` voxels."""
    if min_voxels <= 0 or not mask.any():
        return mask.astype(bool)
    labels, sizes = component_sizes(mask)
    return (sizes >= min_voxels)[labels]


def labels_from_probs(p_vessel: np.ndarray, p_tumour: np.ndarray, threshold: float = 0.5,
                      min_voxels: int = 0) -> np.ndarray:
    """Carte de labels (0 fond, 1 vaisseaux, 2 tumeurs) à partir des probabilités.

    Un voxel est un vaisseau si sa probabilité dépasse `threshold` (puis les petites composantes
    sont supprimées) ; sinon il est tumeur si la tumeur est la classe la plus probable.
    Avec threshold = 0,5 et min_voxels = 0, on retrouve à peu près l'argmax.
    """
    p_vessel = np.asarray(p_vessel, dtype=np.float32)
    p_tumour = np.asarray(p_tumour, dtype=np.float32)
    vessel = remove_small_components(p_vessel >= threshold, min_voxels)
    p_background = 1.0 - p_vessel - p_tumour
    tumour = (p_tumour > p_background) & (p_tumour > p_vessel) & ~vessel
    out = np.zeros(p_vessel.shape, dtype=np.uint8)
    out[vessel] = 1
    out[tumour] = 2
    return out


def weighted_average(arrays: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Moyenne pondérée (poids normalisés) de cartes de probabilités de même forme."""
    if len(arrays) != len(weights) or not arrays:
        raise ValueError("Autant de poids que de cartes sont nécessaires")
    total = float(sum(weights))
    out = np.zeros(arrays[0].shape, dtype=np.float32)
    for array, weight in zip(arrays, weights):
        out += np.asarray(array, dtype=np.float32) * (weight / total)
    return out


def dice_grid(p_vessel: np.ndarray, gt: np.ndarray, thresholds, min_sizes) -> dict[tuple[float, int], float]:
    """Dice des vaisseaux pour chaque couple (seuil, taille minimale), calculé efficacement.

    Pour chaque seuil, les composantes ne sont étiquetées qu'une fois ; le filtrage par taille se
    fait ensuite par une table de correspondance.
    """
    gt = gt.astype(bool)
    n_gt = int(gt.sum())
    out = {}
    for t in thresholds:
        mask = p_vessel >= t
        labels, sizes = component_sizes(mask)
        tp_per_label = np.bincount(labels[gt], minlength=len(sizes))
        tp_per_label[0] = 0
        for m in min_sizes:
            keep = sizes >= m
            keep[0] = False
            tp = int(tp_per_label[keep].sum())
            n_pred = int(sizes[keep].sum())
            denom = n_pred + n_gt
            out[(float(t), int(m))] = 2 * tp / denom if denom else float("nan")
    return out
