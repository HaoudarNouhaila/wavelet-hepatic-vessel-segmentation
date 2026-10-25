"""Analyse des structures vasculaires : calibre, continuité, composantes.

Le calibre local d'un vaisseau est estimé par la distance (en mm) entre un
point de la ligne centrale de référence et le bord du vaisseau. Le rappel
de la ligne centrale est ensuite calculé par classe de rayon, ce qui
permet de savoir si les petits vaisseaux sont perdus avant les gros.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .metrics import _bounding_box, skeleton

DEFAULT_RADIUS_BINS = (0.0, 1.5, 3.0, np.inf)


def radius_bin_names(bins=DEFAULT_RADIUS_BINS) -> list[str]:
    names = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        names.append(f"r>{lo:g}mm" if np.isinf(hi) else f"r{lo:g}-{hi:g}mm")
    return names


def centerline_radius(gt: np.ndarray, spacing) -> tuple[np.ndarray, np.ndarray]:
    """Retourne le squelette de référence et le rayon local (mm) en chaque point."""
    gt = gt.astype(bool)
    sk = skeleton(gt)
    dist = ndimage.distance_transform_edt(gt, sampling=spacing)
    return sk, dist


def centerline_recall_by_radius(pred: np.ndarray, gt: np.ndarray, spacing, bins=DEFAULT_RADIUS_BINS) -> dict[str, float]:
    """Part des points de la ligne centrale de référence couverts par la prédiction, par rayon."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    out = {}
    names = radius_bin_names(bins)
    if not gt.any():
        return {f"cl_recall_{n}": float("nan") for n in names} | {f"cl_points_{n}": 0 for n in names}
    box = _bounding_box(gt, margin=1)
    sk, dist = centerline_radius(gt[box], spacing)
    radii = dist[sk]
    covered = pred[box][sk]
    for name, lo, hi in zip(names, bins[:-1], bins[1:]):
        sel = (radii > lo) & (radii <= hi)
        n = int(sel.sum())
        out[f"cl_recall_{name}"] = float(covered[sel].mean()) if n else float("nan")
        out[f"cl_points_{name}"] = n
    return out


def component_analysis(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Statistiques sur les composantes connexes (connexité 26 en 3D, 8 en 2D).

    - gt_components / pred_components : nombre de composantes ;
    - gt_components_detected : part des composantes de référence touchées
      par au moins un voxel prédit ;
    - pred_components_fp : composantes prédites sans aucun recouvrement avec
      la référence (faux positifs isolés) ;
    - largest_component_fraction : part du volume prédit contenue dans la
      plus grande composante (indicateur de fragmentation).
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    structure = ndimage.generate_binary_structure(pred.ndim, pred.ndim)
    gt_lab, n_gt = ndimage.label(gt, structure)
    pred_lab, n_pred = ndimage.label(pred, structure)
    detected = np.unique(gt_lab[pred & gt])
    detected = detected[detected > 0]
    hit_pred = np.unique(pred_lab[pred & gt])
    hit_pred = hit_pred[hit_pred > 0]
    largest = 0.0
    if pred.any():
        sizes = np.bincount(pred_lab.ravel())[1:]
        largest = float(sizes.max() / sizes.sum())
    return {
        "gt_components": int(n_gt),
        "pred_components": int(n_pred),
        "gt_components_detected": _safe(len(detected), n_gt),
        "pred_components_fp": int(n_pred - len(hit_pred)),
        "largest_component_fraction": largest,
    }


def _safe(num, den) -> float:
    return float(num / den) if den else float("nan")


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Carte des erreurs : 0 = vrai négatif, 1 = vrai positif, 2 = faux positif, 3 = faux négatif."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    out = np.zeros(pred.shape, dtype=np.uint8)
    out[pred & gt] = 1
    out[pred & ~gt] = 2
    out[~pred & gt] = 3
    return out
