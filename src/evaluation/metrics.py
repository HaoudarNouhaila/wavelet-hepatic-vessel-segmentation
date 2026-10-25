"""Métriques de segmentation binaire calculées sur un volume entier.

Conventions :
- les masques sont des tableaux booléens de même forme (2D ou 3D) ;
- `spacing` donne la taille des voxels en mm dans l'ordre des axes ;
- une métrique indéfinie (par exemple HD95 quand un masque est vide)
  vaut NaN et est ignorée lors des moyennes.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


def confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    """Retourne (TP, FP, FN, TN)."""
    pred, gt = pred.astype(bool), gt.astype(bool)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    tn = int(pred.size - tp - fp - fn)
    return tp, fp, fn, tn


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Dice, IoU, précision, rappel et F1 voxel à voxel.

    Pour une segmentation binaire, F1 et Dice sont égaux par définition ;
    le F1 est recalculé à partir de la précision et du rappel pour contrôle.
    Si les deux masques sont vides, Dice et IoU valent NaN (cas non défini).
    """
    tp, fp, fn, _ = confusion(pred, gt)
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    if np.isnan(precision) or np.isnan(recall) or precision + recall == 0:
        f1 = 0.0 if tp + fp + fn > 0 else float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "dice": _ratio(2 * tp, 2 * tp + fp + fn),
        "iou": _ratio(tp, tp + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _bounding_box(mask: np.ndarray, margin: int = 1) -> tuple[slice, ...]:
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(0) - margin, 0)
    hi = np.minimum(coords.max(0) + margin + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def surface(mask: np.ndarray) -> np.ndarray:
    """Voxels du masque ayant au moins un voisin (connexité 1) hors du masque."""
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure, border_value=0)
    return mask & ~eroded


def surface_points(mask: np.ndarray, spacing) -> np.ndarray:
    """Coordonnées physiques (mm) des voxels de surface, calculées dans la boîte englobante."""
    box = _bounding_box(mask, margin=1)
    coords = np.argwhere(surface(mask[box])) + [s.start for s in box]
    return coords * np.asarray(spacing, dtype=np.float64)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing=None) -> float:
    """Distance de Hausdorff au 95e percentile, en mm.

    Définition symétrique : maximum des deux distances dirigées au 95e
    percentile (même définition que MONAI avec directed=False). Les
    distances entre centres de voxels de surface sont obtenues exactement
    par un arbre k-d, ce qui équivaut à une transformée de distance mais
    demande beaucoup moins de mémoire sur les volumes entiers.
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() or not gt.any():
        return float("nan")
    spacing = tuple(spacing) if spacing is not None else (1.0,) * pred.ndim
    pts_pred, pts_gt = surface_points(pred, spacing), surface_points(gt, spacing)
    d_pg = cKDTree(pts_gt).query(pts_pred, workers=-1)[0]
    d_gp = cKDTree(pts_pred).query(pts_gt, workers=-1)[0]
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def skeleton(mask: np.ndarray) -> np.ndarray:
    """Squelette binaire (méthode de Lee en 3D, Zhang en 2D, via scikit-image).

    Le calcul est fait dans la boîte englobante du masque (avec une marge
    d'un voxel vide), ce qui ne change pas le résultat.
    """
    mask = mask.astype(bool)
    out = np.zeros_like(mask, dtype=bool)
    if not mask.any():
        return out
    box = _bounding_box(mask, margin=1)
    out[box] = skeletonize(mask[box]).astype(bool)
    return out


def cldice(pred: np.ndarray, gt: np.ndarray) -> float:
    """centerline Dice (Shit et al., 2021).

    Tprec = |S(P) ∩ G| / |S(P)| : part du squelette prédit contenue dans la vérité.
    Tsens = |S(G) ∩ P| / |S(G)| : part du squelette de référence retrouvée.
    clDice = 2 Tprec Tsens / (Tprec + Tsens).
    """
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return float("nan")
    if not pred.any() or not gt.any():
        return 0.0
    sk_pred, sk_gt = skeleton(pred), skeleton(gt)
    tprec = _ratio(np.count_nonzero(sk_pred & gt), np.count_nonzero(sk_pred))
    tsens = _ratio(np.count_nonzero(sk_gt & pred), np.count_nonzero(sk_gt))
    if np.isnan(tprec) or np.isnan(tsens) or tprec + tsens == 0:
        return 0.0
    return float(2 * tprec * tsens / (tprec + tsens))


def all_metrics(pred: np.ndarray, gt: np.ndarray, spacing=None, with_cldice: bool = True) -> dict[str, float]:
    out = overlap_metrics(pred, gt)
    out["hd95_mm"] = hd95(pred, gt, spacing)
    if with_cldice:
        out["cldice"] = cldice(pred, gt)
    return out
