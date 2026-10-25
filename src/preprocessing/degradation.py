"""Dégradations artificielles de la résolution dans le plan de coupe.

Les dégradations sont appliquées au volume en unités Hounsfield, avant la
normalisation, puis l'image est ramenée sur la grille d'origine. Le masque
de référence n'est pas modifié : on mesure la capacité du modèle à
retrouver l'anatomie réelle à partir d'une image moins résolue.

Ces images simulées servent uniquement à étudier la sensibilité des
modèles à la résolution. Ce ne sont pas des acquisitions réelles à une
autre résolution (ni des paires CT / micro-CT).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def gaussian_blur(volume: np.ndarray, spacing, sigma_mm: float) -> np.ndarray:
    """Flou gaussien dans le plan (axes H et W), sigma exprimé en mm."""
    _, sh, sw = spacing
    sigma = (0.0, sigma_mm / sh, sigma_mm / sw)
    return ndimage.gaussian_filter(volume.astype(np.float32), sigma=sigma, mode="nearest")


def downsample_upsample(volume: np.ndarray, factor: int, order: int = 3) -> np.ndarray:
    """Sous-échantillonnage d'un facteur `factor` dans le plan puis retour à la taille d'origine.

    Un filtre anti-repliement gaussien (sigma = (factor - 1) / 2 pixels,
    comme dans scikit-image) est appliqué avant la décimation. Le retour
    sur la grille d'origine utilise une interpolation spline d'ordre 3.
    """
    if factor <= 1:
        return volume.astype(np.float32)
    z, h, w = volume.shape
    sigma = (factor - 1) / 2
    smooth = ndimage.gaussian_filter(volume.astype(np.float32), sigma=(0, sigma, sigma), mode="nearest")
    low = ndimage.zoom(smooth, (1, 1 / factor, 1 / factor), order=1, mode="nearest", grid_mode=True)
    zoom_back = (1, h / low.shape[1], w / low.shape[2])
    high = ndimage.zoom(low, zoom_back, order=order, mode="nearest", grid_mode=True)
    return high[:, :h, :w].astype(np.float32)


def add_gaussian_noise(volume: np.ndarray, sigma_hu: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (volume + rng.normal(0, sigma_hu, volume.shape)).astype(np.float32)


def degrade(volume: np.ndarray, spacing, condition: dict, seed: int = 0) -> np.ndarray:
    """Applique une condition décrite par un dict, par exemple
    {"blur_sigma_mm": 1.0}, {"downsample": 2} ou {"downsample": 2, "noise_hu": 20}.
    L'ordre est : flou, sous-échantillonnage, bruit.
    """
    out = volume.astype(np.float32)
    if condition.get("blur_sigma_mm"):
        out = gaussian_blur(out, spacing, condition["blur_sigma_mm"])
    if condition.get("downsample", 1) > 1:
        out = downsample_upsample(out, int(condition["downsample"]))
    if condition.get("noise_hu"):
        out = add_gaussian_noise(out, condition["noise_hu"], seed)
    return out
