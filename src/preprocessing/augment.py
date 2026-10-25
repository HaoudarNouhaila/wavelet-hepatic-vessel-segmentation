"""Augmentations 2D appliquées aux patches d'entraînement.

L'image et le masque reçoivent exactement la même transformation spatiale
(interpolation linéaire pour l'image, plus proche voisin pour le masque).
Les augmentations d'intensité ne concernent que l'image. Les
caractéristiques ondelettes sont calculées après l'augmentation, sur
l'image augmentée.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def crop_with_padding(array: np.ndarray, top: int, left: int, size: int, fill: float) -> np.ndarray:
    """Extrait un carré `size` x `size` en complétant par `fill` hors de l'image."""
    h, w = array.shape
    out = np.full((size, size), fill, dtype=array.dtype)
    y0, x0 = max(top, 0), max(left, 0)
    y1, x1 = min(top + size, h), min(left + size, w)
    if y1 > y0 and x1 > x0:
        out[y0 - top : y1 - top, x0 - left : x1 - left] = array[y0:y1, x0:x1]
    return out


def affine_crop(image, label, center_in: float, out_size: int, angle_deg: float, scale: float, fill: float):
    """Rotation + mise à l'échelle autour du centre, puis recadrage à `out_size`.

    Un `scale` > 1 agrandit le contenu (zoom avant).
    """
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) / scale
    c_out = (out_size - 1) / 2
    offset = center_in - rot @ np.array([c_out, c_out])
    img = ndimage.affine_transform(image, rot, offset=offset, output_shape=(out_size, out_size),
                                   order=1, mode="constant", cval=fill)
    lab = ndimage.affine_transform(label, rot, offset=offset, output_shape=(out_size, out_size),
                                   order=0, mode="constant", cval=0)
    return img.astype(np.float32), lab.astype(label.dtype)


class Augmenter2D:
    """Tire et applique une augmentation aléatoire à partir d'un générateur donné."""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.p_affine = cfg.get("p_affine", 0.3)
        self.max_angle = cfg.get("max_angle_deg", 15.0)
        self.scale_range = tuple(cfg.get("scale_range", (0.85, 1.15)))
        self.p_flip = cfg.get("p_flip", 0.5)
        self.intensity_scale = tuple(cfg.get("intensity_scale", (0.9, 1.1)))
        self.intensity_shift = cfg.get("intensity_shift", 0.1)
        self.p_noise = cfg.get("p_noise", 0.15)
        self.noise_std = cfg.get("noise_std", 0.05)

    def spatial(self, big_image, big_label, out_size: int, fill: float, rng: np.random.Generator):
        """`big_*` sont centrés sur le point d'intérêt et plus grands que `out_size`."""
        center = (big_image.shape[0] - 1) / 2
        if rng.random() < self.p_affine:
            angle = rng.uniform(-self.max_angle, self.max_angle)
            scale = rng.uniform(*self.scale_range)
        else:
            angle, scale = 0.0, 1.0
        if angle == 0.0 and scale == 1.0:
            start = (big_image.shape[0] - out_size) // 2
            sl = slice(start, start + out_size)
            image, label = big_image[sl, sl].astype(np.float32), big_label[sl, sl]
        else:
            image, label = affine_crop(big_image, big_label, center, out_size, angle, scale, fill)
        for axis in (0, 1):
            if rng.random() < self.p_flip:
                image, label = np.flip(image, axis), np.flip(label, axis)
        return np.ascontiguousarray(image), np.ascontiguousarray(label)

    def intensity(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        image = image * rng.uniform(*self.intensity_scale) + rng.uniform(-self.intensity_shift, self.intensity_shift)
        if rng.random() < self.p_noise:
            image = image + rng.normal(0, self.noise_std, image.shape)
        return image.astype(np.float32)
