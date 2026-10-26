"""Inférence coupe par coupe sur un volume complet (modèles 2D)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from src.data.dataset2d import make_inputs
from src.wavelets.transforms import WaveletFeatures


def pad_to_multiple(image: np.ndarray, multiple: int, fill: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Complète une coupe (H, W) en bas et à droite pour atteindre un multiple de `multiple`."""
    h, w = image.shape
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return image, (h, w)
    return np.pad(image, ((0, ph), (0, pw)), constant_values=fill), (h, w)


@torch.inference_mode()
def iterate_volume_logits(model, volume: np.ndarray, wavelet: WaveletFeatures | None, fill_value: float,
                          batch_size: int = 8, device="cpu"):
    """Produit (indices des coupes, logits) par lots de coupes, sur la grille du volume."""
    model.eval()
    multiple = model.size_multiple
    z = volume.shape[0]
    for start in range(0, z, batch_size):
        xs, injections, size = [], [], None
        for k in range(start, min(start + batch_size, z)):
            padded, size = pad_to_multiple(np.asarray(volume[k], dtype=np.float32), multiple, fill_value)
            x, inj = make_inputs(padded, wavelet)
            xs.append(x)
            injections.append(inj)
        x = torch.from_numpy(np.stack(xs)).to(device)
        inj = [torch.from_numpy(np.stack([s[i] for s in injections])).to(device) for i in range(len(injections[0]))]
        logits = model(x, inj or None)[..., : size[0], : size[1]]
        yield range(start, start + len(xs)), logits


TTA_FLIPS_2D = [(), (2,), (1,), (1, 2)]  # retournements des axes W et H d'un volume (Z, H, W)


def predict_volume(model, volume: np.ndarray, wavelet: WaveletFeatures | None, fill_value: float,
                   batch_size: int = 8, device="cpu", flips=((),)) -> np.ndarray:
    """Retourne les probabilités softmax (C, Z, H, W), stockées en float16 pour limiter la mémoire.

    Avec plusieurs `flips`, les probabilités sont moyennées sur les versions retournées du volume
    (augmentation au test) ; les caractéristiques ondelettes sont recalculées sur chaque version.
    """
    total = None
    for dims in flips:
        view = np.flip(volume, dims) if dims else volume
        out = None
        for idx, logits in iterate_volume_logits(model, view, wavelet, fill_value, batch_size, device):
            probs = F.softmax(logits, dim=1).cpu().numpy()
            if out is None:
                out = np.zeros((probs.shape[1],) + volume.shape, dtype=np.float32)
            out[:, idx.start : idx.stop] = probs.transpose(1, 0, 2, 3)
        out = np.flip(out, tuple(d + 1 for d in dims)) if dims else out
        total = out if total is None else total + out
    return (total / len(flips)).astype(np.float16)


def probs_to_original_grid(probs: np.ndarray, original_shape) -> np.ndarray:
    """Ramène des probabilités sur la grille d'origine si un rééchantillonnage a été appliqué."""
    if tuple(probs.shape[1:]) == tuple(original_shape):
        return probs
    zoom = [o / s for o, s in zip(original_shape, probs.shape[1:])]
    return np.stack([ndimage.zoom(p, zoom, order=1, grid_mode=True, mode="nearest") for p in probs])


def probs_to_labels(probs: np.ndarray) -> np.ndarray:
    return probs.argmax(axis=0).astype(np.uint8)
