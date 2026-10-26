"""Dataset PyTorch de patches 2D pour l'entraînement.

Chaque échantillon est tiré de façon déterministe à partir de
(seed, époque, indice) : pour une même seed, le U-Net standard et les
variantes ondelettes voient exactement les mêmes patches et les mêmes
augmentations. Seules les entrées calculées à partir de l'image changent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.augment import Augmenter2D, crop_with_padding
from src.preprocessing.pipeline import load_processed
from src.wavelets.transforms import WaveletFeatures


def make_inputs(image: np.ndarray, wavelet: WaveletFeatures | None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Construit l'entrée du réseau à partir d'une coupe normalisée (H, W).

    Retourne (entrée pleine résolution (C, H, W), liste des injections).
    """
    image = image.astype(np.float32)
    if wavelet is None:
        return image[None], []
    feats = wavelet(image)
    if wavelet.mode == "swt":
        return np.concatenate([image[None], feats[0]], axis=0), []
    return image[None], feats


class LazyCases:
    """Accès paresseux aux volumes prétraités (memory-map).

    Seuls les chemins sont transmis aux processus de chargement du DataLoader : chaque processus
    rouvre les fichiers lui-même au premier accès, au lieu de recevoir une copie complète des
    volumes (ce qui échoue sous Windows et coûte beaucoup de mémoire ailleurs).
    """

    def __init__(self, case_dirs):
        self.case_dirs = [Path(d) for d in case_dirs]
        self._cases = None

    @property
    def cases(self) -> list[dict]:
        if self._cases is None:
            self._cases = [load_processed(d) for d in self.case_dirs]
        return self._cases

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cases"] = None
        return state


class PatchDataset2D(LazyCases, Dataset):
    """Patches carrés tirés dans les volumes d'entraînement.

    Args:
        processed_dir: dossier contenant un sous-dossier par cas.
        case_ids: cas utilisés (uniquement ceux de l'ensemble d'entraînement).
        patch_size: côté du patch (multiple de 16).
        samples_per_epoch: nombre de patches par époque.
        fg_prob: probabilité de centrer le patch sur un voxel annoté.
        fill_value: valeur normalisée de l'air pour le padding.
        wavelet: extracteur de caractéristiques, ou None pour le U-Net standard.
        augment_cfg: paramètres d'augmentation, ou None pour désactiver.
        seed: graine des tirages.
    """

    MARGIN = 8  # marge retirée après le calcul des ondelettes (multiple de 4)

    def __init__(self, processed_dir, case_ids, patch_size: int = 192, samples_per_epoch: int = 1000,
                 fg_prob: float = 0.67, fill_value: float = -3.0, wavelet: WaveletFeatures | None = None,
                 augment_cfg: dict | None = None, seed: int = 0):
        if patch_size % 16:
            raise ValueError("patch_size doit être un multiple de 16")
        super().__init__([Path(processed_dir) / cid for cid in case_ids])
        self.case_ids = list(case_ids)
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.fg_prob = fg_prob
        self.fill_value = fill_value
        self.wavelet = wavelet
        self.augmenter = Augmenter2D(augment_cfg) if augment_cfg is not None else None
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _draw_center(self, case: dict, rng: np.random.Generator) -> tuple[int, int, int]:
        z, h, w = case["image"].shape
        coords = case["fg_coords"]
        if len(coords) and rng.random() < self.fg_prob:
            cz, cy, cx = coords[rng.integers(len(coords))]
            return int(cz), int(cy), int(cx)
        return int(rng.integers(z)), int(rng.integers(h)), int(rng.integers(w))

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch, index])
        case_idx = int(rng.integers(len(self.cases)))
        case = self.cases[case_idx]
        cz, cy, cx = self._draw_center(case, rng)
        inner = self.patch_size + 2 * self.MARGIN
        big = int(np.ceil(inner * 1.5 / 4) * 4) if self.augmenter else inner
        top, left = cy - big // 2, cx - big // 2
        top, left = top - top % 4, left - left % 4  # grille alignée pour la DWT
        image = crop_with_padding(np.asarray(case["image"][cz], dtype=np.float32), top, left, big, self.fill_value)
        label = crop_with_padding(np.asarray(case["label"][cz]), top, left, big, 0)
        if self.augmenter:
            image, label = self.augmenter.spatial(image, label, inner, self.fill_value, rng)
            image = self.augmenter.intensity(image, rng)
        x, injections = make_inputs(image, self.wavelet)
        m = self.MARGIN
        sample = {
            "image": torch.from_numpy(np.ascontiguousarray(x[:, m:-m, m:-m])),
            "label": torch.from_numpy(np.ascontiguousarray(label[m:-m, m:-m]).astype(np.int64)),
            "case_idx": case_idx,
        }
        for k, feat in enumerate(injections):
            mk = m // 2 ** (k + 1)
            sample[f"inject_{k}"] = torch.from_numpy(np.ascontiguousarray(feat[:, mk:-mk, mk:-mk]))
        return sample


def injections_from_batch(batch: dict, device) -> list[torch.Tensor] | None:
    keys = sorted(k for k in batch if k.startswith("inject_"))
    return [batch[k].to(device) for k in keys] or None
