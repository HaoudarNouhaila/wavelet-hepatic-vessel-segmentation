"""Extension volumique : U-Net 3D de MONAI entraîné par patches.

Les volumes de Task08 sont très anisotropes (coupes de 1,5 à 8 mm pour
environ 0,8 mm dans le plan). Le premier sous-échantillonnage ne réduit
donc que les axes du plan (stride (1, 2, 2)), les suivants les trois axes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from torch.utils.data import Dataset

from src.preprocessing.pipeline import load_processed


def build_unet3d(cfg: dict) -> UNet:
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=cfg.get("num_classes", 3),
        channels=tuple(cfg.get("channels", (16, 32, 64, 128))),
        strides=tuple(tuple(s) for s in cfg.get("strides", ((1, 2, 2), (2, 2, 2), (2, 2, 2)))),
        num_res_units=cfg.get("num_res_units", 1),
        norm=cfg.get("norm", "batch"),
    )


class PatchDataset3D(Dataset):
    """Patches 3D (Z, H, W) tirés comme en 2D : centrés sur un voxel annoté avec
    probabilité `fg_prob`, sinon au hasard. Augmentations : retournements dans
    le plan et variation d'intensité."""

    def __init__(self, processed_dir, case_ids, patch_size=(16, 128, 128), samples_per_epoch: int = 200,
                 fg_prob: float = 0.67, fill_value: float = -3.0, augment: bool = True, seed: int = 0):
        self.cases = [load_processed(Path(processed_dir) / cid) for cid in case_ids]
        self.patch_size = tuple(patch_size)
        self.samples_per_epoch = samples_per_epoch
        self.fg_prob = fg_prob
        self.fill_value = fill_value
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> dict:
        rng = np.random.default_rng([self.seed, self.epoch, index])
        case = self.cases[int(rng.integers(len(self.cases)))]
        shape = case["image"].shape
        coords = case["fg_coords"]
        if len(coords) and rng.random() < self.fg_prob:
            center = coords[rng.integers(len(coords))]
        else:
            center = [rng.integers(s) for s in shape]
        start = [int(c) - p // 2 for c, p in zip(center, self.patch_size)]
        image = np.full(self.patch_size, self.fill_value, np.float32)
        label = np.zeros(self.patch_size, np.int64)
        src = tuple(slice(max(s, 0), min(s + p, n)) for s, p, n in zip(start, self.patch_size, shape))
        dst = tuple(slice(a.start - s, a.stop - s) for a, s in zip(src, start))
        image[dst] = case["image"][src]
        label[dst] = case["label"][src]
        if self.augment:
            for axis in (1, 2):
                if rng.random() < 0.5:
                    image, label = np.flip(image, axis), np.flip(label, axis)
            image = image * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
        return {"image": torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
                "label": torch.from_numpy(np.ascontiguousarray(label))}


@torch.no_grad()
def predict_volume_3d(model, volume: np.ndarray, roi_size, overlap: float = 0.25, sw_batch_size: int = 2,
                      device="cpu") -> np.ndarray:
    """Inférence par fenêtre glissante (pondération gaussienne) ; retourne (C, Z, H, W)."""
    model.eval()
    x = torch.from_numpy(np.asarray(volume, dtype=np.float32))[None, None].to(device)
    logits = sliding_window_inference(x, roi_size, sw_batch_size, model, overlap=overlap, mode="gaussian")
    return torch.softmax(logits, dim=1)[0].cpu().numpy()
