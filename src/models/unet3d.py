"""Extension volumique : réseaux 3D de MONAI entraînés par patches.

Deux architectures sont disponibles :
- `unet3d` : `monai.networks.nets.UNet`, utilisé pour les premières expériences ;
- `dynunet` : `monai.networks.nets.DynUNet`, réimplémentation de l'architecture de nnU-Net
  (normalisation d'instance, LeakyReLU, noyaux et pas réglables par axe).

Les volumes de Task08 sont très anisotropes (coupes de 1,5 à 8 mm pour environ 0,8 mm dans le
plan). Les premiers sous-échantillonnages ne réduisent donc que les axes du plan, les suivants
les trois axes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet, UNet
from torch.utils.data import Dataset

from src.data.dataset2d import LazyCases

MODEL_TYPES_3D = ("unet3d", "dynunet")


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


def build_dynunet(cfg: dict) -> DynUNet:
    """DynUNet : `strides[0]` concerne la convolution d'entrée (en général 1)."""
    strides = [tuple(s) for s in cfg["strides"]]
    return DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=cfg.get("num_classes", 3),
        kernel_size=[tuple(k) for k in cfg["kernels"]],
        strides=strides,
        upsample_kernel_size=strides[1:],
        filters=list(cfg["filters"]),
        norm_name=("instance", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        res_block=cfg.get("res_block", True),
    )


def build_model_3d(cfg: dict):
    kind = cfg.get("type", "unet3d")
    if kind == "dynunet":
        return build_dynunet(cfg)
    if kind == "unet3d":
        return build_unet3d(cfg)
    raise ValueError(f"Architecture 3D inconnue : {kind}")


class PatchDataset3D(LazyCases, Dataset):
    """Patches 3D (Z, H, W) tirés comme en 2D : centrés sur un voxel annoté avec
    probabilité `fg_prob`, sinon au hasard. Augmentations : retournements dans
    le plan et variation d'intensité."""

    def __init__(self, processed_dir, case_ids, patch_size=(16, 128, 128), samples_per_epoch: int = 200,
                 fg_prob: float = 0.67, fill_value: float = -3.0, augment: bool = True, seed: int = 0,
                 noise_prob: float = 0.0, noise_std: float = 0.05):
        super().__init__([Path(processed_dir) / cid for cid in case_ids])
        self.patch_size = tuple(patch_size)
        self.samples_per_epoch = samples_per_epoch
        self.fg_prob = fg_prob
        self.fill_value = fill_value
        self.augment = augment
        self.seed = seed
        self.noise_prob = noise_prob  # 0 : comportement des premières expériences
        self.noise_std = noise_std
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
            if self.noise_prob and rng.random() < self.noise_prob:
                image = image + rng.normal(0, self.noise_std, image.shape)
        return {"image": torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32)),
                "label": torch.from_numpy(np.ascontiguousarray(label))}


# Retournements dans le plan utilisés pour l'augmentation au test (axes H et W d'un tenseur N, C, Z, H, W).
TTA_FLIPS_3D = [(), (4,), (3,), (3, 4)]


@torch.no_grad()
def predict_volume_3d(model, volume: np.ndarray, roi_size, overlap: float = 0.25, sw_batch_size: int = 2,
                      device="cpu", amp: bool = False, flips=((),)) -> np.ndarray:
    """Inférence par fenêtre glissante (pondération gaussienne) ; retourne (C, Z, H, W).

    Avec plusieurs `flips`, les probabilités sont moyennées sur les versions retournées du
    volume (augmentation au test), chaque prédiction étant remise dans le bon sens.
    """
    model.eval()
    x = torch.from_numpy(np.asarray(volume, dtype=np.float32))[None, None].to(device)
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    total = None
    for dims in flips:
        xi = torch.flip(x, dims) if dims else x
        with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=amp and device_type == "cuda"):
            logits = sliding_window_inference(xi, roi_size, sw_batch_size, model, overlap=overlap, mode="gaussian")
        probs = torch.softmax(logits.float(), dim=1)
        probs = torch.flip(probs, dims) if dims else probs
        total = probs if total is None else total + probs
    return (total / len(flips))[0].cpu().numpy()
