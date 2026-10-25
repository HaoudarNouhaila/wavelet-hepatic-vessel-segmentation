"""U-Net 2D (Ronneberger et al., 2015) et variante guidée par ondelettes.

Le même code sert aux trois modèles comparés :
- U-Net standard : entrée = coupe CT (1 canal) ;
- Wavelet U-Net (entrée) : entrée = CT + 4 sous-bandes SWT (5 canaux) ;
- Wavelet U-Net (multi-échelle) : entrée = CT, et les sous-bandes DWT du
  niveau k sont concaténées aux caractéristiques de l'étage k de l'encodeur,
  qui a exactement la même résolution (H/2^k).
Seuls les canaux d'entrée des blocs concernés changent ; le reste de
l'architecture est identique, ce qui rend la comparaison équitable.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Deux convolutions 3x3 + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    """U-Net 2D avec injection optionnelle de caractéristiques par étage.

    Args:
        in_channels: nombre de canaux de l'entrée pleine résolution.
        num_classes: nombre de classes en sortie (logits, sans softmax).
        features: nombre de filtres par étage, du plus fin au plus grossier.
        injection_channels: canaux supplémentaires concaténés à l'entrée de
            chaque étage de l'encodeur (même longueur que `features`). Le
            premier élément doit être 0 (l'étage 0 reçoit `in_channels`).
        dropout: dropout spatial appliqué au bloc le plus profond.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        features: tuple[int, ...] = (16, 32, 64, 128, 256),
        injection_channels: tuple[int, ...] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        depth = len(features)
        injection = tuple(injection_channels or (0,) * depth)
        if len(injection) != depth or injection[0] != 0:
            raise ValueError("injection_channels doit avoir la longueur de features et commencer par 0")
        self.injection_channels = injection
        self.depth = depth

        self.encoders = nn.ModuleList()
        previous = in_channels
        for level, width in enumerate(features):
            drop = dropout if level == depth - 1 else 0.0
            self.encoders.append(ConvBlock(previous + injection[level], width, drop))
            previous = width
        self.pool = nn.MaxPool2d(2)

        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(depth - 1, 0, -1):
            self.upsamplers.append(nn.ConvTranspose2d(features[level], features[level - 1], 2, stride=2))
            self.decoders.append(ConvBlock(2 * features[level - 1], features[level - 1]))
        self.head = nn.Conv2d(features[0], num_classes, 1)

    @property
    def size_multiple(self) -> int:
        return 2 ** (self.depth - 1)

    def forward(self, x: torch.Tensor, injections: list[torch.Tensor] | None = None) -> torch.Tensor:
        """`injections[k-1]` est concaténé à l'entrée de l'étage k (k >= 1)."""
        if x.shape[-1] % self.size_multiple or x.shape[-2] % self.size_multiple:
            raise ValueError(f"Taille {tuple(x.shape[-2:])} non divisible par {self.size_multiple}")
        skips = []
        for level, encoder in enumerate(self.encoders):
            if level > 0:
                x = self.pool(x)
                if self.injection_channels[level]:
                    extra = injections[level - 1]
                    if extra.shape[-2:] != x.shape[-2:]:
                        raise ValueError(
                            f"Injection de taille {tuple(extra.shape[-2:])} à l'étage {level} "
                            f"de taille {tuple(x.shape[-2:])}"
                        )
                    x = torch.cat([x, extra], dim=1)
            x = encoder(x)
            skips.append(x)
        x = skips.pop()
        for upsample, decoder in zip(self.upsamplers, self.decoders):
            x = upsample(x)
            x = decoder(torch.cat([x, skips.pop()], dim=1))
        return self.head(x)


def build_model_2d(cfg: dict) -> UNet2D:
    """Construit le modèle 2D décrit par la section `model` d'une configuration."""
    features = tuple(cfg.get("features", (16, 32, 64, 128, 256)))
    num_classes = cfg.get("num_classes", 3)
    dropout = cfg.get("dropout", 0.0)
    kind = cfg.get("wavelet_mode", "none")
    if kind == "none":
        return UNet2D(1, num_classes, features, dropout=dropout)
    if kind == "swt":
        return UNet2D(5, num_classes, features, dropout=dropout)
    if kind == "dwt_multilevel":
        levels = cfg.get("wavelet_levels", 2)
        injection = [0] * len(features)
        for level in range(1, levels + 1):
            injection[level] = 4
        return UNet2D(1, num_classes, features, injection_channels=tuple(injection), dropout=dropout)
    raise ValueError(f"wavelet_mode inconnu : {kind}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
