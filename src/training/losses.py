"""Fonctions de perte pour la segmentation multi-classe.

Les vaisseaux occupent une très petite fraction des voxels. La Dice loss
(calculée sur le lot entier, sans le fond) compense ce déséquilibre, et la
cross-entropy stabilise l'apprentissage au début. Leur somme est la
combinaison utilisée par nnU-Net (Isensee et al., 2021).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def one_hot(target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, H, W[, D]) entiers -> (B, C, H, W[, D]) flottants."""
    return F.one_hot(target.long(), num_classes).movedim(-1, 1).float()


class SoftDiceLoss(nn.Module):
    """1 - Dice moyen sur les classes de premier plan.

    Args:
        include_background: inclure la classe 0 dans la moyenne.
        batch_dice: sommer sur tout le lot avant de calculer le Dice (plus
            stable quand certaines images ne contiennent pas une classe).
        smooth: terme de lissage du numérateur et du dénominateur.
    """

    def __init__(self, include_background: bool = False, batch_dice: bool = True, smooth: float = 1e-5):
        super().__init__()
        self.include_background = include_background
        self.batch_dice = batch_dice
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = logits.softmax(dim=1)
        target_1h = one_hot(target, logits.shape[1])
        if not self.include_background:
            probs, target_1h = probs[:, 1:], target_1h[:, 1:]
        spatial = tuple(range(2, logits.ndim))
        axes = (0,) + spatial if self.batch_dice else spatial
        intersection = (probs * target_1h).sum(axes)
        denominator = probs.sum(axes) + target_1h.sum(axes)
        dice = (2 * intersection + self.smooth) / (denominator + self.smooth)
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    """Somme pondérée de la Dice loss et de la cross-entropy."""

    def __init__(
        self,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        class_weights: list[float] | None = None,
        include_background: bool = False,
        batch_dice: bool = True,
    ):
        super().__init__()
        self.dice = SoftDiceLoss(include_background, batch_dice)
        weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights else None
        self.register_buffer("class_weights", weights, persistent=False)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target.long(), weight=self.class_weights)
        return self.dice_weight * self.dice(logits, target) + self.ce_weight * ce


def build_loss(cfg: dict) -> nn.Module:
    return DiceCELoss(
        dice_weight=cfg.get("dice_weight", 1.0),
        ce_weight=cfg.get("ce_weight", 1.0),
        class_weights=cfg.get("class_weights"),
        include_background=cfg.get("include_background", False),
        batch_dice=cfg.get("batch_dice", True),
    )
