import numpy as np
import pytest
import torch
import torch.nn.functional as F
from monai.losses import DiceLoss

from src.models.unet2d import UNet2D, build_model_2d, count_parameters
from src.training.losses import DiceCELoss, SoftDiceLoss

SMALL = (4, 8, 16, 32, 64)


@pytest.mark.parametrize("mode", ["none", "swt", "dwt_multilevel"])
def test_forward_shapes(mode):
    model = build_model_2d({"wavelet_mode": mode, "features": SMALL, "wavelet_levels": 2})
    x = torch.randn(2, 5 if mode == "swt" else 1, 64, 64)
    inj = [torch.randn(2, 4, 32, 32), torch.randn(2, 4, 16, 16)] if mode == "dwt_multilevel" else None
    out = model(x, inj)
    assert out.shape == (2, 3, 64, 64)


def test_parameter_counts_are_comparable():
    counts = {m: count_parameters(build_model_2d({"wavelet_mode": m})) for m in ("none", "swt", "dwt_multilevel")}
    base = counts["none"]
    assert all(abs(c - base) / base < 0.01 for c in counts.values())  # moins de 1 % d'écart


def test_size_and_injection_errors():
    model = UNet2D(1, 3, SMALL, injection_channels=(0, 4, 0, 0, 0))
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 60, 64), [torch.randn(1, 4, 30, 32)])
    with pytest.raises(ValueError):
        model(torch.randn(1, 1, 64, 64), [torch.randn(1, 4, 16, 16)])
    with pytest.raises(ValueError):
        UNet2D(1, 3, SMALL, injection_channels=(4, 0, 0, 0, 0))


def test_dice_loss_values():
    target = torch.zeros(2, 16, 16, dtype=torch.long)
    target[:, 4:8, 4:8] = 1
    target[:, 10:12, 10:14] = 2
    perfect = F.one_hot(target, 3).permute(0, 3, 1, 2).float() * 40 - 20
    assert SoftDiceLoss()(perfect, target).item() < 1e-4
    assert SoftDiceLoss()(-perfect, target).item() > 0.99


def test_dice_loss_matches_monai():
    torch.manual_seed(0)
    logits = torch.randn(3, 3, 20, 20)
    target = torch.randint(0, 3, (3, 20, 20))
    ours = SoftDiceLoss(include_background=False, batch_dice=True)(logits, target)
    ref = DiceLoss(include_background=False, to_onehot_y=True, softmax=True, batch=True,
                   smooth_nr=1e-5, smooth_dr=1e-5)(logits, target[:, None])
    assert ours.item() == pytest.approx(ref.item(), abs=1e-6)


def test_dice_ce_combines_terms():
    torch.manual_seed(1)
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))
    total = DiceCELoss()(logits, target)
    expected = SoftDiceLoss()(logits, target) + F.cross_entropy(logits, target)
    assert total.item() == pytest.approx(expected.item(), abs=1e-6)
    weighted = DiceCELoss(class_weights=[0.2, 1.0, 1.0])(logits, target)
    assert weighted.item() != pytest.approx(total.item())


def test_early_stopping_respects_min_epochs():
    from src.training.trainer import EarlyStopping

    stopper = EarlyStopping(patience=2, min_epochs=5)
    values = [0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3]
    stops = []
    for v in values:
        stopper.step(v)
        stops.append(stopper.should_stop)
    assert stops == [False, False, False, False, False, False, True]
    assert stopper.best == 0.3


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = UNet2D(1, 3, SMALL)
    x = torch.randn(4, 1, 32, 32)
    y = (x[:, 0] > 0.8).long()
    loss_fn = DiceCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(30):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.7 * first
    assert all(np.isfinite(p.detach().numpy()).all() for p in model.parameters())
