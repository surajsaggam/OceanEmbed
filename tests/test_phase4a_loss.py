"""
tests/test_phase4a_loss.py
--------------------------
Unit tests for Phase-4A vertical gradient loss and GradientAwareLoss.
"""

import numpy as np
import pytest
import torch

from models.losses import GradientAwareLoss, MaskedMSELoss, MaskedVerticalGradientLoss


def test_vertical_gradient_nonuniform_depths():
    depths = [0, 10, 30, 70]
    criterion = MaskedVerticalGradientLoss(depths=depths, loss_type="l1")

    # Linear profile: T(z) = 30.0 - 0.1 * z -> dT/dz = -0.1 degC/m
    # depths: [0, 10, 30, 70] -> T: [30.0, 29.0, 27.0, 23.0]
    targ = torch.tensor([30.0, 29.0, 27.0, 23.0]).view(1, 4, 1, 1)
    # Pred with dT/dz = -0.2 degC/m -> T: [30.0, 28.0, 24.0, 16.0]
    pred = torch.tensor([30.0, 28.0, 24.0, 16.0]).view(1, 4, 1, 1)

    # Difference in dT/dz is |-0.2 - (-0.1)| = 0.1 degC/m everywhere
    loss = criterion(pred, targ)
    assert torch.isclose(loss, torch.tensor(0.1), atol=1e-5)


def test_vertical_gradient_masking():
    depths = [0, 10, 20]
    criterion = MaskedVerticalGradientLoss(depths=depths, loss_type="l1")

    # Pred and target have error only at interval 1 (depths 10->20)
    pred = torch.tensor([25.0, 20.0, 10.0]).view(1, 3, 1, 1)
    targ = torch.tensor([25.0, 20.0, 15.0]).view(1, 3, 1, 1)

    # If depth 2 (20m) is masked out:
    mask = torch.tensor([1.0, 1.0, 0.0]).view(1, 3, 1, 1)
    # Interval 0 (0-10m) has zero error (both pred and targ have dT = -5)
    # Interval 1 (10-20m) has error, but depth 2 is invalid, so interval 1 is masked out!
    loss = criterion(pred, targ, mask=mask)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_gradient_loss_empty_valid_region():
    depths = [0, 10, 20]
    criterion = MaskedVerticalGradientLoss(depths=depths, loss_type="l1")
    pred = torch.randn(2, 3, 4, 4)
    targ = torch.randn(2, 3, 4, 4)
    mask = torch.zeros(2, 3, 4, 4)  # fully masked (e.g. land)

    loss = criterion(pred, targ, mask=mask)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_gradient_loss_backpropagation():
    depths = [0, 10, 20]
    criterion = MaskedVerticalGradientLoss(depths=depths, loss_type="l1")
    pred = torch.tensor([[[[25.0]], [[20.0]], [[15.0]]]], requires_grad=True)
    targ = torch.tensor([[[[25.0]], [[18.0]], [[12.0]]]])

    loss = criterion(pred, targ)
    loss.backward()

    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    # At least some gradients must be non-zero
    assert (pred.grad.abs() > 0).any()


def test_gradient_aware_loss_baseline_equivalence():
    depths = [0, 5, 10, 20]
    pred = torch.randn(2, 4, 5, 5, requires_grad=True)
    targ = torch.randn(2, 4, 5, 5)
    mask = torch.ones(2, 4, 5, 5)

    base_criterion = MaskedMSELoss()
    base_loss = base_criterion(pred, targ, mask=mask)

    # 1. With enabled=False
    comb_criterion_disabled = GradientAwareLoss(depths=depths, enabled=False)
    loss_disabled, metrics_d = comb_criterion_disabled(pred, targ, mask=mask)
    assert torch.isclose(loss_disabled, base_loss)
    assert metrics_d["loss_grad"] == 0.0

    # 2. With lambda_grad=0.0
    comb_criterion_zero_lambda = GradientAwareLoss(depths=depths, lambda_grad=0.0)
    loss_zero_l, metrics_z = comb_criterion_zero_lambda(pred, targ, mask=mask)
    assert torch.isclose(loss_zero_l, base_loss)
    assert metrics_z["grad_weighted"] == 0.0


def test_gradient_aware_loss_positive_lambda():
    depths = [0, 5, 10, 20]
    pred = torch.randn(2, 4, 5, 5, requires_grad=True)
    targ = torch.randn(2, 4, 5, 5)
    mask = torch.ones(2, 4, 5, 5)

    comb_criterion = GradientAwareLoss(depths=depths, lambda_grad=2.0, enabled=True)
    total_loss, metrics = comb_criterion(pred, targ, mask=mask)

    assert torch.isfinite(total_loss)
    assert total_loss.item() > metrics["loss_mse"]
    assert np.isclose(metrics["loss_total"], metrics["loss_mse"] + 2.0 * metrics["loss_grad"])

    total_loss.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
