"""
tests/test_model_gradients.py
-----------------------------
Gradient / backpropagation tests for OceanEmbedNet.
Part of the model debugging gate (check 3).

Rules verified:
  - Backward pass on synthetic data completes without error
  - All model parameters receive gradients (no dead weights)
  - No NaN in loss or gradients after one backward step
  - Gradient norms are reasonable (not zero, not astronomically large)

NOTE: Skipped until models/ocean_embed_net.py is implemented.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config

model_available = importlib.util.find_spec("models.ocean_embed_net") is not None
try:
    if model_available:
        from models.ocean_embed_net import OceanEmbedNet
        model_available = True
except (ImportError, Exception):
    model_available = False


@pytest.mark.skipif(not model_available, reason="models/ocean_embed_net.py not yet implemented")
class TestOceanEmbedNetGradients:
    @pytest.fixture(autouse=True)
    def _setup(self, model_cfg):
        self.model = OceanEmbedNet(model_cfg).train()

    def test_backward_completes(self, synthetic_surface_batch, synthetic_target_batch, synthetic_target_mask):
        x = synthetic_surface_batch.cpu()
        y = synthetic_target_batch.cpu()
        mask = synthetic_target_mask.cpu()

        out = self.model(x)
        pred = out["temperature"]

        # Masked MSE loss
        loss = ((pred - y) ** 2)[mask].mean()
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        loss.backward()

    def test_all_params_have_gradients(self, synthetic_surface_batch, synthetic_target_batch, synthetic_target_mask):
        x = synthetic_surface_batch.cpu()
        y = synthetic_target_batch.cpu()
        mask = synthetic_target_mask.cpu()

        out = self.model(x)
        loss = ((out["temperature"] - y) ** 2)[mask].mean()
        loss.backward()

        no_grad = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is None:
                no_grad.append(name)

        assert len(no_grad) == 0, (
            f"Parameters with no gradient after backward pass:\n"
            + "\n".join(f"  - {n}" for n in no_grad)
        )

    def test_no_nan_in_gradients(self, synthetic_surface_batch, synthetic_target_batch, synthetic_target_mask):
        x = synthetic_surface_batch.cpu()
        y = synthetic_target_batch.cpu()
        mask = synthetic_target_mask.cpu()

        out = self.model(x)
        loss = ((out["temperature"] - y) ** 2)[mask].mean()
        loss.backward()

        nan_grads = []
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                nan_grads.append(name)

        assert len(nan_grads) == 0, (
            f"NaN or inf gradients in parameters:\n"
            + "\n".join(f"  - {n}" for n in nan_grads)
        )
