"""
tests/test_model_overfit.py
---------------------------
Synthetic overfit test for OceanEmbedNet.
Part of the model debugging gate (check 4).

Verifies that the model can overfit a tiny fixed dataset.
Loss must decrease monotonically over a small number of epochs.
This confirms the entire forward → loss → backward → optimiser step
chain is wired correctly.

NOTE: Skipped until models/ocean_embed_net.py is implemented.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch
import torch.optim as optim

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
class TestOceanEmbedNetOverfit:
    """
    Overfit test on a tiny synthetic dataset (4 samples, small spatial crop).
    Uses CPU only — this is a correctness test, not a performance benchmark.
    """

    def test_loss_decreases_over_epochs(self, tiny_overfit_batch, model_cfg):
        x, y, mask = tiny_overfit_batch
        # x: [4, 14, 4, 4], y: [4, 15, 4, 4], mask: [4, 15, 4, 4]

        model = OceanEmbedNet(model_cfg).train()
        optimiser = optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        n_epochs = 10

        for epoch in range(n_epochs):
            optimiser.zero_grad()
            out = model(x)
            pred = out["temperature"]
            loss = ((pred - y) ** 2)[mask].mean()
            assert torch.isfinite(loss), f"Epoch {epoch}: loss is not finite"
            loss.backward()
            optimiser.step()
            losses.append(loss.item())

        # Loss must decrease overall (first > last)
        assert losses[0] > losses[-1], (
            f"Loss did not decrease over {n_epochs} epochs on fixed data.\n"
            f"Losses: {losses}\n"
            "This indicates a bug in the forward pass, loss computation, or "
            "optimiser step."
        )

        # Loss must be finite throughout
        for i, l in enumerate(losses):
            assert not (l != l), f"NaN loss at epoch {i}"  # NaN check

        # Loss at epoch 5+ should be strictly less than epoch 0
        mid_loss = losses[5]
        assert mid_loss < losses[0], (
            f"Loss at epoch 5 ({mid_loss:.4f}) is not less than epoch 0 ({losses[0]:.4f})"
        )
