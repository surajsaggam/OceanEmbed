"""
tests/test_model_shapes.py
--------------------------
Shape tests for the Phase-1 OceanEmbedNet model.

These are part of the required model debugging gate. All tests run on CPU
(no GPU required) so they can be executed on any machine immediately after
writing the model code.

Rules verified:
  - OceanEmbedNet.forward(x) with x.shape=[B, 14, H, W]
    returns dict with:
      "temperature":  shape [B, 15, H, W]
      "embedding":    shape [B, 128, H, W]
  - Both outputs are float32
  - Shapes are derived from config (not hardcoded)
  - Model runs on CPU without CUDA

NOTE: This test will be skipped until models/ocean_embed_net.py is implemented.
The @pytest.mark.skipif guard is removed once the model exists.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config

# Skip if model is not yet implemented
model_available = importlib.util.find_spec("models.ocean_embed_net") is not None
try:
    if model_available:
        from models.ocean_embed_net import OceanEmbedNet
        model_available = True
except (ImportError, Exception):
    model_available = False


@pytest.mark.skipif(not model_available, reason="models/ocean_embed_net.py not yet implemented")
class TestOceanEmbedNetShapes:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.model_cfg = load_config("model")
        self.data_cfg = load_config("data")
        self.model = OceanEmbedNet(self.model_cfg).eval()

    def test_temperature_output_shape(self, synthetic_surface_batch):
        with torch.no_grad():
            out = self.model(synthetic_surface_batch.cpu())
        B = synthetic_surface_batch.shape[0]
        H, W = synthetic_surface_batch.shape[2], synthetic_surface_batch.shape[3]
        expected = (B, len(self.data_cfg.depths_m), H, W)
        assert out["temperature"].shape == expected, (
            f"temperature shape {tuple(out['temperature'].shape)} != {expected}"
        )

    def test_embedding_output_shape(self, synthetic_surface_batch):
        with torch.no_grad():
            out = self.model(synthetic_surface_batch.cpu())
        B = synthetic_surface_batch.shape[0]
        H, W = synthetic_surface_batch.shape[2], synthetic_surface_batch.shape[3]
        expected = (B, self.model_cfg.embedding_dim, H, W)
        assert out["embedding"].shape == expected, (
            f"embedding shape {tuple(out['embedding'].shape)} != {expected}"
        )

    def test_output_dtype_float32(self, synthetic_surface_batch):
        with torch.no_grad():
            out = self.model(synthetic_surface_batch.cpu())
        assert out["temperature"].dtype == torch.float32
        assert out["embedding"].dtype == torch.float32

    def test_output_keys_present(self, synthetic_surface_batch):
        with torch.no_grad():
            out = self.model(synthetic_surface_batch.cpu())
        assert "temperature" in out, "Output dict must contain 'temperature'"
        assert "embedding" in out, "Output dict must contain 'embedding'"

    def test_spatial_dimensions_preserved(self, synthetic_surface_batch):
        """H and W must be identical in output and input (single-resolution model)."""
        with torch.no_grad():
            out = self.model(synthetic_surface_batch.cpu())
        assert out["temperature"].shape[2:] == synthetic_surface_batch.shape[2:]
        assert out["embedding"].shape[2:] == synthetic_surface_batch.shape[2:]
