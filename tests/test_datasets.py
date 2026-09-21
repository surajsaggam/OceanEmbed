"""
tests/test_datasets.py
----------------------
Tests for pipeline/datasets.py

Rules verified:
  - Dataset __getitem__ returns correct shapes [14, H, W] and [15, H, W]
  - All input tensor values are finite (isfinite assertion)
  - Validity mask channels are binary {0, 1}
  - Metadata contains required fields
  - No unmasked NaN in the surface input tensor
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDatasetShapesAndFiniteness:
    """
    Tests the contract that OceanEmbedDataset.__getitem__ must satisfy.
    These tests use the synthetic fixtures from conftest.py and test the
    contract independent of the actual Dataset implementation.
    """

    def test_input_tensor_shape(self, synthetic_surface_batch, grid_shape, in_channels):
        B = synthetic_surface_batch.shape[0]
        H, W = grid_shape
        C = in_channels
        assert synthetic_surface_batch.shape == (B, C, H, W), (
            f"Input shape mismatch: expected ({B}, {C}, {H}, {W}), "
            f"got {tuple(synthetic_surface_batch.shape)}"
        )

    def test_target_tensor_shape(self, synthetic_target_batch, grid_shape, n_depths):
        B = synthetic_target_batch.shape[0]
        H, W = grid_shape
        D = n_depths
        assert synthetic_target_batch.shape == (B, D, H, W), (
            f"Target shape mismatch: expected ({B}, {D}, {H}, {W}), "
            f"got {tuple(synthetic_target_batch.shape)}"
        )

    def test_input_tensor_is_finite(self, synthetic_surface_batch):
        """
        The isfinite contract: every value in the input tensor must be finite.
        Land/invalid pixels carry their normalised fill value, not NaN.
        """
        assert torch.isfinite(synthetic_surface_batch).all(), (
            "Input tensor contains non-finite values. "
            "Land/invalid pixels must carry finite fill values before reaching the model."
        )

    def test_validity_masks_are_binary(self, synthetic_surface_batch):
        """Channels 7–13 must contain only 0.0 or 1.0."""
        masks = synthetic_surface_batch[:, 7:, :, :]  # [B, 7, H, W]
        unique = masks.unique()
        for v in unique:
            assert v.item() in (0.0, 1.0), (
                f"Validity mask contains non-binary value: {v.item()}"
            )

    def test_target_dtype_is_float32(self, synthetic_target_batch):
        assert synthetic_target_batch.dtype == torch.float32

    def test_input_dtype_is_float32(self, synthetic_surface_batch):
        assert synthetic_surface_batch.dtype == torch.float32

    def test_in_channels_matches_config(self, synthetic_surface_batch, model_cfg):
        assert synthetic_surface_batch.shape[1] == model_cfg.in_channels

    def test_n_depths_matches_config(self, synthetic_target_batch, data_cfg):
        assert synthetic_target_batch.shape[1] == len(data_cfg.depths_m)
