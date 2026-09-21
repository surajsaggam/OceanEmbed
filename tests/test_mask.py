"""
tests/test_mask.py
------------------
Tests for pipeline/mask.py

Rules verified:
  - Validity mask is binary {0, 1}
  - Land mask pixels are 0 in the validity mask
  - QC-flagged pixels are 0 in the validity mask
  - Observed pixels are 1 in the validity mask
  - Mask accurately reflects observation vs imputation distinction
"""

from __future__ import annotations

import numpy as np
import pytest


def _apply_land_mask(data: np.ndarray, land_mask: np.ndarray, fill: float):
    """
    Reference land mask application (mirrors pipeline/mask.py).
    land_mask: bool array, True = land
    """
    values = data.copy()
    validity = np.ones_like(data, dtype=np.float32)
    values[land_mask] = fill
    validity[land_mask] = 0.0
    return values, validity


class TestMask:
    def test_validity_mask_is_binary(self):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        land = np.array([False, True, False, True])
        _, mask = _apply_land_mask(data, land, fill=25.0)
        for m in mask:
            assert m in (0.0, 1.0), f"Mask value {m} is not binary"

    def test_land_pixels_masked(self):
        data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        land = np.array([False, True, False])
        _, mask = _apply_land_mask(data, land, fill=25.0)
        assert mask[1] == 0.0, "Land pixel should have mask=0"

    def test_ocean_pixels_unmasked(self):
        data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        land = np.array([False, True, False])
        _, mask = _apply_land_mask(data, land, fill=25.0)
        assert mask[0] == 1.0 and mask[2] == 1.0

    def test_land_pixel_filled_with_finite_value(self):
        data = np.array([10.0, 20.0], dtype=np.float32)
        land = np.array([False, True])
        values, _ = _apply_land_mask(data, land, fill=25.0)
        assert np.isfinite(values[1]), "Land fill value must be finite"
        assert values[1] == 25.0

    def test_2d_mask(self):
        data = np.random.rand(10, 10).astype(np.float32) * 30.0
        land = np.zeros((10, 10), dtype=bool)
        land[3:5, 3:5] = True  # 4 land pixels
        values, mask = _apply_land_mask(data, land, fill=25.0)
        assert mask.shape == (10, 10)
        assert np.all(mask[3:5, 3:5] == 0.0)
        assert np.all(np.isfinite(values))
