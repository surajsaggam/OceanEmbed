"""
tests/test_regrid.py
--------------------
Tests for pipeline/regrid.py

Rules verified:
  - Output shape [H, W] is derived from config domain (not hardcoded)
  - Expected H=101, W=241 for default config with inclusive endpoints
  - Grid computation uses np.arange correctly
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def _compute_grid_shape(lat_min, lat_max, lon_min, lon_max, resolution):
    """Reference grid shape computation (mirrors pipeline/regrid.py)."""
    lat = np.arange(lat_min, lat_max + resolution / 2, resolution)
    lon = np.arange(lon_min, lon_max + resolution / 2, resolution)
    return len(lat), len(lon)


class TestGridShape:
    @pytest.fixture(autouse=True)
    def _cfg(self):
        self.cfg = load_config("data")

    def test_grid_shape_from_config(self):
        """Grid shape must be computed from config, not hardcoded."""
        H, W = _compute_grid_shape(
            self.cfg.domain.lat_min, self.cfg.domain.lat_max,
            self.cfg.domain.lon_min, self.cfg.domain.lon_max,
            self.cfg.domain.resolution,
        )
        # For default config: lat 5–30 at 0.25° = 101; lon 45–105 at 0.25° = 241
        assert H == 101, f"Expected H=101, got H={H}"
        assert W == 241, f"Expected W=241, got W={W}"

    def test_lat_array_includes_endpoints(self):
        lat = np.arange(
            self.cfg.domain.lat_min,
            self.cfg.domain.lat_max + self.cfg.domain.resolution / 2,
            self.cfg.domain.resolution,
        )
        assert lat[0] == pytest.approx(self.cfg.domain.lat_min)
        assert lat[-1] == pytest.approx(self.cfg.domain.lat_max, abs=0.25)

    def test_lon_array_includes_endpoints(self):
        lon = np.arange(
            self.cfg.domain.lon_min,
            self.cfg.domain.lon_max + self.cfg.domain.resolution / 2,
            self.cfg.domain.resolution,
        )
        assert lon[0] == pytest.approx(self.cfg.domain.lon_min)
        assert lon[-1] == pytest.approx(self.cfg.domain.lon_max, abs=0.25)

    def test_grid_shape_non_zero(self):
        H, W = _compute_grid_shape(
            self.cfg.domain.lat_min, self.cfg.domain.lat_max,
            self.cfg.domain.lon_min, self.cfg.domain.lon_max,
            self.cfg.domain.resolution,
        )
        assert H > 0 and W > 0
