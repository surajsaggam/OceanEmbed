"""
tests/conftest.py
-----------------
Shared pytest fixtures providing synthetic data for all unit tests.

Design rules:
  - No real NetCDF or Argo files are loaded here.
  - All shapes are derived from the config, not hardcoded constants.
  - Tests run in seconds on any machine, with or without GPU.
  - The fixtures expose the same interface that real pipeline outputs will use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is on path (handles running pytest from any directory)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


# ── Load config once for the whole test session ───────────────────────────

@pytest.fixture(scope="session")
def data_cfg():
    return load_config("data")


@pytest.fixture(scope="session")
def model_cfg():
    return load_config("model")


# ── Derived grid shape from config ────────────────────────────────────────

@pytest.fixture(scope="session")
def grid_shape(data_cfg):
    """Returns (H, W) computed from config domain, not hardcoded."""
    lat = np.arange(
        data_cfg.domain.lat_min,
        data_cfg.domain.lat_max + data_cfg.domain.resolution,
        data_cfg.domain.resolution,
    )
    lon = np.arange(
        data_cfg.domain.lon_min,
        data_cfg.domain.lon_max + data_cfg.domain.resolution,
        data_cfg.domain.resolution,
    )
    return len(lat), len(lon)


# ── Core tensor fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def n_depths(data_cfg):
    return len(data_cfg.depths_m)


@pytest.fixture(scope="session")
def in_channels(model_cfg):
    return model_cfg.in_channels  # 14


@pytest.fixture
def synthetic_surface_batch(grid_shape, in_channels):
    """
    Synthetic input tensor: [B=2, 14, H, W] float32.
    Channels 0–6: normalised surface vars (finite, ~N(0,1))
    Channels 7–13: validity masks (binary float32)
    All values are finite — matches the Dataset isfinite contract.
    """
    B = 2
    H, W = grid_shape
    C = in_channels

    x = torch.zeros(B, C, H, W, dtype=torch.float32)
    # Physical channels: random normal (simulates normalised observations)
    x[:, :7, :, :] = torch.randn(B, 7, H, W)
    # Mask channels: binary {0, 1}
    x[:, 7:, :, :] = torch.randint(0, 2, (B, 7, H, W)).float()

    assert torch.isfinite(x).all(), "Synthetic input contains non-finite values"
    return x


@pytest.fixture
def synthetic_target_batch(grid_shape, n_depths):
    """
    Synthetic target tensor: [B=2, 15, H, W] float32.
    Represents GLORYS temperature (un-normalised, ~[0, 30] °C).
    """
    B = 2
    H, W = grid_shape
    D = n_depths
    return torch.rand(B, D, H, W, dtype=torch.float32) * 30.0


@pytest.fixture
def synthetic_target_mask(grid_shape, n_depths):
    """
    Target validity mask: [B=2, 15, H, W] bool.
    True = valid ocean pixel. ~70% valid.
    """
    B = 2
    H, W = grid_shape
    D = n_depths
    return torch.rand(B, D, H, W) > 0.30


# ── Small overfit fixtures ────────────────────────────────────────────────

@pytest.fixture
def tiny_overfit_batch(grid_shape, in_channels, n_depths):
    """
    Very small fixed batch for the synthetic overfit test.
    Uses a small spatial crop [4, 4] to be fast on CPU.
    """
    B = 4
    C = in_channels
    D = n_depths
    H_crop, W_crop = 4, 4

    torch.manual_seed(42)
    x = torch.zeros(B, C, H_crop, W_crop)
    x[:, :7] = torch.randn(B, 7, H_crop, W_crop)
    x[:, 7:] = (torch.rand(B, 7, H_crop, W_crop) > 0.3).float()

    y = torch.rand(B, D, H_crop, W_crop) * 28.0 + 2.0
    mask = torch.ones(B, D, H_crop, W_crop, dtype=torch.bool)

    return x, y, mask
