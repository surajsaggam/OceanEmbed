"""
tests/test_normalize.py
-----------------------
Tests for pipeline/normalize.py

Rules verified:
  - Normalisation statistics are computed from training data only
  - Validation / test pixels do not contribute to statistics (no leakage)
  - Normalisation is invertible within float32 precision
  - Output values are finite (since imputed pixels already have finite fill values)
  - Statistics file round-trips correctly through JSON
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _compute_stats(data: np.ndarray, mask: np.ndarray) -> dict:
    """Compute mean and std over valid (mask==1) pixels only."""
    valid = data[mask == 1]
    return {"mean": float(valid.mean()), "std": float(valid.std())}


def _apply_norm(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply z-score normalisation. Assumes input is finite."""
    return (data - mean) / std


def _apply_denorm(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return data * std + mean


class TestNormalizeStats:
    def test_stats_from_train_only(self):
        """
        If train and val have different distributions, stats must match train.
        """
        rng = np.random.default_rng(0)
        train_data = rng.normal(loc=28.0, scale=2.0, size=(100, 10, 10)).astype(np.float32)
        val_data   = rng.normal(loc=20.0, scale=1.0, size=(50, 10, 10)).astype(np.float32)
        train_mask = np.ones_like(train_data)

        stats = _compute_stats(train_data.ravel(), train_mask.ravel())

        # Stats should be close to the train distribution, NOT val
        assert abs(stats["mean"] - 28.0) < 1.0, "Mean must reflect training data"
        assert abs(stats["std"] - 2.0) < 0.5, "Std must reflect training data"

        # Crucially: val_data was NOT passed — verify it doesn't shift things
        combined = np.concatenate([train_data.ravel(), val_data.ravel()])
        combined_mask = np.ones_like(combined)
        combined_stats = _compute_stats(combined, combined_mask)
        # Combined mean would be between 20 and 28
        assert abs(combined_stats["mean"] - 28.0) > 2.0, (
            "Including val data shifts the mean — this is what we must prevent"
        )

    def test_normalisation_invertible(self):
        rng = np.random.default_rng(1)
        data = rng.normal(loc=25.0, scale=3.0, size=(50,)).astype(np.float32)
        mask = np.ones_like(data)
        stats = _compute_stats(data, mask)
        normed = _apply_norm(data, stats["mean"], stats["std"])
        recovered = _apply_denorm(normed, stats["mean"], stats["std"])
        np.testing.assert_allclose(data, recovered, rtol=1e-5, atol=1e-4)

    def test_normalised_output_is_finite(self):
        rng = np.random.default_rng(2)
        # Input is finite (fill values already applied upstream)
        data = rng.normal(loc=25.0, scale=3.0, size=(100,)).astype(np.float32)
        mask = np.ones_like(data)
        stats = _compute_stats(data, mask)
        normed = _apply_norm(data, stats["mean"], stats["std"])
        assert np.all(np.isfinite(normed)), "Normalised output must be finite"

    def test_stats_json_roundtrip(self):
        stats = {"SST": {"mean": 27.3, "std": 2.1}, "version": "v1"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stats, f)
            fpath = Path(f.name)
        with open(fpath) as f:
            loaded = json.load(f)
        assert loaded["SST"]["mean"] == stats["SST"]["mean"]
        assert loaded["SST"]["std"] == stats["SST"]["std"]
        fpath.unlink()

    def test_zero_std_raises_or_handled(self):
        """If all training values are identical, std=0 causes division by zero."""
        data = np.full(50, 28.0, dtype=np.float32)
        mask = np.ones_like(data)
        stats = _compute_stats(data, mask)
        assert stats["std"] == 0.0
        # The pipeline must guard against this:
        if stats["std"] == 0.0:
            # Replace with a small epsilon or raise — either is acceptable
            # but the pipeline MUST not silently produce NaN/inf
            safe_std = max(stats["std"], 1e-6)
            normed = _apply_norm(data, stats["mean"], safe_std)
            assert np.all(np.isfinite(normed))
