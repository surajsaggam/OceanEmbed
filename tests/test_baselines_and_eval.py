"""
tests/test_baselines_and_eval.py
--------------------------------
Unit tests for baselines (Climatology, Ridge) and evaluation metrics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.climatology import ClimatologyBaseline
from baselines.ridge import ChunkedRidgeBaseline
from evaluation.metrics import (
    compute_all_depth_metrics,
    depth_wise_bias,
    depth_wise_mae,
    depth_wise_r_and_r2,
    depth_wise_rmse,
)
from inference.predictor import OceanEmbedPredictor


class TestBaselines:
    def test_climatology_baseline(self):
        H, W, D = 10, 10, 15
        base = ClimatologyBaseline(n_depths=D, grid_shape=(H, W))

        # Update with month 1 data
        target = np.ones((D, H, W), dtype=np.float32) * 20.0
        base.update(month=1, target=target)
        base.finalize()

        pred = base.predict(month=1)
        assert pred.shape == (D, H, W)
        np.testing.assert_allclose(pred, 20.0)

    def test_chunked_ridge_baseline(self):
        in_dim = 14
        out_dim = 15
        ridge = ChunkedRidgeBaseline(in_features=in_dim, out_features=out_dim, alpha=1.0)

        rng = np.random.default_rng(42)
        X = rng.normal(size=(in_dim, 20, 20)).astype(np.float32)
        # Linear relationship: y = 2 * x[0] + 5
        Y = np.zeros((out_dim, 20, 20), dtype=np.float32)
        Y[:] = 2.0 * X[0] + 5.0

        ridge.update_chunk(X, Y)
        ridge.finalize()

        assert ridge.is_fitted
        assert ridge.weights.shape == (in_dim, out_dim)
        assert ridge.bias.shape == (out_dim,)

        pred = ridge.predict(X)
        assert pred.shape == (out_dim, 20, 20)
        # Ridge should recover the linear relationship with high R2
        corr = np.corrcoef(pred[0].ravel(), Y[0].ravel())[0, 1]
        assert corr > 0.95, f"Expected strong correlation, got {corr}"

    def test_cnn_only_baseline(self):
        from models.cnn_baseline import CNNOnlyBaseline
        model = CNNOnlyBaseline()
        x = torch.randn(2, 14, 16, 24)
        out = model(x)
        assert "temperature" in out
        assert "embedding" in out
        assert out["temperature"].shape == (2, 15, 16, 24)
        assert out["embedding"].shape == (2, 128, 16, 24)


class TestMetrics:
    def test_depth_metrics_exact(self):
        D, H, W = 15, 8, 8
        pred = np.full((D, H, W), 25.0, dtype=np.float32)
        target = np.full((D, H, W), 23.0, dtype=np.float32)

        depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        results = compute_all_depth_metrics(pred, target, depths)

        # Difference is exactly 2.0 everywhere
        for d in depths:
            assert pytest.approx(results["rmse"][d], abs=1e-4) == 2.0
            assert pytest.approx(results["mae"][d], abs=1e-4) == 2.0
            assert pytest.approx(results["bias"][d], abs=1e-4) == 2.0

    def test_r_and_r2_distinction(self):
        # r measures correlation, R2 measures fraction of variance explained
        rng = np.random.default_rng(0)
        t = rng.normal(size=(1, 50, 50)).astype(np.float32)
        p = t * 2.0 + 10.0  # Perfect correlation (r=1.0), but bad calibration (R2 < 1.0)

        r_list, r2_list = depth_wise_r_and_r2(p, t)
        assert pytest.approx(r_list[0], abs=1e-4) == 1.0
        assert r2_list[0] < 1.0, "R2 must reflect calibration scale error, not just r"


class TestPredictor:
    def test_predictor_synthetic_inference(self):
        predictor = OceanEmbedPredictor(device="cpu")
        x = np.ones((14, 20, 20), dtype=np.float32)
        out = predictor.predict(x, return_embedding=True, return_attention=True)

        assert "temperature" in out
        assert "embedding" in out
        assert "attention" in out
        assert out["temperature"].shape == (15, 20, 20)
        assert out["embedding"].shape == (128, 20, 20)
        assert out["attention"].shape == (1, 20, 20)
