"""
tests/test_phase3_analysis.py
------------------------------
Unit tests for Phase-3 diagnostic and error analysis functions.
"""

import numpy as np
import pytest

from evaluation.phase3_error_analysis import (
    compute_binned_statistics,
    compute_depth_metrics_with_counts,
    compute_spatial_gradient_magnitude,
    compute_time_aggregated_spatial_errors,
    perform_embedding_pca,
)


def test_spatial_gradient_magnitude():
    # Linear ramp in x: f(y, x) = 2.0 * x (dx = 1.0, dy = 1.0) -> gx = 2.0, gy = 0.0 -> mag = 2.0
    field = np.tile(np.linspace(0, 20, 11), (5, 1))
    grad = compute_spatial_gradient_magnitude(field, dx=2.0, dy=1.0)
    assert grad.shape == (5, 11)
    # Interior points should have gx = 1.0, gy = 0.0 -> mag = 1.0
    assert np.allclose(grad[1:4, 1:10], 1.0, atol=1e-5)

    # Test with mask
    mask = np.ones((5, 11))
    mask[0, 0] = 0.0
    grad_masked = compute_spatial_gradient_magnitude(field, dx=2.0, dy=1.0, mask=mask)
    assert np.isnan(grad_masked[0, 0])
    assert np.isfinite(grad_masked[2, 2])


def test_time_aggregated_spatial_errors():
    N, D, H, W = 10, 3, 4, 4
    preds = np.zeros((N, D, H, W), dtype=np.float32)
    targets = np.ones((N, D, H, W), dtype=np.float32)
    masks = np.ones((N, D, H, W), dtype=np.float32)

    # Set some cells invalid
    masks[:, :, 0, 0] = 0.0

    # With preds=0 and targets=1, signed_error = -1, abs_error = 1, rmse = 1
    res = compute_time_aggregated_spatial_errors(preds, targets, masks)

    assert np.isnan(res["signed_error"][:, 0, 0]).all()
    assert np.isnan(res["abs_error"][:, 0, 0]).all()
    assert np.isnan(res["rmse"][:, 0, 0]).all()
    assert (res["valid_count"][:, 0, 0] == 0).all()

    # Valid ocean cells
    assert np.allclose(res["signed_error"][:, 1, 1], -1.0)
    assert np.allclose(res["abs_error"][:, 1, 1], 1.0)
    assert np.allclose(res["rmse"][:, 1, 1], 1.0)
    assert (res["valid_count"][:, 1, 1] == N).all()


def test_compute_depth_metrics_with_counts():
    depths = [0, 50, 100]
    pred = np.array([
        [[10.0, 12.0], [14.0, 16.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[2.0, 2.0], [2.0, 2.0]],
    ])  # [3, 2, 2]
    target = pred + 1.0  # constant error of -1.0
    mask = np.ones((3, 2, 2))
    mask[0, 0, 0] = 0.0  # 1 invalid pixel in depth 0

    metrics = compute_depth_metrics_with_counts(pred, target, depths, mask=mask)

    assert metrics["valid_count"][0] == 3
    assert metrics["valid_count"][50] == 4
    assert metrics["valid_count"][100] == 4

    assert np.isclose(metrics["rmse"][0], 1.0)
    assert np.isclose(metrics["mae"][0], 1.0)
    assert np.isclose(metrics["bias"][0], -1.0)


def test_binned_statistics():
    x = np.linspace(0, 10, 100)
    y = 2.0 * x + np.random.normal(0, 0.01, 100)
    res = compute_binned_statistics(x, y, n_bins=5)

    assert res["sample_count"] == 100
    assert np.isclose(res["overall_pearson_r"], 1.0, atol=1e-2)
    assert len(res["bins"]) == 5
    total_binned = sum(b["count"] for b in res["bins"])
    assert total_binned == 100


def test_perform_embedding_pca():
    embeddings = np.random.randn(50, 128)
    transformed, meta = perform_embedding_pca(embeddings, n_components=3)

    assert transformed.shape == (50, 3)
    assert meta["n_components"] == 3
    assert len(meta["explained_variance_ratio"]) == 3
    assert 0.0 < meta["total_explained_variance"] <= 1.0
