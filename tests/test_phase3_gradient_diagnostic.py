"""
tests/test_phase3_gradient_diagnostic.py
-----------------------------------------
Unit tests for the Phase-3 vertical gradient and profile displacement diagnostic engine.
"""

import numpy as np
import pytest

from evaluation.phase3_gradient_diagnostic import (
    analyze_profile_displacement_population,
    compute_gradient_vs_error_binned,
    compute_interval_valid_masks,
    compute_layer_gradient_metrics,
    compute_vertical_gradients_adjacent,
    estimate_peak_gradient_depth,
)


def test_compute_vertical_gradients_adjacent():
    depths = [0, 10, 30, 70]
    temp = np.array([25.0, 23.0, 19.0, 11.0])  # [4]
    grad, midpoints = compute_vertical_gradients_adjacent(temp, depths)

    assert grad.shape == (3,)
    assert np.allclose(midpoints, [5.0, 20.0, 50.0])
    # dT/dz:
    # 0->10: (23 - 25) / 10 = -0.2
    # 10->30: (19 - 23) / 20 = -0.2
    # 30->70: (11 - 19) / 40 = -0.2
    assert np.allclose(grad, [-0.2, -0.2, -0.2])


def test_compute_interval_valid_masks():
    masks = np.array([
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0, 1.0],
    ])
    interval_m = compute_interval_valid_masks(masks)
    assert interval_m.shape == (2, 3)
    # Row 0: [1&1, 1&1, 1&0] -> [True, True, False]
    assert np.array_equal(interval_m[0], [True, True, False])
    # Row 1: [1&0, 0&1, 1&1] -> [False, False, True]
    assert np.array_equal(interval_m[1], [False, False, True])


def test_compute_layer_gradient_metrics():
    pred_g = np.array([[ -0.1, -0.3, -0.05 ]])
    target_g = np.array([[ -0.15, -0.2, -0.05 ]])
    m_arr = np.array([[ True, True, True ]])
    midpoints = np.array([25.0, 100.0, 400.0])

    layers = {
        "upper": (0.0, 50.0),
        "mid": (50.0, 150.0),
        "deep": (150.0, 500.0),
    }

    metrics = compute_layer_gradient_metrics(pred_g, target_g, m_arr, midpoints, layers)
    # mid layer error: -0.3 - (-0.2) = -0.1 -> MAE = 0.1, RMSE = 0.1, bias = -0.1
    assert np.isclose(metrics["mid"]["mae"], 0.1)
    assert np.isclose(metrics["mid"]["rmse"], 0.1)
    assert np.isclose(metrics["mid"]["bias"], -0.1)


def test_compute_gradient_vs_error_binned():
    grads = np.linspace(0.01, 0.15, 100)
    errors = grads * 10.0 + np.random.normal(0, 0.001, 100)
    binned = compute_gradient_vs_error_binned(grads, errors, n_bins=5)

    assert binned["sample_count"] == 100
    assert np.isclose(binned["pearson_r"], 1.0, atol=1e-2)
    assert np.isclose(binned["spearman_rho"], 1.0, atol=1e-2)
    assert len(binned["bins"]) == 5


def test_estimate_peak_gradient_and_displacement():
    depths = [0, 50, 75, 100, 125, 150, 200]
    # Midpoints: 25, 62.5, 87.5, 112.5, 137.5, 175

    # Target has peak gradient between 75 and 100m (midpoint 87.5m)
    t_temp = np.array([28.0, 27.0, 25.0, 18.0, 16.0, 15.0, 14.0])
    # Pred has peak gradient between 100 and 125m (midpoint 112.5m)
    p_temp = np.array([28.0, 27.5, 26.5, 25.0, 18.0, 16.0, 14.0])

    t_g, mids = compute_vertical_gradients_adjacent(t_temp, depths)
    p_g, _ = compute_vertical_gradients_adjacent(p_temp, depths)

    t_peak_z, _, _ = estimate_peak_gradient_depth(t_g, mids)
    p_peak_z, _, _ = estimate_peak_gradient_depth(p_g, mids)

    assert np.isclose(t_peak_z, 87.5)
    assert np.isclose(p_peak_z, 112.5)

    # Test population analysis with these two profiles
    pred_profiles = np.stack([p_temp, p_temp])
    targ_profiles = np.stack([t_temp, t_temp])
    masks = np.ones_like(pred_profiles)

    res = analyze_profile_displacement_population(
        pred_profiles, targ_profiles, depths, masks, min_eval_depth=200.0, min_peak_grad=0.03
    )

    assert res["eligible_profile_count"] == 2
    assert np.isclose(res["discrete_displacement"]["mean_m"], 25.0)
