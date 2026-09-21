"""
evaluation/phase3_gradient_diagnostic.py
-----------------------------------------
Diagnostic engine testing the vertical-gradient and profile-displacement hypothesis
for OceanEmbed Phase-3.

Scientific & Integrity Rules:
1. Nonuniform Depth Handling: Gradients are computed strictly accounting for nonuniform
   depth spacing (dT/dz = (T[d+1] - T[d]) / (z[d+1] - z[d])).
2. Pure Analysis: Zero model modification, zero weight training.
3. Rigorous Masking: Bathymetry masks are strictly respected; intervals are valid only
   if both bounding depths are valid ocean.
4. Quantized vs Continuous Displacement: Explicitly accounts for the discrete 15-depth
   resolution limits when estimating peak-gradient depths.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_vertical_gradients_adjacent(
    temp: np.ndarray,
    depths: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes vertical temperature gradients dT/dz for adjacent depth intervals.

    Args:
        temp: Temperature array of shape [..., D] where last dimension matches depths.
        depths: List of depth levels in meters (length D).

    Returns:
        grad: Gradient array of shape [..., D-1] in deg C / m (negative values indicate cooling with depth).
        midpoints: 1D array of length D-1 with midpoint depths in meters.
    """
    z = np.array(depths, dtype=np.float64)
    dz = np.diff(z)  # [D-1]
    if np.any(dz <= 0):
        raise ValueError("Depths must be strictly increasing.")

    # dT = T[d+1] - T[d]
    dT = np.diff(temp, axis=-1)  # [..., D-1]
    # Broadcast dz across leading dimensions
    grad = dT / dz
    midpoints = (z[:-1] + z[1:]) / 2.0
    return grad.astype(np.float32), midpoints


def compute_interval_valid_masks(
    masks: np.ndarray,
) -> np.ndarray:
    """
    Computes valid mask for adjacent depth intervals. An interval is valid (1.0)
    if and only if both bounding depths are valid (mask == 1.0).

    Args:
        masks: Binary masks [..., D]

    Returns:
        interval_mask: [..., D-1]
    """
    return (masks[..., :-1] == 1.0) & (masks[..., 1:] == 1.0)


def compute_layer_gradient_metrics(
    pred_grad: np.ndarray,
    target_grad: np.ndarray,
    interval_masks: np.ndarray,
    midpoints: np.ndarray,
    layer_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Computes gradient MAE, RMSE, Bias, and valid count N across depth layers.

    Args:
        pred_grad: [..., D-1]
        target_grad: [..., D-1]
        interval_masks: [..., D-1] (boolean or float)
        midpoints: [D-1]
        layer_ranges: Dict mapping layer name to (min_depth, max_depth) inclusive.

    Returns:
        Dict of layer metrics.
    """
    results = {}
    for layer_name, (z_min, z_max) in layer_ranges.items():
        layer_indices = np.where((midpoints >= z_min) & (midpoints <= z_max))[0]
        if len(layer_indices) == 0:
            continue

        p_layer = pred_grad[..., layer_indices]
        t_layer = target_grad[..., layer_indices]
        m_layer = interval_masks[..., layer_indices]

        valid = (m_layer == 1.0) & np.isfinite(p_layer) & np.isfinite(t_layer)
        count = int(np.sum(valid))

        if count < 1:
            results[layer_name] = {
                "mae": float("nan"),
                "rmse": float("nan"),
                "bias": float("nan"),
                "target_mean_abs_grad": float("nan"),
                "pred_mean_abs_grad": float("nan"),
                "valid_count": count,
            }
            continue

        err = (p_layer[valid] - t_layer[valid]).astype(np.float64)
        t_vals = t_layer[valid].astype(np.float64)
        p_vals = p_layer[valid].astype(np.float64)

        results[layer_name] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "bias": float(np.mean(err)),
            "target_mean_abs_grad": float(np.mean(np.abs(t_vals))),
            "pred_mean_abs_grad": float(np.mean(np.abs(p_vals))),
            "valid_count": count,
        }

    return results


def compute_gradient_vs_error_binned(
    target_abs_grad: np.ndarray,
    temp_abs_err: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes correlation (Pearson and Spearman) and binned statistics
    between target vertical gradient magnitude and temperature absolute error.
    """
    valid = np.isfinite(target_abs_grad) & np.isfinite(temp_abs_err)
    x = target_abs_grad[valid].astype(np.float64)
    y = temp_abs_err[valid].astype(np.float64)

    if len(x) < 5:
        return {
            "pearson_r": float("nan"),
            "pearson_p": float("nan"),
            "spearman_rho": float("nan"),
            "spearman_p": float("nan"),
            "sample_count": int(len(x)),
            "bins": [],
        }

    r_val, r_p = pearsonr(x, y)
    rho_val, rho_p = spearmanr(x, y)

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.unique(np.percentile(x, quantiles))

    bins = []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        in_bin = (x >= low) & (x <= high) if i == len(bin_edges) - 2 else (x >= low) & (x < high)
        count = int(np.sum(in_bin))
        if count > 0:
            y_in = y[in_bin]
            bins.append({
                "bin_low": float(low),
                "bin_high": float(high),
                "grad_mean": float(np.mean(x[in_bin])),
                "mae_mean": float(np.mean(y_in)),
                "mae_median": float(np.median(y_in)),
                "rmse": float(np.sqrt(np.mean(y_in**2))),
                "count": count,
            })

    return {
        "pearson_r": float(r_val),
        "pearson_p": float(r_p),
        "spearman_rho": float(rho_val),
        "spearman_p": float(rho_p),
        "sample_count": int(len(x)),
        "bins": bins,
    }


def estimate_peak_gradient_depth(
    profile_grad: np.ndarray,
    midpoints: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Finds the depth of the maximum absolute vertical gradient.

    Returns:
        discrete_peak_depth: Midpoint depth with maximum |dT/dz|
        discrete_peak_magnitude: Peak |dT/dz|
        subgrid_peak_depth: Parabolic sub-grid interpolation peak depth
    """
    abs_g = np.abs(profile_grad)
    idx = int(np.argmax(abs_g))
    disc_depth = float(midpoints[idx])
    disc_mag = float(abs_g[idx])

    # 3-point parabolic interpolation for subgrid peak if interior point
    if 0 < idx < len(midpoints) - 1:
        y0, y1, y2 = abs_g[idx - 1], abs_g[idx], abs_g[idx + 1]
        denom = 2.0 * (2.0 * y1 - y0 - y2)
        if denom > 1e-12:
            delta = (y0 - y2) / denom  # shift between -0.5 and +0.5 of interval spacing
            # Interval spacing around midpoint
            h = (midpoints[idx + 1] - midpoints[idx - 1]) / 2.0
            subgrid_depth = float(disc_depth + delta * h)
        else:
            subgrid_depth = disc_depth
    else:
        subgrid_depth = disc_depth

    return disc_depth, disc_mag, subgrid_depth


def analyze_profile_displacement_population(
    pred_profiles: np.ndarray,
    target_profiles: np.ndarray,
    depths: List[int],
    profile_masks: np.ndarray,
    min_eval_depth: float = 200.0,
    min_peak_grad: float = 0.03,
) -> Dict[str, Any]:
    """
    Evaluates vertical profile peak-gradient displacement across eligible columns.

    Args:
        pred_profiles: [M, D]
        target_profiles: [M, D]
        depths: List of depth levels (length D)
        profile_masks: [M, D]
        min_eval_depth: Profile must be valid to at least this depth
        min_peak_grad: Minimum peak gradient (deg C / m) to qualify as a strong-gradient profile

    Returns:
        Dict with displacement statistics, distribution, and resolution caveats.
    """
    pred_g, midpoints = compute_vertical_gradients_adjacent(pred_profiles, depths)
    targ_g, _ = compute_vertical_gradients_adjacent(target_profiles, depths)
    interval_m = compute_interval_valid_masks(profile_masks)

    # Search window: restrict search to depths <= min_eval_depth
    search_indices = np.where(midpoints <= min_eval_depth)[0]

    displacements_discrete = []
    displacements_subgrid = []
    profile_maes = []
    profile_rmses = []
    peak_target_grads = []

    M = pred_profiles.shape[0]
    for i in range(M):
        # Check eligibility: all intervals in search window must be valid
        if not np.all(interval_m[i, search_indices] == 1.0):
            continue

        p_g_win = pred_g[i, search_indices]
        t_g_win = targ_g[i, search_indices]
        m_win = midpoints[search_indices]

        t_disc_z, t_peak_g, t_sub_z = estimate_peak_gradient_depth(t_g_win, m_win)
        if t_peak_g < min_peak_grad:
            # Skip profiles without a pronounced vertical gradient
            continue

        p_disc_z, p_peak_g, p_sub_z = estimate_peak_gradient_depth(p_g_win, m_win)

        disp_disc = p_disc_z - t_disc_z
        disp_sub = p_sub_z - t_sub_z

        # Profile error over search window depths
        win_depth_indices = [k for k, d in enumerate(depths) if d <= min_eval_depth]
        err_prof = pred_profiles[i, win_depth_indices] - target_profiles[i, win_depth_indices]
        p_mae = float(np.mean(np.abs(err_prof)))
        p_rmse = float(np.sqrt(np.mean(err_prof**2)))

        displacements_discrete.append(disp_disc)
        displacements_subgrid.append(disp_sub)
        profile_maes.append(p_mae)
        profile_rmses.append(p_rmse)
        peak_target_grads.append(t_peak_g)

    N_eligible = len(displacements_discrete)
    if N_eligible < 1:
        return {
            "eligible_profile_count": N_eligible,
            "error": "Insufficient eligible profiles meeting depth and gradient thresholds",
        }

    disp_disc_arr = np.array(displacements_discrete)
    disp_sub_arr = np.array(displacements_subgrid)
    p_maes_arr = np.array(profile_maes)
    p_rmses_arr = np.array(profile_rmses)

    # Correlation between absolute displacement and profile temperature error
    abs_disp_sub = np.abs(disp_sub_arr)
    if N_eligible >= 2 and np.std(abs_disp_sub) > 1e-12 and np.std(p_maes_arr) > 1e-12:
        corr_disp_mae, _ = pearsonr(abs_disp_sub, p_maes_arr)
        corr_disp_rmse, _ = pearsonr(abs_disp_sub, p_rmses_arr)
        spear_disp_mae, _ = spearmanr(abs_disp_sub, p_maes_arr)
    else:
        corr_disp_mae, corr_disp_rmse, spear_disp_mae = float("nan"), float("nan"), float("nan")

    # Discrete histogram
    unique_vals, counts = np.unique(disp_disc_arr, return_counts=True)
    hist_discrete = {str(float(v)): int(c) for v, c in zip(unique_vals, counts)}

    return {
        "eligible_profile_count": N_eligible,
        "qualification_criteria": {
            "min_eval_depth_m": min_eval_depth,
            "min_peak_gradient_degC_per_m": min_peak_grad,
        },
        "discrete_displacement": {
            "mean_m": float(np.mean(disp_disc_arr)),
            "median_m": float(np.median(disp_disc_arr)),
            "mean_abs_m": float(np.mean(np.abs(disp_disc_arr))),
            "median_abs_m": float(np.median(np.abs(disp_disc_arr))),
            "histogram": hist_discrete,
        },
        "subgrid_displacement": {
            "mean_m": float(np.mean(disp_sub_arr)),
            "median_m": float(np.median(disp_sub_arr)),
            "mean_abs_m": float(np.mean(abs_disp_sub)),
            "median_abs_m": float(np.median(abs_disp_sub)),
            "std_m": float(np.std(disp_sub_arr)),
        },
        "correlation_displacement_vs_temperature_error": {
            "pearson_r_abs_disp_vs_mae": float(corr_disp_mae),
            "spearman_rho_abs_disp_vs_mae": float(spear_disp_mae),
            "pearson_r_abs_disp_vs_rmse": float(corr_disp_rmse),
        },
        "resolution_caveats": (
            "The standard 15-depth grid has 25m interval spacing between 50m and 150m. "
            "Discrete displacement estimates are quantized in steps of +/-25m. "
            "Subgrid estimates use 3-point parabolic interpolation but remain constrained by grid resolution."
        ),
    }
