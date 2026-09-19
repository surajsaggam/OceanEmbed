"""
evaluation/phase3_error_analysis.py
------------------------------------
Core diagnostic and error-analysis engine for OceanEmbed Phase-3.

Scientific & Integrity Rules:
1. Pure Analysis: Zero model modifications, zero weight adjustments, zero tuning.
2. Exact Masking: Respects bathymetry and observation validity masks. Invalid cells
   are strictly preserved as NaN, never zero-filled.
3. Rigorous Statistics: Every metric calculation tracks and reports valid sample counts (N).
4. Physical Units: Input features are un-normalized using frozen training moments
   before gradient and speed calculations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.decomposition import PCA


def compute_spatial_gradient_magnitude(
    field: np.ndarray,
    dx: float = 0.25,
    dy: float = 0.25,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Computes horizontal gradient magnitude sqrt((df/dx)^2 + (df/dy)^2).

    Args:
        field: 2D array [H, W] or 3D array [N, H, W]
        dx: Grid spacing in x (degrees or km)
        dy: Grid spacing in y (degrees or km)
        mask: Optional binary mask (1 = valid ocean, 0 = land/invalid)

    Returns:
        Gradient magnitude array of same shape, with invalid regions set to NaN.
    """
    if field.ndim == 2:
        gy, gx = np.gradient(field, dy, dx)
        grad_mag = np.sqrt(gx**2 + gy**2)
        if mask is not None:
            grad_mag[mask == 0] = np.nan
        return grad_mag
    elif field.ndim == 3:
        # field is [N, H, W]
        N, H, W = field.shape
        grad_mag = np.zeros_like(field, dtype=np.float32)
        for i in range(N):
            f_i = field[i]
            gy, gx = np.gradient(f_i, dy, dx)
            mag = np.sqrt(gx**2 + gy**2)
            if mask is not None:
                m_i = mask[i] if mask.ndim == 3 else mask
                mag[m_i == 0] = np.nan
            grad_mag[i] = mag
        return grad_mag
    else:
        raise ValueError(f"Expected 2D or 3D field, got ndim={field.ndim}")


def compute_time_aggregated_spatial_errors(
    preds: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Computes time-aggregated spatial error fields across N samples.

    Args:
        preds: [N, D, H, W]
        targets: [N, D, H, W]
        masks: [N, D, H, W] (1.0 = valid ocean, 0.0 = invalid/bathymetry)

    Returns:
        Dict containing:
          - "signed_error": [D, H, W] mean(pred - target) over time
          - "abs_error": [D, H, W] mean(|pred - target|) over time
          - "rmse": [D, H, W] sqrt(mean((pred - target)^2)) over time
          - "valid_count": [D, H, W] integer counts of valid time steps
    """
    N, D, H, W = preds.shape
    valid = (masks == 1.0) & np.isfinite(preds) & np.isfinite(targets)

    # Accumulate sums and counts along time axis (N)
    valid_count = np.sum(valid, axis=0)  # [D, H, W]

    diff = np.where(valid, preds - targets, 0.0)
    abs_diff = np.where(valid, np.abs(preds - targets), 0.0)
    sq_diff = np.where(valid, (preds - targets) ** 2, 0.0)

    sum_diff = np.sum(diff, axis=0)
    sum_abs_diff = np.sum(abs_diff, axis=0)
    sum_sq_diff = np.sum(sq_diff, axis=0)

    # Safe division: cells with 0 valid samples become NaN
    signed_error = np.full((D, H, W), np.nan, dtype=np.float32)
    abs_error = np.full((D, H, W), np.nan, dtype=np.float32)
    rmse = np.full((D, H, W), np.nan, dtype=np.float32)

    has_data = valid_count > 0
    signed_error[has_data] = sum_diff[has_data] / valid_count[has_data]
    abs_error[has_data] = sum_abs_diff[has_data] / valid_count[has_data]
    rmse[has_data] = np.sqrt(sum_sq_diff[has_data] / valid_count[has_data])

    return {
        "signed_error": signed_error,
        "abs_error": abs_error,
        "rmse": rmse,
        "valid_count": valid_count,
    }


def compute_depth_metrics_with_counts(
    pred: np.ndarray,
    target: np.ndarray,
    depths: List[int],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[int, Union[float, int]]]:
    """
    Computes depth-wise RMSE, MAE, Bias, Pearson r, R^2, and exact valid sample count N.
    """
    D = pred.shape[1] if pred.ndim == 4 else pred.shape[0]
    results = {
        "rmse": {},
        "mae": {},
        "bias": {},
        "pearson_r": {},
        "r2_score": {},
        "valid_count": {},
    }

    for d in range(D):
        depth_val = depths[d]
        p_d = pred[:, d] if pred.ndim == 4 else pred[d]
        t_d = target[:, d] if target.ndim == 4 else target[d]

        if mask is not None:
            if mask.ndim == pred.ndim:
                m_d = (mask[:, d] == 1.0) if mask.ndim == 4 else (mask[d] == 1.0)
            elif mask.ndim == 2:
                m_d = (mask == 1.0)
            else:
                m_d = np.isfinite(t_d)
            valid = m_d & np.isfinite(p_d) & np.isfinite(t_d)
        else:
            valid = np.isfinite(p_d) & np.isfinite(t_d)

        count = int(np.sum(valid))
        results["valid_count"][depth_val] = count

        if count < 2:
            results["rmse"][depth_val] = float("nan")
            results["mae"][depth_val] = float("nan")
            results["bias"][depth_val] = float("nan")
            results["pearson_r"][depth_val] = float("nan")
            results["r2_score"][depth_val] = float("nan")
            continue

        p_vals = p_d[valid].astype(np.float64)
        t_vals = t_d[valid].astype(np.float64)
        err = p_vals - t_vals

        results["rmse"][depth_val] = float(np.sqrt(np.mean(err**2)))
        results["mae"][depth_val] = float(np.mean(np.abs(err)))
        results["bias"][depth_val] = float(np.mean(err))

        p_mean = np.mean(p_vals)
        t_mean = np.mean(t_vals)
        p_diff = p_vals - p_mean
        t_diff = t_vals - t_mean

        num = np.sum(p_diff * t_diff)
        den = np.sqrt(np.sum(p_diff**2) * np.sum(t_diff**2))
        results["pearson_r"][depth_val] = float(num / den) if den > 1e-12 else float("nan")

        ss_res = np.sum(err**2)
        ss_tot = np.sum(t_diff**2)
        results["r2_score"][depth_val] = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else float("nan")

    return results


def compute_binned_statistics(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Computes binned statistics (mean, std, median, count) of y across equal-frequency or equal-width bins of x.
    """
    valid = np.isfinite(x_vals) & np.isfinite(y_vals)
    x = x_vals[valid]
    y = y_vals[valid]

    if len(x) == 0:
        return {"bins": [], "overall_pearson_r": float("nan"), "sample_count": 0}

    # Pearson r
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sqrt(np.sum((x - x_mean)**2) * np.sum((y - y_mean)**2))
    corr = float(num / den) if den > 1e-12 else float("nan")

    # Quantile bins
    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(x, quantiles)
    # Ensure unique edges
    bin_edges = np.unique(bin_edges)

    bin_data = []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            in_bin = (x >= low) & (x <= high)
        else:
            in_bin = (x >= low) & (x < high)

        count = int(np.sum(in_bin))
        if count > 0:
            bin_data.append({
                "bin_low": float(low),
                "bin_high": float(high),
                "x_mean": float(np.mean(x[in_bin])),
                "y_mean": float(np.mean(y[in_bin])),
                "y_median": float(np.median(y[in_bin])),
                "y_std": float(np.std(y[in_bin])),
                "count": count,
            })

    return {
        "overall_pearson_r": corr,
        "sample_count": int(len(x)),
        "bins": bin_data,
    }


def perform_embedding_pca(
    embeddings: np.ndarray,
    n_components: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Performs PCA on flattened [M, 128] embedding vectors.

    Returns:
        transformed embeddings [M, n_components]
        metadata dict (explained variance ratio, total variance explained)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(embeddings)
    meta = {
        "n_components": n_components,
        "explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "total_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    }
    return transformed, meta
