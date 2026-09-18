"""
evaluation/metrics.py
---------------------
Scientific metrics for subsurface ocean temperature reconstruction.

Supported metrics:
  - Root Mean Square Error (RMSE) by depth
  - Mean Absolute Error (MAE) by depth
  - Mean Bias (pred - target) by depth
  - Pearson correlation coefficient (r) by depth
  - Coefficient of determination (R^2) by depth

Scientific rules:
  - Metrics are evaluated on valid ocean pixels only (mask == 1).
  - Depth-wise reporting is mandatory — never collapse the 15 depths into a single
    misleading scalar accuracy number.
  - Pearson r and R^2 are fundamentally different and calculated separately:
      * r measures linear correlation (-1 to +1).
      * R^2 measures proportion of variance explained (-inf to 1).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def depth_wise_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> List[float]:
    """
    Computes RMSE separately for each depth channel.

    Args:
        pred: Predicted temperature [D, H, W] or [B, D, H, W].
        target: Target temperature [D, H, W] or [B, D, H, W].
        mask: Optional binary mask [H, W] or [D, H, W] or [B, D, H, W].

    Returns:
        List of RMSE values (float) for each depth level.
    """
    if pred.ndim == 4:
        # [B, D, H, W] -> iterate over D
        D = pred.shape[1]
    else:
        D = pred.shape[0]

    rmses = []
    for d in range(D):
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

        if np.sum(valid) == 0:
            rmses.append(float("nan"))
        else:
            err = p_d[valid] - t_d[valid]
            rmses.append(float(np.sqrt(np.mean(err ** 2))))

    return rmses


def depth_wise_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> List[float]:
    """
    Computes MAE separately for each depth channel.
    """
    D = pred.shape[1] if pred.ndim == 4 else pred.shape[0]
    maes = []
    for d in range(D):
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

        if np.sum(valid) == 0:
            maes.append(float("nan"))
        else:
            maes.append(float(np.mean(np.abs(p_d[valid] - t_d[valid]))))
    return maes


def depth_wise_bias(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> List[float]:
    """
    Computes mean bias (pred - target) separately for each depth channel.
    """
    D = pred.shape[1] if pred.ndim == 4 else pred.shape[0]
    biases = []
    for d in range(D):
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

        if np.sum(valid) == 0:
            biases.append(float("nan"))
        else:
            biases.append(float(np.mean(p_d[valid] - t_d[valid])))
    return biases


def depth_wise_r_and_r2(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[List[float], List[float]]:
    """
    Computes Pearson correlation r and coefficient of determination R^2 separately
    for each depth channel.
    """
    D = pred.shape[1] if pred.ndim == 4 else pred.shape[0]
    r_list = []
    r2_list = []

    for d in range(D):
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

        p_vals = p_d[valid]
        t_vals = t_d[valid]

        if len(p_vals) < 2:
            r_list.append(float("nan"))
            r2_list.append(float("nan"))
            continue

        # Pearson r
        p_mean = np.mean(p_vals)
        t_mean = np.mean(t_vals)
        p_diff = p_vals - p_mean
        t_diff = t_vals - t_mean

        numerator = np.sum(p_diff * t_diff)
        denominator = np.sqrt(np.sum(p_diff ** 2) * np.sum(t_diff ** 2))
        r = float(numerator / denominator) if denominator > 1e-12 else float("nan")

        # R^2 = 1 - SS_res / SS_tot
        ss_res = np.sum((t_vals - p_vals) ** 2)
        ss_tot = np.sum(t_diff ** 2)
        r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-12 else float("nan")

        r_list.append(r)
        r2_list.append(r2)

    return r_list, r2_list


def compute_all_depth_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    depths: List[int],
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[int, float]]:
    """
    Computes a comprehensive dictionary of all depth-wise metrics.
    """
    rmses = depth_wise_rmse(pred, target, mask)
    maes = depth_wise_mae(pred, target, mask)
    biases = depth_wise_bias(pred, target, mask)
    r_list, r2_list = depth_wise_r_and_r2(pred, target, mask)

    results = {
        "rmse": {d: rmses[i] for i, d in enumerate(depths)},
        "mae": {d: maes[i] for i, d in enumerate(depths)},
        "bias": {d: biases[i] for i, d in enumerate(depths)},
        "pearson_r": {d: r_list[i] for i, d in enumerate(depths)},
        "r2_score": {d: r2_list[i] for i, d in enumerate(depths)},
    }
    return results
