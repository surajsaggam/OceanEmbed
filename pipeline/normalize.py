"""
pipeline/normalize.py
---------------------
Normalization and denormalization routines for OceanEmbed.

Scientific rules:
  - Z-score normalization statistics (mean, std) must be computed ONLY from
    training split observations where validity_mask == 1.
  - Validation, test, and independent Argo observations MUST NEVER contribute to
    normalization statistics (strict zero-leakage guarantee).
  - Imputed/land fill pixels are normalized with the same training statistics,
    yielding deterministic finite values.
  - Standard deviation near zero is guarded against by defaulting std = 1.0.
  - Statistics are serializable to/from JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


def compute_variable_stats(
    data: np.ndarray,
    mask: np.ndarray,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes mean and std over valid observations (mask == 1) only.

    Args:
        data: Numpy array of observations.
        mask: Binary mask where 1.0 indicates observed valid ocean data.
        eps: Minimum threshold for standard deviation.

    Returns:
        dict with "mean" and "std" as floats.
    """
    valid_data = data[mask == 1.0]
    if len(valid_data) == 0:
        raise ValueError("No valid observations (mask == 1) found to compute statistics.")

    mean_val = float(np.mean(valid_data))
    std_val = float(np.std(valid_data))

    if std_val < eps or not np.isfinite(std_val):
        std_val = 1.0

    return {"mean": mean_val, "std": std_val}


PHYSICAL_KEYS = ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]


def compute_moments(
    data: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, Union[float, int]]:
    """
    Computes sum, sum of squares, and valid count over valid physical observations (mask == 1.0)
    for online / staged multi-year statistics accumulation.

    Guarantees:
      - Valid points must be finite and have mask == 1.0.
      - Uses float64 accumulator to prevent numerical overflow across multi-year data.
    """
    valid = (mask == 1.0) & np.isfinite(data)
    if not np.any(valid):
        return {"sum": 0.0, "sq_sum": 0.0, "count": 0}
    vals = data[valid].astype(np.float64)
    return {
        "sum": float(np.sum(vals)),
        "sq_sum": float(np.sum(vals ** 2)),
        "count": int(len(vals)),
    }


def combine_moments(
    moments_list: list[Dict[str, Dict[str, Union[float, int]]]],
) -> Dict[str, Dict[str, Union[float, int]]]:
    """
    Combines annual or daily moments across multiple batches/years into a single global moments dict.
    
    Guarantees:
      - Strictly associative and commutative (order-independent across years).
      - Combines moments ONLY for the 7 physical variables in PHYSICAL_KEYS.
      - Masks (channels 7..13) are NEVER processed for normalization statistics.
    """
    combined: Dict[str, Dict[str, Union[float, int]]] = {
        k: {"sum": 0.0, "sq_sum": 0.0, "count": 0}
        for k in PHYSICAL_KEYS
    }
    for m_dict in moments_list:
        for k in PHYSICAL_KEYS:
            if k in m_dict:
                combined[k]["sum"] += float(m_dict[k]["sum"])
                combined[k]["sq_sum"] += float(m_dict[k]["sq_sum"])
                combined[k]["count"] += int(m_dict[k]["count"])
    return combined


def compute_stats_from_moments(
    moments: Dict[str, Dict[str, Union[float, int]]],
    eps: float = 1e-6,
) -> Dict[str, Dict[str, float]]:
    """
    Computes exact global mean and std from combined moments across training years.
    
    Guarantees:
      - Computed ONLY for the 7 physical variables (PHYSICAL_KEYS).
      - Binary validity masks (channels 7..13) are NEVER included in stats.
    """
    stats = {}
    for k in PHYSICAL_KEYS:
        if k not in moments:
            raise KeyError(f"Variable '{k}' missing from moments dict!")
        cnt = int(moments[k]["count"])
        if cnt == 0:
            raise ValueError(f"No valid observations for variable '{k}' in training moments!")
        mu = float(moments[k]["sum"]) / cnt
        var = max(eps ** 2, (float(moments[k]["sq_sum"]) / cnt) - (mu ** 2))
        sigma = float(np.sqrt(var))
        if sigma < eps or not np.isfinite(sigma):
            sigma = 1.0
        stats[k] = {"mean": float(mu), "std": float(sigma)}
    return stats


def normalize_input_tensor(
    tensor: Union[np.ndarray, torch.Tensor],
    norm_stats: Dict[str, Dict[str, float]],
    var_keys: Optional[list[str]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalizes a 14-channel input tensor [14, H, W] using frozen normalization statistics.

    CRITICAL SCIENTIFIC CONTRACT:
      - Channels 0..6 (the 7 physical variables) are z-score normalized: (x - mu) / sigma.
      - Channels 7..13 (the 7 validity masks) are NEVER normalized and remain STRICTLY binary 0/1.
      - Output is guaranteed finite.
    """
    keys = var_keys or PHYSICAL_KEYS
    assert len(keys) == 7, f"Expected 7 physical variables, got {len(keys)}"

    is_torch = isinstance(tensor, torch.Tensor)
    if is_torch:
        arr = tensor.detach().cpu().numpy().copy()
    else:
        arr = tensor.copy()

    assert arr.shape[0] == 14, f"Expected 14-channel tensor [14, H, W], got {arr.shape}"

    # 1. Normalize physical channels 0..6
    for c, k in enumerate(keys):
        if k in norm_stats:
            mu = norm_stats[k]["mean"]
            sigma = norm_stats[k]["std"]
            if abs(sigma) < 1e-8:
                sigma = 1.0
            arr[c] = (arr[c] - mu) / sigma

    # 2. Channels 7..13 (masks) remain untouched — verify they are binary 0/1
    mask_channels = arr[7:14]
    unique_vals = np.unique(mask_channels)
    for val in unique_vals:
        assert val in (0.0, 1.0), f"Validity mask contains non-binary value: {val}"

    # Finiteness check
    assert np.all(np.isfinite(arr)), "Normalized array contains non-finite values!"

    if is_torch:
        return torch.from_numpy(arr).to(tensor.dtype).to(tensor.device)
    return arr


def denormalize_input_tensor(
    tensor: Union[np.ndarray, torch.Tensor],
    norm_stats: Dict[str, Dict[str, float]],
    var_keys: Optional[list[str]] = None,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Inverts z-score normalization on channels 0..6, preserving channels 7..13 as binary masks.
    """
    keys = var_keys or PHYSICAL_KEYS
    is_torch = isinstance(tensor, torch.Tensor)
    if is_torch:
        arr = tensor.detach().cpu().numpy().copy()
    else:
        arr = tensor.copy()

    for c, k in enumerate(keys):
        if k in norm_stats:
            mu = norm_stats[k]["mean"]
            sigma = norm_stats[k]["std"]
            arr[c] = arr[c] * sigma + mu

    if is_torch:
        return torch.from_numpy(arr).to(tensor.dtype).to(tensor.device)
    return arr


def normalize_array(
    data: Union[np.ndarray, torch.Tensor],
    mean: float,
    std: float,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Applies z-score normalization: (data - mean) / std.
    """
    if abs(std) < 1e-8:
        std = 1.0
    return (data - mean) / std


def denormalize_array(
    norm_data: Union[np.ndarray, torch.Tensor],
    mean: float,
    std: float,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Inverts z-score normalization: norm_data * std + mean.
    """
    return norm_data * std + mean


def save_norm_stats(stats: Dict[str, Dict[str, float]], save_path: Union[str, Path]) -> None:
    """
    Saves normalization statistics dictionary to a JSON file.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def load_norm_stats(stats_path: Union[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Loads normalization statistics dictionary from a JSON file.
    """
    path = Path(stats_path)
    if not path.exists():
        raise FileNotFoundError(f"Normalization statistics file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        stats = json.load(f)
    return stats
