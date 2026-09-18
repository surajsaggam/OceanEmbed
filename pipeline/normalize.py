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
