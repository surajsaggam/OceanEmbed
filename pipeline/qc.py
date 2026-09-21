"""
pipeline/qc.py
--------------
Quality Control (QC) module for OceanEmbed surface observation variables.

Scientific rules:
  - Each physical variable is checked against physical plausible ranges.
  - Invalid values (out of range, NaN, +/-inf) are flagged:
      * validity_mask = 0.0 (imputed/invalid)
      * data is replaced with a finite fill_value (e.g. regional climatology)
  - Valid values pass through unchanged:
      * validity_mask = 1.0 (observed)
  - Zero is NEVER silently used as a replacement unless scientifically appropriate.
  - All outputs must be strictly finite (no residual NaNs or Infs).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import numpy as np
import torch


def apply_qc(
    values: Union[np.ndarray, torch.Tensor],
    min_val: float,
    max_val: float,
    fill_value: float,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Applies physical range QC to an array or tensor.

    Args:
        values: Numerical array or tensor of observations.
        min_val: Minimum physically plausible value.
        max_val: Maximum physically plausible value.
        fill_value: Finite replacement value for invalid entries.

    Returns:
        values_out: QC'd values with invalid entries replaced by fill_value.
        validity_mask: Binary mask (1.0 = valid/observed, 0.0 = invalid/imputed).
    """
    if not np.isfinite(fill_value):
        raise ValueError(f"fill_value must be a finite number, got {fill_value}")

    if isinstance(values, torch.Tensor):
        mask = torch.ones_like(values, dtype=torch.float32)
        values_out = values.clone().to(torch.float32)

        invalid = (values < min_val) | (values > max_val) | (~torch.isfinite(values))
        values_out[invalid] = fill_value
        mask[invalid] = 0.0
        return values_out, mask

    # NumPy array pathway
    arr = np.asarray(values, dtype=np.float32)
    mask = np.ones_like(arr, dtype=np.float32)
    values_out = arr.copy()

    invalid = (arr < min_val) | (arr > max_val) | (~np.isfinite(arr))
    values_out[invalid] = fill_value
    mask[invalid] = 0.0

    return values_out, mask


def apply_variable_qc(
    var_name: str,
    values: Union[np.ndarray, torch.Tensor],
    qc_limits: Dict[str, Dict[str, float]],
    fill_value: float,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Applies QC using limits looked up from the qc_limits configuration.

    Args:
        var_name: Variable name (e.g. 'SST', 'SSS', 'SSH', 'U_curr', 'V_curr', 'WindU', 'WindV').
        values: Raw observation data.
        qc_limits: Dictionary mapping var_name to {'min': float, 'max': float}.
        fill_value: Finite fill value.

    Returns:
        values_out, validity_mask
    """
    if var_name not in qc_limits:
        raise KeyError(f"Variable '{var_name}' not found in qc_limits dictionary: {list(qc_limits.keys())}")

    lims = qc_limits[var_name]
    min_val = float(lims["min"])
    max_val = float(lims["max"])
    return apply_qc(values, min_val=min_val, max_val=max_val, fill_value=fill_value)
