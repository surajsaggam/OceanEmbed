"""
pipeline/mask.py
----------------
Mask management and land/validity mask processing for OceanEmbed.

Scientific rules:
  - Land pixels are identified via static land mask.
  - For land pixels:
      * validity_mask = 0.0
      * physical value is replaced with a finite fill value (never NaN)
  - For ocean pixels:
      * validity_mask = 1.0 if observed, 0.0 if imputed/cloud-masked
  - Missing-day exclusion:
      * Dates where invalid fraction across all ocean pixels exceeds threshold
        (default 0.90) are flagged for exclusion.
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch


def apply_land_mask(
    data: Union[np.ndarray, torch.Tensor],
    land_mask: Union[np.ndarray, torch.Tensor],
    fill: float,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Applies land mask to data, replacing land pixels with a deterministic finite fill value
    and marking validity mask as 0.0 for land.

    Args:
        data: Physical variable field [H, W] or [..., H, W].
        land_mask: Boolean mask where True indicates land.
        fill: Finite value to fill land pixels.

    Returns:
        values_filled: Data with land pixels set to fill.
        validity_mask: Binary float mask (1.0 = ocean/observed, 0.0 = land).
    """
    if not np.isfinite(fill):
        raise ValueError(f"fill value must be finite, got {fill}")

    if isinstance(data, torch.Tensor):
        values = data.clone().to(torch.float32)
        validity = torch.ones_like(values, dtype=torch.float32)
        land_bool = torch.as_tensor(land_mask, dtype=torch.bool, device=data.device)

        values[land_bool] = fill
        validity[land_bool] = 0.0
        return values, validity

    arr = np.asarray(data, dtype=np.float32).copy()
    validity = np.ones_like(arr, dtype=np.float32)
    land_bool = np.asarray(land_mask, dtype=bool)

    arr[land_bool] = fill
    validity[land_bool] = 0.0
    return arr, validity


def check_missing_day_fraction(
    validity_masks: np.ndarray,
    land_mask: np.ndarray,
    threshold: float = 0.90,
) -> Tuple[bool, float]:
    """
    Checks if a day should be excluded due to excessive missing data over ocean pixels.

    Args:
        validity_masks: [7, H, W] array of validity masks for the 7 variables.
        land_mask: [H, W] boolean mask (True = land).
        threshold: Fraction threshold (e.g. 0.90).

    Returns:
        should_exclude: True if invalid ocean fraction >= threshold.
        invalid_fraction: Fraction of invalid ocean pixels across variables.
    """
    ocean_bool = ~np.asarray(land_mask, dtype=bool)
    n_ocean = np.sum(ocean_bool)
    if n_ocean == 0:
        return True, 1.0

    # For each ocean pixel, check if it is invalid in all 7 channels
    ocean_validity = validity_masks[:, ocean_bool]  # [7, n_ocean]
    all_invalid = np.all(ocean_validity == 0.0, axis=0)
    invalid_fraction = float(np.sum(all_invalid) / n_ocean)

    should_exclude = invalid_fraction >= threshold
    return should_exclude, invalid_fraction
