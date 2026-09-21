"""
baselines/climatology.py
------------------------
Climatological baseline for subsurface ocean temperature reconstruction.

Scientific role:
  Represents the seasonal mean state of the subsurface ocean.
  In oceanography, any deep learning model must demonstrably outperform climatology
  to prove that it captures synoptic and mesoscale dynamical variability rather than
  just the seasonal solar cycle.

Rule:
  Climatology statistics are computed EXCLUSIVELY from training period data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


class ClimatologyBaseline:
    """
    Monthly 3D Climatological Baseline.

    Stores mean temperature field T(month, depth, lat, lon) computed over
    the training years.
    """

    def __init__(self, n_depths: int = 15, grid_shape: Tuple[int, int] = (101, 241)) -> None:
        self.n_depths = n_depths
        self.H, self.W = grid_shape
        # Accumulators: month 1..12 -> [12, n_depths, H, W]
        self.month_sums = np.zeros((12, n_depths, self.H, self.W), dtype=np.float64)
        self.month_counts = np.zeros((12, n_depths, self.H, self.W), dtype=np.int64)
        self.climatology = np.zeros((12, n_depths, self.H, self.W), dtype=np.float32)
        self.is_fitted = False

    def update(self, month: int, target: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """
        Incrementally accumulates training profiles for a given month (1–12).

        Args:
            month: Month integer 1 to 12.
            target: Target temperature field [n_depths, H, W].
            mask: Optional boolean or float mask (1 = valid ocean).
        """
        idx = month - 1
        m = (mask == 1.0) if mask is not None else np.isfinite(target)
        valid_target = np.where(m, target, 0.0)

        self.month_sums[idx] += valid_target
        self.month_counts[idx] += m.astype(np.int64)

    def finalize(self) -> None:
        """
        Computes final monthly mean fields.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(self.month_counts > 0, self.month_sums / self.month_counts, 0.0)
        self.climatology = mean.astype(np.float32)
        self.is_fitted = True

    def predict(self, month: int) -> np.ndarray:
        """
        Predicts temperature field for a given month.

        Args:
            month: Month integer 1 to 12.

        Returns:
            pred: [n_depths, H, W] temperature field.
        """
        if not self.is_fitted:
            raise RuntimeError("Climatology baseline has not been finalized/fitted.")
        return self.climatology[month - 1].copy()

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves climatology arrays to .npz file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            climatology=self.climatology,
            month_counts=self.month_counts,
            n_depths=self.n_depths,
            grid_shape=(self.H, self.W),
        )

    def load(self, file_path: Union[str, Path]) -> None:
        """Loads climatology arrays from .npz file."""
        data = np.load(file_path)
        self.climatology = data["climatology"]
        self.month_counts = data["month_counts"]
        self.n_depths = int(data["n_depths"])
        self.H, self.W = tuple(data["grid_shape"])
        self.is_fitted = True
