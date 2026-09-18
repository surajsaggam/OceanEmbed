"""
baselines/ridge.py
------------------
Memory-efficient Linear / Ridge Regression baseline for OceanEmbed.

Scientific and architectural role:
  Represents the linear baseline mapping surface observations to subsurface temperature.
  Demonstrates how much of the subsurface thermal structure can be explained by linear
  multivariate relationships alone.

Memory-conscious design:
  CRITICAL RULE: Does NOT materialize the complete [N_days x H x W, 14] matrix in RAM.
  Instead, accumulates sufficient statistics (X^T X in R^(15 x 15) and X^T Y in R^(15 x 15))
  incrementally chunk-by-chunk. Memory footprint is strictly O(1) regardless of dataset size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np


class ChunkedRidgeBaseline:
    """
    Ridge regression baseline using chunked sufficient-statistic accumulation.

    Maps 14 surface channels (7 physical + 7 masks) to 15 temperature depths.
    Uses an affine formulation: y = X * W + b.
    """

    def __init__(self, in_features: int = 14, out_features: int = 15, alpha: float = 1.0) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = float(alpha)

        # Dimension with bias appended
        self.d = in_features + 1

        # Sufficient statistics accumulators
        # A = sum(x_aug^T * x_aug), shape [d, d]
        self.XtX = np.zeros((self.d, self.d), dtype=np.float64)
        # B = sum(x_aug^T * y), shape [d, out_features]
        self.XtY = np.zeros((self.d, self.out_features), dtype=np.float64)
        self.n_samples: int = 0

        # Learned parameters
        self.weights = np.zeros((in_features, out_features), dtype=np.float32)
        self.bias = np.zeros((out_features,), dtype=np.float32)
        self.is_fitted = False

    def update_chunk(
        self,
        surface_features: np.ndarray,
        target_temps: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        Incrementally accumulates statistics from a single batch or single day.

        Args:
            surface_features: [in_features, H, W] or [N, in_features] array.
            target_temps: [out_features, H, W] or [N, out_features] array.
            valid_mask: Optional boolean mask selecting valid pixels [H, W] or [N].
        """
        if surface_features.ndim == 3:
            # Flatten spatial dims to pixels [N, in_features]
            C, H, W = surface_features.shape
            X_flat = surface_features.reshape(C, -1).T  # [N, 14]
            Y_flat = target_temps.reshape(self.out_features, -1).T  # [N, 15]

            if valid_mask is not None:
                mask_flat = valid_mask.ravel().astype(bool)
                X_flat = X_flat[mask_flat]
                Y_flat = Y_flat[mask_flat]
        else:
            X_flat = surface_features
            Y_flat = target_temps
            if valid_mask is not None:
                X_flat = X_flat[valid_mask]
                Y_flat = Y_flat[valid_mask]

        if len(X_flat) == 0:
            return

        # Append column of ones for bias term: [N, d]
        ones = np.ones((X_flat.shape[0], 1), dtype=np.float64)
        X_aug = np.hstack([X_flat.astype(np.float64), ones])  # [N, 15]
        Y_dbl = Y_flat.astype(np.float64)  # [N, 15]

        # Accumulate outer products
        self.XtX += X_aug.T @ X_aug  # [15, 15]
        self.XtY += X_aug.T @ Y_dbl  # [15, 15]
        self.n_samples += len(X_flat)

    def finalize(self) -> None:
        """
        Solves regularized normal equations: (X^T X + alpha * I) W = X^T Y.
        Bias parameter is not penalized.
        """
        if self.n_samples == 0:
            raise RuntimeError("Cannot finalize Ridge baseline: no samples accumulated.")

        # Regularization matrix: penalize weights, but do not regularize bias
        reg = self.alpha * np.eye(self.d, dtype=np.float64)
        reg[-1, -1] = 0.0  # Zero regularization on bias

        A = self.XtX + reg
        # Solve A * W_aug = XtY
        W_aug = np.linalg.solve(A, self.XtY)  # [d, out_features]

        self.weights = W_aug[:-1, :].astype(np.float32)  # [in_features, out_features]
        self.bias = W_aug[-1, :].astype(np.float32)  # [out_features]
        self.is_fitted = True

    def predict(self, surface_features: np.ndarray) -> np.ndarray:
        """
        Predicts depth-wise temperature field.

        Args:
            surface_features: [in_features, H, W] array.

        Returns:
            pred: [out_features, H, W] predicted temperature.
        """
        if not self.is_fitted:
            raise RuntimeError("Ridge baseline must be fitted before predict.")

        C, H, W = surface_features.shape
        X_flat = surface_features.reshape(C, -1).T  # [N, 14]
        Y_pred = X_flat @ self.weights + self.bias  # [N, 15]
        return Y_pred.T.reshape(self.out_features, H, W).astype(np.float32)

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves weights, bias, and config to .npz."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            weights=self.weights,
            bias=self.bias,
            alpha=self.alpha,
            in_features=self.in_features,
            out_features=self.out_features,
            n_samples=self.n_samples,
        )

    def load(self, file_path: Union[str, Path]) -> None:
        """Loads weights and bias from .npz."""
        data = np.load(file_path)
        self.weights = data["weights"]
        self.bias = data["bias"]
        self.alpha = float(data["alpha"])
        self.in_features = int(data["in_features"])
        self.out_features = int(data["out_features"])
        self.n_samples = int(data["n_samples"])
        self.is_fitted = True
