"""
models/losses.py
----------------
Loss functions for OceanEmbed subsurface reconstruction.

Includes:
  - MaskedMSELoss: Plain uniform masked MSE over valid ocean pixels.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Phase-1 Plain Uniform Masked MSE Loss.
    Penalizes squared error over valid ocean pixels only.

    Supports:
      - 2D mask [H, W]
      - 3D mask [D, H, W]
      - 4D mask [B, D, H, W]
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes uniform masked MSE.

        Args:
            pred: Predicted temperature [B, D, H, W].
            target: Target temperature [B, D, H, W].
            mask: Optional binary validity mask.

        Returns:
            Scalar loss tensor.
        """
        err_sq = (pred - target) ** 2

        if mask is not None:
            if mask.shape != err_sq.shape:
                if mask.ndim == 2:
                    # [H, W] -> [B, D, H, W]
                    mask = mask.unsqueeze(0).unsqueeze(0).expand_as(err_sq)
                elif mask.ndim == 3 and mask.shape[0] == pred.shape[1]:
                    # [D, H, W] -> [B, D, H, W]
                    mask = mask.unsqueeze(0).expand_as(err_sq)
                elif mask.ndim == 3 and mask.shape[0] == pred.shape[0]:
                    # [B, H, W] -> [B, 1, H, W] -> [B, D, H, W]
                    mask = mask.unsqueeze(1).expand_as(err_sq)
                else:
                    mask = mask.expand_as(err_sq)

            valid = (mask == 1.0) & torch.isfinite(target) & torch.isfinite(pred)
            if valid.sum() == 0:
                return err_sq.mean()
            return err_sq[valid].mean()
        else:
            valid = torch.isfinite(target) & torch.isfinite(pred)
            if valid.sum() == 0:
                return err_sq.mean()
            return err_sq[valid].mean()
