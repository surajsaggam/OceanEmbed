"""
models/losses.py
----------------
Loss functions for OceanEmbed subsurface reconstruction.

Includes:
  - MaskedMSELoss: Plain uniform masked MSE over valid ocean pixels (Phase-1 baseline).
  - MaskedVerticalGradientLoss: Vertical temperature gradient difference using nonuniform depths.
  - GradientAwareLoss: Combined loss L_total = L_MSE + lambda_grad * L_gradient (Phase-4A).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

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


class MaskedVerticalGradientLoss(nn.Module):
    """
    Masked Vertical Gradient Difference Loss.

    Computes:
      dT/dz = (T[k+1] - T[k]) / (z[k+1] - z[k])
    across adjacent depth levels accounting for nonuniform depth intervals.

    Supports L1 or MSE difference over valid depth intervals where both
    depth k and depth k+1 are valid ocean pixels (mask == 1.0).
    """

    def __init__(
        self,
        depths: Optional[List[int]] = None,
        loss_type: str = "l1",
    ) -> None:
        super().__init__()
        if depths is None:
            depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
        self.depths = list(depths)
        z = torch.tensor(self.depths, dtype=torch.float32)
        dz = z[1:] - z[:-1]
        if torch.any(dz <= 0):
            raise ValueError("Depths must be strictly increasing.")
        self.register_buffer("dz", dz.view(1, -1, 1, 1))  # [1, D-1, 1, 1]
        self.loss_type = loss_type.lower()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes vertical gradient difference loss.

        Args:
            pred: Predicted temperature [B, D, H, W].
            target: Target temperature [B, D, H, W].
            mask: Optional binary validity mask [B, D, H, W] or broadcastable.

        Returns:
            Scalar gradient loss tensor.
        """
        dT_pred = pred[:, 1:] - pred[:, :-1]      # [B, D-1, H, W]
        dT_targ = target[:, 1:] - target[:, :-1]  # [B, D-1, H, W]

        g_pred = dT_pred / self.dz                # [B, D-1, H, W] in degC / m
        g_targ = dT_targ / self.dz                # [B, D-1, H, W] in degC / m

        if self.loss_type == "mse":
            diff = (g_pred - g_targ) ** 2
        else:
            diff = torch.abs(g_pred - g_targ)

        if mask is not None:
            if mask.shape != pred.shape:
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0).expand_as(pred)
                elif mask.ndim == 3 and mask.shape[0] == pred.shape[1]:
                    mask = mask.unsqueeze(0).expand_as(pred)
                elif mask.ndim == 3 and mask.shape[0] == pred.shape[0]:
                    mask = mask.unsqueeze(1).expand_as(pred)
                else:
                    mask = mask.expand_as(pred)

            # Valid interval requires both bounding depths to be valid ocean
            interval_valid = (
                (mask[:, 1:] == 1.0)
                & (mask[:, :-1] == 1.0)
                & torch.isfinite(g_targ)
                & torch.isfinite(g_pred)
            )
            if interval_valid.sum() == 0:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            return diff[interval_valid].mean()
        else:
            valid = torch.isfinite(g_targ) & torch.isfinite(g_pred)
            if valid.sum() == 0:
                return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
            return diff[valid].mean()


class GradientAwareLoss(nn.Module):
    """
    Phase-4A Combined Objective:
      L_total = L_MSE + lambda_grad * L_gradient

    Guarantees:
      - When enabled=False or lambda_grad=0.0, reproduces baseline MaskedMSELoss.
      - Returns (loss_total, metrics_dict) for transparent monitoring.
    """

    def __init__(
        self,
        depths: Optional[List[int]] = None,
        lambda_grad: float = 2.0,
        gradient_loss_type: str = "l1",
        enabled: bool = True,
    ) -> None:
        super().__init__()
        self.mse_loss = MaskedMSELoss()
        self.lambda_grad = float(lambda_grad)
        self.enabled = bool(enabled)
        self.gradient_loss_type = gradient_loss_type
        self.grad_loss = MaskedVerticalGradientLoss(depths=depths, loss_type=gradient_loss_type)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes combined loss.

        Returns:
            Tuple of (loss_total: torch.Tensor, metrics: Dict[str, float])
        """
        loss_mse = self.mse_loss(pred, target, mask=mask)

        if not self.enabled or self.lambda_grad <= 0.0:
            return loss_mse, {
                "loss_total": loss_mse.item(),
                "loss_mse": loss_mse.item(),
                "loss_grad": 0.0,
                "grad_weighted": 0.0,
            }

        loss_g = self.grad_loss(pred, target, mask=mask)
        grad_weighted = self.lambda_grad * loss_g
        loss_total = loss_mse + grad_weighted

        metrics = {
            "loss_total": loss_total.item(),
            "loss_mse": loss_mse.item(),
            "loss_grad": loss_g.item(),
            "grad_weighted": grad_weighted.item(),
        }
        return loss_total, metrics
