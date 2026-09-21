"""
models/cnn_baseline.py
-----------------------
CNN-Only Baseline Architecture for OceanEmbed (Path A without Path B).

Scientific role:
  Ablation and architectural baseline isolating the multi-scale spatial CNN (Path A).
  Unlike the full dual-path OceanEmbedNet, this model does NOT include:
    - Path B (Pointwise MLP / 1x1 local channel interactions)
    - Dual-path feature fusion
  Spatial features F_spatial in R^(B x 128 x H x W) are projected directly through
  a spatial decoder to predict the 15 output depths [B, 15, H, W].
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from models.multi_scale_cnn import MultiScaleSpatialCNN


class CNNOnlyBaseline(nn.Module):
    """
    CNN-only spatial baseline model.

    Args:
        in_channels: Number of input channels (14: 7 physical + 7 masks).
        out_depths: Number of target depths (15).
        spatial_channels: Multi-scale CNN output channels (default 128).
    """

    def __init__(
        self,
        in_channels: int = 14,
        out_depths: int = 15,
        spatial_channels: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_depths = out_depths
        self.spatial_channels = spatial_channels

        # Path A: Multi-Scale Spatial CNN
        self.path_a = MultiScaleSpatialCNN(
            in_channels=in_channels,
            local_channels=64,
            dilated_channels=64,
            dilation=2,
            out_channels=spatial_channels,
        )

        # CNN-only decoder head (reconstructs 15 depths directly from spatial features)
        self.decoder = nn.Sequential(
            nn.Conv2d(spatial_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            nn.Conv2d(32, out_depths, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Surface input tensor [B, 14, H, W].

        Returns:
            Dictionary with:
              - "temperature": [B, 15, H, W]
              - "embedding": [B, 128, H, W] (spatial features)
        """
        f_spatial = self.path_a(x)
        temp_pred = self.decoder(f_spatial)
        return {
            "temperature": temp_pred,
            "embedding": f_spatial,
        }
