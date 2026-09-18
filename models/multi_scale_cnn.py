"""
models/multi_scale_cnn.py
-------------------------
Path A: Multi-Scale Spatial CNN for OceanEmbed.

Scientific and architectural role:
  Captures spatial context across two receptive-field scales:
    1. Local branch: standard 3×3 convolution (padding=1) for fine-scale gradients.
    2. Dilated branch: 3×3 convolution with dilation=2 (padding=2) for broader context
       (~5×5 effective receptive field) without spatial downsampling.
  The two branches are concatenated and refined through a residual block to produce
  the spatial feature representation F_spatial ∈ R^(B × 128 × H × W).

Single-resolution design:
  Spatial dimensions (H, W) are strictly preserved across all operations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultiScaleSpatialCNN(nn.Module):
    """
    Multi-scale spatial CNN encoder branch (Path A).

    Args:
        in_channels: Number of input channels (default 14: 7 physical + 7 masks).
        local_channels: Number of output channels from local 3x3 branch (default 64).
        dilated_channels: Number of output channels from dilated 3x3 branch (default 64).
        dilation: Dilation rate for dilated branch (default 2, effective RF ~5x5).
        out_channels: Projected spatial output channels (default 128).
    """

    def __init__(
        self,
        in_channels: int = 14,
        local_channels: int = 64,
        dilated_channels: int = 64,
        dilation: int = 2,
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.local_channels = local_channels
        self.dilated_channels = dilated_channels
        self.dilation = dilation
        self.out_channels = out_channels

        # Local branch: 3x3 conv, stride 1, padding 1
        self.local_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                local_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, local_channels),
            nn.GELU(),
        )

        # Dilated branch: 3x3 conv, stride 1, dilation d, padding d
        self.dilated_branch = nn.Sequential(
            nn.Conv2d(
                in_channels,
                dilated_channels,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(1, dilated_channels),
            nn.GELU(),
        )

        concat_channels = local_channels + dilated_channels

        # Residual refinement block
        self.project = (
            nn.Conv2d(concat_channels, out_channels, kernel_size=1, bias=False)
            if concat_channels != out_channels
            else nn.Identity()
        )

        self.res_conv1 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res_norm1 = nn.GroupNorm(1, out_channels)
        self.act1 = nn.GELU()

        self.res_conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.res_norm2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Path A.

        Args:
            x: Input surface tensor of shape [B, in_channels, H, W].

        Returns:
            F_spatial: Spatial feature tensor of shape [B, out_channels, H, W].
        """
        f_local = self.local_branch(x)
        f_dilated = self.dilated_branch(x)

        # Concatenate multi-scale representations
        f_cat = torch.cat([f_local, f_dilated], dim=1)

        # Refine through residual block
        f_proj = self.project(f_cat)
        res = self.res_conv1(f_proj)
        res = self.res_norm1(res)
        res = self.act1(res)
        res = self.res_conv2(res)
        res = self.res_norm2(res)

        f_spatial = self.act2(f_proj + res)
        return f_spatial
