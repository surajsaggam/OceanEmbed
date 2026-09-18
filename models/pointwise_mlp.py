"""
models/pointwise_mlp.py
-----------------------
Path B: Pointwise MLP / 1x1 Convolution for OceanEmbed.

Scientific and architectural role:
  Encodes the 14-channel surface observation column at each geographic pixel
  independently without mixing spatial neighbors. This preserves local 1D column
  relationships (e.g., surface density and wind stress at that exact point) that
  spatial filtering might otherwise blur.

Single-resolution design:
  Spatial dimensions (H, W) are strictly preserved across all operations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PointwiseMLP(nn.Module):
    """
    Pointwise MLP encoder branch (Path B).

    Args:
        in_channels: Number of input channels (default 14).
        hidden_channels: Hidden dimension of pointwise MLP (default 64).
        out_channels: Output feature dimension (default 64).
        norm: Normalization type ("none", "layernorm", "batchnorm"). Default "none".
    """

    def __init__(
        self,
        in_channels: int = 14,
        hidden_channels: int = 64,
        out_channels: int = 64,
        norm: str = "none",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.norm_type = norm.lower()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=True)
        self.act1 = nn.GELU()

        if self.norm_type == "layernorm":
            self.norm1 = nn.GroupNorm(1, hidden_channels)
            self.norm2 = nn.GroupNorm(1, out_channels)
        elif self.norm_type == "batchnorm":
            self.norm1 = nn.BatchNorm2d(hidden_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=True)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Path B.

        Args:
            x: Input surface tensor of shape [B, in_channels, H, W].

        Returns:
            F_local: Local column feature tensor of shape [B, out_channels, H, W].
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act1(h)

        h = self.conv2(h)
        h = self.norm2(h)
        f_local = self.act2(h)
        return f_local
