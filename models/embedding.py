"""
models/embedding.py
-------------------
Feature Fusion and Explicit Ocean Embedding Module for OceanEmbed.

Scientific and architectural role:
  Combines the multi-scale spatial features F_spatial (Path A) and local column
  features F_local (Path B) into a unified, explicit 128-dimensional Ocean Embedding
  Z ∈ R^(B × 128 × H × W).

  Normalization:
    Uses GroupNorm(1, embedding_dim), which normalises over the channel dimension C
    for NCHW tensors per sample, avoiding artificial spatial coupling across geographic
    coordinates while guaranteeing finite, zero-mean, unit-variance embedding activations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OceanEmbeddingFusion(nn.Module):
    """
    Fuses spatial and pointwise feature maps into an explicit Ocean Embedding.

    Args:
        spatial_channels: Input channels from Path A (default 128).
        local_channels: Input channels from Path B (default 64).
        embedding_dim: Dimension of explicit Ocean Embedding Z (default 128).
    """

    def __init__(
        self,
        spatial_channels: int = 128,
        local_channels: int = 64,
        embedding_dim: int = 128,
    ) -> None:
        super().__init__()
        self.spatial_channels = spatial_channels
        self.local_channels = local_channels
        self.embedding_dim = embedding_dim

        in_fused = spatial_channels + local_channels

        # 1x1 projection into embedding space
        self.proj = nn.Conv2d(in_fused, embedding_dim, kernel_size=1, bias=False)
        # Channel-wise normalization over C for NCHW feature maps
        self.norm = nn.GroupNorm(num_groups=1, num_channels=embedding_dim)
        self.act = nn.GELU()

    def forward(
        self, f_spatial: torch.Tensor, f_local: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features into explicit Ocean Embedding Z.

        Args:
            f_spatial: Feature tensor from Path A [B, spatial_channels, H, W].
            f_local: Feature tensor from Path B [B, local_channels, H, W].

        Returns:
            Z: Explicit Ocean Embedding tensor [B, embedding_dim, H, W].
        """
        fused = torch.cat([f_spatial, f_local], dim=1)
        z = self.proj(fused)
        z = self.norm(z)
        z = self.act(z)
        return z
