"""
models/attention_decoder.py
---------------------------
Attention-Guided Decoder with Skip Connections for OceanEmbed.

Scientific and architectural role:
  Reconstructs the 15-depth temperature profile field from the explicit 128-D
  Ocean Embedding Z.
  To preserve fine-scale oceanographic gradients (such as coastal fronts and
  sharp thermal boundaries) without blurring, the decoder incorporates encoder
  skip features (F_spatial from Path A) modulated by an attention gate.

Single-resolution design:
  Because OceanEmbed operates at a constant 0.25° grid without spatial downsampling,
  the attention gate operates directly at native spatial resolution [H, W], computing
  either:
    - Spatial attention: alpha in [B, 1, H, W] via additive attention gating (default).
    - Channel attention: alpha in [B, C, 1, 1] via squeeze-and-excitation gating.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate2D(nn.Module):
    """
    Attention Gate for 2D single-resolution feature maps.

    Computes gating coefficients alpha to filter skip connections based on the
    decoder gating signal (e.g., Ocean Embedding Z).

    Args:
        gate_channels: Number of channels in gating signal g (default 128).
        skip_channels: Number of channels in skip feature x (default 128).
        inter_channels: Intermediate channel dimension for attention map (default 64).
        gate_type: "spatial" for [B, 1, H, W] spatial map, or "channel" for [B, C, 1, 1].
    """

    def __init__(
        self,
        gate_channels: int = 128,
        skip_channels: int = 128,
        inter_channels: int = 64,
        gate_type: str = "spatial",
    ) -> None:
        super().__init__()
        self.gate_type = gate_type.lower()
        self.skip_channels = skip_channels

        if self.gate_type == "channel":
            # Squeeze-and-excitation style channel attention
            self.fc1 = nn.Conv2d(gate_channels + skip_channels, inter_channels, kernel_size=1)
            self.fc2 = nn.Conv2d(inter_channels, skip_channels, kernel_size=1)
        else:
            # Additive spatial attention gate (Oktay et al., Attention U-Net)
            self.W_g = nn.Sequential(
                nn.Conv2d(gate_channels, inter_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, inter_channels),
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(skip_channels, inter_channels, kernel_size=1, bias=False),
                nn.GroupNorm(1, inter_channels),
            )
            self.psi = nn.Sequential(
                nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True),
                nn.Sigmoid(),
            )

    def forward(
        self, g: torch.Tensor, x_skip: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            g: Gating feature map [B, gate_channels, H, W].
            x_skip: Skip feature map [B, skip_channels, H, W].

        Returns:
            gated_x: Modulated skip feature [B, skip_channels, H, W].
            alpha: Attention weight map.
        """
        if self.gate_type == "channel":
            # Global pooling across spatial dimensions
            pool_g = F.adaptive_avg_pool2d(g, (1, 1))
            pool_x = F.adaptive_avg_pool2d(x_skip, (1, 1))
            pooled = torch.cat([pool_g, pool_x], dim=1)
            alpha = torch.sigmoid(self.fc2(F.gelu(self.fc1(pooled))))
            gated_x = alpha * x_skip
        else:
            # Additive spatial attention
            inter = F.gelu(self.W_g(g) + self.W_x(x_skip))
            alpha = self.psi(inter)  # [B, 1, H, W]
            gated_x = alpha * x_skip

        return gated_x, alpha


class AttentionGuidedDecoder(nn.Module):
    """
    Attention-guided decoder mapping Ocean Embedding Z to 15 temperature depths.

    Args:
        embedding_dim: Channels of embedding Z (default 128).
        skip_channels: Channels of encoder skip connection (default 128).
        dec_channels_1: Channels of first decoder conv stage (default 64).
        dec_channels_2: Channels of second decoder conv stage (default 32).
        out_depths: Number of target depth levels (default 15).
        attention_gate: "spatial" or "channel".
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        skip_channels: int = 128,
        dec_channels_1: int = 64,
        dec_channels_2: int = 32,
        out_depths: int = 15,
        attention_gate: str = "spatial",
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.out_depths = out_depths

        # Attention gate combining embedding Z and encoder skip features
        self.gate = AttentionGate2D(
            gate_channels=embedding_dim,
            skip_channels=skip_channels,
            inter_channels=dec_channels_1,
            gate_type=attention_gate,
        )

        # First stage combines Z and gated skip features
        merged_channels = embedding_dim + skip_channels
        self.stage1 = nn.Sequential(
            nn.Conv2d(
                merged_channels,
                dec_channels_1,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, dec_channels_1),
            nn.GELU(),
        )

        # Second stage refinement
        self.stage2 = nn.Sequential(
            nn.Conv2d(
                dec_channels_1,
                dec_channels_2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, dec_channels_2),
            nn.GELU(),
        )

        # Depth projection to 15 standard ocean depths
        self.out_proj = nn.Conv2d(
            dec_channels_2,
            out_depths,
            kernel_size=1,
            bias=True,
        )

    def forward(
        self,
        z: torch.Tensor,
        skip_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of decoder.

        Args:
            z: Ocean Embedding tensor [B, embedding_dim, H, W].
            skip_features: Encoder skip features [B, skip_channels, H, W].
            return_attention: If True, also returns attention gate weights.

        Returns:
            temp_out: Temperature reconstruction [B, out_depths, H, W].
            alpha: Attention map (if return_attention=True, else None).
        """
        gated_skip, alpha = self.gate(z, skip_features)
        merged = torch.cat([z, gated_skip], dim=1)

        h1 = self.stage1(merged)
        h2 = self.stage2(h1)
        temp_out = self.out_proj(h2)

        if return_attention:
            return temp_out, alpha
        return temp_out, None
