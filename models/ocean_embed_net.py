"""
models/ocean_embed_net.py
-------------------------
OceanEmbedNet: Full Phase-1 Deep Learning Architecture.

Architecture:
  Input: 14 channels (7 physical surface variables + 7 validity masks) [B, 14, H, W]
  Path A: Multi-Scale Spatial CNN (local 3x3 + dilated 3x3 -> 128 channels)
  Path B: Pointwise MLP / 1x1 Convolution (64 channels)
  Feature Fusion: Concatenation -> 1x1 projection -> GroupNorm(1, 128) -> GELU
  Explicit Ocean Embedding: Z in R^(B x 128 x H x W)
  Attention-Guided Decoder: Skip connection from Path A modulated by attention gate -> 15 depths
  Output: Temperature at 15 depths [B, 15, H, W]

All spatial dimensions H, W are strictly preserved (single-resolution design).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn

from models.attention_decoder import AttentionGuidedDecoder
from models.embedding import OceanEmbeddingFusion
from models.multi_scale_cnn import MultiScaleSpatialCNN
from models.pointwise_mlp import PointwiseMLP


class OceanEmbedNet(nn.Module):
    """
    Phase-1 OceanEmbed Neural Network.

    Args:
        cfg: Configuration object (ConfigNode, dict, or Any with attribute access)
             containing model hyperparameters.
    """

    def __init__(self, cfg: Any = None, **kwargs: Any) -> None:
        super().__init__()

        # Helper to extract configuration value with fallback
        def _get_val(key: str, default: Any) -> Any:
            if key in kwargs:
                return kwargs[key]
            if cfg is not None:
                if isinstance(cfg, dict):
                    return cfg.get(key, default)
                if hasattr(cfg, key):
                    return getattr(cfg, key)
            return default

        self.in_channels: int = int(_get_val("in_channels", 14))
        self.embedding_dim: int = int(_get_val("embedding_dim", 128))
        self.out_depths: int = int(_get_val("out_depths", 15))

        # Path A configs
        self.cnn_local_channels: int = int(_get_val("cnn_local_channels", 64))
        self.cnn_dilated_channels: int = int(_get_val("cnn_dilated_channels", 64))
        self.cnn_dilation: int = int(_get_val("cnn_dilation", 2))
        self.cnn_spatial_out_channels: int = int(_get_val("cnn_spatial_out_channels", 128))

        # Path B configs
        self.mlp_hidden_channels: int = int(_get_val("mlp_hidden_channels", 64))
        self.mlp_out_channels: int = int(_get_val("mlp_out_channels", 64))
        self.mlp_norm: str = str(_get_val("mlp_norm", "none"))

        # Decoder configs
        self.dec_channels_1: int = int(_get_val("dec_channels_1", 64))
        self.dec_channels_2: int = int(_get_val("dec_channels_2", 32))
        self.attention_gate: str = str(_get_val("attention_gate", "spatial"))

        # Path A: Multi-Scale Spatial CNN
        self.path_a = MultiScaleSpatialCNN(
            in_channels=self.in_channels,
            local_channels=self.cnn_local_channels,
            dilated_channels=self.cnn_dilated_channels,
            dilation=self.cnn_dilation,
            out_channels=self.cnn_spatial_out_channels,
        )

        # Path B: Pointwise MLP
        self.path_b = PointwiseMLP(
            in_channels=self.in_channels,
            hidden_channels=self.mlp_hidden_channels,
            out_channels=self.mlp_out_channels,
            norm=self.mlp_norm,
        )

        # Feature Fusion -> Explicit Ocean Embedding Z
        self.fusion = OceanEmbeddingFusion(
            spatial_channels=self.cnn_spatial_out_channels,
            local_channels=self.mlp_out_channels,
            embedding_dim=self.embedding_dim,
        )

        # Attention-Guided Decoder
        self.decoder = AttentionGuidedDecoder(
            embedding_dim=self.embedding_dim,
            skip_channels=self.cnn_spatial_out_channels,
            dec_channels_1=self.dec_channels_1,
            dec_channels_2=self.dec_channels_2,
            out_depths=self.out_depths,
            attention_gate=self.attention_gate,
        )

    def encode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode input into spatial features, local features, and Ocean Embedding Z.

        Args:
            x: Surface input tensor [B, in_channels, H, W].

        Returns:
            f_spatial: Multi-scale spatial features [B, cnn_spatial_out_channels, H, W].
            f_local: Pointwise local features [B, mlp_out_channels, H, W].
            z: Explicit Ocean Embedding [B, embedding_dim, H, W].
        """
        f_spatial = self.path_a(x)
        f_local = self.path_b(x)
        z = self.fusion(f_spatial, f_local)
        return f_spatial, f_local, z

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract explicit Ocean Embedding Z for downstream analysis (PCA, UMAP, etc.).

        Args:
            x: Surface input tensor [B, in_channels, H, W].

        Returns:
            Z: Ocean Embedding tensor [B, embedding_dim, H, W].
        """
        _, _, z = self.encode(x)
        return z

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through OceanEmbedNet.

        Args:
            x: Input surface tensor of shape [B, 14, H, W].
            return_attention: If True, includes attention gate map in output dict.

        Returns:
            Dictionary containing:
                "temperature": Reconstructed temperature [B, 15, H, W]
                "embedding": Explicit Ocean Embedding Z [B, 128, H, W]
                "attention" (optional): Attention gate weights [B, 1, H, W]
        """
        f_spatial, _, z = self.encode(x)
        temp_pred, alpha = self.decoder(z, f_spatial, return_attention=return_attention)

        out = {
            "temperature": temp_pred,
            "embedding": z,
        }
        if return_attention and alpha is not None:
            out["attention"] = alpha

        return out
