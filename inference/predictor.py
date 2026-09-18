"""
inference/predictor.py
----------------------
Inference interface for OceanEmbed.

Provides a self-contained, clean prediction class for loading checkpoints
and generating 15-depth subsurface reconstructions and 128-D Ocean Embeddings.
Designed for seamless integration into downstream prototyping or deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from models.ocean_embed_net import OceanEmbedNet
from utils.config import load_config


class OceanEmbedPredictor:
    """
    High-level predictor for OceanEmbed model inference.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model configuration
        self.cfg = load_config("model" if config_path is None else config_path)
        self.model = OceanEmbedNet(self.cfg)

        if checkpoint_path is not None:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict(
        self,
        surface_tensor: Union[np.ndarray, torch.Tensor],
        return_embedding: bool = True,
        return_attention: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Runs inference on a single 14-channel surface tensor or batch.

        Args:
            surface_tensor: [14, H, W] or [B, 14, H, W] array/tensor.
            return_embedding: Whether to extract explicit 128-D Ocean Embedding Z.
            return_attention: Whether to extract attention gate map.

        Returns:
            Dict containing "temperature" [..., 15, H, W] and optionally "embedding", "attention".
        """
        if isinstance(surface_tensor, np.ndarray):
            x = torch.from_numpy(surface_tensor).to(torch.float32)
        else:
            x = surface_tensor.clone().to(torch.float32)

        added_batch_dim = False
        if x.ndim == 3:
            x = x.unsqueeze(0)  # [1, 14, H, W]
            added_batch_dim = True

        x = x.to(self.device)
        with torch.no_grad():
            out = self.model(x, return_attention=return_attention)
            temp = out["temperature"].cpu().numpy()
            z = out["embedding"].cpu().numpy()
            attn = out.get("attention")
            if attn is not None:
                attn = attn.cpu().numpy()

        if added_batch_dim:
            temp = temp[0]
            z = z[0]
            if attn is not None:
                attn = attn[0]

        result = {"temperature": temp}
        if return_embedding:
            result["embedding"] = z
        if return_attention and attn is not None:
            result["attention"] = attn

        return result
