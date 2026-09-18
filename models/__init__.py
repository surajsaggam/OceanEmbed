"""
models package for OceanEmbed.
"""

from models.attention_decoder import AttentionGate2D, AttentionGuidedDecoder
from models.embedding import OceanEmbeddingFusion
from models.losses import MaskedMSELoss
from models.multi_scale_cnn import MultiScaleSpatialCNN
from models.ocean_embed_net import OceanEmbedNet
from models.pointwise_mlp import PointwiseMLP

__all__ = [
    "MultiScaleSpatialCNN",
    "PointwiseMLP",
    "OceanEmbeddingFusion",
    "AttentionGate2D",
    "AttentionGuidedDecoder",
    "OceanEmbedNet",
    "MaskedMSELoss",
]
