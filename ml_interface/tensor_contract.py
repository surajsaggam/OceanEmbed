"""Tensor dimension contracts and verification utilities for the ML teammate.

Validates that any PyTorch model or ONNX graph conforms to the
Phase-1 14-channel input and 15-channel depth output specifications.
"""

from typing import Tuple, List
from api.config import settings

# 14 Channels: 7 physical variables + 7 missing-data binary validity masks
EXPECTED_INPUT_CHANNELS = 14
EXPECTED_INPUT_VARIABLES = [
    "SST", "SSS", "SSH", "U_curr", "V_curr", "U_wind", "V_wind",
    "mask_SST", "mask_SSS", "mask_SSH", "mask_U_curr", "mask_V_curr", "mask_U_wind", "mask_V_wind"
]

# 15 Depth Channels
EXPECTED_OUTPUT_DEPTHS: List[int] = settings.STANDARD_DEPTHS
EXPECTED_OUTPUT_CHANNELS = len(EXPECTED_OUTPUT_DEPTHS)

# Latent Bottleneck Embedding Dimension
EXPECTED_EMBEDDING_DIM = 128


def validate_input_tensor_shape(shape: Tuple[int, ...]) -> None:
    """Validate that the input surface observation tensor matches the model contract.
    
    Accepts:
        - 4D Gridded Batch: [Batch, 14, Height, Width]
        - 2D Pointwise Column Batch: [Batch, 14]
    """
    if len(shape) == 4:
        if shape[1] != EXPECTED_INPUT_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_INPUT_CHANNELS} input channels at dim 1, got shape {shape}"
            )
    elif len(shape) == 2:
        if shape[1] != EXPECTED_INPUT_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_INPUT_CHANNELS} input features at dim 1, got shape {shape}"
            )
    else:
        raise ValueError(f"Input tensor must be 2D or 4D, received shape {shape}")


def validate_output_tensor_shape(shape: Tuple[int, ...]) -> None:
    """Validate that the reconstructed temperature tensor has 15 depth channels.
    
    Accepts:
        - 4D Gridded Batch: [Batch, 15, Height, Width]
        - 2D Pointwise Column Batch: [Batch, 15]
    """
    if len(shape) == 4:
        if shape[1] != EXPECTED_OUTPUT_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_OUTPUT_CHANNELS} output depth channels at dim 1, got shape {shape}"
            )
    elif len(shape) == 2:
        if shape[1] != EXPECTED_OUTPUT_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_OUTPUT_CHANNELS} output depth values at dim 1, got shape {shape}"
            )
    else:
        raise ValueError(f"Output tensor must be 2D or 4D, received shape {shape}")


def validate_embedding_tensor_shape(shape: Tuple[int, ...]) -> None:
    """Validate that the latent Ocean Embedding bottleneck has 128 dimensions."""
    if len(shape) == 4 and shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_DIM} latent embedding dimensions at dim 1, got shape {shape}"
        )
    elif len(shape) == 2 and shape[1] != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_DIM} latent embedding features at dim 1, got shape {shape}"
        )
