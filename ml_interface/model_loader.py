"""Model loading utility for PyTorch checkpoints and ONNX graphs.

Designed to be invoked by RealOceanEmbedProvider when the ML teammate
provides the trained model weights.
"""

import os
import hashlib
from typing import Optional, Tuple, Any


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA256 checksum of a model file for provenance verification."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()


def detect_optimal_device() -> str:
    """Detect CUDA GPU or CPU availability."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_pytorch_checkpoint(checkpoint_path: str) -> Tuple[Optional[Any], Optional[str]]:
    """Safely load a PyTorch model checkpoint from disk.
    
    Returns:
        (model, checkpoint_hash) or (None, None) if file does not exist.
    """
    if not os.path.exists(checkpoint_path):
        return None, None

    try:
        import torch
        device = detect_optimal_device()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        file_hash = compute_file_sha256(checkpoint_path)
        return checkpoint, file_hash
    except Exception as e:
        print(f"Error loading PyTorch checkpoint from {checkpoint_path}: {e}")
        return None, None
