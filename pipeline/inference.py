"""
pipeline/inference.py
---------------------
Re-exports the production OceanEmbed inference predictor and extraction utilities.
"""

from inference.predictor import (
    CLIMATOLOGICAL_FILL_VALUES,
    DEFAULT_QC_LIMITS,
    FROZEN_PHASE1_SHA256,
    PHYSICAL_KEYS,
    STANDARD_DEPTHS,
    OceanEmbedPredictor,
    compute_file_sha256,
    extract_profile,
    lookup_argo_profile,
)

__all__ = [
    "OceanEmbedPredictor",
    "STANDARD_DEPTHS",
    "PHYSICAL_KEYS",
    "DEFAULT_QC_LIMITS",
    "CLIMATOLOGICAL_FILL_VALUES",
    "FROZEN_PHASE1_SHA256",
    "compute_file_sha256",
    "extract_profile",
    "lookup_argo_profile",
]
