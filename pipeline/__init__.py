"""
pipeline package for OceanEmbed.
"""

from pipeline.align import generate_date_range, get_split_date_ranges, get_split_for_date
from pipeline.datasets import OceanEmbedDataset, SyntheticOceanDataset, create_dataloader
from pipeline.mask import apply_land_mask, check_missing_day_fraction
from pipeline.normalize import (
    compute_variable_stats,
    denormalize_array,
    load_norm_stats,
    normalize_array,
    save_norm_stats,
)
from pipeline.qc import apply_qc, apply_variable_qc
from pipeline.regrid import compute_grid_shape, compute_target_coords, regrid_2d

__all__ = [
    "apply_qc",
    "apply_variable_qc",
    "compute_target_coords",
    "compute_grid_shape",
    "regrid_2d",
    "apply_land_mask",
    "check_missing_day_fraction",
    "compute_variable_stats",
    "normalize_array",
    "denormalize_array",
    "save_norm_stats",
    "load_norm_stats",
    "get_split_date_ranges",
    "get_split_for_date",
    "generate_date_range",
    "OceanEmbedDataset",
    "SyntheticOceanDataset",
    "create_dataloader",
]
