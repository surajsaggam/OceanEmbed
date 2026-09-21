"""
pipeline/regrid.py
------------------
Grid generation and spatial interpolation module for OceanEmbed.

Standard target grid:
  - North Indian Ocean domain: 5°N–30°N, 45°E–105°E
  - Resolution: 0.25° × 0.25°
  - Expected spatial shape: H=101, W=241 with inclusive endpoints
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from utils.config import load_config


def compute_target_coords(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 1D target latitude and longitude coordinate arrays with inclusive endpoints.

    Args:
        lat_min: Minimum latitude in degrees North.
        lat_max: Maximum latitude in degrees North.
        lon_min: Minimum longitude in degrees East.
        lon_max: Maximum longitude in degrees East.
        resolution: Spatial resolution in degrees.

    Returns:
        lat: 1D array of latitude coordinates.
        lon: 1D array of longitude coordinates.
    """
    # Use + resolution/2 to ensure endpoint inclusion with floating point precision
    lat = np.arange(lat_min, lat_max + resolution / 2.0, resolution, dtype=np.float64)
    lon = np.arange(lon_min, lon_max + resolution / 2.0, resolution, dtype=np.float64)
    return lat, lon


def compute_grid_shape(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    resolution: float,
) -> Tuple[int, int]:
    """
    Returns (H, W) for the given domain bounds and resolution.
    """
    lat, lon = compute_target_coords(lat_min, lat_max, lon_min, lon_max, resolution)
    return len(lat), len(lon)


def get_default_target_coords() -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads target coordinates specified in configs/data.yaml.
    """
    cfg = load_config("data")
    return compute_target_coords(
        cfg.domain.lat_min,
        cfg.domain.lat_max,
        cfg.domain.lon_min,
        cfg.domain.lon_max,
        cfg.domain.resolution,
    )


def regrid_2d(
    data: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    dst_lat: np.ndarray,
    dst_lon: np.ndarray,
    method: str = "linear",
    fill_value: float = np.nan,
) -> np.ndarray:
    """
    Regrids a 2D regular grid [len(src_lat), len(src_lon)] to target coordinates [len(dst_lat), len(dst_lon)].

    Args:
        data: 2D array of source data.
        src_lat: 1D ascending array of source latitudes.
        src_lon: 1D ascending array of source longitudes.
        dst_lat: 1D ascending array of destination latitudes.
        dst_lon: 1D ascending array of destination longitudes.
        method: 'linear' or 'nearest'.
        fill_value: Value used for out-of-bounds queries.

    Returns:
        regridded: 2D array of shape [len(dst_lat), len(dst_lon)].
    """
    # Ensure src coords are ascending
    lat_flip = False
    lon_flip = False
    if src_lat[1] < src_lat[0]:
        src_lat = src_lat[::-1]
        data = data[::-1, :]
        lat_flip = True
    if src_lon[1] < src_lon[0]:
        src_lon = src_lon[::-1]
        data = data[:, ::-1]
        lon_flip = True

    interp = RegularGridInterpolator(
        (src_lat, src_lon),
        data,
        method=method,
        bounds_error=False,
        fill_value=fill_value,
    )

    dst_lat_grid, dst_lon_grid = np.meshgrid(dst_lat, dst_lon, indexing="ij")
    points = np.stack([dst_lat_grid.ravel(), dst_lon_grid.ravel()], axis=-1)
    regridded = interp(points).reshape(len(dst_lat), len(dst_lon))
    return regridded.astype(np.float32)
