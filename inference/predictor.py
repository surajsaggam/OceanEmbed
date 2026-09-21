"""
inference/predictor.py
----------------------
Production ML Inference Module for OceanEmbed Phase-1 Primary Model.

Scientific & Engineering Contracts:
  - Frozen Model: OceanEmbed Phase-1 (525,040 parameters)
  - Checkpoint: checkpoints/phase1/best.pt
  - Expected SHA256: f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b
  - Input: 14 channels (7 physical surface variables + 7 binary validity masks)
  - Grid: North Indian Ocean 0.25° grid (H=101, W=241, lat 5–30°N, lon 45–105°E)
  - Output:
      * 3D Temperature reconstruction: [15, 101, 241] in °C
      * Explicit 128-D Ocean Embedding: [128, 101, 241]
      * Spatial Attention Gate Map: [1, 101, 241]
      * Observation Validity Masks: [7, 101, 241]
  - Preprocessing Semantics:
      * Invalid observations replaced with regional climatological fill values
      * Validity mask set to 0.0 for invalid/imputed/land pixels, 1.0 for valid observations
      * Physical channels z-score normalized with frozen 2015–2017 training statistics
      * Validity masks (channels 7–13) remain strictly binary {0.0, 1.0}
  - Profile Extraction:
      * Standard 15 depths: [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] m
  - Invariance:
      * Pure evaluation mode: model.eval(), torch.no_grad(), requires_grad=False
      * Zero optimizer or training state loaded
      * Independent Argo validation strictly decoupled
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from models.ocean_embed_net import OceanEmbedNet
from pipeline.normalize import load_norm_stats
from pipeline.regrid import compute_target_coords
from utils.config import load_config


# ── Constants & Scientific Contracts ──────────────────────────────────────────

FROZEN_PHASE1_SHA256 = (
    "f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b"
)

STANDARD_DEPTHS: List[int] = [
    0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
]

PHYSICAL_KEYS: List[str] = [
    "SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"
]

VARIABLE_ALIASES: Dict[str, List[str]] = {
    "SST": ["sst", "sst_c", "analysed_sst", "temperature_surface"],
    "SSS": ["sss", "sos", "salinity_surface"],
    "SSH": ["ssh", "sla", "sea_surface_height", "sea_level_anomaly"],
    "U_curr": ["u_curr", "ucurr", "u_current", "u"],
    "V_curr": ["v_curr", "vcurr", "v_current", "v"],
    "WindU": ["windu", "u_wind", "wind_u", "uwnd"],
    "WindV": ["windv", "v_wind", "wind_v", "vwnd"],
}

DEFAULT_QC_LIMITS: Dict[str, Dict[str, float]] = {
    "SST": {"min": -2.0, "max": 36.0},
    "SSS": {"min": 0.0, "max": 45.0},
    "SSH": {"min": -2.0, "max": 2.0},
    "U_curr": {"min": -3.0, "max": 3.0},
    "V_curr": {"min": -3.0, "max": 3.0},
    "WindU": {"min": -30.0, "max": 30.0},
    "WindV": {"min": -30.0, "max": 30.0},
}

# Regional Climatological Fill Values used during Phase-1 training
# (INVALID OBSERVATION -> regional climatology fill + validity mask = 0)
CLIMATOLOGICAL_FILL_VALUES: Dict[str, float] = {
    "SST": 28.0,   # Regional mean surface temperature (°C)
    "SSS": 34.0,   # Regional mean surface salinity (PSU)
    "SSH": 0.0,    # Mean sea level anomaly (m)
    "U_curr": 0.0, # Neutral surface zonal current (m/s)
    "V_curr": 0.0, # Neutral surface meridional current (m/s)
    "WindU": 0.0,  # Neutral zonal wind (m/s)
    "WindV": 0.0,  # Neutral meridional wind (m/s)
}


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes hexadecimal SHA256 checksum for a given file."""
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {p}")
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class OceanEmbedPredictor:
    """
    High-level, production-ready inference interface for the frozen Phase-1 OceanEmbed model.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: Optional[Union[str, Path]] = None,
        norm_stats_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        verify_sha256: bool = True,
    ) -> None:
        """
        Initializes the predictor with frozen weights and statistics.

        Args:
            checkpoint_path: Path to Phase-1 best.pt checkpoint. Defaults to 'checkpoints/phase1/best.pt'.
            config_path: Path to model configuration YAML. Defaults to 'configs/model.yaml'.
            norm_stats_path: Path to frozen 2015–2017 normalization stats JSON.
            device: 'cuda', 'cpu', or None (auto-detect).
            verify_sha256: If True, validates checkpoint SHA256 against frozen Phase-1 hash.
        """
        self.project_root = Path(__file__).resolve().parent.parent

        # 1. Resolve Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 2. Resolve Checkpoint & SHA256 Verification
        if checkpoint_path is None:
            self.checkpoint_path = self.project_root / "checkpoints" / "phase1" / "best.pt"
        else:
            self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.checkpoint_path}")

        self.checkpoint_sha256 = compute_file_sha256(self.checkpoint_path)
        if verify_sha256:
            if self.checkpoint_sha256.lower() != FROZEN_PHASE1_SHA256.lower():
                raise ValueError(
                    f"Checkpoint SHA256 mismatch!\n"
                    f"  Expected: {FROZEN_PHASE1_SHA256}\n"
                    f"  Found:    {self.checkpoint_sha256}\n"
                    f"Aborting inference to prevent unverified model execution."
                )

        # 3. Load Model Configuration
        if config_path is None:
            self.cfg_model = load_config("model")
        else:
            self.cfg_model = load_config(config_path)

        # 4. Initialize Network
        self.model = OceanEmbedNet(self.cfg_model)

        # 5. Load State Dict (strictly weights only, no optimizer state)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict)

        # Freeze all parameters and set to eval mode
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

        # 6. Load Normalization Statistics
        if norm_stats_path is None:
            self.norm_stats_path = self.project_root / "data" / "norm_stats" / "train_stats.json"
        else:
            self.norm_stats_path = Path(norm_stats_path)

        if not self.norm_stats_path.exists():
            raise FileNotFoundError(f"Normalization statistics file not found at {self.norm_stats_path}")

        self.norm_stats = load_norm_stats(self.norm_stats_path)

        # 7. Grid Specifications (North Indian Ocean Domain)
        self.lat_min = 5.0
        self.lat_max = 30.0
        self.lon_min = 45.0
        self.lon_max = 105.0
        self.resolution = 0.25
        self.lat_coords, self.lon_coords = compute_target_coords(
            self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.resolution
        )
        self.H = len(self.lat_coords)  # 101
        self.W = len(self.lon_coords)  # 241
        self.standard_depths = STANDARD_DEPTHS
        self.num_depths = len(self.standard_depths)

    def prepare_input_tensor(
        self,
        surface_observations: Union[Dict[str, np.ndarray], np.ndarray, torch.Tensor],
        masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Prepares a validated 14-channel model input tensor following exact training semantics.

        Args:
            surface_observations:
                Either a dictionary containing the 7 surface variables:
                  'SST', 'SSS', 'SSH', 'U_curr', 'V_curr', 'WindU', 'WindV'
                  (with shape [101, 241] for each variable),
                OR a pre-formed [14, 101, 241] or [B, 14, 101, 241] array/tensor.
            masks:
                Optional dictionary of binary masks {var_name: mask_array}.
                If not provided for dictionary inputs, invalid/NaN/Inf values are automatically
                assigned mask=0.0 and imputed with regional climatological values.

        Returns:
            input_tensor: [1, 14, 101, 241] or [B, 14, 101, 241] FloatTensor.
            validity_masks: [7, 101, 241] or [B, 7, 101, 241] numpy array of masks.
        """
        # Case A: Input is already a 14-channel array or tensor
        if isinstance(surface_observations, (np.ndarray, torch.Tensor)):
            arr = (
                surface_observations.detach().cpu().numpy()
                if isinstance(surface_observations, torch.Tensor)
                else surface_observations.copy()
            )
            added_batch_dim = False
            if arr.ndim == 3:
                if arr.shape[0] != 14:
                    raise ValueError(
                        f"Expected 3D input of shape (14, H, W), got {arr.shape}"
                    )
                arr = arr[np.newaxis, ...]  # [1, 14, H, W]
                added_batch_dim = True
            elif arr.ndim == 4:
                if arr.shape[1] != 14:
                    raise ValueError(
                        f"Expected 4D batched input of shape (B, 14, H, W), got {arr.shape}"
                    )
            else:
                raise ValueError(
                    f"Unsupported array dimension {arr.ndim}. Expected 3D [14, H, W] or 4D [B, 14, H, W]."
                )

            if not np.all(np.isfinite(arr)):
                raise ValueError("Input array contains non-finite values (NaN or Inf).")

            # Validate that mask channels (7..13) are binary {0.0, 1.0}
            mask_channels = arr[:, 7:14, :, :]
            unique_vals = np.unique(mask_channels)
            for v in unique_vals:
                if v not in (0.0, 1.0):
                    raise ValueError(f"Validity mask channels contain non-binary value: {v}")

            masks_out = mask_channels[0] if added_batch_dim else mask_channels
            t_tensor = torch.from_numpy(arr).to(torch.float32)
            return t_tensor, masks_out

        # Case B: Input is a dictionary of 7 physical variables
        if not isinstance(surface_observations, dict):
            raise TypeError(
                f"surface_observations must be a dict or a 14-channel array/tensor, got {type(surface_observations)}"
            )

        norm_phys_channels = []
        mask_channels = []
        first_shape = None

        for key in PHYSICAL_KEYS:
            # Match key directly or via supported aliases
            val_arr = None
            if key in surface_observations:
                val_arr = surface_observations[key]
            else:
                for alias in VARIABLE_ALIASES.get(key, []):
                    if alias in surface_observations:
                        val_arr = surface_observations[alias]
                        break
                    # Case-insensitive check
                    for k_in in surface_observations.keys():
                        if k_in.lower() == alias.lower():
                            val_arr = surface_observations[k_in]
                            break
                    if val_arr is not None:
                        break

            if val_arr is None:
                raise KeyError(
                    f"Missing required physical surface variable: '{key}'. "
                    f"Provided keys: {list(surface_observations.keys())}"
                )

            arr = np.asarray(val_arr, dtype=np.float32).copy()
            if arr.ndim != 2:
                raise ValueError(f"Variable '{key}' must be a 2D array, got shape {arr.shape}.")

            if first_shape is None:
                first_shape = arr.shape
            elif arr.shape != first_shape:
                raise ValueError(
                    f"Variable '{key}' shape {arr.shape} does not match expected shape {first_shape}."
                )

            # Determine validity mask
            explicit_mask = None
            if masks and key in masks:
                explicit_mask = np.asarray(masks[key], dtype=np.float32)
            elif f"mask_{key}" in surface_observations:
                explicit_mask = np.asarray(surface_observations[f"mask_{key}"], dtype=np.float32)

            lims = DEFAULT_QC_LIMITS[key]
            fill_val = CLIMATOLOGICAL_FILL_VALUES[key]

            # Identify invalid/missing values (NaN, Inf, out of QC bounds)
            is_nan_inf = ~np.isfinite(arr)
            is_out_of_bounds = (arr < lims["min"]) | (arr > lims["max"])
            invalid = is_nan_inf | is_out_of_bounds

            if explicit_mask is not None:
                if explicit_mask.shape != first_shape:
                    raise ValueError(f"Mask for '{key}' has shape {explicit_mask.shape}, expected {first_shape}")
                m = np.where(invalid | (explicit_mask == 0.0), 0.0, 1.0).astype(np.float32)
            else:
                m = np.where(invalid, 0.0, 1.0).astype(np.float32)

            # Impute invalid entries with regional climatological fill value
            arr[m == 0.0] = fill_val

            # Apply training-only z-score normalization
            mu = float(self.norm_stats[key]["mean"])
            sigma = float(self.norm_stats[key]["std"])
            if abs(sigma) < 1e-8:
                sigma = 1.0
            norm_arr = (arr - mu) / sigma

            norm_phys_channels.append(norm_arr)
            mask_channels.append(m)

        # Assemble into 14 channels
        phys_stack = np.stack(norm_phys_channels, axis=0)  # [7, 101, 241]
        mask_stack = np.stack(mask_channels, axis=0)        # [7, 101, 241]
        input_14 = np.concatenate([phys_stack, mask_stack], axis=0).astype(np.float32)  # [14, 101, 241]

        assert np.all(np.isfinite(input_14)), "Input assembly produced non-finite values!"
        t_tensor = torch.from_numpy(input_14[np.newaxis, ...]).to(torch.float32)  # [1, 14, 101, 241]
        return t_tensor, mask_stack

    def predict(
        self,
        surface_observations: Union[Dict[str, np.ndarray], np.ndarray, torch.Tensor],
        masks: Optional[Dict[str, np.ndarray]] = None,
        date: Optional[str] = None,
        return_embedding: bool = True,
        return_attention: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes deterministic inference on surface observations.

        Args:
            surface_observations: Dict of 7 variables or [14, H, W] array/tensor.
            masks: Optional explicit validity masks for dictionary inputs.
            date: Optional date string (e.g. '2019-06-15') for metadata traceability.
            return_embedding: Whether to extract the 128-D Ocean Embedding Z.
            return_attention: Whether to extract the spatial attention map.

        Returns:
            Dictionary containing:
              - 'temperature': [15, 101, 241] numpy array in °C
              - 'embedding': [128, 101, 241] numpy array (if return_embedding)
              - 'attention': [1, 101, 241] numpy array (if return_attention)
              - 'validity_mask': [7, 101, 241] numpy array
              - 'ocean_mask': [101, 241] boolean array (True where SST is valid ocean)
              - 'metadata': Structured metadata dictionary
        """
        input_tensor, masks_arr = self.prepare_input_tensor(surface_observations, masks=masks)

        input_device = input_tensor.to(self.device)

        with torch.no_grad():
            out = self.model(input_device, return_attention=return_attention)
            temp = out["temperature"].cpu().numpy()
            z = out["embedding"].cpu().numpy() if return_embedding else None
            attn = out.get("attention")
            if attn is not None:
                attn = attn.cpu().numpy()

        is_batched = input_tensor.shape[0] > 1
        if not is_batched:
            temp = temp[0]  # [15, 101, 241]
            if z is not None:
                z = z[0]    # [128, 101, 241]
            if attn is not None:
                attn = attn[0]  # [1, 101, 241]
            if masks_arr.ndim == 4:
                masks_arr = masks_arr[0]

        # Ocean mask: valid ocean where SST observation is valid
        ocean_mask = (masks_arr[0] == 1.0) if masks_arr.ndim == 3 else (masks_arr[:, 0] == 1.0)

        cur_h, cur_w = temp.shape[-2], temp.shape[-1]
        is_default_grid = (cur_h == self.H) and (cur_w == self.W)

        result: Dict[str, Any] = {
            "temperature": temp,
            "validity_mask": masks_arr,
            "ocean_mask": ocean_mask,
            "metadata": {
                "model_name": "OceanEmbed Phase-1 Primary Model",
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_sha256": self.checkpoint_sha256,
                "date": date,
                "depths_m": self.standard_depths,
                "grid": {
                    "lat": self.lat_coords.copy() if is_default_grid else np.arange(cur_h, dtype=np.float64),
                    "lon": self.lon_coords.copy() if is_default_grid else np.arange(cur_w, dtype=np.float64),
                    "resolution": self.resolution if is_default_grid else None,
                    "shape": (cur_h, cur_w),
                    "lat_min": self.lat_min if is_default_grid else 0.0,
                    "lat_max": self.lat_max if is_default_grid else float(cur_h - 1),
                    "lon_min": self.lon_min if is_default_grid else 0.0,
                    "lon_max": self.lon_max if is_default_grid else float(cur_w - 1),
                },
                "units": {
                    "temperature": "°C",
                    "depth": "m",
                    "embedding": "dimensionless",
                    "attention": "dimensionless",
                },
                "device": str(self.device),
                "eval_mode": not self.model.training,
            },
        }

        if return_embedding and z is not None:
            result["embedding"] = z
        if return_attention and attn is not None:
            result["attention"] = attn

        return result


def extract_profile(
    pred_result: Dict[str, Any],
    lat: float,
    lon: float,
) -> Dict[str, Any]:
    """
    Extracts the 15-depth vertical temperature profile at a requested geographic location.

    Args:
        pred_result: Output dictionary from OceanEmbedPredictor.predict().
        lat: Latitude in degrees North (must be in [5.0, 30.0]).
        lon: Longitude in degrees East (must be in [45.0, 105.0]).

    Returns:
        Dictionary containing:
          - 'depth_m': Standard 15 depths [0, 5, ..., 1000] m
          - 'temperature_C': Extracted temperatures [15]
          - 'lat_requested': Requested latitude
          - 'lon_requested': Requested longitude
          - 'lat_grid': Nearest grid cell latitude
          - 'lon_grid': Nearest grid cell longitude
          - 'grid_idx': (lat_idx, lon_idx) tuple
          - 'is_valid_ocean': bool indicating if surface point was validly observed ocean
          - 'embedding': Pointwise 128-D embedding vector (if available)
          - 'attention_weight': Pointwise attention gate weight (if available)
    """
    grid_meta = pred_result["metadata"]["grid"]
    lat_min = grid_meta["lat_min"]
    lat_max = grid_meta["lat_max"]
    lon_min = grid_meta["lon_min"]
    lon_max = grid_meta["lon_max"]
    resolution = grid_meta["resolution"]
    H, W = grid_meta["shape"]

    if not (lat_min <= lat <= lat_max) or not (lon_min <= lon <= lon_max):
        raise ValueError(
            f"Requested coordinates ({lat}°N, {lon}°E) fall outside the North Indian Ocean "
            f"domain [{lat_min}–{lat_max}°N, {lon_min}–{lon_max}°E]."
        )

    lat_coords = grid_meta["lat"]
    lon_coords = grid_meta["lon"]

    # Nearest grid cell index
    lat_idx = int(np.clip(np.round((lat - lat_min) / resolution), 0, H - 1))
    lon_idx = int(np.clip(np.round((lon - lon_min) / resolution), 0, W - 1))

    grid_lat = float(lat_coords[lat_idx])
    grid_lon = float(lon_coords[lon_idx])

    temp_field = pred_result["temperature"]
    if temp_field.ndim == 4:
        temp_profile = temp_field[0, :, lat_idx, lon_idx]
    else:
        temp_profile = temp_field[:, lat_idx, lon_idx]

    ocean_mask = pred_result.get("ocean_mask")
    is_valid = bool(ocean_mask[lat_idx, lon_idx]) if ocean_mask is not None else True

    profile_dict: Dict[str, Any] = {
        "depth_m": list(pred_result["metadata"]["depths_m"]),
        "temperature_C": temp_profile.tolist(),
        "lat_requested": float(lat),
        "lon_requested": float(lon),
        "lat_grid": grid_lat,
        "lon_grid": grid_lon,
        "grid_idx": (lat_idx, lon_idx),
        "is_valid_ocean": is_valid,
    }

    if "embedding" in pred_result:
        emb_field = pred_result["embedding"]
        point_emb = emb_field[0, :, lat_idx, lon_idx] if emb_field.ndim == 4 else emb_field[:, lat_idx, lon_idx]
        profile_dict["embedding"] = point_emb.tolist()

    if "attention" in pred_result:
        attn_field = pred_result["attention"]
        point_attn = attn_field[0, 0, lat_idx, lon_idx] if attn_field.ndim == 4 else attn_field[0, lat_idx, lon_idx]
        profile_dict["attention_weight"] = float(point_attn)

    return profile_dict


def lookup_argo_profile(
    date: str,
    lat: float,
    lon: float,
    max_dist_deg: float = 0.5,
    parquet_path: Optional[Union[str, Path]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Demonstration read-only lookup utility querying the local 2019 Argo in-situ cache.

    CRITICAL SCIENTIFIC SAFETY:
      - This function is STRICTLY READ-ONLY and isolated.
      - In-situ Argo observations never enter model inference or affect normalization.
      - Provided strictly for demonstration and side-by-side UI/dashboard comparison.

    Args:
        date: Calendar date 'YYYY-MM-DD' (e.g. '2019-06-15').
        lat: Target latitude in degrees North.
        lon: Target longitude in degrees East.
        max_dist_deg: Maximum spatial distance radius in degrees (default 0.5°).
        parquet_path: Optional path to Argo parquet file.

    Returns:
        Dict with matched Argo profile soundings and metadata, or None if no match found.
    """
    if parquet_path is None:
        p_path = Path(__file__).resolve().parent.parent / "data" / "argo" / "argo_2019_raw_measurements.parquet"
    else:
        p_path = Path(parquet_path)

    if not p_path.exists():
        return None

    try:
        import pandas as pd
        from pipeline.argo_pipeline import interpolate_argo_profile
    except ImportError:
        return None

    df = pd.read_parquet(p_path)
    # Match date prefix
    date_clean = str(date).strip()[:10]
    df_date = df[df["time"].astype(str).str.startswith(date_clean)]
    if df_date.empty:
        return None

    # Find unique profiles
    unique_profiles = df_date[["PLATFORM_NUMBER", "CYCLE_NUMBER", "latitude", "longitude", "time"]].drop_duplicates()
    if unique_profiles.empty:
        return None

    dists = np.sqrt(
        (unique_profiles["latitude"] - lat) ** 2 + (unique_profiles["longitude"] - lon) ** 2
    )
    min_idx = dists.idxmin()
    min_dist = float(dists.loc[min_idx])

    if min_dist > max_dist_deg:
        return None

    best_prof = unique_profiles.loc[min_idx]
    plat = best_prof["PLATFORM_NUMBER"]
    cycle = best_prof["CYCLE_NUMBER"]

    prof_soundings = df_date[
        (df_date["PLATFORM_NUMBER"] == plat) & (df_date["CYCLE_NUMBER"] == cycle)
    ]

    pressures = prof_soundings["PRES"].values.astype(np.float32)
    temperatures = prof_soundings["TEMP"].values.astype(np.float32)
    qc_flags = prof_soundings["TEMP_QC"].values

    interp_temp, valid_mask = interpolate_argo_profile(
        pressures=pressures,
        temperatures=temperatures,
        target_depths=STANDARD_DEPTHS,
        qc_flags=qc_flags,
    )

    # Convert NaNs to None for clean JSON serialization
    temp_list = [float(t) if np.isfinite(t) else None for t in interp_temp]

    return {
        "source": "in_situ_argo_observation",
        "platform_number": str(plat),
        "cycle_number": int(cycle),
        "time": str(best_prof["time"]),
        "lat": float(best_prof["latitude"]),
        "lon": float(best_prof["longitude"]),
        "distance_deg": round(min_dist, 4),
        "depth_m": STANDARD_DEPTHS,
        "temperature_C": temp_list,
        "valid_depths_mask": valid_mask.tolist(),
        "num_valid_depths": int(valid_mask.sum()),
    }
