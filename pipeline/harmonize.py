"""
pipeline/harmonize.py
---------------------
Harmonization and multi-source data fusion for OceanEmbed.

Coordinates:
  1. Physical units conversion (e.g. OSTIA SST Kelvin -> Celsius).
  2. Variable selection (confirming SLA is used rather than ADT for sea surface height).
  3. Temporal harmonization (CCMP 6-hourly winds -> daily mean).
  4. Coordinate order and dimension standardization (e.g. OSCAR transposition and coordinate mapping).
  5. 2D horizontal regridding to common 0.25° NIO target grid (H=101, W=241).
  6. GLORYS 3D vertical interpolation to 15 standard depths with zero extrapolation and shallow-water bathymetric masking.
  7. Production of 14-channel model input tensor and 15-channel target tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d
import torch
import xarray as xr

from pipeline.mask import apply_land_mask
from pipeline.qc import apply_variable_qc
from pipeline.regrid import get_default_target_coords, regrid_2d
from utils.config import load_config


class DataHarmonizer:
    """
    Harmonizes multi-source raw satellite observations and GLORYS target data
    onto the common 0.25° North Indian Ocean grid.
    """

    def __init__(self, data_cfg: Optional[Any] = None) -> None:
        self.data_cfg = data_cfg or load_config("data")
        self.lat_target, self.lon_target = get_default_target_coords()
        self.H = len(self.lat_target)  # 101
        self.W = len(self.lon_target)  # 241
        self.target_depths = np.array(self.data_cfg.depths_m, dtype=np.float32)  # 15 depths
        self.qc_limits = self.data_cfg.qc_limits

    def harmonize_sst(self, sst_file: Union[str, Path], time_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmonizes OSTIA SST:
          - Converts Kelvin to Celsius (T_C = T_K - 273.15).
          - Regrids from native 0.05° to common 0.25° grid.
          - Applies physical range QC [-2.0, 36.0]°C.

        Returns:
            sst_regrid: [101, 241] array of SST in °C.
            mask: [101, 241] binary validity mask.
        """
        ds = xr.open_dataset(sst_file)
        # Slices 2D array
        sst_k = ds.analysed_sst.isel(time=time_idx).values
        sst_c = (sst_k - 273.15).astype(np.float32)

        lat_src = ds.latitude.values
        lon_src = ds.longitude.values
        ds.close()

        sst_regrid = regrid_2d(sst_c, lat_src, lon_src, self.lat_target, self.lon_target)
        # QC range check
        fill = float(self.qc_limits.SST.min) if hasattr(self.qc_limits, "SST") else 25.0
        sst_clean, mask = apply_variable_qc("SST", sst_regrid, self.qc_limits, fill_value=25.0)
        return sst_clean, mask

    def harmonize_sss(self, sss_file: Union[str, Path], time_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmonizes SMAP/SMOS SSS:
          - Units are PSU.
          - Regrids from native 0.125° to common 0.25° grid.
          - Applies physical range QC [0.0, 45.0] PSU.
        """
        ds = xr.open_dataset(sss_file)
        var = ds.sos.isel(time=time_idx)
        if "depth" in var.dims:
            var = var.isel(depth=0)
        sss_raw = var.values.astype(np.float32)

        lat_src = ds.latitude.values
        lon_src = ds.longitude.values
        ds.close()

        sss_regrid = regrid_2d(sss_raw, lat_src, lon_src, self.lat_target, self.lon_target)
        sss_clean, mask = apply_variable_qc("SSS", sss_regrid, self.qc_limits, fill_value=34.0)
        return sss_clean, mask

    def harmonize_ssh(self, ssh_file: Union[str, Path], time_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmonizes DUACS Sea Level Anomaly (SLA):
          - Explicitly selects 'sla' (Sea Level Anomaly) rather than 'adt'.
          - Units are meters.
          - Regrids from native 0.25° to common 0.25° grid.
          - Applies physical range QC [-2.0, 2.0] m.
        """
        ds = xr.open_dataset(ssh_file)
        sla_raw = ds.sla.isel(time=time_idx).values.astype(np.float32)

        lat_src = ds.latitude.values
        lon_src = ds.longitude.values
        ds.close()

        sla_regrid = regrid_2d(sla_raw, lat_src, lon_src, self.lat_target, self.lon_target)
        sla_clean, mask = apply_variable_qc("SSH", sla_regrid, self.qc_limits, fill_value=0.0)
        return sla_clean, mask

    def harmonize_currents(self, currents_file: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Harmonizes OSCAR Surface Currents:
          - Accounts for non-standard dimension order ('time', 'longitude', 'latitude').
          - Transposes to ('latitude', 'longitude') using ds.lat and ds.lon coordinates.
          - Selects longitude range [45, 105] from [0, 360).
          - Regrids u and v components to common 0.25° grid.
          - Applies physical range QC [-3.0, 3.0] m/s.
        """
        ds = xr.open_dataset(currents_file)
        u_raw = ds.u.isel(time=0).transpose("latitude", "longitude").values.astype(np.float32)
        v_raw = ds.v.isel(time=0).transpose("latitude", "longitude").values.astype(np.float32)

        lat_src = ds.lat.values
        lon_src = ds.lon.values
        ds.close()

        u_regrid = regrid_2d(u_raw, lat_src, lon_src, self.lat_target, self.lon_target)
        v_regrid = regrid_2d(v_raw, lat_src, lon_src, self.lat_target, self.lon_target)

        u_clean, mask_u = apply_variable_qc("U_curr", u_regrid, self.qc_limits, fill_value=0.0)
        v_clean, mask_v = apply_variable_qc("V_curr", v_regrid, self.qc_limits, fill_value=0.0)
        return u_clean, mask_u, v_clean, mask_v

    def harmonize_winds(self, winds_file: Union[str, Path], method: str = "daily_mean") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Harmonizes CCMP v3.1 6-Hourly Winds to daily:
          - method="daily_mean": Averages across the 4 synoptic times (00:00, 06:00, 12:00, 18:00 UTC)
            to represent daily wind stress forcing.
          - method="12utc": Selects the 12:00 UTC synoptic snapshot.
          - Regrids uwnd and vwnd to common 0.25° grid.
          - Applies physical range QC [-30.0, 30.0] m/s.
        """
        ds = xr.open_dataset(winds_file)
        if method == "daily_mean":
            u_daily = ds.uwnd.mean(dim="time").values.astype(np.float32)
            v_daily = ds.vwnd.mean(dim="time").values.astype(np.float32)
        else:
            # 12:00 UTC slice is index 2
            u_daily = ds.uwnd.isel(time=2).values.astype(np.float32)
            v_daily = ds.vwnd.isel(time=2).values.astype(np.float32)

        lat_src = ds.latitude.values
        lon_src = ds.longitude.values
        ds.close()

        u_regrid = regrid_2d(u_daily, lat_src, lon_src, self.lat_target, self.lon_target)
        v_regrid = regrid_2d(v_daily, lat_src, lon_src, self.lat_target, self.lon_target)

        u_clean, mask_u = apply_variable_qc("WindU", u_regrid, self.qc_limits, fill_value=0.0)
        v_clean, mask_v = apply_variable_qc("WindV", v_regrid, self.qc_limits, fill_value=0.0)
        return u_clean, mask_u, v_clean, mask_v

    def harmonize_glorys_target(
        self, glorys_file: Union[str, Path], time_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmonizes GLORYS12V1 3D temperature target:
          - Regrids horizontally from native 0.0833° to target 0.25° grid.
          - Vertically interpolates from native 36 levels to 15 standard depths.
          - ZERO below-deepest extrapolation (1000m bracketed by 902.339m and 1062.44m).
          - Proper shallow-water bathymetric masking (levels deeper than seafloor remain unobserved/masked).

        Returns:
            target_15d: [15, 101, 241] temperature array in °C.
            target_mask: [15, 101, 241] binary validity mask.
        """
        ds = xr.open_dataset(glorys_file)
        t_raw = ds.thetao.isel(time=time_idx).values.astype(np.float32)  # [36, 301, 721]
        native_depths = ds.depth.values.astype(np.float32)
        lat_src = ds.latitude.values
        lon_src = ds.longitude.values
        ds.close()

        assert native_depths[-1] >= 1000.0, (
            f"GLORYS native depths terminate at {native_depths[-1]:.2f}m (< 1000m)! "
            f"Cannot interpolate to 1000m without forbidden extrapolation."
        )

        n_native = len(native_depths)
        # 1. Regrid horizontally
        t_regrid_native = np.zeros((n_native, self.H, self.W), dtype=np.float32)
        for k in range(n_native):
            t_regrid_native[k] = regrid_2d(t_raw[k], lat_src, lon_src, self.lat_target, self.lon_target)

        # 2. Vertical interpolation per spatial column
        t_15d = np.full((len(self.target_depths), self.H, self.W), np.nan, dtype=np.float32)

        for i in range(self.H):
            for j in range(self.W):
                col = t_regrid_native[:, i, j]
                valid_idx = np.where(np.isfinite(col))[0]
                if len(valid_idx) < 2:
                    continue

                z_valid = native_depths[valid_idx]
                t_valid = col[valid_idx]

                # Surface extension: top level (0.494m) extends to 0.0m
                if z_valid[0] < 5.0:
                    z_ext = np.insert(z_valid, 0, 0.0)
                    t_ext = np.insert(t_valid, 0, t_valid[0])
                else:
                    z_ext, t_ext = z_valid, t_valid

                # Strictly bounded linear interpolation: fill_value=np.nan prohibits extrapolation
                f_interp = interp1d(z_ext, t_ext, kind="linear", bounds_error=False, fill_value=np.nan)
                t_15d[:, i, j] = f_interp(self.target_depths)

        target_mask = np.isfinite(t_15d).astype(np.float32)
        return t_15d, target_mask

    def build_daily_sample(
        self,
        sst_file: Union[str, Path],
        sss_file: Union[str, Path],
        ssh_file: Union[str, Path],
        currents_file: Union[str, Path],
        winds_file: Union[str, Path],
        glorys_file: Union[str, Path],
        time_idx: int = 0,
        norm_stats: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Builds a full 14-channel input tensor and 15-channel target tensor for a single day.
        Guarantees torch.isfinite(input).all() is True.
        """
        # 1. Harmonize each variable
        sst, m_sst = self.harmonize_sst(sst_file, time_idx=time_idx)
        sss, m_sss = self.harmonize_sss(sss_file, time_idx=time_idx)
        sla, m_sla = self.harmonize_ssh(ssh_file, time_idx=time_idx)
        u_curr, m_uc, v_curr, m_vc = self.harmonize_currents(currents_file)
        u_wind, m_uw, v_wind, m_vw = self.harmonize_winds(winds_file)

        phys_vars = [sst, sss, sla, u_curr, v_curr, u_wind, v_wind]
        masks = [m_sst, m_sss, m_sla, m_uc, m_vc, m_uw, m_vw]
        var_keys = ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]

        # 2. Normalization (using provided stats or standard scaling)
        norm_phys = []
        for v_arr, v_key in zip(phys_vars, var_keys):
            if norm_stats and v_key in norm_stats:
                mu = norm_stats[v_key]["mean"]
                sigma = norm_stats[v_key]["std"]
                v_norm = (v_arr - mu) / sigma
            else:
                # Default unit scaling
                v_norm = v_arr.copy()
            norm_phys.append(v_norm)

        # 3. Stack into [14, H, W] input tensor
        phys_stack = np.stack(norm_phys, axis=0)  # [7, 101, 241]
        mask_stack = np.stack(masks, axis=0)      # [7, 101, 241]

        input_arr = np.concatenate([phys_stack, mask_stack], axis=0).astype(np.float32)  # [14, 101, 241]

        # Finiteness guarantee
        assert np.all(np.isfinite(input_arr)), "Non-finite values found in input array!"

        # 4. GLORYS Target [15, H, W]
        target_15d, target_mask = self.harmonize_glorys_target(glorys_file, time_idx=time_idx)

        # Land/missing values on target carry finite fill for tensor storage,
        # with target_mask preserving the exact valid ocean footprint
        target_clean = np.where(np.isfinite(target_15d), target_15d, 0.0).astype(np.float32)

        in_tensor = torch.from_numpy(input_arr).to(torch.float32)
        target_tensor = torch.from_numpy(target_clean).to(torch.float32)
        target_mask_tensor = torch.from_numpy(target_mask).to(torch.float32)

        meta = {
            "time_idx": time_idx,
            "H": self.H,
            "W": self.W,
            "n_depths": len(self.target_depths),
            "target_mask": target_mask_tensor,
        }
        return in_tensor, target_tensor, meta
