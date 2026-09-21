"""
scripts/preprocess_year.py
--------------------------
Production staged preprocessor for a single calendar year in OceanEmbed.

Scientific contract:
  1. Generates unnormalized daily tensors in data/interim/<YYYY>/
  2. Input: [14, 101, 241]
     - Channels 0..6: 7 physical surface variables in native physical units (°C, PSU, m, m/s, m/s)
     - Channels 7..13: 7 binary validity masks (1.0 = observed ocean, 0.0 = land/imputed)
  3. Target: [15, 101, 241] GLORYS thetao interpolated to 15 standard depths down to 1000m.
  4. Target mask: [15, 101, 241] binary validity mask.
  5. Strict Finiteness: All stored tensors guaranteed finite.
  6. Normalization Moments: Accumulates double-precision moments (sum, sq_sum, count)
     for the 7 physical variables over mask == 1.0 and saves to moments_<YYYY>.json.
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import xarray as xr

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.cleanup import get_expected_dates_for_year
from pipeline.harmonize import DataHarmonizer
from pipeline.normalize import (
    PHYSICAL_KEYS,
    compute_moments,
    save_norm_stats,
)
from pipeline.regrid import get_default_target_coords
from utils.config import load_config


def preprocess_single_year(
    year: int,
    raw_dir: Optional[Path] = None,
    interim_dir: Optional[Path] = None,
    norm_stats_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Preprocesses all calendar days for a given year into data/interim/<YYYY>/.
    """
    data_cfg = load_config("data")
    raw_root = Path(raw_dir or data_cfg.raw_dir)
    interim_root = Path(interim_dir or "data/interim") / str(year)
    norm_stats_root = Path(norm_stats_dir or "data/norm_stats")

    interim_root.mkdir(parents=True, exist_ok=True)
    norm_stats_root.mkdir(parents=True, exist_ok=True)

    harmonizer = DataHarmonizer(data_cfg)
    lat_tgt, lon_tgt = get_default_target_coords()
    H, W = len(lat_tgt), len(lon_tgt)
    target_depths = list(data_cfg.depths_m)

    expected_dates = get_expected_dates_for_year(year)
    total_days = len(expected_dates)

    print("=" * 80)
    print(f"OCEANEMBED PRODUCTION PREPROCESSOR — YEAR {year} ({total_days} Days)")
    print(f"  Target interim output: {interim_root}")
    print("=" * 80)

    # 1. Discover raw files
    sst_file = raw_root / "sst" / str(year) / f"sst_{year}.nc"
    sss_file = raw_root / "sss" / str(year) / f"sss_{year}.nc"
    ssh_file = raw_root / "ssh" / str(year) / f"ssh_{year}.nc"
    glorys_dir = raw_root / "glorys" / str(year)
    currents_dir = raw_root / "currents" / str(year)
    winds_dir = raw_root / "winds" / str(year)

    assert sst_file.exists(), f"Missing SST file: {sst_file}"
    assert sss_file.exists(), f"Missing SSS file: {sss_file}"
    assert ssh_file.exists(), f"Missing SSH file: {ssh_file}"

    currents_files = sorted(list(currents_dir.glob("*.nc")))
    winds_files = sorted(list(winds_dir.glob("*.nc")))
    glorys_files = sorted(list(glorys_dir.glob(f"glorys_{year}_m*.nc")))

    assert len(currents_files) >= total_days, f"Expected {total_days} currents files, found {len(currents_files)}"
    assert len(winds_files) >= total_days, f"Expected {total_days} winds files, found {len(winds_files)}"
    assert len(glorys_files) == 12, f"Expected 12 GLORYS monthly files, found {len(glorys_files)}"

    print(f"[Raw Data Inventory] All input datasets verified ({total_days} daily currents/winds, 12 GLORYS months).")

    # 2. Open full-year NetCDF handles once for high throughput
    print("[Preprocessing] Opening annual handles for SST, SSS, and SSH...")
    ds_sst = xr.open_dataset(sst_file)
    ds_sss = xr.open_dataset(sss_file)
    ds_ssh = xr.open_dataset(ssh_file)

    # Moments accumulator for training split
    annual_moments = {
        k: {"sum": 0.0, "sq_sum": 0.0, "count": 0}
        for k in PHYSICAL_KEYS
    }

    current_month = -1
    ds_glorys_month = None

    start_time = time.time()
    processed_count = 0

    try:
        for day_idx, date_str in enumerate(expected_dates):
            # Parse year, month, day
            parts = date_str.split("-")
            month = int(parts[1])
            day_of_month = int(parts[2])

            # Manage monthly GLORYS dataset handle
            if month != current_month:
                if ds_glorys_month is not None:
                    ds_glorys_month.close()
                current_month = month
                m_file = glorys_dir / f"glorys_{year}_m{month:02d}.nc"
                ds_glorys_month = xr.open_dataset(m_file)

            # ── A. Harmonize Surface Variables ──────────────────────────────
            # 1. SST (°C)
            sst_k = ds_sst.analysed_sst.isel(time=day_idx).values.astype(np.float32)
            sst_c = sst_k - 273.15
            lat_src = ds_sst.latitude.values if "latitude" in ds_sst else ds_sst.lat.values
            lon_src = ds_sst.longitude.values if "longitude" in ds_sst else ds_sst.lon.values
            from pipeline.regrid import regrid_2d
            from pipeline.qc import apply_variable_qc
            sst_regrid = regrid_2d(sst_c, lat_src, lon_src, lat_tgt, lon_tgt)
            sst_clean, m_sst = apply_variable_qc("SST", sst_regrid, harmonizer.qc_limits, fill_value=28.0)

            # 2. SSS (PSU)
            var_sss = ds_sss.sos.isel(time=day_idx)
            if "depth" in var_sss.dims:
                var_sss = var_sss.isel(depth=0)
            sss_raw = var_sss.values.astype(np.float32)
            lat_src = ds_sss.latitude.values if "latitude" in ds_sss else ds_sss.lat.values
            lon_src = ds_sss.longitude.values if "longitude" in ds_sss else ds_sss.lon.values
            sss_regrid = regrid_2d(sss_raw, lat_src, lon_src, lat_tgt, lon_tgt)
            sss_clean, m_sss = apply_variable_qc("SSS", sss_regrid, harmonizer.qc_limits, fill_value=34.0)

            # 3. SSH / SLA (m)
            sla_raw = ds_ssh.sla.isel(time=day_idx).values.astype(np.float32)
            lat_src = ds_ssh.latitude.values if "latitude" in ds_ssh else ds_ssh.lat.values
            lon_src = ds_ssh.longitude.values if "longitude" in ds_ssh else ds_ssh.lon.values
            sla_regrid = regrid_2d(sla_raw, lat_src, lon_src, lat_tgt, lon_tgt)
            sla_clean, m_sla = apply_variable_qc("SSH", sla_regrid, harmonizer.qc_limits, fill_value=0.0)

            # 4. Currents (m/s)
            u_curr, m_uc, v_curr, m_vc = harmonizer.harmonize_currents(currents_files[day_idx])

            # 5. Winds (m/s)
            u_wind, m_uw, v_wind, m_vw = harmonizer.harmonize_winds(winds_files[day_idx], method="daily_mean")

            # ── B. GLORYS 3D Target ─────────────────────────────────────────
            # Day of month index in monthly file (0-indexed)
            t_glorys_idx = day_of_month - 1
            t_raw = ds_glorys_month.thetao.isel(time=t_glorys_idx).values.astype(np.float32)  # [depth, lat, lon]
            native_depths = ds_glorys_month.depth.values.astype(np.float32)
            lat_src = ds_glorys_month.latitude.values if "latitude" in ds_glorys_month else ds_glorys_month.lat.values
            lon_src = ds_glorys_month.longitude.values if "longitude" in ds_glorys_month else ds_glorys_month.lon.values

            n_native = len(native_depths)
            t_regrid_native = np.zeros((n_native, H, W), dtype=np.float32)
            for k in range(n_native):
                t_regrid_native[k] = regrid_2d(t_raw[k], lat_src, lon_src, lat_tgt, lon_tgt)

            from scipy.interpolate import interp1d
            t_15d = np.full((len(target_depths), H, W), np.nan, dtype=np.float32)
            for i in range(H):
                for j in range(W):
                    col = t_regrid_native[:, i, j]
                    valid_idx = np.where(np.isfinite(col))[0]
                    if len(valid_idx) < 2:
                        continue
                    z_valid = native_depths[valid_idx]
                    t_valid = col[valid_idx]
                    if z_valid[0] < 5.0:
                        z_ext = np.insert(z_valid, 0, 0.0)
                        t_ext = np.insert(t_valid, 0, t_valid[0])
                    else:
                        z_ext, t_ext = z_valid, t_valid
                    f_interp = interp1d(z_ext, t_ext, kind="linear", bounds_error=False, fill_value=np.nan)
                    t_15d[:, i, j] = f_interp(target_depths)

            target_mask = np.isfinite(t_15d).astype(np.float32)
            target_clean = np.where(np.isfinite(t_15d), t_15d, 0.0).astype(np.float32)

            # ── C. Assemble 14-Channel Unnormalized Input Tensor ─────────────
            phys_vars = [sst_clean, sss_clean, sla_clean, u_curr, v_curr, u_wind, v_wind]
            masks = [m_sst, m_sss, m_sla, m_uc, m_vc, m_uw, m_vw]

            # Accumulate double-precision moments for training statistics
            for k, arr, m in zip(PHYSICAL_KEYS, phys_vars, masks):
                m_dict = compute_moments(arr, m)
                annual_moments[k]["sum"] += m_dict["sum"]
                annual_moments[k]["sq_sum"] += m_dict["sq_sum"]
                annual_moments[k]["count"] += m_dict["count"]

            phys_stack = np.stack(phys_vars, axis=0).astype(np.float32)  # [7, 101, 241]
            mask_stack = np.stack(masks, axis=0).astype(np.float32)      # [7, 101, 241]
            input_tensor = np.concatenate([phys_stack, mask_stack], axis=0)  # [14, 101, 241]

            # Strict validations
            assert np.all(np.isfinite(input_tensor)), f"Non-finite input on {date_str}"
            assert np.all(np.isfinite(target_clean)), f"Non-finite target on {date_str}"
            assert np.all(np.isin(input_tensor[7:], [0.0, 1.0])), f"Non-binary mask on {date_str}"

            # Save daily .npz
            out_file = interim_root / f"oceanembed_{date_str}.npz"
            np.savez_compressed(
                out_file,
                input=input_tensor,
                target=target_clean,
                target_mask=target_mask,
                date=date_str,
                day_idx=day_idx,
            )
            processed_count += 1

            if (day_idx + 1) % 30 == 0 or (day_idx + 1) == total_days:
                elapsed = time.time() - start_time
                rate = (day_idx + 1) / elapsed
                print(f"  Processed {day_idx + 1}/{total_days} days ({rate:.2f} days/s) -> {date_str}")

    finally:
        ds_sst.close()
        ds_sss.close()
        ds_ssh.close()
        if ds_glorys_month is not None:
            ds_glorys_month.close()

    # 3. Save Moments JSON
    moments_payload = {
        "year": year,
        "is_train_split": True,
        "num_days": processed_count,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "moments": annual_moments,
    }

    # Save to data/norm_stats/moments_<YYYY>.json and data/interim/<YYYY>/moments.json
    stats_out_1 = norm_stats_root / f"moments_{year}.json"
    stats_out_2 = interim_root / "moments.json"
    with open(stats_out_1, "w", encoding="utf-8") as f:
        json.dump(moments_payload, f, indent=2)
    with open(stats_out_2, "w", encoding="utf-8") as f:
        json.dump(moments_payload, f, indent=2)

    print(f"\n[Moments Saved] Saved training moments to {stats_out_1} and {stats_out_2}")
    for k in PHYSICAL_KEYS:
        cnt = annual_moments[k]["count"]
        mean_val = annual_moments[k]["sum"] / max(1, cnt)
        var_val = (annual_moments[k]["sq_sum"] / max(1, cnt)) - (mean_val ** 2)
        std_val = float(np.sqrt(max(1e-6, var_val)))
        print(f"  {k:7s} -> mean: {mean_val:8.4f}, std: {std_val:8.4f} (from {cnt} valid points)")

    # 4. Save Year Metadata
    meta_payload = {
        "year": year,
        "total_days": processed_count,
        "date_start": expected_dates[0],
        "date_end": expected_dates[-1],
        "domain": {"H": H, "W": W, "lat": [float(lat_tgt[0]), float(lat_tgt[-1])], "lon": [float(lon_tgt[0]), float(lon_tgt[-1])]},
        "target_depths": target_depths,
        "is_unnormalized": True,
        "status": "complete",
    }
    with open(interim_root / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2)

    total_time = time.time() - start_time
    print(f"\n[Success] Completed preprocessing for Year {year}: {processed_count} days in {total_time:.1f}s ({processed_count/total_time:.2f} days/s).")
    return meta_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a calendar year for OceanEmbed")
    parser.add_argument("--year", type=int, default=2015, help="Year to preprocess")
    args = parser.parse_args()
    preprocess_single_year(args.year)
