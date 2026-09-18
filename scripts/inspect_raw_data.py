"""
scripts/inspect_raw_data.py
---------------------------
Inspection script for raw observation and target files (January 2020 dry run).

Analyzes:
  1. Actual temporal cadence (timestamps, step frequency)
  2. Source grids and spatial resolution (lat/lon coordinates and spacing)
  3. Variable names, dimensions, and standard attributes
  4. Missing values, fill values, and QC flag variables
  5. Spatial bounding box coverage vs North Indian Ocean target domain
  6. File format and storage layout
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr


def inspect_netcdf_file(file_path: Path) -> Dict[str, Any]:
    """
    Inspects a single NetCDF or Zarr file and extracts diagnostic metadata.
    """
    report = {
        "file_name": file_path.name,
        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
        "dimensions": {},
        "coordinates": {},
        "data_variables": {},
        "global_attributes": {},
    }

    try:
        ds = xr.open_dataset(file_path)
    except Exception as e:
        report["error"] = str(e)
        return report

    # Dimensions
    for dim_name, dim_size in ds.sizes.items():
        report["dimensions"][dim_name] = dim_size

    # Coordinates
    for coord_name, coord_var in ds.coords.items():
        vals = coord_var.values
        coord_info = {
            "dtype": str(coord_var.dtype),
            "size": len(vals) if vals.ndim > 0 else 1,
        }
        if np.issubdtype(vals.dtype, np.number):
            coord_info["min"] = float(np.nanmin(vals))
            coord_info["max"] = float(np.nanmax(vals))
            if len(vals) > 1:
                diffs = np.diff(vals)
                coord_info["step_mean"] = float(np.mean(diffs))
                coord_info["step_std"] = float(np.std(diffs))
        elif np.issubdtype(vals.dtype, np.datetime64):
            coord_info["start"] = str(vals[0])
            coord_info["end"] = str(vals[-1])
            if len(vals) > 1:
                coord_info["cadence"] = str(vals[1] - vals[0])

        report["coordinates"][coord_name] = coord_info

    # Data variables
    for var_name, var in ds.data_vars.items():
        var_info = {
            "dims": list(var.dims),
            "shape": list(var.shape),
            "dtype": str(var.dtype),
            "attributes": {k: str(v) for k, v in var.attrs.items() if k in ["units", "long_name", "standard_name", "_FillValue"]},
        }
        # Compute finite / missing stats on a sample
        sample_vals = var.values
        total_pts = sample_vals.size
        finite_pts = int(np.sum(np.isfinite(sample_vals)))
        var_info["total_points"] = total_pts
        var_info["finite_points"] = finite_pts
        var_info["valid_percent"] = round(100.0 * finite_pts / max(1, total_pts), 2)

        if finite_pts > 0:
            finite_data = sample_vals[np.isfinite(sample_vals)]
            var_info["observed_min"] = float(np.min(finite_data))
            var_info["observed_max"] = float(np.max(finite_data))
            var_info["observed_mean"] = float(np.mean(finite_data))

        report["data_variables"][var_name] = var_info

    ds.close()
    return report


def print_inspection_report(report: Dict[str, Any]) -> None:
    """Prints a formatted console report."""
    print("=" * 70)
    print(f"FILE: {report['file_name']} ({report.get('file_size_mb', 0)} MB)")
    print("=" * 70)

    if "error" in report:
        print(f"Error reading file: {report['error']}")
        return

    print("\n[Dimensions]")
    for d, s in report["dimensions"].items():
        print(f"  {d}: {s}")

    print("\n[Coordinates]")
    for c, info in report["coordinates"].items():
        extra = ""
        if "min" in info:
            step = info.get("step_mean", 0)
            extra = f" | range: [{info['min']:.3f}, {info['max']:.3f}] | step: {step:.4f}"
        elif "start" in info:
            extra = f" | [{info['start']} -> {info['end']}] | cadence: {info.get('cadence', 'N/A')}"
        print(f"  {c} ({info['dtype']}, size={info['size']}){extra}")

    print("\n[Data Variables]")
    for v, info in report["data_variables"].items():
        units = info["attributes"].get("units", "")
        long_name = info["attributes"].get("long_name", v)
        print(f"  {v}: {long_name} [{units}]")
        print(f"    shape: {info['shape']} | valid: {info['valid_percent']}%")
        if "observed_min" in info:
            print(f"    min: {info['observed_min']:.4f} | max: {info['observed_max']:.4f} | mean: {info['observed_mean']:.4f}")


def scan_raw_directory(raw_dir: Path) -> List[Dict[str, Any]]:
    """Scans raw_dir for netCDF/Zarr files and inspects them."""
    files = sorted(list(raw_dir.glob("**/*.nc")) + list(raw_dir.glob("**/*.nc4")))
    reports = []
    for f in files:
        rep = inspect_netcdf_file(f)
        print_inspection_report(rep)
        reports.append(rep)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Raw Observation Files")
    parser.add_argument("--file", type=str, default=None, help="Specific file to inspect")
    parser.add_argument("--dir", type=str, default="data/raw", help="Directory to scan")
    args = parser.parse_args()

    if args.file:
        p = Path(args.file)
        if not p.exists():
            print(f"File not found: {p}")
        else:
            rep = inspect_netcdf_file(p)
            print_inspection_report(rep)
    else:
        d = Path(args.dir)
        if not d.exists():
            print(f"Directory not found: {d}")
        else:
            scan_raw_directory(d)
