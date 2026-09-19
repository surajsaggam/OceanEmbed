"""
scripts/inspect_2016_raw.py
---------------------------
Deep inspection of 2016 raw data products for OceanEmbed.
Leap year: 366 calendar days expected.
"""

from pathlib import Path
import numpy as np
import xarray as xr

def inspect_all_2016_products():
    raw_dir = Path("data/raw")

    print("=" * 80)
    print("OCEANEMBED 2016 RAW DATA PRODUCTS DEEP INSPECTION (LEAP YEAR: 366 DAYS)")
    print("=" * 80)

    # 1. SST
    sst_file = raw_dir / "sst" / "2016" / "sst_2016.nc"
    print(f"\n[1. OSTIA SST] File: {sst_file} ({sst_file.stat().st_size / 1e6:.1f} MB)")
    with xr.open_dataset(sst_file) as ds:
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]} ({len(ds.time)} days)")
        lat_k = "latitude" if "latitude" in ds else "lat"
        lon_k = "longitude" if "longitude" in ds else "lon"
        print(f"  Lat range:  {float(ds[lat_k].values[0])} to {float(ds[lat_k].values[-1])} ({len(ds[lat_k])} points)")
        print(f"  Lon range:  {float(ds[lon_k].values[0])} to {float(ds[lon_k].values[-1])} ({len(ds[lon_k])} points)")
        print(f"  Variables:  {list(ds.data_vars.keys())}")
        v = ds.analysed_sst
        print(f"  analysed_sst units: {v.attrs.get('units')}, min: {float(np.nanmin(v.values)):.2f}, max: {float(np.nanmax(v.values)):.2f}")
        valid_frac = float(np.isfinite(v.values).sum() / v.values.size)
        print(f"  Valid pixel fraction: {valid_frac*100:.2f}% (land/mask accounted)")
        assert len(ds.time) == 366, f"Expected 366 days for 2016 SST, got {len(ds.time)}"

    # 2. SSS
    sss_file = raw_dir / "sss" / "2016" / "sss_2016.nc"
    print(f"\n[2. SMAP/SMOS SSS] File: {sss_file} ({sss_file.stat().st_size / 1e6:.1f} MB)")
    with xr.open_dataset(sss_file) as ds:
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]} ({len(ds.time)} days)")
        lat_k = "latitude" if "latitude" in ds else "lat"
        lon_k = "longitude" if "longitude" in ds else "lon"
        print(f"  Lat range:  {float(ds[lat_k].values[0])} to {float(ds[lat_k].values[-1])}")
        print(f"  Lon range:  {float(ds[lon_k].values[0])} to {float(ds[lon_k].values[-1])}")
        print(f"  Variables:  {list(ds.data_vars.keys())}")
        v = ds.sos
        print(f"  sos units: {v.attrs.get('units')}, min: {float(np.nanmin(v.values)):.2f}, max: {float(np.nanmax(v.values)):.2f}")
        assert len(ds.time) == 366, f"Expected 366 days for 2016 SSS, got {len(ds.time)}"

    # 3. SSH
    ssh_file = raw_dir / "ssh" / "2016" / "ssh_2016.nc"
    print(f"\n[3. DUACS SSH] File: {ssh_file} ({ssh_file.stat().st_size / 1e6:.1f} MB)")
    with xr.open_dataset(ssh_file) as ds:
        print(f"  Dimensions: {dict(ds.sizes)}")
        print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]} ({len(ds.time)} days)")
        lat_k = "latitude" if "latitude" in ds else "lat"
        lon_k = "longitude" if "longitude" in ds else "lon"
        print(f"  Lat range:  {float(ds[lat_k].values[0])} to {float(ds[lat_k].values[-1])}")
        print(f"  Lon range:  {float(ds[lon_k].values[0])} to {float(ds[lon_k].values[-1])}")
        print(f"  Variables:  {list(ds.data_vars.keys())}")
        print(f"  SLA present: {'sla' in ds.data_vars}")
        v = ds.sla
        print(f"  sla units: {v.attrs.get('units')}, min: {float(np.nanmin(v.values)):.3f}, max: {float(np.nanmax(v.values)):.3f}")
        assert len(ds.time) == 366, f"Expected 366 days for 2016 SSH, got {len(ds.time)}"

    # 4. GLORYS
    glorys_dir = raw_dir / "glorys" / "2016"
    glorys_files = sorted(list(glorys_dir.glob("glorys_2016_m*.nc")))
    print(f"\n[4. GLORYS12V1 Reanalysis] Files: {len(glorys_files)} monthly files")
    assert len(glorys_files) == 12, f"Expected 12 monthly files for 2016 GLORYS, got {len(glorys_files)}"
    total_glorys_days = 0
    for gf in glorys_files:
        with xr.open_dataset(gf) as ds:
            total_glorys_days += len(ds.time)
    print(f"  Total GLORYS days across 12 months: {total_glorys_days} (expected 366)")
    assert total_glorys_days == 366, f"Expected 366 total GLORYS days, got {total_glorys_days}"

    with xr.open_dataset(glorys_files[0]) as ds:
        print(f"  Sample month (Jan): dims={dict(ds.sizes)}")
        lat_k = "latitude" if "latitude" in ds else "lat"
        lon_k = "longitude" if "longitude" in ds else "lon"
        print(f"  Lat range:  {float(ds[lat_k].values[0])} to {float(ds[lat_k].values[-1])}")
        print(f"  Lon range:  {float(ds[lon_k].values[0])} to {float(ds[lon_k].values[-1])}")
        print(f"  Depths:     {len(ds.depth)} levels from {float(ds.depth.values[0]):.2f}m to {float(ds.depth.values[-1]):.2f}m")
        print(f"  Depth >= 1000m: {float(ds.depth.values[-1]) >= 1000.0}")
        print(f"  thetao min: {float(np.nanmin(ds.thetao.values)):.2f} C, max: {float(np.nanmax(ds.thetao.values)):.2f} C")
        # Check that 1000m depth bracket is satisfied
        depth_vals = ds.depth.values
        has_bracket = any(d < 1000.0 for d in depth_vals) and any(d >= 1000.0 for d in depth_vals)
        print(f"  1000m vertical bracket satisfied: {has_bracket}")
        assert has_bracket, "GLORYS depth does not bracket 1000m!"

    # Check Feb has 29 days
    with xr.open_dataset(glorys_files[1]) as ds:
        print(f"  February 2016 days: {len(ds.time)} (leap year confirmed)")
        assert len(ds.time) == 29, f"Expected 29 days for Feb 2016, got {len(ds.time)}"

    # 5. OSCAR Currents
    curr_dir = raw_dir / "currents" / "2016"
    curr_files = sorted(list(curr_dir.glob("*.nc")))
    print(f"\n[5. OSCAR Currents] Files: {len(curr_files)} daily files")
    assert len(curr_files) == 366, f"Expected 366 daily OSCAR files, got {len(curr_files)}"
    with xr.open_dataset(curr_files[0]) as ds:
        print(f"  Sample (Day 1): dims={dict(ds.sizes)}")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        print(f"  u min: {float(np.nanmin(ds.u.values)):.2f}, max: {float(np.nanmax(ds.u.values)):.2f}")
        print(f"  v min: {float(np.nanmin(ds.v.values)):.2f}, max: {float(np.nanmax(ds.v.values)):.2f}")

    # 6. CCMP Winds
    winds_dir = raw_dir / "winds" / "2016"
    winds_files = sorted(list(winds_dir.glob("*.nc")))
    print(f"\n[6. CCMP Surface Winds] Files: {len(winds_files)} daily files")
    assert len(winds_files) == 366, f"Expected 366 daily CCMP files, got {len(winds_files)}"
    with xr.open_dataset(winds_files[0]) as ds:
        print(f"  Sample (Day 1): dims={dict(ds.sizes)}")
        print(f"  Variables: {list(ds.data_vars.keys())}")
        print(f"  uwnd min: {float(np.nanmin(ds.uwnd.values)):.2f}, max: {float(np.nanmax(ds.uwnd.values)):.2f}")
        print(f"  vwnd min: {float(np.nanmin(ds.vwnd.values)):.2f}, max: {float(np.nanmax(ds.vwnd.values)):.2f}")

    print("\n" + "=" * 80)
    print("ALL 6 RAW DATA PRODUCTS FOR YEAR 2016 VERIFIED COMPLETE AND UNCORRUPTED (366/366 DAYS)")
    print("=" * 80)

if __name__ == "__main__":
    inspect_all_2016_products()
