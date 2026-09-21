"""
scripts/inspect_2019_raw.py
---------------------------
Deep inspection of 2019 raw data products for OceanEmbed.
Standard year: 365 calendar days expected.
"""

from pathlib import Path
import numpy as np
import xarray as xr

def inspect_all_2019_products():
    raw_dir = Path("data/raw")

    print("=" * 80)
    print("OCEANEMBED 2019 RAW DATA PRODUCTS DEEP INSPECTION (365 DAYS)")
    print("=" * 80)

    # 1. SST
    sst_file = raw_dir / "sst" / "2019" / "sst_2019.nc"
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
        assert len(ds.time) == 365, f"Expected 365 days for 2019 SST, got {len(ds.time)}"

    # 2. SSS
    sss_file = raw_dir / "sss" / "2019" / "sss_2019.nc"
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
        assert len(ds.time) == 365, f"Expected 365 days for 2019 SSS, got {len(ds.time)}"

    # 3. SSH
    ssh_file = raw_dir / "ssh" / "2019" / "ssh_2019.nc"
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
        assert len(ds.time) == 365, f"Expected 365 days for 2019 SSH, got {len(ds.time)}"

    # 4. GLORYS
    glorys_dir = raw_dir / "glorys" / "2019"
    glorys_files = sorted(list(glorys_dir.glob("glorys_2019_m*.nc")))
    print(f"\n[4. GLORYS12V1 Reanalysis] Files: {len(glorys_files)} monthly files")
    assert len(glorys_files) == 12, f"Expected 12 monthly GLORYS files for 2019, got {len(glorys_files)}"
    total_days = 0
    with xr.open_dataset(glorys_files[0]) as ds:
        depth_coords = ds.depth.values
        print(f"  GLORYS depth coordinates count: {len(depth_coords)}")
        print(f"  Depth range: {depth_coords[0]:.2f}m to {depth_coords[-1]:.2f}m")
        print(f"  1000m bracket present: {any(d >= 1000.0 for d in depth_coords)}")
        assert any(d >= 1000.0 for d in depth_coords), "Missing depth >= 1000m in GLORYS!"
    for gf in glorys_files:
        with xr.open_dataset(gf) as ds:
            total_days += len(ds.time)
    print(f"  Total GLORYS days across 12 files: {total_days}")
    assert total_days == 365, f"Expected 365 GLORYS days for 2019, got {total_days}"

    # 5. OSCAR Currents
    oscar_dir = raw_dir / "currents" / "2019"
    oscar_files = sorted(list(oscar_dir.glob("oscar_*.nc")))
    print(f"\n[5. OSCAR Surface Currents] Daily files: {len(oscar_files)}")
    assert len(oscar_files) == 365, f"Expected 365 OSCAR files for 2019, got {len(oscar_files)}"
    with xr.open_dataset(oscar_files[0]) as ds:
        print(f"  Sample dimensions: {dict(ds.sizes)}")
        print(f"  Variables: u in ds -> {'u' in ds}, v in ds -> {'v' in ds}")
        assert "u" in ds and "v" in ds

    # 6. CCMP Winds
    ccmp_dir = raw_dir / "winds" / "2019"
    ccmp_files = sorted(list(ccmp_dir.glob("ccmp_*.nc")))
    print(f"\n[6. CCMP Surface Winds] Daily files: {len(ccmp_files)}")
    assert len(ccmp_files) == 365, f"Expected 365 CCMP files for 2019, got {len(ccmp_files)}"
    with xr.open_dataset(ccmp_files[0]) as ds:
        print(f"  Sample dimensions: {dict(ds.sizes)}")
        print(f"  Variables: uwnd in ds -> {'uwnd' in ds}, vwnd in ds -> {'vwnd' in ds}")
        assert "uwnd" in ds and "vwnd" in ds

    # Total Storage
    total_bytes = 0
    all_2019_files = list(raw_dir.glob("**/2019/*")) + list(raw_dir.glob("**/2019/*/*"))
    for f in all_2019_files:
        if f.is_file():
            total_bytes += f.stat().st_size
    print(f"\n[Total 2019 Raw Storage]: {total_bytes / (1024**3):.3f} GB across {len(all_2019_files)} files")
    print("=" * 80)
    print("2019 RAW DATA PRODUCTS 100% VERIFIED AND COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    inspect_all_2019_products()
