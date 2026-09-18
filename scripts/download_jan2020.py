"""
scripts/download_jan2020.py
---------------------------
Automated data downloader for the January 2020 dry run of OceanEmbed.

Datasets acquired:
  1. OSTIA SST                -> Copernicus Marine (CMEMS)
  2. SMAP/SMOS SSS            -> Copernicus Marine (CMEMS)
  3. DUACS SSH/SLA            -> Copernicus Marine (CMEMS)
  4. OSCAR surface currents   -> PO.DAAC / NASA Earthdata
  5. CCMP v3.1 surface winds  -> PO.DAAC / NASA Earthdata
  6. GLORYS12V1 reanalysis    -> Copernicus Marine (CMEMS)

Security:
  - Credentials read from .env or environment variables.
  - No secrets or credentials are ever printed or logged.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import copernicusmarine
import earthaccess

from utils.config import load_config
from utils.credentials import (
    check_credentials_status,
    get_copernicus_credentials,
    get_earthdata_credentials,
    load_env,
)


def verify_credentials() -> bool:
    """
    Checks that credentials for Copernicus Marine and NASA Earthdata are present.
    Prints instructions if missing.
    """
    status = check_credentials_status()
    missing = []
    if not status["copernicus_configured"]:
        missing.append("Copernicus Marine (COPERNICUSMARINE_SERVICE_USERNAME, COPERNICUSMARINE_SERVICE_PASSWORD)")
    if not status["earthdata_configured"]:
        missing.append("NASA Earthdata (EARTHDATA_USERNAME, EARTHDATA_PASSWORD)")

    if missing:
        print("\n" + "=" * 70)
        print("[CREDENTIAL CONFIGURATION REQUIRED]")
        print("=" * 70)
        print("The following credentials were not found in your environment or .env file:")
        for m in missing:
            print(f"  - {m}")
        print("\nTo configure them safely without exposing secrets:")
        print("1. Create a '.env' file in the project root: d:\\OceanEmbed\\.env")
        print("2. Add your credentials (see .env.example for template):")
        print("     COPERNICUSMARINE_SERVICE_USERNAME=your_username")
        print("     COPERNICUSMARINE_SERVICE_PASSWORD=your_password")
        print("     EARTHDATA_USERNAME=your_earthdata_username")
        print("     EARTHDATA_PASSWORD=your_earthdata_password")
        print("3. The .env file is strictly gitignored and will never be committed.")
        print("=" * 70 + "\n")
        return False

    print("[Credentials] Copernicus Marine and NASA Earthdata credentials verified.")
    return True


def download_copernicus_variable(
    var_name: str,
    dataset_id: str,
    output_dir: Path,
    variables: Optional[List[str]],
    data_cfg: Any,
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-31",
    max_depth: Optional[float] = None,
) -> None:
    """
    Downloads spatial subset from Copernicus Marine via the official Python client.
    """
    cm_user, cm_pwd = get_copernicus_credentials()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[CMEMS Download] Requesting {var_name.upper()} ({dataset_id})...")
    print(f"  Domain: lat [{data_cfg.domain.lat_min}, {data_cfg.domain.lat_max}] | lon [{data_cfg.domain.lon_min}, {data_cfg.domain.lon_max}]")
    print(f"  Dates:  {start_date} to {end_date}")

    try:
        copernicusmarine.subset(
            dataset_id=dataset_id,
            username=cm_user,
            password=cm_pwd,
            variables=variables,
            minimum_longitude=float(data_cfg.domain.lon_min),
            maximum_longitude=float(data_cfg.domain.lon_max),
            minimum_latitude=float(data_cfg.domain.lat_min),
            maximum_latitude=float(data_cfg.domain.lat_max),
            minimum_depth=0.0 if max_depth is not None else None,
            maximum_depth=max_depth,
            start_datetime=start_date,
            end_datetime=end_date,
            output_directory=output_dir,
            output_filename=f"{var_name}_jan2020.nc",
            file_format="netcdf",
            overwrite=True,
        )
        print(f"  -> Successfully downloaded {var_name.upper()} to {output_dir}")
    except Exception as e:
        print(f"  [Error] Failed to download {var_name.upper()}: {e}")


def download_earthdata_variable(
    var_name: str,
    short_name: str,
    output_dir: Path,
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-31",
) -> None:
    """
    Searches and downloads data from PO.DAAC / NASA Earthdata using earthaccess.
    """
    load_env()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[NASA Earthdata Download] Requesting {var_name.upper()} ({short_name})...")
    print(f"  Dates: {start_date} to {end_date}")

    try:
        # Authenticate
        auth = earthaccess.login(strategy="environment")
        if not auth.authenticated:
            print("  [Error] Earthdata authentication failed. Check EARTHDATA_USERNAME/EARTHDATA_PASSWORD.")
            return

        granules = earthaccess.search_data(
            short_name=short_name,
            temporal=(start_date, end_date),
        )
        print(f"  Found {len(granules)} granules for {var_name.upper()}. Downloading...")
        earthaccess.download(granules, str(output_dir))
        print(f"  -> Successfully downloaded {var_name.upper()} to {output_dir}")
    except Exception as e:
        print(f"  [Error] Failed to download {var_name.upper()}: {e}")


def run_dry_run_downloads(
    variable: str = "all",
    start_date: str = "2020-01-01",
    end_date: str = "2020-01-31",
    check_only: bool = False,
) -> None:
    data_cfg = load_config("data")
    raw_dir = Path(data_cfg.raw_dir)

    if not verify_credentials():
        return

    if check_only:
        print("\n[Dry Run Check Mode] Testing Earthdata search connection...")
        auth = earthaccess.login(strategy="environment")
        print(f"  Earthdata authenticated: {auth.authenticated}")
        oscar_g = earthaccess.search_data(short_name="OSCAR_L4_OC_FINAL_V2.0", temporal=(start_date, end_date))
        print(f"  OSCAR granules accessible: {len(oscar_g)}")
        ccmp_g = earthaccess.search_data(short_name="CCMP_WINDS_10M6HR_L4_V3.1", temporal=(start_date, end_date))
        print(f"  CCMP granules accessible: {len(ccmp_g)}")
        print("[Dry Run Check Mode] All checks complete.")
        return

    var = variable.lower()

    # 1. OSTIA SST
    if var in ["all", "sst"]:
        download_copernicus_variable(
            var_name="sst",
            dataset_id="METOFFICE-GLO-SST-L4-REP-OBS-SST",
            output_dir=raw_dir / "sst",
            variables=["analysed_sst"],
            data_cfg=data_cfg,
            start_date=start_date,
            end_date=end_date,
        )

    # 2. SMAP/SMOS SSS
    if var in ["all", "sss"]:
        download_copernicus_variable(
            var_name="sss",
            dataset_id="cmems_obs-mob_glo_phy-sss_my_multi_P1D",
            output_dir=raw_dir / "sss",
            variables=["sos"],
            data_cfg=data_cfg,
            start_date=start_date,
            end_date=end_date,
        )

    # 3. DUACS SSH/SLA
    if var in ["all", "ssh"]:
        download_copernicus_variable(
            var_name="ssh",
            dataset_id="c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D",
            output_dir=raw_dir / "ssh",
            variables=["sla", "adt", "ugosa", "vgosa"],
            data_cfg=data_cfg,
            start_date=start_date,
            end_date=end_date,
        )

    # 4. OSCAR Currents
    if var in ["all", "currents"]:
        download_earthdata_variable(
            var_name="currents",
            short_name="OSCAR_L4_OC_FINAL_V2.0",
            output_dir=raw_dir / "currents",
            start_date=start_date,
            end_date=end_date,
        )

    # 5. CCMP Winds
    if var in ["all", "winds"]:
        download_earthdata_variable(
            var_name="winds",
            short_name="CCMP_WINDS_10M6HR_L4_V3.1",
            output_dir=raw_dir / "winds",
            start_date=start_date,
            end_date=end_date,
        )

    # 6. GLORYS12V1 Temperature Target
    if var in ["all", "glorys"]:
        download_copernicus_variable(
            var_name="glorys",
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            output_dir=raw_dir / "glorys",
            variables=["thetao"],
            data_cfg=data_cfg,
            start_date=start_date,
            end_date=end_date,
            max_depth=1100.0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed January 2020 Data Downloader")
    parser.add_argument("--variable", type=str, default="all", choices=["all", "sst", "sss", "ssh", "currents", "winds", "glorys"], help="Variable to download")
    parser.add_argument("--start", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2020-01-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--check-only", action="store_true", help="Verify credentials and search API without downloading files")
    args = parser.parse_args()

    run_dry_run_downloads(
        variable=args.variable,
        start_date=args.start,
        end_date=args.end,
        check_only=args.check_only,
    )
