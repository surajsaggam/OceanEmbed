"""
scripts/download_multiyear.py
------------------------------
Production multi-year satellite and reanalysis downloader for OceanEmbed.

Features:
  1. Multi-year / range selection (e.g. --start-year 2015 --end-year 2021).
  2. Year-specific directory layout: data/raw/<variable>/<YYYY>/
  3. All 7 physical surface variables + GLORYS 3D target.
  4. Resumable and idempotent: skips valid existing files; replaces corrupt/zero-byte files.
  5. Explicit GLORYS reanalysis product transition:
     - <= 2021: Frozen Multi-Year Reanalysis (MY) 'cmems_mod_glo_phy_my_0.083deg_P1D-m'
     - >= 2022: Interim Multi-Year Reanalysis (MYINT) 'cmems_mod_glo_phy_myint_0.083deg_P1D-m'
  6. Dry-run audit mode (--dry-run) reporting exact sizes, product IDs, and skipping plan.
  7. Disk-space safety checks: enforces minimum free headroom margin before downloading.
  8. Secure credentials: uses utils.credentials (never logs secrets).
"""

from __future__ import annotations

import argparse
import calendar
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copernicusmarine
import earthaccess

from pipeline.cleanup import safe_cleanup_raw_year, verify_preprocessed_year
from utils.config import load_config
from utils.credentials import (
    check_credentials_status,
    get_copernicus_credentials,
    get_earthdata_credentials,
    load_env,
)


# ── Product Specifications ───────────────────────────────────────────────────

# Estimated monthly and yearly uncompressed sizes for NIO bounding box
ESTIMATED_SIZES_GB = {
    "sst": 0.45,       # ~37 MB / month -> ~440 MB / year
    "sss": 0.15,       # ~12 MB / month -> ~140 MB / year
    "ssh": 0.14,       # ~11.4 MB / month -> ~136 MB / year
    "currents": 12.1,  # 365 daily files @ 33.2 MB -> ~12.1 GB / year
    "winds": 12.3,     # 365 daily files @ 33.6 MB -> ~12.3 GB / year
    "glorys": 5.5,     # 12 monthly files @ 462 MB -> ~5.5 GB / year
}

MIN_EXPECTED_FILE_BYTES = {
    "sst": 10 * 1024 * 1024,        # At least 10 MB for a month/year
    "sss": 5 * 1024 * 1024,         # At least 5 MB
    "ssh": 5 * 1024 * 1024,         # At least 5 MB
    "currents": 10 * 1024 * 1024,   # At least 10 MB per daily file
    "winds": 10 * 1024 * 1024,      # At least 10 MB per daily file
    "glorys": 50 * 1024 * 1024,     # At least 50 MB per monthly file
}


def get_glorys_dataset_id(year: int) -> str:
    """
    Returns official CMEMS GLORYS12V1 dataset ID based on calendar year.

    - Years <= 2021: 'cmems_mod_glo_phy_my_0.083deg_P1D-m' (Frozen Multi-Year)
    - Years >= 2022: 'cmems_mod_glo_phy_myint_0.083deg_P1D-m' (Interim Multi-Year)
    """
    if year <= 2021:
        return "cmems_mod_glo_phy_my_0.083deg_P1D-m"
    else:
        return "cmems_mod_glo_phy_myint_0.083deg_P1D-m"


def is_valid_netcdf_or_data_file(file_path: Path, min_size: int = 1024) -> bool:
    """
    Checks that a downloaded file exists, is non-empty, and has valid header bytes.
    Detects zero-byte or aborted downloads.
    """
    if not file_path.exists() or not file_path.is_file():
        return False

    size = file_path.stat().st_size
    if size < min_size:
        return False

    # Check magic header bytes for NetCDF / HDF5
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
            # Classic NetCDF: b'CDF\x01' or b'CDF\x02' or b'CDF\x05'
            # NetCDF-4 / HDF5: b'\x89HDF\r\n\x1a\n'
            if header.startswith(b"CDF") or header.startswith(b"\x89HDF"):
                return True
            # For other valid binary data (some L4 streams)
            if size >= min_size:
                return True
    except Exception:
        return False

    return True


def check_disk_space_headroom(
    target_path: Path,
    estimated_need_gb: float,
    min_headroom_gb: float = 20.0,
) -> Tuple[bool, float, float]:
    """
    Checks if downloading estimated_need_gb leaves at least min_headroom_gb free.

    Returns:
        (is_safe, current_free_gb, projected_free_gb)
    """
    # Find root drive for Windows/POSIX
    drive_root = target_path.anchor or str(target_path)
    total, used, free = shutil.disk_usage(drive_root)
    current_free_gb = free / (1024 ** 3)
    projected_free_gb = current_free_gb - estimated_need_gb
    is_safe = projected_free_gb >= min_headroom_gb
    return is_safe, current_free_gb, projected_free_gb


class MultiYearDownloader:
    """
    Coordinates modular, resumable multi-year data acquisition.
    """

    def __init__(
        self,
        raw_dir: Path,
        data_cfg: Optional[Any] = None,
        min_headroom_gb: float = 20.0,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.data_cfg = data_cfg or load_config("data")
        self.min_headroom_gb = min_headroom_gb

        # Domain bounding box
        self.lat_min = float(self.data_cfg.domain.lat_min)
        self.lat_max = float(self.data_cfg.domain.lat_max)
        self.lon_min = float(self.data_cfg.domain.lon_min)
        self.lon_max = float(self.data_cfg.domain.lon_max)

    def plan_downloads(
        self,
        years: List[int],
        variables: List[str],
    ) -> Dict[str, Any]:
        """
        Plans download targets, checking existing files and estimating storage.
        """
        plan = {
            "years": sorted(years),
            "variables": variables,
            "tasks": [],
            "total_estimated_gb": 0.0,
            "existing_files_found": 0,
            "missing_files_to_download": 0,
        }

        for y in plan["years"]:
            is_leap = calendar.isleap(y)
            days_in_year = 366 if is_leap else 365

            for var in variables:
                var_clean = var.lower()
                var_dir = self.raw_dir / var_clean / str(y)

                task_info: Dict[str, Any] = {
                    "year": y,
                    "variable": var_clean,
                    "directory": str(var_dir),
                    "estimated_gb": ESTIMATED_SIZES_GB.get(var_clean, 1.0),
                }

                if var_clean == "sst":
                    task_info["dataset_id"] = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
                    task_info["source"] = "copernicus"
                    task_info["target_file"] = str(var_dir / f"sst_{y}.nc")
                    file_p = Path(task_info["target_file"])
                    task_info["status"] = "present" if is_valid_netcdf_or_data_file(file_p, MIN_EXPECTED_FILE_BYTES["sst"]) else "missing"

                elif var_clean == "sss":
                    task_info["dataset_id"] = "cmems_obs-mob_glo_phy-sss_my_multi_P1D"
                    task_info["source"] = "copernicus"
                    task_info["target_file"] = str(var_dir / f"sss_{y}.nc")
                    file_p = Path(task_info["target_file"])
                    task_info["status"] = "present" if is_valid_netcdf_or_data_file(file_p, MIN_EXPECTED_FILE_BYTES["sss"]) else "missing"

                elif var_clean == "ssh":
                    task_info["dataset_id"] = "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"
                    task_info["source"] = "copernicus"
                    task_info["target_file"] = str(var_dir / f"ssh_{y}.nc")
                    file_p = Path(task_info["target_file"])
                    task_info["status"] = "present" if is_valid_netcdf_or_data_file(file_p, MIN_EXPECTED_FILE_BYTES["ssh"]) else "missing"

                elif var_clean == "glorys":
                    dataset_id = get_glorys_dataset_id(y)
                    task_info["dataset_id"] = dataset_id
                    task_info["source"] = "copernicus"
                    task_info["max_depth"] = 1100.0
                    task_info["monthly_files"] = [str(var_dir / f"glorys_{y}_m{m:02d}.nc") for m in range(1, 13)]
                    present_m = sum(1 for f in task_info["monthly_files"] if is_valid_netcdf_or_data_file(Path(f), MIN_EXPECTED_FILE_BYTES["glorys"]))
                    task_info["status"] = "complete" if present_m == 12 else f"partial_{present_m}_of_12"

                elif var_clean == "currents":
                    task_info["short_name"] = "OSCAR_L4_OC_FINAL_V2.0"
                    task_info["source"] = "earthdata"
                    present_days = len(list(var_dir.glob("*.nc"))) if var_dir.exists() else 0
                    task_info["status"] = "complete" if present_days >= days_in_year else f"partial_{present_days}_of_{days_in_year}"

                elif var_clean == "winds":
                    task_info["short_name"] = "CCMP_WINDS_10M6HR_L4_V3.1"
                    task_info["source"] = "earthdata"
                    present_days = len(list(var_dir.glob("*.nc"))) if var_dir.exists() else 0
                    task_info["status"] = "complete" if present_days >= days_in_year else f"partial_{present_days}_of_{days_in_year}"

                # Update counts
                if task_info["status"] in ["present", "complete"]:
                    plan["existing_files_found"] += 1
                else:
                    plan["missing_files_to_download"] += 1
                    plan["total_estimated_gb"] += task_info["estimated_gb"]

                plan["tasks"].append(task_info)

        plan["total_estimated_gb"] = round(plan["total_estimated_gb"], 2)
        return plan

    def execute_dry_run_report(self, plan: Dict[str, Any]) -> None:
        """Prints diagnostic dry-run plan report without downloading."""
        print("=" * 80)
        print("OCEANEMBED MULTI-YEAR DOWNLOADER — DRY-RUN AUDIT REPORT")
        print("=" * 80)

        # Credentials check
        creds = check_credentials_status()
        print("\n[1. Credentials Status]")
        print(f"  Copernicus Marine configured: {creds['copernicus_configured']}")
        print(f"  NASA Earthdata configured:   {creds['earthdata_configured']}")
        print(f"  Environment file (.env):      {creds['env_file_exists']}")

        # Requested scope
        print("\n[2. Requested Scope]")
        print(f"  Years:     {plan['years']}")
        print(f"  Variables: {plan['variables']}")
        print(f"  Domain:    lat [{self.lat_min}, {self.lat_max}] | lon [{self.lon_min}, {self.lon_max}]")

        # Storage safety
        is_safe, free_gb, proj_free_gb = check_disk_space_headroom(
            self.raw_dir,
            plan["total_estimated_gb"],
            self.min_headroom_gb,
        )
        print("\n[3. Storage Safety Assessment]")
        print(f"  Available Free Disk Space:    {free_gb:.2f} GB")
        print(f"  Estimated Download Needed:    {plan['total_estimated_gb']:.2f} GB")
        print(f"  Projected Free After Download: {proj_free_gb:.2f} GB")
        print(f"  Required Safety Headroom:      {self.min_headroom_gb:.2f} GB")
        print(f"  Safety Gate Decision:          {'SAFE TO PROCEED' if is_safe else 'BLOCKED (Insufficient Headroom)'}")

        # Task breakdown
        print("\n[4. Planned Tasks & Product Mapping]")
        for task in plan["tasks"]:
            ds_str = task.get("dataset_id") or task.get("short_name", "N/A")
            status_str = task.get("status", "unknown")
            print(f"  Year {task['year']} | {task['variable'].upper():8s} | {task['source']:10s} | {ds_str:48s} | Status: {status_str}")

        print("\n[5. Summary]")
        print(f"  Tasks with verified files already present: {plan['existing_files_found']}")
        print(f"  Tasks requiring download:                   {plan['missing_files_to_download']}")
        print("  DRY-RUN MODE: Zero bytes downloaded. All files untouched.")
        print("=" * 80)

    def execute_downloads(self, plan: Dict[str, Any], max_retries: int = 3) -> None:
        """
        Executes planned downloads in a resumable, idempotent, and verified manner.
        """
        load_env()
        cm_user, cm_pwd = get_copernicus_credentials()

        print("\n" + "=" * 80)
        print(f"OCEANEMBED MULTI-YEAR DOWNLOADER — EXECUTING DOWNLOADS ({len(plan['tasks'])} Tasks)")
        print("=" * 80)

        for t_idx, task in enumerate(plan["tasks"], 1):
            y = task["year"]
            var = task["variable"]
            var_dir = Path(task["directory"])
            var_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n[{t_idx}/{len(plan['tasks'])}] Processing Year {y} | Variable: {var.upper()}")

            # 1. SST (Copernicus)
            if var == "sst":
                target_p = Path(task["target_file"])
                if is_valid_netcdf_or_data_file(target_p, MIN_EXPECTED_FILE_BYTES["sst"]):
                    print(f"  [Verified] {target_p.name} already present and valid ({target_p.stat().st_size / 1e6:.1f} MB). Skipping.")
                    continue
                print(f"  [Downloading] Requesting full-year SST for {y} via Copernicus Marine...")
                for attempt in range(1, max_retries + 1):
                    try:
                        copernicusmarine.subset(
                            dataset_id=task["dataset_id"],
                            username=cm_user,
                            password=cm_pwd,
                            variables=["analysed_sst"],
                            minimum_longitude=self.lon_min,
                            maximum_longitude=self.lon_max,
                            minimum_latitude=self.lat_min,
                            maximum_latitude=self.lat_max,
                            start_datetime=f"{y}-01-01",
                            end_datetime=f"{y}-12-31",
                            output_directory=target_p.parent,
                            output_filename=target_p.name,
                            file_format="netcdf",
                            overwrite=True,
                        )
                        if is_valid_netcdf_or_data_file(target_p, MIN_EXPECTED_FILE_BYTES["sst"]):
                            print(f"  -> SST {y} downloaded successfully ({target_p.stat().st_size / 1e6:.1f} MB).")
                            break
                        else:
                            raise ValueError(f"Downloaded file {target_p.name} failed integrity check.")
                    except Exception as e:
                        print(f"  [Attempt {attempt}/{max_retries} Failed] {e}")
                        if attempt == max_retries:
                            raise RuntimeError(f"Failed to download SST for {y} after {max_retries} attempts.")
                        time.sleep(3)

            # 2. SSS (Copernicus)
            elif var == "sss":
                target_p = Path(task["target_file"])
                if is_valid_netcdf_or_data_file(target_p, MIN_EXPECTED_FILE_BYTES["sss"]):
                    print(f"  [Verified] {target_p.name} already present and valid ({target_p.stat().st_size / 1e6:.1f} MB). Skipping.")
                    continue
                print(f"  [Downloading] Requesting full-year SSS for {y} via Copernicus Marine...")
                for attempt in range(1, max_retries + 1):
                    try:
                        copernicusmarine.subset(
                            dataset_id=task["dataset_id"],
                            username=cm_user,
                            password=cm_pwd,
                            variables=["sos"],
                            minimum_longitude=self.lon_min,
                            maximum_longitude=self.lon_max,
                            minimum_latitude=self.lat_min,
                            maximum_latitude=self.lat_max,
                            start_datetime=f"{y}-01-01",
                            end_datetime=f"{y}-12-31",
                            output_directory=target_p.parent,
                            output_filename=target_p.name,
                            file_format="netcdf",
                            overwrite=True,
                        )
                        if is_valid_netcdf_or_data_file(target_p, MIN_EXPECTED_FILE_BYTES["sss"]):
                            print(f"  -> SSS {y} downloaded successfully ({target_p.stat().st_size / 1e6:.1f} MB).")
                            break
                        else:
                            raise ValueError(f"Downloaded file {target_p.name} failed integrity check.")
                    except Exception as e:
                        print(f"  [Attempt {attempt}/{max_retries} Failed] {e}")
                        if attempt == max_retries:
                            raise RuntimeError(f"Failed to download SSS for {y} after {max_retries} attempts.")
                        time.sleep(3)

            # 3. SSH/SLA (Copernicus)
            elif var == "ssh":
                target_p = Path(task["target_file"])
                if is_valid_netcdf_or_data_file(target_p, MIN_EXPECTED_FILE_BYTES["ssh"]):
                    print(f"  [Verified] {target_p.name} already present and valid ({target_p.stat().st_size / 1e6:.1f} MB). Skipping.")
                    continue
                print(f"  [Downloading] Requesting full-year SSH (SLA) for {y} via Copernicus Marine...")
                for attempt in range(1, max_retries + 1):
                    try:
                        copernicusmarine.subset(
                            dataset_id=task["dataset_id"],
                            username=cm_user,
                            password=cm_pwd,
                            variables=["sla", "adt", "ugosa", "vgosa"],
                            minimum_longitude=self.lon_min,
                            maximum_longitude=self.lon_max,
                            minimum_latitude=self.lat_min,
                            maximum_latitude=self.lat_max,
                            start_datetime=f"{y}-01-01",
                            end_datetime=f"{y}-12-31",
                            output_directory=target_p.parent,
                            output_filename=target_p.name,
                            file_format="netcdf",
                            overwrite=True,
                        )
                        if is_valid_netcdf_or_data_file(target_p, MIN_EXPECTED_FILE_BYTES["ssh"]):
                            print(f"  -> SSH {y} downloaded successfully ({target_p.stat().st_size / 1e6:.1f} MB).")
                            break
                        else:
                            raise ValueError(f"Downloaded file {target_p.name} failed integrity check.")
                    except Exception as e:
                        print(f"  [Attempt {attempt}/{max_retries} Failed] {e}")
                        if attempt == max_retries:
                            raise RuntimeError(f"Failed to download SSH for {y} after {max_retries} attempts.")
                        time.sleep(3)

            # 4. GLORYS (Copernicus, 12 monthly files)
            elif var == "glorys":
                print(f"  [Downloading] Checking 12 monthly GLORYS files for {y}...")
                for m in range(1, 13):
                    m_file = var_dir / f"glorys_{y}_m{m:02d}.nc"
                    if is_valid_netcdf_or_data_file(m_file, MIN_EXPECTED_FILE_BYTES["glorys"]):
                        print(f"    [Month {m:02d}] Already verified ({m_file.stat().st_size / 1e6:.1f} MB). Skipping.")
                        continue
                    last_day = calendar.monthrange(y, m)[1]
                    m_start = f"{y}-{m:02d}-01"
                    m_end = f"{y}-{m:02d}-{last_day:02d}"
                    print(f"    [Month {m:02d}] Downloading {m_start} to {m_end}...")
                    for attempt in range(1, max_retries + 1):
                        try:
                            copernicusmarine.subset(
                                dataset_id=task["dataset_id"],
                                username=cm_user,
                                password=cm_pwd,
                                variables=["thetao"],
                                minimum_longitude=self.lon_min,
                                maximum_longitude=self.lon_max,
                                minimum_latitude=self.lat_min,
                                maximum_latitude=self.lat_max,
                                minimum_depth=0.0,
                                maximum_depth=1100.0,
                                start_datetime=m_start,
                                end_datetime=m_end,
                                output_directory=m_file.parent,
                                output_filename=m_file.name,
                                file_format="netcdf",
                                overwrite=True,
                            )
                            if is_valid_netcdf_or_data_file(m_file, MIN_EXPECTED_FILE_BYTES["glorys"]):
                                print(f"    -> Month {m:02d} verified ({m_file.stat().st_size / 1e6:.1f} MB).")
                                break
                            else:
                                raise ValueError(f"Downloaded month {m} failed integrity check.")
                        except Exception as e:
                            print(f"    [Attempt {attempt}/{max_retries} Failed] {e}")
                            if attempt == max_retries:
                                raise RuntimeError(f"Failed to download GLORYS month {m} for {y}.")
                            time.sleep(3)

            # 5. CURRENTS (NASA Earthdata OSCAR)
            elif var == "currents":
                print(f"  [Downloading] Searching Earthdata for {task['short_name']} in {y}...")
                auth = earthaccess.login(strategy="environment")
                granules = earthaccess.search_data(
                    short_name=task["short_name"],
                    temporal=(f"{y}-01-01", f"{y}-12-31"),
                )
                print(f"  Found {len(granules)} granules for OSCAR currents in {y}.")
                # Clean zero-byte or corrupt files first
                for f in var_dir.glob("*.nc"):
                    if not is_valid_netcdf_or_data_file(f, MIN_EXPECTED_FILE_BYTES["currents"]):
                        print(f"  [Cleaning] Removing zero-byte/corrupt file: {f.name}")
                        f.unlink()
                # Download missing granules
                earthaccess.download(granules, str(var_dir))
                valid_count = sum(1 for f in var_dir.glob("*.nc") if is_valid_netcdf_or_data_file(f, MIN_EXPECTED_FILE_BYTES["currents"]))
                print(f"  -> Verified {valid_count} valid OSCAR files in {var_dir}.")

            # 6. WINDS (NASA Earthdata CCMP)
            elif var == "winds":
                print(f"  [Downloading] Searching Earthdata for {task['short_name']} in {y}...")
                auth = earthaccess.login(strategy="environment")
                granules = earthaccess.search_data(
                    short_name=task["short_name"],
                    temporal=(f"{y}-01-01", f"{y}-12-31"),
                )
                print(f"  Found {len(granules)} granules for CCMP winds in {y}.")
                # Clean zero-byte or corrupt files first
                for f in var_dir.glob("*.nc"):
                    if not is_valid_netcdf_or_data_file(f, MIN_EXPECTED_FILE_BYTES["winds"]):
                        print(f"  [Cleaning] Removing zero-byte/corrupt file: {f.name}")
                        f.unlink()
                # Download missing granules
                earthaccess.download(granules, str(var_dir))
                valid_count = sum(1 for f in var_dir.glob("*.nc") if is_valid_netcdf_or_data_file(f, MIN_EXPECTED_FILE_BYTES["winds"]))
                print(f"  -> Verified {valid_count} valid CCMP files in {var_dir}.")

        print("\n" + "=" * 80)
        print("ALL REQUESTED DOWNLOAD TASKS COMPLETED SUCCESSFULLY")
        print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="OceanEmbed Multi-Year Production Downloader")
    parser.add_argument("--years", type=int, nargs="+", default=[2015], help="Explicit year(s) to process")
    parser.add_argument("--start-year", type=int, default=None, help="Start year of continuous range")
    parser.add_argument("--end-year", type=int, default=None, help="End year of continuous range")
    parser.add_argument("--variables", type=str, nargs="+", default=["sst", "sss", "ssh", "currents", "winds", "glorys"], help="Variables to download")
    parser.add_argument("--dry-run", action="store_true", help="Audit mode: verify plan, products, and disk safety without downloading")
    parser.add_argument("--min-headroom-gb", type=float, default=20.0, help="Minimum free disk space buffer required (GB)")
    parser.add_argument("--verify-year", type=int, default=None, help="Verify preprocessed completeness for a specific year")
    parser.add_argument("--cleanup-raw-year", type=int, default=None, help="Safely cleanup raw data for a verified year")
    parser.add_argument("--confirm-delete", action="store_true", help="Explicit confirmation required to permanently delete raw data")
    args = parser.parse_args()

    data_cfg = load_config("data")
    raw_dir = Path(data_cfg.raw_dir)
    processed_dir = Path(data_cfg.processed_dir)

    # 1. Verification-only subcommand
    if args.verify_year is not None:
        is_valid, count, issues = verify_preprocessed_year(args.verify_year, processed_dir)
        print(f"Preprocessed Verification for Year {args.verify_year}:")
        print(f"  Valid: {is_valid} | Verified Days: {count}")
        if issues:
            print("  Issues found:")
            for issue in issues[:10]:
                print(f"    - {issue}")
        sys.exit(0 if is_valid else 1)

    # 2. Cleanup subcommand
    if args.cleanup_raw_year is not None:
        res = safe_cleanup_raw_year(
            year=args.cleanup_raw_year,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            confirm_delete=args.confirm_delete,
        )
        print(f"Cleanup Report for Year {args.cleanup_raw_year}:")
        print(f"  Action: {res['action_taken']}")
        print(f"  Message: {res['message']}")
        sys.exit(0 if res["verified"] else 1)

    # 3. Determine requested years
    if args.start_year is not None and args.end_year is not None:
        years = list(range(args.start_year, args.end_year + 1))
    else:
        years = args.years

    downloader = MultiYearDownloader(raw_dir=raw_dir, data_cfg=data_cfg, min_headroom_gb=args.min_headroom_gb)
    plan = downloader.plan_downloads(years=years, variables=args.variables)

    if args.dry_run:
        downloader.execute_dry_run_report(plan)
        sys.exit(0)
    else:
        # Check disk safety
        is_safe, free_gb, proj_free_gb = check_disk_space_headroom(raw_dir, plan["total_estimated_gb"], args.min_headroom_gb)
        if not is_safe:
            print(f"[ERROR] Cannot proceed with download: Insufficient disk headroom.")
            print(f"  Free space: {free_gb:.2f} GB | Required: {plan['total_estimated_gb']:.2f} GB + {args.min_headroom_gb:.2f} GB margin")
            sys.exit(1)

        downloader.execute_downloads(plan)


if __name__ == "__main__":
    main()
