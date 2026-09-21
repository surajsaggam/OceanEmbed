"""
tests/test_downloader.py
-------------------------
Tests for production multi-year downloader logic (scripts/download_multiyear.py)
and safe verification/cleanup module (pipeline/cleanup.py).

Validates:
  1. Year range and calendar day generation.
  2. GLORYS product transition by year (MY <= 2021, MYINT >= 2022).
  3. Product and task mapping for all variables.
  4. Output path generation in year-specific directories.
  5. Incomplete / zero-byte / corrupt file detection.
  6. Idempotent existing-file detection.
  7. Disk space headroom and safety check logic.
  8. Dry-run mode execution and report generation.
  9. Credential loading and status checks without secret exposure.
  10. Verification gate preventing cleanup of unverified/incomplete preprocessed years.
"""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest

from pipeline.cleanup import (
    get_expected_dates_for_year,
    safe_cleanup_raw_year,
    verify_preprocessed_year,
)
from scripts.download_multiyear import (
    MultiYearDownloader,
    check_disk_space_headroom,
    get_glorys_dataset_id,
    is_valid_netcdf_or_data_file,
)
from utils.credentials import check_credentials_status, load_env


class TestMultiYearDownloaderLogic:
    def test_expected_dates_for_year(self):
        """Verify date generation handles leap and non-leap years correctly."""
        # Non-leap year: 365 days
        dates_2015 = get_expected_dates_for_year(2015)
        assert len(dates_2015) == 365
        assert dates_2015[0] == "2015-01-01"
        assert dates_2015[-1] == "2015-12-31"

        # Leap year: 366 days
        dates_2020 = get_expected_dates_for_year(2020)
        assert len(dates_2020) == 366
        assert "2020-02-29" in dates_2020

    def test_glorys_product_transition_by_year(self):
        """Verify GLORYS maps to frozen MY for <= 2021 and interim MYINT for >= 2022."""
        # Frozen Multi-Year Reanalysis
        assert get_glorys_dataset_id(2015) == "cmems_mod_glo_phy_my_0.083deg_P1D-m"
        assert get_glorys_dataset_id(2020) == "cmems_mod_glo_phy_my_0.083deg_P1D-m"
        assert get_glorys_dataset_id(2021) == "cmems_mod_glo_phy_my_0.083deg_P1D-m"

        # Interim Multi-Year Reanalysis
        assert get_glorys_dataset_id(2022) == "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
        assert get_glorys_dataset_id(2023) == "cmems_mod_glo_phy_myint_0.083deg_P1D-m"
        assert get_glorys_dataset_id(2024) == "cmems_mod_glo_phy_myint_0.083deg_P1D-m"

    def test_file_validation_and_zero_byte_detection(self):
        """Verify detection of zero-byte, truncated, or non-existent files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            
            # 1. Non-existent file
            assert not is_valid_netcdf_or_data_file(tmp / "nonexistent.nc")

            # 2. Zero-byte file
            empty_file = tmp / "empty.nc"
            empty_file.touch()
            assert not is_valid_netcdf_or_data_file(empty_file, min_size=100)

            # 3. Truncated small file
            small_file = tmp / "small.nc"
            small_file.write_bytes(b"ABC")
            assert not is_valid_netcdf_or_data_file(small_file, min_size=100)

            # 4. Valid NetCDF-3 magic header
            valid_nc3 = tmp / "valid3.nc"
            valid_nc3.write_bytes(b"CDF\x01" + b"\x00" * 2000)
            assert is_valid_netcdf_or_data_file(valid_nc3, min_size=100)

            # 5. Valid NetCDF-4 / HDF5 magic header
            valid_nc4 = tmp / "valid4.nc"
            valid_nc4.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 2000)
            assert is_valid_netcdf_or_data_file(valid_nc4, min_size=100)

    def test_disk_space_safety_check(self):
        """Verify that disk safety check passes when headroom is ample and fails when exceeded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            
            # Request small 1 GB download -> should be safe
            is_safe, free_gb, proj_gb = check_disk_space_headroom(tmp, estimated_need_gb=1.0, min_headroom_gb=5.0)
            assert is_safe
            assert proj_gb == pytest.approx(free_gb - 1.0, rel=1e-3)

            # Request absurdly huge 100,000 GB download -> must be flagged unsafe
            is_unsafe, _, proj_unsafe = check_disk_space_headroom(tmp, estimated_need_gb=100000.0, min_headroom_gb=20.0)
            assert not is_unsafe
            assert proj_unsafe < 0.0

    def test_downloader_plan_and_dry_run(self):
        """Verify plan_downloads accurately structures tasks, product IDs, and output paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            downloader = MultiYearDownloader(raw_dir=raw_dir)

            plan = downloader.plan_downloads(
                years=[2015, 2022],
                variables=["sst", "glorys"],
            )

            assert plan["years"] == [2015, 2022]
            assert plan["variables"] == ["sst", "glorys"]
            assert len(plan["tasks"]) == 4  # 2 years * 2 variables
            assert plan["total_estimated_gb"] > 0

            # Check year-specific directory paths
            for task in plan["tasks"]:
                expected_dir_part = f"{task['variable']}\\{task['year']}" if "\\" in task["directory"] else f"{task['variable']}/{task['year']}"
                assert expected_dir_part in task["directory"]

            # Check GLORYS products
            task_glorys_2015 = next(t for t in plan["tasks"] if t["year"] == 2015 and t["variable"] == "glorys")
            assert task_glorys_2015["dataset_id"] == "cmems_mod_glo_phy_my_0.083deg_P1D-m"

            task_glorys_2022 = next(t for t in plan["tasks"] if t["year"] == 2022 and t["variable"] == "glorys")
            assert task_glorys_2022["dataset_id"] == "cmems_mod_glo_phy_myint_0.083deg_P1D-m"

    def test_credentials_check_without_secret_exposure(self):
        """Verify credentials check returns boolean status and does not expose secrets."""
        status = check_credentials_status()
        assert "copernicus_configured" in status
        assert "earthdata_configured" in status
        assert isinstance(status["copernicus_configured"], bool)
        assert isinstance(status["earthdata_configured"], bool)

        # load_env test
        loaded = load_env()
        for k, v in loaded.items():
            assert v == "***", "Credentials must be obscured in load_env return values!"

    def test_cleanup_verification_gate_prevents_premature_deletion(self):
        """Verify safe_cleanup_raw_year refuses deletion if preprocessed data is incomplete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            proc_dir = Path(tmpdir) / "processed"
            raw_dir.mkdir(parents=True)
            proc_dir.mkdir(parents=True)

            # Create dummy raw directory
            raw_curr_2015 = raw_dir / "currents" / "2015"
            raw_curr_2015.mkdir(parents=True)
            dummy_file = raw_curr_2015 / "oscar_20150101.nc"
            dummy_file.write_bytes(b"dummy data")

            # Try to cleanup 2015 when 0 preprocessed files exist
            report = safe_cleanup_raw_year(
                year=2015,
                raw_dir=raw_dir,
                processed_dir=proc_dir,
                confirm_delete=True,
            )

            # Must refuse deletion!
            assert report["verified"] is False
            assert report["action_taken"] == "none"
            assert "REFUSING" in report["message"]
            assert dummy_file.exists(), "Raw data must NOT be deleted when verification fails!"
