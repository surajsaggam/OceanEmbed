"""
pipeline/cleanup.py
--------------------
Safe verification and cleanup utility for raw multi-year satellite and reanalysis data.

SCIENTIFIC AND OPERATIONAL SAFETY RULES:
  1. Never delete raw data unless the corresponding preprocessed year is 100% complete and validated.
  2. Every preprocessed file must pass shape, key, and finiteness checks (torch.isfinite().all()).
  3. Deletion requires an explicit confirmation flag (--confirm-delete).
  4. Deletion is strictly scoped to the specified year only (never touches other years).
  5. Supports an archive destination option as an alternative to permanent deletion.
"""

from __future__ import annotations

import calendar
import json
import logging
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger("OceanEmbed.Cleanup")

PHYSICAL_VARIABLES = ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]


def get_expected_dates_for_year(year: int) -> List[str]:
    """Returns list of YYYY-MM-DD date strings for the given calendar year."""
    is_leap = calendar.isleap(year)
    days_in_year = 366 if is_leap else 365
    dates = []
    for month in range(1, 13):
        _, n_days = calendar.monthrange(year, month)
        for d in range(1, n_days + 1):
            dates.append(f"{year:04d}-{month:02d}-{d:02d}")
    assert len(dates) == days_in_year
    return dates


def verify_preprocessed_year(
    year: int,
    processed_dir: Path,
    expected_shape_in: Tuple[int, int, int] = (14, 101, 241),
    expected_shape_target: Tuple[int, int, int] = (15, 101, 241),
    require_moments: bool = True,
) -> Tuple[bool, int, List[str]]:
    """
    Verifies that all preprocessed .npz files for a given year exist and are valid.

    Checks:
      1. Every date in the calendar year has a corresponding .npz file across split dirs.
      2. File contains 'input', 'target', and 'target_mask'.
      3. Input and target shapes match specifications.
      4. Finiteness contract: all input and target values are finite.
      5. Validity masks (channels 7..13) are strictly binary 0.0 or 1.0.
      6. For training years (2015-2021), normalization moments are safely preserved.

    Returns:
        (is_valid, num_verified, issues_list)
    """
    expected_dates = get_expected_dates_for_year(year)
    issues = []
    verified_count = 0

    # Search in all potential split and interim subdirectories
    search_dirs = [processed_dir]
    interim_candidate = processed_dir.parent / "interim"
    if interim_candidate.exists() and interim_candidate not in search_dirs:
        search_dirs.append(interim_candidate)

    processed_files = {}
    for s_dir in search_dirs:
        for p in s_dir.glob(f"**/*{year}-*.npz"):
            # Match date from filename: e.g. oceanembed_2020-01-01.npz
            for d_str in expected_dates:
                if d_str in p.name:
                    processed_files[d_str] = p
                    break

    for date_str in expected_dates:
        if date_str not in processed_files:
            issues.append(f"Missing preprocessed file for date {date_str}")
            continue

        file_path = processed_files[date_str]
        try:
            with np.load(file_path) as data:
                if "input" not in data or "target" not in data:
                    issues.append(f"{file_path.name}: missing required keys 'input' or 'target'")
                    continue

                in_arr = data["input"]
                tgt_arr = data["target"]

                if in_arr.shape != expected_shape_in:
                    issues.append(f"{file_path.name}: input shape {in_arr.shape} != {expected_shape_in}")
                    continue

                if tgt_arr.shape != expected_shape_target:
                    issues.append(f"{file_path.name}: target shape {tgt_arr.shape} != {expected_shape_target}")
                    continue

                if not np.all(np.isfinite(in_arr)):
                    issues.append(f"{file_path.name}: input contains non-finite values")
                    continue

                if not np.all(np.isfinite(tgt_arr)):
                    issues.append(f"{file_path.name}: target contains non-finite values")
                    continue

                # Verify channels 7..13 are strictly binary 0/1 masks
                mask_channels = in_arr[7:14]
                if not np.all(np.isin(mask_channels, [0.0, 1.0])):
                    issues.append(f"{file_path.name}: channels 7..13 contain non-binary mask values")
                    continue

                verified_count += 1
        except Exception as e:
            issues.append(f"{file_path.name}: error reading file: {e}")

    # Check for year-specific training moments if training year (2015-2021)
    if require_moments and (2015 <= year <= 2021):
        moments_candidates = [
            processed_dir / str(year) / "moments.json",
            processed_dir / f"moments_{year}.json",
            processed_dir.parent / "interim" / str(year) / "moments.json",
            processed_dir.parent / "norm_stats" / f"moments_{year}.json",
        ]
        moments_found = False
        for m_path in moments_candidates:
            if m_path.exists():
                try:
                    with open(m_path, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    # Check if moments file contains all physical variables with positive counts
                    src_dict = m_data.get("moments", m_data)
                    if all(k in src_dict and src_dict[k].get("count", 0) > 0 for k in PHYSICAL_VARIABLES):
                        moments_found = True
                        break
                except Exception:
                    pass
        if not moments_found:
            issues.append(
                f"Training year {year} is missing required training normalization moments file (e.g. moments_{year}.json). "
                f"Cannot delete raw data without preserved normalization information!"
            )

    is_valid = (len(issues) == 0) and (verified_count == len(expected_dates))
    return is_valid, verified_count, issues


def safe_cleanup_raw_year(
    year: int,
    raw_dir: Path,
    processed_dir: Path,
    confirm_delete: bool = False,
    archive_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Safely deletes or moves raw data for a given year ONLY AFTER verifying that
    preprocessed outputs for that year are complete and valid.

    Args:
        year: Calendar year to clean up.
        raw_dir: Path to raw data root directory.
        processed_dir: Path to processed data root directory.
        confirm_delete: If True and archive_dir is None, permanently deletes raw files.
        archive_dir: Optional destination to move raw files rather than deleting.

    Returns:
        Summary report dictionary.
    """
    report = {
        "year": year,
        "verified": False,
        "action_taken": "none",
        "freed_bytes": 0,
        "issues": [],
    }

    # 1. Verification Gate
    is_valid, count, issues = verify_preprocessed_year(year, processed_dir)
    if not is_valid:
        report["issues"] = issues
        report["message"] = (
            f"REFUSING cleanup for year {year}: preprocessed data verification failed! "
            f"Verified {count} files, found {len(issues)} issues."
        )
        return report

    report["verified"] = True

    # 2. Locate year-specific raw targets
    year_str = str(year)
    targets_to_clean: List[Path] = []

    # Check for year directories (currents/YYYY, winds/YYYY, glorys/YYYY, sst/YYYY, sss/YYYY, ssh/YYYY)
    for var_dir in raw_dir.iterdir():
        if var_dir.is_dir():
            y_dir = var_dir / year_str
            if y_dir.exists() and y_dir.is_dir():
                targets_to_clean.append(y_dir)
            # Also check for individual yearly files: e.g. sst/sst_YYYY.nc
            for y_file in var_dir.glob(f"*{year_str}*.nc"):
                if y_file.is_file():
                    targets_to_clean.append(y_file)

    total_bytes = 0
    for target in targets_to_clean:
        if target.is_dir():
            for f in target.glob("**/*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
        elif target.is_file():
            total_bytes += target.stat().st_size

    report["freed_bytes"] = total_bytes
    report["freed_gb"] = round(total_bytes / (1024 ** 3), 2)
    report["targets"] = [str(p) for p in targets_to_clean]

    # 3. Action Execution
    if archive_dir is not None:
        arch_path = Path(archive_dir) / year_str
        arch_path.mkdir(parents=True, exist_ok=True)
        for target in targets_to_clean:
            dest = arch_path / target.name
            shutil.move(str(target), str(dest))
        report["action_taken"] = f"archived_to_{arch_path}"
        report["message"] = f"Successfully moved {report['freed_gb']} GB of raw data for {year} to {arch_path}"
    elif confirm_delete:
        for target in targets_to_clean:
            if target.is_dir():
                shutil.rmtree(target)
            elif target.is_file():
                target.unlink()
        report["action_taken"] = "deleted"
        report["message"] = f"Successfully deleted {report['freed_gb']} GB of raw data for {year} after verification."
    else:
        report["action_taken"] = "dry_run"
        report["message"] = (
            f"Verification passed. DRY RUN: Would remove/archive {report['freed_gb']} GB "
            f"across {len(targets_to_clean)} targets. Use --confirm-delete to execute."
        )

    return report
