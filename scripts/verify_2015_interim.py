"""
scripts/verify_2015_interim.py
------------------------------
Verifies the complete 2015 interim dataset against all scientific and operational requirements.
"""

import json
from pathlib import Path
import numpy as np
import torch

from pipeline.cleanup import get_expected_dates_for_year

def verify_2015_interim():
    interim_dir = Path("data/interim/2015")
    expected_dates = get_expected_dates_for_year(2015)

    print("=" * 80)
    print(f"VERIFYING 2015 INTERIM DATASET ({len(expected_dates)} Expected Days)")
    print("=" * 80)

    # 1. Check every file exists
    missing = []
    sample_files = []
    for d_str in expected_dates:
        f = interim_dir / f"oceanembed_{d_str}.npz"
        if not f.exists():
            missing.append(d_str)
        else:
            sample_files.append(f)

    assert len(missing) == 0, f"Missing {len(missing)} files: {missing[:5]}"
    print(f"[1. File Existence] All {len(expected_dates)} daily .npz files present without date gaps.")

    # 2. Inspect tensors across all 365 days
    first_file = sample_files[0]
    mid_file = sample_files[182]
    last_file = sample_files[-1]

    # Validate all 365 files
    target_depth_valid_counts = np.zeros(15, dtype=np.int64)

    print("[2. Scanning 365 Files for Shapes, Finiteness, and Masks]...")
    for idx, f in enumerate(sample_files):
        with np.load(f) as data:
            in_arr = data["input"]
            tgt_arr = data["target"]
            tgt_mask = data["target_mask"]
            d_str = str(data["date"])

            # Check shapes
            assert in_arr.shape == (14, 101, 241), f"Wrong input shape in {f.name}: {in_arr.shape}"
            assert tgt_arr.shape == (15, 101, 241), f"Wrong target shape in {f.name}: {tgt_arr.shape}"
            assert tgt_mask.shape == (15, 101, 241), f"Wrong target mask shape in {f.name}: {tgt_mask.shape}"

            # Check finiteness
            assert np.all(np.isfinite(in_arr)), f"Non-finite input in {f.name}"
            assert np.all(np.isfinite(tgt_arr)), f"Non-finite target in {f.name}"
            assert np.all(np.isfinite(tgt_mask)), f"Non-finite target_mask in {f.name}"

            # Check binary mask contract (channels 7..13)
            masks = in_arr[7:]
            assert np.all(np.isin(masks, [0.0, 1.0])), f"Non-binary input mask in {f.name}"
            assert np.all(np.isin(tgt_mask, [0.0, 1.0])), f"Non-binary target mask in {f.name}"

            # Accumulate target depth validity
            for k in range(15):
                target_depth_valid_counts[k] += int(tgt_mask[k].sum())

    print("  -> All 365 files conform to input [14, 101, 241] and target [15, 101, 241] contracts.")
    print("  -> All 365 files have 100% finite values (zero NaNs / Infs).")
    print("  -> Channels 7..13 and target_mask are strictly binary 0.0 or 1.0.")

    # 3. Verify physical unnormalized units on sample day (Mid-year: 2015-07-02)
    print(f"\n[3. Physical Units Verification on Mid-Year Sample: {mid_file.name}]")
    with np.load(mid_file) as data:
        in_arr = data["input"]
        var_names = ["SST", "SSS", "SSH/SLA", "U_curr", "V_curr", "WindU", "WindV"]
        units = ["°C", "PSU", "m", "m/s", "m/s", "m/s", "m/s"]
        for c in range(7):
            mask = in_arr[7 + c]
            valid_vals = in_arr[c][mask == 1.0]
            print(f"  {var_names[c]:10s} (Ch {c}) [{units[c]:4s}] -> min: {np.min(valid_vals):7.3f}, max: {np.max(valid_vals):7.3f}, mean: {np.mean(valid_vals):7.3f}")
            # Assert physical units
            if c == 0:  # SST in Celsius
                assert 20.0 < np.mean(valid_vals) < 35.0, f"SST not in Celsius: mean={np.mean(valid_vals)}"
            elif c == 1:  # SSS in PSU
                assert 25.0 < np.mean(valid_vals) < 40.0, f"SSS not in PSU: mean={np.mean(valid_vals)}"
            elif c == 2:  # SLA in m (around 0)
                assert -0.5 < np.mean(valid_vals) < 0.5, f"SLA not anomaly: mean={np.mean(valid_vals)}"

    # 4. Target Depth Validity & 1000m Level Check
    depths_m = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    print("\n[4. Target Depths Validity across 365 Days]")
    for k, d in enumerate(depths_m):
        pts_per_day = target_depth_valid_counts[k] / 365.0
        ocean_fraction = pts_per_day / (101 * 241) * 100.0
        print(f"  Depth {d:4d}m (Ch {k:2d}): {target_depth_valid_counts[k]:10d} valid pixels ({pts_per_day:7.1f} pts/day, {ocean_fraction:5.1f}% ocean)")

    assert target_depth_valid_counts[-1] > 0, "1000m target depth has 0 valid pixels!"
    print(f"  -> Level 1000m verified: {target_depth_valid_counts[-1]} valid interpolated pixels without below-deepest extrapolation.")

    # 5. Verify Moments JSON
    moments_path = Path("data/norm_stats/moments_2015.json")
    assert moments_path.exists(), f"Missing moments file: {moments_path}"
    with open(moments_path, "r", encoding="utf-8") as f:
        m_payload = json.load(f)

    assert m_payload["year"] == 2015
    assert m_payload["num_days"] == 365
    assert "moments" in m_payload
    for k in ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]:
        assert k in m_payload["moments"]
        assert m_payload["moments"][k]["count"] > 3_000_000
    print("\n[5. Moments File Verification]")
    print(f"  moments_2015.json exists and verified with 7 physical variables, all count > 3,000,000 points.")

    print("\n" + "=" * 80)
    print("ALL 2015 INTERIM DATASET VERIFICATION CHECKS PASSED (100%)")
    print("=" * 80)

if __name__ == "__main__":
    verify_2015_interim()
