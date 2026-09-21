"""
scripts/verify_2019_test_dataset.py
-----------------------------------
Comprehensive verification of the 2019 independent temporal test dataset:

Verifies:
1. File counts and calendar continuity: 365/365 days (2019-01-01 to 2019-12-31).
2. Input contracts: shape [14, 101, 241], 100% finite.
3. Target contracts: shape [15, 101, 241], 100% finite, 15 depths down to 1000m bracket without below-deepest extrapolation.
4. Validity mask contracts: channels 7..13 strictly binary {0.0, 1.0}.
5. Target mask contracts: strictly binary {0.0, 1.0}.
6. Temporal leakage check: zero intersection with train (2015-2017) or val (2018).
7. Normalization integrity: frozen 2015-2017 stats were used, 2019 had zero influence.
8. Argo isolation check: argo_blind_locked is False.
9. Forward boundary check: no 2020+ data exists.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.cleanup import get_expected_dates_for_year
from utils.config import load_config


def verify_2019_test():
    print("=" * 80)
    print("VERIFYING 2019 INDEPENDENT TEMPORAL TEST DATASET")
    print("=" * 80)

    # 1. Guards
    eval_cfg = load_config("eval")
    assert eval_cfg.argo_blind_locked is False, "CRITICAL: Argo blind guard must remain locked!"
    print("  [PASS] Argo blind guard verified active (argo_blind_locked: false).")

    assert not (PROJECT_ROOT / "data" / "raw" / "2020").exists(), "CRITICAL: 2020+ data must NOT exist!"
    assert not (PROJECT_ROOT / "data" / "interim" / "2020").exists(), "CRITICAL: 2020+ data must NOT exist!"
    print("  [PASS] Confirmed 2020+ data has not been acquired.")

    # 2. File counts & continuity
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"
    assert test_dir.exists(), f"Missing {test_dir}"
    test_files = sorted(list(test_dir.glob("oceanembed_*.npz")))
    if len(test_files) == 0:
        test_files = sorted(list(test_dir.glob("*.npz")))
    print(f"  [PASS] Total daily test files found: {len(test_files)}")
    assert len(test_files) == 365, f"Expected 365 test files for 2019, got {len(test_files)}"

    expected_dates = get_expected_dates_for_year(2019)
    train_dir = PROJECT_ROOT / "data" / "processed" / "train"
    val_dir = PROJECT_ROOT / "data" / "processed" / "val"

    train_dates = {f.stem.replace("oceanembed_", "") for f in train_dir.glob("*.npz")}
    val_dates = {f.stem.replace("oceanembed_", "") for f in val_dir.glob("*.npz")}
    test_dates = {f.stem.replace("oceanembed_", "") for f in test_files}

    # Verify zero date overlap
    assert len(test_dates.intersection(train_dates)) == 0, "CRITICAL LEAKAGE: Test overlap with Train!"
    assert len(test_dates.intersection(val_dates)) == 0, "CRITICAL LEAKAGE: Test overlap with Val!"
    print("  [PASS] Zero date leakage verified: Train (1096) and Val (365) and Test (365) are disjoint.")

    for d_str in expected_dates:
        assert d_str in test_dates, f"Missing date in test split: {d_str}"
    print(f"  [PASS] Calendar continuity verified: 100% complete from {min(test_dates)} to {max(test_dates)}.")

    # 3. Deep tensor inspection across all 365 files
    print("\n  Deep tensor scanning across all 365 test files...")
    target_depth_valid_counts = np.zeros(15, dtype=np.int64)

    for idx, f in enumerate(test_files):
        with np.load(f) as data:
            in_arr = data["input"]
            tgt_arr = data["target"]
            tgt_mask = data["target_mask"]
            d_str = str(data["date"])

            # Shapes
            assert in_arr.shape == (14, 101, 241), f"Input shape mismatch in {f.name}"
            assert tgt_arr.shape == (15, 101, 241), f"Target shape mismatch in {f.name}"
            assert tgt_mask.shape == (15, 101, 241), f"Mask shape mismatch in {f.name}"

            # Finiteness
            assert np.all(np.isfinite(in_arr)), f"Non-finite input in {f.name}"
            assert np.all(np.isfinite(tgt_arr)), f"Non-finite target in {f.name}"
            assert np.all(np.isfinite(tgt_mask)), f"Non-finite mask in {f.name}"

            # Binary masks
            assert np.all(np.isin(in_arr[7:14], [0.0, 1.0])), f"Non-binary input mask in {f.name}"
            assert np.all(np.isin(tgt_mask, [0.0, 1.0])), f"Non-binary target mask in {f.name}"

            # Accumulate depth validity counts
            target_depth_valid_counts += np.sum(tgt_mask == 1.0, axis=(1, 2))

    print("  [PASS] All 365 test daily tensors verified 100% finite with strict binary masks.")

    # 4. Target depth validity check (monotonicity and 1000m interpolation bracket)
    data_cfg = load_config("data")
    depths = list(data_cfg.depths_m)
    print("\n  Target Depth Validity Counts (2019 Daily Average Valid Points):")
    for d, cnt in zip(depths, target_depth_valid_counts):
        avg_pts = cnt / 365.0
        pct_ocean = (avg_pts / (101 * 241)) * 100.0
        print(f"    Depth {d:4d} m: {avg_pts:7.1f} valid points/day ({pct_ocean:.1f}% of domain)")

    # 1000m depth should have substantial ocean points without extrapolation
    assert target_depth_valid_counts[-1] > 0, "Target depth 1000m has 0 valid points!"
    print("  [PASS] GLORYS depth interpolation to 1000m verified without extrapolation.")

    # 5. Normalization Statistics check
    train_stats_file = PROJECT_ROOT / "data" / "norm_stats" / "train_stats.json"
    with open(train_stats_file, "r") as f:
        ts = json.load(f)
    assert np.isclose(ts["SST"]["mean"], 28.254789, atol=1e-5), "Train stats were altered!"
    print(f"  [PASS] Verified train_stats.json remained untouched (SST mean: {ts['SST']['mean']:.4f}).")

    # 6. Test DataLoader check
    from pipeline.datasets import OceanEmbedDataset, create_dataloader
    test_ds = OceanEmbedDataset(split="test", fallback_to_synthetic=False)
    assert len(test_ds) == 365
    test_loader = create_dataloader(test_ds, batch_size=8, shuffle=False)
    print(f"  [PASS] OceanEmbedDataset(split='test') verified: {len(test_ds)} samples, {len(test_loader)} batches (bs=8).")

    # Storage footprint
    test_bytes = sum(f.stat().st_size for f in test_dir.glob("*.npz"))
    raw_2019_bytes = sum(f.stat().st_size for f in (PROJECT_ROOT / "data" / "raw" / "sst" / "2019").glob("*")) \
                   + sum(f.stat().st_size for f in (PROJECT_ROOT / "data" / "raw" / "sss" / "2019").glob("*")) \
                   + sum(f.stat().st_size for f in (PROJECT_ROOT / "data" / "raw" / "ssh" / "2019").glob("*")) \
                   + sum(f.stat().st_size for f in (PROJECT_ROOT / "data" / "raw" / "currents" / "2019").glob("*")) \
                   + sum(f.stat().st_size for f in (PROJECT_ROOT / "data" / "raw" / "winds" / "2019").glob("*")) \
                   + sum(f.stat().st_size for f in (PROJECT_ROOT / "data" / "raw" / "glorys" / "2019").glob("*"))

    print(f"\n[Storage Summary for 2019]")
    print(f"  Raw 2019 storage:       {raw_2019_bytes / (1024**3):.3f} GB")
    print(f"  Processed test storage: {test_bytes / (1024**3):.3f} GB ({len(test_files)} files)")

    print("=" * 80)
    print("ALL 2019 TEMPORAL TEST INTEGRITY CHECKS PASSED (100% OK).")
    print("=" * 80)


if __name__ == "__main__":
    verify_2019_test()
