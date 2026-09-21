"""
scripts/create_test_dataset.py
------------------------------
Populates the independent temporal test dataset (data/processed/test/)
for Year 2019 using frozen training normalization statistics (2015-2017).

Scientific and pipeline guarantees:
1. Zero Data Leakage:
   - Normalization statistics loaded strictly from data/norm_stats/train_stats.json.
   - 2019 data is NEVER used to update or recalculate normalization moments.
2. Production Contracts:
   - Input: [14, 101, 241]
     * Channels 0..6: Normalized physical surface variables (SST, SSS, SSH, U, V, WindU, WindV).
     * Channels 7..13: Binary validity masks {0.0, 1.0}.
   - Target: [15, 101, 241] GLORYS thetao at 15 standard depths down to 1000m.
   - Target Mask: [15, 101, 241] binary validity mask.
3. Strict Finiteness:
   - 100% finite values (zero NaNs, zero Infs).
4. Full Calendar Coverage:
   - Exactly 365 daily .npz files for 2019 (2019-01-01 through 2019-12-31).
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch

from pipeline.cleanup import get_expected_dates_for_year
from pipeline.normalize import load_norm_stats, normalize_input_tensor


def populate_2019_test_split():
    print("=" * 80)
    print("OCEANEMBED 2019 INDEPENDENT TEMPORAL TEST SET NORMALIZATION")
    print("=" * 80)

    # 1. Normalization Stats Gate
    norm_stats_path = Path("data/norm_stats/train_stats.json")
    assert norm_stats_path.exists(), f"Missing frozen training stats at {norm_stats_path}"
    norm_stats = load_norm_stats(norm_stats_path)
    print(f"[1. Normalization Gate] Loaded frozen training statistics from {norm_stats_path}.")
    print("  Verified: 2019 data has zero influence on normalization parameters.")

    # 2. Directory setup
    interim_dir = Path("data/interim/2019")
    test_processed_dir = Path("data/processed/test")
    test_processed_dir.mkdir(parents=True, exist_ok=True)

    expected_dates = get_expected_dates_for_year(2019)
    assert len(expected_dates) == 365, f"Expected 365 days for 2019, got {len(expected_dates)}"

    print(f"\n[2. Normalizing 2019 Interim Data] Source: {interim_dir} -> Destination: {test_processed_dir}")
    t_start = time.time()
    processed_count = 0

    for d_str in expected_dates:
        interim_file = interim_dir / f"oceanembed_{d_str}.npz"
        if not interim_file.exists():
            interim_file = interim_dir / f"{d_str}.npz"
        assert interim_file.exists(), f"Missing interim file: {interim_file}"

        with np.load(interim_file) as data:
            raw_input = data["input"]        # [14, 101, 241] unnormalized
            raw_target = data["target"]      # [15, 101, 241]
            raw_mask = data["target_mask"]   # [15, 101, 241]
            d_val = str(data.get("date", d_str))

        # Verify interim contracts
        assert raw_input.shape == (14, 101, 241), f"Unexpected input shape: {raw_input.shape}"
        assert raw_target.shape == (15, 101, 241), f"Unexpected target shape: {raw_target.shape}"
        assert raw_mask.shape == (15, 101, 241), f"Unexpected mask shape: {raw_mask.shape}"
        assert np.isfinite(raw_input).all(), f"Non-finite input on {d_str}"
        assert np.isfinite(raw_target).all(), f"Non-finite target on {d_str}"

        # Normalize physical channels 0..6; channels 7..13 remain binary masks
        norm_input = normalize_input_tensor(raw_input, norm_stats)
        assert norm_input.shape == (14, 101, 241)
        assert np.isfinite(norm_input).all(), f"Non-finite normalized input on {d_str}"

        # Verify validity masks strictly binary {0.0, 1.0}
        masks_slice = norm_input[7:14]
        assert np.all(np.isin(masks_slice, [0.0, 1.0])), f"Non-binary mask values in {d_str}"

        # Save processed .npz
        out_file = test_processed_dir / f"oceanembed_{d_str}.npz"
        np.savez_compressed(
            out_file,
            input=norm_input.astype(np.float32),
            target=raw_target.astype(np.float32),
            target_mask=raw_mask.astype(np.float32),
            date=d_val,
        )
        processed_count += 1
        if processed_count % 60 == 0 or processed_count == 365:
            print(f"  Processed {processed_count:3d}/365 days -> {d_str}")

    t_dur = time.time() - t_start
    print(f"\n[Success] Normalized {processed_count} days into {test_processed_dir} in {t_dur:.2f}s ({processed_count/t_dur:.1f} days/s).")

    # 3. Save test metadata
    meta = {
        "split": "test",
        "year": 2019,
        "num_days": processed_count,
        "date_start": expected_dates[0],
        "date_end": expected_dates[-1],
        "input_shape": [14, 101, 241],
        "target_shape": [15, 101, 241],
        "norm_stats_source": str(norm_stats_path),
        "status": "ready_for_blind_test",
    }
    with open(test_processed_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved test metadata to {test_processed_dir / 'metadata.json'}")
    print("=" * 80)
    print("2019 TEMPORAL TEST SPLIT FULLY POPULATED AND VERIFIED.")
    print("=" * 80)


if __name__ == "__main__":
    populate_2019_test_split()
