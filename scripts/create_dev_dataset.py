"""
scripts/create_dev_dataset.py
-----------------------------
Generates the Phase-1 Development Dataset from interim 2015–2018 datasets.

Scientific & Pipeline Rules:
  - Chronological Development Split:
      * Train: 2015–2017 (1,096 days)
      * Val:   2018 (365 days)
      * Test:  None (explicitly documented: no independent final temporal test year exists yet)
  - Zero-Leakage Normalization:
      * Frozen training statistics (2015–2017) are applied to physical channels 0–6.
      * Validity mask channels 7–13 remain strictly binary {0.0, 1.0}.
      * Target (15 standard depths) and target_mask remain intact and finite.
  - Strict Finiteness: All processed tensors are 100% finite.
  - Measures disk space before and after normalization.
"""

from pathlib import Path
import shutil
import time
import json
import numpy as np
import torch

from pipeline.cleanup import get_expected_dates_for_year
from pipeline.normalize import load_norm_stats, normalize_input_tensor

def get_dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

def get_disk_free_bytes(path: Path = Path("data")) -> int:
    total, used, free = shutil.disk_usage(path if path.exists() else ".")
    return free

def main():
    print("=" * 80)
    print("OCEANEMBED PHASE-1 DEVELOPMENT DATASET GENERATOR (2015–2018)")
    print("=" * 80)

    # 1. Pre-normalization disk audit
    raw_dir = Path("data/raw")
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")
    norm_stats_file = Path("data/norm_stats/train_stats.json")

    bytes_before = {
        "raw": get_dir_size_bytes(raw_dir),
        "interim": get_dir_size_bytes(interim_dir),
        "processed": get_dir_size_bytes(processed_dir),
        "free": get_disk_free_bytes(),
    }

    print("\n[1. Disk Usage BEFORE Normalization]")
    print(f"  Raw Storage:       {bytes_before['raw'] / (1024**3):.3f} GB")
    print(f"  Interim Storage:   {bytes_before['interim'] / (1024**3):.3f} GB")
    print(f"  Processed Storage: {bytes_before['processed'] / (1024**3):.3f} GB")
    print(f"  Free Disk Space:   {bytes_before['free'] / (1024**3):.2f} GB")

    # 2. Verify frozen training normalization statistics exist
    assert norm_stats_file.exists(), f"Missing frozen stats file: {norm_stats_file}"
    stats = load_norm_stats(norm_stats_file)
    print("\n[2. Frozen Normalization Statistics Loaded (2015–2017 Training Split)]")
    for k, v in stats.items():
        print(f"  {k:10s} -> mean: {v['mean']:8.4f}, std: {v['std']:8.4f}")

    # 3. Clean and prepare processed directories
    train_dir = processed_dir / "train"
    val_dir = processed_dir / "val"
    test_dir = processed_dir / "test"

    for d in [train_dir, val_dir, test_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # 4. Process Training Split (2015, 2016, 2017)
    print("\n[3. Normalizing Training Split: 2015–2017 (1,096 Days)]...")
    t0 = time.time()
    train_days_count = 0

    for year in [2015, 2016, 2017]:
        dates = get_expected_dates_for_year(year)
        y_dir = interim_dir / str(year)
        for d_str in dates:
            src_file = y_dir / f"oceanembed_{d_str}.npz"
            assert src_file.exists(), f"Missing interim file: {src_file}"

            with np.load(src_file) as data:
                in_raw = data["input"]
                tgt_arr = data["target"]
                tgt_mask = data["target_mask"]
                d_val = str(data["date"])

            # Verify unnormalized
            assert in_raw.shape == (14, 101, 241)
            # Normalize channels 0..6, keeping channels 7..13 binary
            in_norm = normalize_input_tensor(in_raw, stats)

            # Contract verification
            assert in_norm.shape == (14, 101, 241)
            assert tgt_arr.shape == (15, 101, 241)
            assert tgt_mask.shape == (15, 101, 241)
            assert np.all(np.isfinite(in_norm))
            assert np.all(np.isfinite(tgt_arr))
            assert np.all(np.isfinite(tgt_mask))
            assert np.all(np.isin(in_norm[7:14], [0.0, 1.0]))
            assert np.all(np.isin(tgt_mask, [0.0, 1.0]))

            # Save normalized tensor
            dst_file = train_dir / f"oceanembed_{d_str}.npz"
            np.savez_compressed(
                dst_file,
                input=in_norm.astype(np.float32),
                target=tgt_arr.astype(np.float32),
                target_mask=tgt_mask.astype(np.float32),
                date=d_val,
            )
            train_days_count += 1

    t_train = time.time() - t0
    print(f"  -> Generated {train_days_count} normalized training files in {t_train:.1f}s ({train_days_count/t_train:.1f} days/s).")

    # 5. Process Validation Split (2018)
    print("\n[4. Normalizing Validation Split: 2018 (365 Days)]...")
    t0_val = time.time()
    val_days_count = 0
    y_dir_2018 = interim_dir / "2018"
    dates_2018 = get_expected_dates_for_year(2018)

    for d_str in dates_2018:
        src_file = y_dir_2018 / f"oceanembed_{d_str}.npz"
        assert src_file.exists(), f"Missing interim file: {src_file}"

        with np.load(src_file) as data:
            in_raw = data["input"]
            tgt_arr = data["target"]
            tgt_mask = data["target_mask"]
            d_val = str(data["date"])

        # Normalize with SAME training stats (zero leakage)
        in_norm = normalize_input_tensor(in_raw, stats)

        # Contract verification
        assert in_norm.shape == (14, 101, 241)
        assert tgt_arr.shape == (15, 101, 241)
        assert tgt_mask.shape == (15, 101, 241)
        assert np.all(np.isfinite(in_norm))
        assert np.all(np.isfinite(tgt_arr))
        assert np.all(np.isfinite(tgt_mask))
        assert np.all(np.isin(in_norm[7:14], [0.0, 1.0]))
        assert np.all(np.isin(tgt_mask, [0.0, 1.0]))

        # Save normalized tensor
        dst_file = val_dir / f"oceanembed_{d_str}.npz"
        np.savez_compressed(
            dst_file,
            input=in_norm.astype(np.float32),
            target=tgt_arr.astype(np.float32),
            target_mask=tgt_mask.astype(np.float32),
            date=d_val,
        )
        val_days_count += 1

    t_val = time.time() - t0_val
    print(f"  -> Generated {val_days_count} normalized validation files in {t_val:.1f}s ({val_days_count/t_val:.1f} days/s).")

    # 6. Test Split Status
    print("\n[5. Test Split Status]")
    test_files_count = len(list(test_dir.glob("*.npz")))
    print(f"  Test split directory: {test_dir} ({test_files_count} files)")
    print("  EXPLICIT SCIENTIFIC NOTE: No independent final temporal test year exists yet.")
    print("  Per OceanEmbed technical plan, temporal test horizon is locked to 2023–2024 and remains unacquired.")

    # 7. Post-normalization disk audit
    bytes_after = {
        "raw": get_dir_size_bytes(raw_dir),
        "interim": get_dir_size_bytes(interim_dir),
        "processed": get_dir_size_bytes(processed_dir),
        "free": get_disk_free_bytes(),
    }

    print("\n[6. Disk Usage AFTER Normalization]")
    print(f"  Raw Storage:       {bytes_after['raw'] / (1024**3):.3f} GB (preserved untouched)")
    print(f"  Interim Storage:   {bytes_after['interim'] / (1024**3):.3f} GB (preserved untouched)")
    print(f"  Processed Storage: {bytes_after['processed'] / (1024**3):.3f} GB (new {train_days_count + val_days_count} files)")
    print(f"  Free Disk Space:   {bytes_after['free'] / (1024**3):.2f} GB")
    print(f"  Processed Net Growth: +{(bytes_after['processed'] - bytes_before['processed']) / (1024**3):.3f} GB")

    # 8. Save development split metadata
    split_meta = {
        "dataset_name": "OceanEmbed Phase-1 2015-2018 Development Dataset",
        "total_days": train_days_count + val_days_count,
        "train_years": [2015, 2016, 2017],
        "train_days": train_days_count,
        "train_date_range": ["2015-01-01", "2017-12-31"],
        "val_years": [2018],
        "val_days": val_days_count,
        "val_date_range": ["2018-01-01", "2018-12-31"],
        "test_status": "Unpopulated. No independent final temporal test year exists yet (2023-2024 unacquired).",
        "norm_stats_file": str(norm_stats_file),
        "frozen_training_stats": stats,
        "tensor_contract": {
            "input_shape": [14, 101, 241],
            "target_shape": [15, 101, 241],
            "target_mask_shape": [15, 101, 241],
            "physical_channels": "0..6 (z-score normalized with 2015-2017 training stats)",
            "mask_channels": "7..13 (strictly binary 0/1)",
            "finiteness": "100% finite across all tensors",
        },
    }
    with open(processed_dir / "dev_split_metadata.json", "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)

    print("\n" + "=" * 80)
    print("PHASE-1 DEVELOPMENT DATASET GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
