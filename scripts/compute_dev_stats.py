"""
scripts/compute_dev_stats.py
----------------------------
Computes and freezes global normalization statistics using sufficient statistics (moments).
Supports:
  1. 2015-2017 training split moments (zero-leakage training statistics)
  2. 2015-2018 complete development dataset moments
"""

import json
from pathlib import Path
from pipeline.normalize import combine_moments, compute_stats_from_moments, save_norm_stats

def main():
    norm_dir = Path("data/norm_stats")
    norm_dir.mkdir(parents=True, exist_ok=True)

    moments_by_year = {}
    for y in [2015, 2016, 2017, 2018]:
        p = norm_dir / f"moments_{y}.json"
        with open(p, "r", encoding="utf-8") as f:
            moments_by_year[y] = json.load(f)["moments"]

    # 1. 2015-2017 Training Split
    m_train = combine_moments([moments_by_year[2015], moments_by_year[2016], moments_by_year[2017]])
    stats_train = compute_stats_from_moments(m_train)

    # 2. 2015-2018 Complete Development Horizon
    m_all = combine_moments([moments_by_year[2015], moments_by_year[2016], moments_by_year[2017], moments_by_year[2018]])
    stats_all = compute_stats_from_moments(m_all)

    print("=" * 80)
    print("GLOBAL NORMALIZATION STATISTICS (SUFFICIENT STATISTICS)")
    print("=" * 80)

    print("\n[A. 2015-2017 Chronological Training Split Statistics]")
    for k, v in stats_train.items():
        print(f"  {k:10s} -> mean: {v['mean']:8.4f}, std: {v['std']:8.4f} (count: {m_train[k]['count']:,})")

    print("\n[B. 2015-2018 Complete Horizon Statistics]")
    for k, v in stats_all.items():
        print(f"  {k:10s} -> mean: {v['mean']:8.4f}, std: {v['std']:8.4f} (count: {m_all[k]['count']:,})")

    # Save frozen statistics
    # 1) train_stats.json -> training split 2015-2017 (used by DataLoader / models for strict zero-leakage training)
    save_norm_stats(stats_train, norm_dir / "train_stats_2015_2017.json")
    save_norm_stats(stats_train, norm_dir / "train_stats.json")

    # 2) dev_stats_2015_2018.json -> 2015-2018 combined
    save_norm_stats(stats_all, norm_dir / "dev_stats_2015_2018.json")

    # Also save combined moments metadata
    combined_payload = {
        "dataset": "OceanEmbed Phase-1 2015-2018 Development Dataset",
        "train_years": [2015, 2016, 2017],
        "val_years": [2018],
        "test_years": [],
        "note": "Initial Phase-1 development dataset. No independent final temporal test year exists yet (2019-2024 unacquired).",
        "train_stats": stats_train,
        "dev_stats_all": stats_all,
        "combined_train_moments": m_train,
        "combined_all_moments": m_all,
    }
    with open(norm_dir / "global_dev_metadata.json", "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, indent=2)

    print("\n[C. Frozen Artifacts Written]")
    print(f"  - {norm_dir / 'train_stats.json'} (2015-2017 training split, zero leakage)")
    print(f"  - {norm_dir / 'train_stats_2015_2017.json'}")
    print(f"  - {norm_dir / 'dev_stats_2015_2018.json'}")
    print(f"  - {norm_dir / 'global_dev_metadata.json'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
