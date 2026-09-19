"""
scripts/compare_2015_2018.py
----------------------------
Scientific data-quality comparison across 2015, 2016, 2017, and 2018 interim datasets and raw storage.
Does NOT interpret differences as model performance.
"""

import json
import shutil
from pathlib import Path
import numpy as np

def dir_size_gb(p: Path) -> float:
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 ** 3)

def run_comparison():
    print("=" * 80)
    print("OCEANEMBED DATA-QUALITY COMPARISON: 2015 vs 2016 vs 2017 vs 2018")
    print("=" * 80)

    # 1. Storage Comparison
    raw_sizes = {}
    interim_sizes = {}
    years = [2015, 2016, 2017, 2018]
    days_map = {2015: 365, 2016: 366, 2017: 365, 2018: 365}

    for y in years:
        raw_sizes[y] = sum(dir_size_gb(Path(f"data/raw/{v}/{y}")) for v in ["sst", "sss", "ssh", "glorys", "currents", "winds"])
        interim_sizes[y] = dir_size_gb(Path(f"data/interim/{y}"))

    # Free disk space
    total, used, free = shutil.disk_usage("data")
    free_gb = free / (1024 ** 3)

    print("\n[1. Storage Metrics]")
    for y in years:
        print(f"  Raw {y} Storage:     {raw_sizes[y]:.3f} GB ({days_map[y]} days)")
        print(f"  Interim {y} Storage: {interim_sizes[y]:.3f} GB ({days_map[y]} files)")
    print(f"  Current Free Disk:    {free_gb:.2f} GB")

    # 2. Moments Comparison (7 physical variables)
    moments_dict = {}
    for y in years:
        p = Path(f"data/norm_stats/moments_{y}.json")
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                moments_dict[y] = json.load(f)["moments"]

    if all(y in moments_dict for y in years):
        print("\n[2. Physical Variables Distributional Comparison (Moments)]")
        print(f"  {'Variable':10s} | {'2015 Mean':10s} | {'2016 Mean':10s} | {'2017 Mean':10s} | {'2018 Mean':10s} | {'2015 Std':10s} | {'2018 Std':10s}")
        print("  " + "-" * 85)
        for var in ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]:
            means = {}
            stds = {}
            for y in years:
                m = moments_dict[y][var]
                mean_v = m["sum"] / m["count"]
                var_v = max(0.0, (m["sq_sum"] / m["count"]) - (mean_v ** 2))
                means[y] = mean_v
                stds[y] = np.sqrt(var_v)
            print(f"  {var:10s} | {means[2015]:10.4f} | {means[2016]:10.4f} | {means[2017]:10.4f} | {means[2018]:10.4f} | {stds[2015]:10.4f} | {stds[2018]:10.4f}")

    # 3. Valid Counts per Physical Variable
    if all(y in moments_dict for y in years):
        print("\n[3. Valid Pixel Counts (Per Day Average)]")
        print(f"  {'Variable':10s} | {'2015 Pts/Day':14s} | {'2016 Pts/Day':14s} | {'2017 Pts/Day':14s} | {'2018 Pts/Day':14s}")
        print("  " + "-" * 75)
        for var in ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]:
            c15 = moments_dict[2015][var]["count"] / 365.0
            c16 = moments_dict[2016][var]["count"] / 366.0
            c17 = moments_dict[2017][var]["count"] / 365.0
            c18 = moments_dict[2018][var]["count"] / 365.0
            print(f"  {var:10s} | {c15:14.1f} | {c16:14.1f} | {c17:14.1f} | {c18:14.1f}")

    # 4. Target Depth Valid Counts
    print("\n[4. Target Depth Valid Pixels Comparison]")
    depths_m = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    def get_target_counts(year):
        counts = np.zeros(15, dtype=np.int64)
        for f in Path(f"data/interim/{year}").glob("*.npz"):
            with np.load(f) as data:
                counts += data["target_mask"].sum(axis=(1, 2)).astype(np.int64)
        return counts

    if all(Path(f"data/interim/{y}").exists() for y in years):
        t_counts = {y: get_target_counts(y) for y in years}
        print(f"  {'Depth':6s} | {'2015 Pts/Day':14s} | {'2016 Pts/Day':14s} | {'2017 Pts/Day':14s} | {'2018 Pts/Day':14s} | {'Coverage %':12s}")
        print("  " + "-" * 85)
        for k, d in enumerate(depths_m):
            avg15 = t_counts[2015][k] / 365.0
            avg16 = t_counts[2016][k] / 366.0
            avg17 = t_counts[2017][k] / 365.0
            avg18 = t_counts[2018][k] / 365.0
            pct = (avg18 / (101 * 241)) * 100.0
            print(f"  {d:4d}m  | {avg15:14.1f} | {avg16:14.1f} | {avg17:14.1f} | {avg18:14.1f} | {pct:11.2f}%")

    print("\n" + "=" * 80)
    print("DATA-QUALITY COMPARISON COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    run_comparison()
