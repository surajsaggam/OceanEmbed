"""
scripts/compare_2015_2016.py
----------------------------
Scientific data-quality comparison between 2015 and 2016 interim datasets and raw storage.
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
    print("OCEANEMBED DATA-QUALITY COMPARISON: 2015 vs 2016")
    print("=" * 80)

    # 1. Storage Comparison
    raw_2015 = sum(dir_size_gb(Path(f"data/raw/{v}/2015")) for v in ["sst", "sss", "ssh", "glorys", "currents", "winds"])
    raw_2016 = sum(dir_size_gb(Path(f"data/raw/{v}/2016")) for v in ["sst", "sss", "ssh", "glorys", "currents", "winds"])
    interim_2015 = dir_size_gb(Path("data/interim/2015"))
    interim_2016 = dir_size_gb(Path("data/interim/2016"))

    # Free disk space
    total, used, free = shutil.disk_usage("data")
    free_gb = free / (1024 ** 3)

    print("\n[1. Storage Metrics]")
    print(f"  Raw 2015 Storage:     {raw_2015:.3f} GB (365 days)")
    print(f"  Raw 2016 Storage:     {raw_2016:.3f} GB (366 days)")
    print(f"  Interim 2015 Storage: {interim_2015:.3f} GB (365 files)")
    print(f"  Interim 2016 Storage: {interim_2016:.3f} GB (366 files)")
    print(f"  Current Free Disk:    {free_gb:.2f} GB")

    # 2. Moments Comparison (7 physical variables)
    m15_path = Path("data/norm_stats/moments_2015.json")
    m16_path = Path("data/norm_stats/moments_2016.json")

    if m15_path.exists() and m16_path.exists():
        with open(m15_path, "r", encoding="utf-8") as f:
            m15 = json.load(f)["moments"]
        with open(m16_path, "r", encoding="utf-8") as f:
            m16 = json.load(f)["moments"]

        print("\n[2. Physical Variables Distributional Comparison (Moments)]")
        print(f"  {'Variable':10s} | {'2015 Mean':10s} | {'2016 Mean':10s} | {'Diff Mean':10s} | {'2015 Std':10s} | {'2016 Std':10s} | {'Diff Std':10s}")
        print("  " + "-" * 75)
        for var in ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]:
            mean15 = m15[var]["sum"] / m15[var]["count"]
            mean16 = m16[var]["sum"] / m16[var]["count"]
            var15 = max(0.0, (m15[var]["sq_sum"] / m15[var]["count"]) - (mean15 ** 2))
            var16 = max(0.0, (m16[var]["sq_sum"] / m16[var]["count"]) - (mean16 ** 2))
            std15 = np.sqrt(var15)
            std16 = np.sqrt(var16)
            diff_m = mean16 - mean15
            diff_s = std16 - std15
            print(f"  {var:10s} | {mean15:10.4f} | {mean16:10.4f} | {diff_m:+10.4f} | {std15:10.4f} | {std16:10.4f} | {diff_s:+10.4f}")

    # 3. Valid Counts per Physical Variable
    vars_list = ["SST", "SSS", "SSH/SLA", "U_curr", "V_curr", "WindU", "WindV"]
    print("\n[3. Valid Pixel Counts (Per Day Average)]")
    print(f"  {'Variable':10s} | {'2015 Pts/Day':14s} | {'2016 Pts/Day':14s} | {'Ratio (16/15)':14s}")
    print("  " + "-" * 60)
    for var in ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]:
        count15 = m15[var]["count"] / 365.0
        count16 = m16[var]["count"] / 366.0
        ratio = count16 / count15 if count15 > 0 else 1.0
        print(f"  {var:10s} | {count15:14.1f} | {count16:14.1f} | {ratio:14.4f}")

    # 4. Target Depth Valid Counts
    print("\n[4. Target Depth Valid Pixels Comparison]")
    depths_m = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    # Quick scan of 2015 and 2016
    def get_target_counts(year, n_days):
        counts = np.zeros(15, dtype=np.int64)
        for f in Path(f"data/interim/{year}").glob("*.npz"):
            with np.load(f) as data:
                counts += data["target_mask"].sum(axis=(1, 2)).astype(np.int64)
        return counts

    if Path("data/interim/2016").exists():
        t15 = get_target_counts(2015, 365)
        t16 = get_target_counts(2016, 366)
        print(f"  {'Depth':6s} | {'2015 Pts/Day':14s} | {'2016 Pts/Day':14s} | {'Coverage %':12s}")
        print("  " + "-" * 55)
        for k, d in enumerate(depths_m):
            avg15 = t15[k] / 365.0
            avg16 = t16[k] / 366.0
            pct = (avg16 / (101 * 241)) * 100.0
            print(f"  {d:4d}m  | {avg15:14.1f} | {avg16:14.1f} | {pct:11.2f}%")

    print("\n" + "=" * 80)
    print("DATA-QUALITY COMPARISON COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    run_comparison()
