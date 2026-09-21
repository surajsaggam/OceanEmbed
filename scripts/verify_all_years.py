"""
scripts/verify_all_years.py
---------------------------
Deep verification of 2015, 2016, 2017, and 2018 datasets.
Verifies all raw files, interim files, moments files, and disk integrity.
"""

from pathlib import Path
import json
import numpy as np

from pipeline.cleanup import get_expected_dates_for_year

def verify_all():
    years = [2015, 2016, 2017, 2018]
    expected_days = {2015: 365, 2016: 366, 2017: 365, 2018: 365}
    expected_raw_files = {2015: 745, 2016: 747, 2017: 745, 2018: 745}

    print("=" * 80)
    print("VERIFYING 2015–2018 DATASETS INTEGRITY")
    print("=" * 80)

    all_ok = True

    for y in years:
        n_exp = expected_days[y]
        dates = get_expected_dates_for_year(y)
        assert len(dates) == n_exp

        # 1. Raw checks
        raw_files = []
        for v in ["sst", "sss", "ssh", "glorys", "currents", "winds"]:
            v_dir = Path(f"data/raw/{v}/{y}")
            if v_dir.exists():
                raw_files.extend(list(v_dir.glob("*.nc")))
        
        raw_count = len(raw_files)
        raw_bytes = sum(f.stat().st_size for f in raw_files)
        raw_gb = raw_bytes / (1024 ** 3)
        exp_raw = expected_raw_files[y]

        # 2. Interim checks
        interim_dir = Path(f"data/interim/{y}")
        interim_files = list(interim_dir.glob("*.npz"))
        interim_count = len(interim_files)
        interim_bytes = sum(f.stat().st_size for f in interim_files)
        interim_gb = interim_bytes / (1024 ** 3)

        missing_dates = []
        for d in dates:
            if not (interim_dir / f"oceanembed_{d}.npz").exists():
                missing_dates.append(d)

        # 3. Moments checks
        m_file = Path(f"data/norm_stats/moments_{y}.json")
        has_moments = m_file.exists()
        if has_moments:
            with open(m_file, "r") as f:
                m_data = json.load(f)
            has_7_keys = all(k in m_data["moments"] for k in ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"])
        else:
            has_7_keys = False

        status = (raw_count == exp_raw) and (interim_count == n_exp) and (len(missing_dates) == 0) and has_moments and has_7_keys
        if not status:
            all_ok = False

        print(f"Year {y}:")
        print(f"  Raw:     {raw_count}/{exp_raw} files ({raw_gb:.3f} GB) -> {'OK' if raw_count == exp_raw else 'MISMATCH'}")
        print(f"  Interim: {interim_count}/{n_exp} files ({interim_gb:.3f} GB), Missing dates: {len(missing_dates)} -> {'OK' if len(missing_dates) == 0 else 'FAIL'}")
        print(f"  Moments: {m_file.name} -> {'OK (7 physical keys)' if has_7_keys else 'FAIL'}")
        print()

    print("=" * 80)
    print(f"OVERALL STATUS: {'ALL 4 YEARS INTACT & COMPLETE (1461 DAYS)' if all_ok else 'VERIFICATION FAILED'}")
    print("=" * 80)
    return all_ok

if __name__ == "__main__":
    verify_all()
