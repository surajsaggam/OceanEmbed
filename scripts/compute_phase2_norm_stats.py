"""
scripts/compute_phase2_norm_stats.py
------------------------------------
Computes frozen normalization statistics (mean, std) for Phase-2 temporal features:
  - Delta_SST = SST(t) - SST(t-1)
  - Delta_SSH = SSH(t) - SSH(t-1)

Scientific & Integrity Rules:
1. Training Split Only:
   Statistics are computed strictly and exclusively from the 2015-01-01 to 2017-12-31
   training observations.
2. Zero Leakage:
   Observations from 2018 (validation) and 2019 (test) MUST NOT be accessed or used.
3. Valid Ocean Only:
   Only pixels where both day t and day t-1 are valid ocean observations
   (M_delta == 1.0) contribute to the statistics.
4. Immutability:
   Phase-1 stats file (data/norm_stats/train_stats.json) is left completely untouched.
   Outputs are saved separately to data/norm_stats/phase2_train_stats.json.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.temporal_features import compute_temporal_deltas


def compute_phase2_stats(
    train_dir: Path = PROJECT_ROOT / "data" / "processed" / "train",
    phase1_stats_path: Path = PROJECT_ROOT / "data" / "norm_stats" / "train_stats.json",
    out_path: Path = PROJECT_ROOT / "data" / "norm_stats" / "phase2_train_stats.json",
) -> dict:
    print(f"[Phase-2 Normalization] Reading Phase-1 stats from {phase1_stats_path}...")
    with open(phase1_stats_path, "r") as f:
        phase1_stats = json.load(f)

    sst_mean = float(phase1_stats["SST"]["mean"])
    sst_std = float(phase1_stats["SST"]["std"])
    ssh_mean = float(phase1_stats["SSH"]["mean"])
    ssh_std = float(phase1_stats["SSH"]["std"])

    train_files = sorted(list(train_dir.glob("oceanembed_*.npz")))
    if not train_files:
        raise FileNotFoundError(f"No processed training files found in {train_dir}")

    print(f"[Phase-2 Normalization] Found {len(train_files)} training files (2015–2017).")
    print(f"  First: {train_files[0].name}")
    print(f"  Last:  {train_files[-1].name}")

    # Accumulators in float64 for exact precision
    sst_sum = 0.0
    sst_sq_sum = 0.0
    sst_count = 0

    ssh_sum = 0.0
    ssh_sq_sum = 0.0
    ssh_count = 0

    prev_sst = None
    prev_mask_sst = None
    prev_ssh = None
    prev_mask_ssh = None

    valid_transitions = 0

    for i, file_path in enumerate(train_files):
        with np.load(file_path) as data:
            inp = data["input"]  # [14, H, W]
            date_str = str(data.get("date", file_path.stem))

        # Unnormalize SST and SSH on day t
        # Channel 0: SST, Channel 2: SSH
        # Channel 7: M_SST, Channel 9: M_SSH
        sst_norm = inp[0]
        mask_sst = inp[7]
        ssh_norm = inp[2]
        mask_ssh = inp[9]

        sst_t = sst_norm * sst_std + sst_mean
        ssh_t = ssh_norm * ssh_std + ssh_mean

        if prev_sst is not None:
            # Compute temporal differences and joint validity masks
            d_sst, m_sst, d_ssh, m_ssh = compute_temporal_deltas(
                sst_t=sst_t,
                mask_sst_t=mask_sst,
                ssh_t=ssh_t,
                mask_ssh_t=mask_ssh,
                sst_t_minus_1=prev_sst,
                mask_sst_t_minus_1=prev_mask_sst,
                ssh_t_minus_1=prev_ssh,
                mask_ssh_t_minus_1=prev_mask_ssh,
            )

            # Accumulate for Delta_SST
            valid_sst = (m_sst == 1.0) & np.isfinite(d_sst)
            vals_sst = d_sst[valid_sst].astype(np.float64)
            if len(vals_sst) > 0:
                sst_sum += float(np.sum(vals_sst))
                sst_sq_sum += float(np.sum(vals_sst ** 2))
                sst_count += int(len(vals_sst))

            # Accumulate for Delta_SSH
            valid_ssh = (m_ssh == 1.0) & np.isfinite(d_ssh)
            vals_ssh = d_ssh[valid_ssh].astype(np.float64)
            if len(vals_ssh) > 0:
                ssh_sum += float(np.sum(vals_ssh))
                ssh_sq_sum += float(np.sum(vals_ssh ** 2))
                ssh_count += int(len(vals_ssh))

            valid_transitions += 1

        # Advance t-1
        prev_sst = sst_t
        prev_mask_sst = mask_sst
        prev_ssh = ssh_t
        prev_mask_ssh = mask_ssh

        if (i + 1) % 200 == 0 or (i + 1) == len(train_files):
            print(f"  Processed {i+1}/{len(train_files)} files ({date_str})...")

    if sst_count == 0 or ssh_count == 0:
        raise RuntimeError("No valid temporal observations found in training set!")

    # Compute final moments
    mu_sst = sst_sum / sst_count
    var_sst = max(1e-12, (sst_sq_sum / sst_count) - (mu_sst ** 2))
    std_sst = float(np.sqrt(var_sst))

    mu_ssh = ssh_sum / ssh_count
    var_ssh = max(1e-12, (ssh_sq_sum / ssh_count) - (mu_ssh ** 2))
    std_ssh = float(np.sqrt(var_ssh))

    phase2_stats = {
        "delta_SST": {
            "mean": float(mu_sst),
            "std": float(std_sst),
        },
        "delta_SSH": {
            "mean": float(mu_ssh),
            "std": float(std_ssh),
        },
        "metadata": {
            "training_period": "2015-01-01 to 2017-12-31",
            "total_training_days": len(train_files),
            "valid_transitions": valid_transitions,
            "valid_observations_sst": sst_count,
            "valid_observations_ssh": ssh_count,
            "units": {
                "delta_SST": "deg C / day",
                "delta_SSH": "m / day",
            },
        },
    }

    print("\n[Phase-2 Normalization Results]")
    print(f"  Delta_SST: mean = {mu_sst:.6f} deg C, std = {std_sst:.6f} deg C (N={sst_count:,})")
    print(f"  Delta_SSH: mean = {mu_ssh:.6f} m,     std = {std_ssh:.6f} m     (N={ssh_count:,})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(phase2_stats, f, indent=2)
    print(f"[Phase-2 Normalization] Saved stats to {out_path}")

    return phase2_stats


if __name__ == "__main__":
    compute_phase2_stats()
