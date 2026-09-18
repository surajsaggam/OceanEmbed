"""
pipeline/preprocess.py
-----------------------
Production batch preprocessing pipeline for OceanEmbed.

Processes raw satellite observations and GLORYS reanalysis into model-ready
tensors, adhering strictly to the scientific contracts:
  - 14 input channels: 7 physical surface variables + 7 binary validity masks
  - 15 target channels: GLORYS thetao interpolated to 15 standard depths
  - Zero unmasked NaNs: torch.isfinite(input).all() guaranteed
  - Training-only normalization: Statistics computed exclusively on training dates
  - Strict temporal splits: Disjoint date ranges for train, val, and test
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr

from pipeline.harmonize import DataHarmonizer
from pipeline.normalize import compute_variable_stats, normalize_array, save_norm_stats
from pipeline.regrid import get_default_target_coords
from utils.config import load_config


PHYSICAL_KEYS = ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]


class ProductionPreprocessor:
    """
    Coordinates batch preprocessing from raw NetCDF files into normalized,
    regridded daily .npz tensors.
    """

    def __init__(
        self,
        raw_dir: Union[str, Path] = "data/raw",
        processed_dir: Union[str, Path] = "data/processed",
        norm_stats_path: Union[str, Path] = "data/norm_stats/train_stats.json",
        data_cfg: Optional[Any] = None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.norm_stats_path = Path(norm_stats_path)
        self.data_cfg = data_cfg or load_config("data")
        self.harmonizer = DataHarmonizer(self.data_cfg)

        self.lat_tgt, self.lon_tgt = get_default_target_coords()
        self.H = len(self.lat_tgt)
        self.W = len(self.lon_tgt)
        self.target_depths = list(self.data_cfg.depths_m)

    def discover_raw_files(self) -> Dict[str, Any]:
        """
        Locates and validates raw input files in raw_dir.
        """
        sst_file = self.raw_dir / "sst" / "sst_jan2020.nc"
        sss_file = self.raw_dir / "sss" / "sss_jan2020.nc"
        ssh_file = self.raw_dir / "ssh" / "ssh_jan2020.nc"
        currents_files = sorted(list((self.raw_dir / "currents").glob("*.nc")))
        winds_files = sorted(list((self.raw_dir / "winds").glob("*.nc")))
        glorys_file = self.raw_dir / "glorys" / "glorys_jan2020.nc"

        missing = []
        for name, p in [("SST", sst_file), ("SSS", sss_file), ("SSH", ssh_file), ("GLORYS", glorys_file)]:
            if not p.exists():
                missing.append(f"{name} ({p})")
        if len(currents_files) == 0:
            missing.append("Currents (data/raw/currents/*.nc)")
        if len(winds_files) == 0:
            missing.append("Winds (data/raw/winds/*.nc)")

        if missing:
            raise FileNotFoundError(f"Missing required raw files: {', '.join(missing)}")

        return {
            "sst": sst_file,
            "sss": sss_file,
            "ssh": ssh_file,
            "currents": currents_files,
            "winds": winds_files,
            "glorys": glorys_file,
            "num_days": min(
                31,
                len(currents_files),
                len(winds_files),
            ),
        }

    def compute_training_normalization_stats(
        self,
        raw_files: Dict[str, Any],
        train_indices: List[int],
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes z-score normalization statistics (mean, std) exclusively from
        the training dates over valid ocean pixels (mask == 1).
        """
        print(f"[Preprocessing] Computing normalization statistics from {len(train_indices)} training days...")

        accumulators = {
            k: {"sum": 0.0, "sq_sum": 0.0, "count": 0}
            for k in PHYSICAL_KEYS
        }

        for idx in train_indices:
            # Harmonize single day un-normalized
            sst, m_sst = self.harmonizer.harmonize_sst(raw_files["sst"], time_idx=idx)
            sss, m_sss = self.harmonizer.harmonize_sss(raw_files["sss"], time_idx=idx)
            sla, m_sla = self.harmonizer.harmonize_ssh(raw_files["ssh"], time_idx=idx)
            uc, m_uc, vc, m_vc = self.harmonizer.harmonize_currents(raw_files["currents"][idx])
            uw, m_uw, vw, m_vw = self.harmonizer.harmonize_winds(raw_files["winds"][idx])

            day_vars = {
                "SST": (sst, m_sst),
                "SSS": (sss, m_sss),
                "SSH": (sla, m_sla),
                "U_curr": (uc, m_uc),
                "V_curr": (vc, m_vc),
                "WindU": (uw, m_uw),
                "WindV": (vw, m_vw),
            }

            for k, (arr, mask) in day_vars.items():
                valid = (mask == 1.0) & np.isfinite(arr)
                if np.any(valid):
                    v_vals = arr[valid].astype(np.float64)
                    accumulators[k]["sum"] += np.sum(v_vals)
                    accumulators[k]["sq_sum"] += np.sum(v_vals ** 2)
                    accumulators[k]["count"] += len(v_vals)

        stats = {}
        for k, acc in accumulators.items():
            cnt = acc["count"]
            if cnt == 0:
                raise ValueError(f"No valid training samples for variable '{k}'!")
            mu = acc["sum"] / cnt
            variance = max(1e-8, (acc["sq_sum"] / cnt) - (mu ** 2))
            sigma = float(np.sqrt(variance))
            stats[k] = {"mean": float(mu), "std": float(sigma)}
            print(f"  {k:6s} -> mean: {mu:8.4f}, std: {sigma:8.4f} (from {cnt} valid points)")

        # Save to disk
        save_norm_stats(stats, self.norm_stats_path)
        print(f"[Preprocessing] Normalization stats saved to {self.norm_stats_path}")
        return stats

    def run_preprocessing(
        self,
        train_days: int = 21,
        val_days: int = 5,
        test_days: int = 5,
    ) -> Dict[str, int]:
        """
        Executes full preprocessing pipeline across all days and writes .npz files.
        """
        raw_files = self.discover_raw_files()
        total_days = raw_files["num_days"]
        assert train_days + val_days + test_days <= total_days, (
            f"Requested {train_days}+{val_days}+{test_days} days > available {total_days} days"
        )

        train_indices = list(range(0, train_days))
        val_indices = list(range(train_days, train_days + val_days))
        test_indices = list(range(train_days + val_days, train_days + val_days + test_days))

        # 1. Compute training normalization statistics (zero leakage)
        norm_stats = self.compute_training_normalization_stats(raw_files, train_indices)

        # 2. Process all splits
        split_map = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

        counts = {"train": 0, "val": 0, "test": 0}

        for split_name, indices in split_map.items():
            split_dir = self.processed_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            print(f"[Preprocessing] Generating {len(indices)} samples for '{split_name}' split...")

            for t_idx in indices:
                day_num = t_idx + 1
                date_str = f"2020-01-{day_num:02d}"

                in_tensor, target_tensor, meta = self.harmonizer.build_daily_sample(
                    sst_file=raw_files["sst"],
                    sss_file=raw_files["sss"],
                    ssh_file=raw_files["ssh"],
                    currents_file=raw_files["currents"][t_idx],
                    winds_file=raw_files["winds"][t_idx],
                    glorys_file=raw_files["glorys"],
                    time_idx=t_idx,
                    norm_stats=norm_stats,
                )

                # Strict finiteness validation
                assert torch.isfinite(in_tensor).all(), f"Non-finite input on {date_str}"
                assert torch.isfinite(target_tensor).all(), f"Non-finite target on {date_str}"

                target_mask = meta["target_mask"].numpy()

                out_path = split_dir / f"oceanembed_{date_str}.npz"
                np.savez_compressed(
                    out_path,
                    input=in_tensor.numpy(),
                    target=target_tensor.numpy(),
                    target_mask=target_mask,
                    date=date_str,
                    day_idx=t_idx,
                )
                counts[split_name] += 1

        # 3. Save grid and metadata
        meta_path = self.processed_dir / "grid_metadata.json"
        metadata = {
            "domain": {
                "lat_min": float(self.lat_tgt[0]),
                "lat_max": float(self.lat_tgt[-1]),
                "lon_min": float(self.lon_tgt[0]),
                "lon_max": float(self.lon_tgt[-1]),
                "resolution": float(self.lat_tgt[1] - self.lat_tgt[0]),
                "H": self.H,
                "W": self.W,
            },
            "depths_m": self.target_depths,
            "physical_variables": PHYSICAL_KEYS,
            "num_channels_input": 14,
            "num_channels_target": len(self.target_depths),
            "splits": {
                "train": {"count": counts["train"], "start": "2020-01-01", "end": f"2020-01-{train_days:02d}"},
                "val": {"count": counts["val"], "start": f"2020-01-{train_days+1:02d}", "end": f"2020-01-{train_days+val_days:02d}"},
                "test": {"count": counts["test"], "start": f"2020-01-{train_days+val_days+1:02d}", "end": f"2020-01-{total_days:02d}"},
            },
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Preprocessing] Complete! Generated {sum(counts.values())} files in {self.processed_dir}")
        return counts
