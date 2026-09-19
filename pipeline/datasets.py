"""
pipeline/datasets.py
--------------------
PyTorch Dataset and DataLoader implementations for OceanEmbed (Phase-1 and Phase-2).

Scientific contract:
  - Input tensor shape:
      * Phase 1 (in_channels=14): [14, H, W]
          Channels 0–6: 7 physical surface variables (normalized, finite)
          Channels 7–13: 7 binary validity masks (1.0 = observed ocean, 0.0 = imputed/land)
      * Phase 2 (in_channels=16): [16, H, W]
          Channels 0–13: Identical to Phase 1
          Channel 14: Normalized Delta_SST = SST(t) - SST(t-1) (0.0 fill where invalid)
          Channel 15: Normalized Delta_SSH = SSH(t) - SSH(t-1) (0.0 fill where invalid)
          meta["delta_mask"]: [2, H, W] binary masks for Delta_SST and Delta_SSH validity
  - Target tensor shape: [15, H, W]
      * 15 standard ocean depths: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
  - Strict Finiteness Guarantee:
      * torch.isfinite(input_tensor).all() is verified. No NaNs or Infs enter the model.
  - Zero Leakage & Boundary Handling:
      * 2015-01-01: Previous day unavailable -> Delta is unobserved (mask=0.0, norm_fill=0.0).
      * 2018-01-01 (val start): Uses 2017-12-31 historical training data.
      * 2019-01-01 (test start): Uses 2018-12-31 historical validation data.
      * Future information is NEVER accessible or referenced.
  - Zero Storage Duplication:
      * Delta features are constructed dynamically on-the-fly from existing processed files.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.regrid import compute_grid_shape
from pipeline.temporal_features import compute_temporal_deltas, normalize_temporal_deltas
from utils.config import load_config


PHYSICAL_VARIABLES = ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]


class SyntheticOceanDataset(Dataset):
    """
    Synthetic OceanEmbed dataset for testing, model debugging gates,
    and offline integration tests without requiring downloaded satellite NetCDF files.
    Supports both 14-channel (Phase 1) and 16-channel (Phase 2) contracts.
    """

    def __init__(
        self,
        num_samples: int = 16,
        grid_shape: Tuple[int, int] = (101, 241),
        n_depths: int = 15,
        in_channels: int = 14,
        seed: int = 42,
    ) -> None:
        super().__init__()
        if in_channels not in (14, 16):
            raise ValueError(f"Unsupported in_channels={in_channels}. Must be 14 or 16.")

        self.num_samples = num_samples
        self.H, self.W = grid_shape
        self.n_depths = n_depths
        self.in_channels = in_channels

        rng = np.random.default_rng(seed)

        # Generate realistic land mask (e.g. northern & eastern land boundaries)
        self.land_mask = np.zeros((self.H, self.W), dtype=bool)
        self.land_mask[80:, :] = True  # Northern boundary land
        self.land_mask[:, 190:] = True  # Eastern boundary land

        # Pre-generate synthetic samples to ensure fast, deterministic retrieval
        self.samples = []
        for i in range(num_samples):
            # 7 physical channels [7, H, W]
            phys = rng.normal(loc=0.0, scale=1.0, size=(7, self.H, self.W)).astype(np.float32)

            # 7 validity masks [7, H, W]
            masks = np.ones((7, self.H, self.W), dtype=np.float32)
            # Apply land mask
            masks[:, self.land_mask] = 0.0
            # Some random cloud/missingness on ocean
            cloud_mask = (rng.random(size=(7, self.H, self.W)) < 0.05) & (~self.land_mask)
            masks[cloud_mask] = 0.0

            # Land and missing pixels get finite fill (normalized fill = 0.0)
            phys[masks == 0.0] = 0.0

            # 14-channel base input: concat physical + masks
            base_tensor = np.concatenate([phys, masks], axis=0)  # [14, H, W]

            meta: Dict[str, Any] = {
                "date": f"2020-01-{i+1:02d}",
                "index": i,
            }

            if in_channels == 16:
                # Synthetic Delta_SST and Delta_SSH
                # Valid iff ocean is valid
                m_delta_sst = masks[0].copy()
                m_delta_ssh = masks[2].copy()

                # Boundary: first sample has no previous day
                if i == 0:
                    m_delta_sst[:] = 0.0
                    m_delta_ssh[:] = 0.0

                delta_sst = rng.normal(loc=0.0, scale=0.25, size=(self.H, self.W)).astype(np.float32)
                delta_ssh = rng.normal(loc=0.0, scale=0.01, size=(self.H, self.W)).astype(np.float32)

                # Normalize (dummy scale for synthetic)
                norm_d_sst = delta_sst / 0.25
                norm_d_ssh = delta_ssh / 0.01

                norm_d_sst[m_delta_sst == 0.0] = 0.0
                norm_d_ssh[m_delta_ssh == 0.0] = 0.0

                in_tensor = np.concatenate(
                    [base_tensor, norm_d_sst[np.newaxis, :, :], norm_d_ssh[np.newaxis, :, :]],
                    axis=0,
                )  # [16, H, W]
                meta["delta_mask"] = torch.from_numpy(
                    np.stack([m_delta_sst, m_delta_ssh], axis=0)
                ).to(torch.float32)
            else:
                in_tensor = base_tensor  # [14, H, W]

            # 15 target depths [15, H, W]
            depth_means = np.linspace(28.0, 4.0, self.n_depths).reshape(-1, 1, 1).astype(np.float32)
            target = depth_means + rng.normal(
                loc=0.0, scale=0.5, size=(self.n_depths, self.H, self.W)
            ).astype(np.float32)
            target_mask = np.ones((self.n_depths, self.H, self.W), dtype=np.float32)
            target_mask[:, self.land_mask] = 0.0
            meta["target_mask"] = torch.from_numpy(target_mask).to(torch.float32)

            self.samples.append((
                torch.from_numpy(in_tensor).to(torch.float32),
                torch.from_numpy(target).to(torch.float32),
                meta,
            ))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        in_tensor, target_tensor, meta = self.samples[idx]
        assert torch.isfinite(in_tensor).all(), "Input tensor contains non-finite values!"
        return in_tensor, target_tensor, meta


class OceanEmbedDataset(Dataset):
    """
    Production OceanEmbed PyTorch Dataset reading harmonized preprocessed files.
    Supports both 14-channel (Phase 1) and 16-channel (Phase 2) contracts.

    For Phase 2 (in_channels=16):
      Constructs normalized Delta_SST and Delta_SSH dynamically on-the-fly with
      zero disk storage duplication and strict temporal causality.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/processed",
        split: str = "train",
        fallback_to_synthetic: bool = True,
        num_synthetic_samples: int = 16,
        in_channels: int = 14,
        phase1_stats_path: Union[str, Path] = "data/norm_stats/train_stats.json",
        phase2_stats_path: Union[str, Path] = "data/norm_stats/phase2_train_stats.json",
    ) -> None:
        super().__init__()
        if in_channels not in (14, 16):
            raise ValueError(f"Unsupported in_channels={in_channels}. Must be 14 or 16.")

        self.data_dir = Path(data_dir)
        self.split = split
        self.in_channels = in_channels
        self.cfg_data = load_config("data")

        # Spatial grid shape
        self.H, self.W = compute_grid_shape(
            self.cfg_data.domain.lat_min,
            self.cfg_data.domain.lat_max,
            self.cfg_data.domain.lon_min,
            self.cfg_data.domain.lon_max,
            self.cfg_data.domain.resolution,
        )
        self.n_depths = len(self.cfg_data.depths_m)

        # Check for preprocessed data files
        split_dir = self.data_dir / split
        self.file_list = sorted(list(split_dir.glob("oceanembed_*.npz"))) if split_dir.exists() else []

        if len(self.file_list) == 0:
            if fallback_to_synthetic:
                self._synthetic = SyntheticOceanDataset(
                    num_samples=num_synthetic_samples,
                    grid_shape=(self.H, self.W),
                    n_depths=self.n_depths,
                    in_channels=self.in_channels,
                )
            else:
                raise FileNotFoundError(f"No preprocessed files found in {split_dir}")
        else:
            self._synthetic = None

        # If Phase-2 (16 channels), initialize normalization stats and historical date map
        if self._synthetic is None and self.in_channels == 16:
            p1_path = Path(phase1_stats_path)
            p2_path = Path(phase2_stats_path)
            if not p1_path.exists():
                raise FileNotFoundError(f"Phase-1 normalization stats not found at {p1_path}")
            if not p2_path.exists():
                raise FileNotFoundError(f"Phase-2 normalization stats not found at {p2_path}")

            with open(p1_path, "r") as f:
                self.phase1_stats = json.load(f)
            with open(p2_path, "r") as f:
                self.phase2_stats = json.load(f)

            self.sst_mean = float(self.phase1_stats["SST"]["mean"])
            self.sst_std = float(self.phase1_stats["SST"]["std"])
            self.ssh_mean = float(self.phase1_stats["SSH"]["mean"])
            self.ssh_std = float(self.phase1_stats["SSH"]["std"])

            # Build strictly historical date map (never indexing future splits)
            self._historical_date_map: Dict[str, Path] = {}

            # Current split files
            for p in self.file_list:
                date_str = p.stem.replace("oceanembed_", "")
                self._historical_date_map[date_str] = p

            # If validation: add historical train split (allows 2018-01-01 to use 2017-12-31)
            if self.split == "val":
                train_dir = self.data_dir / "train"
                if train_dir.exists():
                    for p in train_dir.glob("oceanembed_*.npz"):
                        date_str = p.stem.replace("oceanembed_", "")
                        self._historical_date_map[date_str] = p

            # If test: add historical val and train splits (allows 2019-01-01 to use 2018-12-31)
            if self.split == "test":
                for past_split in ("train", "val"):
                    past_dir = self.data_dir / past_split
                    if past_dir.exists():
                        for p in past_dir.glob("oceanembed_*.npz"):
                            date_str = p.stem.replace("oceanembed_", "")
                            self._historical_date_map[date_str] = p

            # Single-entry read cache for fast sequential reads
            self._cache_date: Optional[str] = None
            self._cache_data: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None

    def __len__(self) -> int:
        if self._synthetic is not None:
            return len(self._synthetic)
        return len(self.file_list)

    def _load_historical_day(
        self, date_str: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Loads unnormalized SST and SSH for a given historical date.
        Returns (sst, mask_sst, ssh, mask_ssh) or None if unavailable.
        """
        if self._cache_date == date_str and self._cache_data is not None:
            return self._cache_data

        if date_str not in self._historical_date_map:
            return None

        file_path = self._historical_date_map[date_str]
        with np.load(file_path) as data:
            inp = data["input"]

        sst = inp[0] * self.sst_std + self.sst_mean
        mask_sst = inp[7]
        ssh = inp[2] * self.ssh_std + self.ssh_mean
        mask_ssh = inp[9]

        res = (sst, mask_sst, ssh, mask_ssh)
        self._cache_date = date_str
        self._cache_data = res
        return res

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self._synthetic is not None:
            return self._synthetic[idx]

        file_path = self.file_list[idx]
        with np.load(file_path) as data:
            raw_input = data["input"]  # [14, H, W]
            target = torch.from_numpy(data["target"]).to(torch.float32)
            date_str = str(data.get("date", file_path.stem.replace("oceanembed_", "")))
            meta: Dict[str, Any] = {"date": date_str, "file": str(file_path), "index": idx}
            if "target_mask" in data:
                meta["target_mask"] = torch.from_numpy(data["target_mask"]).to(torch.float32)

        if self.in_channels == 14:
            in_tensor = torch.from_numpy(raw_input).to(torch.float32)
        elif self.in_channels == 16:
            # Reconstruct day t physical SST and SSH
            sst_t = raw_input[0] * self.sst_std + self.sst_mean
            mask_sst_t = raw_input[7]
            ssh_t = raw_input[2] * self.ssh_std + self.ssh_mean
            mask_ssh_t = raw_input[9]

            # Determine previous day date (strict temporal causality: t-1 only)
            curr_date = datetime.date.fromisoformat(date_str)
            prev_date = curr_date - datetime.timedelta(days=1)
            prev_date_str = prev_date.isoformat()

            prev_vals = self._load_historical_day(prev_date_str)
            if prev_vals is not None:
                sst_prev, mask_sst_prev, ssh_prev, mask_ssh_prev = prev_vals
            else:
                sst_prev, mask_sst_prev, ssh_prev, mask_ssh_prev = None, None, None, None

            # Compute physical differences and joint validity masks
            d_sst, m_sst, d_ssh, m_ssh = compute_temporal_deltas(
                sst_t=sst_t,
                mask_sst_t=mask_sst_t,
                ssh_t=ssh_t,
                mask_ssh_t=mask_ssh_t,
                sst_t_minus_1=sst_prev,
                mask_sst_t_minus_1=mask_sst_prev,
                ssh_t_minus_1=ssh_prev,
                mask_ssh_t_minus_1=mask_ssh_prev,
            )

            # Apply frozen Phase-2 normalization (invalid pixels receive 0.0 fill)
            norm_sst, norm_ssh = normalize_temporal_deltas(
                d_sst, m_sst, d_ssh, m_ssh, self.phase2_stats
            )

            # Concatenate channels: [14, H, W] + [1, H, W] + [1, H, W] -> [16, H, W]
            inp_16 = np.concatenate(
                [raw_input, norm_sst[np.newaxis, :, :], norm_ssh[np.newaxis, :, :]],
                axis=0,
            )
            in_tensor = torch.from_numpy(inp_16).to(torch.float32)

            # Record explicit delta validity masks in metadata
            meta["delta_mask"] = torch.from_numpy(
                np.stack([m_sst, m_ssh], axis=0)
            ).to(torch.float32)

            # Cache day t for potential subsequent access
            self._cache_date = date_str
            self._cache_data = (sst_t, mask_sst_t, ssh_t, mask_ssh_t)
        else:
            raise ValueError(f"Unsupported in_channels={self.in_channels}")

        # Enforce the isfinite contract
        assert torch.isfinite(in_tensor).all(), f"Input tensor from {file_path} contains non-finite values"
        return in_tensor, target, meta


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader configured according to train settings.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
