"""
pipeline/datasets.py
--------------------
PyTorch Dataset and DataLoader implementations for OceanEmbed.

Scientific contract:
  - Input tensor shape: [14, H, W]
      * Channels 0–6: 7 physical surface variables (normalized, finite)
      * Channels 7–13: 7 binary validity masks (1.0 = observed ocean, 0.0 = imputed/land)
  - Target tensor shape: [15, H, W]
      * 15 standard ocean depths: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
  - Strict Finiteness Guarantee:
      * torch.isfinite(input_tensor).all() is verified. No NaNs or Infs enter the model.
  - Land/QC masks:
      * Land pixels carry their normalized deterministic fill value, with validity mask = 0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.regrid import compute_grid_shape
from utils.config import load_config


PHYSICAL_VARIABLES = ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]


class SyntheticOceanDataset(Dataset):
    """
    Synthetic OceanEmbed dataset for testing, model debugging gates,
    and offline integration tests without requiring downloaded satellite NetCDF files.
    """

    def __init__(
        self,
        num_samples: int = 16,
        grid_shape: Tuple[int, int] = (101, 241),
        n_depths: int = 15,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.H, self.W = grid_shape
        self.n_depths = n_depths

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

            # 14-channel input: concat physical + masks
            in_tensor = np.concatenate([phys, masks], axis=0)  # [14, H, W]

            # 15 target depths [15, H, W]
            # Surface warm (~28C), deep cold (~4C)
            depth_means = np.linspace(28.0, 4.0, self.n_depths).reshape(-1, 1, 1).astype(np.float32)
            target = depth_means + rng.normal(loc=0.0, scale=0.5, size=(self.n_depths, self.H, self.W)).astype(np.float32)
            # Land on target can carry fill
            # Target mask [15, H, W]
            target_mask = np.ones((self.n_depths, self.H, self.W), dtype=np.float32)
            target_mask[:, self.land_mask] = 0.0

            self.samples.append((
                torch.from_numpy(in_tensor).to(torch.float32),
                torch.from_numpy(target).to(torch.float32),
                {
                    "date": f"2020-01-{i+1:02d}",
                    "index": i,
                    "target_mask": torch.from_numpy(target_mask).to(torch.float32),
                }
            ))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        in_tensor, target_tensor, meta = self.samples[idx]
        assert torch.isfinite(in_tensor).all(), "Input tensor contains non-finite values!"
        return in_tensor, target_tensor, meta


class OceanEmbedDataset(Dataset):
    """
    Production OceanEmbed PyTorch Dataset reading harmonized preprocessed files or Zarr stores.

    If data directory does not contain processed files, falls back seamlessly
    to SyntheticOceanDataset if fallback_to_synthetic is True.
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/processed",
        split: str = "train",
        fallback_to_synthetic: bool = True,
        num_synthetic_samples: int = 16,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.cfg_data = load_config("data")

        # Determine spatial grid shape from config
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
        self.file_list = sorted(list(split_dir.glob("*.npz"))) if split_dir.exists() else []

        if len(self.file_list) == 0:
            if fallback_to_synthetic:
                self._synthetic = SyntheticOceanDataset(
                    num_samples=num_synthetic_samples,
                    grid_shape=(self.H, self.W),
                    n_depths=self.n_depths,
                )
            else:
                raise FileNotFoundError(f"No preprocessed files found in {split_dir}")
        else:
            self._synthetic = None

    def __len__(self) -> int:
        if self._synthetic is not None:
            return len(self._synthetic)
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self._synthetic is not None:
            return self._synthetic[idx]

        file_path = self.file_list[idx]
        with np.load(file_path) as data:
            in_tensor = torch.from_numpy(data["input"]).to(torch.float32)
            target = torch.from_numpy(data["target"]).to(torch.float32)
            date_str = str(data.get("date", file_path.stem))
            meta = {"date": date_str, "file": str(file_path), "index": idx}
            if "target_mask" in data:
                meta["target_mask"] = torch.from_numpy(data["target_mask"]).to(torch.float32)

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
