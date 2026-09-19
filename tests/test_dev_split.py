"""
tests/test_dev_split.py
-----------------------
Unit tests for the multi-year Phase-1 Development Dataset (2015–2018):
  - Chronological split: 2015–2017 train, 2018 val, test unpopulated
  - Zero date leakage across train and validation splits
  - Sufficient-statistics moment combination mathematical correctness
  - Zero leakage in normalization statistics (train stats strictly independent of 2018)
  - Tensor contracts: [14, 101, 241] input, [15, 101, 241] target
  - Binary validity masks and strict finiteness
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from pipeline.datasets import OceanEmbedDataset, create_dataloader
from pipeline.normalize import (
    PHYSICAL_KEYS,
    combine_moments,
    compute_stats_from_moments,
    load_norm_stats,
    normalize_input_tensor,
)

PROCESSED_DIR = Path("data/processed")
NORM_STATS_DIR = Path("data/norm_stats")
DEV_DATA_EXISTS = (
    (PROCESSED_DIR / "train").exists()
    and len(list((PROCESSED_DIR / "train").glob("*.npz"))) == 1096
    and (PROCESSED_DIR / "val").exists()
    and len(list((PROCESSED_DIR / "val").glob("*.npz"))) == 365
)


class TestMultiYearNormalizationLogic:
    def test_combine_moments_mathematical_invariance(self):
        """Verify that combining moments is strictly order-independent and matches single-pass stats."""
        # Simulated two years of data
        m1 = {
            k: {"sum": 1000.0, "sq_sum": 50000.0, "count": 100}
            for k in PHYSICAL_KEYS
        }
        m2 = {
            k: {"sum": 2000.0, "sq_sum": 110000.0, "count": 100}
            for k in PHYSICAL_KEYS
        }

        # Combine 1 then 2, vs 2 then 1
        c12 = combine_moments([m1, m2])
        c21 = combine_moments([m2, m1])
        assert c12 == c21

        stats = compute_stats_from_moments(c12)
        for k in PHYSICAL_KEYS:
            expected_mean = (1000.0 + 2000.0) / 200.0  # 15.0
            expected_sq = (50000.0 + 110000.0) / 200.0  # 800.0
            expected_var = expected_sq - (expected_mean ** 2)  # 800 - 225 = 575.0
            expected_std = float(np.sqrt(expected_var))
            assert np.isclose(stats[k]["mean"], expected_mean)
            assert np.isclose(stats[k]["std"], expected_std)

    def test_train_stats_strictly_independent_of_2018(self):
        """Verify that frozen train_stats.json matches 2015-2017 moments, NOT 2018 moments."""
        train_stats = load_norm_stats(NORM_STATS_DIR / "train_stats.json")
        dev_stats = load_norm_stats(NORM_STATS_DIR / "dev_stats_2015_2018.json")

        # 2018 has distinct mean for SSH (0.0666 vs 0.0850)
        assert train_stats["SSH"]["mean"] != dev_stats["SSH"]["mean"]
        assert np.isclose(train_stats["SSH"]["mean"], 0.0850, atol=1e-3)


@pytest.mark.skipif(not DEV_DATA_EXISTS, reason="Phase-1 development dataset (1461 files) not found")
class TestPhase1DevelopmentDataset:
    @classmethod
    def setup_class(cls):
        cls.train_ds = OceanEmbedDataset(split="train", fallback_to_synthetic=False)
        cls.val_ds = OceanEmbedDataset(split="val", fallback_to_synthetic=False)

    def test_development_split_sample_counts(self):
        """Train must have exactly 1,096 days (2015-2017) and Val exactly 365 days (2018)."""
        assert len(self.train_ds) == 1096
        assert len(self.val_ds) == 365

    def test_zero_date_leakage_between_splits(self):
        """Dates in train (2015-2017) and val (2018) must have zero overlap."""
        train_dates = set()
        for f in self.train_ds.file_list:
            date_str = f.stem.replace("oceanembed_", "")
            train_dates.add(date_str)

        val_dates = set()
        for f in self.val_ds.file_list:
            date_str = f.stem.replace("oceanembed_", "")
            val_dates.add(date_str)

        overlap = train_dates.intersection(val_dates)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping dates between train and val!"

        # Verify all train dates are in 2015-2017
        for d in train_dates:
            year = int(d.split("-")[0])
            assert 2015 <= year <= 2017, f"Train date outside 2015-2017: {d}"

        # Verify all val dates are in 2018
        for d in val_dates:
            year = int(d.split("-")[0])
            assert year == 2018, f"Val date outside 2018: {d}"

    def test_dataloader_batch_contract_and_finiteness(self):
        """Verify DataLoader batch tensor contracts and binary masks."""
        loader = create_dataloader(self.train_ds, batch_size=4, shuffle=True)
        batch_in, batch_tgt, meta = next(iter(loader))

        assert batch_in.shape == (4, 14, 101, 241)
        assert batch_tgt.shape == (4, 15, 101, 241)
        assert torch.isfinite(batch_in).all()
        assert torch.isfinite(batch_tgt).all()

        # Binary masks
        masks = batch_in[:, 7:]
        for u in masks.unique():
            assert u.item() in (0.0, 1.0)
