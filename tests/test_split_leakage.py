"""
tests/test_split_leakage.py
---------------------------
Tests for temporal split integrity.

Rules verified:
  - Train, validation, and test date ranges are disjoint
  - No date appears in more than one split
  - Split boundaries match config settings
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config


def _make_splits(cfg) -> dict[str, pd.DatetimeIndex]:
    """Build date ranges from config."""
    return {
        "train": pd.date_range(cfg.splits.train_start, cfg.splits.train_end, freq="D"),
        "val":   pd.date_range(cfg.splits.val_start,   cfg.splits.val_end,   freq="D"),
        "test":  pd.date_range(cfg.splits.test_start,  cfg.splits.test_end,  freq="D"),
    }


class TestSplitLeakage:
    @pytest.fixture(autouse=True)
    def _cfg(self):
        self.cfg = load_config("data")
        self.splits = _make_splits(self.cfg)

    def test_train_val_disjoint(self):
        overlap = self.splits["train"].intersection(self.splits["val"])
        assert len(overlap) == 0, (
            f"Train and val overlap on {len(overlap)} dates: {overlap[:5].tolist()}"
        )

    def test_train_test_disjoint(self):
        overlap = self.splits["train"].intersection(self.splits["test"])
        assert len(overlap) == 0, (
            f"Train and test overlap on {len(overlap)} dates"
        )

    def test_val_test_disjoint(self):
        overlap = self.splits["val"].intersection(self.splits["test"])
        assert len(overlap) == 0, (
            f"Val and test overlap on {len(overlap)} dates"
        )

    def test_all_splits_non_empty(self):
        for name, dates in self.splits.items():
            assert len(dates) > 0, f"{name} split is empty"

    def test_train_before_val(self):
        assert self.splits["train"].max() < self.splits["val"].min(), (
            "Latest training date must precede earliest validation date"
        )

    def test_val_before_test(self):
        assert self.splits["val"].max() < self.splits["test"].min(), (
            "Latest validation date must precede earliest test date"
        )

    def test_split_boundaries_from_config(self):
        """Split boundaries must come from config, not be hardcoded."""
        cfg_train_end = pd.Timestamp(self.cfg.splits.train_end)
        assert self.splits["train"].max() == cfg_train_end
