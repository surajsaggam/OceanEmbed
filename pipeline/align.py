"""
pipeline/align.py
-----------------
Temporal alignment and date harmonization for OceanEmbed.

Scientific rules:
  - Surface observations and target GLORYS fields are aligned to daily timestamps
    (12:00 UTC reference).
  - Train, validation, and test splits are strictly temporal:
      * Train: e.g. 2015-01-01 to 2021-12-31
      * Val:   e.g. 2022-01-01 to 2022-12-31
      * Test:  e.g. 2023-01-01 to 2024-12-31
  - No random date shuffling across splits (preserves zero-leakage guarantee).
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import pandas as pd

from utils.config import load_config


def parse_date_str(d: str) -> pd.Timestamp:
    """
    Parses date string to a normalized UTC pd.Timestamp.
    """
    return pd.to_datetime(d).tz_localize(None).floor("D")


def get_split_date_ranges(split_cfg: Optional[Dict] = None) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Retrieves the start and end timestamps for each split from config.
    """
    if split_cfg is None:
        cfg = load_config("data")
        split_cfg = cfg.splits

    splits = {}
    for name in ["train", "val", "test"]:
        start_key = f"{name}_start"
        end_key = f"{name}_end"
        if hasattr(split_cfg, start_key) and hasattr(split_cfg, end_key):
            start = parse_date_str(getattr(split_cfg, start_key))
            end = parse_date_str(getattr(split_cfg, end_key))
        elif isinstance(split_cfg, dict) and start_key in split_cfg and end_key in split_cfg:
            start = parse_date_str(split_cfg[start_key])
            end = parse_date_str(split_cfg[end_key])
        else:
            sc = getattr(split_cfg, name) if hasattr(split_cfg, name) else split_cfg[name]
            start = parse_date_str(sc.start if hasattr(sc, "start") else sc["start"])
            end = parse_date_str(sc.end if hasattr(sc, "end") else sc["end"])
        splits[name] = (start, end)
    return splits


def get_split_for_date(target_date: Union[str, datetime, date, pd.Timestamp], splits: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]) -> Optional[str]:
    """
    Determines which split a date belongs to.
    """
    dt = parse_date_str(str(target_date))
    for split_name, (start, end) in splits.items():
        if start <= dt <= end:
            return split_name
    return None


def generate_date_range(start: str, end: str) -> List[str]:
    """
    Generates list of daily date strings formatted as YYYY-MM-DD.
    """
    dr = pd.date_range(start=start, end=end, freq="D")
    return [d.strftime("%Y-%m-%d") for d in dr]
