"""
evaluation/ablations.py
-----------------------
Ablation experiment framework for OceanEmbed.

Scientific protocol:
  - Each ablation alters EXACTLY ONE component relative to the Phase-1 baseline.
  - Candidate ablations:
      1. Path A only (w/o Pointwise MLP)
      2. Path B only (w/o Multi-Scale CNN)
      3. Embedding dimension: 64 vs 128 vs 256
      4. Decoder attention gate: spatial vs channel vs none
      5. Loss: uniform MSE vs thermocline-weighted vs smoothness penalty
  - Results are recorded as depth-wise RMSE differences relative to the baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class AblationTracker:
    """
    Tracks and tabulates ablation experiment outcomes.
    """

    def __init__(self, baseline_name: str = "phase1_baseline") -> None:
        self.baseline_name = baseline_name
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def register_experiment(
        self,
        exp_name: str,
        depth_rmse: Dict[int, float],
        description: str = "",
    ) -> None:
        """
        Registers an experiment's depth-wise RMSE results.
        """
        self.experiments[exp_name] = {
            "description": description,
            "depth_rmse": depth_rmse,
        }

    def summary_dataframe(self) -> pd.DataFrame:
        """
        Returns a DataFrame comparing depth-wise RMSE across all registered experiments.
        """
        records = []
        for name, data in self.experiments.items():
            row = {"experiment": name, "description": data["description"]}
            row.update({f"RMSE_{d}m": v for d, v in data["depth_rmse"].items()})
            records.append(row)
        return pd.DataFrame(records)

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves registered ablation outcomes to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.experiments, f, indent=2)
