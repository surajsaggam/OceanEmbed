"""
evaluation/glorys_eval.py
-------------------------
Evaluation against GLORYS12V1 reanalysis target.

Scientific framing:
  GLORYS is a data-assimilative reanalysis product used as the training and reference
  target. Metrics against GLORYS measure reconstruction fidelity across the 15 standard
  depth levels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from evaluation.metrics import compute_all_depth_metrics, depth_wise_rmse
from pipeline.regrid import compute_target_coords
from utils.config import load_config


class GlorysEvaluator:
    """
    Evaluator computing depth-wise metrics, seasonal breakdowns, and regional analyses
    against GLORYS reanalysis target fields.
    """

    def __init__(self, eval_cfg: Optional[Any] = None, data_cfg: Optional[Any] = None) -> None:
        self.eval_cfg = eval_cfg or load_config("eval")
        self.data_cfg = data_cfg or load_config("data")
        self.depths = list(self.eval_cfg.depths_m)
        self.lat, self.lon = compute_target_coords(
            self.data_cfg.domain.lat_min,
            self.data_cfg.domain.lat_max,
            self.data_cfg.domain.lon_min,
            self.data_cfg.domain.lon_max,
            self.data_cfg.domain.resolution,
        )

    def evaluate_dataset(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: Union[str, torch.device] = "cpu",
    ) -> Dict[str, Any]:
        """
        Runs model over dataloader and calculates depth-wise metrics against GLORYS.

        Returns:
            Dictionary containing depth metrics and sample count.
        """
        model.eval()
        model.to(device)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[0], batch[1]
                else:
                    x = batch["input"]
                    y = batch["target"]

                x = x.to(device)
                out = model(x)
                pred = out["temperature"] if isinstance(out, dict) else out

                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.numpy() if isinstance(y, torch.Tensor) else y)

        preds_arr = np.concatenate(all_preds, axis=0)      # [N, 15, H, W]
        targets_arr = np.concatenate(all_targets, axis=0)  # [N, 15, H, W]

        # Overall depth metrics
        metrics = compute_all_depth_metrics(preds_arr, targets_arr, self.depths)

        # Spatial RMSE maps [15, H, W]
        spatial_rmse = np.sqrt(np.mean((preds_arr - targets_arr) ** 2, axis=0))

        results = {
            "num_samples": int(preds_arr.shape[0]),
            "depths_m": self.depths,
            "metrics": metrics,
            "spatial_rmse_mean": {d: float(np.mean(spatial_rmse[i])) for i, d in enumerate(self.depths)},
        }
        return results

    def save_results(self, results: Dict[str, Any], output_path: Union[str, Path]) -> None:
        """
        Saves evaluation metrics dictionary to JSON.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
