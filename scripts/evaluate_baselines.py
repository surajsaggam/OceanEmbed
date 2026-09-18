"""
scripts/evaluate_baselines.py
------------------------------
Fits and evaluates Climatology and Ridge baselines on real January 2020 data.
Reports depth-wise RMSE, MAE, bias, Pearson r, and R^2.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from baselines.climatology import ClimatologyBaseline
from baselines.ridge import ChunkedRidgeBaseline
from evaluation.metrics import compute_all_depth_metrics
from pipeline.datasets import OceanEmbedDataset, create_dataloader
from utils.config import load_config


def main() -> None:
    print("=" * 75)
    print("OceanEmbed — Baseline Training & Evaluation (January 2020 Dry Run)")
    print("=" * 75)

    data_cfg = load_config("data")
    depths = list(data_cfg.depths_m)

    # 1. Load Datasets
    train_ds = OceanEmbedDataset(split="train", fallback_to_synthetic=False)
    val_ds = OceanEmbedDataset(split="val", fallback_to_synthetic=False)
    test_ds = OceanEmbedDataset(split="test", fallback_to_synthetic=False)

    print(f"Loaded datasets: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # 2. Climatology Baseline
    print("\n--- Training Climatology Baseline ---")
    clim = ClimatologyBaseline(n_depths=len(depths), grid_shape=(101, 241))
    for i in range(len(train_ds)):
        x, y, meta = train_ds[i]
        mask = meta.get("target_mask", None)
        mask_np = mask.numpy() if mask is not None else None
        # Month 1 = January
        clim.update(month=1, target=y.numpy(), mask=mask_np)
    clim.finalize()
    clim_path = Path("checkpoints/baselines/climatology_jan2020.npz")
    clim.save(clim_path)
    print(f"Climatology baseline fitted and saved to {clim_path}")

    # 3. Ridge Baseline
    print("\n--- Training Chunked Ridge Baseline (O(1) RAM) ---")
    ridge = ChunkedRidgeBaseline(in_features=14, out_features=15, alpha=1.0)
    for i in range(len(train_ds)):
        x, y, meta = train_ds[i]
        mask = meta.get("target_mask", None)
        # Use surface ocean validity mask or target ocean validity mask
        ocean_mask = (mask[0].numpy() == 1.0) if mask is not None else None
        ridge.update_chunk(x.numpy(), y.numpy(), valid_mask=ocean_mask)
    ridge.finalize()
    ridge_path = Path("checkpoints/baselines/ridge_jan2020.npz")
    ridge.save(ridge_path)
    print(f"Ridge baseline fitted on {ridge.n_samples} pixel-samples and saved to {ridge_path}")

    # 4. Evaluate both baselines on Test Split
    print("\n--- Evaluating on Test Split (Jan 27 - Jan 31) ---")
    test_targets = []
    test_masks = []
    clim_preds = []
    ridge_preds = []

    for i in range(len(test_ds)):
        x, y, meta = test_ds[i]
        mask = meta.get("target_mask", None)
        test_targets.append(y.numpy())
        test_masks.append(mask.numpy() if mask is not None else np.ones_like(y.numpy()))

        # Climatology prediction for January
        c_pred = clim.predict(month=1)
        clim_preds.append(c_pred)

        # Ridge prediction
        r_pred = ridge.predict(x.numpy())
        ridge_preds.append(r_pred)

    targets_arr = np.stack(test_targets, axis=0)  # [N, 15, H, W]
    masks_arr = np.stack(test_masks, axis=0)      # [N, 15, H, W]
    clim_arr = np.stack(clim_preds, axis=0)        # [N, 15, H, W]
    ridge_arr = np.stack(ridge_preds, axis=0)      # [N, 15, H, W]

    clim_metrics = compute_all_depth_metrics(clim_arr, targets_arr, depths, mask=masks_arr)
    ridge_metrics = compute_all_depth_metrics(ridge_arr, targets_arr, depths, mask=masks_arr)

    # Print comparative table
    print(f"\n{'Depth (m)':<10} | {'Clim RMSE (°C)':<15} | {'Ridge RMSE (°C)':<15} | {'Clim R²':<10} | {'Ridge R²':<10}")
    print("-" * 75)
    for d in depths:
        c_rmse = clim_metrics["rmse"][d]
        r_rmse = ridge_metrics["rmse"][d]
        c_r2 = clim_metrics["r2_score"][d]
        r_r2 = ridge_metrics["r2_score"][d]
        print(f"{d:<10.0f} | {c_rmse:<15.4f} | {r_rmse:<15.4f} | {c_r2:<10.4f} | {r_r2:<10.4f}")

    # Save metrics JSON
    results = {
        "depths_m": depths,
        "climatology": clim_metrics,
        "ridge": ridge_metrics,
    }
    out_json = Path("evaluation/results/baseline_results_jan2020.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline evaluation results to {out_json}")
    print("=" * 75)


if __name__ == "__main__":
    main()
