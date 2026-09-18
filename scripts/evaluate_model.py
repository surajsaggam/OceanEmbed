"""
scripts/evaluate_model.py
--------------------------
Evaluates a trained OceanEmbed checkpoint against GLORYS reanalysis on test/val split.
Computes depth-wise RMSE, MAE, bias, Pearson r, and R^2 score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from evaluation.metrics import compute_all_depth_metrics
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, create_dataloader
from utils.config import load_config


def evaluate_checkpoint(
    checkpoint_path: str = "checkpoints/phase1/best.pt",
    split: str = "test",
    output_json: str = "evaluation/results/model_results_jan2020.json",
) -> dict:
    print("=" * 75)
    print(f"OceanEmbed — Evaluating Checkpoint: {checkpoint_path} on '{split}' split")
    print("=" * 75)

    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}")

    data_cfg = load_config("data")
    model_cfg = load_config("model")
    depths = list(data_cfg.depths_m)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    # Load dataset
    ds = OceanEmbedDataset(split=split, fallback_to_synthetic=False)
    loader = create_dataloader(ds, batch_size=1, shuffle=False)
    print(f"Loaded {len(ds)} samples for '{split}' split.")

    # Load model
    model = OceanEmbedNet(model_cfg).to(device)
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            out = model(x)
            pred = out["temperature"].cpu().numpy()
            target = y.numpy()
            mask = meta["target_mask"].numpy() if "target_mask" in meta else np.ones_like(target)

            all_preds.append(pred)
            all_targets.append(target)
            all_masks.append(mask)

    preds_arr = np.concatenate(all_preds, axis=0)      # [N, 15, H, W]
    targets_arr = np.concatenate(all_targets, axis=0)  # [N, 15, H, W]
    masks_arr = np.concatenate(all_masks, axis=0)      # [N, 15, H, W]

    metrics = compute_all_depth_metrics(preds_arr, targets_arr, depths, mask=masks_arr)

    print(f"\n{'Depth (m)':<10} | {'RMSE (°C)':<12} | {'MAE (°C)':<12} | {'Bias (°C)':<12} | {'Pearson r':<12} | {'R²':<10}")
    print("-" * 75)
    for d in depths:
        rmse = metrics["rmse"][d]
        mae = metrics["mae"][d]
        bias = metrics["bias"][d]
        r = metrics["pearson_r"][d]
        r2 = metrics["r2_score"][d]
        print(f"{d:<10.0f} | {rmse:<12.4f} | {mae:<12.4f} | {bias:<12.4f} | {r:<12.4f} | {r2:<10.4f}")

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": str(ckpt_file),
        "split": split,
        "num_samples": len(ds),
        "depths_m": depths,
        "metrics": metrics,
    }
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation metrics to {out_p}")
    print("=" * 75)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate OceanEmbed Checkpoint")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/phase1/best.pt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default="evaluation/results/model_results_jan2020.json")
    args = parser.parse_args()

    evaluate_checkpoint(checkpoint_path=args.checkpoint, split=args.split, output_json=args.output)
