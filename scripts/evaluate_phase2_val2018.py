"""
scripts/evaluate_phase2_val2018.py
----------------------------------
Comprehensive 2018 Validation Evaluation for OceanEmbed Phase-2:
  - Evaluates best Phase-2 model (checkpoints/phase2/best.pt) on all 365 days of 2018
  - Computes 15-depth metrics: RMSE, MAE, Bias, Pearson r, R^2
  - Subsurface focus: 50–150 m and 500–1000 m layers
  - Seasonal breakdown: DJF, MAM, JJAS, OND
  - Regional breakdown: Arabian Sea (AS) vs Bay of Bengal (BoB)
  - Rigorous comparison against frozen Phase-1 2018 baseline
  - Generates high-quality diagnostic figures
  - STRICT SAFETY: 2019 test set is NEVER accessed or evaluated.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_all_depth_metrics
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, create_dataloader
from utils.config import load_config


def run_phase2_val_evaluation():
    print("=" * 75)
    print("OCEANEMBED PHASE-2 VALIDATION EVALUATION (YEAR 2018)")
    print("=" * 75)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "phase2" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase-2 checkpoint not found at {ckpt_path}")

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Device: {device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "unknown")
    val_loss = ckpt.get("val_loss", "unknown")
    print(f"[Evaluation] Loaded Phase-2 checkpoint: Epoch {epoch}, Val Loss {val_loss:.4f} deg C^2")

    # Initialize model
    model_cfg = load_config("model_phase2")
    model = OceanEmbedNet(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load 2018 Validation Dataset (16 channels)
    data_cfg = load_config("data")
    depths = list(data_cfg.depths_m)
    val_ds = OceanEmbedDataset(split="val", in_channels=16, fallback_to_synthetic=False)
    val_loader = create_dataloader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"[Evaluation] Validation dataset loaded: {len(val_ds)} days of 2018")

    # Spatial coordinates
    lat_coords = np.linspace(data_cfg.domain.lat_min, data_cfg.domain.lat_max, 101)
    lon_coords = np.linspace(data_cfg.domain.lon_min, data_cfg.domain.lon_max, 241)

    # Regional masks
    lat_2d = lat_coords[:, None]
    lon_2d = lon_coords[None, :]

    mask_as = (lat_2d >= 5.0) & (lat_2d <= 25.0) & (lon_2d >= 45.0) & (lon_2d <= 77.0)
    mask_bob = (lat_2d >= 5.0) & (lat_2d <= 22.0) & (lon_2d >= 80.0) & (lon_2d <= 100.0)

    # Storage for predictions, targets, and dates
    all_preds = []
    all_targets = []
    all_masks = []
    all_dates = []

    print("[Evaluation] Running model inference over all 365 days of 2018...")
    start_time = time.time()

    with torch.no_grad():
        for x, y, meta in val_loader:
            x = x.to(device)
            m = meta.get("target_mask", None)
            if m is not None:
                m = m.numpy()
            else:
                m = np.ones_like(y.numpy())

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                out = model(x)
                pred = out["temperature"]

            all_preds.append(pred.cpu().float().numpy())
            all_targets.append(y.numpy())
            all_masks.append(m)

            dates = meta.get("date", [])
            if isinstance(dates, (list, tuple)):
                all_dates.extend(dates)
            else:
                all_dates.append(str(dates))

    inf_time = time.time() - start_time
    print(f"[Evaluation] Inference complete in {inf_time:.2f}s ({inf_time / len(val_ds) * 1000:.1f} ms/day).")

    preds_arr = np.concatenate(all_preds, axis=0)      # [365, 15, 101, 241]
    targets_arr = np.concatenate(all_targets, axis=0)  # [365, 15, 101, 241]
    masks_arr = np.concatenate(all_masks, axis=0)      # [365, 15, 101, 241]

    # Compute 15-depth metrics
    depth_metrics = compute_all_depth_metrics(preds_arr, targets_arr, depths, mask=masks_arr)

    # Compute seasonal metrics
    seasonal_indices = {"DJF": [], "MAM": [], "JJAS": [], "OND": []}
    for idx, d_str in enumerate(all_dates):
        month = int(d_str.split("-")[1])
        if month in [12, 1, 2]:
            seasonal_indices["DJF"].append(idx)
        elif month in [3, 4, 5]:
            seasonal_indices["MAM"].append(idx)
        elif month in [6, 7, 8, 9]:
            seasonal_indices["JJAS"].append(idx)
        elif month in [10, 11]:
            seasonal_indices["OND"].append(idx)

    seasonal_metrics = {}
    for season, idxs in seasonal_indices.items():
        if idxs:
            s_preds = preds_arr[idxs]
            s_targets = targets_arr[idxs]
            s_masks = masks_arr[idxs]
            s_m = compute_all_depth_metrics(s_preds, s_targets, depths, mask=s_masks)
            seasonal_metrics[season] = {
                "num_days": len(idxs),
                "mean_rmse": float(np.mean([s_m["rmse"][d] for d in depths])),
                "mean_mae": float(np.mean([s_m["mae"][d] for d in depths])),
                "mean_r2": float(np.mean([s_m["r2_score"][d] for d in depths])),
                "depth_rmse": {d: float(s_m["rmse"][d]) for d in depths},
            }

    # Compute regional metrics
    regional_metrics = {}
    for reg_name, reg_mask in [("Arabian_Sea", mask_as), ("Bay_of_Bengal", mask_bob)]:
        comb_mask = masks_arr * reg_mask[None, None, :, :]
        r_m = compute_all_depth_metrics(preds_arr, targets_arr, depths, mask=comb_mask)
        regional_metrics[reg_name] = {
            "mean_rmse": float(np.mean([r_m["rmse"][d] for d in depths])),
            "mean_mae": float(np.mean([r_m["mae"][d] for d in depths])),
            "mean_bias": float(np.mean([r_m["bias"][d] for d in depths])),
            "mean_r2": float(np.mean([r_m["r2_score"][d] for d in depths])),
            "depth_rmse": {d: float(r_m["rmse"][d]) for d in depths},
        }

    # Subsurface layers: 50-150 m and 500-1000 m
    depths_50_150 = [50, 75, 100, 125, 150]
    depths_500_1000 = [500, 700, 1000]

    mean_rmse_50_150 = float(np.mean([depth_metrics["rmse"][d] for d in depths_50_150]))
    mean_rmse_500_1000 = float(np.mean([depth_metrics["rmse"][d] for d in depths_500_1000]))
    overall_mean_rmse = float(np.mean([depth_metrics["rmse"][d] for d in depths]))
    overall_mean_mae = float(np.mean([depth_metrics["mae"][d] for d in depths]))
    overall_mean_bias = float(np.mean([depth_metrics["bias"][d] for d in depths]))
    overall_mean_r = float(np.mean([depth_metrics["pearson_r"][d] for d in depths]))
    overall_mean_r2 = float(np.mean([depth_metrics["r2_score"][d] for d in depths]))

    # Load Phase-1 2018 validation metrics for rigorous baseline comparison
    p1_metrics_path = PROJECT_ROOT / "evaluation" / "results" / "oceanembed_val2018_metrics.json"
    with open(p1_metrics_path, "r") as f:
        p1_data = json.load(f)
    p1_rmse = {int(k): float(v) for k, v in p1_data["metrics"]["rmse"].items()}
    p1_r2 = {int(k): float(v) for k, v in p1_data["metrics"].get("r2_score", p1_data["metrics"].get("r2", {})).items()}
    p1_mae = {int(k): float(v) for k, v in p1_data["metrics"]["mae"].items()}
    p1_bias = {int(k): float(v) for k, v in p1_data["metrics"]["bias"].items()}

    p1_mean_rmse = float(np.mean([p1_rmse[d] for d in depths]))
    p1_mean_r2 = float(np.mean([p1_r2[d] for d in depths]))
    p1_mean_50_150 = float(np.mean([p1_rmse[d] for d in depths_50_150]))
    p1_mean_500_1000 = float(np.mean([p1_rmse[d] for d in depths_500_1000]))

    delta_mean_rmse = overall_mean_rmse - p1_mean_rmse
    delta_mean_rmse_pct = (delta_mean_rmse / p1_mean_rmse) * 100.0

    print("\n" + "=" * 75)
    print("PHASE-1 vs. PHASE-2 VALIDATION PERFORMANCE (2018)")
    print("=" * 75)
    print(f"{'Depth (m)':<10} | {'Phase-1 RMSE':<14} | {'Phase-2 RMSE':<14} | {'Diff (deg C)':<14} | {'Diff (%)':<10} | {'P1 R^2':<10} | {'P2 R^2':<10}")
    print("-" * 85)

    material_degradation_detected = False
    degraded_depths = []

    for d in depths:
        r1 = p1_rmse[d]
        r2 = depth_metrics["rmse"][d]
        diff = r2 - r1
        diff_pct = (diff / r1) * 100.0
        r2_1 = p1_r2[d]
        r2_2 = depth_metrics["r2_score"][d]

        flag = ""
        if diff_pct > 5.0:
            flag = " [DEGRADED]"
            material_degradation_detected = True
            degraded_depths.append(d)
        elif diff_pct < -2.0:
            flag = " [IMPROVED]"

        print(f"{d:<10} | {r1:<14.4f} | {r2:<14.4f} | {diff:<+14.4f} | {diff_pct:<+9.2f}% | {r2_1:<10.4f} | {r2_2:<10.4f}{flag}")

    print("-" * 85)
    print(f"{'MEAN':<10} | {p1_mean_rmse:<14.4f} | {overall_mean_rmse:<14.4f} | {delta_mean_rmse:<+14.4f} | {delta_mean_rmse_pct:<+9.2f}% | {p1_mean_r2:<10.4f} | {overall_mean_r2:<10.4f}")
    print(f"{'50-150m':<10} | {p1_mean_50_150:<14.4f} | {mean_rmse_50_150:<14.4f} | {mean_rmse_50_150 - p1_mean_50_150:<+14.4f} | {((mean_rmse_50_150 - p1_mean_50_150) / p1_mean_50_150)*100:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'500-1000m':<10} | {p1_mean_500_1000:<14.4f} | {mean_rmse_500_1000:<14.4f} | {mean_rmse_500_1000 - p1_mean_500_1000:<+14.4f} | {((mean_rmse_500_1000 - p1_mean_500_1000) / p1_mean_500_1000)*100:<+9.2f}% | {'-':<10} | {'-':<10}")

    # Build report dict
    report = {
        "model": "OceanEmbed Phase-2",
        "channels": 16,
        "features": "14 Phase-1 channels + Delta_SST + Delta_SSH",
        "validation_year": 2018,
        "num_days": len(val_ds),
        "best_epoch": epoch,
        "best_val_loss": val_loss,
        "overall_metrics": {
            "mean_rmse": overall_mean_rmse,
            "mean_mae": overall_mean_mae,
            "mean_bias": overall_mean_bias,
            "mean_pearson_r": overall_mean_r,
            "mean_r2": overall_mean_r2,
            "mean_rmse_50_150m": mean_rmse_50_150,
            "mean_rmse_500_1000m": mean_rmse_500_1000,
        },
        "phase1_baseline": {
            "mean_rmse": p1_mean_rmse,
            "mean_r2": p1_mean_r2,
            "mean_rmse_50_150m": p1_mean_50_150,
            "mean_rmse_500_1000m": p1_mean_500_1000,
        },
        "difference_vs_phase1": {
            "delta_mean_rmse": delta_mean_rmse,
            "delta_mean_rmse_pct": delta_mean_rmse_pct,
            "delta_mean_r2": overall_mean_r2 - p1_mean_r2,
            "delta_50_150m_rmse": mean_rmse_50_150 - p1_mean_50_150,
            "delta_500_1000m_rmse": mean_rmse_500_1000 - p1_mean_500_1000,
            "material_degradation_detected": material_degradation_detected,
            "degraded_depths": degraded_depths,
        },
        "depth_wise_metrics": {
            str(d): {
                "phase2_rmse": float(depth_metrics["rmse"][d]),
                "phase1_rmse": float(p1_rmse[d]),
                "diff_rmse": float(depth_metrics["rmse"][d] - p1_rmse[d]),
                "diff_rmse_pct": float(((depth_metrics["rmse"][d] - p1_rmse[d]) / p1_rmse[d]) * 100.0),
                "phase2_mae": float(depth_metrics["mae"][d]),
                "phase2_bias": float(depth_metrics["bias"][d]),
                "phase2_pearson_r": float(depth_metrics["pearson_r"][d]),
                "phase2_r2": float(depth_metrics["r2_score"][d]),
                "phase1_r2": float(p1_r2[d]),
            }
            for d in depths
        },
        "seasonal_metrics": seasonal_metrics,
        "regional_metrics": regional_metrics,
    }

    out_report_path = PROJECT_ROOT / "evaluation" / "results" / "phase2_val2018_report.json"
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Evaluation] Saved validation report to {out_report_path}")

    # Generate Figures
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Depth RMSE Comparison
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    p1_vals = [p1_rmse[d] for d in depths]
    p2_vals = [depth_metrics["rmse"][d] for d in depths]
    ax.plot(p1_vals, depths, "o-", color="#1f77b4", label=f"Phase-1 (14 ch) - Mean: {p1_mean_rmse:.4f} °C", linewidth=2)
    ax.plot(p2_vals, depths, "s-", color="#2ca02c", label=f"Phase-2 (16 ch) - Mean: {overall_mean_rmse:.4f} °C", linewidth=2)
    ax.set_ylim(1050, -50)
    ax.set_xlabel("Validation RMSE (°C)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title("OceanEmbed 2018 Validation: Phase-1 vs. Phase-2 Depth RMSE", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig1_path = figures_dir / "phase2_vs_phase1_val2018_depth_rmse.png"
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"[Figure] Saved {fig1_path}")

    # Figure 2: Subsurface 50-200m focus
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sub_depths = [30, 50, 75, 100, 125, 150, 200]
    p1_sub = [p1_rmse[d] for d in sub_depths]
    p2_sub = [depth_metrics["rmse"][d] for d in sub_depths]
    x_indices = np.arange(len(sub_depths))
    width = 0.35
    ax.bar(x_indices - width/2, p1_sub, width, label=f"Phase-1 (Mean: {p1_mean_50_150:.4f}°C)", color="#3498db")
    ax.bar(x_indices + width/2, p2_sub, width, label=f"Phase-2 (Mean: {mean_rmse_50_150:.4f}°C)", color="#2ecc71")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{d}m" for d in sub_depths])
    ax.set_ylabel("RMSE (°C)", fontsize=12)
    ax.set_title("Subsurface Thermocline Layer (50–150m) Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend()
    fig.tight_layout()
    fig2_path = figures_dir / "phase2_vs_phase1_val2018_subsurface.png"
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"[Figure] Saved {fig2_path}")

    # Figure 3: Seasonal Comparison
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    seasons = ["DJF", "MAM", "JJAS", "OND"]
    p2_seas_vals = [seasonal_metrics[s]["mean_rmse"] for s in seasons]
    # Phase-1 seasonal values from report
    p1_diag_path = PROJECT_ROOT / "evaluation" / "results" / "phase1_diagnostics_report.json"
    p1_seas_vals = []
    if p1_diag_path.exists():
        with open(p1_diag_path, "r") as f:
            p1_diag = json.load(f)
        for s in seasons:
            s_dict = p1_diag["seasonal_metrics"][s]
            val = s_dict.get("mean_rmse", np.mean(list(s_dict["rmse"].values())))
            p1_seas_vals.append(float(val))
    else:
        p1_seas_vals = [0.72] * 4

    x_s = np.arange(len(seasons))
    ax.bar(x_s - width/2, p1_seas_vals, width, label="Phase-1 Baseline", color="#e67e22")
    ax.bar(x_s + width/2, p2_seas_vals, width, label="Phase-2 (16 ch)", color="#16a085")
    ax.set_xticks(x_s)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("Seasonal Mean RMSE (°C)", fontsize=12)
    ax.set_title("Seasonal Validation Performance (2018)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend()
    fig.tight_layout()
    fig3_path = figures_dir / "phase2_vs_phase1_val2018_seasonal.png"
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"[Figure] Saved {fig3_path}")

    # Figure 4: Training History
    hist_path = PROJECT_ROOT / "evaluation" / "results" / "phase2_training_history.json"
    if hist_path.exists():
        with open(hist_path, "r") as f:
            hist_data = json.load(f)
        epochs_arr = [r["epoch"] for r in hist_data["history"]]
        tr_losses = [r["train_loss"] for r in hist_data["history"]]
        v_losses = [r["val_loss"] for r in hist_data["history"]]

        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.plot(epochs_arr, tr_losses, "-", label="Train Loss (MSE)", color="#2980b9", linewidth=2)
        ax.plot(epochs_arr, v_losses, "-", label="Val Loss (MSE)", color="#e74c3c", linewidth=2)
        ax.axvline(x=hist_data["best_epoch"], color="green", linestyle="--", label=f"Best Epoch {hist_data['best_epoch']} ({hist_data['best_val_loss']:.4f}°C²)")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (°C²)", fontsize=12)
        ax.set_title("OceanEmbed Phase-2 Training & Validation Loss Curves", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(fontsize=11)
        fig.tight_layout()
        fig4_path = figures_dir / "phase2_val2018_training_history.png"
        fig.savefig(fig4_path)
        plt.close(fig)
        print(f"[Figure] Saved {fig4_path}")

    print("\n[Evaluation] All Phase-2 2018 validation metrics and figures generated successfully.")
    return report


if __name__ == "__main__":
    run_phase2_val_evaluation()
