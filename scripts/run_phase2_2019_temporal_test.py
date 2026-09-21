"""
scripts/run_phase2_2019_temporal_test.py
----------------------------------------
Final Pre-Specified Phase-2 Temporal Test Evaluation on 2019 Out-of-Sample Dataset.

Scientific & Integrity Rules:
1. Frozen Checkpoint: Evaluates ONLY checkpoints/phase2/best.pt (Epoch 95, 527,472 parameters).
   Strictly zero retraining, zero fine-tuning, zero hyperparameter adjustment.
2. Unbiased Evaluation: Evaluated after the 2018 validation decision. Zero model-selection feedback.
3. Dataset: 365 days of 2019 (data/processed/test/) dynamically assembling 16 channels.
4. Checksum Invariance: Confirms both Phase-1 and Phase-2 checkpoint SHA256 hashes are invariant.
5. Zero 2020+ Acquisition: Verifies 2020+ horizon remains strictly untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.metrics import compute_all_depth_metrics
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, create_dataloader
from utils.config import load_config


def run_phase2_test():
    print("=" * 85)
    print("OCEANEMBED FINAL PHASE-2 2019 TEMPORAL TEST EVALUATION")
    print("=" * 85)

    # 1. Guards and Checkpoint Verification
    p1_ckpt = PROJECT_ROOT / "checkpoints" / "phase1" / "best.pt"
    p2_ckpt = PROJECT_ROOT / "checkpoints" / "phase2" / "best.pt"

    expected_p1 = "f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b"
    expected_p2 = "8bd977fc87412d5209081a218745325f214e384cc821d6b951ec51fe50c34d8e"

    actual_p1 = hashlib.sha256(open(p1_ckpt, "rb").read()).hexdigest()
    actual_p2 = hashlib.sha256(open(p2_ckpt, "rb").read()).hexdigest()

    assert actual_p1 == expected_p1, f"Phase-1 checkpoint altered! Got {actual_p1}"
    assert actual_p2 == expected_p2, f"Phase-2 checkpoint altered! Got {actual_p2}"

    print(f"  [PASS] Phase-1 best.pt SHA256 verified invariant: {actual_p1}")
    print(f"  [PASS] Phase-2 best.pt SHA256 verified invariant: {actual_p2}")

    assert not (PROJECT_ROOT / "data" / "raw" / "2020").exists(), "CRITICAL: 2020+ data must not exist!"
    assert not (PROJECT_ROOT / "data" / "interim" / "2020").exists(), "CRITICAL: 2020+ data must not exist!"
    print("  [PASS] Confirmed 2020+ data has not been acquired.")

    # 2. Setup model and loader
    data_cfg = load_config("data")
    model_cfg = load_config("model_phase2")
    depths = list(data_cfg.depths_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OceanEmbedNet(model_cfg).to(device)
    ckpt = torch.load(p2_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  [PASS] Loaded Phase-2 checkpoint: Epoch {ckpt.get('epoch')}, Val Loss {ckpt.get('val_loss'):.4f} deg C^2.")

    test_ds = OceanEmbedDataset(split="test", in_channels=16, fallback_to_synthetic=False)
    assert len(test_ds) == 365, f"Expected 365 test days, got {len(test_ds)}"
    test_loader = create_dataloader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"  [PASS] Loaded 2019 test dataset: {len(test_ds)} calendar days.")

    # Spatial coordinates
    lat_coords = np.linspace(data_cfg.domain.lat_min, data_cfg.domain.lat_max, 101)
    lon_coords = np.linspace(data_cfg.domain.lon_min, data_cfg.domain.lon_max, 241)
    lat_2d = lat_coords[:, None]
    lon_2d = lon_coords[None, :]

    mask_as = (lat_2d >= 5.0) & (lat_2d <= 25.0) & (lon_2d >= 45.0) & (lon_2d <= 77.0)
    mask_bob = (lat_2d >= 5.0) & (lat_2d <= 22.0) & (lon_2d >= 80.0) & (lon_2d <= 100.0)

    # Inference loop
    all_preds = []
    all_targets = []
    all_masks = []
    all_dates = []

    print("\n[Evaluation] Running Phase-2 model inference on all 365 days of 2019...")
    start_time = time.time()

    with torch.no_grad():
        for x, y, meta in test_loader:
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
    print(f"[Evaluation] Inference completed in {inf_time:.2f}s ({inf_time / len(test_ds) * 1000:.1f} ms/day).")

    preds_arr = np.concatenate(all_preds, axis=0)      # [365, 15, 101, 241]
    targets_arr = np.concatenate(all_targets, axis=0)  # [365, 15, 101, 241]
    masks_arr = np.concatenate(all_masks, axis=0)      # [365, 15, 101, 241]

    # Compute 15-depth metrics on 2019
    depth_metrics = compute_all_depth_metrics(preds_arr, targets_arr, depths, mask=masks_arr)

    # Seasonal breakdown
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

    # Regional breakdown
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

    # Layers
    depths_50_150 = [50, 75, 100, 125, 150]
    depths_500_1000 = [500, 700, 1000]

    mean_rmse_50_150 = float(np.mean([depth_metrics["rmse"][d] for d in depths_50_150]))
    mean_rmse_500_1000 = float(np.mean([depth_metrics["rmse"][d] for d in depths_500_1000]))
    overall_mean_rmse = float(np.mean([depth_metrics["rmse"][d] for d in depths]))
    overall_mean_mae = float(np.mean([depth_metrics["mae"][d] for d in depths]))
    overall_mean_bias = float(np.mean([depth_metrics["bias"][d] for d in depths]))
    overall_mean_r = float(np.mean([depth_metrics["pearson_r"][d] for d in depths]))
    overall_mean_r2 = float(np.mean([depth_metrics["r2_score"][d] for d in depths]))

    # Load frozen Phase-1 2019 test metrics
    p1_report_path = PROJECT_ROOT / "evaluation" / "results" / "phase1_test2019_report.json"
    with open(p1_report_path, "r") as f:
        p1_report = json.load(f)

    p1_rmse = {int(k): float(v) for k, v in p1_report["oceanembed_2019_metrics"]["rmse"].items()}
    p1_mae = {int(k): float(v) for k, v in p1_report["oceanembed_2019_metrics"]["mae"].items()}
    p1_bias = {int(k): float(v) for k, v in p1_report["oceanembed_2019_metrics"]["bias"].items()}
    p1_r = {int(k): float(v) for k, v in p1_report["oceanembed_2019_metrics"]["pearson_r"].items()}
    p1_r2 = {int(k): float(v) for k, v in p1_report["oceanembed_2019_metrics"]["r2_score"].items()}

    p1_mean_rmse = float(np.mean([p1_rmse[d] for d in depths]))
    p1_mean_mae = float(np.mean([p1_mae[d] for d in depths]))
    p1_mean_bias = float(np.mean([p1_bias[d] for d in depths]))
    p1_mean_r = float(np.mean([p1_r[d] for d in depths]))
    p1_mean_r2 = float(np.mean([p1_r2[d] for d in depths]))
    p1_mean_50_150 = float(np.mean([p1_rmse[d] for d in depths_50_150]))
    p1_mean_500_1000 = float(np.mean([p1_rmse[d] for d in depths_500_1000]))

    delta_mean_rmse = overall_mean_rmse - p1_mean_rmse
    delta_mean_rmse_pct = (delta_mean_rmse / p1_mean_rmse) * 100.0

    # Load Phase-2 2018 validation metrics for 2018 vs 2019 stability comparison
    p2_val_report_path = PROJECT_ROOT / "evaluation" / "results" / "phase2_val2018_report.json"
    with open(p2_val_report_path, "r") as f:
        p2_val_data = json.load(f)

    p2_val_mean_rmse = p2_val_data["overall_metrics"]["mean_rmse"]
    p2_val_mean_r2 = p2_val_data["overall_metrics"]["mean_r2"]
    diff_val_test_rmse = overall_mean_rmse - p2_val_mean_rmse
    diff_val_test_rmse_pct = (diff_val_test_rmse / p2_val_mean_rmse) * 100.0

    print("\n" + "=" * 90)
    print("PHASE-1 vs. PHASE-2 FINAL 2019 TEMPORAL TEST COMPARISON")
    print("=" * 90)
    print(f"{'Depth (m)':<10} | {'Phase-1 RMSE':<14} | {'Phase-2 RMSE':<14} | {'Diff (deg C)':<14} | {'Diff (%)':<10} | {'P1 R^2':<10} | {'P2 R^2':<10}")
    print("-" * 90)

    for d in depths:
        r1 = p1_rmse[d]
        r2 = depth_metrics["rmse"][d]
        diff = r2 - r1
        diff_pct = (diff / r1) * 100.0
        r2_1 = p1_r2[d]
        r2_2 = depth_metrics["r2_score"][d]

        status = ""
        if diff_pct < -2.0:
            status = " [IMPROVED]"
        elif diff_pct > 5.0:
            status = " [DEGRADED]"

        print(f"{d:<10} | {r1:<14.4f} | {r2:<14.4f} | {diff:<+14.4f} | {diff_pct:<+9.2f}% | {r2_1:<10.4f} | {r2_2:<10.4f}{status}")

    print("-" * 90)
    print(f"{'MEAN':<10} | {p1_mean_rmse:<14.4f} | {overall_mean_rmse:<14.4f} | {delta_mean_rmse:<+14.4f} | {delta_mean_rmse_pct:<+9.2f}% | {p1_mean_r2:<10.4f} | {overall_mean_r2:<10.4f}")
    print(f"{'50-150m':<10} | {p1_mean_50_150:<14.4f} | {mean_rmse_50_150:<14.4f} | {mean_rmse_50_150 - p1_mean_50_150:<+14.4f} | {((mean_rmse_50_150 - p1_mean_50_150)/p1_mean_50_150)*100:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'500-1000m':<10} | {p1_mean_500_1000:<14.4f} | {mean_rmse_500_1000:<14.4f} | {mean_rmse_500_1000 - p1_mean_500_1000:<+14.4f} | {((mean_rmse_500_1000 - p1_mean_500_1000)/p1_mean_500_1000)*100:<+9.2f}% | {'-':<10} | {'-':<10}")

    print("\n" + "=" * 90)
    print("PHASE-2 TEMPORAL GENERALIZATION STABILITY: 2018 VAL vs. 2019 TEST")
    print("=" * 90)
    print(f"  2018 Val Mean RMSE: {p2_val_mean_rmse:.4f} °C | 2019 Test Mean RMSE: {overall_mean_rmse:.4f} °C | Diff: {diff_val_test_rmse:+.4f} °C ({diff_val_test_rmse_pct:+.2f}%)")
    print(f"  2018 Val Mean R^2:  {p2_val_mean_r2:.4f}   | 2019 Test Mean R^2:  {overall_mean_r2:.4f}   | Diff: {overall_mean_r2 - p2_val_mean_r2:+.4f}")

    # Build report dict
    report = {
        "evaluation_scope": "Phase-2 Final 2019 Temporal Test",
        "checkpoint_used": str(p2_ckpt),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_loss_2018": ckpt.get("val_loss"),
        "checkpoint_sha256": actual_p2,
        "phase1_checkpoint_sha256": actual_p1,
        "test_year": 2019,
        "num_days": len(test_ds),
        "oceanembed_phase2_2019_metrics": {
            "mean_rmse": overall_mean_rmse,
            "mean_mae": overall_mean_mae,
            "mean_bias": overall_mean_bias,
            "mean_pearson_r": overall_mean_r,
            "mean_r2": overall_mean_r2,
            "mean_rmse_50_150m": mean_rmse_50_150,
            "mean_rmse_500_1000m": mean_rmse_500_1000,
            "rmse": {d: float(depth_metrics["rmse"][d]) for d in depths},
            "mae": {d: float(depth_metrics["mae"][d]) for d in depths},
            "bias": {d: float(depth_metrics["bias"][d]) for d in depths},
            "pearson_r": {d: float(depth_metrics["pearson_r"][d]) for d in depths},
            "r2_score": {d: float(depth_metrics["r2_score"][d]) for d in depths},
        },
        "phase1_comparison_2019": {
            "phase1_mean_rmse": p1_mean_rmse,
            "phase1_mean_r2": p1_mean_r2,
            "delta_mean_rmse": delta_mean_rmse,
            "delta_mean_rmse_pct": delta_mean_rmse_pct,
            "delta_mean_r2": overall_mean_r2 - p1_mean_r2,
            "delta_50_150m_rmse": mean_rmse_50_150 - p1_mean_50_150,
            "delta_500_1000m_rmse": mean_rmse_500_1000 - p1_mean_500_1000,
        },
        "phase2_temporal_stability": {
            "2018_val_mean_rmse": p2_val_mean_rmse,
            "2019_test_mean_rmse": overall_mean_rmse,
            "diff_rmse": diff_val_test_rmse,
            "diff_rmse_pct": diff_val_test_rmse_pct,
            "2018_val_mean_r2": p2_val_mean_r2,
            "2019_test_mean_r2": overall_mean_r2,
            "diff_r2": overall_mean_r2 - p2_val_mean_r2,
        },
        "depth_wise_comparison_2019": {
            str(d): {
                "phase2_rmse": float(depth_metrics["rmse"][d]),
                "phase1_rmse": float(p1_rmse[d]),
                "diff_rmse": float(depth_metrics["rmse"][d] - p1_rmse[d]),
                "diff_rmse_pct": float(((depth_metrics["rmse"][d] - p1_rmse[d]) / p1_rmse[d]) * 100.0),
                "phase2_mae": float(depth_metrics["mae"][d]),
                "phase1_mae": float(p1_mae[d]),
                "phase2_bias": float(depth_metrics["bias"][d]),
                "phase1_bias": float(p1_bias[d]),
                "phase2_pearson_r": float(depth_metrics["pearson_r"][d]),
                "phase1_pearson_r": float(p1_r[d]),
                "phase2_r2": float(depth_metrics["r2_score"][d]),
                "phase1_r2": float(p1_r2[d]),
            }
            for d in depths
        },
        "seasonal_metrics_2019": seasonal_metrics,
        "regional_metrics_2019": regional_metrics,
    }

    out_report_path = PROJECT_ROOT / "evaluation" / "results" / "phase2_test2019_report.json"
    with open(out_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Evaluation] Saved 2019 test report to {out_report_path}")

    # Generate Figures
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: 15-Depth Comparison on 2019
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    p1_vals = [p1_rmse[d] for d in depths]
    p2_vals = [depth_metrics["rmse"][d] for d in depths]
    ax.plot(p1_vals, depths, "o-", color="#1f77b4", label=f"Phase-1 Baseline (Mean: {p1_mean_rmse:.4f} °C)", linewidth=2)
    ax.plot(p2_vals, depths, "s-", color="#e74c3c", label=f"Phase-2 (16 ch) (Mean: {overall_mean_rmse:.4f} °C)", linewidth=2)
    ax.set_ylim(1050, -50)
    ax.set_xlabel("2019 Test RMSE (°C)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title("OceanEmbed 2019 Temporal Test: Phase-1 vs. Phase-2 Depth RMSE", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig1_path = figures_dir / "phase2_vs_phase1_test2019_depth_rmse.png"
    fig.savefig(fig1_path)
    plt.close(fig)
    print(f"[Figure] Saved {fig1_path}")

    # Figure 2: Temporal Generalization (2018 Val vs 2019 Test for Phase-2)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    p2_val_depths = [p2_val_data["depth_wise_metrics"][str(d)]["phase2_rmse"] for d in depths]
    ax.plot(p2_val_depths, depths, "o-", color="#27ae60", label=f"2018 Validation (Mean: {p2_val_mean_rmse:.4f} °C)", linewidth=2)
    ax.plot(p2_vals, depths, "s-", color="#8e44ad", label=f"2019 Out-of-Sample Test (Mean: {overall_mean_rmse:.4f} °C)", linewidth=2)
    ax.set_ylim(1050, -50)
    ax.set_xlabel("RMSE (°C)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title("Phase-2 Temporal Generalization Stability: 2018 Val vs. 2019 Test", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig2_path = figures_dir / "phase2_temporal_generalization_2018_vs_2019.png"
    fig.savefig(fig2_path)
    plt.close(fig)
    print(f"[Figure] Saved {fig2_path}")

    # Figure 3: Subsurface Thermocline 2019
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sub_depths = [30, 50, 75, 100, 125, 150, 200]
    p1_sub = [p1_rmse[d] for d in sub_depths]
    p2_sub = [depth_metrics["rmse"][d] for d in sub_depths]
    x_indices = np.arange(len(sub_depths))
    width = 0.35
    ax.bar(x_indices - width/2, p1_sub, width, label=f"Phase-1 (Mean: {p1_mean_50_150:.4f}°C)", color="#2980b9")
    ax.bar(x_indices + width/2, p2_sub, width, label=f"Phase-2 (Mean: {mean_rmse_50_150:.4f}°C)", color="#e67e22")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{d}m" for d in sub_depths])
    ax.set_ylabel("2019 RMSE (°C)", fontsize=12)
    ax.set_title("2019 Subsurface Thermocline Layer Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend()
    fig.tight_layout()
    fig3_path = figures_dir / "phase2_vs_phase1_test2019_subsurface.png"
    fig.savefig(fig3_path)
    plt.close(fig)
    print(f"[Figure] Saved {fig3_path}")

    # Figure 4: Seasonal Comparison on 2019
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    seasons = ["DJF", "MAM", "JJAS", "OND"]
    p2_seas = [seasonal_metrics[s]["mean_rmse"] for s in seasons]
    p1_seas_dict = p1_report.get("seasonal_2019_metrics", p1_report.get("seasonal_metrics", {}))
    p1_seas = [np.mean(list(p1_seas_dict[s]["rmse"].values())) for s in seasons]

    x_s = np.arange(len(seasons))
    ax.bar(x_s - width/2, p1_seas, width, label="Phase-1 Baseline", color="#34495e")
    ax.bar(x_s + width/2, p2_seas, width, label="Phase-2 (16 ch)", color="#d35400")
    ax.set_xticks(x_s)
    ax.set_xticklabels(seasons)
    ax.set_ylabel("2019 Seasonal Mean RMSE (°C)", fontsize=12)
    ax.set_title("Seasonal Temporal Test Performance (2019)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend()
    fig.tight_layout()
    fig4_path = figures_dir / "phase2_vs_phase1_test2019_seasonal.png"
    fig.savefig(fig4_path)
    plt.close(fig)
    print(f"[Figure] Saved {fig4_path}")

    print("\n[Evaluation] All Phase-2 2019 test metrics and figures generated successfully.")
    return report


if __name__ == "__main__":
    run_phase2_test()
