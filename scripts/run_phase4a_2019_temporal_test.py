"""
scripts/run_phase4a_2019_temporal_test.py
-----------------------------------------
Final Locked Phase-4A Temporal Test Evaluation on 2019 Out-of-Sample Dataset.

Scientific & Integrity Rules:
1. Frozen Checkpoint: Evaluates ONLY checkpoints/phase4a/best.pt (Epoch 76, 525,040 parameters).
   Strictly zero retraining, zero fine-tuning, zero hyperparameter adjustment.
2. Unbiased Evaluation: Evaluated after the 2018 validation decision. Zero model-selection feedback.
3. Dataset: 365 days of 2019 (data/processed/test/) using 14 Phase-1 channels.
4. Checksum Invariance: Confirms Phase-1, Phase-2, and Phase-4A checkpoint SHA256 hashes are invariant.
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


def run_phase4a_test():
    print("=" * 85)
    print("OCEANEMBED FINAL PHASE-4A 2019 TEMPORAL TEST EVALUATION")
    print("=" * 85)

    # 1. Guards and Checkpoint Verification
    p1_ckpt = PROJECT_ROOT / "checkpoints" / "phase1" / "best.pt"
    p2_ckpt = PROJECT_ROOT / "checkpoints" / "phase2" / "best.pt"
    p4a_ckpt = PROJECT_ROOT / "checkpoints" / "phase4a" / "best.pt"

    expected_p1 = "f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b"
    expected_p2 = "8bd977fc87412d5209081a218745325f214e384cc821d6b951ec51fe50c34d8e"
    expected_p4a = "eb2e1e1272d3695a9fabf0932306838c5e6893a87d71306283bde0968a57f7c3"

    actual_p1 = hashlib.sha256(open(p1_ckpt, "rb").read()).hexdigest()
    actual_p2 = hashlib.sha256(open(p2_ckpt, "rb").read()).hexdigest()
    actual_p4a = hashlib.sha256(open(p4a_ckpt, "rb").read()).hexdigest()

    assert actual_p1 == expected_p1, f"Phase-1 checkpoint altered! Got {actual_p1}"
    assert actual_p2 == expected_p2, f"Phase-2 checkpoint altered! Got {actual_p2}"
    assert actual_p4a == expected_p4a, f"Phase-4A checkpoint altered! Got {actual_p4a}"

    print(f"  [PASS] Phase-1 best.pt SHA256 verified invariant: {actual_p1}")
    print(f"  [PASS] Phase-2 best.pt SHA256 verified invariant: {actual_p2}")
    print(f"  [PASS] Phase-4A best.pt SHA256 verified invariant: {actual_p4a}")

    assert not (PROJECT_ROOT / "data" / "raw" / "2020").exists(), "CRITICAL: 2020+ data must not exist!"
    assert not (PROJECT_ROOT / "data" / "interim" / "2020").exists(), "CRITICAL: 2020+ data must not exist!"
    print("  [PASS] Confirmed 2020+ data has not been acquired.")

    # 2. Setup model and loader (14 channels)
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    depths = list(data_cfg.depths_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OceanEmbedNet(model_cfg).to(device)
    ckpt = torch.load(p4a_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"  [PASS] Loaded Phase-4A checkpoint: Epoch {ckpt.get('epoch')}, Val MSE {ckpt.get('val_loss_mse'):.4f} deg C^2.")

    test_ds = OceanEmbedDataset(split="test", in_channels=14, fallback_to_synthetic=False)
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

    print("\n[Inference] Running zero-gradient evaluation on 365 test days of 2019...")
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
    print(f"  [PASS] Inference completed in {inf_time:.2f}s ({inf_time / len(test_ds) * 1000:.1f} ms/day).")

    preds_arr = np.concatenate(all_preds, axis=0)      # [365, 15, 101, 241]
    targets_arr = np.concatenate(all_targets, axis=0)  # [365, 15, 101, 241]
    masks_arr = np.concatenate(all_masks, axis=0)      # [365, 15, 101, 241]

    # Compute 15-depth metrics
    depth_metrics = compute_all_depth_metrics(preds_arr, targets_arr, depths, mask=masks_arr)

    # Compute valid pixel count per depth
    valid_n_per_depth = {d: int(np.sum(masks_arr[:, idx, :, :] > 0.5)) for idx, d in enumerate(depths)}

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

    # Subsurface layers
    depths_50_150 = [50, 75, 100, 125, 150]
    depths_75_150 = [75, 100, 125, 150]
    depths_500_1000 = [500, 700, 1000]

    mean_rmse_all = float(np.mean([depth_metrics["rmse"][d] for d in depths]))
    mean_mae_all = float(np.mean([depth_metrics["mae"][d] for d in depths]))
    mean_bias_all = float(np.mean([depth_metrics["bias"][d] for d in depths]))
    mean_r_all = float(np.mean([depth_metrics["pearson_r"][d] for d in depths]))
    mean_r2_all = float(np.mean([depth_metrics["r2_score"][d] for d in depths]))

    mean_rmse_50_150 = float(np.mean([depth_metrics["rmse"][d] for d in depths_50_150]))
    mean_rmse_75_150 = float(np.mean([depth_metrics["rmse"][d] for d in depths_75_150]))
    mean_rmse_500_1000 = float(np.mean([depth_metrics["rmse"][d] for d in depths_500_1000]))

    rmse_100m = float(depth_metrics["rmse"][100])
    rmse_125m = float(depth_metrics["rmse"][125])

    # 3. Load Phase-1 2019 Temporal Test Metrics
    p1_report_path = PROJECT_ROOT / "evaluation" / "results" / "phase1_test2019_report.json"
    with open(p1_report_path, "r") as f:
        p1_data = json.load(f)

    p1_rmse = {int(k): float(v) for k, v in p1_data["oceanembed_2019_metrics"]["rmse"].items()}
    p1_mae = {int(k): float(v) for k, v in p1_data["oceanembed_2019_metrics"]["mae"].items()}
    p1_bias = {int(k): float(v) for k, v in p1_data["oceanembed_2019_metrics"]["bias"].items()}
    p1_r = {int(k): float(v) for k, v in p1_data["oceanembed_2019_metrics"]["pearson_r"].items()}
    p1_r2 = {int(k): float(v) for k, v in p1_data["oceanembed_2019_metrics"]["r2_score"].items()}

    p1_mean_rmse = float(np.mean([p1_rmse[d] for d in depths]))
    p1_mean_mae = float(np.mean([p1_mae[d] for d in depths]))
    p1_mean_bias = float(np.mean([p1_bias[d] for d in depths]))
    p1_mean_r = float(np.mean([p1_r[d] for d in depths]))
    p1_mean_r2 = float(np.mean([p1_r2[d] for d in depths]))

    p1_mean_50_150 = float(np.mean([p1_rmse[d] for d in depths_50_150]))
    p1_mean_75_150 = float(np.mean([p1_rmse[d] for d in depths_75_150]))
    p1_mean_500_1000 = float(np.mean([p1_rmse[d] for d in depths_500_1000]))
    p1_rmse_100m = float(p1_rmse[100])
    p1_rmse_125m = float(p1_rmse[125])

    delta_mean_rmse = mean_rmse_all - p1_mean_rmse
    delta_mean_rmse_pct = (delta_mean_rmse / p1_mean_rmse) * 100.0
    delta_75_150 = mean_rmse_75_150 - p1_mean_75_150
    delta_75_150_pct = (delta_75_150 / p1_mean_75_150) * 100.0
    delta_500_1000 = mean_rmse_500_1000 - p1_mean_500_1000
    delta_500_1000_pct = (delta_500_1000 / p1_mean_500_1000) * 100.0

    print("\n" + "=" * 85)
    print("PHASE-1 vs. PHASE-4A TEMPORAL TEST PERFORMANCE (2019)")
    print("=" * 85)
    print(f"{'Depth (m)':<10} | {'Phase-1 RMSE':<14} | {'Phase-4A RMSE':<14} | {'Diff (deg C)':<14} | {'Diff (%)':<10} | {'P1 R^2':<10} | {'P4A R^2':<10}")
    print("-" * 85)

    depth_diffs = {}
    for d in depths:
        r1 = p1_rmse[d]
        r4 = depth_metrics["rmse"][d]
        diff = r4 - r1
        diff_pct = (diff / r1) * 100.0
        r2_1 = p1_r2[d]
        r2_4 = depth_metrics["r2_score"][d]

        flag = ""
        if diff_pct < -1.0:
            flag = " [IMPROVED]"
        elif diff_pct > +2.0:
            flag = " [DEGRADED]"

        depth_diffs[d] = {
            "diff_degC": float(diff),
            "diff_pct": float(diff_pct),
            "p1_rmse": float(r1),
            "p4a_rmse": float(r4),
            "p1_r2": float(r2_1),
            "p4a_r2": float(r2_4),
            "valid_n": valid_n_per_depth[d],
        }

        print(f"{d:<10} | {r1:<14.4f} | {r4:<14.4f} | {diff:<+14.4f} | {diff_pct:<+9.2f}% | {r2_1:<10.4f} | {r2_4:<10.4f}{flag}")

    print("-" * 85)
    print(f"{'OVERALL':<10} | {p1_mean_rmse:<14.4f} | {mean_rmse_all:<14.4f} | {delta_mean_rmse:<+14.4f} | {delta_mean_rmse_pct:<+9.2f}% | {p1_mean_r2:<10.4f} | {mean_r2_all:<10.4f}")
    print(f"{'50-150m':<10} | {p1_mean_50_150:<14.4f} | {mean_rmse_50_150:<14.4f} | {mean_rmse_50_150 - p1_mean_50_150:<+14.4f} | {((mean_rmse_50_150 - p1_mean_50_150)/p1_mean_50_150)*100:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'75-150m':<10} | {p1_mean_75_150:<14.4f} | {mean_rmse_75_150:<14.4f} | {delta_75_150:<+14.4f} | {delta_75_150_pct:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'100m':<10} | {p1_rmse_100m:<14.4f} | {rmse_100m:<14.4f} | {rmse_100m - p1_rmse_100m:<+14.4f} | {((rmse_100m - p1_rmse_100m)/p1_rmse_100m)*100:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'125m':<10} | {p1_rmse_125m:<14.4f} | {rmse_125m:<14.4f} | {rmse_125m - p1_rmse_125m:<+14.4f} | {((rmse_125m - p1_rmse_125m)/p1_rmse_125m)*100:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'500-1000m':<10} | {p1_mean_500_1000:<14.4f} | {mean_rmse_500_1000:<14.4f} | {delta_500_1000:<+14.4f} | {delta_500_1000_pct:<+9.2f}% | {'-':<10} | {'-':<10}")

    # 4. Temporal Generalization: 2018 Val vs 2019 Test for Phase-4A
    p4a_val_path = PROJECT_ROOT / "evaluation" / "results" / "phase4a_val2018_report.json"
    with open(p4a_val_path, "r") as f:
        p4a_val = json.load(f)

    p4a_val_rmse = p4a_val["overall_metrics"]["mean_rmse"]
    p4a_val_r2 = p4a_val["overall_metrics"]["mean_r2"]
    gen_delta_rmse = mean_rmse_all - p4a_val_rmse
    gen_delta_r2 = mean_r2_all - p4a_val_r2

    print("\n" + "=" * 85)
    print("PHASE-4A TEMPORAL GENERALIZATION (2018 VAL vs. 2019 TEST)")
    print("=" * 85)
    print(f"2018 Val RMSE:  {p4a_val_rmse:.4f} °C  |  2018 Val R²:  {p4a_val_r2:.4f}")
    print(f"2019 Test RMSE: {mean_rmse_all:.4f} °C  |  2019 Test R²: {mean_r2_all:.4f}")
    print(f"Generalization Delta RMSE: {gen_delta_rmse:+.4f} °C ({(gen_delta_rmse / p4a_val_rmse)*100:+.2f}%)")
    print(f"Generalization Delta R²:   {gen_delta_r2:+.4f}")

    # Phase-1 Generalization Delta for comparison:
    p1_val_rmse = 0.7178
    p1_test_rmse = 0.7203
    p1_gen_delta = p1_test_rmse - p1_val_rmse
    print(f"Phase-1 Generalization Delta: {p1_gen_delta:+.4f} °C")

    # 5. Predeclared Decision Framework
    # Case A: Phase-4A improves 2019 overall and does not materially degrade important depth ranges
    # Case B: Phase-4A improves selected depths but overall performance is similar or worse
    # Case C: Phase-4A clearly degrades 2019
    if delta_mean_rmse < -0.005 and delta_500_1000 <= 0.01:
        case = "CASE A: Phase-4A improves 2019 overall and preserves column integrity -> Candidate for final model consideration"
    elif delta_mean_rmse > 0.01:
        case = "CASE C: Phase-4A clearly degrades 2019 -> Reject Phase-4A as final model; retain Phase-1"
    else:
        case = "CASE B: Phase-4A improves selected depths but overall performance is similar or slightly different -> Retain Phase-1 as primary baseline; retain Phase-4A as informative ablation"

    print("\n" + "=" * 85)
    print("DECISION FRAMEWORK EVALUATION")
    print("=" * 85)
    print(f"  {case}")
    print("=" * 85)

    # 6. Save Report JSON
    report = {
        "evaluation_scope": "Phase-4A Final 2019 Temporal Test",
        "checkpoint_used": str(p4a_ckpt),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_val_loss_2018": ckpt.get("val_loss_mse"),
        "test_year": 2019,
        "num_days": len(test_ds),
        "overall_metrics": {
            "mean_rmse": mean_rmse_all,
            "mean_mae": mean_mae_all,
            "mean_bias": mean_bias_all,
            "mean_pearson_r": mean_r_all,
            "mean_r2": mean_r2_all,
            "mean_rmse_50_150m": mean_rmse_50_150,
            "mean_rmse_75_150m": mean_rmse_75_150,
            "rmse_100m": rmse_100m,
            "rmse_125m": rmse_125m,
            "mean_rmse_500_1000m": mean_rmse_500_1000,
        },
        "phase1_baseline": {
            "mean_rmse": p1_mean_rmse,
            "mean_mae": p1_mean_mae,
            "mean_bias": p1_mean_bias,
            "mean_pearson_r": p1_mean_r,
            "mean_r2": p1_mean_r2,
            "mean_rmse_50_150m": p1_mean_50_150,
            "mean_rmse_75_150m": p1_mean_75_150,
            "rmse_100m": p1_rmse_100m,
            "rmse_125m": p1_rmse_125m,
            "mean_rmse_500_1000m": p1_mean_500_1000,
        },
        "differences_vs_phase1": {
            "delta_mean_rmse": delta_mean_rmse,
            "delta_mean_rmse_pct": delta_mean_rmse_pct,
            "delta_mean_r2": mean_r2_all - p1_mean_r2,
            "delta_50_150m": mean_rmse_50_150 - p1_mean_50_150,
            "delta_75_150m": delta_75_150,
            "delta_75_150m_pct": delta_75_150_pct,
            "delta_100m": rmse_100m - p1_rmse_100m,
            "delta_125m": rmse_125m - p1_rmse_125m,
            "delta_500_1000m": delta_500_1000,
            "delta_500_1000m_pct": delta_500_1000_pct,
            "depth_wise_diffs": depth_diffs,
        },
        "depth_metrics": {
            "rmse": {d: float(depth_metrics["rmse"][d]) for d in depths},
            "mae": {d: float(depth_metrics["mae"][d]) for d in depths},
            "bias": {d: float(depth_metrics["bias"][d]) for d in depths},
            "pearson_r": {d: float(depth_metrics["pearson_r"][d]) for d in depths},
            "r2_score": {d: float(depth_metrics["r2_score"][d]) for d in depths},
            "valid_n": valid_n_per_depth,
        },
        "seasonal_metrics": seasonal_metrics,
        "regional_metrics": regional_metrics,
        "temporal_generalization": {
            "val2018_mean_rmse": p4a_val_rmse,
            "test2019_mean_rmse": mean_rmse_all,
            "delta_rmse": gen_delta_rmse,
            "delta_rmse_pct": (gen_delta_rmse / p4a_val_rmse) * 100.0,
            "val2018_mean_r2": p4a_val_r2,
            "test2019_mean_r2": mean_r2_all,
            "delta_r2": gen_delta_r2,
            "phase1_gen_delta_rmse": p1_gen_delta,
        },
        "decision_framework": {
            "selected_case": case,
        },
    }

    report_path = PROJECT_ROOT / "evaluation" / "results" / "phase4a_test2019_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Artifact Saved] Machine-readable report: {report_path}")

    # 7. Generate Diagnostic Figures
    fig_dir = PROJECT_ROOT / "reports" / "figures" / "phase4a"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Depth RMSE comparison (Phase-1 vs Phase-4A on 2019)
    plt.figure(figsize=(10, 6))
    plt.plot([p1_rmse[d] for d in depths], depths, "o-", label="Phase-1 Baseline (Plain MSE)", color="#1f77b4", linewidth=2)
    plt.plot([depth_metrics["rmse"][d] for d in depths], depths, "s--", label="Phase-4A (+Gradient Loss $\\lambda=2.0$)", color="#d62728", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE (°C)", fontsize=12)
    plt.ylabel("Depth (m)", fontsize=12)
    plt.title("OceanEmbed 2019 Temporal Test: Phase-1 vs. Phase-4A Vertical RMSE Profile", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axhspan(75, 150, color="orange", alpha=0.15, label="Target Subsurface Zone (75–150m)")
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig1_path = fig_dir / "phase4a_vs_phase1_test2019_depth_rmse.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[Figure Saved] {fig1_path}")

    # Figure 2: Focused 50–150 m bar comparison
    sub_depths = [50, 75, 100, 125, 150]
    x_pos = np.arange(len(sub_depths))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x_pos - width/2, [p1_rmse[d] for d in sub_depths], width, label="Phase-1 (MSE)", color="#1f77b4", alpha=0.85)
    plt.bar(x_pos + width/2, [depth_metrics["rmse"][d] for d in sub_depths], width, label="Phase-4A (+Gradient Loss)", color="#d62728", alpha=0.85)
    plt.xticks(x_pos, [f"{d} m" for d in sub_depths], fontsize=11)
    plt.ylabel("RMSE (°C)", fontsize=12)
    plt.title("2019 Temporal Test: Subsurface Thermocline Layer (50–150 m) RMSE Comparison", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig2_path = fig_dir / "phase4a_vs_phase1_test2019_subsurface.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[Figure Saved] {fig2_path}")

    # Figure 3: Seasonal comparison
    seasons = ["DJF", "MAM", "JJAS", "OND"]
    p1_seas_dict = p1_data.get("seasonal_2019_metrics", p1_data.get("seasonal_metrics", {}))
    p1_seasonal_rmse = [float(np.mean(list(p1_seas_dict[s]["rmse"].values()))) for s in seasons]
    p4a_seasonal_rmse = [seasonal_metrics[s]["mean_rmse"] for s in seasons]
    x_pos = np.arange(len(seasons))
    plt.figure(figsize=(9, 5))
    plt.bar(x_pos - width/2, p1_seasonal_rmse, width, label="Phase-1 Baseline", color="#1f77b4", alpha=0.85)
    plt.bar(x_pos + width/2, p4a_seasonal_rmse, width, label="Phase-4A (+Gradient Loss)", color="#d62728", alpha=0.85)
    plt.xticks(x_pos, seasons, fontsize=11)
    plt.ylabel("Mean Column RMSE (°C)", fontsize=12)
    plt.title("2019 Temporal Test: Seasonal Performance Comparison", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig3_path = fig_dir / "phase4a_vs_phase1_test2019_seasonal.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"[Figure Saved] {fig3_path}")

    # Figure 4: Phase-4A 2018 Val vs 2019 Test Depth Profile
    val_d_rmse = [p4a_val["depth_metrics"]["rmse"][str(d)] for d in depths]
    test_d_rmse = [depth_metrics["rmse"][d] for d in depths]
    plt.figure(figsize=(10, 6))
    plt.plot(val_d_rmse, depths, "o-", label="Phase-4A (2018 Validation)", color="#2ca02c", linewidth=2)
    plt.plot(test_d_rmse, depths, "s--", label="Phase-4A (2019 Temporal Test)", color="#d62728", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE (°C)", fontsize=12)
    plt.ylabel("Depth (m)", fontsize=12)
    plt.title("Phase-4A Temporal Stability: 2018 Validation vs. 2019 Test", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig4_path = fig_dir / "phase4a_temporal_stability_2018_vs_2019.png"
    plt.savefig(fig4_path, dpi=200)
    plt.close()
    print(f"[Figure Saved] {fig4_path}")

    print("\n[Complete] Phase-4A 2019 Temporal Test Evaluation Finished.")
    return report


if __name__ == "__main__":
    run_phase4a_test()
