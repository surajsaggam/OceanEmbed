"""
scripts/evaluate_phase4a_val2018.py
-----------------------------------
Phase-4A 2018 Validation Set Evaluation.

Rigorous evaluation of the Phase-4A gradient-aware model (checkpoints/phase4a/best.pt)
against the frozen Phase-1 baseline on the 365 days of 2018.

Metrics computed:
1. Overall: RMSE, MAE, Bias, Pearson r, R^2
2. Depth-wise: All 15 standard depths
3. Focused:
   - 50–150 m mean RMSE
   - 75–150 m mean RMSE
   - 100 m RMSE
   - 125 m RMSE
   - 500–1000 m mean RMSE
4. Vertical gradient diagnostics:
   - dT/dz MAE, RMSE, Bias across 14 depth intervals (nonuniform dz)
5. Profile displacement diagnostics:
   - Peak gradient depth error, profile displacement statistics
6. Direct Phase-4A vs Phase-1 diffs and percent change
7. Decision gate evaluation

STRICT BOUNDARY:
- 2019 test set is NEVER accessed or evaluated.
- Phase-1 and Phase-2 checkpoints are strictly read-only.
"""

from __future__ import annotations

import hashlib
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


def compute_gradient_metrics(preds: np.ndarray, targets: np.ndarray, masks: np.ndarray, depths: list[int]):
    """
    Compute vertical temperature gradient metrics using nonuniform depth intervals.
    depths: [15] list of depths in meters.
    preds, targets, masks: [N, 15, H, W]
    """
    z = np.array(depths, dtype=np.float32)
    dz = np.diff(z)  # [14] in meters

    # Gradients along depth axis (axis=1)
    # grad[k] = (T[k+1] - T[k]) / dz[k]  in degC/m
    grad_pred = (preds[:, 1:, :, :] - preds[:, :-1, :, :]) / dz[None, :, None, None]
    grad_target = (targets[:, 1:, :, :] - targets[:, :-1, :, :]) / dz[None, :, None, None]

    # Valid gradient mask requires BOTH bounding depths to be valid
    grad_mask = (masks[:, 1:, :, :] > 0.5) & (masks[:, :-1, :, :] > 0.5)

    grad_diff = grad_pred - grad_target
    grad_abs_diff = np.abs(grad_diff)
    grad_sq_diff = grad_diff ** 2

    interval_names = [f"{depths[k]}-{depths[k+1]}m" for k in range(len(dz))]
    results = {}

    total_valid = 0
    sum_abs = 0.0
    sum_sq = 0.0
    sum_diff = 0.0

    interval_mae = {}
    interval_rmse = {}
    interval_bias = {}

    for k, name in enumerate(interval_names):
        m_k = grad_mask[:, k, :, :]
        n_valid = np.sum(m_k)
        if n_valid > 0:
            diff_k = grad_diff[:, k, :, :][m_k]
            abs_k = grad_abs_diff[:, k, :, :][m_k]
            sq_k = grad_sq_diff[:, k, :, :][m_k]

            mae_k = float(np.mean(abs_k))
            rmse_k = float(np.sqrt(np.mean(sq_k)))
            bias_k = float(np.mean(diff_k))

            total_valid += n_valid
            sum_abs += np.sum(abs_k)
            sum_sq += np.sum(sq_k)
            sum_diff += np.sum(diff_k)
        else:
            mae_k = float("nan")
            rmse_k = float("nan")
            bias_k = float("nan")

        interval_mae[name] = mae_k
        interval_rmse[name] = rmse_k
        interval_bias[name] = bias_k

    overall_mae = float(sum_abs / max(1, total_valid))
    overall_rmse = float(np.sqrt(sum_sq / max(1, total_valid)))
    overall_bias = float(sum_diff / max(1, total_valid))

    results["overall_mae"] = overall_mae
    results["overall_rmse"] = overall_rmse
    results["overall_bias"] = overall_bias
    results["interval_mae"] = interval_mae
    results["interval_rmse"] = interval_rmse
    results["interval_bias"] = interval_bias
    results["depth_intervals"] = interval_names

    return results, grad_pred, grad_target, grad_mask


def compute_profile_displacement_metrics(grad_pred, grad_target, grad_mask, depths):
    """
    Compute peak vertical gradient depth displacement metrics.
    grad_pred, grad_target: [N, 14, H, W]
    grad_mask: [N, 14, H, W]
    """
    z_mid = (np.array(depths[:-1]) + np.array(depths[1:])) / 2.0  # midpoint depths

    # Subsurface depth mask: intervals between 20m and 300m (indices 3 to 10)
    subsurface_indices = [k for k, zm in enumerate(z_mid) if 20 <= zm <= 300]

    # Find peak gradient depth for each profile where at least 4 intervals are valid
    valid_counts = np.sum(grad_mask[:, subsurface_indices, :, :], axis=1)  # [N, H, W]
    profile_mask = valid_counts >= len(subsurface_indices)  # strictly valid profiles

    target_mag = np.abs(grad_target[:, subsurface_indices, :, :])
    pred_mag = np.abs(grad_pred[:, subsurface_indices, :, :])

    # Argmax depth in subsurface
    target_peak_idx = np.argmax(target_mag, axis=1)  # [N, H, W]
    pred_peak_idx = np.argmax(pred_mag, axis=1)

    sub_z_mid = z_mid[subsurface_indices]
    target_peak_z = sub_z_mid[target_peak_idx]
    pred_peak_z = sub_z_mid[pred_peak_idx]

    displacement = pred_peak_z[profile_mask] - target_peak_z[profile_mask]
    abs_displacement = np.abs(displacement)

    return {
        "mean_displacement": float(np.mean(displacement)),
        "mean_abs_displacement": float(np.mean(abs_displacement)),
        "rmse_displacement": float(np.sqrt(np.mean(displacement ** 2))),
        "num_valid_profiles": int(np.sum(profile_mask)),
    }


def run_evaluation():
    print("=" * 75)
    print("OCEANEMBED PHASE-4A VALIDATION EVALUATION (YEAR 2018)")
    print("=" * 75)

    # Checkpoint integrity
    p1_ckpt = PROJECT_ROOT / "checkpoints" / "phase1" / "best.pt"
    p2_ckpt = PROJECT_ROOT / "checkpoints" / "phase2" / "best.pt"
    expected_p1 = "f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b"
    expected_p2 = "8bd977fc87412d5209081a218745325f214e384cc821d6b951ec51fe50c34d8e"
    actual_p1 = hashlib.sha256(open(p1_ckpt, "rb").read()).hexdigest()
    actual_p2 = hashlib.sha256(open(p2_ckpt, "rb").read()).hexdigest()
    assert actual_p1 == expected_p1, "Phase-1 checkpoint corrupted!"
    assert actual_p2 == expected_p2, "Phase-2 checkpoint corrupted!"
    print(f"[Integrity Gate] Phase-1 SHA256 verified invariant: {actual_p1}")
    print(f"[Integrity Gate] Phase-2 SHA256 verified invariant: {actual_p2}")

    ckpt_path = PROJECT_ROOT / "checkpoints" / "phase4a" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase-4A checkpoint not found at {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluation] Device: {device}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", "unknown")
    val_mse = ckpt.get("val_loss_mse", ckpt.get("val_loss", "unknown"))
    val_grad = ckpt.get("val_loss_grad", "unknown")
    val_total = ckpt.get("val_loss_total", "unknown")
    lambda_grad = ckpt.get("lambda_grad", 2.0)
    print(f"[Evaluation] Loaded Phase-4A checkpoint:")
    print(f"  Best Epoch:     {epoch}")
    print(f"  Val MSE Loss:   {val_mse:.4f} degC^2")
    print(f"  Val Grad Loss:  {val_grad:.4f} degC/m")
    print(f"  Val Total Loss: {val_total:.4f}")
    print(f"  Lambda Grad:    {lambda_grad}")

    # Initialize model (14 channels)
    model_cfg = load_config("model")
    model = OceanEmbedNet(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load 2018 Validation Dataset (14 channels)
    data_cfg = load_config("data")
    depths = list(data_cfg.depths_m)
    val_ds = OceanEmbedDataset(split="val", in_channels=14, fallback_to_synthetic=False)
    val_loader = create_dataloader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"[Evaluation] 2018 Validation dataset loaded: {len(val_ds)} days")

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

    # 1. 15-depth metrics
    depth_metrics = compute_all_depth_metrics(preds_arr, targets_arr, depths, mask=masks_arr)

    # 2. Gradient metrics
    grad_metrics, grad_pred, grad_target, grad_mask = compute_gradient_metrics(
        preds_arr, targets_arr, masks_arr, depths
    )

    # 3. Profile displacement metrics
    disp_metrics = compute_profile_displacement_metrics(grad_pred, grad_target, grad_mask, depths)

    # 4. Layer summaries
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

    # 5. Load Phase-1 2018 validation metrics
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
    print("PHASE-1 vs. PHASE-4A VALIDATION PERFORMANCE (2018)")
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
        }

        print(f"{d:<10} | {r1:<14.4f} | {r4:<14.4f} | {diff:<+14.4f} | {diff_pct:<+9.2f}% | {r2_1:<10.4f} | {r2_4:<10.4f}{flag}")

    print("-" * 85)
    print(f"{'OVERALL':<10} | {p1_mean_rmse:<14.4f} | {mean_rmse_all:<14.4f} | {delta_mean_rmse:<+14.4f} | {delta_mean_rmse_pct:<+9.2f}% | {p1_mean_r2:<10.4f} | {mean_r2_all:<10.4f}")
    print(f"{'50-150m':<10} | {p1_mean_50_150:<14.4f} | {mean_rmse_50_150:<14.4f} | {mean_rmse_50_150 - p1_mean_50_150:<+14.4f} | {((mean_rmse_50_150 - p1_mean_50_150)/p1_mean_50_150)*100:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'75-150m':<10} | {p1_mean_75_150:<14.4f} | {mean_rmse_75_150:<14.4f} | {delta_75_150:<+14.4f} | {delta_75_150_pct:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'100m':<10} | {p1_rmse_100m:<14.4f} | {rmse_100m:<14.4f} | {rmse_100m - p1_rmse_100m:<+14.4f} | {((rmse_100m - p1_rmse_100m)/p1_rmse_100m)*100:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'125m':<10} | {p1_rmse_125m:<14.4f} | {rmse_125m:<14.4f} | {rmse_125m - p1_rmse_125m:<+14.4f} | {((rmse_125m - p1_rmse_125m)/p1_rmse_125m)*100:<+9.2f}% | {'-':<10} | {'-':<10}")
    print(f"{'500-1000m':<10} | {p1_mean_500_1000:<14.4f} | {mean_rmse_500_1000:<14.4f} | {delta_500_1000:<+14.4f} | {delta_500_1000_pct:<+9.2f}% | {'-':<10} | {'-':<10}")

    print("\n" + "=" * 85)
    print("VERTICAL GRADIENT DIAGNOSTICS")
    print("=" * 85)
    print(f"Overall Gradient MAE:  {grad_metrics['overall_mae']:.6f} degC/m")
    print(f"Overall Gradient RMSE: {grad_metrics['overall_rmse']:.6f} degC/m")
    print(f"Overall Gradient Bias: {grad_metrics['overall_bias']:+.6f} degC/m")
    print(f"Peak Gradient Depth Mean Abs Displacement: {disp_metrics['mean_abs_displacement']:.2f} m")

    # Evaluate Decision Gate
    gate_75_150_improved = delta_75_150 < 0
    gate_overall_ok = delta_mean_rmse <= 0.01  # no material degradation (>0.01 C)
    gate_deep_ok = delta_500_1000 <= 0.01
    gate_surface_ok = (depth_metrics["rmse"][0] - p1_rmse[0]) <= 0.02

    promising_result = gate_75_150_improved and gate_overall_ok and gate_deep_ok and gate_surface_ok

    print("\n" + "=" * 85)
    print("DECISION GATE EVALUATION")
    print("=" * 85)
    print(f"  [Gate 1] 75–150 m improved?           {'YES (' + f'{delta_75_150:+.4f} C, {delta_75_150_pct:+.2f}%)' if gate_75_150_improved else 'NO (' + f'{delta_75_150:+.4f} C)'}")
    print(f"  [Gate 2] Overall RMSE no degradation? {'YES (' + f'{delta_mean_rmse:+.4f} C)' if gate_overall_ok else 'NO (' + f'{delta_mean_rmse:+.4f} C)'}")
    print(f"  [Gate 3] 500–1000 m no degradation?  {'YES (' + f'{delta_500_1000:+.4f} C)' if gate_deep_ok else 'NO (' + f'{delta_500_1000:+.4f} C)'}")
    surf_diff = depth_metrics['rmse'][0] - p1_rmse[0]
    print(f"  [Gate 4] Surface layer no degrad?     {'YES (' + f'{surf_diff:+.4f} C)' if gate_surface_ok else 'NO'}")
    print(f"  --> EXPERIMENT STATUS: {'PROMISING ABLATION' if promising_result else 'NEGATIVE / TRADEOFF ABLATION'}")
    print("=" * 85)

    # Build report dictionary
    report = {
        "experiment": "Phase-4A Vertical Gradient-Aware Loss Ablation",
        "validation_year": 2018,
        "num_days": len(val_ds),
        "best_epoch": epoch,
        "lambda_grad": lambda_grad,
        "gradient_loss_type": "l1",
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
        },
        "gradient_metrics": grad_metrics,
        "profile_displacement_metrics": disp_metrics,
        "decision_gates": {
            "gate_75_150_improved": gate_75_150_improved,
            "gate_overall_no_material_degradation": gate_overall_ok,
            "gate_deep_no_material_degradation": gate_deep_ok,
            "gate_surface_no_material_degradation": gate_surface_ok,
            "promising_result": promising_result,
        },
    }

    report_path = PROJECT_ROOT / "evaluation" / "results" / "phase4a_val2018_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Artifact Saved] Machine-readable report: {report_path}")

    # Generate Figures
    fig_dir = PROJECT_ROOT / "reports" / "figures" / "phase4a"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Depth RMSE comparison
    plt.figure(figsize=(10, 6))
    plt.plot([p1_rmse[d] for d in depths], depths, "o-", label="Phase-1 Baseline (Plain MSE)", color="#1f77b4", linewidth=2)
    plt.plot([depth_metrics["rmse"][d] for d in depths], depths, "s--", label=f"Phase-4A (Gradient Loss $\\lambda={lambda_grad}$)", color="#d62728", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("RMSE (°C)", fontsize=12)
    plt.ylabel("Depth (m)", fontsize=12)
    plt.title("OceanEmbed 2018 Validation: Phase-1 vs. Phase-4A Vertical RMSE Profile", fontsize=14, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.axhspan(75, 150, color="orange", alpha=0.15, label="Target High-Error Zone (75–150m)")
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig1_path = fig_dir / "phase4a_vs_phase1_val2018_depth_rmse.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[Figure Saved] {fig1_path}")

    # Figure 2: Focused Subsurface Bar Comparison
    sub_depths = [50, 75, 100, 125, 150]
    x_pos = np.arange(len(sub_depths))
    width = 0.35
    plt.figure(figsize=(9, 5))
    plt.bar(x_pos - width/2, [p1_rmse[d] for d in sub_depths], width, label="Phase-1 (MSE)", color="#1f77b4", alpha=0.85)
    plt.bar(x_pos + width/2, [depth_metrics["rmse"][d] for d in sub_depths], width, label="Phase-4A (+Gradient Loss)", color="#d62728", alpha=0.85)
    plt.xticks(x_pos, [f"{d} m" for d in sub_depths], fontsize=11)
    plt.ylabel("RMSE (°C)", fontsize=12)
    plt.title("Subsurface Thermocline Layer (50–150 m) RMSE Comparison", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    fig2_path = fig_dir / "phase4a_vs_phase1_val2018_subsurface.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[Figure Saved] {fig2_path}")

    # Figure 3: Gradient RMSE by depth interval
    intervals = grad_metrics["depth_intervals"]
    g_rmse = [grad_metrics["interval_rmse"][k] for k in intervals]
    plt.figure(figsize=(11, 5))
    plt.bar(range(len(intervals)), g_rmse, color="#2ca02c", alpha=0.8)
    plt.xticks(range(len(intervals)), intervals, rotation=45, ha="right", fontsize=10)
    plt.ylabel("Gradient RMSE (°C/m)", fontsize=12)
    plt.title("Phase-4A Vertical Temperature Gradient Reconstruction Error by Interval", fontsize=13, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig3_path = fig_dir / "phase4a_gradient_rmse_by_interval.png"
    plt.savefig(fig3_path, dpi=200)
    plt.close()
    print(f"[Figure Saved] {fig3_path}")

    print("\n[Complete] Phase-4A 2018 Validation Evaluation Finished.")
    return report


if __name__ == "__main__":
    run_evaluation()
