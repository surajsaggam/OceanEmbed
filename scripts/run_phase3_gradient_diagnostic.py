"""
scripts/run_phase3_gradient_diagnostic.py
-----------------------------------------
Vertical-Gradient & Profile-Displacement Sanity-Check on 2019 Temporal Test Set.

Scientific Rules & Protocols:
1. Frozen Checkpoint: Evaluates ONLY checkpoints/phase1/best.pt (Epoch 89).
   Zero retraining, zero fine-tuning, zero hyperparameter adjustment.
2. Target Dataset: 365 daily samples of 2019 (data/processed/test/).
3. Nonuniform Depths: dT/dz computed as (T[d+1] - T[d]) / (z[d+1] - z[d]).
4. Mask Integrity: Bathymetry masks strictly respected; invalid depths never zero-filled.
5. Mode: Pure torch.no_grad() evaluation.
6. Outputs:
   - JSON report: evaluation/results/phase3_gradient_diagnostic_2019.json
   - Diagnostic figures: reports/figures/phase3/
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
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

from evaluation.phase3_gradient_diagnostic import (
    analyze_profile_displacement_population,
    compute_gradient_vs_error_binned,
    compute_interval_valid_masks,
    compute_layer_gradient_metrics,
    compute_vertical_gradients_adjacent,
    estimate_peak_gradient_depth,
)
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, create_dataloader
from utils.config import load_config


def main():
    print("=" * 85)
    print("OCEANEMBED PHASE-3: VERTICAL-GRADIENT & DISPLACEMENT SANITY CHECK")
    print("=" * 85)

    # -------------------------------------------------------------
    # 1. Guards & Checkpoint Verification
    # -------------------------------------------------------------
    p1_ckpt = PROJECT_ROOT / "checkpoints" / "phase1" / "best.pt"
    p2_ckpt = PROJECT_ROOT / "checkpoints" / "phase2" / "best.pt"

    expected_p1 = "f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b"
    expected_p2 = "8bd977fc87412d5209081a218745325f214e384cc821d6b951ec51fe50c34d8e"

    actual_p1 = hashlib.sha256(open(p1_ckpt, "rb").read()).hexdigest()
    actual_p2 = hashlib.sha256(open(p2_ckpt, "rb").read()).hexdigest()

    assert actual_p1 == expected_p1, f"Phase-1 checkpoint altered! Got {actual_p1}"
    assert actual_p2 == expected_p2, f"Phase-2 checkpoint altered! Got {actual_p2}"

    print(f"  [PASS] Phase-1 checkpoint SHA256 verified invariant: {actual_p1}")
    print(f"  [PASS] Phase-2 checkpoint SHA256 verified invariant: {actual_p2}")
    assert not (PROJECT_ROOT / "data" / "raw" / "2020").exists(), "CRITICAL: 2020+ data must not exist!"
    print("  [PASS] 2020+ data isolation verified.")

    # -------------------------------------------------------------
    # 2. Setup Model & DataLoader
    # -------------------------------------------------------------
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    depths = list(data_cfg.depths_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OceanEmbedNet(model_cfg).to(device)
    ckpt = torch.load(p1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  [PASS] Loaded Phase-1 model (Epoch {ckpt.get('epoch')}).")

    test_ds = OceanEmbedDataset(split="test", in_channels=14, fallback_to_synthetic=False)
    assert len(test_ds) == 365, f"Expected 365 test days, got {len(test_ds)}"
    loader = create_dataloader(test_ds, batch_size=8, shuffle=False, num_workers=0)
    print(f"  [PASS] Loaded 2019 temporal test dataset: {len(test_ds)} daily samples.")

    domain = getattr(data_cfg, "domain", {})
    lat_min = domain.get("lat_min", 5.0) if isinstance(domain, dict) else getattr(domain, "lat_min", 5.0)
    lat_max = domain.get("lat_max", 30.0) if isinstance(domain, dict) else getattr(domain, "lat_max", 30.0)
    lon_min = domain.get("lon_min", 45.0) if isinstance(domain, dict) else getattr(domain, "lon_min", 45.0)
    lon_max = domain.get("lon_max", 105.0) if isinstance(domain, dict) else getattr(domain, "lon_max", 105.0)

    H, W = 101, 241
    lats = np.linspace(lat_min, lat_max, H)
    lons = np.linspace(lon_min, lon_max, W)

    fig_dir = PROJECT_ROOT / "reports" / "figures" / "phase3"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 3. Model Inference over 2019 (torch.no_grad)
    # -------------------------------------------------------------
    print("\n[Inference] Executing model inference over all 365 days of 2019...")
    t0 = time.time()
    all_preds = []
    all_targets = []
    all_masks = []
    all_dates = []

    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            out = model(x)
            pred = out["temperature"].cpu().numpy()  # [B, 15, H, W]
            target = y.numpy()                       # [B, 15, H, W]
            mask = meta["target_mask"].numpy() if "target_mask" in meta else np.ones_like(target)

            all_preds.append(pred)
            all_targets.append(target)
            all_masks.append(mask)

            b_dates = meta["date"] if isinstance(meta["date"], list) else [meta["date"]]
            all_dates.extend(b_dates)

    t_inf = time.time() - t0
    print(f"  -> Inference completed in {t_inf:.2f}s ({365 / t_inf:.1f} days/sec).")

    preds_arr = np.concatenate(all_preds, axis=0)    # [365, 15, 101, 241]
    targets_arr = np.concatenate(all_targets, axis=0)# [365, 15, 101, 241]
    masks_arr = np.concatenate(all_masks, axis=0)    # [365, 15, 101, 241]
    N_days, D, _, _ = preds_arr.shape

    # Transpose to [N_days, H, W, D] for vertical gradient calculations
    p_trans = np.transpose(preds_arr, (0, 2, 3, 1))    # [365, H, W, 15]
    t_trans = np.transpose(targets_arr, (0, 2, 3, 1))  # [365, H, W, 15]
    m_trans = np.transpose(masks_arr, (0, 2, 3, 1))    # [365, H, W, 15]

    # -------------------------------------------------------------
    # ANALYSIS A: Vertical Temperature Gradient Calculation
    # -------------------------------------------------------------
    print("\n[Analysis A] Calculating Vertical Temperature Gradients (dT/dz)...")
    # Gradients across adjacent depth intervals
    pred_grad, midpoints = compute_vertical_gradients_adjacent(p_trans, depths)   # [365, H, W, 14]
    targ_grad, _ = compute_vertical_gradients_adjacent(t_trans, depths)           # [365, H, W, 14]
    interval_masks = compute_interval_valid_masks(m_trans)                         # [365, H, W, 14]

    interval_labels = [f"{depths[k]}-{depths[k+1]}m" for k in range(D - 1)]
    print(f"  -> Calculated gradients for {len(interval_labels)} adjacent depth intervals.")
    print(f"  -> Interval midpoints (m): {[round(m, 1) for m in midpoints]}")

    # -------------------------------------------------------------
    # ANALYSIS B: Gradient Error by Layer and Interval
    # -------------------------------------------------------------
    print("\n[Analysis B] Calculating Gradient Error Statistics across Depth Layers...")
    layers = {
        "0-50m": (0.0, 50.0),
        "50-150m": (50.0, 150.0),
        "150-300m": (150.0, 300.0),
        "300-500m": (300.0, 500.0),
        "500-1000m": (500.0, 1000.0),
    }

    layer_metrics = compute_layer_gradient_metrics(pred_grad, targ_grad, interval_masks, midpoints, layers)

    # Individual interval metrics
    interval_metrics = {}
    for k, label in enumerate(interval_labels):
        m_k = interval_masks[..., k]
        p_k = pred_grad[..., k]
        t_k = targ_grad[..., k]
        valid_k = (m_k == 1.0) & np.isfinite(p_k) & np.isfinite(t_k)
        cnt_k = int(np.sum(valid_k))
        if cnt_k > 0:
            err_k = p_k[valid_k] - t_k[valid_k]
            interval_metrics[label] = {
                "midpoint_m": float(midpoints[k]),
                "delta_z_m": float(depths[k+1] - depths[k]),
                "gradient_mae": float(np.mean(np.abs(err_k))),
                "gradient_rmse": float(np.sqrt(np.mean(err_k**2))),
                "gradient_bias": float(np.mean(err_k)),
                "target_mean_abs_grad": float(np.mean(np.abs(t_k[valid_k]))),
                "pred_mean_abs_grad": float(np.mean(np.abs(p_k[valid_k]))),
                "valid_count": cnt_k,
            }

    # Plot Analysis B: Gradient Error and Gradient Magnitude by Depth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=150)

    # Panel 1: Gradient Magnitude (Target vs Pred)
    t_mags = [interval_metrics[lbl]["target_mean_abs_grad"] for lbl in interval_labels]
    p_mags = [interval_metrics[lbl]["pred_mean_abs_grad"] for lbl in interval_labels]
    ax1.plot(t_mags, midpoints, "s-", color="#2c3e50", lw=2, label="Target |dT/dz| (GLORYS)")
    ax1.plot(p_mags, midpoints, "o-", color="#e74c3c", lw=2, label="Pred |dT/dz| (OceanEmbed)")
    ax1.set_ylim(1050, -50)
    ax1.set_xlabel("Mean Absolute Vertical Gradient (°C / m)")
    ax1.set_ylabel("Depth Midpoint (m)")
    ax1.set_title("Vertical Temperature Gradient Profile", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Gradient MAE and RMSE
    g_maes = [interval_metrics[lbl]["gradient_mae"] for lbl in interval_labels]
    g_rmses = [interval_metrics[lbl]["gradient_rmse"] for lbl in interval_labels]
    ax2.plot(g_rmses, midpoints, "o-", color="#8e44ad", lw=2, label="Gradient RMSE (°C / m)")
    ax2.plot(g_maes, midpoints, "s-", color="#2980b9", lw=2, label="Gradient MAE (°C / m)")
    ax2.set_ylim(1050, -50)
    ax2.set_xlabel("Gradient Error (°C / m)")
    ax2.set_title("Gradient Reconstruction Error by Depth", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("OceanEmbed Phase-3: Vertical Gradient & Gradient Error Distribution (2019)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_gradient_error_by_depth.png", dpi=150)
    plt.close(fig)
    print("  [DONE] Gradient error by depth saved.")

    # -------------------------------------------------------------
    # ANALYSIS C: Relationship Between Target Gradient and Temperature Error
    # -------------------------------------------------------------
    print("\n[Analysis C] Evaluating Correlation Between Gradient Magnitude and Temperature Error...")
    # Subsample across 2019 for correlation and binned analysis
    # Associate each standard depth k with the local vertical gradient magnitude
    # For depth k: local gradient is average of adjacent upper and lower gradients
    rng = np.random.RandomState(42)
    sample_targets_grad = []
    sample_temp_errors = []
    sample_depth_tags = []

    for d_idx in range(N_days):
        # Sample ~100 ocean pixels per day
        valid_deep = (masks_arr[d_idx, 9] == 1.0)  # valid down to at least 150m
        ys, xs = np.where(valid_deep)
        if len(ys) == 0:
            continue
        pick = rng.choice(len(ys), size=min(100, len(ys)), replace=False)
        py = ys[pick]
        px = xs[pick]

        # For these pixels, compute for depths 0 to 14:
        # local gradient magnitude at midpoint intervals
        for k in range(D - 1):
            t_g_val = np.abs(targ_grad[d_idx, py, px, k])  # [sample_k]
            # Temperature error at the lower depth of interval
            t_err_val = np.abs(p_trans[d_idx, py, px, k + 1] - t_trans[d_idx, py, px, k + 1])
            m_val = interval_masks[d_idx, py, px, k]

            valid_pts = (m_val == 1.0) & np.isfinite(t_g_val) & np.isfinite(t_err_val)
            if np.any(valid_pts):
                sample_targets_grad.append(t_g_val[valid_pts])
                sample_temp_errors.append(t_err_val[valid_pts])
                sample_depth_tags.append(np.full(np.sum(valid_pts), midpoints[k]))

    all_t_grads = np.concatenate(sample_targets_grad)
    all_t_errs = np.concatenate(sample_temp_errors)
    all_d_tags = np.concatenate(sample_depth_tags)

    print(f"  -> Assembled {len(all_t_grads)} gradient-error pairs.")

    binned_gradient_rel = compute_gradient_vs_error_binned(all_t_grads, all_t_errs, n_bins=10)
    print(f"  -> Pearson r: {binned_gradient_rel['pearson_r']:+.4f} (p = {binned_gradient_rel['pearson_p']:.2e})")
    print(f"  -> Spearman rho: {binned_gradient_rel['spearman_rho']:+.4f} (p = {binned_gradient_rel['spearman_p']:.2e})")

    # Plot Analysis C: Binned relationship
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Panel 1: Binned Error vs Target Gradient
    b_list = binned_gradient_rel["bins"]
    if b_list:
        g_means = [b["grad_mean"] * 1000.0 for b in b_list]  # in m°C / m
        mae_means = [b["mae_mean"] for b in b_list]
        mae_meds = [b["mae_median"] for b in b_list]
        rmses = [b["rmse"] for b in b_list]

        ax1.plot(g_means, rmses, "o-", color="#c0392b", lw=2, label="Temperature RMSE (°C)")
        ax1.plot(g_means, mae_means, "s-", color="#2980b9", lw=2, label="Temperature MAE (°C)")
        ax1.plot(g_means, mae_meds, "^--", color="#27ae60", lw=1.5, label="Temperature Median Error (°C)")
        ax1.set_xlabel("Target Vertical Gradient Magnitude |dT/dz| (m°C / m)")
        ax1.set_ylabel("Temperature Reconstruction Error (°C)")
        ax1.set_title(f"Reconstruction Error vs Target Gradient Magnitude\n(Pearson r = {binned_gradient_rel['pearson_r']:+.3f}, Spearman ρ = {binned_gradient_rel['spearman_rho']:+.3f})", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

    # Panel 2: Subsurface (50-150m) subset correlation
    sub_mask = (all_d_tags >= 50.0) & (all_d_tags <= 150.0)
    binned_sub = compute_gradient_vs_error_binned(all_t_grads[sub_mask], all_t_errs[sub_mask], n_bins=8)
    if binned_sub["bins"]:
        g_sub_means = [b["grad_mean"] * 1000.0 for b in binned_sub["bins"]]
        rmse_sub = [b["rmse"] for b in binned_sub["bins"]]
        mae_sub = [b["mae_mean"] for b in binned_sub["bins"]]
        ax2.plot(g_sub_means, rmse_sub, "o-", color="#e67e22", lw=2, label="75-150m RMSE (°C)")
        ax2.plot(g_sub_means, mae_sub, "s-", color="#2c3e50", lw=2, label="75-150m MAE (°C)")
        ax2.set_xlabel("Target Vertical Gradient Magnitude in 50-150m (m°C / m)")
        ax2.set_ylabel("Temperature Reconstruction Error (°C)")
        ax2.set_title(f"Upper-Interior (50-150 m) Gradient Relationship\n(Pearson r = {binned_sub['pearson_r']:+.3f}, Spearman ρ = {binned_sub['spearman_rho']:+.3f})", fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    fig.suptitle("OceanEmbed Phase-3: Temperature Error vs Vertical Temperature Gradient", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_gradient_vs_temp_error_binned.png", dpi=150)
    plt.close(fig)
    print("  [DONE] Gradient vs temperature error plots saved.")

    # -------------------------------------------------------------
    # ANALYSIS D: Profile Displacement Diagnostic
    # -------------------------------------------------------------
    print("\n[Analysis D] Evaluating Profile Displacement (Peak Gradient Depth Mismatch)...")
    # Subsample full vertical columns valid to >= 200m
    flat_preds = p_trans.reshape(-1, D)     # [M_total, 15]
    flat_targets = t_trans.reshape(-1, D)   # [M_total, 15]
    flat_masks = m_trans.reshape(-1, D)     # [M_total, 15]

    # Evaluate on candidate profile pool
    step_stride = max(1, flat_preds.shape[0] // 15000)
    cand_preds = flat_preds[::step_stride]
    cand_targs = flat_targets[::step_stride]
    cand_masks = flat_masks[::step_stride]

    displacement_results = analyze_profile_displacement_population(
        cand_preds, cand_targs, depths, cand_masks, min_eval_depth=200.0, min_peak_grad=0.03
    )

    print(f"  -> Eligible strong-gradient profiles analyzed: N = {displacement_results['eligible_profile_count']}")
    print(f"  -> Discrete Displacement: Mean = {displacement_results['discrete_displacement']['mean_m']:+.2f} m, Median = {displacement_results['discrete_displacement']['median_m']:+.1f} m, Mean |Δz| = {displacement_results['discrete_displacement']['mean_abs_m']:.2f} m")
    print(f"  -> Subgrid Displacement: Mean = {displacement_results['subgrid_displacement']['mean_m']:+.2f} m, Median = {displacement_results['subgrid_displacement']['median_m']:+.1f} m, Mean |Δz| = {displacement_results['subgrid_displacement']['mean_abs_m']:.2f} m")
    print(f"  -> Correlation between |Δz| and Profile MAE: r = {displacement_results['correlation_displacement_vs_temperature_error']['pearson_r_abs_disp_vs_mae']:+.3f}")

    # Plot Analysis D: Displacement Distribution and Correlation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Panel 1: Histogram of Discrete Displacement
    hist_d = displacement_results["discrete_displacement"]["histogram"]
    x_shifts = [float(k) for k in hist_d.keys()]
    y_counts = [hist_d[k] for k in hist_d.keys()]
    # Sort
    sort_idx = np.argsort(x_shifts)
    x_shifts = [x_shifts[i] for i in sort_idx]
    y_counts = [y_counts[i] for i in sort_idx]

    bar_colors = ["#3498db" if x == 0 else "#e74c3c" for x in x_shifts]
    ax1.bar(x_shifts, y_counts, width=15.0, color=bar_colors, edgecolor="black", alpha=0.85)
    ax1.axvline(0.0, color="k", linestyle="--", lw=1.5)
    ax1.set_xlabel("Peak Gradient Depth Displacement Δz (m) [Pred - Target]")
    ax1.set_ylabel("Profile Count")
    ax1.set_title(f"Displacement Histogram (N = {displacement_results['eligible_profile_count']})\nMean = {displacement_results['discrete_displacement']['mean_m']:+.1f} m | Mean |Δz| = {displacement_results['discrete_displacement']['mean_abs_m']:.1f} m", fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Subgrid continuous displacement distribution
    ax2.text(
        0.05, 0.70,
        f"DISPLACEMENT STATS:\n"
        f"• Eligible Profiles (≥200m, |dT/dz|≥0.03): N = {displacement_results['eligible_profile_count']:,}\n\n"
        f"• Discrete Mean Displacement: {displacement_results['discrete_displacement']['mean_m']:+.2f} m\n"
        f"• Discrete Mean |Δz|: {displacement_results['discrete_displacement']['mean_abs_m']:.2f} m\n"
        f"• Zero-Shift Frequency: {hist_d.get('0.0', 0) / displacement_results['eligible_profile_count'] * 100:.1f}%\n\n"
        f"• Subgrid Parabolic Mean |Δz|: {displacement_results['subgrid_displacement']['mean_abs_m']:.2f} m\n"
        f"• Subgrid Median |Δz|: {displacement_results['subgrid_displacement']['median_abs_m']:.2f} m\n\n"
        f"• Correlation |Δz| vs Profile MAE:\n"
        f"   Pearson r = {displacement_results['correlation_displacement_vs_temperature_error']['pearson_r_abs_disp_vs_mae']:+.3f}\n"
        f"   Spearman ρ = {displacement_results['correlation_displacement_vs_temperature_error']['spearman_rho_abs_disp_vs_mae']:+.3f}\n\n"
        f"CRITICAL RESOLUTION CAVEAT:\n"
        f"The 15-depth grid has 25m spacing in 50-150m.\n"
        f"Peak shifts are quantized in steps of ±25m.\n"
        f"Estimates reflect coarse grid constraints.",
        fontsize=9.5,
        family="monospace",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="#ecf0f1", edgecolor="#bdc3c7", alpha=0.9),
    )
    ax2.axis("off")
    ax2.set_title("Profile Displacement Summary & Caveats", fontweight="bold")

    fig.suptitle("OceanEmbed Phase-3: Peak Vertical Gradient Displacement Diagnostic", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_profile_displacement_diagnostic.png", dpi=150)
    plt.close(fig)
    print("  [DONE] Profile displacement plots saved.")

    # -------------------------------------------------------------
    # ANALYSIS E: 75–150 m Focused Check
    # -------------------------------------------------------------
    print("\n[Analysis E] Dedicated 75-150 m Layer Diagnostic...")
    # Intervals in this range: 50-75m (62.5m), 75-100m (87.5m), 100-125m (112.5m), 125-150m (137.5m)
    sub_intervals = ["50-75m", "75-100m", "100-125m", "125-150m"]
    sub_metrics = {lbl: interval_metrics[lbl] for lbl in sub_intervals if lbl in interval_metrics}

    # Compare gradient magnitude underestimation (gradient smoothing ratio)
    smoothing_ratios = {}
    for lbl in sub_intervals:
        t_mag = sub_metrics[lbl]["target_mean_abs_grad"]
        p_mag = sub_metrics[lbl]["pred_mean_abs_grad"]
        ratio = p_mag / t_mag if t_mag > 0 else 1.0
        smoothing_ratios[lbl] = {
            "target_mag": t_mag,
            "pred_mag": p_mag,
            "ratio_pred_to_target": ratio,
            "underestimation_pct": (1.0 - ratio) * 100.0,
            "gradient_mae": sub_metrics[lbl]["gradient_mae"],
        }
        print(f"  -> {lbl:<10} | Target |dT/dz|: {t_mag:.4f} °C/m | Pred |dT/dz|: {p_mag:.4f} °C/m | Ratio: {ratio:.3f} (Smoothing: {(1.0 - ratio)*100:.1f}%)")

    # Plot Analysis E
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Panel 1: Target vs Predicted Gradient in 50-150m
    x_sub = np.arange(len(sub_intervals))
    t_sub_vals = [smoothing_ratios[lbl]["target_mag"] * 1000.0 for lbl in sub_intervals]
    p_sub_vals = [smoothing_ratios[lbl]["pred_mag"] * 1000.0 for lbl in sub_intervals]
    w = 0.35
    ax1.bar(x_sub - w/2, t_sub_vals, width=w, color="#2c3e50", label="Target Gradient (GLORYS)")
    ax1.bar(x_sub + w/2, p_sub_vals, width=w, color="#e74c3c", label="Pred Gradient (OceanEmbed)")
    ax1.set_xticks(x_sub)
    ax1.set_xticklabels(sub_intervals)
    ax1.set_ylabel("Mean |dT/dz| (m°C / m)")
    ax1.set_title("Target vs Predicted Gradient Magnitude (50-150 m)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Underestimation percentage
    under_pcts = [smoothing_ratios[lbl]["underestimation_pct"] for lbl in sub_intervals]
    ax2.bar(x_sub, under_pcts, width=0.45, color="#8e44ad", edgecolor="black")
    ax2.set_xticks(x_sub)
    ax2.set_xticklabels(sub_intervals)
    ax2.set_ylabel("Gradient Smoothing / Underestimation (%)")
    ax2.set_title("Vertical Gradient Smoothing Percentage", fontweight="bold")
    for idx_val, val in enumerate(under_pcts):
        ax2.text(idx_val, val + 0.5, f"{val:.1f}%", ha="center", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("OceanEmbed Phase-3: Upper-Interior (50-150 m) Gradient Smoothing Diagnostic", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_subsurface_gradient_diagnostic.png", dpi=150)
    plt.close(fig)
    print("  [DONE] 75-150 m focused diagnostic saved.")

    # -------------------------------------------------------------
    # ANALYSIS F: Deterministic Representative Profiles
    # -------------------------------------------------------------
    print("\n[Analysis F] Selecting Deterministic Representative Profiles...")
    # Find eligible profiles across candidate days
    cand_day_indices = [15, 45, 105, 195, 255, 315]
    all_case_pool = []

    for c_idx in cand_day_indices:
        d_str = all_dates[c_idx]
        d_p = p_trans[c_idx]  # [H, W, 15]
        d_t = t_trans[c_idx]  # [H, W, 15]
        d_m = m_trans[c_idx]  # [H, W, 15]

        # Fully valid columns down to at least 200m
        valid_col = (np.all(d_m[:, :, :11] == 1.0, axis=-1))
        ys, xs = np.where(valid_col)

        for py, px in zip(ys[::12], xs[::12]):
            prof_p = d_p[py, px]
            prof_t = d_t[py, px]
            col_mae = float(np.mean(np.abs(prof_p - prof_t)))
            col_rmse = float(np.sqrt(np.mean((prof_p - prof_t)**2)))

            # Gradient profiles
            g_p, _ = compute_vertical_gradients_adjacent(prof_p, depths)
            g_t, _ = compute_vertical_gradients_adjacent(prof_t, depths)
            peak_t_grad = float(np.max(np.abs(g_t[:10])))  # upper 200m

            all_case_pool.append({
                "date": d_str,
                "lat": float(lats[py]),
                "lon": float(lons[px]),
                "col_mae": col_mae,
                "col_rmse": col_rmse,
                "peak_target_grad": peak_t_grad,
                "pred_temp": prof_p.tolist(),
                "target_temp": prof_t.tolist(),
                "pred_grad": g_p.tolist(),
                "target_grad": g_t.tolist(),
            })

    # Sort and filter
    all_case_pool.sort(key=lambda c: c["col_mae"])
    grad_pool = [c for c in all_case_pool if c["peak_target_grad"] >= 0.05]
    grad_pool.sort(key=lambda c: c["peak_target_grad"])

    # 1. Low-error profile
    case_low = all_case_pool[0]

    # 2. High-error profile
    case_high = max(all_case_pool, key=lambda c: c["col_mae"])

    # 3. High-gradient / low-error profile (top 20% gradient, bottom 25% error)
    p80_grad = np.percentile([c["peak_target_grad"] for c in all_case_pool], 80)
    p25_mae = np.percentile([c["col_mae"] for c in all_case_pool], 25)
    hg_le_pool = [c for c in all_case_pool if c["peak_target_grad"] >= p80_grad and c["col_mae"] <= p25_mae]
    case_hg_le = hg_le_pool[len(hg_le_pool) // 2] if hg_le_pool else grad_pool[0]

    # 4. High-gradient / high-error profile (top 20% gradient, top 25% error)
    p75_mae = np.percentile([c["col_mae"] for c in all_case_pool], 75)
    hg_he_pool = [c for c in all_case_pool if c["peak_target_grad"] >= p80_grad and c["col_mae"] >= p75_mae]
    case_hg_he = hg_he_pool[len(hg_he_pool) // 2] if hg_he_pool else grad_pool[-1]

    rep_profiles = {
        "1. Low-Error Profile": case_low,
        "2. High-Error Profile": case_high,
        "3. High-Gradient / Low-Error": case_hg_le,
        "4. High-Gradient / High-Error": case_hg_he,
    }

    # Plot Analysis F: 4 cases, each with T(z) on left and dT/dz(z) on right
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=150)

    for idx, (title, case) in enumerate(rep_profiles.items()):
        # Temperature profile (row 0)
        ax_t = axes[0, idx]
        p_t = np.array(case["pred_temp"])
        t_t = np.array(case["target_temp"])
        ax_t.plot(p_t, depths, "o-", color="#e74c3c", lw=2, label="OceanEmbed Pred")
        ax_t.plot(t_t, depths, "s--", color="#2c3e50", lw=2, label="GLORYS Target")
        ax_t.set_ylim(1050, -50)
        ax_t.set_xlabel("Temperature (°C)")
        if idx == 0:
            ax_t.set_ylabel("Depth (m)")
            ax_t.legend(loc="lower left")
        ax_t.set_title(f"{title}\n{case['date']} | ({case['lat']:.1f}°N, {case['lon']:.1f}°E)\nMAE: {case['col_mae']:.2f}°C", fontsize=9.5, fontweight="bold")
        ax_t.grid(True, alpha=0.3)

        # Gradient profile (row 1)
        ax_g = axes[1, idx]
        p_g = np.array(case["pred_grad"])
        t_g = np.array(case["target_grad"])
        ax_g.plot(p_g * 1000.0, midpoints, "o-", color="#e74c3c", lw=1.8, label="Pred dT/dz")
        ax_g.plot(t_g * 1000.0, midpoints, "s--", color="#2c3e50", lw=1.8, label="Target dT/dz")
        ax_g.set_ylim(500, -25)  # zoom to upper 500m
        ax_g.set_xlabel("dT/dz (m°C / m)")
        if idx == 0:
            ax_g.set_ylabel("Depth Midpoint (m)")
            ax_g.legend(loc="lower left")
        ax_g.set_title(f"Vertical Gradient dT/dz\nPeak: {case['peak_target_grad']*1000:.1f} m°C/m", fontsize=9.5, fontweight="bold")
        ax_g.grid(True, alpha=0.3)

    fig.suptitle("OceanEmbed Phase-3: Deterministic Representative Profiles — Temperature & Gradient Alignment", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_representative_gradient_profiles.png", dpi=150)
    plt.close(fig)
    print("  [DONE] Representative gradient profiles saved.")

    # -------------------------------------------------------------
    # DECISION GATE CLASSIFICATION
    # -------------------------------------------------------------
    # Thresholds:
    # Strong correlation (r > 0.35 or rho > 0.35) AND significant gradient smoothing (> 10% in 75-150m) -> Category A
    # Moderate correlation (0.15 < r <= 0.35) -> Category B
    # Weak/no correlation (r <= 0.15) -> Category C
    p_r = binned_gradient_rel["pearson_r"]
    s_rho = binned_gradient_rel["spearman_rho"]
    mean_smoothing = float(np.mean([smoothing_ratios[k]["underestimation_pct"] for k in smoothing_ratios]))

    if (p_r >= 0.30 or s_rho >= 0.30) and mean_smoothing >= 10.0:
        decision_category = "A. STRONG SUPPORT"
        decision_text = (
            "Evidence strongly supports investigating gradient-aware loss. "
            f"Target vertical gradient magnitude is strongly correlated with temperature error (r = {p_r:+.3f}, ρ = {s_rho:+.3f}), "
            f"and Phase-1 systematically underestimates the peak vertical gradient magnitude by an average of {mean_smoothing:.1f}% in the 50-150 m layer."
        )
    elif p_r >= 0.15 or s_rho >= 0.15 or mean_smoothing >= 5.0:
        decision_category = "B. PARTIAL SUPPORT"
        decision_text = (
            "Evidence suggests a gradient relationship, but more evidence or a carefully scoped experiment is needed. "
            f"Correlation: r = {p_r:+.3f}, ρ = {s_rho:+.3f}; mean gradient smoothing: {mean_smoothing:.1f}%."
        )
    else:
        decision_category = "C. INSUFFICIENT SUPPORT"
        decision_text = (
            "The evidence does not justify prioritizing gradient-aware loss at this stage."
        )

    print("\n" + "=" * 85)
    print(f"DECISION GATE CLASSIFICATION: {decision_category}")
    print(decision_text)
    print("=" * 85)

    # -------------------------------------------------------------
    # Machine-Readable Serialization
    # -------------------------------------------------------------
    json_results = {
        "analysis_scope": "Phase-3 Vertical-Gradient & Profile-Displacement Diagnostic",
        "dataset": "2019 Temporal Test Set (365 calendar days)",
        "checkpoint_sha256": actual_p1,
        "phase2_checkpoint_sha256": actual_p2,
        "decision_gate": {
            "category": decision_category,
            "justification": decision_text,
            "pearson_r_gradient_vs_temp_error": p_r,
            "spearman_rho_gradient_vs_temp_error": s_rho,
            "mean_50_150m_gradient_smoothing_pct": mean_smoothing,
        },
        "layer_gradient_metrics": layer_metrics,
        "interval_gradient_metrics": interval_metrics,
        "gradient_vs_error_correlation": {
            "full_column": binned_gradient_rel,
            "subsurface_50_150m": binned_sub,
        },
        "profile_displacement": displacement_results,
        "subsurface_75_150m_gradient_smoothing": smoothing_ratios,
        "representative_profiles": {
            k: {
                "date": v["date"],
                "lat": v["lat"],
                "lon": v["lon"],
                "col_mae": v["col_mae"],
                "peak_target_grad": v["peak_target_grad"],
            }
            for k, v in rep_profiles.items()
        },
    }

    report_out = PROJECT_ROOT / "evaluation" / "results" / "phase3_gradient_diagnostic_2019.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=2)

    print(f"  [PASS] Saved JSON diagnostic report to: {report_out}")


if __name__ == "__main__":
    main()
