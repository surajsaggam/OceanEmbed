"""
scripts/run_phase3_error_analysis.py
------------------------------------
Phase-3 Scientific Error Analysis for OceanEmbed on the 2019 Temporal Test Set.

Scientific Rules & Protocols:
1. Frozen Checkpoint: Evaluates ONLY checkpoints/phase1/best.pt (Epoch 89).
   Zero retraining, zero fine-tuning, zero hyperparameter adjustment.
2. Dataset: 365 daily samples of 2019 (data/processed/test/).
3. Physical Variables: Un-normalized using data/norm_stats/train_stats.json.
4. Mask Integrity: Invalid ocean and land pixels are strictly NaN, never zero-filled.
5. All reported metrics track exact valid sample counts (N).
6. Outputs:
   - JSON report: evaluation/results/phase3_error_analysis_2019.json
   - High-resolution figures: reports/figures/phase3/
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

from evaluation.phase3_error_analysis import (
    compute_binned_statistics,
    compute_depth_metrics_with_counts,
    compute_spatial_gradient_magnitude,
    compute_time_aggregated_spatial_errors,
    perform_embedding_pca,
)
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, create_dataloader
from utils.config import load_config


def main():
    print("=" * 85)
    print("OCEANEMBED PHASE-3 SCIENTIFIC ERROR ANALYSIS (2019 TEMPORAL TEST SET)")
    print("=" * 85)

    # -------------------------------------------------------------
    # 1. Integrity Guards & Checkpoint Verification
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

    norm_stats_p = PROJECT_ROOT / "data" / "norm_stats" / "train_stats.json"
    with open(norm_stats_p, "r", encoding="utf-8") as f:
        norm_stats = json.load(f)
    assert np.isclose(norm_stats["SST"]["mean"], 28.254789, atol=1e-5), "Train stats altered!"
    print("  [PASS] Normalization stats verified (2015-2017 frozen moments).")

    # -------------------------------------------------------------
    # 2. Setup Model, Dataset, and Spatial Domain
    # -------------------------------------------------------------
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    depths = list(data_cfg.depths_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OceanEmbedNet(model_cfg).to(device)
    ckpt = torch.load(p1_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  [PASS] Loaded Phase-1 model (Epoch {ckpt.get('epoch')}, val_loss: {ckpt.get('val_loss'):.4f}).")

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
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Regional bounding boxes
    as_mask_2d = (lat_grid >= 5.0) & (lat_grid <= 25.0) & (lon_grid >= 45.0) & (lon_grid <= 77.5)
    bob_mask_2d = (lat_grid >= 5.0) & (lat_grid <= 23.0) & (lon_grid >= 80.0) & (lon_grid <= 100.0)

    # Output figures directory
    fig_dir = PROJECT_ROOT / "reports" / "figures" / "phase3"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 3. Model Inference & Diagnostic Subsampling
    # -------------------------------------------------------------
    print("\n[Inference] Running Phase-1 model over all 365 days of 2019...")
    t0 = time.time()

    all_preds = []
    all_targets = []
    all_masks = []
    all_dates = []

    # Containers for surface diagnostic features and embeddings subsample
    sampled_embeddings = []
    sampled_err_col = []
    sampled_err_75_150 = []
    sampled_sst = []
    sampled_sss = []
    sampled_ssh = []
    sampled_curr_speed = []
    sampled_wind_speed = []
    sampled_grad_sst = []
    sampled_grad_ssh = []
    sampled_missing_frac = []
    sampled_seasons = []
    sampled_regions = []

    # Depth index mapping for 75-150m
    idx_75_150 = [depths.index(d) for d in [75, 100, 125, 150]]

    date_regex = re.compile(r"\d{4}-(\d{2})-\d{2}")
    rng = np.random.RandomState(42)

    with torch.no_grad():
        for b_idx, (x, y, meta) in enumerate(loader):
            x = x.to(device)
            out = model(x)
            pred = out["temperature"].cpu().numpy()  # [B, 15, H, W]
            emb = out["embedding"].cpu().numpy()     # [B, 128, H, W]
            target = y.numpy()                       # [B, 15, H, W]
            mask = meta["target_mask"].numpy() if "target_mask" in meta else np.ones_like(target)

            all_preds.append(pred)
            all_targets.append(target)
            all_masks.append(mask)

            b_dates = meta["date"] if isinstance(meta["date"], list) else [meta["date"]]
            all_dates.extend(b_dates)

            # Un-normalize physical surface inputs:
            # x[:, 0] = SST, x[:, 1] = SSS, x[:, 2] = SSH, x[:, 3] = U_curr, x[:, 4] = V_curr, x[:, 5] = WindU, x[:, 6] = WindV
            # x[:, 7:14] = masks
            x_cpu = x.cpu().numpy()
            sst_raw = x_cpu[:, 0] * norm_stats["SST"]["std"] + norm_stats["SST"]["mean"]
            sss_raw = x_cpu[:, 1] * norm_stats["SSS"]["std"] + norm_stats["SSS"]["mean"]
            ssh_raw = x_cpu[:, 2] * norm_stats["SSH"]["std"] + norm_stats["SSH"]["mean"]
            u_curr = x_cpu[:, 3] * norm_stats["U_curr"]["std"] + norm_stats["U_curr"]["mean"]
            v_curr = x_cpu[:, 4] * norm_stats["V_curr"]["std"] + norm_stats["V_curr"]["mean"]
            curr_spd = np.sqrt(u_curr**2 + v_curr**2)
            u_wind = x_cpu[:, 5] * norm_stats["WindU"]["std"] + norm_stats["WindU"]["mean"]
            v_wind = x_cpu[:, 6] * norm_stats["WindV"]["std"] + norm_stats["WindV"]["mean"]
            wind_spd = np.sqrt(u_wind**2 + v_wind**2)

            # Missing fraction across the 7 surface variables
            surface_valid_masks = x_cpu[:, 7:14]  # [B, 7, H, W]
            missing_frac = 1.0 - (np.sum(surface_valid_masks, axis=1) / 7.0)  # [B, H, W]

            # Gradients
            grad_sst = compute_spatial_gradient_magnitude(sst_raw, dx=0.25, dy=0.25)
            grad_ssh = compute_spatial_gradient_magnitude(ssh_raw, dx=0.25, dy=0.25)

            # Subsample ~60 valid ocean points per day for feature/embedding correlation
            for i in range(pred.shape[0]):
                d_str = b_dates[i]
                mo = int(date_regex.search(d_str).group(1))
                if mo in [12, 1, 2]:
                    season = "DJF"
                elif mo in [3, 4, 5]:
                    season = "MAM"
                elif mo in [6, 7, 8, 9]:
                    season = "JJAS"
                else:
                    season = "OND"

                # Valid ocean pixels where depths 75-150m are all valid ocean (bathymetry >= 150m)
                valid_ocean = np.all(mask[i, idx_75_150] == 1.0, axis=0)
                valid_ys, valid_xs = np.where(valid_ocean)
                if len(valid_ys) == 0:
                    continue

                sample_k = min(60, len(valid_ys))
                pick_idx = rng.choice(len(valid_ys), size=sample_k, replace=False)
                py = valid_ys[pick_idx]
                px = valid_xs[pick_idx]

                # Prediction errors (strictly over valid depths)
                p_col = pred[i, :, py, px]  # [sample_k, 15]
                t_col = target[i, :, py, px]  # [sample_k, 15]
                m_col = mask[i, :, py, px]  # [sample_k, 15]
                err_matrix = np.abs(p_col - t_col)  # [sample_k, 15]
                col_err = np.sum(err_matrix * m_col, axis=1) / np.maximum(np.sum(m_col, axis=1), 1.0)
                sub_err = np.mean(err_matrix[:, idx_75_150], axis=1)  # [sample_k] (all valid by construction)

                # Embeddings [sample_k, 128]
                e_pts = emb[i, :, py, px]  # [sample_k, 128]

                sampled_embeddings.append(e_pts)
                sampled_err_col.append(col_err)
                sampled_err_75_150.append(sub_err)
                sampled_sst.append(sst_raw[i, py, px])
                sampled_sss.append(sss_raw[i, py, px])
                sampled_ssh.append(ssh_raw[i, py, px])
                sampled_curr_speed.append(curr_spd[i, py, px])
                sampled_wind_speed.append(wind_spd[i, py, px])
                sampled_grad_sst.append(grad_sst[i, py, px])
                sampled_grad_ssh.append(grad_ssh[i, py, px])
                sampled_missing_frac.append(missing_frac[i, py, px])

                for y_coord, x_coord in zip(py, px):
                    sampled_seasons.append(season)
                    if as_mask_2d[y_coord, x_coord]:
                        sampled_regions.append("Arabian Sea")
                    elif bob_mask_2d[y_coord, x_coord]:
                        sampled_regions.append("Bay of Bengal")
                    else:
                        sampled_regions.append("Other")

    elapsed = time.time() - t0
    print(f"  -> Inference completed in {elapsed:.2f}s ({365 / elapsed:.1f} days/sec).")

    preds_arr = np.concatenate(all_preds, axis=0)    # [365, 15, 101, 241]
    targets_arr = np.concatenate(all_targets, axis=0)# [365, 15, 101, 241]
    masks_arr = np.concatenate(all_masks, axis=0)    # [365, 15, 101, 241]
    N, D, _, _ = preds_arr.shape

    # Flatten subsampled diagnostic points
    emb_all = np.vstack(sampled_embeddings)          # [M, 128]
    err_col_all = np.concatenate(sampled_err_col)    # [M]
    err_sub_all = np.concatenate(sampled_err_75_150) # [M]
    sst_all = np.concatenate(sampled_sst)            # [M]
    sss_all = np.concatenate(sampled_sss)            # [M]
    ssh_all = np.concatenate(sampled_ssh)            # [M]
    curr_all = np.concatenate(sampled_curr_speed)    # [M]
    wind_all = np.concatenate(sampled_wind_speed)    # [M]
    grad_sst_all = np.concatenate(sampled_grad_sst)  # [M]
    grad_ssh_all = np.concatenate(sampled_grad_ssh)  # [M]
    missing_all = np.concatenate(sampled_missing_frac)# [M]
    seasons_all = np.array(sampled_seasons)          # [M]
    regions_all = np.array(sampled_regions)          # [M]

    print(f"  -> Assembled {len(err_sub_all)} stratified diagnostic verification samples.")

    # -------------------------------------------------------------
    # SECTION A: Depth-Wise Spatial Error Fields & Maps
    # -------------------------------------------------------------
    print("\n[Objective A] Computing Depth-Wise Spatial Error Fields...")
    spatial_errors = compute_time_aggregated_spatial_errors(preds_arr, targets_arr, masks_arr)

    target_eval_depths = [0, 10, 50, 75, 100, 125, 150, 300, 500, 1000]
    eval_depth_indices = [depths.index(d) for d in target_eval_depths]

    # Generate dedicated figures for 100m as explicitly specified:
    # 1. phase1_test2019_error_100m.png
    # 2. phase1_test2019_abs_error_100m.png
    idx_100m = depths.index(100)

    # 1. Signed Error 100m
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    im = ax.imshow(
        spatial_errors["signed_error"][idx_100m],
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap="coolwarm",
        vmin=-2.0,
        vmax=2.0,
    )
    plt.colorbar(im, ax=ax, label="Signed Error (°C) [Prediction - Target]")
    ax.set_title("OceanEmbed Phase-1 2019 Temporal Test — Signed Error at 100 m Depth", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase1_test2019_error_100m.png", dpi=150)
    plt.close(fig)

    # 2. Absolute Error 100m
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    im = ax.imshow(
        spatial_errors["abs_error"][idx_100m],
        origin="lower",
        extent=[lon_min, lon_max, lat_min, lat_max],
        cmap="YlOrRd",
        vmin=0.0,
        vmax=2.5,
    )
    plt.colorbar(im, ax=ax, label="Mean Absolute Error (°C)")
    ax.set_title("OceanEmbed Phase-1 2019 Temporal Test — Absolute Error (MAE) at 100 m Depth", fontsize=11, fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase1_test2019_abs_error_100m.png", dpi=150)
    plt.close(fig)

    # Multi-panel spatial error grid across all requested depths (0, 10, 50, 75, 100, 125, 150, 300, 500, 1000 m)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=150, sharex=True, sharey=True)
    axes = axes.flatten()
    for i, d in enumerate(target_eval_depths):
        d_idx = depths.index(d)
        ax = axes[i]
        mae_map = spatial_errors["abs_error"][d_idx]
        im = ax.imshow(
            mae_map,
            origin="lower",
            extent=[lon_min, lon_max, lat_min, lat_max],
            cmap="inferno",
            vmin=0.0,
            vmax=2.0,
        )
        mean_val = np.nanmean(mae_map)
        ax.set_title(f"{d} m (Mean MAE: {mean_val:.2f} °C)", fontsize=10, fontweight="bold")
        if i >= 5:
            ax.set_xlabel("Lon (°E)")
        if i % 5 == 0:
            ax.set_ylabel("Lat (°N)")
    fig.suptitle("OceanEmbed Phase-1 2019: Multi-Depth Spatial Absolute Error (MAE) Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Mean Absolute Error (°C)")
    fig.savefig(fig_dir / "phase1_test2019_multidepth_abs_error.png", dpi=150)
    plt.close(fig)

    # Multi-panel signed error grid
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=150, sharex=True, sharey=True)
    axes = axes.flatten()
    for i, d in enumerate(target_eval_depths):
        d_idx = depths.index(d)
        ax = axes[i]
        bias_map = spatial_errors["signed_error"][d_idx]
        im = ax.imshow(
            bias_map,
            origin="lower",
            extent=[lon_min, lon_max, lat_min, lat_max],
            cmap="coolwarm",
            vmin=-1.5,
            vmax=1.5,
        )
        mean_bias = np.nanmean(bias_map)
        ax.set_title(f"{d} m (Mean Bias: {mean_bias:+.2f} °C)", fontsize=10, fontweight="bold")
        if i >= 5:
            ax.set_xlabel("Lon (°E)")
        if i % 5 == 0:
            ax.set_ylabel("Lat (°N)")
    fig.suptitle("OceanEmbed Phase-1 2019: Multi-Depth Spatial Signed Bias Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.91, 0.96])
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Signed Error (°C) [Pred - Target]")
    fig.savefig(fig_dir / "phase1_test2019_multidepth_signed_error.png", dpi=150)
    plt.close(fig)

    print("  [DONE] Spatial error maps saved to reports/figures/phase3/.")

    # -------------------------------------------------------------
    # SECTION B: Regional Error Analysis (Arabian Sea vs Bay of Bengal)
    # -------------------------------------------------------------
    print("\n[Objective B] Computing Regional Metrics (Arabian Sea vs Bay of Bengal)...")
    as_mask_4d = np.broadcast_to(as_mask_2d[None, None, :, :], (N, D, H, W)) & (masks_arr == 1.0)
    bob_mask_4d = np.broadcast_to(bob_mask_2d[None, None, :, :], (N, D, H, W)) & (masks_arr == 1.0)

    as_metrics = compute_depth_metrics_with_counts(preds_arr, targets_arr, depths, mask=as_mask_4d.astype(np.float32))
    bob_metrics = compute_depth_metrics_with_counts(preds_arr, targets_arr, depths, mask=bob_mask_4d.astype(np.float32))
    overall_metrics = compute_depth_metrics_with_counts(preds_arr, targets_arr, depths, mask=masks_arr)

    # Plot Regional Comparison across depths
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    ax1.plot([as_metrics["rmse"][d] for d in depths], depths, "o-", color="#1f77b4", label="Arabian Sea")
    ax1.plot([bob_metrics["rmse"][d] for d in depths], depths, "s-", color="#ff7f0e", label="Bay of Bengal")
    ax1.plot([overall_metrics["rmse"][d] for d in depths], depths, "--", color="gray", label="Domain Total")
    ax1.set_ylim(1050, -50)
    ax1.set_xlabel("RMSE (°C)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("RMSE by Depth", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot([as_metrics["mae"][d] for d in depths], depths, "o-", color="#1f77b4", label="Arabian Sea")
    ax2.plot([bob_metrics["mae"][d] for d in depths], depths, "s-", color="#ff7f0e", label="Bay of Bengal")
    ax2.set_ylim(1050, -50)
    ax2.set_xlabel("MAE (°C)")
    ax2.set_title("MAE by Depth", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    ax3.plot([as_metrics["bias"][d] for d in depths], depths, "o-", color="#1f77b4", label="Arabian Sea")
    ax3.plot([bob_metrics["bias"][d] for d in depths], depths, "s-", color="#ff7f0e", label="Bay of Bengal")
    ax3.axvline(0.0, color="k", linestyle=":", alpha=0.6)
    ax3.set_ylim(1050, -50)
    ax3.set_xlabel("Bias (°C)")
    ax3.set_title("Signed Bias by Depth", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("OceanEmbed Phase-1 2019: Regional Error Comparison (Arabian Sea vs Bay of Bengal)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_regional_depth_comparison.png", dpi=150)
    plt.close(fig)

    print("  [DONE] Regional metrics computed.")

    # -------------------------------------------------------------
    # SECTION C: Seasonal Error Analysis (DJF, MAM, JJAS, OND)
    # -------------------------------------------------------------
    print("\n[Objective C] Computing Seasonal Metrics...")
    months = np.array([int(date_regex.search(d).group(1)) for d in all_dates])
    season_indices = {
        "DJF": np.where((months == 12) | (months == 1) | (months == 2))[0],
        "MAM": np.where((months >= 3) & (months <= 5))[0],
        "JJAS": np.where((months >= 6) & (months <= 9))[0],
        "OND": np.where((months >= 10) & (months <= 11))[0],
    }

    seasonal_metrics = {}
    season_colors = {"DJF": "#3498db", "MAM": "#e67e22", "JJAS": "#2ecc71", "OND": "#9b59b6"}

    for s_name, s_idx in season_indices.items():
        seasonal_metrics[s_name] = compute_depth_metrics_with_counts(
            preds_arr[s_idx], targets_arr[s_idx], depths, mask=masks_arr[s_idx]
        )

    # Plot Seasonal RMSE by Depth with focus on 50-150m
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    for s_name, col in season_colors.items():
        s_rmse = [seasonal_metrics[s_name]["rmse"][d] for d in depths]
        ax1.plot(s_rmse, depths, "o-", color=col, label=f"{s_name} (N_days={len(season_indices[s_name])})")
    ax1.set_ylim(1050, -50)
    ax1.set_xlabel("RMSE (°C)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_title("Full Column Seasonal RMSE (0-1000 m)", fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subsurface zoom (30 to 200 m)
    for s_name, col in season_colors.items():
        sub_d = [30, 50, 75, 100, 125, 150, 200]
        s_rmse = [seasonal_metrics[s_name]["rmse"][d] for d in sub_d]
        ax2.plot(s_rmse, sub_d, "o-", color=col, lw=2, label=s_name)
    ax2.set_ylim(210, 25)
    ax2.set_xlabel("RMSE (°C)")
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Subsurface Peak Layer (30-200 m)", fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("OceanEmbed Phase-1 2019: Seasonal Error Profiles Across Depths", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_seasonal_depth_error.png", dpi=150)
    plt.close(fig)

    print("  [DONE] Seasonal metrics computed.")

    # -------------------------------------------------------------
    # SECTION D: Error vs Surface-State Features
    # -------------------------------------------------------------
    print("\n[Objective D] Analyzing Error vs Surface-State Features...")
    # Features to analyze against 75-150m subsurface MAE:
    surface_feature_dict = {
        "SST (°C)": sst_all,
        "SSS (PSU)": sss_all,
        "SSH / SLA (m)": ssh_all,
        "Current Speed (m/s)": curr_all,
        "Wind Speed (m/s)": wind_all,
        "|∇SST| (°C/deg)": grad_sst_all,
        "|∇SSH| (m/deg)": grad_ssh_all,
        "Missing Data Fraction": missing_all,
    }

    feature_binned_results = {}
    for feat_name, feat_vals in surface_feature_dict.items():
        feature_binned_results[feat_name] = compute_binned_statistics(feat_vals, err_sub_all, n_bins=10)

    # Multi-panel binned relationship plot
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), dpi=150)
    axes = axes.flatten()

    for idx, (feat_name, b_res) in enumerate(feature_binned_results.items()):
        ax = axes[idx]
        bins = b_res["bins"]
        if bins:
            x_m = [b["x_mean"] for b in bins]
            y_m = [b["y_mean"] for b in bins]
            y_s = [b["y_std"] for b in bins]
            ax.errorbar(x_m, y_m, yerr=y_s, fmt="o-", color="#2c3e50", ecolor="#bdc3c7", capsize=3, lw=1.5)
            r_val = b_res["overall_pearson_r"]
            ax.set_title(f"{feat_name}\n(r = {r_val:+.3f}, N = {b_res['sample_count']})", fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No finite bins", ha="center")
        ax.set_xlabel(feat_name)
        ax.set_ylabel("Subsurface MAE 75-150m (°C)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("OceanEmbed Phase-1 2019: Subsurface Error (75-150m) vs Surface Physical Variables", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_error_vs_surface_features.png", dpi=150)
    plt.close(fig)

    print("  [DONE] Error vs surface-state feature correlations computed.")

    # -------------------------------------------------------------
    # SECTION E: 128-D Ocean Embedding PCA & Projections
    # -------------------------------------------------------------
    print("\n[Objective E] Performing PCA on 128-D Ocean Embeddings...")
    # Standardize sample size for PCA (sample 10,000 random points)
    pca_n = min(10000, len(emb_all))
    pca_idx = rng.choice(len(emb_all), size=pca_n, replace=False)

    emb_sub = emb_all[pca_idx]
    err_col_sub = err_col_all[pca_idx]
    err_75_150_sub = err_sub_all[pca_idx]
    seasons_sub = seasons_all[pca_idx]
    regions_sub = regions_all[pca_idx]
    sst_sub = sst_all[pca_idx]
    curr_sub = curr_all[pca_idx]

    pca_proj, pca_meta = perform_embedding_pca(emb_sub, n_components=3, random_state=42)
    print(f"  -> PCA Explained Variance: PC1={pca_meta['explained_variance_ratio'][0]:.3f}, PC2={pca_meta['explained_variance_ratio'][1]:.3f}, Total 3-comp={pca_meta['total_explained_variance']:.3f}")

    # Plot 4-panel embedding PCA projection
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), dpi=150)

    # Panel 1: Colored by Subsurface Error (75-150m)
    ax = axes[0, 0]
    sc = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=err_75_150_sub, cmap="plasma", s=10, alpha=0.6, vmin=0.2, vmax=1.8)
    plt.colorbar(sc, ax=ax, label="Subsurface MAE (75-150m) [°C]")
    ax.set_title("Colored by Subsurface Error (75-150 m)", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca_meta['explained_variance_ratio'][0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_meta['explained_variance_ratio'][1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)

    # Panel 2: Colored by Season
    ax = axes[0, 1]
    for s_name, col in season_colors.items():
        s_mask = (seasons_sub == s_name)
        ax.scatter(pca_proj[s_mask, 0], pca_proj[s_mask, 1], c=col, label=s_name, s=10, alpha=0.5)
    ax.set_title("Colored by Season (DJF, MAM, JJAS, OND)", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca_meta['explained_variance_ratio'][0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_meta['explained_variance_ratio'][1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=3)

    # Panel 3: Colored by Region (Arabian Sea vs Bay of Bengal)
    ax = axes[1, 0]
    reg_colors = {"Arabian Sea": "#1f77b4", "Bay of Bengal": "#ff7f0e", "Other": "#7f7f7f"}
    for r_name, col in reg_colors.items():
        r_mask = (regions_sub == r_name)
        ax.scatter(pca_proj[r_mask, 0], pca_proj[r_mask, 1], c=col, label=r_name, s=10, alpha=0.5)
    ax.set_title("Colored by Region (AS vs BoB vs Other)", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca_meta['explained_variance_ratio'][0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_meta['explained_variance_ratio'][1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=3)

    # Panel 4: Colored by Sea Surface Temperature (SST)
    ax = axes[1, 1]
    sc4 = ax.scatter(pca_proj[:, 0], pca_proj[:, 1], c=sst_sub, cmap="Spectral_r", s=10, alpha=0.6)
    plt.colorbar(sc4, ax=ax, label="SST (°C)")
    ax.set_title("Colored by Sea Surface Temperature (SST)", fontweight="bold")
    ax.set_xlabel(f"PC1 ({pca_meta['explained_variance_ratio'][0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_meta['explained_variance_ratio'][1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("OceanEmbed Phase-1: 128-D Ocean Embedding Latent Space (PCA Projection, N=10,000)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_embedding_pca_analysis.png", dpi=150)
    plt.close(fig)

    print("  [DONE] Embedding PCA analysis saved.")

    # -------------------------------------------------------------
    # SECTION F: Representative Temperature Profiles
    # -------------------------------------------------------------
    print("\n[Objective F] Selecting Representative Temperature Profiles...")
    # Find representative profiles from the 2019 dataset:
    # 1. Low-error profile
    # 2. High-error profile
    # 3. Arabian Sea profile
    # 4. Bay of Bengal profile
    # 5. Monsoon profile (JJAS)
    # 6. Non-monsoon profile (DJF)

    # Compute column MAE for each valid ocean pixel across time
    # To be efficient, evaluate candidate days:
    candidate_days = [15, 75, 135, 195, 255, 315]  # spread throughout the year
    case_candidates = []

    for c_day in candidate_days:
        d_date = all_dates[c_day]
        d_pred = preds_arr[c_day]   # [15, H, W]
        d_targ = targets_arr[c_day] # [15, H, W]
        d_mask = masks_arr[c_day]   # [15, H, W]

        # Valid ocean column (all 15 depths valid)
        full_ocean_col = (np.sum(d_mask == 1.0, axis=0) == 15)
        ys, xs = np.where(full_ocean_col)

        for py, px in zip(ys[::10], xs[::10]):
            p_profile = d_pred[:, py, px]
            t_profile = d_targ[:, py, px]
            mae = float(np.mean(np.abs(p_profile - t_profile)))
            rmse = float(np.sqrt(np.mean((p_profile - t_profile)**2)))
            sub_mae = float(np.mean(np.abs(p_profile[idx_75_150] - t_profile[idx_75_150])))
            lat_pt = float(lats[py])
            lon_pt = float(lons[px])
            is_as = as_mask_2d[py, px]
            is_bob = bob_mask_2d[py, px]
            mo = int(d_date.split("-")[1])
            is_monsoon = (mo in [6, 7, 8, 9])

            case_candidates.append({
                "date": d_date,
                "lat": lat_pt,
                "lon": lon_pt,
                "py": int(py),
                "px": int(px),
                "mae": mae,
                "rmse": rmse,
                "sub_mae": sub_mae,
                "is_as": is_as,
                "is_bob": is_bob,
                "is_monsoon": is_monsoon,
                "pred": p_profile.tolist(),
                "target": t_profile.tolist(),
            })

    # Select representative cases
    # 1. Low-error case (overall lowest MAE)
    case_candidates.sort(key=lambda c: c["mae"])
    case_low = case_candidates[0]

    # 2. High-error case (highest subsurface MAE)
    case_high = max(case_candidates, key=lambda c: c["sub_mae"])

    # 3. Arabian Sea case (moderate error, is_as)
    as_cases = [c for c in case_candidates if c["is_as"]]
    case_as = as_cases[len(as_cases) // 2] if as_cases else case_candidates[1]

    # 4. Bay of Bengal case (moderate error, is_bob)
    bob_cases = [c for c in case_candidates if c["is_bob"]]
    case_bob = bob_cases[len(bob_cases) // 2] if bob_cases else case_candidates[2]

    # 5. Monsoon case (JJAS)
    monsoon_cases = [c for c in case_candidates if c["is_monsoon"]]
    case_monsoon = monsoon_cases[len(monsoon_cases) // 2] if monsoon_cases else case_candidates[3]

    # 6. Non-monsoon case (DJF / MAM)
    non_monsoon_cases = [c for c in case_candidates if not c["is_monsoon"]]
    case_non_monsoon = non_monsoon_cases[len(non_monsoon_cases) // 2] if non_monsoon_cases else case_candidates[4]

    representative_cases = {
        "Low-Error Case": case_low,
        "High-Error Subsurface Case": case_high,
        "Arabian Sea Case": case_as,
        "Bay of Bengal Case": case_bob,
        "Monsoon Case (JJAS)": case_monsoon,
        "Non-Monsoon Case": case_non_monsoon,
    }

    # Plot representative profiles
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=150)
    axes = axes.flatten()

    for idx, (title, case) in enumerate(representative_cases.items()):
        ax = axes[idx]
        p_vals = np.array(case["pred"])
        t_vals = np.array(case["target"])
        ax.plot(p_vals, depths, "o-", color="#e74c3c", lw=2, label="OceanEmbed Pred")
        ax.plot(t_vals, depths, "s--", color="#2c3e50", lw=2, label="GLORYS Target")
        ax.set_ylim(1050, -50)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("Depth (m)")
        ax.set_title(
            f"{title}\nDate: {case['date']} | ({case['lat']:.1f}°N, {case['lon']:.1f}°E)\nCol MAE: {case['mae']:.2f}°C | Sub MAE: {case['sub_mae']:.2f}°C",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="lower left")

    fig.suptitle("OceanEmbed Phase-1 2019: Representative Vertical Temperature Profiles (15 Depths)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_representative_profiles.png", dpi=150)
    plt.close(fig)

    print("  [DONE] Representative vertical profiles saved.")

    # -------------------------------------------------------------
    # SECTION G: Missing-Data & Validity Mask Error Diagnostic
    # -------------------------------------------------------------
    print("\n[Objective G] Computing Missing-Data & Validity Mask Diagnostics...")
    # Missing data statistics
    missing_binned = compute_binned_statistics(missing_all, err_sub_all, n_bins=5)

    # Individual surface variable validity impact on error
    # Compute error conditioned on whether SST/SSS/SSH mask == 1 vs 0
    # From subsampled data:
    var_mask_names = ["SST", "SSS", "SSH", "Currents", "Winds"]
    # Plot missing fraction vs error
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Panel 1: Binned Missing Fraction vs Subsurface MAE
    if missing_binned["bins"]:
        x_mf = [b["x_mean"] * 100.0 for b in missing_binned["bins"]]
        y_mf = [b["y_mean"] for b in missing_binned["bins"]]
        y_mf_std = [b["y_std"] for b in missing_binned["bins"]]
        ax1.errorbar(x_mf, y_mf, yerr=y_mf_std, fmt="s-", color="#c0392b", capsize=4, lw=1.8)
        ax1.set_title(f"Subsurface Error vs Surface Missing Fraction\n(Pearson r = {missing_binned['overall_pearson_r']:+.3f}, N = {missing_binned['sample_count']})", fontweight="bold")
    ax1.set_xlabel("Surface Variables Missingness (%)")
    ax1.set_ylabel("Subsurface MAE (75-150m) [°C]")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Distribution of column MAE for zero vs partial missingness
    zero_missing = err_sub_all[missing_all == 0.0]
    some_missing = err_sub_all[missing_all > 0.0]

    box_data = [zero_missing, some_missing]
    labels = [f"Complete Data (N={len(zero_missing)})", f"Missing ≥1 Inputs (N={len(some_missing)})"]
    ax2.boxplot(box_data, patch_artist=True, showmeans=True,
                boxprops=dict(facecolor="#ecf0f1", color="#2c3e50"),
                meanprops=dict(marker="o", markeredgecolor="red", markerfacecolor="red"))
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Subsurface MAE (75-150m) [°C]")
    ax2.set_title("Error Distribution: Complete vs Partially Missing Surface Inputs", fontweight="bold")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("OceanEmbed Phase-1 2019: Missing-Data and Validity Impact Diagnostics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_missing_data_error_diagnostic.png", dpi=150)
    plt.close(fig)

    print("  [DONE] Missing data analysis saved.")

    # -------------------------------------------------------------
    # SECTION H: Dedicated Diagnostic for 75-150m Layer
    # -------------------------------------------------------------
    print("\n[Objective H] Dedicated Subsurface 75-150 m Diagnostic...")
    sub_depths = [75, 100, 125, 150]
    sub_depth_metrics = {
        d: {
            "overall_rmse": overall_metrics["rmse"][d],
            "overall_mae": overall_metrics["mae"][d],
            "overall_bias": overall_metrics["bias"][d],
            "overall_r": overall_metrics["pearson_r"][d],
            "overall_r2": overall_metrics["r2_score"][d],
            "valid_count": overall_metrics["valid_count"][d],
            "as_rmse": as_metrics["rmse"][d],
            "bob_rmse": bob_metrics["rmse"][d],
            "seasonal_rmse": {s: seasonal_metrics[s]["rmse"][d] for s in seasonal_metrics},
        }
        for d in sub_depths
    }

    # Generate dedicated diagnostic figure for 75-150m
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # 1. Depth profile of RMSE and MAE in 75-150m
    ax = axes[0, 0]
    ax.plot([overall_metrics["rmse"][d] for d in sub_depths], sub_depths, "o-", color="#e74c3c", lw=2, label="RMSE")
    ax.plot([overall_metrics["mae"][d] for d in sub_depths], sub_depths, "s-", color="#2980b9", lw=2, label="MAE")
    ax.set_ylim(160, 65)
    ax.set_xlabel("Error (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("RMSE & MAE across 75-150 m Layer", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Regional comparison in 75-150m
    ax = axes[0, 1]
    ax.plot([as_metrics["rmse"][d] for d in sub_depths], sub_depths, "o-", color="#1f77b4", lw=2, label="Arabian Sea RMSE")
    ax.plot([bob_metrics["rmse"][d] for d in sub_depths], sub_depths, "s-", color="#ff7f0e", lw=2, label="Bay of Bengal RMSE")
    ax.set_ylim(160, 65)
    ax.set_xlabel("RMSE (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Regional Comparison (AS vs BoB)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Seasonal comparison in 75-150m
    ax = axes[1, 0]
    for s_name, col in season_colors.items():
        s_rmse = [seasonal_metrics[s_name]["rmse"][d] for d in sub_depths]
        ax.plot(s_rmse, sub_depths, "o-", color=col, lw=1.8, label=s_name)
    ax.set_ylim(160, 65)
    ax.set_xlabel("RMSE (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Seasonal Comparison across 75-150 m", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Bias across 75-150m
    ax = axes[1, 1]
    ax.plot([overall_metrics["bias"][d] for d in sub_depths], sub_depths, "o-", color="#8e44ad", lw=2, label="Domain Bias")
    ax.plot([as_metrics["bias"][d] for d in sub_depths], sub_depths, "--", color="#1f77b4", lw=1.5, label="Arabian Sea Bias")
    ax.plot([bob_metrics["bias"][d] for d in sub_depths], sub_depths, "--", color="#ff7f0e", lw=1.5, label="Bay of Bengal Bias")
    ax.axvline(0.0, color="k", linestyle=":", alpha=0.6)
    ax.set_ylim(160, 65)
    ax.set_xlabel("Signed Bias (°C)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Signed Bias across 75-150 m", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle("OceanEmbed Phase-1 2019: Focused Diagnostic of 75-150 m Upper-Interior Layer", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "phase3_subsurface_75_150m_diagnostic.png", dpi=150)
    plt.close(fig)

    print("  [DONE] Dedicated 75-150 m diagnostics saved.")

    # -------------------------------------------------------------
    # SECTION I: Machine-Readable Report Serialization (JSON)
    # -------------------------------------------------------------
    print("\n[Serialization] Compiling Machine-Readable JSON Report...")

    # Calculate layer summaries
    def calc_mean_stat(metric_dict, depth_list):
        return float(np.mean([metric_dict[d] for d in depth_list]))

    json_report = {
        "analysis_scope": "OceanEmbed Phase-3 Diagnostic Error Analysis",
        "evaluation_dataset": "2019 Temporal Test Set (365 calendar days)",
        "model_checkpoint": str(p1_ckpt),
        "checkpoint_sha256": actual_p1,
        "phase2_checkpoint_sha256_unmodified": actual_p2,
        "overall_column_metrics": {
            "mean_rmse": calc_mean_stat(overall_metrics["rmse"], depths),
            "mean_mae": calc_mean_stat(overall_metrics["mae"], depths),
            "mean_bias": calc_mean_stat(overall_metrics["bias"], depths),
            "mean_pearson_r": calc_mean_stat(overall_metrics["pearson_r"], depths),
            "mean_r2": calc_mean_stat(overall_metrics["r2_score"], depths),
            "subsurface_75_150m_rmse": calc_mean_stat(overall_metrics["rmse"], sub_depths),
            "deep_500_1000m_rmse": calc_mean_stat(overall_metrics["rmse"], [500, 700, 1000]),
            "depth_wise": {
                str(d): {
                    "rmse": overall_metrics["rmse"][d],
                    "mae": overall_metrics["mae"][d],
                    "bias": overall_metrics["bias"][d],
                    "pearson_r": overall_metrics["pearson_r"][d],
                    "r2_score": overall_metrics["r2_score"][d],
                    "valid_sample_count": overall_metrics["valid_count"][d],
                }
                for d in depths
            },
        },
        "regional_analysis": {
            "Arabian_Sea": {
                "mean_rmse": calc_mean_stat(as_metrics["rmse"], depths),
                "mean_mae": calc_mean_stat(as_metrics["mae"], depths),
                "mean_bias": calc_mean_stat(as_metrics["bias"], depths),
                "subsurface_75_150m_rmse": calc_mean_stat(as_metrics["rmse"], sub_depths),
                "depth_wise": {
                    str(d): {
                        "rmse": as_metrics["rmse"][d],
                        "mae": as_metrics["mae"][d],
                        "bias": as_metrics["bias"][d],
                        "pearson_r": as_metrics["pearson_r"][d],
                        "r2_score": as_metrics["r2_score"][d],
                        "valid_sample_count": as_metrics["valid_count"][d],
                    }
                    for d in depths
                },
            },
            "Bay_of_Bengal": {
                "mean_rmse": calc_mean_stat(bob_metrics["rmse"], depths),
                "mean_mae": calc_mean_stat(bob_metrics["mae"], depths),
                "mean_bias": calc_mean_stat(bob_metrics["bias"], depths),
                "subsurface_75_150m_rmse": calc_mean_stat(bob_metrics["rmse"], sub_depths),
                "depth_wise": {
                    str(d): {
                        "rmse": bob_metrics["rmse"][d],
                        "mae": bob_metrics["mae"][d],
                        "bias": bob_metrics["bias"][d],
                        "pearson_r": bob_metrics["pearson_r"][d],
                        "r2_score": bob_metrics["r2_score"][d],
                        "valid_sample_count": bob_metrics["valid_count"][d],
                    }
                    for d in depths
                },
            },
        },
        "seasonal_analysis": {
            s: {
                "num_days": len(season_indices[s]),
                "mean_rmse": calc_mean_stat(seasonal_metrics[s]["rmse"], depths),
                "mean_mae": calc_mean_stat(seasonal_metrics[s]["mae"], depths),
                "subsurface_75_150m_rmse": calc_mean_stat(seasonal_metrics[s]["rmse"], sub_depths),
                "depth_rmse": {str(d): seasonal_metrics[s]["rmse"][d] for d in depths},
                "valid_sample_count": {str(d): seasonal_metrics[s]["valid_count"][d] for d in depths},
            }
            for s in seasonal_metrics
        },
        "surface_feature_correlations": {
            feat_name: {
                "pearson_r": b_res["overall_pearson_r"],
                "sample_count": b_res["sample_count"],
                "bins": b_res["bins"],
            }
            for feat_name, b_res in feature_binned_results.items()
        },
        "embedding_pca_metadata": pca_meta,
        "representative_profiles": {
            k: {
                "date": v["date"],
                "lat": v["lat"],
                "lon": v["lon"],
                "column_mae": v["mae"],
                "subsurface_mae": v["sub_mae"],
            }
            for k, v in representative_cases.items()
        },
        "subsurface_75_150m_focus": sub_depth_metrics,
        "missing_data_analysis": {
            "missing_fraction_pearson_r": missing_binned["overall_pearson_r"],
            "zero_missing_mae_mean": float(np.mean(zero_missing)) if len(zero_missing) > 0 else None,
            "partial_missing_mae_mean": float(np.mean(some_missing)) if len(some_missing) > 0 else None,
            "zero_missing_sample_count": int(len(zero_missing)),
            "partial_missing_sample_count": int(len(some_missing)),
        },
    }

    report_path = PROJECT_ROOT / "evaluation" / "results" / "phase3_error_analysis_2019.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    print(f"  [PASS] Saved JSON report to: {report_path}")
    print("\n" + "=" * 85)
    print("PHASE-3 SCIENTIFIC ERROR ANALYSIS COMPLETE.")
    print("=" * 85)


if __name__ == "__main__":
    main()
