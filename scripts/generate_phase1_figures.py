"""
scripts/generate_phase1_figures.py
----------------------------------
Generates publication-grade diagnostic plots and verification artifacts
for the Phase-1 OceanEmbed evaluation on the 2018 validation dataset:

1. depth_metrics_comparison.png:
   - Depth-wise RMSE and R^2 comparing OceanEmbed vs Climatology vs Ridge across all 15 depths.
2. spatial_error_maps.png:
   - Spatial error (MAE / RMSE) maps across the North Indian Ocean at 0m, 100m, 500m, 1000m.
3. representative_profiles.png:
   - Reconstructed vertical temperature profiles vs ground-truth across Arabian Sea, Bay of Bengal, and Equatorial Indian Ocean.
4. loss_curves.png:
   - Epoch-by-epoch training and validation loss curves, learning rate trajectory, and best-epoch indicator.
5. embedding_distribution.png & embedding_verification.json:
   - 128-dimensional latent representation extraction, spatial variance per channel, and mode-collapse verification.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, create_dataloader
from utils.config import load_config


def plot_loss_curves(history_path: Path, output_path: Path):
    if not history_path.exists():
        print(f"Warning: {history_path} does not exist, skipping loss curve plot.")
        return

    with open(history_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    history = data.get("history", [])
    if not history:
        print("Empty history in loss record.")
        return

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    lrs = [h["learning_rate"] for h in history]
    best_epoch = data.get("best_epoch", epochs[np.argmin(val_loss)])
    best_val = data.get("best_val_loss", min(val_loss))

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

    color_train = "#1f77b4"
    color_val = "#d62728"
    color_lr = "#2ca02c"

    ax1.plot(epochs, train_loss, label="Training Loss (Masked MSE)", color=color_train, linewidth=2)
    ax1.plot(epochs, val_loss, label="Validation Loss (Masked MSE)", color=color_val, linewidth=2)
    ax1.axvline(best_epoch, color="#7f7f7f", linestyle="--", alpha=0.8, label=f"Best Epoch ({best_epoch}: {best_val:.4f})")
    ax1.scatter([best_epoch], [best_val], color=color_val, s=80, zorder=5)

    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss (°C²)", fontsize=12, fontweight="bold")
    ax1.set_title("OceanEmbed Phase-1 Training & Validation Loss Dynamics", fontsize=13, fontweight="bold", pad=12)

    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, label="Learning Rate", color=color_lr, linestyle=":", alpha=0.7, linewidth=1.5)
    ax2.set_ylabel("Learning Rate", fontsize=12, fontweight="bold", color=color_lr)
    ax2.tick_params(axis="y", labelcolor=color_lr)
    ax2.grid(False)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right", frameon=True)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss curve to {output_path}")


def plot_depth_metrics(
    oceanembed_path: Path,
    clim_path: Path,
    ridge_path: Path,
    output_path: Path,
):
    with open(oceanembed_path, "r", encoding="utf-8") as f:
        oe_res = json.load(f)["metrics"]
    with open(clim_path, "r", encoding="utf-8") as f:
        clim_res = json.load(f)
    with open(ridge_path, "r", encoding="utf-8") as f:
        ridge_res = json.load(f)

    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    depth_keys = [str(d) for d in depths]

    oe_rmse = [float(oe_res["rmse"].get(str(d), oe_res["rmse"].get(d))) for d in depths]
    clim_rmse = [float(clim_res["rmse"][k]) for k in depth_keys]
    ridge_rmse = [float(ridge_res["rmse"][k]) for k in depth_keys]

    oe_r2 = [float(oe_res["r2_score"].get(str(d), oe_res["r2_score"].get(d))) for d in depths]
    clim_r2 = [float(clim_res.get("r2_score", clim_res.get("r2"))[k]) for k in depth_keys]
    ridge_r2 = [float(ridge_res.get("r2_score", ridge_res.get("r2"))[k]) for k in depth_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), dpi=300)

    # 1. Depth vs RMSE
    ax1.plot(oe_rmse, depths, marker="o", color="#1f77b4", linewidth=2.2, label="OceanEmbed (Dual-Path)")
    ax1.plot(clim_rmse, depths, marker="s", color="#ff7f0e", linewidth=1.8, linestyle="--", label="Climatology Baseline")
    ax1.plot(ridge_rmse, depths, marker="^", color="#2ca02c", linewidth=1.8, linestyle="-.", label="Chunked Ridge Baseline")
    ax1.invert_yaxis()
    ax1.set_xlabel("RMSE (°C)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Depth (m)", fontsize=11, fontweight="bold")
    ax1.set_title("Root Mean Square Error by Depth", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, alpha=0.3)

    # 2. Depth vs R^2
    ax2.plot(oe_r2, depths, marker="o", color="#1f77b4", linewidth=2.2, label="OceanEmbed (Dual-Path)")
    ax2.plot(clim_r2, depths, marker="s", color="#ff7f0e", linewidth=1.8, linestyle="--", label="Climatology Baseline")
    ax2.plot(ridge_r2, depths, marker="^", color="#2ca02c", linewidth=1.8, linestyle="-.", label="Chunked Ridge Baseline")
    ax2.invert_yaxis()
    ax2.set_xlabel("Coefficient of Determination ($R^2$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Depth (m)", fontsize=11, fontweight="bold")
    ax2.set_title("Variance Explained ($R^2$) by Depth", fontsize=12, fontweight="bold")
    ax2.set_xlim(-1.5, 1.05)
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.7)
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Phase-1 Model Evaluation vs Baselines (2018 Validation Split, 365 Days)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved depth metrics comparison to {output_path}")


def generate_spatial_and_profile_figures(
    checkpoint_path: Path,
    spatial_map_path: Path,
    profiles_path: Path,
    embedding_plot_path: Path,
    embedding_json_path: Path,
):
    data_cfg = load_config("data")
    model_cfg = load_config("model")
    depths = list(data_cfg.depths_m)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = OceanEmbedNet(model_cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = OceanEmbedDataset(split="val", fallback_to_synthetic=False)
    loader = create_dataloader(ds, batch_size=8, shuffle=False)

    all_preds = []
    all_targets = []
    all_masks = []
    sample_embeddings = []

    with torch.no_grad():
        for i, (x, y, meta) in enumerate(loader):
            x = x.to(device)
            out = model(x)
            pred = out["temperature"].cpu().numpy()
            target = y.numpy()
            mask = meta["target_mask"].numpy() if "target_mask" in meta else np.ones_like(target)

            all_preds.append(pred)
            all_targets.append(target)
            all_masks.append(mask)

            if i < 4:
                sample_embeddings.append(out["embedding"].cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)      # [N, 15, H, W]
    targets_arr = np.concatenate(all_targets, axis=0)  # [N, 15, H, W]
    masks_arr = np.concatenate(all_masks, axis=0)      # [N, 15, H, W]
    embeddings_arr = np.concatenate(sample_embeddings, axis=0) # [B_sub, 128, H, W]

    # --- 1. SPATIAL ERROR MAPS ---
    # Compute MAE across time dimension N for each pixel [15, H, W]
    abs_err = np.abs(preds_arr - targets_arr)
    valid_mask = (masks_arr == 1.0) & np.isfinite(abs_err)
    abs_err[~valid_mask] = np.nan
    mean_mae_map = np.nanmean(abs_err, axis=0)  # [15, H, W]

    selected_depth_indices = [0, 7, 12, 14] # 0m, 100m, 500m, 1000m
    selected_depth_names = ["Surface (0 m)", "Thermocline (100 m)", "Intermediate (500 m)", "Deep (1000 m)"]

    domain = getattr(data_cfg, "domain", {})
    lat_min = domain.get("lat_min", 5.0) if isinstance(domain, dict) else getattr(domain, "lat_min", 5.0)
    lat_max = domain.get("lat_max", 30.0) if isinstance(domain, dict) else getattr(domain, "lat_max", 30.0)
    lon_min = domain.get("lon_min", 45.0) if isinstance(domain, dict) else getattr(domain, "lon_min", 45.0)
    lon_max = domain.get("lon_max", 105.0) if isinstance(domain, dict) else getattr(domain, "lon_max", 105.0)

    lats = np.linspace(lat_min, lat_max, 101)
    lons = np.linspace(lon_min, lon_max, 241)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=300)
    for ax, d_idx, name in zip(axes.flat, selected_depth_indices, selected_depth_names):
        data = mean_mae_map[d_idx]
        masked_data = np.ma.masked_invalid(data)
        cmap = plt.cm.plasma
        im = ax.pcolormesh(lon_grid, lat_grid, masked_data, cmap=cmap, vmin=0, vmax=2.5, shading="auto")
        ax.set_title(f"Mean Absolute Error — {name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        fig.colorbar(im, ax=ax, label="MAE (°C)", fraction=0.046, pad=0.04)

    fig.suptitle("Spatial Error Distribution across North Indian Ocean (2018 Validation Split)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    spatial_map_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(spatial_map_path)
    plt.close()
    print(f"Saved spatial error maps to {spatial_map_path}")

    # --- 2. REPRESENTATIVE VERTICAL PROFILES ---
    # Locations:
    # 1. Arabian Sea: ~15°N, 65°E (lat idx: (15-5)*4 = 40, lon idx: (65-45)*4 = 80)
    # 2. Bay of Bengal: ~15°N, 90°E (lat idx: 40, lon idx: 180)
    # 3. Equatorial IO: ~6°N, 75°E (lat idx: 4, lon idx: 120)
    locs = [
        ("Central Arabian Sea (15°N, 65°E)", 40, 80),
        ("Bay of Bengal (15°N, 90°E)", 40, 180),
        ("Equatorial Indian Ocean (6°N, 75°E)", 4, 120),
    ]

    day_idx = 180  # mid-year (summer monsoon)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), dpi=300)

    for ax, (title, r_idx, c_idx) in zip(axes, locs):
        p_prof = preds_arr[day_idx, :, r_idx, c_idx]
        t_prof = targets_arr[day_idx, :, r_idx, c_idx]
        m_prof = masks_arr[day_idx, :, r_idx, c_idx]

        valid_idx = (m_prof == 1.0) & np.isfinite(t_prof)
        valid_depths = np.array(depths)[valid_idx]
        valid_p = p_prof[valid_idx]
        valid_t = t_prof[valid_idx]

        ax.plot(valid_p, valid_depths, marker="o", color="#1f77b4", linewidth=2.2, label="OceanEmbed Reconstruction")
        ax.plot(valid_t, valid_depths, marker="s", color="#d62728", linewidth=2.0, linestyle="--", label="Ground Truth (GLORYS)")
        ax.invert_yaxis()
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperature (°C)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Depth (m)", fontsize=10, fontweight="bold")
        ax.legend(loc="lower left", frameon=True)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Representative Vertical Temperature Profiles — Day {day_idx} (Mid-2018)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    profiles_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(profiles_path)
    plt.close()
    print(f"Saved representative profiles to {profiles_path}")

    # --- 3. EMBEDDING EXTRACTION VERIFICATION ---
    # Shape: [B_sub, 128, H, W]
    channel_stds = np.std(embeddings_arr, axis=(0, 2, 3))
    channel_means = np.mean(embeddings_arr, axis=(0, 2, 3))
    total_active_channels = int(np.sum(channel_stds > 1e-4))
    mean_std = float(np.mean(channel_stds))
    min_std = float(np.min(channel_stds))
    max_std = float(np.max(channel_stds))

    emb_results = {
        "embedding_dim": 128,
        "sample_tensors_analyzed": int(embeddings_arr.shape[0]),
        "total_active_channels": total_active_channels,
        "mean_spatial_std": mean_std,
        "min_spatial_std": min_std,
        "max_spatial_std": max_std,
        "mode_collapse_detected": bool(total_active_channels < 100),
    }

    with open(embedding_json_path, "w", encoding="utf-8") as f:
        json.dump(emb_results, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    ax1.bar(range(128), channel_stds, color="#2b5c8f", width=0.8)
    ax1.set_xlabel("Embedding Channel Index (0–127)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Spatial Standard Deviation", fontsize=11, fontweight="bold")
    ax1.set_title("Spatial Feature Variance Across 128 Latent Channels", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # 2D PCA or average spatial activation map
    avg_activation_map = np.mean(embeddings_arr[0], axis=0) # [H, W]
    im = ax2.pcolormesh(lon_grid, lat_grid, avg_activation_map, cmap="viridis", shading="auto")
    ax2.set_xlabel("Longitude (°E)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Latitude (°N)", fontsize=11, fontweight="bold")
    ax2.set_title("Mean Latent Activation Map Z (Spatial Domain)", fontsize=12, fontweight="bold")
    fig.colorbar(im, ax=ax2, label="Activation Intensity")

    fig.suptitle("OceanEmbed 128-Dimensional Explicit Latent Embedding Verification", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    embedding_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(embedding_plot_path)
    plt.close()
    print(f"Saved embedding distribution to {embedding_plot_path}")
    print(f"Saved embedding metrics to {embedding_json_path}")


def main():
    results_dir = PROJECT_ROOT / "evaluation" / "results"
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    history_p = results_dir / "phase1_training_history.json"
    loss_curve_p = figures_dir / "loss_curves.png"
    plot_loss_curves(history_p, loss_curve_p)

    oceanembed_p = results_dir / "oceanembed_val2018_metrics.json"
    clim_p = results_dir / "climatology_val2018_metrics.json"
    ridge_p = results_dir / "ridge_val2018_metrics.json"
    depth_plot_p = figures_dir / "depth_metrics_comparison.png"

    if oceanembed_p.exists() and clim_p.exists() and ridge_p.exists():
        plot_depth_metrics(oceanembed_p, clim_p, ridge_p, depth_plot_p)

    ckpt_p = PROJECT_ROOT / "checkpoints" / "phase1" / "best.pt"
    spatial_map_p = figures_dir / "spatial_error_maps.png"
    profiles_p = figures_dir / "representative_profiles.png"
    embedding_plot_p = figures_dir / "embedding_distribution.png"
    embedding_json_p = results_dir / "embedding_verification.json"

    if ckpt_p.exists():
        generate_spatial_and_profile_figures(ckpt_p, spatial_map_p, profiles_p, embedding_plot_p, embedding_json_p)


if __name__ == "__main__":
    main()
