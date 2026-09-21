"""
scripts/train_phase4a.py
------------------------
Phase-4A Vertical Gradient-Aware Loss Training Script.

A strictly controlled ablation testing whether vertical temperature gradient
alignment improves the 75–150 m subsurface region without degrading the column.

Configuration:
  - Input: 14 channels (7 physical surface variables + 7 validity masks) [B, 14, 101, 241]
  - Output: 15 standard ocean depths [B, 15, 101, 241]
  - Architecture: Identical OceanEmbedNet (525,040 parameters, 128-D embedding)
  - Loss: L_total = L_MSE + lambda_grad * L_gradient (lambda_grad = 2.0, L1 gradient loss)
  - Train: 2015-01-01 to 2017-12-31 (1,096 days)
  - Validation: 2018-01-01 to 2018-12-31 (365 days)
  - 2019 Test: LOCKED (Strictly untouched)
  - Precision: bf16 AMP on GPU
  - Optimizer: Adam (lr=1e-4, grad_accumulation=2, grad_clip=1.0)
  - Schedule: CosineAnnealingLR (eta_min=1e-6, max_epochs=100)
  - Checkpoint: checkpoints/phase4a/best.pt (Phase-1 & Phase-2 checkpoints are read-only)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.losses import GradientAwareLoss, MaskedMSELoss
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, SyntheticOceanDataset, create_dataloader
from utils.config import load_config


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_checkpoint_immutability():
    """Verify that Phase-1 and Phase-2 checkpoints are 100% untouched."""
    p1_ckpt = PROJECT_ROOT / "checkpoints" / "phase1" / "best.pt"
    p2_ckpt = PROJECT_ROOT / "checkpoints" / "phase2" / "best.pt"
    expected_p1 = "f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b"
    expected_p2 = "8bd977fc87412d5209081a218745325f214e384cc821d6b951ec51fe50c34d8e"

    actual_p1 = hashlib.sha256(open(p1_ckpt, "rb").read()).hexdigest()
    actual_p2 = hashlib.sha256(open(p2_ckpt, "rb").read()).hexdigest()

    assert actual_p1 == expected_p1, f"Phase-1 checkpoint corrupted! Got {actual_p1}"
    assert actual_p2 == expected_p2, f"Phase-2 checkpoint corrupted! Got {actual_p2}"
    print(f"[Integrity Gate] Phase-1 best.pt SHA256 verified invariant: {actual_p1}")
    print(f"[Integrity Gate] Phase-2 best.pt SHA256 verified invariant: {actual_p2}")


def run_training(
    epochs: Optional[int] = None,
    use_synthetic: bool = False,
    num_synthetic_samples: int = 32,
) -> None:
    print("=" * 75)
    print("OCEANEMBED PHASE-4A: VERTICAL GRADIENT-AWARE LOSS ABLATION TRAINING")
    print("=" * 75)

    # 1. Pre-training integrity verification
    verify_checkpoint_immutability()

    train_cfg = load_config("train_phase4a")
    model_cfg = load_config("model")
    data_cfg = load_config("data")

    set_seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = train_cfg.precision
    use_amp = (precision in ["bf16", "fp16"]) and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    print(f"[Training] Device: {device} | Precision: {precision} | AMP: {use_amp}")
    print(f"[Training] Model in_channels: {model_cfg.in_channels} (Expected 14)")

    # Datasets (14 channels)
    if use_synthetic:
        print("[Training] Using synthetic data mode.")
        train_ds = SyntheticOceanDataset(
            num_samples=num_synthetic_samples,
            in_channels=14,
            seed=train_cfg.seed,
        )
        val_ds = SyntheticOceanDataset(
            num_samples=max(8, num_synthetic_samples // 4),
            in_channels=14,
            seed=train_cfg.seed + 1,
        )
    else:
        train_ds = OceanEmbedDataset(split="train", in_channels=14, fallback_to_synthetic=False)
        val_ds = OceanEmbedDataset(split="val", in_channels=14, fallback_to_synthetic=False)

    print(f"[Training] Train samples (2015-2017): {len(train_ds)}")
    print(f"[Training] Val samples (2018):         {len(val_ds)}")
    print("[Training] 2019 Test Set:             LOCKED (Untouched)")

    train_loader = create_dataloader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=train_cfg.shuffle_train,
        num_workers=0,  # 0 workers avoids Windows multiprocessing overhead
        pin_memory=train_cfg.pin_memory and (device.type == "cuda"),
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=train_cfg.pin_memory and (device.type == "cuda"),
    )

    # Model Initialization
    model = OceanEmbedNet(model_cfg).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Training] OceanEmbedNet parameters: {param_count:,} (Expected: 525,040)")
    assert param_count == 525040, f"Expected 525,040 params, got {param_count}"

    # Criterion: GradientAwareLoss
    lambda_grad = float(train_cfg.gradient_loss.get("lambda_grad", 2.0))
    grad_type = str(train_cfg.gradient_loss.get("type", "l1"))
    grad_enabled = bool(train_cfg.gradient_loss.get("enabled", True))
    depths = list(data_cfg.depths_m)

    criterion = GradientAwareLoss(
        depths=depths,
        lambda_grad=lambda_grad,
        gradient_loss_type=grad_type,
        enabled=grad_enabled,
    ).to(device)

    print(f"[Loss Config] GradientAwareLoss: lambda={lambda_grad}, type={grad_type}, enabled={grad_enabled}")

    # Optimiser & Scheduler
    if train_cfg.optimizer.lower() == "adamw":
        optimiser = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    else:
        optimiser = optim.Adam(model.parameters(), lr=train_cfg.learning_rate)

    max_epochs = epochs or train_cfg.epochs
    scheduler = CosineAnnealingLR(optimiser, T_max=max_epochs, eta_min=train_cfg.lr_min)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("evaluation/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "phase4a_training_history.json"

    best_val_loss = float("inf")  # Monitored: val_mse (Phase-1 stopping criterion)
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = checkpoint_dir / "best.pt"
    history = []
    train_start_time = time.time()

    print(f"\n[Training] Beginning execution for up to {max_epochs} epochs (patience={train_cfg.patience})...\n")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_total_accum = 0.0
        train_loss_mse_accum = 0.0
        train_loss_grad_accum = 0.0
        n_batches = 0

        optimiser.zero_grad()

        for step, (x, y, meta) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            m = meta.get("target_mask", None)
            if m is not None and isinstance(m, torch.Tensor):
                m = m.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                out = model(x)
                loss, loss_dict = criterion(out["temperature"], y, mask=m)
                scaled_loss = loss / train_cfg.grad_accumulation

            if precision == "fp16":
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if (step + 1) % train_cfg.grad_accumulation == 0 or (step + 1) == len(train_loader):
                if precision == "fp16":
                    if train_cfg.gradient_clip > 0:
                        scaler.unscale_(optimiser)
                        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
                    scaler.step(optimiser)
                    scaler.update()
                else:
                    if train_cfg.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), train_cfg.gradient_clip)
                    optimiser.step()
                optimiser.zero_grad()

            train_loss_total_accum += loss_dict["loss_total"]
            train_loss_mse_accum += loss_dict["loss_mse"]
            train_loss_grad_accum += loss_dict["loss_grad"]
            n_batches += 1

        avg_train_total = train_loss_total_accum / max(1, n_batches)
        avg_train_mse = train_loss_mse_accum / max(1, n_batches)
        avg_train_grad = train_loss_grad_accum / max(1, n_batches)

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Validation evaluation
        model.eval()
        val_loss_total_accum = 0.0
        val_loss_mse_accum = 0.0
        val_loss_grad_accum = 0.0
        val_batches = 0

        with torch.no_grad():
            for x_v, y_v, meta_v in val_loader:
                x_v = x_v.to(device)
                y_v = y_v.to(device)
                m_v = meta_v.get("target_mask", None)
                if m_v is not None and isinstance(m_v, torch.Tensor):
                    m_v = m_v.to(device)

                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    out_v = model(x_v)
                    _, v_loss_dict = criterion(out_v["temperature"], y_v, mask=m_v)

                val_loss_total_accum += v_loss_dict["loss_total"]
                val_loss_mse_accum += v_loss_dict["loss_mse"]
                val_loss_grad_accum += v_loss_dict["loss_grad"]
                val_batches += 1

        avg_val_total = val_loss_total_accum / max(1, val_batches)
        avg_val_mse = val_loss_mse_accum / max(1, val_batches)
        avg_val_grad = val_loss_grad_accum / max(1, val_batches)

        epoch_dur = time.time() - epoch_start
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

        # Phase-1 validation stopping criterion: val_mse
        is_best = avg_val_mse < best_val_loss
        marker = " *" if is_best else ""

        print(
            f"Epoch [{epoch:03d}/{max_epochs:03d}] "
            f"Train Total: {avg_train_total:.4f} (MSE: {avg_train_mse:.4f}, Grad: {avg_train_grad:.4f}) | "
            f"Val Total: {avg_val_total:.4f} (MSE: {avg_val_mse:.4f}, Grad: {avg_val_grad:.4f}){marker} | "
            f"LR: {current_lr:.2e} | "
            f"Dur: {epoch_dur:.1f}s | "
            f"Peak VRAM: {peak_vram_mb:.1f} MB",
            flush=True,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss_total": float(avg_train_total),
            "train_loss_mse": float(avg_train_mse),
            "train_loss_grad": float(avg_train_grad),
            "val_loss_total": float(avg_val_total),
            "val_loss_mse": float(avg_val_mse),
            "val_loss_grad": float(avg_val_grad),
            "learning_rate": float(current_lr),
            "duration_s": float(epoch_dur),
            "peak_vram_mb": float(peak_vram_mb),
        }
        history.append(epoch_record)

        # Checkpoint if best based on validation MSE (Phase-1 criterion)
        if is_best:
            best_val_loss = avg_val_mse
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimiser.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss_total": avg_train_total,
                    "train_loss_mse": avg_train_mse,
                    "train_loss_grad": avg_train_grad,
                    "val_loss_total": avg_val_total,
                    "val_loss_mse": avg_val_mse,
                    "val_loss_grad": avg_val_grad,
                    "lambda_grad": lambda_grad,
                    "gradient_loss_type": grad_type,
                    "in_channels": 14,
                    "out_depths": 15,
                },
                best_ckpt_path,
            )
        else:
            patience_counter += 1

        # Also save latest checkpoint for resilience
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimiser.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch_record": epoch_record,
            },
            checkpoint_dir / "latest.pt",
        )

        # Persist history JSON incrementally
        with open(history_path, "w") as f:
            json.dump(
                {
                    "experiment": "phase4a_gradient_loss",
                    "best_epoch": best_epoch,
                    "best_val_mse": float(best_val_loss),
                    "lambda_grad": lambda_grad,
                    "gradient_loss_type": grad_type,
                    "history": history,
                },
                f,
                indent=2,
            )

        if patience_counter >= train_cfg.patience:
            print(f"\n[Early Stopping] Triggered at epoch {epoch}. No val MSE improvement for {patience_counter} epochs.")
            break

    total_time_m = (time.time() - train_start_time) / 60.0
    print("\n" + "=" * 75)
    print(f"PHASE-4A TRAINING COMPLETE in {total_time_m:.1f} minutes")
    print(f"Best Checkpoint: Epoch {best_epoch} with Val MSE = {best_val_loss:.4f} degC^2")
    print(f"Saved to: {best_ckpt_path}")
    print("=" * 75)

    # Post-training integrity verification
    verify_checkpoint_immutability()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed Phase-4A Training")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic data")
    args = parser.parse_args()

    run_training(epochs=args.epochs, use_synthetic=args.synthetic)
