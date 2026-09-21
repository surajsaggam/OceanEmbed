"""
scripts/train_phase1.py
-----------------------
Training script for Phase-1 OceanEmbed baseline model.

Scientific and training protocol:
  - Input: 14-channel daily surface observations [B, 14, H, W]
  - Target: 15 standard ocean depths from GLORYS [B, 15, H, W]
  - Loss: Uniform depth-wise masked MSE over valid ocean pixels (Phase-1 baseline)
  - Precision: bf16 mixed precision on NVIDIA RTX 5060 GPU
  - Validation: Evaluated at each epoch on validation split (2022)
  - Checkpointing: Saves best validation checkpoints to checkpoints/phase1/
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.losses import MaskedMSELoss
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, SyntheticOceanDataset, create_dataloader
from utils.config import load_config


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    epochs: Optional[int] = None,
    use_synthetic: bool = False,
    num_synthetic_samples: int = 32,
) -> None:
    train_cfg = load_config("train")
    model_cfg = load_config("model")
    data_cfg = load_config("data")

    set_seed(train_cfg.seed)

    # Device & Precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = train_cfg.precision
    use_amp = (precision in ["bf16", "fp16"]) and (device.type == "cuda")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    print(f"[Training] Device: {device} | Precision: {precision} | AMP: {use_amp}")

    # Datasets
    if use_synthetic:
        print("[Training] Using synthetic data mode for pipeline verification.")
        train_ds = SyntheticOceanDataset(num_samples=num_synthetic_samples, seed=train_cfg.seed)
        val_ds = SyntheticOceanDataset(num_samples=max(8, num_synthetic_samples // 4), seed=train_cfg.seed + 1)
    else:
        train_ds = OceanEmbedDataset(split="train", fallback_to_synthetic=True)
        val_ds = OceanEmbedDataset(split="val", fallback_to_synthetic=True)

    train_loader = create_dataloader(
        train_ds,
        batch_size=train_cfg.batch_size,
        shuffle=train_cfg.shuffle_train,
        num_workers=0,  # 0 workers avoids Windows multiprocessing overhead in scripts
        pin_memory=train_cfg.pin_memory and (device.type == "cuda"),
    )
    val_loader = create_dataloader(
        val_ds,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=train_cfg.pin_memory and (device.type == "cuda"),
    )

    # Model
    model = OceanEmbedNet(model_cfg).to(device)
    criterion = MaskedMSELoss()

    # Optimiser
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
    history_path = results_dir / "phase1_training_history.json"

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_ckpt_path = checkpoint_dir / "best.pt"
    history = []
    train_start_time = time.time()

    print(f"[Training] Starting {max_epochs} epochs...")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_accum = 0.0
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
                loss = criterion(out["temperature"], y, mask=m)
                loss = loss / train_cfg.grad_accumulation

            if precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

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

            train_loss_accum += loss.item() * train_cfg.grad_accumulation
            n_batches += 1

        avg_train_loss = train_loss_accum / max(1, n_batches)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # Validation
        model.eval()
        val_loss_accum = 0.0
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
                    v_loss = criterion(out_v["temperature"], y_v, mask=m_v)
                val_loss_accum += v_loss.item()
                val_batches += 1

        avg_val_loss = val_loss_accum / max(1, val_batches)
        epoch_dur = time.time() - epoch_start
        peak_vram_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0

        print(
            f"Epoch [{epoch:03d}/{max_epochs:03d}] "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Dur: {epoch_dur:.1f}s | "
            f"Peak VRAM: {peak_vram_mb:.1f} MB"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": float(avg_train_loss),
            "val_loss": float(avg_val_loss),
            "learning_rate": float(current_lr),
            "duration_s": float(epoch_dur),
            "peak_vram_mb": float(peak_vram_mb),
        }
        history.append(epoch_record)

        # Checkpoint if best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                    "val_loss": avg_val_loss,
                    "train_loss": avg_train_loss,
                    "model_cfg": model_cfg.__dict__,
                },
                best_ckpt_path,
            )
            print(f"  -> Saved new best checkpoint to {best_ckpt_path} (epoch {best_epoch}, val_loss: {avg_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                print(f"[Training] Early stopping triggered after {patience_counter} epochs without improvement.")
                break

        # Save history incrementally
        summary_payload = {
            "experiment_name": train_cfg.experiment_name,
            "best_epoch": best_epoch,
            "best_val_loss": float(best_val_loss),
            "checkpoint_path": str(best_ckpt_path),
            "total_duration_s": float(time.time() - train_start_time),
            "peak_vram_mb": float(peak_vram_mb),
            "epochs_completed": epoch,
            "history": history,
        }
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

    total_training_time = time.time() - train_start_time
    print(f"\n[Training] Finished in {total_training_time:.1f}s ({total_training_time / 60.0:.2f} min).")
    print(f"  Best Epoch: {best_epoch} | Best Val Loss: {best_val_loss:.4f}")
    print(f"  Checkpoint Path: {best_ckpt_path}")
    print(f"  Peak GPU Memory: {peak_vram_mb:.1f} MB")
    print(f"  History saved to: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OceanEmbed Phase-1 Model")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for pipeline verification")
    parser.add_argument("--samples", type=int, default=32, help="Number of synthetic samples")
    args = parser.parse_args()

    run_training(epochs=args.epochs, use_synthetic=args.synthetic, num_synthetic_samples=args.samples)
