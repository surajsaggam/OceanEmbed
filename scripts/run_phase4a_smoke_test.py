"""
scripts/run_phase4a_smoke_test.py
---------------------------------
Phase-4A Smoke and Overfit Test Gate.

Verifies:
1. 14-channel input shape contract [B, 14, 101, 241]
2. 15-channel output shape contract [B, 15, 101, 241]
3. GradientAwareLoss computation (total loss, MSE loss, gradient loss)
4. Backward pass and finite non-zero gradients across model parameters
5. Monotonic / significant loss decrease over 20 optimization steps on a fixed batch
6. Checkpoint save/load round-trip in checkpoints/phase4a/smoke_test.pt
7. Phase-1 and Phase-2 checkpoint SHA256 immutability
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.losses import GradientAwareLoss
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, create_dataloader
from utils.config import load_config


def run_smoke_test():
    print("=" * 65)
    print("OceanEmbed Phase-4A Smoke & Overfit Verification")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # 1. Dataset & DataLoader (14 channels)
    print("\n[Step 1] Loading 14-channel training dataset...")
    ds = OceanEmbedDataset(split="train", in_channels=14, fallback_to_synthetic=False)
    print(f"  Dataset size: {len(ds)} days")

    batch_size = 4
    loader = create_dataloader(ds, batch_size=batch_size, shuffle=False)
    batch_x, batch_y, meta = next(iter(loader))

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    target_mask = meta["target_mask"].to(device)

    print(f"  Input shape:       {batch_x.shape} (Expected [4, 14, 101, 241])")
    print(f"  Target shape:      {batch_y.shape} (Expected [4, 15, 101, 241])")
    print(f"  Target mask shape: {target_mask.shape}")

    assert batch_x.shape == (batch_size, 14, 101, 241), f"Unexpected input shape {batch_x.shape}"
    assert batch_y.shape == (batch_size, 15, 101, 241), f"Unexpected target shape {batch_y.shape}"
    assert torch.isfinite(batch_x).all(), "Non-finite values found in input tensor!"
    assert torch.isfinite(batch_y).all(), "Non-finite values found in target tensor!"
    print("  -> Finiteness & Shapes Verified.")

    # 2. Model Initialization (14 channels)
    print("\n[Step 2] Initializing OceanEmbedNet (14-channel Phase-1 architecture)...")
    model_cfg = load_config("model")
    model = OceanEmbedNet(model_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total trainable parameters: {total_params:,} (Expected: 525,040)")
    assert total_params == 525040, f"Parameter count mismatch: {total_params}"

    # 3. Forward Pass & Loss Computation
    print("\n[Step 3] Forward pass & GradientAwareLoss evaluation...")
    data_cfg = load_config("data")
    depths = list(data_cfg.depths_m)
    criterion = GradientAwareLoss(depths=depths, lambda_grad=2.0, gradient_loss_type="l1", enabled=True).to(device)

    out = model(batch_x)
    pred = out["temperature"]
    emb = out["embedding"]

    assert pred.shape == (batch_size, 15, 101, 241), f"Unexpected output shape {pred.shape}"
    assert emb.shape == (batch_size, 128, 101, 241), f"Unexpected embedding shape {emb.shape}"
    assert torch.isfinite(pred).all(), "Non-finite values in prediction!"

    loss, metrics = criterion(pred, batch_y, mask=target_mask)
    print(f"  Total Combined Loss: {metrics['loss_total']:.6f}")
    print(f"  - MSE Loss:         {metrics['loss_mse']:.6f} degC^2")
    print(f"  - Gradient Loss:    {metrics['loss_grad']:.6f} degC/m")
    print(f"  - Weighted Grad:    {metrics['grad_weighted']:.6f} (lambda=2.0)")
    pct_contrib = (metrics["grad_weighted"] / metrics["loss_total"]) * 100.0
    print(f"  - Grad Contribution:{pct_contrib:.2f}% of total loss")

    assert torch.isfinite(loss), "Combined loss is not finite!"
    assert metrics["loss_mse"] > 0.0, "MSE loss is non-positive!"
    assert metrics["loss_grad"] > 0.0, "Gradient loss is non-positive!"

    # 4. Backward Pass & Gradient Flow
    print("\n[Step 4] Backward pass & parameter gradient flow...")
    loss.backward()

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has None gradient!"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients!"

    # Check non-zero gradients in CNN and MLP paths
    cnn_grad_norm = model.path_a.local_branch[0].weight.grad.norm().item()
    mlp_grad_norm = model.path_b.conv1.weight.grad.norm().item()
    dec_grad_norm = model.decoder.out_proj.weight.grad.norm().item()
    print(f"  CNN local conv grad norm:     {cnn_grad_norm:.6f}")
    print(f"  Pointwise MLP conv grad norm: {mlp_grad_norm:.6f}")
    print(f"  Decoder output proj grad norm:{dec_grad_norm:.6f}")

    assert cnn_grad_norm > 0.0, "CNN branch has zero gradients!"
    assert mlp_grad_norm > 0.0, "MLP branch has zero gradients!"
    assert dec_grad_norm > 0.0, "Decoder branch has zero gradients!"
    print("  -> Gradient flow confirmed across all network pathways.")

    # 5. Overfit Test on Fixed Batch (20 steps)
    print("\n[Step 5] Running 20 overfit steps on fixed batch...")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    for step in range(20):
        optimizer.zero_grad()
        pred = model(batch_x)["temperature"]
        step_loss, step_metrics = criterion(pred, batch_y, mask=target_mask)
        step_loss.backward()
        optimizer.step()
        losses.append(step_loss.item())

        if step % 5 == 0 or step == 19:
            print(f"  Step {step:02d} | Total Loss: {step_loss.item():.6f} | MSE: {step_metrics['loss_mse']:.6f} | Grad: {step_metrics['loss_grad']:.6f}")

    assert losses[-1] < losses[0], f"Loss did not decrease! Start: {losses[0]:.4f}, End: {losses[-1]:.4f}"
    rel_drop = (losses[0] - losses[-1]) / losses[0] * 100.0
    print(f"  -> Initial Loss: {losses[0]:.4f} -> Final Loss: {losses[-1]:.4f} ({rel_drop:.1f}% reduction).")

    # 6. Checkpoint Save/Load Roundtrip
    print("\n[Step 6] Verifying checkpoint serialization round-trip...")
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "phase4a"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    smoke_ckpt_p = ckpt_dir / "smoke_test.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "step": 20,
            "loss": losses[-1],
            "lambda_grad": 2.0,
        },
        smoke_ckpt_p,
    )
    assert smoke_ckpt_p.exists(), "Smoke test checkpoint file not found!"

    # Reload into fresh model instance
    reloaded_model = OceanEmbedNet(model_cfg).to(device)
    loaded_data = torch.load(smoke_ckpt_p, map_location=device, weights_only=False)
    reloaded_model.load_state_dict(loaded_data["model_state_dict"])
    reloaded_model.eval()

    with torch.no_grad():
        orig_out = model(batch_x)["temperature"]
        reloaded_out = reloaded_model(batch_x)["temperature"]
        diff = (orig_out - reloaded_out).abs().max().item()
        print(f"  Max output difference after reload: {diff:.8f}")
        assert diff < 1e-6, "Reloaded model output differs from original!"
    print("  -> Checkpoint roundtrip verified.")

    # 7. Checkpoint Invariance Confirmation
    print("\n[Step 7] Confirming Phase-1 & Phase-2 checkpoint immutability...")
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

    print("\n" + "=" * 65)
    print("ALL PHASE-4A PRE-TRAINING GATES PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_smoke_test()
