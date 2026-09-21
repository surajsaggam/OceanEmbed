"""
scripts/run_phase2_smoke_test.py
--------------------------------
Phase-2 Smoke and Overfit Test Gate.

Verifies:
1. Dynamic 16-channel loading from real preprocessed data
2. Strict finiteness contract
3. Model forward pass with 16 input channels
4. Masked uniform MSE loss computation
5. Backward pass and gradient flow through channels 14 and 15
6. Loss decrease over 20 optimization steps on a fixed mini-batch
7. Checkpoint saving to checkpoints/phase2/smoke_test.pt and roundtrip reloading
8. Confirmation of Phase-1 best.pt checkpoint immutability
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

from models.losses import MaskedMSELoss
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, create_dataloader


def run_smoke_test():
    print("=" * 60)
    print("OceanEmbed Phase-2 Smoke & Overfit Verification")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")

    # 1. Dataset & DataLoader (16 channels)
    print("\n[Step 1] Loading 16-channel training dataset...")
    ds = OceanEmbedDataset(split="train", in_channels=16)
    print(f"  Dataset size: {len(ds)} days")

    # Take first 8 samples for fixed overfit batch
    batch_size = 4
    loader = create_dataloader(ds, batch_size=batch_size, shuffle=False)
    batch_x, batch_y, meta = next(iter(loader))

    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    target_mask = meta["target_mask"].to(device)

    print(f"  Input shape:       {batch_x.shape} (Expected [4, 16, 101, 241])")
    print(f"  Target shape:      {batch_y.shape} (Expected [4, 15, 101, 241])")
    print(f"  Target mask shape: {target_mask.shape}")
    print(f"  Delta mask shape:  {meta['delta_mask'].shape}")

    assert batch_x.shape == (batch_size, 16, 101, 241)
    assert batch_y.shape == (batch_size, 15, 101, 241)
    assert torch.isfinite(batch_x).all(), "Non-finite values found in input tensor!"
    assert torch.isfinite(batch_y).all(), "Non-finite values found in target tensor!"
    print("  -> Finiteness & Shapes Verified.")

    # 2. Model Initialization
    print("\n[Step 2] Initializing OceanEmbedNet (in_channels=16)...")
    model = OceanEmbedNet(in_channels=16).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {param_count:,} (Phase-1 was 525,040)")
    assert param_count == 527472, f"Expected 527,472 params, got {param_count}"

    # 3. Forward Pass
    print("\n[Step 3] Forward pass on fixed mini-batch...")
    batch_x.requires_grad_(True)
    out = model(batch_x)
    pred_temp = out["temperature"]
    embedding = out["embedding"]

    print(f"  Predicted temperature shape: {pred_temp.shape}")
    print(f"  Ocean Embedding Z shape:     {embedding.shape}")
    assert pred_temp.shape == (batch_size, 15, 101, 241)
    assert embedding.shape == (batch_size, 128, 101, 241)
    assert torch.isfinite(pred_temp).all()
    assert torch.isfinite(embedding).all()
    print("  -> Forward pass successful.")

    # 4. Loss & Backward Pass
    print("\n[Step 4] Computing Masked Uniform MSE Loss & Backward pass...")
    criterion = MaskedMSELoss()
    loss = criterion(pred_temp, batch_y, target_mask)
    print(f"  Initial Loss: {loss.item():.6f} deg C^2")
    assert torch.isfinite(loss), "Loss must be finite!"

    loss.backward()
    assert batch_x.grad is not None
    grad_ch14 = batch_x.grad[:, 14].abs().sum().item()
    grad_ch15 = batch_x.grad[:, 15].abs().sum().item()
    print(f"  Gradient norm through Ch 14 (Delta_SST): {grad_ch14:.6e}")
    print(f"  Gradient norm through Ch 15 (Delta_SSH): {grad_ch15:.6e}")
    assert grad_ch14 > 0.0, "Delta_SST channel received 0 gradient!"
    assert grad_ch15 > 0.0, "Delta_SSH channel received 0 gradient!"
    print("  -> Backward pass & gradient flow verified.")

    # 5. Overfit on fixed mini-batch (20 optimization steps)
    print("\n[Step 5] Running 20 optimization steps on fixed mini-batch...")
    batch_x_fixed = batch_x.detach()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    initial_loss = None
    final_loss = None

    for step in range(1, 21):
        optimizer.zero_grad()
        out = model(batch_x_fixed)
        pred = out["temperature"]
        step_loss = criterion(pred, batch_y, target_mask)

        if step == 1:
            initial_loss = step_loss.item()

        step_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 5 == 0 or step == 1 or step == 20:
            print(f"  Step {step:02d}/20: Loss = {step_loss.item():.6f} deg C^2")

        final_loss = step_loss.item()

    loss_reduction_pct = (initial_loss - final_loss) / initial_loss * 100.0
    print(f"\n  Initial Loss:    {initial_loss:.6f} deg C^2")
    print(f"  Final Loss:      {final_loss:.6f} deg C^2")
    print(f"  Loss Reduction:  {loss_reduction_pct:.2f}%")
    assert final_loss < initial_loss, "Loss did not decrease during optimization steps!"
    assert loss_reduction_pct > 5.0, f"Expected noticeable loss decrease, got {loss_reduction_pct:.2f}%"
    print("  -> Overfit test passed.")

    # 6. Checkpoint Save and Load Roundtrip
    print("\n[Step 6] Verifying Phase-2 checkpoint save and reload...")
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "phase2"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    smoke_ckpt_path = ckpt_dir / "smoke_test.pt"

    torch.save({
        "in_channels": 16,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
    }, smoke_ckpt_path)
    print(f"  Saved smoke checkpoint to: {smoke_ckpt_path}")

    # Reload
    loaded_ckpt = torch.load(smoke_ckpt_path, map_location=device, weights_only=False)
    loaded_model = OceanEmbedNet(in_channels=loaded_ckpt["in_channels"]).to(device)
    loaded_model.load_state_dict(loaded_ckpt["model_state"])

    with torch.no_grad():
        out_orig = model(batch_x_fixed)["temperature"]
        out_loaded = loaded_model(batch_x_fixed)["temperature"]
    torch.testing.assert_close(out_orig, out_loaded)
    print("  -> Checkpoint save and reload matched exactly.")

    # 7. Phase-1 Immutability Check
    print("\n[Step 7] Verifying Phase-1 checkpoint SHA256 immutability...")
    phase1_ckpt = PROJECT_ROOT / "checkpoints" / "phase1" / "best.pt"
    expected_sha256 = "f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b"
    actual_sha256 = hashlib.sha256(open(phase1_ckpt, "rb").read()).hexdigest()
    assert actual_sha256 == expected_sha256, f"Phase-1 checkpoint corrupted! Got {actual_sha256}"
    print(f"  Phase-1 best.pt SHA256: {actual_sha256} (VERIFIED INVARIANT)")

    print("\n" + "=" * 60)
    print("Phase-2 Smoke & Overfit Test: ALL 7 CHECKS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_smoke_test()
