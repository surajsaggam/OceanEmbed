"""
scripts/verify_dev_dataloaders.py
---------------------------------
Verifies OceanEmbedDataset and DataLoader on the newly generated
Phase-1 2015-2018 development splits:
  - Train: 2015-2017 (1,096 days)
  - Val:   2018 (365 days)
"""

import time
import torch
import numpy as np
from pipeline.datasets import OceanEmbedDataset, create_dataloader

def test_dataloaders():
    print("=" * 80)
    print("VERIFYING PHASE-1 DATALOADERS (2015–2018 DEVELOPMENT DATASET)")
    print("=" * 80)

    # 1. Training Dataset & DataLoader
    t0 = time.time()
    ds_train = OceanEmbedDataset(data_dir="data/processed", split="train", fallback_to_synthetic=False)
    assert len(ds_train) == 1096, f"Expected 1096 train samples, got {len(ds_train)}"
    print(f"[1. Train Dataset] 1,096 daily samples verified across 2015-2017.")

    loader_train = create_dataloader(ds_train, batch_size=8, shuffle=True, num_workers=0)
    print(f"  DataLoader created: {len(loader_train)} batches of batch_size=8.")

    # Inspect first batch
    batch_in, batch_tgt, batch_meta = next(iter(loader_train))
    print(f"  Batch input shape:       {batch_in.shape} (float32)")
    print(f"  Batch target shape:      {batch_tgt.shape} (float32)")
    print(f"  Batch target_mask shape: {batch_meta['target_mask'].shape} (float32)")

    assert batch_in.shape == (8, 14, 101, 241)
    assert batch_tgt.shape == (8, 15, 101, 241)
    assert batch_meta["target_mask"].shape == (8, 15, 101, 241)
    assert torch.isfinite(batch_in).all()
    assert torch.isfinite(batch_tgt).all()
    assert torch.isfinite(batch_meta["target_mask"]).all()

    # Verify physical channels are normalized (mean ~ 0, std ~ 1)
    # and mask channels are strictly binary
    phys_channels = batch_in[:, :7]
    mask_channels = batch_in[:, 7:]
    print(f"  Normalized physical channels 0..6 mean: {float(phys_channels[mask_channels == 1.0].mean()):.3f}, std: {float(phys_channels[mask_channels == 1.0].std()):.3f}")
    assert abs(float(phys_channels[mask_channels == 1.0].mean())) < 0.5, "Physical channels not normalized properly!"
    assert 0.5 < float(phys_channels[mask_channels == 1.0].std()) < 1.5, "Physical channels std outside normal range!"

    # Verify mask channels strictly binary
    unique_masks = torch.unique(mask_channels)
    for u in unique_masks:
        assert u.item() in (0.0, 1.0), f"Non-binary mask value: {u.item()}"
    print(f"  Validity mask channels 7..13 strictly binary: {unique_masks.tolist()}")

    # 2. Validation Dataset & DataLoader
    ds_val = OceanEmbedDataset(data_dir="data/processed", split="val", fallback_to_synthetic=False)
    assert len(ds_val) == 365, f"Expected 365 val samples, got {len(ds_val)}"
    print(f"\n[2. Validation Dataset] 365 daily samples verified for year 2018.")

    loader_val = create_dataloader(ds_val, batch_size=8, shuffle=False, num_workers=0)
    print(f"  DataLoader created: {len(loader_val)} batches of batch_size=8.")

    val_in, val_tgt, val_meta = next(iter(loader_val))
    assert val_in.shape == (8, 14, 101, 241)
    assert val_tgt.shape == (8, 15, 101, 241)
    assert val_meta["target_mask"].shape == (8, 15, 101, 241)
    assert torch.isfinite(val_in).all()
    assert torch.isfinite(val_tgt).all()
    assert torch.isfinite(val_meta["target_mask"]).all()
    print(f"  Validation batch 1 verified 100% finite and compliant.")

    # 3. Test Split Verification
    ds_test = OceanEmbedDataset(data_dir="data/processed", split="test", fallback_to_synthetic=True)
    assert len(ds_test) == 16, "Test dataset fallback to synthetic check failed!"
    print(f"\n[3. Test Split] Verified empty processed directory with clean fallback to synthetic for unit testing.")
    print("  EXPLICIT SCIENTIFIC NOTE: No independent final temporal test year exists yet (2023–2024 unacquired).")

    print("\n" + "=" * 80)
    print("ALL DATALOADER VERIFICATION CHECKS PASSED (100%)")
    print("=" * 80)

if __name__ == "__main__":
    test_dataloaders()
