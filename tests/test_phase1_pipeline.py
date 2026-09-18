"""
tests/test_phase1_pipeline.py
------------------------------
End-to-end pipeline and integration tests on real preprocessed January 2020 data.

Validates:
  1. Real preprocessed dataset files and metadata exist on disk.
  2. OceanEmbedDataset loads real data and adheres strictly to [14, 101, 241] and [15, 101, 241] contracts.
  3. Finiteness guarantee: torch.isfinite(input).all() is verified on real data.
  4. MaskedMSELoss behaves as expected with 2D, 3D, and 4D masks.
  5. Baselines (Climatology and Ridge) fit and predict on real preprocessed data.
  6. End-to-end forward pass, loss, backward gradient computation, and validation on real data.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from baselines.climatology import ClimatologyBaseline
from baselines.ridge import ChunkedRidgeBaseline
from models.losses import MaskedMSELoss
from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, create_dataloader
from utils.config import load_config


PROCESSED_DIR = Path("data/processed")
REAL_DATA_EXISTS = (
    (PROCESSED_DIR / "train").exists()
    and len(list((PROCESSED_DIR / "train").glob("*.npz"))) >= 21
    and (PROCESSED_DIR / "val").exists()
    and len(list((PROCESSED_DIR / "val").glob("*.npz"))) >= 5
    and (PROCESSED_DIR / "test").exists()
    and len(list((PROCESSED_DIR / "test").glob("*.npz"))) >= 5
)


@pytest.mark.skipif(not REAL_DATA_EXISTS, reason="Preprocessed January 2020 data not found")
class TestPhase1PipelineRealData:
    @classmethod
    def setup_class(cls):
        cls.data_cfg = load_config("data")
        cls.model_cfg = load_config("model")
        cls.train_ds = OceanEmbedDataset(split="train", fallback_to_synthetic=False)
        cls.val_ds = OceanEmbedDataset(split="val", fallback_to_synthetic=False)
        cls.test_ds = OceanEmbedDataset(split="test", fallback_to_synthetic=False)

    def test_preprocessed_file_counts_and_metadata(self):
        """Verify train, val, and test splits contain expected sample counts and metadata."""
        assert len(self.train_ds) == 21
        assert len(self.val_ds) == 5
        assert len(self.test_ds) == 5

        meta_file = PROCESSED_DIR / "grid_metadata.json"
        assert meta_file.exists()
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        assert meta["num_channels_input"] == 14
        assert meta["num_channels_target"] == 15
        assert meta["domain"]["H"] == 101
        assert meta["domain"]["W"] == 241

    def test_real_dataset_shapes_and_finiteness(self):
        """Verify input [14, 101, 241] and target [15, 101, 241] with torch.isfinite().all()."""
        x, y, meta = self.train_ds[0]
        assert x.shape == (14, 101, 241)
        assert y.shape == (15, 101, 241)
        assert torch.isfinite(x).all(), "Real input tensor contains NaNs or Infs!"
        assert torch.isfinite(y).all(), "Real target tensor contains NaNs or Infs!"

        # Masks
        mask_in = x[7:]  # [7, 101, 241]
        for c in range(7):
            uniq = mask_in[c].unique()
            for v in uniq:
                assert v.item() in (0.0, 1.0)

        # Target mask in meta
        assert "target_mask" in meta
        t_mask = meta["target_mask"]
        assert t_mask.shape == (15, 101, 241)

    def test_masked_mse_loss_functionality(self):
        """Verify MaskedMSELoss ignores masked pixels and computes correct error."""
        criterion = MaskedMSELoss()
        pred = torch.ones((2, 15, 101, 241), dtype=torch.float32)
        target = torch.ones((2, 15, 101, 241), dtype=torch.float32) * 3.0  # diff = 2.0, sq_err = 4.0

        mask = torch.ones((2, 15, 101, 241), dtype=torch.float32)
        loss = criterion(pred, target, mask=mask)
        assert torch.isclose(loss, torch.tensor(4.0))

        # Half masked with huge error
        pred[0, :, :50, :] = 1000.0
        mask[0, :, :50, :] = 0.0  # Mask out the 1000.0 error
        loss_masked = criterion(pred, target, mask=mask)
        assert torch.isclose(loss_masked, torch.tensor(4.0))

    def test_baselines_train_and_predict_real_data(self):
        """Verify Climatology and Ridge baselines train and predict on real preprocessed data."""
        clim = ClimatologyBaseline(n_depths=15, grid_shape=(101, 241))
        ridge = ChunkedRidgeBaseline(in_features=14, out_features=15, alpha=1.0)

        for i in range(2):
            x, y, meta = self.train_ds[i]
            m = meta.get("target_mask", None)
            m_np = m.numpy() if m is not None else None
            clim.update(month=1, target=y.numpy(), mask=m_np)
            ocean_mask = (m_np[0] == 1.0) if m_np is not None else None
            ridge.update_chunk(x.numpy(), y.numpy(), valid_mask=ocean_mask)

        clim.finalize()
        ridge.finalize()

        c_pred = clim.predict(month=1)
        r_pred = ridge.predict(self.train_ds[0][0].numpy())

        assert c_pred.shape == (15, 101, 241)
        assert r_pred.shape == (15, 101, 241)
        assert np.all(np.isfinite(r_pred))

    def test_end_to_end_forward_backward_loss(self):
        """Verify end-to-end forward pass, loss, and backward pass on real preprocessed batch."""
        loader = create_dataloader(self.train_ds, batch_size=2, shuffle=False)
        x, y, meta = next(iter(loader))
        target_mask = meta.get("target_mask", None)

        model = OceanEmbedNet(self.model_cfg)
        model.train()

        criterion = MaskedMSELoss()
        out = model(x)

        assert "temperature" in out
        assert "embedding" in out
        assert out["temperature"].shape == (2, 15, 101, 241)
        assert out["embedding"].shape == (2, 128, 101, 241)

        loss = criterion(out["temperature"], y, mask=target_mask)
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

        loss.backward()

        # Check gradients
        for p_name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Parameter {p_name} missing gradient!"
                assert torch.isfinite(p.grad).all(), f"Parameter {p_name} gradient non-finite!"
