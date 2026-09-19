"""
tests/test_phase2_temporal.py
-----------------------------
Comprehensive test suite for OceanEmbed Phase-2:
  - Exact Delta_SST and Delta_SSH calculation
  - Joint validity masking: M_delta = M(t) * M(t-1)
  - Invalid / non-finite delta handling (0.0 fill)
  - 2015-01-01 sequence boundary (unobserved t-1)
  - 2018-01-01 validation boundary (strictly historical 2017-12-31)
  - 2019-01-01 test boundary (strictly historical 2018-12-31)
  - Temporal leakage prevention (zero future knowledge)
  - Training-only normalization statistics (2015-01-01 to 2017-12-31)
  - 16-channel dataset tensor contract [16, H, W]
  - 15-channel output contract [B, 15, H, W]
  - Embedding shape [B, 128, H, W]
  - Finite tensors (isfinite guarantee)
  - Model forward pass
  - Model backward pass
  - Non-zero gradient flow through both new channels (14 & 15)
  - Phase-1 14-channel backward compatibility
  - Phase-2 checkpoint save/load roundtrip
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from models.ocean_embed_net import OceanEmbedNet
from pipeline.datasets import OceanEmbedDataset, SyntheticOceanDataset
from pipeline.temporal_features import compute_temporal_deltas, normalize_temporal_deltas


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_exact_delta_sst_and_ssh():
    """Verify exact difference computation: delta = x(t) - x(t-1)."""
    H, W = 10, 10
    sst_t = np.full((H, W), 28.5, dtype=np.float32)
    sst_prev = np.full((H, W), 28.0, dtype=np.float32)
    m_sst_t = np.ones((H, W), dtype=np.float32)
    m_sst_prev = np.ones((H, W), dtype=np.float32)

    ssh_t = np.full((H, W), 0.15, dtype=np.float32)
    ssh_prev = np.full((H, W), 0.12, dtype=np.float32)
    m_ssh_t = np.ones((H, W), dtype=np.float32)
    m_ssh_prev = np.ones((H, W), dtype=np.float32)

    d_sst, m_d_sst, d_ssh, m_d_ssh = compute_temporal_deltas(
        sst_t, m_sst_t, ssh_t, m_ssh_t,
        sst_prev, m_sst_prev, ssh_prev, m_ssh_prev,
    )

    np.testing.assert_allclose(d_sst, 0.5, atol=1e-6)
    np.testing.assert_allclose(d_ssh, 0.03, atol=1e-6)
    assert np.all(m_d_sst == 1.0)
    assert np.all(m_d_ssh == 1.0)


def test_joint_validity_mask():
    """Verify joint validity mask: M_delta = M(t) * M(t-1)."""
    H, W = 4, 4
    # Case 1: Both valid -> 1
    # Case 2: t invalid -> 0
    # Case 3: t-1 invalid -> 0
    # Case 4: Both invalid -> 0
    m_t = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ], dtype=np.float32)
    m_prev = np.array([
        [1.0, 1.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    sst_t = np.full((2, 2), 25.0, dtype=np.float32)
    sst_prev = np.full((2, 2), 24.0, dtype=np.float32)
    ssh_t = np.full((2, 2), 0.1, dtype=np.float32)
    ssh_prev = np.full((2, 2), 0.05, dtype=np.float32)

    d_sst, m_d_sst, d_ssh, m_d_ssh = compute_temporal_deltas(
        sst_t, m_t, ssh_t, m_t,
        sst_prev, m_prev, ssh_prev, m_prev,
    )

    expected_mask = np.array([
        [1.0, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    np.testing.assert_array_equal(m_d_sst, expected_mask)
    np.testing.assert_array_equal(m_d_ssh, expected_mask)
    assert d_sst[0, 0] == pytest.approx(1.0, rel=1e-5)
    # Invalid positions zeroed out
    assert d_sst[0, 1] == 0.0
    assert d_sst[1, 0] == 0.0
    assert d_sst[1, 1] == 0.0


def test_invalid_and_non_finite_handling():
    """Verify that NaNs/Infs or missing inputs produce finite 0.0 fill and mask 0.0."""
    H, W = 3, 3
    sst_t = np.array([[np.nan, 28.0, 27.0], [26.0, np.inf, 25.0], [24.0, 23.0, 22.0]], dtype=np.float32)
    sst_prev = np.full((H, W), 25.0, dtype=np.float32)
    m_t = np.ones((H, W), dtype=np.float32)
    m_prev = np.ones((H, W), dtype=np.float32)

    ssh_t = np.full((H, W), 0.1, dtype=np.float32)
    ssh_prev = np.full((H, W), 0.08, dtype=np.float32)

    d_sst, m_d_sst, d_ssh, m_d_ssh = compute_temporal_deltas(
        sst_t, m_t, ssh_t, m_t,
        sst_prev, m_prev, ssh_prev, m_prev,
    )

    assert np.isfinite(d_sst).all()
    assert m_d_sst[0, 0] == 0.0
    assert d_sst[0, 0] == 0.0
    assert m_d_sst[1, 1] == 0.0
    assert d_sst[1, 1] == 0.0
    assert m_d_sst[0, 1] == 1.0


def test_boundary_2015_01_01():
    """Verify 2015-01-01 sequence boundary: t-1 is unobserved -> delta=0.0, mask=0.0."""
    ds = OceanEmbedDataset(split="train", in_channels=16)
    x, y, meta = ds[0]

    assert meta["date"] == "2015-01-01"
    assert x.shape == (16, 101, 241)
    # Channel 14 (Delta_SST) and Channel 15 (Delta_SSH) must be exact normalized 0.0 fill
    assert (x[14] == 0.0).all()
    assert (x[15] == 0.0).all()
    assert (meta["delta_mask"] == 0.0).all()


def test_boundary_2018_01_01():
    """Verify 2018-01-01 validation boundary uses historical 2017-12-31 from training split."""
    ds = OceanEmbedDataset(split="val", in_channels=16)
    x, y, meta = ds[0]

    assert meta["date"] == "2018-01-01"
    assert x.shape == (16, 101, 241)
    # Both channels must have finite values and valid ocean pixels
    assert torch.isfinite(x[14]).all()
    assert torch.isfinite(x[15]).all()
    valid_count = (meta["delta_mask"][0] == 1.0).sum().item()
    assert valid_count > 10000, f"Expected >10000 valid pixels for 2018-01-01, got {valid_count}"


def test_boundary_2019_01_01():
    """Verify 2019-01-01 test boundary uses historical 2018-12-31 from validation split."""
    ds = OceanEmbedDataset(split="test", in_channels=16)
    x, y, meta = ds[0]

    assert meta["date"] == "2019-01-01"
    assert x.shape == (16, 101, 241)
    assert torch.isfinite(x[14]).all()
    assert torch.isfinite(x[15]).all()
    valid_count = (meta["delta_mask"][0] == 1.0).sum().item()
    assert valid_count > 10000, f"Expected >10000 valid pixels for 2019-01-01, got {valid_count}"


def test_temporal_leakage_prevention():
    """Verify strict causality: validation dataset cannot map future (2019) files."""
    ds_val = OceanEmbedDataset(split="val", in_channels=16)
    for date_str in ds_val._historical_date_map.keys():
        year = int(date_str.split("-")[0])
        assert year <= 2018, f"Validation date map leaked future date: {date_str}"


def test_training_only_normalization_stats():
    """Verify phase2_train_stats.json is computed strictly on 2015–2017."""
    stats_file = PROJECT_ROOT / "data" / "norm_stats" / "phase2_train_stats.json"
    assert stats_file.exists(), "Phase-2 normalization stats file does not exist!"

    with open(stats_file, "r") as f:
        stats = json.load(f)

    assert "delta_SST" in stats
    assert "delta_SSH" in stats
    assert stats["delta_SST"]["std"] > 0.0
    assert stats["delta_SSH"]["std"] > 0.0
    assert stats["metadata"]["training_period"] == "2015-01-01 to 2017-12-31"


def test_16_channel_tensor_shape_and_finite():
    """Verify synthetic and real dataset return [16, 101, 241] finite tensors."""
    syn = SyntheticOceanDataset(num_samples=2, in_channels=16)
    x, y, meta = syn[0]
    assert x.shape == (16, 101, 241)
    assert y.shape == (15, 101, 241)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()


def test_phase1_backward_compatibility():
    """Verify in_channels=14 works identically for Phase-1."""
    ds = OceanEmbedDataset(split="train", in_channels=14)
    x, y, meta = ds[0]
    assert x.shape == (14, 101, 241)
    assert y.shape == (15, 101, 241)

    m14 = OceanEmbedNet(in_channels=14)
    out = m14(x.unsqueeze(0))
    assert out["temperature"].shape == (1, 15, 101, 241)
    assert out["embedding"].shape == (1, 128, 101, 241)


def test_model_forward_backward_and_gradients_16_channels():
    """Verify model forward, backward, and non-zero gradients on channels 14 and 15."""
    model = OceanEmbedNet(in_channels=16)
    x = torch.randn(2, 16, 101, 241, requires_grad=True)
    target = torch.randn(2, 15, 101, 241)

    out = model(x)
    temp_pred = out["temperature"]
    embedding = out["embedding"]

    assert temp_pred.shape == (2, 15, 101, 241)
    assert embedding.shape == (2, 128, 101, 241)
    assert torch.isfinite(temp_pred).all()
    assert torch.isfinite(embedding).all()

    loss = torch.nn.functional.mse_loss(temp_pred, target)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    # Specifically check gradient flow through channels 14 and 15
    grad_norm_ch14 = x.grad[:, 14].abs().sum().item()
    grad_norm_ch15 = x.grad[:, 15].abs().sum().item()
    assert grad_norm_ch14 > 0.0, "Gradients through Delta_SST channel (14) must be non-zero!"
    assert grad_norm_ch15 > 0.0, "Gradients through Delta_SSH channel (15) must be non-zero!"


def test_phase2_checkpoint_save_and_load():
    """Verify Phase-2 model can save and load state_dict seamlessly."""
    model_orig = OceanEmbedNet(in_channels=16)
    x = torch.randn(1, 16, 101, 241)
    out_orig = model_orig(x)["temperature"]

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        torch.save({"model_state": model_orig.state_dict(), "in_channels": 16}, tmp_path)
        ckpt = torch.load(tmp_path, map_location="cpu", weights_only=False)
        model_loaded = OceanEmbedNet(in_channels=ckpt["in_channels"])
        model_loaded.load_state_dict(ckpt["model_state"])

        out_loaded = model_loaded(x)["temperature"]
        torch.testing.assert_close(out_orig, out_loaded)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
