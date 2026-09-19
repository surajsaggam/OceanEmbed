"""
tests/test_inference.py
-----------------------
Unit tests for OceanEmbed Phase-1 ML inference interface and utilities.

Covers:
  1. Checkpoint SHA256 verification
  2. Checkpoint loading
  3. Eval mode guarantee
  4. Parameter requires_grad=False guarantee
  5. 14-channel input tensor contract
  6. 7-variable dictionary input ingestion and normalization
  7. Pre-formed 14-channel input ingestion
  8. Output tensor shapes (temperature [15, 101, 241], embedding [128, 101, 241], attention [1, 101, 241])
  9. Strict finiteness of outputs
  10. Deterministic repeated inference
  11. 15-depth vertical profile extraction
  12. Domain boundary handling (lat/lon out-of-bounds)
  13. Invalid input validation (missing variable, non-finite, wrong shape, non-binary mask)
  14. Argo lookup demonstration isolation
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.predictor import (
    CLIMATOLOGICAL_FILL_VALUES,
    FROZEN_PHASE1_SHA256,
    PHYSICAL_KEYS,
    STANDARD_DEPTHS,
    OceanEmbedPredictor,
    compute_file_sha256,
    extract_profile,
    lookup_argo_profile,
)


@pytest.fixture(scope="module")
def predictor():
    """Initializes the frozen Phase-1 predictor on CPU for testing."""
    return OceanEmbedPredictor(device="cpu", verify_sha256=True)


@pytest.fixture
def synthetic_7var_dict():
    """Creates a synthetic dictionary with 7 physical variables on the 101x241 grid."""
    H, W = 101, 241
    rng = np.random.default_rng(42)
    return {
        "SST": rng.uniform(20.0, 32.0, size=(H, W)).astype(np.float32),
        "SSS": rng.uniform(30.0, 37.0, size=(H, W)).astype(np.float32),
        "SSH": rng.uniform(-0.5, 0.5, size=(H, W)).astype(np.float32),
        "U_curr": rng.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32),
        "V_curr": rng.uniform(-1.0, 1.0, size=(H, W)).astype(np.float32),
        "WindU": rng.uniform(-15.0, 15.0, size=(H, W)).astype(np.float32),
        "WindV": rng.uniform(-15.0, 15.0, size=(H, W)).astype(np.float32),
    }


@pytest.fixture
def synthetic_14ch_input():
    """Creates a valid pre-formed 14-channel input array [14, 101, 241]."""
    H, W = 101, 241
    rng = np.random.default_rng(123)
    phys = rng.normal(loc=0.0, scale=1.0, size=(7, H, W)).astype(np.float32)
    masks = np.ones((7, H, W), dtype=np.float32)
    # Simulate some coastal land
    masks[:, :10, :10] = 0.0
    phys[:, :10, :10] = 0.0
    return np.concatenate([phys, masks], axis=0).astype(np.float32)


class TestInferenceIntegrity:
    """Tests 1-4: Checkpoint hash, loading, eval mode, and gradient isolation."""

    def test_checkpoint_sha256(self):
        ckpt_path = PROJECT_ROOT / "checkpoints" / "phase1" / "best.pt"
        assert ckpt_path.exists(), f"Phase-1 checkpoint not found at {ckpt_path}"
        computed_hash = compute_file_sha256(ckpt_path)
        assert computed_hash.lower() == FROZEN_PHASE1_SHA256.lower(), (
            f"Phase-1 SHA256 mismatch!\nExpected: {FROZEN_PHASE1_SHA256}\nGot: {computed_hash}"
        )

    def test_checkpoint_loading(self, predictor):
        assert predictor.model is not None
        assert predictor.checkpoint_sha256.lower() == FROZEN_PHASE1_SHA256.lower()
        # Verify 525,040 parameters
        num_params = sum(p.numel() for p in predictor.model.parameters())
        assert num_params == 525040, f"Expected 525,040 parameters, got {num_params}"

    def test_eval_mode(self, predictor):
        assert predictor.model.training is False, "Model must be in eval() mode"

    def test_requires_grad_false(self, predictor):
        for name, param in predictor.model.named_parameters():
            assert param.requires_grad is False, f"Parameter '{name}' must have requires_grad=False"


class TestInputHandling:
    """Tests 5-7: 14-channel contract, dict input, pre-formed input."""

    def test_14_channel_input_tensor(self, predictor, synthetic_14ch_input):
        t_in, m_out = predictor.prepare_input_tensor(synthetic_14ch_input)
        assert isinstance(t_in, torch.Tensor)
        assert t_in.shape == (1, 14, 101, 241)
        assert m_out.shape == (7, 101, 241)
        assert torch.isfinite(t_in).all()

    def test_dictionary_input_with_missingness_and_climatology(self, predictor, synthetic_7var_dict):
        # Insert NaNs and out-of-bounds values to test climatological fill
        synthetic_7var_dict["SST"][20, 30] = np.nan
        synthetic_7var_dict["SST"][25, 35] = 99.0  # unphysical
        synthetic_7var_dict["SSS"][40, 50] = np.nan

        t_in, m_out = predictor.prepare_input_tensor(synthetic_7var_dict)
        assert t_in.shape == (1, 14, 101, 241)
        assert torch.isfinite(t_in).all()

        # The corrupted pixels must have validity mask = 0.0
        assert m_out[0, 20, 30] == 0.0
        assert m_out[0, 25, 35] == 0.0
        assert m_out[1, 40, 50] == 0.0

        # And uncorrupted pixels must have validity mask = 1.0
        assert m_out[0, 0, 0] == 1.0

        # In normalized channel 0, the invalid pixel carries (28.0 - mu)/sigma
        sst_mu = predictor.norm_stats["SST"]["mean"]
        sst_sigma = predictor.norm_stats["SST"]["std"]
        expected_norm_val = (CLIMATOLOGICAL_FILL_VALUES["SST"] - sst_mu) / sst_sigma
        actual_norm_val = t_in[0, 0, 20, 30].item()
        assert np.isclose(actual_norm_val, expected_norm_val, atol=1e-5)

    def test_preformed_14_channel_input(self, predictor, synthetic_14ch_input):
        res = predictor.predict(synthetic_14ch_input)
        assert "temperature" in res
        assert "embedding" in res
        assert "attention" in res
        assert "metadata" in res


class TestOutputContractAndFiniteness:
    """Tests 8-10: Output shapes, finiteness, and determinism."""

    def test_output_shapes(self, predictor, synthetic_7var_dict):
        res = predictor.predict(synthetic_7var_dict, date="2019-07-15")
        assert res["temperature"].shape == (15, 101, 241)
        assert res["embedding"].shape == (128, 101, 241)
        assert res["attention"].shape == (1, 101, 241)
        assert res["validity_mask"].shape == (7, 101, 241)
        assert res["ocean_mask"].shape == (101, 241)
        assert res["metadata"]["date"] == "2019-07-15"
        assert res["metadata"]["depths_m"] == STANDARD_DEPTHS

    def test_outputs_finite(self, predictor, synthetic_7var_dict):
        res = predictor.predict(synthetic_7var_dict)
        assert np.all(np.isfinite(res["temperature"])), "Temperature output must be 100% finite"
        assert np.all(np.isfinite(res["embedding"])), "Embedding output must be 100% finite"
        assert np.all(np.isfinite(res["attention"])), "Attention output must be 100% finite"

    def test_deterministic_repeated_inference(self, predictor, synthetic_7var_dict):
        res1 = predictor.predict(synthetic_7var_dict)
        res2 = predictor.predict(synthetic_7var_dict)
        assert np.array_equal(res1["temperature"], res2["temperature"]), (
            "Model predictions must be bitwise deterministic for identical inputs"
        )
        assert np.array_equal(res1["embedding"], res2["embedding"]), (
            "Embeddings must be bitwise deterministic for identical inputs"
        )
        assert np.array_equal(res1["attention"], res2["attention"]), (
            "Attention gate weights must be bitwise deterministic for identical inputs"
        )


class TestProfileExtraction:
    """Tests 11-12: 15-depth vertical profile extraction and boundary checks."""

    def test_profile_extraction_valid(self, predictor, synthetic_7var_dict):
        pred_res = predictor.predict(synthetic_7var_dict)
        # Query central Arabian Sea location
        lat, lon = 15.0, 65.0
        prof = extract_profile(pred_res, lat=lat, lon=lon)

        assert prof["depth_m"] == STANDARD_DEPTHS
        assert len(prof["temperature_C"]) == 15
        assert np.all(np.isfinite(prof["temperature_C"]))
        assert np.isclose(prof["lat_grid"], 15.0, atol=0.25)
        assert np.isclose(prof["lon_grid"], 65.0, atol=0.25)
        assert isinstance(prof["is_valid_ocean"], bool)
        assert len(prof["embedding"]) == 128
        assert isinstance(prof["attention_weight"], float)

    def test_domain_boundary_handling(self, predictor, synthetic_7var_dict):
        pred_res = predictor.predict(synthetic_7var_dict)
        # Lat out of range (< 5.0)
        with pytest.raises(ValueError, match="outside the North Indian Ocean domain"):
            extract_profile(pred_res, lat=3.0, lon=65.0)

        # Lat out of range (> 30.0)
        with pytest.raises(ValueError, match="outside the North Indian Ocean domain"):
            extract_profile(pred_res, lat=32.0, lon=65.0)

        # Lon out of range (< 45.0)
        with pytest.raises(ValueError, match="outside the North Indian Ocean domain"):
            extract_profile(pred_res, lat=15.0, lon=40.0)

        # Lon out of range (> 105.0)
        with pytest.raises(ValueError, match="outside the North Indian Ocean domain"):
            extract_profile(pred_res, lat=15.0, lon=110.0)


class TestInvalidInputHandling:
    """Test 13: Error handling on malformed inputs."""

    def test_missing_variable_in_dict(self, predictor, synthetic_7var_dict):
        del synthetic_7var_dict["SST"]
        with pytest.raises(KeyError, match="Missing required physical surface variable: 'SST'"):
            predictor.predict(synthetic_7var_dict)

    def test_wrong_spatial_shape(self, predictor, synthetic_7var_dict):
        synthetic_7var_dict["SST"] = np.zeros((50, 50), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match expected shape"):
            predictor.predict(synthetic_7var_dict)

    def test_non_finite_preformed_input(self, predictor, synthetic_14ch_input):
        synthetic_14ch_input[0, 10, 10] = np.nan
        with pytest.raises(ValueError, match="contains non-finite values"):
            predictor.predict(synthetic_14ch_input)

    def test_non_binary_mask_in_preformed_input(self, predictor, synthetic_14ch_input):
        synthetic_14ch_input[7, 10, 10] = 0.5  # not 0.0 or 1.0
        with pytest.raises(ValueError, match="contain non-binary value"):
            predictor.predict(synthetic_14ch_input)


class TestArgoLookupIsolation:
    """Test 14: Argo lookup helper demonstration and model isolation."""

    def test_argo_lookup_demonstration(self):
        # Query date that exists in 2019 cache
        res = lookup_argo_profile(date="2019-01-01", lat=12.0, lon=85.0, max_dist_deg=10.0)
        if res is not None:
            assert res["source"] == "in_situ_argo_observation"
            assert "platform_number" in res
            assert len(res["depth_m"]) == 15
            assert len(res["temperature_C"]) == 15
            assert "valid_depths_mask" in res

    def test_argo_lookup_out_of_bounds(self):
        # Far out of bounds coordinates should return None
        res = lookup_argo_profile(date="2019-01-01", lat=-50.0, lon=-100.0, max_dist_deg=0.5)
        assert res is None
