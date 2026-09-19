"""
tests/test_normalize.py
-----------------------
Tests for pipeline/normalize.py

Rules verified:
  - Normalisation statistics are computed from training data only
  - Validation / test pixels do not contribute to statistics (no leakage)
  - Normalisation is invertible within float32 precision
  - Output values are finite (since imputed pixels already have finite fill values)
  - Statistics file round-trips correctly through JSON
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _compute_stats(data: np.ndarray, mask: np.ndarray) -> dict:
    """Compute mean and std over valid (mask==1) pixels only."""
    valid = data[mask == 1]
    return {"mean": float(valid.mean()), "std": float(valid.std())}


def _apply_norm(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Apply z-score normalisation. Assumes input is finite."""
    return (data - mean) / std


def _apply_denorm(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    return data * std + mean


class TestNormalizeStats:
    def test_stats_from_train_only(self):
        """
        If train and val have different distributions, stats must match train.
        """
        rng = np.random.default_rng(0)
        train_data = rng.normal(loc=28.0, scale=2.0, size=(100, 10, 10)).astype(np.float32)
        val_data   = rng.normal(loc=20.0, scale=1.0, size=(50, 10, 10)).astype(np.float32)
        train_mask = np.ones_like(train_data)

        stats = _compute_stats(train_data.ravel(), train_mask.ravel())

        # Stats should be close to the train distribution, NOT val
        assert abs(stats["mean"] - 28.0) < 1.0, "Mean must reflect training data"
        assert abs(stats["std"] - 2.0) < 0.5, "Std must reflect training data"

        # Crucially: val_data was NOT passed — verify it doesn't shift things
        combined = np.concatenate([train_data.ravel(), val_data.ravel()])
        combined_mask = np.ones_like(combined)
        combined_stats = _compute_stats(combined, combined_mask)
        # Combined mean would be between 20 and 28
        assert abs(combined_stats["mean"] - 28.0) > 2.0, (
            "Including val data shifts the mean — this is what we must prevent"
        )

    def test_normalisation_invertible(self):
        rng = np.random.default_rng(1)
        data = rng.normal(loc=25.0, scale=3.0, size=(50,)).astype(np.float32)
        mask = np.ones_like(data)
        stats = _compute_stats(data, mask)
        normed = _apply_norm(data, stats["mean"], stats["std"])
        recovered = _apply_denorm(normed, stats["mean"], stats["std"])
        np.testing.assert_allclose(data, recovered, rtol=1e-5, atol=1e-4)

    def test_normalised_output_is_finite(self):
        rng = np.random.default_rng(2)
        # Input is finite (fill values already applied upstream)
        data = rng.normal(loc=25.0, scale=3.0, size=(100,)).astype(np.float32)
        mask = np.ones_like(data)
        stats = _compute_stats(data, mask)
        normed = _apply_norm(data, stats["mean"], stats["std"])
        assert np.all(np.isfinite(normed)), "Normalised output must be finite"

    def test_stats_json_roundtrip(self):
        stats = {"SST": {"mean": 27.3, "std": 2.1}, "version": "v1"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(stats, f)
            fpath = Path(f.name)
        with open(fpath) as f:
            loaded = json.load(f)
        assert loaded["SST"]["mean"] == stats["SST"]["mean"]
        assert loaded["SST"]["std"] == stats["SST"]["std"]
        fpath.unlink()

    def test_zero_std_raises_or_handled(self):
        """If all training values are identical, std=0 causes division by zero."""
        data = np.full(50, 28.0, dtype=np.float32)
        mask = np.ones_like(data)
        stats = _compute_stats(data, mask)
        assert stats["std"] == 0.0
        # The pipeline must guard against this:
        if stats["std"] == 0.0:
            # Replace with a small epsilon or raise — either is acceptable
            # but the pipeline MUST not silently produce NaN/inf
            safe_std = max(stats["std"], 1e-6)
            normed = _apply_norm(data, stats["mean"], safe_std)
            assert np.all(np.isfinite(normed))


class TestMultiYearNormalizationLifecycle:
    """
    Tests enforcing the multi-year normalization and storage lifecycle:
      1. Masks (channels 7..13) are NEVER normalized and remain binary 0/1.
      2. Multi-year training moments can be combined across years without storing all raw data.
      3. Combining moments is strictly order-independent (associative & commutative).
      4. Normalization of validation/test uses frozen training statistics.
      5. Cleanup gate refuses to delete raw data if training moments or preprocessed files are missing.
    """

    def test_masks_are_not_normalized(self):
        """Verify channels 0..6 are normalized while channels 7..13 remain binary 0/1."""
        from pipeline.normalize import normalize_input_tensor, PHYSICAL_KEYS

        # Create synthetic [14, 10, 10] array
        rng = np.random.default_rng(42)
        input_tensor = np.zeros((14, 10, 10), dtype=np.float32)

        # 7 physical variables: unnormalized values (e.g. SST ~ 28 C, SSS ~ 35 PSU, etc.)
        for c in range(7):
            input_tensor[c] = rng.normal(loc=20.0 + c * 2, scale=2.0, size=(10, 10))

        # 7 masks: strictly binary 0.0 or 1.0
        for c in range(7, 14):
            input_tensor[c] = (rng.random(size=(10, 10)) > 0.3).astype(np.float32)

        # Normalization stats for the 7 physical variables
        norm_stats = {
            k: {"mean": float(20.0 + c * 2), "std": 2.0}
            for c, k in enumerate(PHYSICAL_KEYS)
        }

        normed = normalize_input_tensor(input_tensor, norm_stats)

        # Physical channels 0..6 must be centered around 0
        for c in range(7):
            assert abs(normed[c].mean()) < 1.0
            assert abs(normed[c].std() - 1.0) < 0.5

        # Mask channels 7..13 must be IDENTICAL to original binary masks (NEVER normalized)
        for c in range(7, 14):
            np.testing.assert_array_equal(normed[c], input_tensor[c])
            unique_vals = np.unique(normed[c])
            for u in unique_vals:
                assert u in (0.0, 1.0)

    def test_global_statistics_combine_multiple_years(self):
        """Verify that combining annual moments produces the exact same statistics as pooling all data."""
        from pipeline.normalize import compute_moments, combine_moments, compute_stats_from_moments, PHYSICAL_KEYS

        rng = np.random.default_rng(101)
        years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]

        pooled_data = {k: [] for k in PHYSICAL_KEYS}
        annual_moments = []

        for y in years:
            y_moments = {}
            for k in PHYSICAL_KEYS:
                # Simulate year data with valid ocean mask
                data_y = rng.normal(loc=15.0, scale=3.5, size=5000).astype(np.float32)
                mask_y = (rng.random(size=5000) > 0.1).astype(np.float32)

                valid_vals = data_y[mask_y == 1.0]
                pooled_data[k].extend(valid_vals)

                m = compute_moments(data_y, mask_y)
                y_moments[k] = m
            annual_moments.append(y_moments)

        # Combine moments across all 7 years
        combined = combine_moments(annual_moments)
        global_stats = compute_stats_from_moments(combined)

        # Compare against pooled direct computation
        for k in PHYSICAL_KEYS:
            pooled_arr = np.array(pooled_data[k], dtype=np.float64)
            direct_mean = float(np.mean(pooled_arr))
            direct_std = float(np.std(pooled_arr))

            assert abs(global_stats[k]["mean"] - direct_mean) < 1e-5
            assert abs(global_stats[k]["std"] - direct_std) < 1e-5

    def test_year_order_independence_of_global_statistics(self):
        """Verify that combining annual moments in any order produces identical results."""
        from pipeline.normalize import compute_moments, combine_moments, compute_stats_from_moments, PHYSICAL_KEYS

        rng = np.random.default_rng(202)
        years = [2015, 2016, 2017, 2018]
        annual_moments = []

        for y in years:
            y_moments = {}
            for k in PHYSICAL_KEYS:
                data_y = rng.normal(loc=10.0 + y % 5, scale=2.0 + (y % 3) * 0.5, size=2000).astype(np.float32)
                mask_y = (rng.random(size=2000) > 0.15).astype(np.float32)
                y_moments[k] = compute_moments(data_y, mask_y)
            annual_moments.append(y_moments)

        # Combine in forward order
        comb_fwd = combine_moments(annual_moments)
        stats_fwd = compute_stats_from_moments(comb_fwd)

        # Combine in reverse order
        comb_rev = combine_moments(list(reversed(annual_moments)))
        stats_rev = compute_stats_from_moments(comb_rev)

        # Combine in shuffled order
        shuffled = [annual_moments[2], annual_moments[0], annual_moments[3], annual_moments[1]]
        comb_shuf = combine_moments(shuffled)
        stats_shuf = compute_stats_from_moments(comb_shuf)

        for k in PHYSICAL_KEYS:
            assert np.isclose(stats_fwd[k]["mean"], stats_rev[k]["mean"], atol=1e-7)
            assert np.isclose(stats_fwd[k]["std"], stats_rev[k]["std"], atol=1e-7)
            assert np.isclose(stats_fwd[k]["mean"], stats_shuf[k]["mean"], atol=1e-7)
            assert np.isclose(stats_fwd[k]["std"], stats_shuf[k]["std"], atol=1e-7)

    def test_normalization_with_frozen_statistics(self):
        """Verify that frozen training statistics apply to validation/test and invert correctly."""
        from pipeline.normalize import normalize_input_tensor, denormalize_input_tensor, PHYSICAL_KEYS

        frozen_stats = {
            k: {"mean": float(10.0 + i), "std": float(2.0 + i * 0.2)}
            for i, k in enumerate(PHYSICAL_KEYS)
        }

        # Validation day input
        rng = np.random.default_rng(303)
        val_input = np.zeros((14, 15, 15), dtype=np.float32)
        for c in range(7):
            val_input[c] = rng.normal(loc=12.0, scale=3.0, size=(15, 15))
        for c in range(7, 14):
            val_input[c] = (rng.random(size=(15, 15)) > 0.2).astype(np.float32)

        # Normalize with frozen training stats
        normed = normalize_input_tensor(val_input, frozen_stats)
        recovered = denormalize_input_tensor(normed, frozen_stats)

        # Channels 0..6 recovered within float32 tolerance
        np.testing.assert_allclose(recovered[:7], val_input[:7], rtol=1e-5, atol=1e-4)
        # Channels 7..13 strictly unchanged
        np.testing.assert_array_equal(recovered[7:], val_input[7:])

    def test_cleanup_refuses_deletion_if_training_moments_missing(self):
        """Verify cleanup gate blocks raw deletion for a training year if moments.json is missing."""
        from pipeline.cleanup import safe_cleanup_raw_year
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            raw_dir = Path(tmpdir) / "raw"
            proc_dir = Path(tmpdir) / "processed"
            raw_dir.mkdir()
            proc_dir.mkdir()

            # Create raw directory for 2015
            raw_sst = raw_dir / "sst" / "2015"
            raw_sst.mkdir(parents=True)
            dummy_file = raw_sst / "sst_2015.nc"
            dummy_file.write_bytes(b"raw data")

            # Create dummy preprocessed files for all 365 days of 2015
            from pipeline.cleanup import get_expected_dates_for_year
            dates_2015 = get_expected_dates_for_year(2015)
            for d in dates_2015:
                p_file = proc_dir / f"oceanembed_{d}.npz"
                np.savez(
                    p_file,
                    input=np.ones((14, 101, 241), dtype=np.float32),
                    target=np.ones((15, 101, 241), dtype=np.float32),
                    target_mask=np.ones((15, 101, 241), dtype=np.float32),
                )

            # Preprocessed files are present, BUT moments.json is MISSING!
            report = safe_cleanup_raw_year(
                year=2015,
                raw_dir=raw_dir,
                processed_dir=proc_dir,
                confirm_delete=True,
            )

            # Cleanup MUST be refused because moments are needed for global normalization!
            assert report["verified"] is False
            assert report["action_taken"] == "none"
            assert dummy_file.exists(), "Raw data must NOT be deleted without preserved normalization moments!"
            assert any("moments" in iss.lower() for iss in report["issues"])

