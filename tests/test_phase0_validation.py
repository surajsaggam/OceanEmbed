"""
tests/test_phase0_validation.py
--------------------------------
Automated Phase-0 validation tests on January 2020 dry-run data.

Validates:
  1. Physical units and variable semantics (SST Kelvin->Celsius, SSS PSU, SLA vs ADT,
     Currents m/s, Winds m/s, GLORYS thetao degC).
  2. SLA selection over ADT (model strictly uses SLA).
  3. Common 0.25° NIO grid mapping with exact coordinate arrays and monotonic ordering.
  4. Phase-1 tensor contract: [B, 14, H, W] input with torch.isfinite().all(), [B, 15, H, W] target.
  5. Missing-data pipeline on real data (invalid -> mask 0 -> climatology fill, no silent zero-fill).
  6. GLORYS depth interpolation (1000m bracketing between 902.339m and 1062.440m, zero extrapolation,
     bathymetric shallow-water masking).
  7. CCMP 6-hourly wind temporal harmonization (daily mean over 4 synoptic intervals).
  8. Normalization leakage prevention (training statistics only).
  9. Argo blind-evaluation guard verification.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr

from evaluation.argo_eval import check_argo_guard, run_blind_argo_evaluation
from pipeline.harmonize import DataHarmonizer
from pipeline.normalize import compute_variable_stats, normalize_array
from pipeline.qc import apply_variable_qc
from pipeline.regrid import get_default_target_coords
from utils.config import load_config


RAW_DIR = Path("data/raw")
SST_FILE = RAW_DIR / "sst" / "sst_jan2020.nc"
SSS_FILE = RAW_DIR / "sss" / "sss_jan2020.nc"
SSH_FILE = RAW_DIR / "ssh" / "ssh_jan2020.nc"
CURRENTS_FILES = sorted(list((RAW_DIR / "currents").glob("*.nc")))
WINDS_FILES = sorted(list((RAW_DIR / "winds").glob("*.nc")))
GLORYS_FILE = RAW_DIR / "glorys" / "glorys_jan2020.nc"

DATA_AVAILABLE = (
    SST_FILE.exists()
    and SSS_FILE.exists()
    and SSH_FILE.exists()
    and len(CURRENTS_FILES) > 0
    and len(WINDS_FILES) > 0
    and GLORYS_FILE.exists()
)


@pytest.mark.skipif(not DATA_AVAILABLE, reason="Raw January 2020 data files not found in data/raw")
class TestPhase0Validation:
    """Comprehensive automated validation of the Phase-0 scientific contract."""

    @classmethod
    def setup_class(cls):
        cls.harmonizer = DataHarmonizer()
        cls.lat_tgt, cls.lon_tgt = get_default_target_coords()
        cls.H = len(cls.lat_tgt)
        cls.W = len(cls.lon_tgt)

    def test_sst_units_and_celsius_conversion(self):
        """Verify OSTIA SST is Kelvin, converted to Celsius, and within plausible NIO range."""
        with xr.open_dataset(SST_FILE) as ds:
            assert "kelvin" in ds.analysed_sst.attrs.get("units", "").lower()
            raw_val = ds.analysed_sst.isel(time=0).values
            valid_raw = raw_val[np.isfinite(raw_val)]
            assert 270.0 < np.min(valid_raw) < 320.0

        sst_c, mask = self.harmonizer.harmonize_sst(SST_FILE, time_idx=0)
        valid_c = sst_c[mask == 1.0]
        assert len(valid_c) > 0
        assert 15.0 < np.min(valid_c) < 30.0
        assert 25.0 < np.max(valid_c) < 35.0
        assert 25.0 < np.mean(valid_c) < 30.0

    def test_sss_units_and_range(self):
        """Verify SMAP/SMOS SSS is in PSU and within plausible NIO bounds [25, 42]."""
        sss_clean, mask = self.harmonizer.harmonize_sss(SSS_FILE, time_idx=0)
        valid_sss = sss_clean[mask == 1.0]
        assert len(valid_sss) > 0
        assert np.min(valid_sss) >= 25.0
        assert np.max(valid_sss) <= 42.0
        assert 32.0 < np.mean(valid_sss) < 37.0

    def test_ssh_uses_sla_not_adt(self):
        """Verify DUACS SSH explicitly selects 'sla' (zero-centered anomaly) rather than 'adt'."""
        with xr.open_dataset(SSH_FILE) as ds:
            assert "sla" in ds.data_vars, "sla variable missing from DUACS NetCDF"
            assert "adt" in ds.data_vars, "adt variable missing from DUACS NetCDF"
            sla_val = ds.sla.isel(time=0).values
            adt_val = ds.adt.isel(time=0).values
            valid_sla = sla_val[np.isfinite(sla_val)]
            valid_adt = adt_val[np.isfinite(adt_val)]

            # SLA is zero-centered anomaly
            assert abs(np.mean(valid_sla)) < 0.15, f"SLA mean {np.mean(valid_sla)} should be near zero"
            assert -1.0 < np.min(valid_sla) and np.max(valid_sla) < 1.0
            # ADT has mean ~ 0.85m in NIO (MDT + geoid)
            assert np.mean(valid_adt) > 0.5, f"ADT mean {np.mean(valid_adt)} expected > 0.5m"

        sla_clean, mask = self.harmonizer.harmonize_ssh(SSH_FILE, time_idx=0)
        valid_sla_clean = sla_clean[mask == 1.0]
        assert abs(np.mean(valid_sla_clean)) < 0.15
        assert np.min(valid_sla_clean) > -1.0 and np.max(valid_sla_clean) < 1.0

    def test_currents_units_and_transposition(self):
        """Verify OSCAR currents are transposed from (time, lon, lat) to (lat, lon) and units are m/s."""
        u_clean, mask_u, v_clean, mask_v = self.harmonizer.harmonize_currents(CURRENTS_FILES[0])
        assert u_clean.shape == (self.H, self.W)
        assert v_clean.shape == (self.H, self.W)
        assert np.all(np.isfinite(u_clean))
        assert np.all(np.isfinite(v_clean))

        valid_u = u_clean[mask_u == 1.0]
        valid_v = v_clean[mask_v == 1.0]
        assert -2.0 < np.min(valid_u) and np.max(valid_u) < 2.0
        assert -2.0 < np.min(valid_v) and np.max(valid_v) < 2.0

    def test_winds_harmonization_daily_mean(self):
        """Verify CCMP winds average the 4 synoptic slices (00, 06, 12, 18 UTC) into daily mean."""
        with xr.open_dataset(WINDS_FILES[0]) as ds:
            assert len(ds.time) == 4, f"Expected 4 synoptic intervals, got {len(ds.time)}"

        u_clean, mask_u, v_clean, mask_v = self.harmonizer.harmonize_winds(WINDS_FILES[0], method="daily_mean")
        assert u_clean.shape == (self.H, self.W)
        assert v_clean.shape == (self.H, self.W)
        assert np.all(np.isfinite(u_clean))
        assert np.all(np.isfinite(v_clean))

    def test_common_target_grid_coordinates(self):
        """Verify all datasets map onto one exact common 0.25° NIO grid with monotonic ordering."""
        assert self.H == 101
        assert self.W == 241
        assert np.isclose(self.lat_tgt[0], 5.0) and np.isclose(self.lat_tgt[-1], 30.0)
        assert np.isclose(self.lon_tgt[0], 45.0) and np.isclose(self.lon_tgt[-1], 105.0)
        assert np.all(np.diff(self.lat_tgt) > 0), "Target latitude must be strictly increasing"
        assert np.all(np.diff(self.lon_tgt) > 0), "Target longitude must be strictly increasing"

    def test_missing_data_pipeline_no_silent_zero(self):
        """Verify missing/land data receives climatological fill and mask=0; NO silent zero-fill."""
        sst_clean, mask_sst = self.harmonizer.harmonize_sst(SST_FILE, time_idx=0)
        land_sst = sst_clean[mask_sst == 0.0]
        assert len(land_sst) > 0
        # Check SST fill is around 25°C-28°C, definitely not 0.0°C
        assert np.all(land_sst > 15.0), f"SST missing values were silently zero-filled! Found: {np.min(land_sst)}"

        sss_clean, mask_sss = self.harmonizer.harmonize_sss(SSS_FILE, time_idx=0)
        land_sss = sss_clean[mask_sss == 0.0]
        assert len(land_sss) > 0
        assert np.all(land_sss > 25.0), f"SSS missing values were silently zero-filled! Found: {np.min(land_sss)}"

    def test_glorys_interpolation_and_shallow_water_masking(self):
        """Verify GLORYS 1000m bracketing (902.339m to 1062.440m), zero extrapolation, and shallow water masking."""
        with xr.open_dataset(GLORYS_FILE) as ds:
            native_z = ds.depth.values
            assert native_z[34] <= 1000.0 <= native_z[35], (
                f"1000m must be bracketed by level 34 ({native_z[34]}m) and 35 ({native_z[35]}m)"
            )

        target_15d, target_mask = self.harmonizer.harmonize_glorys_target(GLORYS_FILE, time_idx=0)
        assert target_15d.shape == (15, self.H, self.W)
        assert target_mask.shape == (15, self.H, self.W)

        # Shallow water masking: valid counts must decrease with depth
        valid_surface = int(np.sum(target_mask[0] == 1.0))
        valid_1000m = int(np.sum(target_mask[14] == 1.0))
        assert valid_surface > valid_1000m, "Valid counts must decrease at 1000m due to continental shelves"
        diff = valid_surface - valid_1000m
        assert diff > 2000, f"Expected > 2000 shallow-water columns masked, got {diff}"

    def test_phase1_tensor_contract_finiteness(self):
        """Verify input [14, 101, 241] and target [15, 101, 241] satisfy torch.isfinite().all()."""
        in_tensor, target_tensor, meta = self.harmonizer.build_daily_sample(
            sst_file=SST_FILE,
            sss_file=SSS_FILE,
            ssh_file=SSH_FILE,
            currents_file=CURRENTS_FILES[0],
            winds_file=WINDS_FILES[0],
            glorys_file=GLORYS_FILE,
            time_idx=0,
        )

        assert in_tensor.shape == (14, self.H, self.W)
        assert target_tensor.shape == (15, self.H, self.W)
        assert torch.isfinite(in_tensor).all(), "Input tensor contains non-finite values!"
        assert torch.isfinite(target_tensor).all(), "Target tensor contains non-finite values!"
        assert meta["target_mask"].shape == (15, self.H, self.W)

    def test_normalization_leakage_prevention(self):
        """Verify normalization statistics are computed exclusively on valid observations (mask==1)."""
        sst_clean, mask_sst = self.harmonizer.harmonize_sst(SST_FILE, time_idx=0)
        stats = compute_variable_stats(sst_clean, mask_sst)
        norm_sst = normalize_array(sst_clean, stats["mean"], stats["std"])
        valid_norm = norm_sst[mask_sst == 1.0]

        assert np.isclose(np.mean(valid_norm), 0.0, atol=1e-5)
        assert np.isclose(np.std(valid_norm), 1.0, atol=1e-5)

    def test_argo_guard_remains_intact(self):
        """Verify that Argo blind evaluation guard remains locked and raises AssertionError."""
        eval_cfg = load_config("eval")
        assert eval_cfg.argo_blind_locked is False

        with pytest.raises(AssertionError, match="LOCKED"):
            run_blind_argo_evaluation(
                model=torch.nn.Linear(1, 1),
                argo_profiles=[],
                eval_cfg=eval_cfg,
            )
