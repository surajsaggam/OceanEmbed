# OceanEmbed — Final Machine Learning Release Record

**Document Status:** OFFICIAL SCIENTIFIC & ENGINEERING RELEASE RECORD  
**Date:** September 19, 2026  
**Milestone:** Phase-1 Production Freeze & Public Handoff  

---

## 1. Final Primary Model

- **Model Identifier:** **OceanEmbed Phase-1 Primary Model**
- **Model Architecture:** `OceanEmbedNet` (Dual-path multi-scale spatial CNN + Pointwise MLP + 128-D Ocean Embedding + Attention-guided decoder)
- **Trainable Parameters:** **525,040**

---

## 2. Frozen Checkpoint Path & Integrity Hash

- **Official Checkpoint:** `checkpoints/phase1/best.pt`
- **Checkpoint Checksum (SHA256):**  
  `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b`
- **Freeze Status:** **PERMANENTLY FROZEN**. The checkpoint file is immutable and locked.

---

## 3. Architecture Summary

```
                       ┌──────────────────────────────────────────────┐
                       │   14-Channel Surface Observations [B,14,H,W] │
                       └──────────────────────┬───────────────────────┘
                                              │
                       ┌──────────────────────┴──────────────────────┐
                       ▼                                             ▼
            [Path A: Multi-Scale CNN]                     [Path B: Pointwise MLP]
           Local 3x3 (64) + Dilated 3x3 (64)               1x1 Conv Encoder (64)
           Concat -> 1x1 Conv -> 128 ch                    Local column relationships
                       │                                             │
                       └──────────────────────┬──────────────────────┘
                                              ▼
                             [Ocean Embedding Fusion]
                             Concat [128 + 64 = 192 ch]
                             1x1 Conv -> GroupNorm(1, 128) -> GELU
                                              │
                                              ▼
                          Explicit 128-D Ocean Embedding Z [B, 128, H, W]
                                              │
                                              ▼
                              [Attention-Guided Decoder]
                      Skip from Path A modulated by Attention Gate
                      Stage 1: Conv3x3 (64) -> GroupNorm -> GELU
                      Stage 2: Conv3x3 (32) -> GroupNorm -> GELU
                      Output Proj: 1x1 Conv -> 15 Standard Depths
                                              │
                                              ▼
                        Reconstructed 3D Temperature [B, 15, H, W]
```

---

## 4. Input Contract

- **Domain:** North Indian Ocean ($5.0^\circ\text{N}–30.0^\circ\text{N}, 45.0^\circ\text{E}–105.0^\circ\text{E}$), regular $0.25^\circ \times 0.25^\circ$ grid ($101 \times 241$ cells).
- **Physical Variables (Channels 0–6):**
  - `SST`: Sea Surface Temperature (°C)
  - `SSS`: Sea Surface Salinity (PSU)
  - `SSH`: Sea Level Anomaly / SLA (m)
  - `U_curr`: Surface Zonal Current (m/s)
  - `V_curr`: Surface Meridional Current (m/s)
  - `WindU`: 10m Surface Zonal Wind (m/s)
  - `WindV`: 10m Surface Meridional Wind (m/s)
- **Validity Masks (Channels 7–13):**
  - Binary validity masks $\{0.0, 1.0\}$ indicating observed ocean pixels ($1.0$) vs. missing/cloud-masked/land pixels ($0.0$).
- **Imputation:** Invalid/missing pixels are replaced with regional climatological values (`SST=28.0` °C, `SSS=34.0` PSU, `SSH=0.0` m, currents/winds=`0.0` m/s).
- **Normalization:** Scaled using frozen 2015–2017 training statistics (`data/norm_stats/train_stats.json`).

---

## 5. Output Contract

Structured dictionary containing:
- **`temperature`:** $[15, 101, 241]$ array in **°C**, 100% finite.
- **`embedding`:** $[128, 101, 241]$ array (latent Ocean Embedding for water mass analysis).
- **`attention`:** $[1, 101, 241]$ array (spatial attention gate weights in $[0.0, 1.0]$).
- **`validity_mask`:** $[7, 101, 241]$ binary array.
- **`ocean_mask`:** $[101, 241]$ boolean array.
- **`metadata`:** Grid coordinates, resolution ($0.25^\circ$), 15 standard depths, model name, date, and SHA256.

### 15 Standard Depths
$$\mathcal{D} = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] \text{ meters}$$

---

## 6. Training, Validation, and Test Periods

| Dataset Split | Time Period | Calendar Days | Purpose | Primary Metric |
| :--- | :---: | :---: | :--- | :---: |
| **Training Split** | 2015-01-01 to 2017-12-31 | 1,096 | Model parameter optimization | Masked Uniform MSE |
| **Validation Split** | 2018-01-01 to 2018-12-31 | 365 | Model selection & stopping rule | Validation Loss = $0.5705^\circ\text{C}^2$ |
| **Temporal Test Split** | 2019-01-01 to 2019-12-31 | 365 | Out-of-sample temporal benchmark | Column RMSE = **0.7203°C** ($R^2=0.8376$) |

*Boundary Rule: Zero data from 2020 or later was acquired or utilized.*

---

## 7. Independent In-Situ Argo Blind Validation Status

- **Status:** **COMPLETED & LOCKED**.
- **Dataset:** 2,165 quality-controlled in-situ Argo profiling floats (569,215 raw soundings) from INCOIS ERDDAP across all 365 days of 2019.
- **Overall Mean In-Situ RMSE:** **0.9514°C** across all 15 depths (GLORYS reference: $0.8157^\circ$C).
- **Deep Ocean Fidelity (500–1000m):** **0.24°C to 0.27°C RMSE** ($R^2 > 0.92$).
- **Isolation Guard:** Restored to default locked state (`argo_blind_locked: false` in `configs/eval.yaml`).
- **Ablations:** Phase-2 ($\Delta\text{SST} + \Delta\text{SSH}$) and Phase-4A (gradient-aware loss) evaluated as informative controlled ablations; neither displaced Phase-1 as the primary model.

---

## 8. Inference Interface & Public Utilities

```python
from inference import OceanEmbedPredictor, extract_profile, lookup_argo_profile

# Initialize predictor (supports CUDA GPU and CPU)
predictor = OceanEmbedPredictor()

# Execute forward inference
result = predictor.predict(surface_observations, date="2019-06-15")

# Extract 15-depth vertical column at requested coordinate
profile = extract_profile(result, lat=15.0, lon=65.0)

# Optional isolated Argo in-situ observation lookup for visual UI comparison
argo_obs = lookup_argo_profile(date="2019-06-15", lat=15.0, lon=65.0)
```

- **Reference Script:** [`examples/inference_example.py`](file:///d:/OceanEmbed/examples/inference_example.py)
- **Technical Specification:** [`docs/ML_INFERENCE_INTERFACE.md`](file:///d:/OceanEmbed/docs/ML_INFERENCE_INTERFACE.md)
- **Handoff Checklist:** [`docs/ML_HANDOFF_CHECKLIST.md`](file:///d:/OceanEmbed/docs/ML_HANDOFF_CHECKLIST.md)

---

## 9. Automated Regression Test Suite

- **Total Automated Tests:** **144**
- **Test Pass Rate:** **144 / 144 (100% pass rate in 16.37s)**
- **Test Modules:**
  - `tests/test_inference.py` (20 tests)
  - `tests/test_phase1_pipeline.py` (5 tests)
  - `tests/test_phase2_temporal.py` (12 tests)
  - `tests/test_phase3_analysis.py` (5 tests)
  - `tests/test_phase3_gradient_diagnostic.py` (6 tests)
  - `tests/test_phase4a_loss.py` (5 tests)
  - `tests/test_baselines_and_eval.py` (6 tests)
  - `tests/test_argo_guard.py` (4 tests)
  - `tests/test_normalize.py` (10 tests)
  - `tests/test_model_shapes.py` (5 tests)
  - `tests/test_model_gradients.py` (3 tests)
  - `tests/test_model_overfit.py` (1 test)
  - `tests/test_layernorm_nchw.py` (5 tests)
  - `tests/test_mask.py` (5 tests)
  - `tests/test_qc.py` (8 tests)
  - `tests/test_regrid.py` (4 tests)
  - `tests/test_split_leakage.py` (7 tests)
  - `tests/test_downloader.py` (7 tests)
  - `tests/test_datasets.py` (8 tests)
  - `tests/test_dev_split.py` (6 tests)
  - `tests/test_phase0_validation.py` (11 tests)
  - `tests/test_credentials_and_inspection.py` (2 tests)

---

## 10. Known Scientific Limitations

1. **Diagnostic Reconstruction:** OceanEmbed infers contemporary subsurface thermal structure from concurrent surface satellite observations; it does not forecast future ocean states.
2. **Thermocline Uncertainty Peak:** Error is non-uniform vertically, peaking in the high-gradient thermocline layer (75–150 m, $\sim 1.1^\circ\text{C}$ RMSE) and reaching minimal error in the deep ocean ($< 0.28^\circ\text{C}$ RMSE at 500–1000 m).
3. **Statistical Mapping:** The framework maps statistical spatial correlations between surface expressions and subsurface water columns; it does not solve the primitive Navier-Stokes equations.

---

## 11. Final Freeze Declaration

> [!IMPORTANT]
> The OceanEmbed machine learning model, weights, preprocessing rules, and evaluation baselines are **OFFICIALLY FROZEN**.
> No further retraining, architectural modifications, loss function experiments, or data acquisitions will take place.
> The ML layer is ready for full production integration by the frontend and backend engineering teams.
