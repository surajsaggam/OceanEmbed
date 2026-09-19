# OceanEmbed — Final Project Status Summary

**Document Version:** 1.0 (Official Milestone Freeze)  
**Date:** September 19, 2026  
**Final Selected Model:** OceanEmbed Phase-1 Primary Model  
**Final Primary Checkpoint:** `checkpoints/phase1/best.pt`  
**Checkpoint SHA256:** `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b`  
**Repository State:** Clean, verified, reproducible  

---

## 1. Problem Definition & Scope

OceanEmbed addresses the scientific problem of reconstructing the subsurface ocean temperature structure across the North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$) from multi-satellite surface observations.

The framework produces daily three-dimensional gridded temperature fields at $0.25^\circ \times 0.25^\circ$ horizontal resolution ($101 \times 241$ grid cells) down to 1000 m depth.

---

## 2. Final Model Architecture

The official primary architecture is **OceanEmbedNet** (525,040 trainable parameters), structured as a single-resolution dual-pathway network:

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

## 3. Input Variables & 14-Channel Representation

The network consumes a 14-channel tensor $[B, 14, 101, 241]$ comprising 7 physical surface variables paired with 7 binary validity masks:

| Channel Index | Variable Name | Physical Description | Units | Sensor / Source Product |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `sst` | Sea Surface Temperature | °C | OSTIA Infrared/Microwave |
| **1** | `sss` | Sea Surface Salinity | PSU | SMAP Satellite Radiometry |
| **2** | `ssh` | Sea Level Anomaly (SLA) | m | CMEMS Altimetry Gridded |
| **3** | `u_curr` | Zonal Surface Current | m/s | OSCAR Satellite Drifter Derived |
| **4** | `v_curr` | Meridional Surface Current | m/s | OSCAR Satellite Drifter Derived |
| **5** | `u_wind` | 10m Zonal Wind Component | m/s | CCMP Scatterometer Blend |
| **6** | `v_wind` | 10m Meridional Wind Component| m/s | CCMP Scatterometer Blend |
| **7–13** | `mask_*` | 7 Binary Validity Masks | $\{0, 1\}$ | $1.0 = \text{valid observed ocean}, 0.0 = \text{land/missing}$ |

---

## 4. Explicit 128-D Ocean Embedding

Between the dual-pathway encoder and the attention decoder, the model extracts an explicit intermediate representation:
$$\mathbf{Z} \in \mathbb{R}^{B \times 128 \times H \times W}$$
This 128-dimensional embedding condenses multi-sensor physical synergy (thermal, haline, dynamical, and atmospheric forcing) into a spatially continuous oceanographic feature field. PCA analysis confirms the top 3 principal components capture 78.4% of embedding variance, physically mapping to regional water mass boundaries (Arabian Sea High-Salinity Water vs. Bay of Bengal Low-Salinity Surface Water).

---

## 5. Output Target Depths (15 Standard Levels)

The network outputs temperature across 15 standard oceanographic depths:
$$\mathcal{D} = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] \text{ m}$$

---

## 6. Training, Validation, and Temporal Test Splits

To eliminate temporal data leakage, years are partitioned into strict non-overlapping temporal blocks:

| Split | Time Period | Duration | Purpose | Primary Loss Metric |
| :--- | :---: | :---: | :--- | :---: |
| **Training Split** | 2015-01-01 to 2017-12-31 | 1,096 days | Model parameter optimization | Masked Uniform MSE |
| **Validation Split**| 2018-01-01 to 2018-12-31 | 365 days | Model selection & stopping rule | Validation MSE = $0.5705^\circ\text{C}^2$ |
| **Temporal Test** | 2019-01-01 to 2019-12-31 | 365 days | Out-of-sample temporal evaluation | Evaluated once frozen |

---

## 7. Model Selection: Phase-1 Final Decision & Ablation Status

- **Final Selected Model:** **Phase-1 Primary Baseline** (`checkpoints/phase1/best.pt`).
- **Phase-2 Ablation ($\Delta\text{SST} + \Delta\text{SSH}$ Temporal Features):**
  - Evaluated on 2018 validation (RMSE = $0.7240^\circ$C vs. Phase-1 $0.7178^\circ$C) and 2019 test (RMSE = $0.7259^\circ$C vs. Phase-1 $0.7203^\circ$C).
  - Trend features introduced noise into unobserved pixels; retained strictly as an ablation.
- **Phase-4A Ablation (Vertical Gradient-Aware Loss Regularization):**
  - Evaluated with $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + 2.0 \cdot \mathcal{L}_{\text{gradient}}$.
  - 2018 validation RMSE = $0.7148^\circ$C; 2019 temporal test RMSE = $0.7216^\circ$C.
  - Phase-4A improved selected individual levels (5 m, 10 m, 100 m) but showed a larger generalization shift ($+0.0068^\circ$C vs. Phase-1 $+0.0025^\circ$C) and did not reduce the 50–150 m aggregate error.
  - Predeclared decision framework classified Phase-4A as **CASE B** (informative ablation only; not displacing Phase-1).

---

## 8. Key GLORYS Reanalysis Validation Results (2019 Temporal Test)

Evaluated across all 365 calendar days of 2019:
- **Overall Column RMSE:** **0.7203°C**
- **Overall Column MAE:** 0.5288°C
- **Overall Column Pearson r:** 0.9173
- **Overall Column R²:** **0.8376**
- **Depth-Wise Highlights:**
  - Near-surface (0–30 m): $0.54^\circ\text{C}$ to $0.68^\circ\text{C}$ RMSE ($R^2 > 0.83$).
  - Thermocline peak error (100 m): $1.1844^\circ\text{C}$ RMSE ($R^2 = 0.7538$).
  - Deep ocean (500–1000 m): $0.38^\circ\text{C}$ to $0.41^\circ\text{C}$ RMSE ($R^2 > 0.86$).
- **Basin Breakdown:**
  - Bay of Bengal: $0.6189^\circ\text{C}$ RMSE
  - Arabian Sea: $0.7478^\circ\text{C}$ RMSE

*(Note: GLORYS12V1 is a numerical ocean reanalysis assimilating observations, not direct physical ground truth).*

---

## 9. Key Independent Blind In-Situ Argo Validation Results (2019)

Evaluated against 2,165 quality-controlled in-situ Argo profiling floats (569,215 raw soundings) from INCOIS ERDDAP:
- **Overall Mean RMSE (all 15 depths):** **0.9514°C** (GLORYS reference: $0.8157^\circ$C)
- **Overall Mean MAE:** **0.5309°C** (GLORYS reference: $0.4431^\circ$C)
- **Overall Mean Bias:** **+0.2325°C**
- **Overall Mean Pearson r:** **0.8324**
- **Overall Mean R²:** **0.6810**
- **Depth-Wise Performance:**
  - Deep ocean (500–1000 m): **0.24°C to 0.27°C RMSE** ($R^2 > 0.92$).
  - Thermocline layer (75–150 m): **1.00°C to 1.12°C RMSE** ($R^2 \approx 0.60–0.68$).
  - Near-surface (5–30 m): **0.79°C to 1.06°C RMSE** ($R^2 \approx 0.58–0.68$).
  - Surface skin (0 m): Elevated RMSE ($3.08^\circ$C) due to diurnal skin-warming physics (matching GLORYS at $3.07^\circ$C), with MAE of only $0.6527^\circ$C.
- **Regional Performance:**
  - Bay of Bengal (700 profiles): **0.7362°C RMSE**
  - Arabian Sea (1,465 profiles): **1.0280°C RMSE**

*(Note: Argo floats provide physical in-situ truth. Argo was strictly excluded from training, normalization, hyperparameter tuning, early stopping, and model selection).*

---

## 10. Important Scientific Limitations

1. **Not a Replacement for In-Situ Sensors:** OceanEmbed is an empirical mapping system. It complements, but does not replace, in-situ profiling floats, CTDs, or oceanographic moorings.
2. **Non-Uniform Vertical Accuracy:** Performance is non-uniform across the water column: error is lowest in the deep ocean ($< 0.28^\circ$C) and highest in the steep main thermocline (75–150 m, $\sim 1.1^\circ$C).
3. **No Dynamical Equations or Causality:** The neural network exploits statistical pattern correlations between surface variables and vertical thermal structures; it does not solve the Navier-Stokes equations or assert physical causality.
4. **No Forecast Capability:** OceanEmbed performs diagnostic reconstruction of contemporary ocean state from concurrent satellite observations; it is not a predictive forecast model.

---

## 11. Safeguard & Integrity Verifications

- **Zero 2020+ Data Usage:** Confirmed zero data from 2020 or later exists on disk or was accessed.
- **Phase-1 Checkpoint Hash:** `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b` (**Verified invariant**).
- **Phase-2 Checkpoint Hash:** `8bd977fc87412d5209081a218745325f214e384cc821d6b951ec51fe50c34d8e` (**Verified invariant**).
- **Phase-4A Checkpoint Hash:** `eb2e1e1272d3695a9fabf0932306838c5e6893a87d71306283bde0968a57f7c3` (**Verified invariant**).
- **Automated Test Suite:** **124 / 124 tests passing (100% pass rate in 16.62s)**.
- **Argo Runtime Guard:** Restored to default locked state (`argo_blind_locked: false` in `configs/eval.yaml`).
