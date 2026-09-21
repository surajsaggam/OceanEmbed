# OceanEmbed — Final Independent In-Situ Argo Blind Validation Report

**Document Version:** 1.0 (Official Scientific Record)  
**Date:** September 19, 2026  
**Evaluated Checkpoint:** Frozen Phase-1 Primary Model (`checkpoints/phase1/best.pt`)  
**Checkpoint SHA256:** `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b`  
**In-Situ Data Source:** INCOIS ERDDAP (`Indian_ARGO_Floats`), 2019 Out-of-Sample Year  
**Target Domain:** North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$)  
**Target Depths (15 standard levels):** $0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{ m}$  

---

## 1. Executive Summary & Profile Accounting

The final independent validation evaluates the frozen OceanEmbed Phase-1 primary candidate model directly against independent in-situ profiling float measurements from the international Argo program over the entire 2019 calendar year.

### In-Situ Profile Accounting

| Accounting Category | Count | Percentage | Description / Filtering Rule |
| :--- | :---: | :---: | :--- |
| **Total Downloaded Measurements** | 569,215 | 100.0% | Raw sounding records across all 2019 float profiles |
| **Unique In-Situ Profiles Identified** | 2,433 | 100.0% | Discrete vertical profile soundings in target domain |
| **QC-Passed Profiles** | **2,165** | **88.98%** | Passing strict QC: `TEMP_QC ∈ {'1', '2'}` and $\ge 5$ depths |
| **Rejection: Bad QC / < 3 Soundings** | 261 | 10.73% | Profiles with unphysical readings or insufficient soundings |
| **Rejection: Insufficient Depth Levels**| 7 | 0.29% | Profiles spanning $< 5$ standard OceanEmbed target depths |
| **Rejection: Off-Grid / Coastal Mask** | 0 | 0.00% | Profiles exceeding the $0.5^\circ$ spatial matching radius |
| **Matched In-Situ Profiles** | **2,165** | **100.0%** | Successfully co-located with daily satellite predictions |

---

## 2. Overall Performance Metrics (OceanEmbed vs. In-Situ Argo)

Averaged across all 15 standard depth levels over all 2,165 matched in-situ profiles:

| Validation Metric | OceanEmbed Phase-1 | GLORYS12V1 Reanalysis Reference | Unit / Scale |
| :--- | :---: | :---: | :---: |
| **Overall Mean RMSE** | **0.9514°C** | 0.8157°C | °C |
| **Overall Mean MAE** | **0.5309°C** | 0.4431°C | °C |
| **Overall Mean Bias** | **+0.2325°C** | +0.0712°C | °C |
| **Overall Mean Pearson r** | **0.8324** | 0.8654 | Dimensionless linear correlation |
| **Overall Mean R²** | **0.6810** | 0.7382 | Explained variance ratio |
| **Average Observations / Depth** | **2,082** | 2,082 | Matched soundings per vertical level |

---

## 3. Depth-Wise 15-Depth Breakdown (OceanEmbed vs. In-Situ Argo & GLORYS)

> [!IMPORTANT]
> **Clarification on Ground Truth:**  
> In-situ Argo profiling floats provide the **only physical ground truth** in this evaluation. GLORYS12V1 is a numerical ocean reanalysis product (which assimilates satellite and in-situ data), **not physical ground truth**. It is included strictly as a state-of-the-art operational reference.

| Depth (m) | Matched N | OceanEmbed RMSE (°C) | OceanEmbed MAE (°C) | OceanEmbed Bias (°C) | Pearson r | OceanEmbed R² | GLORYS RMSE (°C) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 1,721 | 3.0817* | 0.6527 | +0.4975 | 0.4534 | 0.1784 | 3.0736* |
| **5** | 1,721 | 1.0314 | 0.4015 | +0.2825 | 0.7829 | 0.5812 | 0.9706 |
| **10** | 2,135 | 1.0673 | 0.4041 | +0.2892 | 0.7763 | 0.5711 | 1.0052 |
| **20** | 2,149 | 0.9853 | 0.4358 | +0.3208 | 0.7978 | 0.5897 | 0.8861 |
| **30** | 2,152 | 0.7946 | 0.4592 | +0.2273 | 0.8406 | 0.6805 | 0.6664 |
| **50** | 2,160 | 0.9239 | 0.6612 | +0.1655 | 0.7832 | 0.6005 | 0.7267 |
| **75** | 2,159 | 1.0558 | 0.7861 | +0.1451 | 0.7813 | 0.6030 | 0.8323 |
| **100** | 2,160 | 1.1128 | 0.8511 | +0.3095 | 0.8234 | 0.6504 | 0.9297 |
| **125** | 2,160 | 1.1214 | 0.8899 | +0.4138 | 0.8407 | 0.6574 | 0.9088 |
| **150** | 2,160 | 1.0042 | 0.8034 | +0.3584 | 0.8572 | 0.6859 | 0.7617 |
| **200** | 2,157 | 0.7869 | 0.6250 | +0.3631 | 0.9023 | 0.7537 | 0.5163 |
| **300** | 2,156 | 0.5198 | 0.3907 | +0.2002 | 0.9391 | 0.8548 | 0.3496 |
| **500** | 2,117 | 0.2744 | 0.2093 | +0.0222 | 0.9709 | 0.9409 | 0.1862 |
| **700** | 2,105 | 0.2672 | 0.2017 | -0.0560 | 0.9707 | 0.9392 | 0.2182 |
| **1000** | 2,019 | 0.2437 | 0.1922 | -0.0517 | 0.9664 | 0.9286 | 0.2034 |

*\*Note on 0 m RMSE:*  
The large 0-m discrepancy is consistent with differences between satellite/reanalysis surface representation and the measurement depth of Argo profiles. Notice that both OceanEmbed ($3.0817^\circ$C) and GLORYS ($3.0736^\circ$C) exhibit identical behavior, while the MAE at 0 m is only $0.6527^\circ$C. Immediately below the surface layer (at 5 m), RMSE plunges to $1.0314^\circ$C, reaching $0.2437^\circ$C at 1000 m.

---

## 4. Regional Performance Breakdown

| Sub-Basin | Matched Profiles | Mean RMSE (°C) | Mean MAE (°C) | Mean Bias (°C) | Mean R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Arabian Sea (AS)** | 1,465 | 1.0280 | 0.5671 | +0.2648 | 0.4894 |
| **Bay of Bengal (BoB)** | 700 | 0.7362 | 0.4552 | +0.1686 | 0.4465 |

- **Bay of Bengal:** Shows significantly lower reconstruction error ($0.7362^\circ\text{C}$ RMSE), benefiting from strong salinity stratification that locks the near-surface thermal structure.
- **Arabian Sea:** Higher error ($1.0280^\circ\text{C}$ RMSE) is concentrated in the energetic western boundary upwelling systems and the high-salinity Arabian Sea water core.

---

## 5. Seasonal Performance Breakdown

| Season | Matched Profiles | Mean RMSE (°C) | Mean MAE (°C) | Mean Bias (°C) | Mean R² |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DJF (Winter Monsoon)** | 483 | 0.8491 | 0.4908 | +0.2309 | 0.7018 |
| **MAM (Spring Transition)** | 474 | 0.8017 | 0.4705 | +0.1907 | 0.6888 |
| **JJAS (Summer Monsoon)** | 812 | 0.9547 | 0.5563 | +0.2719 | 0.6059 |
| **OND (Fall Transition)** | 396 | 0.9611 | 0.5367 | +0.2018 | 0.6687 |

- Reconstruction accuracy peaks during the calmer Spring Transition (MAM: $0.8017^\circ\text{C}$ RMSE).
- Error increases moderately during high-wind forcing regimes (JJAS summer monsoon: $0.9547^\circ\text{C}$; OND fall transition: $0.9611^\circ\text{C}$).

---

## 6. Independent-Validation Scientific Interpretation

To maintain strict scientific integrity and avoid overclaiming:

1. **Not a Direct Measurement:**  
   OceanEmbed is a statistical satellite-to-subsurface estimation framework. It does not replace physical in-situ sensors (Argo, CTDs, moorings) that directly measure thermodynamic state variables.
2. **Non-Uniform Accuracy Across Depth:**  
   Accuracy varies by vertical regime:
   - Deep ocean (500–1000 m): Exceptionally accurate ($0.24^\circ\text{C}$ to $0.27^\circ\text{C}$ RMSE, $R^2 > 0.92$).
   - Upper/intermediate subsurface (75–150 m): The larger errors in the upper/intermediate subsurface indicate limitations in reconstructing subsurface variability from surface observations alone. These errors were not consistently reduced by the tested temporal-delta or gradient-aware ablations.
   - Surface (0 m): The large 0-m discrepancy is consistent with differences between satellite/reanalysis surface representation and the measurement depth of Argo profiles.
3. **No Causal Physical Mechanisms Claimed:**  
   The model maps correlated spatial patterns of surface expression (SST, SSS, SLA, wind, currents) to subsurface thermal structure. It does not integrate the Navier-Stokes equations or assert physical causation.
4. **No Forecasting Capability Claimed:**  
   This framework performs diagnostic reconstruction of the contemporary water column from concurrent surface observations. It is not an ocean forecasting system.
5. **Strict Data Hygiene & Isolation Guard Confirmed:**  
   Argo in-situ data was strictly isolated throughout the development lifecycle:
   - Argo data were excluded from training.
   - Argo data were excluded from normalization.
   - Argo data were excluded from checkpoint selection.
   - Argo data were excluded from model tuning.
   - Blind validation was performed only after the model was frozen.
   - Any runtime guard (`argo_blind_locked`) was intentionally restored/disabled after completion of the blind validation.

---

## 7. Reproducibility Metadata

- **Evaluation Script:** [`scripts/run_final_argo_validation.py`](file:///d:/OceanEmbed/scripts/run_final_argo_validation.py)
- **Machine-Readable Report:** [`evaluation/results/final_argo_validation_report.json`](file:///d:/OceanEmbed/evaluation/results/final_argo_validation_report.json)
- **Model Checkpoint:** `checkpoints/phase1/best.pt` (SHA256: `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b`)
- **Execution Timestamp:** 2026-09-19T12:35:50Z
- **Operating Environment:** Windows, Python 3.11.3, PyTorch 2.11.0+cu128, NVIDIA GeForce RTX 5060 Laptop GPU.
- **Diagnostic Figures Generated:**
  - `reports/figures/argo_spatial_matched_distribution.png`
  - `reports/figures/argo_depth_metrics_comparison.png`
  - `reports/figures/argo_representative_matched_profiles.png`
  - `reports/figures/argo_seasonal_regional_breakdown.png`
