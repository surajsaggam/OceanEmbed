# OceanEmbed Phase-4A: Vertical Gradient-Aware Loss Ablation Report

**Document Version:** 1.0 (Final Official Scientific Report)  
**Date:** September 19, 2026  
**Primary Baseline:** Phase-1 Frozen Checkpoint (`checkpoints/phase1/best.pt`)  
**SHA256:** `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b`  
**Phase-2 Ablation Checkpoint:** `checkpoints/phase2/best.pt`  
**SHA256:** `8bd977fc87412d5209081a218745325f214e384cc821d6b951ec51fe50c34d8e`  
**Phase-4A Ablation Checkpoint:** `checkpoints/phase4a/best.pt`  
**SHA256:** `eb2e1e1272d3695a9fabf0932306838c5e6893a87d71306283bde0968a57f7c3`  
**Temporal Test Set 2019:** LOCKED (Strictly untouched and unevaluated)  

---

## 1. Scientific Motivation & Hypothesis

In Phase-3, scientific error analysis and vertical-gradient diagnostics on the primary frozen Phase-1 model established the following key physical observations:
1. **Upper-Interior High-Error Layer:** The 75–150 m ocean layer accounts for the highest temperature reconstruction errors across all seasons and sub-basins (100 m RMSE $\approx 1.1844^\circ$C vs. column mean $0.7203^\circ$C and 500–1000 m mean $0.3983^\circ$C).
2. **Correlation with Target Vertical Gradients:** Target vertical gradient magnitude $|\partial T/\partial z|$ exhibits a statistically significant positive relationship with reconstruction error ($r = +0.352$).
3. **Absence of Gradient Collapse:** Phase-3 definitively refuted the "gradient collapse" hypothesis. Average predicted and target gradient magnitudes across the main thermocline were nearly identical ($0.0520^\circ$C/m vs. $0.0523^\circ$C/m across 50–150 m, an aggregate discrepancy of only $-0.5\%$).
4. **Vertical Displacement Hypothesis:** High errors are instead consistent with vertical profile / thermocline position displacement where the model reconstructs sharp gradients at slightly displaced depths.

**Phase-4A Objective:**  
Conduct a strictly controlled ablation testing whether adding a small vertical-gradient alignment term ($\mathcal{L}_{\text{gradient}}$) to the existing uniform depth-wise MSE objective ($\mathcal{L}_{\text{MSE}}$) improves the difficult 75–150 m region without degrading the rest of the reconstructed water column.

---

## 2. Exact Loss Formulation

The total objective function is defined as:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda_{\text{grad}} \cdot \mathcal{L}_{\text{gradient}}$$

Where:
- $\mathcal{L}_{\text{MSE}}$ is the standard masked uniform depth-wise mean squared error:
  $$\mathcal{L}_{\text{MSE}}(\hat{T}, T) = \frac{\sum_{b, k, i, j} M_{b, k, i, j} \cdot (\hat{T}_{b, k, i, j} - T_{b, k, i, j})^2}{\sum_{b, k, i, j} M_{b, k, i, j} + \epsilon}$$
  with $M_{b, k, i, j} \in \{0, 1\}$ being the binary target/bathymetry validity mask.

- $\mathcal{L}_{\text{gradient}}$ is the masked vertical temperature gradient discrepancy:
  $$\mathcal{L}_{\text{gradient}}(\hat{T}, T) = \frac{\sum_{b, k, i, j} M^{\text{grad}}_{b, k, i, j} \cdot \left| \left(\frac{\partial \hat{T}}{\partial z}\right)_{b, k, i, j} - \left(\frac{\partial T}{\partial z}\right)_{b, k, i, j} \right|}{\sum_{b, k, i, j} M^{\text{grad}}_{b, k, i, j} + \epsilon}$$

An L1 gradient norm was selected because:
1. It is robust to extreme localized thermocline gradient outliers.
2. It has clear physical units ($^\circ$C/m).
3. It prevents gradient loss explosion in sharp thermal front regimes.

When `enabled=False` or $\lambda_{\text{grad}} = 0.0$, the loss reverts strictly and identically to Phase-1 `MaskedMSELoss`.

---

## 3. Nonuniform Vertical Gradient Calculation

Standard oceanographic depths are nonuniform:
$$z = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]\text{ m}$$

The 14 adjacent vertical intervals $\Delta z_k = z_{k+1} - z_k$ are:
$$\Delta z = [5, 5, 10, 10, 20, 25, 25, 25, 25, 50, 100, 200, 200, 300]\text{ m}$$

Vertical temperature gradients are computed explicitly using these physical interval widths:
$$\left(\frac{\partial T}{\partial z}\right)_k = \frac{T_{k+1} - T_k}{z_{k+1} - z_k}$$

Adjacent depth indices are **not** treated as equally spaced.

---

## 4. Masking & Bathymetry Integrity

For any vertical interval $[z_k, z_{k+1}]$, the gradient is physically valid **if and only if** both bounding depth levels are valid:
$$M^{\text{grad}}_{b, k, i, j} = M_{b, k, i, j} \land M_{b, k+1, i, j}$$

- If either $T_k$ or $T_{k+1}$ is masked (e.g. land, shallow continental shelf bathymetry), that interval is excluded from both the numerator and denominator.
- Zero valid gradient pixels (e.g. entirely shallow or land tiles) safely return a gradient loss of `0.0` without dividing by zero or producing `NaN`/`Inf`.
- Invalid target depths are never replaced or imputed.

---

## 5. $\lambda$ Selection and Numerical Scale Analysis

Before training, numerical scaling was evaluated on 10 representative batches of the 2015–2017 training set:
- **Baseline MSE Scale:** $\mathcal{L}_{\text{MSE}} \approx 0.2881^\circ\text{C}^2$
- **Gradient Loss Scale:** $\mathcal{L}_{\text{gradient}} \approx 0.0133^\circ\text{C/m}$
- **Selected Regularization Weight:** $\lambda_{\text{grad}} = 2.0$
- **Resulting Weighted Gradient Term:** $\lambda_{\text{grad}} \cdot \mathcal{L}_{\text{grad}} \approx 0.0266^\circ\text{C}^2$
- **Relative Contribution:** $\approx 9.2\%$ of the MSE scale.

This ensured $\mathcal{L}_{\text{gradient}}$ acts as a gentle regularizing alignment term that informs vertical profile shape without overwhelming or destabilizing the primary MSE temperature reconstruction objective.

---

## 6. Unit Tests & Pre-Training Smoke Test

### Unit Tests (`tests/test_phase4a_loss.py`)
Six dedicated unit tests were implemented and passed:
1. `test_nonuniform_depth_gradient_calculation`: Confirms exact gradient calculation against analytical finite differences.
2. `test_gradient_masking_contract`: Confirms that invalid depths do not contribute to gradient loss.
3. `test_empty_mask_handling`: Confirms numerical stability and zero loss on completely masked inputs.
4. `test_gradient_flow_to_prediction`: Confirms non-zero, finite gradients $\frac{\partial \mathcal{L}_{\text{grad}}}{\partial \hat{T}}$.
5. `test_phase1_reproducibility_when_disabled`: Confirms identical loss to Phase-1 `MaskedMSELoss` when `enabled=False` or $\lambda=0$.
6. `test_gradient_aware_loss_metrics_dict`: Confirms clean dictionary decomposition (`loss_total`, `loss_mse`, `loss_grad`, `grad_weighted`).

**Full test suite status:** 124 passed in 14.14s.

### Smoke & Overfit Gate (`scripts/run_phase4a_smoke_test.py`)
All 7 pre-training gates passed cleanly:
- [Gate 1] 14-channel input shape contract verified `[4, 14, 101, 241]`
- [Gate 2] 15-channel output shape contract verified `[4, 15, 101, 241]`
- [Gate 3] Finiteness & `GradientAwareLoss` computation verified
- [Gate 4] Parameter gradient flow verified across all network pathways
- [Gate 5] 20 overfit steps on fixed batch: Loss dropped from 481.94 to 429.52 (10.9% reduction)
- [Gate 6] Checkpoint save/load round-trip verified (diff $< 10^{-6}$)
- [Gate 7] Phase-1 and Phase-2 checkpoint SHA256 immutability confirmed

---

## 7. Training Execution & Convergence

Training was conducted under the exact Phase-1 training protocol:
- **Input Channels:** 14 channels (7 surface variables + 7 validity masks)
- **Model:** `OceanEmbedNet` (525,040 parameters, 128-D embedding)
- **Train Period:** 2015-01-01 to 2017-12-31 (1,096 days)
- **Val Period:** 2018-01-01 to 2018-12-31 (365 days)
- **Test Period (2019):** **LOCKED (Strictly untouched)**
- **Optimizer:** Adam ($lr = 10^{-4}$, grad clip 1.0, grad accum 2, batch size 4)
- **Scheduler:** Cosine annealing ($\eta_{\min} = 10^{-6}$, max epochs 100)
- **Precision:** bf16 AMP on NVIDIA RTX 5060 GPU
- **Stopping Rule:** Monitored validation MSE with patience = 15

### Training Outcome:
- **Total Duration:** 50.3 minutes (91 epochs executed).
- **Early Stopping:** Triggered at Epoch 91 (no validation MSE improvement for 15 consecutive epochs).
- **Best Validation Epoch:** Epoch 76.
  - Validation MSE Loss: $0.5663^\circ\text{C}^2$ (Phase-1 best was $0.5705^\circ\text{C}^2$ at epoch 89).
  - Validation Gradient Loss: $0.0149^\circ\text{C/m}$.
  - Validation Total Loss: $0.5960$.
- **Checkpoint Saved:** `checkpoints/phase4a/best.pt` (Phase-1 `best.pt` strictly preserved and untouched).

---

## 8. 2018 Validation Results & Phase-1 Comparison

Direct evaluation of `checkpoints/phase4a/best.pt` over all 365 daily fields of 2018 against the frozen Phase-1 baseline yields the following comparison:

### Overall Metrics Summary (2018 Validation Set)

| Metric | Phase-1 Baseline | Phase-4A (+Grad Loss) | Difference (P4A - P1) | Relative Change (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Column RMSE** | **0.7178°C** | **0.7148°C** | **-0.0030°C** | **-0.42%** |
| **Overall Column MAE** | 0.5226°C | 0.5206°C | -0.0020°C | -0.38% |
| **Overall Column Pearson r** | 0.9179 | 0.9191 | +0.0012 | +0.13% |
| **Overall Column R²** | **0.8380** | **0.8391** | **+0.0011** | **+0.13%** |
| **Overall Column Bias** | +0.0884°C | +0.0929°C | +0.0045°C | — |

---

## 9. Full Depth-Wise Comparison (All 15 Standard Depths)

| Depth (m) | Phase-1 RMSE (°C) | Phase-4A RMSE (°C) | Diff (°C) | Diff (%) | Phase-1 R² | Phase-4A R² | Classification |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 0.5823 | 0.5954 | +0.0131 | +2.25% | 0.8904 | 0.8854 | Slight Tradeoff |
| **5** | 0.6209 | 0.6017 | -0.0192 | -3.10% | 0.8739 | 0.8816 | **Improved** |
| **10** | 0.5631 | 0.5296 | -0.0335 | -5.95% | 0.8905 | 0.9031 | **Improved** |
| **20** | 0.6143 | 0.6088 | -0.0056 | -0.91% | 0.8714 | 0.8738 | Neutral / Flat |
| **30** | 0.7295 | 0.7278 | -0.0017 | -0.24% | 0.8323 | 0.8331 | Neutral / Flat |
| **50** | 0.9126 | 0.8845 | -0.0280 | -3.07% | 0.7915 | 0.8041 | **Improved** |
| **75** | 1.0522 | 1.0412 | -0.0110 | -1.05% | 0.7802 | 0.7848 | **Improved** |
| **100** | 1.1399 | 1.1355 | -0.0044 | -0.39% | 0.7511 | 0.7530 | Slight Improvement |
| **125** | 1.0956 | 1.0963 | +0.0006 | +0.06% | 0.7551 | 0.7549 | Neutral / Flat |
| **150** | 0.9612 | 0.9808 | +0.0196 | +2.04% | 0.7896 | 0.7810 | Slight Tradeoff |
| **200** | 0.7236 | 0.7343 | +0.0107 | +1.49% | 0.8430 | 0.8383 | Neutral / Flat |
| **300** | 0.5753 | 0.5813 | +0.0060 | +1.04% | 0.8622 | 0.8594 | Neutral / Flat |
| **500** | 0.4080 | 0.4117 | +0.0037 | +0.91% | 0.8909 | 0.8889 | Invariant |
| **700** | 0.4175 | 0.4202 | +0.0028 | +0.66% | 0.8830 | 0.8815 | Invariant |
| **1000** | 0.3704 | 0.3723 | +0.0019 | +0.52% | 0.8653 | 0.8639 | Invariant |

---

## 10. Focused Subsurface Layer Comparison (50–150 m & 75–150 m)

The core scientific hypothesis tested in Phase-4A was whether vertical gradient regularization could mitigate errors in the thermocline layer:
- **50–150 m Mean RMSE:**
  - Phase-1: **1.0323°C**
  - Phase-4A: **1.0277°C**
  - Improvement: **-0.0046°C (-0.45%)**
  - Notably, 50 m improved by **-3.07%** (0.9126°C $\to$ 0.8845°C), 75 m improved by **-1.05%** (1.0522°C $\to$ 1.0412°C), and 100 m improved by **-0.39%** (1.1399°C $\to$ 1.1355°C).
- **75–150 m Mean RMSE:**
  - Phase-1: **1.0622°C**
  - Phase-4A: **1.0634°C**
  - Difference: **+0.0012°C (+0.11%)**
  - While 75 m and 100 m improved, 150 m showed an error increase of $+0.0196^\circ$C (+2.04%), resulting in a net neutral/flat result for the aggregate 75–150 m window.

---

## 11. Deep-Ocean Comparison (500–1000 m)

A crucial safety constraint of Phase-4A was to avoid degrading deep-water reconstruction:
- **Phase-1 Deep Mean RMSE (500–1000 m):** **0.3986°C**
- **Phase-4A Deep Mean RMSE (500–1000 m):** **0.4014°C**
- **Difference:** **+0.0028°C (+0.70%)**

The deep ocean was preserved without material degradation (well within the predeclared tolerance of $< 0.01^\circ$C). The vertical gradient term did not distort deep profile stability.

---

## 12. Vertical Gradient & Profile Displacement Diagnostics

### Gradient Reconstruction Accuracy:
- **Overall Gradient MAE:** $0.014868^\circ\text{C/m}$
- **Overall Gradient RMSE:** $0.025564^\circ\text{C/m}$
- **Overall Gradient Bias:** $-0.000124^\circ\text{C/m}$ (negligible mean gradient bias)
- The highest gradient RMSE is localized in the 20–30 m ($0.0395^\circ$C/m) and 10–20 m ($0.0341^\circ$C/m) intervals, reflecting strong seasonal mixed-layer base gradients.

### Profile Displacement:
- **Phase-3 Baseline Profile Displacement:** $\approx 17.8\text{ m}$ mean absolute displacement.
- **Phase-4A Profile Displacement:** **17.28 m** mean absolute displacement across 3,449,250 valid ocean profiles.
- Vertical gradient regularization produced a modest reduction in vertical peak displacement ($\approx 0.5\text{ m}$ mean alignment improvement).

---

## 13. Limitations & Tradeoff Analysis

1. **Tradeoff Between Upper and Lower Thermocline:**  
   Gradient alignment successfully improved the upper thermocline (50 m by $-3.1\%$, 75 m by $-1.1\%$, 100 m by $-0.4\%$) and near-surface layers (5 m by $-3.1\%$, 10 m by $-6.0\%$), but incurred a modest penalty at the thermocline base (150 m: $+2.0\%$) and surface skin (0 m: $+2.3\%$).
2. **Magnitude of Subsurface Improvement:**  
   While 100 m RMSE improved from 1.1399°C to 1.1355°C, the absolute reduction is small (0.0044°C). The fundamental constraint on reconstructing the 100 m thermocline from surface observations remains primarily information-theoretic (lack of subsurface density/stratification sensors) rather than purely loss misalignment.
3. **No Material Column Degradation:**  
   Unlike Phase-2 (which degraded overall RMSE from 0.7178°C to 0.7240°C), Phase-4A improved overall column RMSE slightly (0.7178°C $\to$ 0.7148°C) and maintained $R^2$ (0.8380 $\to$ 0.8391).

---

## 14. Predeclared Decision Gate & Official Recommendation

### Decision Gate Evaluation (Predeclared Criteria)

| Gate | Criterion | Result | Status |
| :---: | :--- | :---: | :---: |
| **Gate 1** | **Improvement in 75–150 m?** | 75 m & 100 m improved; 150 m degraded; net +0.0012°C (+0.11%) | **PARTIAL / MIXED** |
| **Gate 2** | **No material degradation in overall RMSE?** | Improved by -0.0030°C (-0.42%) | **PASSED** |
| **Gate 3** | **No material degradation in 500–1000 m?** | Invariant (+0.0028°C / +0.70%) | **PASSED** |
| **Gate 4** | **No major degradation at surface?** | 0 m: +0.0131°C (+2.25%); 5–10 m improved | **PASSED** |
| **Gate 5** | **Gradient / Profile diagnostics improved?** | Displacement reduced from 17.8 m to 17.28 m | **PASSED** |

### Official Scientific Recommendation:

1. **Phase-1 Remains the Primary Official OceanEmbed Model:**  
   Although Phase-4A demonstrates clean stability and modest improvements at 50 m, 75 m, 100 m, and overall RMSE (0.7148°C vs 0.7178°C), the 75–150 m layer as a whole does not exhibit a decisive, breakthrough error reduction (0.11% net difference). Phase-1 remains the official primary model.
2. **Phase-4A Serves as a Valid Regularized Ablation:**  
   Phase-4A proves that vertical gradient alignment can be applied stably without collapsing deep ocean accuracy or destabilizing training.
3. **2019 Rule Followed:**  
   As strictly mandated, the **2019 temporal test set has NOT been evaluated**. It remains locked and untampered until explicit authorization.
4. **Next Step:**  
   STOP and await user review before taking any further action.
