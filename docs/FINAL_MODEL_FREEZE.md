# OceanEmbed — Final Model Freeze Record

**Document Status:** OFFICIAL SCIENTIFIC RECORD  
**Date:** September 19, 2026  
**Primary Baseline / Final Candidate:** OceanEmbed Phase-1  

---

## 1. Executive Freeze Summary

Following thorough scientific error analysis (Phase-3) and two rigorous controlled ablation experiments (Phase-2 temporal trend ablation and Phase-4A vertical gradient-aware loss ablation), the empirical evidence confirms:

```text
FINAL PRIMARY MODEL:
Phase-1

CHECKPOINT:
checkpoints/phase1/best.pt

CHECKPOINT SHA256:
f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b

TRAINING:
2015–2017 (1,096 days)

VALIDATION:
2018 (365 days)

TEMPORAL TEST:
2019 (365 days)

PHASE-4A:
Informative ablation only; not selected as primary model.
```

---

## 2. Scientific Justification for Selection

| Model Experiment | Architecture & Configuration | 2018 Val RMSE | 2019 Test RMSE | Generalization Shift ($\Delta$) | Empirical Determination |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Phase-1 (Selected Primary)** | 14 ch (7 surface + 7 masks), 128-D embedding, Masked MSE | **0.7178°C** | **0.7203°C** | **+0.0025°C** | **Best overall accuracy, parsimony, and tightest temporal generalization.** |
| **Phase-2 (Ablation)** | 16 ch (Phase-1 + $\Delta$SST + $\Delta$SSH) | 0.7240°C | 0.7259°C | +0.0019°C | Ablation only; trend channels added noise without column benefit. |
| **Phase-4A (Ablation)** | 14 ch, $\mathcal{L}_{\text{MSE}} + 2.0 \cdot \mathcal{L}_{\text{gradient}}$ | 0.7148°C | 0.7216°C | +0.0068°C | Informative ablation only; did not improve 50–150m aggregate, larger generalization shift. Predeclared Case B selected. |

---

## 3. Architecture & Checkpoint Specifications

- **Model Architecture:** `OceanEmbedNet` (525,040 parameters)
  - Path A: `MultiScaleSpatialCNN` (local $3 \times 3$ + dilated $3 \times 3 \to 128$ channels)
  - Path B: `PointwiseMLP` ($1 \times 1$ conv column encoder $\to 64$ channels)
  - Fusion: `OceanEmbeddingFusion` ($192 \to 128$ projected explicit Ocean Embedding)
  - Decoder: `AttentionGuidedDecoder` (skip features gated by spatial attention gate $\to 15$ depth projection)
- **Input Channels (14 channels):**
  - Physical surface variables: SST, SSS, SSH/SLA, U-current, V-current, U-wind, V-wind
  - Binary validity masks: 7 corresponding spatial coverage masks
- **Output Channels (15 standard depths):**
  $0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{ m}$
- **Spatial Resolution:** $0.25^\circ \times 0.25^\circ$ ($101 \times 241$ grid) over the North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$)
- **Checkpoint Epoch:** Epoch 89 (Validation loss = $0.5705^\circ\text{C}^2$)
- **Checkpoint Hash:** `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b`

---

## 4. Integrity and Safeguard Guarantees

1. **Frozen Checkpoint:** The Phase-1 checkpoint file remains 100% read-only and immutable.
2. **Zero Temporal Leakage:** Neither 2018 nor 2019 data was ever accessible or utilized during model training (trained strictly on 2015–2017).
3. **Inference Purity:** All test and validation evaluations are strictly zero-gradient (`torch.no_grad()`), evaluation mode (`model.eval()`), with no optimizer state active.
4. **Argo Independence & Lifecycle:** Argo data were excluded from training, normalization, checkpoint selection, and model tuning. Blind validation was performed only after the model was frozen, and any runtime guard (`argo_blind_locked`) was intentionally restored/disabled after completion of the blind validation.
