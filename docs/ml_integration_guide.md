# OceanEmbed — ML Model Integration Guide

This guide is for the **ML/DL teammate** responsible for model development, training against GLORYS reanalysis, and validation against INCOIS Argo floats.

---

## 1. Scope Boundary

The frontend dashboard and FastAPI backend are fully operational and currently running on the `MockInferenceProvider`.
When your PyTorch checkpoint is ready, you do **not** need to touch the frontend or rebuild the user interface.

All integration happens through `ml_interface/` and `api/services/real_provider.py`.

---

## 2. Checkpoint & Asset Requirements

To plug in your model, you will supply:

1. **Model Weights Checkpoint:**  
   `checkpoints/oceanembed_weights.pt` or ONNX model.
2. **Training Normalization Parameters:**  
   `checkpoints/norm_stats.json` containing $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ computed strictly on the training period (e.g. 2015–2021).
3. **Latent 2D Projection Transformer:**  
   `checkpoints/pca_transformer.joblib` to map the 128-D bottleneck vector $Z$ to `(pca_1, pca_2)` for the dashboard scatter plot.

---

## 3. Tensor Specifications

- **Input Tensor Shape:**
  - 4D Batch: `[B, 14, H, W]`
  - 2D Pointwise: `[B, 14]`
  - Channels (14 total):
    1. SST (OSTIA)
    2. SSS (SMAP/SMOS)
    3. SSH / SLA (DUACS)
    4. $U$ Current (OSCAR)
    5. $V$ Current (OSCAR)
    6. $U$ 10m Wind (CCMP / ASCAT)
    7. $V$ 10m Wind (CCMP / ASCAT)
    8–14. Binary observation validity masks for each variable ($1 = \text{observed}, 0 = \text{imputed}$)

- **Output Tensor Shape:**
  - 4D Batch: `[B, 15, H, W]`
  - 2D Pointwise: `[B, 15]`
  - Standard Depths: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters.
  - Output Units: Degrees Celsius ($^\circ\text{C}$).

- **Latent Embedding:**
  - Shape: `[B, 128, H, W]` or `[B, 128]`.

---

## 4. Enabling the Real Provider

In `api/config.py` (or via environment variable):
```bash
DEFAULT_PROVIDER=real
OCEANEMBED_CHECKPOINT_PATH=checkpoints/oceanembed_weights.pt
```

The application will verify the checkpoint SHA256, execute inference, and automatically flip the UI badge from `DEMO / MOCK DATA` to `LIVE MODEL INFERENCE`.
