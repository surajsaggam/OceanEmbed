# OceanEmbed — ML Teammate Integration Guide

This directory defines the boundary between the **separately developed ML/DL model** and the **OceanEmbed prototype application**.

---

## 1. Scope & Separation of Concerns

- **ML Teammate Responsibilities:**
  - Data ingestion, harmonisation (xESMF), regridding to $0.25^\circ \times 0.25^\circ$.
  - Dual-path CNN+MLP model architecture and training against GLORYS12V1 reanalysis.
  - Blind evaluation against independent INCOIS Argo floats.
  - Exporting model weights and normalization constants.

- **Prototype Responsibilities:**
  - Serving inferences via FastAPI (`/api/reconstruct`).
  - Interactive React + Leaflet + Plotly dashboard.
  - Scientific visualization of the 15-depth profile, Argo float comparisons, and 2D embedding scatter.
  - Managing mock vs real provider execution and displaying data provenance.

---

## 2. Tensor & Output Contract

### Input Specification
- Single-day, 14-channel normalized surface observation tensor:
  - 7 physical variables: SST, SSS, SSH, $U_{\text{curr}}, V_{\text{curr}}, U_{\text{wind}}, V_{\text{wind}}$
  - 7 binary validity masks ($1 = \text{observed}, 0 = \text{imputed}$)
- Batch shape: `[Batch, 14, Height, Width]` or `[Batch, 14]` for pointwise columns.
- Inputs must be normalized using **only training-year statistics**:
  $$x_{\text{norm}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

### Output Specification
- Subsurface temperature across the **15 standard depths**:
  `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters.
- Batch shape: `[Batch, 15, Height, Width]` or `[Batch, 15]`.
- Output units: Degrees Celsius ($^\circ\text{C}$).

### Latent Ocean Embedding Bottleneck
- The network's feature fusion layer produces a 128-dimensional embedding $Z \in \mathbb{R}^{B \times 128 \times H \times W}$.
- Supply a fitted 2D PCA or UMAP projection matrix or transformer (`pca_transformer.joblib`) so the active sample's embedding can be mapped to 2D coordinates `(pca_1, pca_2)` for dashboard visualization.

---

## 3. How to Plug In Your Model

When your model is trained:
1. Save your weights checkpoint (e.g. `checkpoints/oceanembed_v1.pt`).
2. Save your training normalization statistics (`norm_stats.json`).
3. Save your fitted PCA projector (`pca_2d.joblib` or 2D projection matrix).
4. Update `api/services/real_provider.py` to instantiate your PyTorch model and forward-pass the input tensor.
5. In your `.env` or `api/config.py`, set:
   ```bash
   DEFAULT_PROVIDER=real
   ```

The dashboard and API schemas require **zero changes**.
