# OceanEmbed — ML Integration Handoff Checklist

**Document Version:** 1.0 (Official Model Handoff)  
**Date:** September 19, 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Frozen Model & Checkpoint Verification

| Component | Verified Specification |
| :--- | :--- |
| **Model Architecture** | `OceanEmbedNet` (Phase-1 Primary Baseline, 525,040 parameters) |
| **Checkpoint Path** | `checkpoints/phase1/best.pt` |
| **Checkpoint SHA256** | `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b` |
| **Integrity Status** | Immutable / Read-only / Verified Invariant |

---

## 2. Public Import Interface

```python
from inference import OceanEmbedPredictor, extract_profile, lookup_argo_profile
```

*(Alternatively accessible via `from pipeline.inference import ...`)*

---

## 3. Quickstart Example

```python
from inference import OceanEmbedPredictor, extract_profile

# 1. Initialize predictor (auto-detects CUDA / CPU)
predictor = OceanEmbedPredictor()

# 2. Run inference on 7 surface observation fields (each shape [101, 241])
result = predictor.predict(surface_observations, date="2019-06-15")

# 3. Extract 15-depth vertical temperature profile at target coordinate
profile = extract_profile(result, lat=15.0, lon=65.0)
```

---

## 4. Input Contract

- **Spatial Domain:** North Indian Ocean ($5.0^\circ\text{N}–30.0^\circ\text{N}, 45.0^\circ\text{E}–105.0^\circ\text{E}$), $0.25^\circ \times 0.25^\circ$ regular grid ($101 \times 241$ cells).
- **Format:** `Dict[str, np.ndarray]` with 7 physical variables:
  - `SST` (°C)
  - `SSS` (PSU)
  - `SSH` / `sla` (m)
  - `U_curr` (m/s)
  - `V_curr` (m/s)
  - `WindU` (m/s)
  - `WindV` (m/s)
- **Pre-formed Tensor:** Also accepts `[14, 101, 241]` or `[B, 14, 101, 241]` preprocessed arrays directly.

---

## 5. Output Contract

Structured dictionary containing:
- `temperature`: $[15, 101, 241]$ numpy array in °C (100% finite).
- `embedding`: $[128, 101, 241]$ numpy array (latent Ocean Embedding for water mass clustering).
- `attention`: $[1, 101, 241]$ numpy array (spatial attention gate weights).
- `validity_mask`: $[7, 101, 241]$ numpy array (binary observation validity).
- `ocean_mask`: $[101, 241]$ boolean array (true ocean coverage).
- `metadata`: Grid bounds, resolution ($0.25^\circ$), 15 standard depths, model name, and SHA256 checksum.

---

## 6. Profile Extraction

- **Function:** `extract_profile(result, lat, lon)`
- **Target Depths (15 levels):**
  $$0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{ m}$$
- **Output:** Returns `{"depth_m": [...], "temperature_C": [...], "lat_grid": float, "lon_grid": float, "is_valid_ocean": bool, "grid_idx": (row, col)}`.
- **Bounds:** Strictly rejects out-of-domain queries ($<5^\circ\text{N}$, $>30^\circ\text{N}$, $<45^\circ\text{E}$, $>105^\circ\text{E}$) with clear `ValueError`.

---

## 7. Preprocessing & Missing-Data Behavior

- **Automatic Imputation:** Invalid, cloud-masked, NaN, Inf, or out-of-range physical inputs are imputed with regional climatological values:
  `SST=28.0` °C, `SSS=34.0` PSU, `SSH=0.0` m, currents/winds=`0.0` m/s.
- **Mask Propagation:** Validity mask is assigned $0.0$ for invalid/imputed pixels, $1.0$ for valid observations.
- **Normalization:** Physical channels are z-score scaled using frozen 2015–2017 training statistics (`data/norm_stats/train_stats.json`).

---

## 8. Deployment & Hardware Footprint

- **GPU (CUDA):** $\approx 5–8\text{ ms}$ per daily inference run; $< 80\text{ MB}$ dynamic VRAM footprint.
- **CPU:** $\approx 45–65\text{ ms}$ per daily inference run; $\approx 50\text{ MB}$ RAM footprint.
- **Checkpoint File Size:** 6.35 MB (`checkpoints/phase1/best.pt`).

---

## 9. Automated Test Status

- **Unit Inference Tests:** **20 / 20 passed** (`tests/test_inference.py`).
- **Full Repository Test Suite:** **144 / 144 passed** (`pytest -q`).
- **Deterministic Smoke Test:** Verified bitwise identical repeated inference on real satellite data.

---

## 10. Known Scientific Limitations

1. **Diagnostic Reconstruction, Not a Forecast:** OceanEmbed diagnoses contemporary subsurface structure from concurrent satellite observations; it is not a predictive forecast model.
2. **Thermocline Peak Uncertainty:** Accuracy is highest in the deep ocean ($< 0.28^\circ\text{C}$ RMSE at 500–1000 m) and lowest in the steep thermocline (75–150 m, $\sim 1.1^\circ\text{C}$ RMSE).
3. **Not In-Situ Truth:** The model provides statistical inference from surface satellite signatures. It complements, but does not replace, in-situ profiling floats (Argo) or moorings.

---

## 11. Division of Team Responsibilities

### ML / Inference Team Responsibility (COMPLETED)
- [x] Train and scientifically validate Phase-1 model.
- [x] Lock and freeze primary checkpoint (`checkpoints/phase1/best.pt`).
- [x] Provide clean Python inference interface (`OceanEmbedPredictor`).
- [x] Provide coordinate-to-profile extraction utility (`extract_profile`).
- [x] Verify determinism, finiteness, and error handling with automated unit tests.
- [x] Deliver complete technical contract (`docs/ML_INFERENCE_INTERFACE.md`).

### Frontend / Application Team Responsibility (DOWNSTREAM)
- [ ] Implement user dashboard / React application / UI widgets.
- [ ] Connect Leaflet / Mapbox / OpenLayers for 2D spatial map rendering.
- [ ] Connect Plotly / Chart.js for 15-depth vertical profile line charts.
- [ ] Connect FastAPI / web server endpoints to invoke `OceanEmbedPredictor.predict()` and `extract_profile()`.
- [ ] Manage user authentication, file uploads, and browser sessions.
