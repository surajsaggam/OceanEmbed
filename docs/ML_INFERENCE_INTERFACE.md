# OceanEmbed — ML Inference Interface Specification

**Document Version:** 1.0 (Production Model Handoff)  
**Target Audience:** Frontend Team, Backend/API Engineers, Downstream Integration Engineers  
**Scope:** Machine Learning Model Inference & Data Contract Only (No UI/Server code)  
**Frozen Model Checkpoint:** `checkpoints/phase1/best.pt`  
**Checkpoint SHA256:** `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b`  

---

## A. Purpose & Overview

`OceanEmbed` is a deep learning system that reconstructs the three-dimensional subsurface ocean temperature structure across the North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$) from multi-satellite surface observations.

This module provides the clean, reproducible, and deterministic Python inference interface (`OceanEmbedPredictor`) that wraps the frozen Phase-1 primary model (525,040 parameters). The interface accepts surface observations, executes pure zero-gradient evaluation, and yields:
1. Reconstructed 3D temperature fields at 15 standard depths down to 1000 m.
2. Latent 128-dimensional Ocean Embedding fields for water mass characterization.
3. Learned 2D spatial attention gate maps.
4. Pointwise 15-depth vertical temperature profile extraction at arbitrary coordinates.

---

## B. Required Surface Observation Inputs

To generate a subsurface reconstruction, the model requires 7 physical surface observation fields on the standard $0.25^\circ \times 0.25^\circ$ regular grid ($101 \times 241$ cells):

| Variable Key | Description | Physical Units | Valid Range | Climatological Fill | Sensor / Product Source |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`SST`** | Sea Surface Temperature | °C | $[-2.0, 36.0]$ | `28.0` °C | OSTIA Infrared/Microwave Blend |
| **`SSS`** | Sea Surface Salinity | PSU | $[0.0, 45.0]$ | `34.0` PSU | SMAP Satellite Radiometry |
| **`SSH`** | Sea Level Anomaly (SLA) | m | $[-2.0, 2.0]$ | `0.0` m | CMEMS Multi-Satellite Altimeter |
| **`U_curr`** | Surface Zonal Current | m/s | $[-3.0, 3.0]$ | `0.0` m/s | OSCAR Satellite Drifter Derived |
| **`V_curr`** | Surface Meridional Current | m/s | $[-3.0, 3.0]$ | `0.0` m/s | OSCAR Satellite Drifter Derived |
| **`WindU`** | 10m Surface Zonal Wind | m/s | $[-30.0, 30.0]$ | `0.0` m/s | CCMP Scatterometer Blend |
| **`WindV`** | 10m Surface Meridional Wind | m/s | $[-30.0, 30.0]$ | `0.0` m/s | CCMP Scatterometer Blend |

The predictor supports common aliases (e.g., `'sst'`, `'sla'`, `'u_wind'`, `'v_wind'`) in input dictionaries.

---

## C. Exact 14-Channel Model Input Ordering

The neural network ingests a 14-channel tensor of shape $[B, 14, 101, 241]$:

```
Channel Index │ Variable Name   │ Content Type
──────────────┼─────────────────┼─────────────────────────────────────────────
      0       │ SST             │ Z-score normalized Sea Surface Temperature
      1       │ SSS             │ Z-score normalized Sea Surface Salinity
      2       │ SSH             │ Z-score normalized Sea Level Anomaly
      3       │ U_curr          │ Z-score normalized Zonal Current
      4       │ V_curr          │ Z-score normalized Meridional Current
      5       │ WindU           │ Z-score normalized Zonal Wind
      6       │ WindV           │ Z-score normalized Meridional Wind
──────────────┼─────────────────┼─────────────────────────────────────────────
      7       │ mask_SST        │ Binary validity mask (1.0 = valid, 0.0 = invalid/land)
      8       │ mask_SSS        │ Binary validity mask
      9       │ mask_SSH        │ Binary validity mask
     10       │ mask_U_curr     │ Binary validity mask
     11       │ mask_V_curr     │ Binary validity mask
     12       │ mask_WindU      │ Binary validity mask
     13       │ mask_WindV      │ Binary validity mask
```

---

## D. Preprocessing Behavior

Inference strictly adheres to the approved Phase-1 training preprocessing semantics:

1. **Validity Masking**:
   - Valid ocean observations: `mask = 1.0`.
   - Missing data, cloud-occluded pixels, non-finite values (NaN/Inf), or land: `mask = 0.0`.
2. **Missing-Data Imputation**:
   - Invalid pixels are **never** silently left as unnormalized zeros or NaNs.
   - Invalid pixels are imputed with regional climatological values (SST = 28.0 °C, SSS = 34.0 PSU, etc.) prior to normalization.
3. **Z-Score Normalization**:
   - Normalized physical channels: $x_{\text{norm}} = \frac{x - \mu}{\sigma}$.
   - Validity masks (channels 7–13) remain strictly binary $\{0.0, 1.0\}$ and are never normalized.

---

## E. Frozen Normalization Statistics

All inputs are scaled using the official 2015–2017 training split statistics stored in `data/norm_stats/train_stats.json`:

| Variable | Mean ($\mu$) | Standard Deviation ($\sigma$) |
| :--- | :---: | :---: |
| **`SST`** | `28.254789` | `1.847708` |
| **`SSS`** | `34.711783` | `2.045608` |
| **`SSH`** | `0.085016` | `0.097401` |
| **`U_curr`** | `0.011031` | `0.226067` |
| **`V_curr`** | `0.010126` | `0.209320` |
| **`WindU`** | `0.701621` | `3.519643` |
| **`WindV`** | `0.197075` | `3.296156` |

---

## F. Model Loading & Instantiation

```python
from inference import OceanEmbedPredictor

# Automatically loads checkpoints/phase1/best.pt, verifies SHA256,
# and puts the model in eval mode on GPU (if available) or CPU.
predictor = OceanEmbedPredictor(device="cuda") # or device="cpu"
```

The predictor guarantees:
- `model.eval()` is set.
- All model parameter gradients are disabled (`requires_grad = False`).
- Optimizer state is omitted to minimize RAM footprint (~6.35 MB weight footprint).

---

## G. Temperature Output Contract

- **Tensor Key:** `result["temperature"]`
- **Shape:** `[15, 101, 241]` (or `[B, 15, 101, 241]` for batched inputs)
- **Data Type:** `numpy.ndarray` (float32)
- **Physical Unit:** Degrees Celsius (°C)
- **Finiteness:** 100% guaranteed finite (no NaNs, no Infs).

---

## H. 15 Standard Depths

The 15 vertical target depths correspond to indices $0 \dots 14$ along axis 0 of the temperature array:

$$\mathcal{D} = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] \text{ meters}$$

```python
# Temperature at 100m depth across the entire domain:
temp_100m = result["temperature"][7, :, :]  # index 7 corresponds to 100m
```

---

## I. 128-Dimensional Ocean Embedding

- **Tensor Key:** `result["embedding"]`
- **Shape:** `[128, 101, 241]` (or `[B, 128, 101, 241]`)
- **Data Type:** `numpy.ndarray` (float32)
- **Role:** Explicit latent representation fusing thermal, haline, dynamical, and wind forcing. Can be directly passed to dimensionality reduction algorithms (PCA, UMAP, t-SNE) for water mass clustering or visualization.

---

## J. Attention Gate Output

- **Tensor Key:** `result["attention"]`
- **Shape:** `[1, 101, 241]` (or `[B, 1, 101, 241]`)
- **Data Type:** `numpy.ndarray` (float32, values in $[0.0, 1.0]$)
- **Role:** Spatial attention gate weights showing where the decoder selectively focuses multi-scale surface gradient features to sharpen subsurface thermal boundaries.

---

## K. Point/Profile Extraction Utility

To extract the 15-depth vertical temperature profile at any geographic point:

```python
from inference import extract_profile

# Extract water column at 15.25°N, 65.50°E
profile = extract_profile(result, lat=15.25, lon=65.50)
```

### Returned Profile Object Contract:

```json
{
  "depth_m": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
  "temperature_C": [28.45, 28.42, 28.38, 28.10, 27.50, 25.20, 22.10, 19.40, 17.10, 15.30, 13.05, 10.80, 8.20, 6.45, 5.10],
  "lat_requested": 15.25,
  "lon_requested": 65.50,
  "lat_grid": 15.25,
  "lon_grid": 65.50,
  "grid_idx": [41, 82],
  "is_valid_ocean": true,
  "embedding": [ ... 128 float values ... ],
  "attention_weight": 0.742
}
```

---

## L. Error Handling & Edge Cases

| Scenario | Behavior / Exception | Resolution |
| :--- | :--- | :--- |
| **Coordinate Out-of-Bounds** | Raises `ValueError` with bounds specification | Query within lat $[5.0, 30.0]^\circ\text{N}$, lon $[45.0, 105.0]^\circ\text{E}$ |
| **Missing Surface Variable** | Raises `KeyError` specifying missing key | Provide all 7 physical variables |
| **Mismatched Grid Shapes** | Raises `ValueError` specifying mismatched shape | Ensure all variables have matching 2D shapes |
| **Non-finite Preformed Tensor** | Raises `ValueError` | Ensure inputs are finite; use dict input for auto-imputation |
| **Corrupted Checkpoint File** | Raises `ValueError` on SHA256 mismatch | Restore verified checkpoint `checkpoints/phase1/best.pt` |

---

## M. CPU vs. GPU Deployment

- **CUDA GPU:** Under GPU execution with batch size 1, inference takes $\approx 5–8\text{ ms}$ per day.
- **CPU:** Under standard x86 CPU execution, inference takes $\approx 45–65\text{ ms}$ per day.
- **VRAM Footprint:** Model parameters require $\approx 2.1\text{ MB}$; inference forward pass uses $< 80\text{ MB}$ of VRAM.

---

## N. Complete Python Usage Example

```python
import numpy as np
from inference import OceanEmbedPredictor, extract_profile

# 1. Initialize predictor
predictor = OceanEmbedPredictor(device="cuda")

# 2. Prepare surface observations (dictionary of 7 variables, shape [101, 241])
# In production, these come from your daily NetCDF/Zarr data ingestion pipeline.
surface_obs = {
    "SST": np.full((101, 241), 28.5, dtype=np.float32),
    "SSS": np.full((101, 241), 35.0, dtype=np.float32),
    "SSH": np.full((101, 241), 0.05, dtype=np.float32),
    "U_curr": np.full((101, 241), 0.1, dtype=np.float32),
    "V_curr": np.full((101, 241), -0.05, dtype=np.float32),
    "WindU": np.full((101, 241), 2.5, dtype=np.float32),
    "WindV": np.full((101, 241), 1.0, dtype=np.float32),
}

# 3. Execute inference
result = predictor.predict(surface_obs, date="2019-06-15")

# 4. Access outputs
temp_3d = result["temperature"]       # [15, 101, 241] (°C)
emb_3d = result["embedding"]          # [128, 101, 241]
attn_2d = result["attention"]         # [1, 101, 241]

# 5. Extract single column profile at target location
point_profile = extract_profile(result, lat=12.0, lon=70.0)
print(f"Location: {point_profile['lat_grid']}°N, {point_profile['lon_grid']}°E")
print(f"Surface Temp: {point_profile['temperature_C'][0]:.2f}°C")
print(f"100m Temp:    {point_profile['temperature_C'][7]:.2f}°C")
print(f"1000m Temp:   {point_profile['temperature_C'][14]:.2f}°C")
```

---

## O. Frozen Checkpoint Verification

To verify that your local checkpoint file is identical to the scientifically verified Phase-1 model:

```powershell
Get-FileHash -Algorithm SHA256 checkpoints/phase1/best.pt
```

**Expected Hexadecimal Checksum:**  
`f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b`
