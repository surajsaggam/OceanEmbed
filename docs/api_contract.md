# OceanEmbed API Contract Documentation

**Version:** 0.1.0  
**Base URL:** `http://127.0.0.1:8000/api`  
**Domain:** North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}$, $45^\circ\text{E}–105^\circ\text{E}$)

---

## 1. Overview & Architecture

OceanEmbed provides RESTful HTTP/JSON endpoints for retrieving reconstructed subsurface ocean temperatures across **15 standard depths**:
`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000` meters.

The backend uses a **pluggable provider design**:
- `MockInferenceProvider`: High-fidelity synthetic oceanographic climatology generator for UI development and testing.
- `RealOceanEmbedProvider`: Plug-in interface for loading trained PyTorch weights and executing GPU/CPU inference without frontend changes.

---

## 2. Endpoints

### 2.1 Health Check
- **Route:** `GET /api/health`
- **Description:** Checks service status, active provider, domain constraints, and depths.
- **Response `200 OK`:**
```json
{
  "status": "healthy",
  "project": "OceanEmbed API",
  "version": "0.1.0",
  "active_provider": "mock_climatology",
  "is_mock": true,
  "domain": {
    "lat_min": 5.0,
    "lat_max": 30.0,
    "lon_min": 45.0,
    "lon_max": 105.0
  },
  "depths_m": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
}
```

---

### 2.2 Reconstruct Subsurface Temperature Profile
- **Route:** `POST /api/reconstruct`
- **Description:** Reconstructs the 15-depth vertical temperature profile for a given date and location.
- **Request Body:**
```json
{
  "date": "2023-06-15",
  "latitude": 18.5,
  "longitude": 88.25
}
```
- **Validation Rules:**
  - `date`: Valid ISO string `YYYY-MM-DD`.
  - `latitude`: Float bounded between `5.0` and `30.0`.
  - `longitude`: Float bounded between `45.0` and `105.0`.
- **Response `200 OK`:**
```json
{
  "request_id": "rec_20230615_1850_08825",
  "date": "2023-06-15",
  "latitude": 18.5,
  "longitude": 88.25,
  "depths_m": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
  "temperature_c": [29.8, 29.72, 29.65, 29.5, 29.2, 26.5, 22.8, 19.1, 15.8, 13.9, 11.8, 9.4, 7.6, 6.2, 5.4],
  "surface_context": {
    "sst_c": 29.8,
    "sss_psu": 32.2,
    "ssh_m": 0.19,
    "current_u_ms": 0.22,
    "current_v_ms": -0.16,
    "wind_u_ms": -4.2,
    "wind_v_ms": 5.5
  },
  "d26_depth_m": 53.2,
  "mixed_layer_depth_m": 31.0,
  "argo_comparison": {
    "float_id": "INCOIS-2902145",
    "date": "2023-06-15",
    "latitude": 18.5,
    "longitude": 88.25,
    "distance_km": 0.0,
    "depths_m": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
    "temperature_c": [29.65, 29.58, 29.45, 29.25, 28.7, 25.85, 21.9, 18.4, 15.2, 13.5, 11.45, 9.15, 7.45, 6.1, 5.35],
    "rmse": 0.42,
    "mae": 0.38,
    "bias": 0.35,
    "is_mock": true
  },
  "embedding": {
    "pca_1": 2.57,
    "pca_2": -1.09,
    "regime_label": "Bay of Bengal Freshwater Plume",
    "vector_dim": 128
  },
  "model": {
    "name": "OceanEmbed",
    "version": "0.1.0-mock",
    "provider_type": "mock_climatology",
    "checkpoint_hash": null,
    "inference_time_ms": 1.25
  },
  "is_mock": true,
  "provenance": "Synthetic Oceanographic Climatology Generator (Mock Provider).",
  "timestamp": "2026-09-19T12:00:00Z"
}
```

---

### 2.3 Ocean Embedding 2D Scatter Data
- **Route:** `GET /api/embedding`
- **Description:** Returns historical 2D PCA/UMAP background points representing oceanographic regimes across the basin.
- **Response `200 OK`:**
```json
{
  "points": [
    {
      "id": "pt-01",
      "pca_1": 2.8,
      "pca_2": -1.3,
      "regime": "Bay of Bengal Freshwater Plume",
      "region": "Bay of Bengal",
      "season": "SW Monsoon",
      "latitude": 19.5,
      "longitude": 89.0
    }
  ],
  "regimes": [
    "Arabian Sea High-Salinity Water",
    "Bay of Bengal Freshwater Plume",
    "Central Arabian Sea",
    "Central Bay of Bengal",
    "Equatorial Warm Pool",
    "Somali / Oman Upwelling Zone"
  ],
  "total_points": 20,
  "is_mock": true
}
```

---

### 2.4 Find Nearby In-Situ Argo Float
- **Route:** `GET /api/argo/nearby?date=2023-06-15&latitude=18.5&longitude=88.25`
- **Description:** Searches for independent blind Argo float profiles within temporal/spatial search radius.
- **Response `200 OK`:** `ArgoObservation` or `null`.
