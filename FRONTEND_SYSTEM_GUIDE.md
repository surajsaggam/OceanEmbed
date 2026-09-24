# OceanIQ — Comprehensive Frontend & Integration Guide for ML Engineers

> **Target Audience:** ML / DL Team Members (Rishabh & Model Authors)  
> **Purpose:** This document explains how the OceanEmbed machine learning model, weights, input tensors, and validation datasets are integrated into the live OceanIQ web platform, how the frontend and backend bridge work, and how to run, inspect, and update the application.

---

## 1. Executive Summary & Quick Start

The **OceanIQ** application is a local-first, publication-grade oceanographic analysis platform that provides an interactive graphical interface for the **OceanEmbed** deep-learning model.

While you developed the neural network architecture, training pipeline, and validation notebooks, the frontend team built an interactive user interface, an API serving layer, an Argo float collocation matching engine, and an automated 4-page PDF technical report generator.

### Quick Start (Running the Entire Platform Locally)

The system runs on two lightweight local processes:

#### Terminal 1: Backend (FastAPI + PyTorch Inference)
```bash
# From workspace root: S:\OceanEmbed
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
* **API Documentation & Swagger UI:** `http://127.0.0.1:8000/docs`
* **Health Check:** `http://127.0.0.1:8000/api/health`

#### Terminal 2: Frontend (React + TypeScript + Vite)
```bash
# From workspace root: S:\OceanEmbed
cd dashboard
npm run dev
```
* **Live Web Dashboard:** `http://localhost:5173`

---

## 2. End-to-End System Architecture

Here is the exact data flow from when a user interacts with the map to when your neural network produces the final 15-depth reconstruction:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        USER BROWSER (React + TS)                       │
│  - Interactive North Indian Ocean Map (Arabian Sea & Bay of Bengal)    │
│  - User selects Date + Coordinates (e.g. 18.5°N, 88.25°E)              │
│  - Clicks "Run Reconstruction"                                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP POST /api/reconstruct
                                    │ JSON: {"date": "...", "latitude": 18.5, "longitude": 88.25}
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND BRIDGE (`api/`)                   │
│                                                                        │
│ 1. Coordinate & Domain Validation (5°N–30°N, 45°E–105°E)               │
│ 2. Data Provider (`api/services/real_provider.py`):                    │
│    - Extracts 7 surface variables from `oceanembed_2019-01-01.npz`     │
│    - Extracts 7 binary quality validity masks (14 channels total)      │
│    - Applies z-score normalization via `train_stats.json`              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ Tensor: [1, 14, 1, 1]
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     FROZEN ML PREDICTOR (`inference/`)                 │
│                                                                        │
│ 1. Loads checkpoint: `checkpoints/phase1/best.pt`                      │
│ 2. Verifies SHA256 integrity:                                          │
│    `f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b`│
│ 3. Forward pass under `torch.no_grad()` (~40 ms latency)               │
│ 4. Un-normalizes output using target stats (mean, std)                 │
│ 5. Returns:                                                            │
│    - 15 subsurface temperatures (°C) at authoritative depths           │
│    - 128-dimensional latent representation vector Z                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   POST-PROCESSING & VALIDATION ENGINE                  │
│                                                                        │
│ 1. Calculates MLD (Mixed Layer Depth, ΔT = 0.5°C threshold from skin)  │
│ 2. Calculates D26 (Depth of 26°C isotherm, only if profile crosses 26) │
│ 3. 2D PCA projection of 128-D embedding for visual manifold            │
│ 4. Independent Argo Spatial Matchup:                                   │
│    - Queries `argo_2019_raw_measurements.parquet` within 50 km radius  │
│    - Computes real RMSE, MAE, Bias against collocated in-situ float    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    │ Typed JSON Response (ReconstructionResponse)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     RICH FRONTEND DASHBOARD RENDERING                  │
│  - Vertical Thermal Sounding Curve (0–1000m) with Argo overlay         │
│  - Discrete 15-Depth Thermal Matrix & Sounding Schedule                │
│  - Surface Driver breakdown (SST, SSS, SSH, Currents, Winds)          │
│  - 128-D Latent Manifold Projection scatter plot                      │
│  - In-Situ Argo Ground Truth scorecard (RMSE, MAE, Bias)              │
│  - PDF Report Generator: downloads 4-page publication-grade PDF        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. How Your ML Artifacts Are Utilized

The frontend does **NOT** retrain, fine-tune, or modify your model. It interacts with your exact frozen files:

| Artifact Path | SHA256 / Verification | How It Is Used in the Application |
| :--- | :--- | :--- |
| `checkpoints/phase1/best.pt` | `f3d99a9b9214efe...` | Loaded into PyTorch by `inference/predictor.py`. Strict SHA256 integrity check verified at startup. |
| `data/norm_stats/train_stats.json` | JSON schema | Provides channel-wise `mean` and `std` for normalizing the 7 surface variables before model entry. |
| `data/norm_stats/global_dev_metadata.json` | JSON schema | Contains domain coordinates, depth grids, and target un-normalization statistics. |
| `data/processed/test/oceanembed_2019-01-01.npz` | 0.25° daily grid | Provides the true multi-source satellite observations (`sst`, `sss`, `ssh`, `u_curr`, `v_curr`, `u_wind`, `v_wind`) across the NIO basin. |
| `data/argo/argo_2019_raw_measurements.parquet` | 11,353 in-situ profiles | Used **only** as independent ground truth validation. If a float was within 50 km on that date, it computes true RMSE/MAE/Bias. **Never used as input to the neural network.** |

---

## 4. API Contract & Interface Specification

The frontend and ML backend communicate exclusively over typed HTTP JSON contracts.

### Endpoint: `POST /api/reconstruct`

#### Request Payload
```json
{
  "date": "2019-01-01",
  "latitude": 18.5,
  "longitude": 88.25
}
```

#### Response Payload (Authoritative Structure)
```json
{
  "date": "2019-01-01",
  "latitude": 18.5,
  "longitude": 88.25,
  "depths_m": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
  "temperature_c": [27.27, 27.31, 27.34, 27.62, 27.88, 28.61, 27.73, 25.74, 22.99, 20.12, 16.18, 12.48, 10.42, 8.95, 6.95],
  "mixed_layer_depth_m": 87.1,
  "d26_depth_m": 96.7,
  "surface_context": {
    "sst_c": 27.27,
    "sss_psu": 32.14,
    "ssh_m": 0.082,
    "current_u_ms": -0.154,
    "current_v_ms": 0.089,
    "wind_u_ms": 2.31,
    "wind_v_ms": -1.84,
    "data_source": "OceanIQ observation archive"
  },
  "embedding": {
    "dimension": 128,
    "pca_1": 1.42,
    "pca_2": -0.68,
    "regime_label": "Bay of Bengal Central Gyre"
  },
  "argo_comparison": {
    "float_id": "ARGO-2902685",
    "distance_km": 14.8,
    "date": "2019-01-01",
    "temperature_c": [27.15, 27.20, 27.22, 27.45, 27.70, 28.40, 27.50, 25.50, 22.80, 19.90, 16.00, 12.30, 10.20, 8.80, 6.90],
    "rmse": 0.21,
    "mae": 0.17,
    "bias": 0.12
  },
  "model": {
    "name": "OceanEmbed",
    "version": "Phase-1",
    "checkpoint_hash": "f3d99a9b9214efe92a4e8fb11bd49b759a62cfb5d991d1225fd1360362876b9b",
    "provider_type": "PyTorch (Evaluation Mode)",
    "inference_time_ms": 38.4
  },
  "provenance": "Phase-1 frozen PyTorch model checkpoint evaluated on 2019-01-01 multi-source observations.",
  "is_mock": false
}
```

### Critical Oceanographic Rules Implemented in the Engine
1. **D26 Isotherm Handling**: If the reconstructed temperature profile never reaches 26.0°C (e.g., cold regions or high latitudes where maximum profile temperature $< 26.0^\circ\text{C}$), `d26_depth_m` is strictly `null`. It is **never** reported as `0.0 m`. The UI and PDF report display `"Not reached"`.
2. **MLD Calculation**: Evaluated as the depth where temperature drops by $0.5^\circ\text{C}$ relative to the ocean surface skin temperature.
3. **Data Provenance**: To avoid inventing unverified satellite mission names, all multi-source input observations are attributed to the `"OceanIQ observation archive"`.

---

## 5. Frontend Architecture & Component Directory

The web frontend is written in **React 18 + TypeScript**, styled using modern **Tailwind CSS**, and compiled via **Vite**.

```
dashboard/
├── src/
│   ├── components/
│   │   ├── shell/
│   │   │   └── AppHeader.tsx               # Top navigation, status indicator, model badge
│   │   ├── targeting/
│   │   │   └── TargetingStationBar.tsx     # Date/coord inputs, "Run Reconstruction" & "Generate Report"
│   │   ├── map/
│   │   │   └── BasinLocationPicker.tsx     # Interactive Leaflet map (5°N–30°N, 45°E–105°E)
│   │   ├── profile/
│   │   │   ├── ThermalSoundingPlot.tsx     # 0–1000m vertical profile curve with Argo overlay
│   │   │   └── ThermalMetricsMatrix.tsx   # D26, MLD, Argo RMSE scorecards
│   │   ├── workbench/
│   │   │   └── AnalysisWorkbench.tsx       # Main tabbed view: Profile, Depth Matrix, History
│   │   ├── surface/
│   │   │   └── SurfaceDriversPanel.tsx     # Breakdown of the 7 input surface features
│   │   ├── manifold/
│   │   │   └── LatentManifoldView.tsx      # 128-D latent representation 2D PCA projection
│   │   ├── validation/
│   │   │   └── ArgoValidationPanel.tsx     # Collocated Argo float details, RMSE/MAE/Bias
│   │   ├── audit/
│   │   │   └── ScientificProvenanceCard.tsx# SHA256 integrity, parameter counts, model provenance
│   │   └── dialogs/
│   │       ├── ModelSpecsDialog.tsx        # Technical modal of neural architecture & depths
│   │       └── AboutDialog.tsx             # Scientific caveats and oceanographic background
│   ├── services/
│   │   └── api.ts                          # Axios/Fetch client calling FastAPI backend
│   ├── types/
│   │   └── reconstruction.ts               # TypeScript interfaces matching backend Pydantic models
│   ├── App.tsx                             # Master application layout
│   └── main.tsx                            # React entry point
```

### Detailed Component Overview

#### 1. Header & System Badge (`AppHeader.tsx`)
* Displays application branding (**OceanIQ**).
* Shows the active backend status badge: `"Phase-1 Frozen Model: ACTIVE"`.
* Shows the checkpoint SHA256 badge (`f3d99a9b...`) to guarantee to oceanographers that the model weights are unmodified.
* Contains modals for technical specifications and scientific methodology.

#### 2. Station Controls (`TargetingStationBar.tsx`)
* Allows the user to select dates and enter latitude/longitude.
* Provides preset oceanographic stations (e.g., *Bay of Bengal Central Gyre*, *Arabian Sea Upwelling Zone*, *Somali Current*, *Equatorial Indian Ocean*).
* Hosts the primary **Run Reconstruction** button (triggers neural inference in ~40ms).
* Hosts the **Generate Report** button (compiles and downloads the 4-page PDF).

#### 3. Basin Location Picker (`BasinLocationPicker.tsx`)
* Leaflet map covering the North Indian Ocean domain ($5^\circ\text{N} - 30^\circ\text{N}, 45^\circ\text{E} - 105^\circ\text{E}$).
* Users can click anywhere on the ocean surface to set coordinates.
* Shows held-out Argo float pins across the basin so users can test locations where real in-situ validation exists.

#### 4. Thermal Sounding Curve (`ThermalSoundingPlot.tsx`)
* Displays ocean depth on the vertical descending axis ($0\text{ m}$ at the top down to $1000\text{ m}$ at the bottom).
* Temperature is plotted on the horizontal axis ($4^\circ\text{C}$ to $32^\circ\text{C}$).
* Displays point dots for all **15 standard depths**: `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m]`.
* Provides an interactive toggle to overlay the collocated in-situ Argo float curve when available.
* Annotates D26 (green dashed isobath) and MLD (orange dotted isobath).

#### 5. Depth Matrix View (`AnalysisWorkbench.tsx`)
* A dedicated tab rendering a dual-view thermal matrix:
  * **Left:** A continuous physical sounding ribbon showing the vertical temperature gradient from surface to abyss.
  * **Right:** 15 discrete tiles for each standard depth level, colored using the scientific Turbo colormap with high-contrast temperature readouts and oceanographic zone tags (*Surface Ocean Skin*, *Epipelagic Layer*, *Thermocline Zone*, *Mesopelagic Water*, *Deep NIO Abyss*).

#### 6. Latent Manifold Visualizer (`LatentManifoldView.tsx`)
* Visualizes your model's **128-dimensional latent representation $Z$**.
* Projects the vector into a 2D PCA feature space ($PC_1$ vs $PC_2$).
* Displays **Contextual Reference Points** representing regional water clusters across the North Indian Ocean.
* Plots the active reconstruction as an indigo star, showing how your model maps the current surface state within its learned manifold.

#### 7. Independent Argo In-Situ Validation (`ArgoValidationPanel.tsx`)
* When an Argo float is collocated ($< 50\text{ km}$):
  * Displays platform WMO ID (e.g. `ARGO-2902685`), cycle number, and distance.
  * Displays overall **RMSE**, **MAE**, and **Bias** against the in-situ float.
  * Shows a depth-by-depth residual schedule ($\Delta T = T_{\text{pred}} - T_{\text{argo}}$) with color-coded errors.
* When no float is collocated:
  * Explicitly displays a scientific notice that no float was within radius, benchmarked against Phase-1 baseline metrics without fabricating synthetic float curves.

---

## 6. Automated PDF Technical Report Generation

When users click **"Generate Report"**, the backend service ([api/services/pdf_report_service.py](file:///s:/OceanEmbed/api/services/pdf_report_service.py)) compiles an authoritative 4-page technical oceanographic report:

* **Engine:** Built with **ReportLab** (Platypus flowables) and **Matplotlib** (headless Agg backend, 220 DPI rendering).
* **Page 1: Cover & Observation Coordinates**: Title, coordinate badges, oceanographic regime, and summary metrics card (D26, MLD, Argo RMSE).
* **Page 2: Vertical Profile & 15-Depth Table**: High-resolution sounding curve image and a 15-depth tabular schedule with Argo residuals.
* **Page 3: Depth Matrix & Surface Drivers**: Dual-column continuous sounding column + discrete 15-depth matrix chart, plus a breakdown table and kinematics bar chart of the 7 surface variables.
* **Page 4: Argo Ground Truth & Latent Space**: Float validation scorecard, 2D PCA projection of the 128-D embedding, model lineage provenance, and mandatory scientific disclaimer.

---

## 7. How the ML Developer Can Extend the System

As you train new checkpoints (e.g., Phase-2, multi-day temporal models, or improved loss functions), you can plug them directly into the platform without touching the frontend:

### 1. Adding a New Model Checkpoint
1. Place your PyTorch `.pt` checkpoint file into `checkpoints/phase2/best.pt`.
2. Compute its SHA256 hash:
   ```bash
   certutil -hashfile checkpoints/phase2/best.pt SHA256
   ```
3. Update [api/services/real_provider.py](file:///s:/OceanEmbed/api/services/real_provider.py) to point to the new checkpoint path and expected SHA256.
4. Ensure your model's forward pass returns:
   - Reconstructed temperatures: tensor of shape `[batch, 15]`
   - Latent embeddings: tensor of shape `[batch, 128]`

### 2. Adding New Observation Dates
1. Prepare your harmonized `.npz` file containing the 7 surface channels + quality masks.
2. Save it to `data/processed/test/oceanembed_YYYY-MM-DD.npz`.
3. The backend provider automatically detects available `.npz` files matching the requested date.

### 3. Adding New Argo Datasets
* If you acquire new Argo validation parquets (e.g., 2020 or 2021 floats), drop the parquet into `data/argo/`.
* The spatial lookup logic in `real_provider.py` will automatically query it for collocation matches.

---

## 8. Directory & File Reference

| File / Folder | Role & Description |
| :--- | :--- |
| `api/main.py` | FastAPI application entry point, CORS configuration, API routes. |
| `api/services/real_provider.py` | Inference orchestration: loads model, extracts `.npz` slice, runs PyTorch, matches Argo. |
| `api/services/pdf_report_service.py` | ReportLab PDF compiler for the 4-page scientific report. |
| `api/schemas/` | Pydantic data schemas defining request and response structures. |
| `inference/predictor.py` | Low-level PyTorch wrapper: SHA256 verification and tensor evaluation. |
| `checkpoints/phase1/best.pt` | Frozen Phase-1 neural network weights (525,040 parameters). |
| `data/norm_stats/` | Normalization statistics (`train_stats.json`, `global_dev_metadata.json`). |
| `data/processed/test/` | Multi-source surface observation grids (`.npz`). |
| `data/argo/` | Verified independent in-situ Argo float profiles (`.parquet`). |
| `dashboard/src/` | Complete React + TypeScript frontend dashboard source code. |
| `tests/backend/` | Comprehensive test suite (30 unit & integration tests). |

---

## 9. Summary for Rishabh

The platform is designed so that you can focus entirely on the deep-learning model, loss formulations, and oceanographic validation. The frontend and backend layer handles user interactions, geographic visualization, Argo collocation calculations, and automated report generation while preserving the scientific integrity of your model outputs.

To see the platform running live with your model right now, simply run:
1. `python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload`
2. `cd dashboard && npm run dev`
3. Open `http://localhost:5173` in your browser!
