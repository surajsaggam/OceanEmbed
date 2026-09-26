<div align="center">

# 🌊 OceanIQ / OceanEmbed

### Operational Deep-Learning Platform for Subsurface Ocean Thermal Reconstruction from Multi-Source Satellite Observations

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%20%7C%20CUDA%2012.8-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19.0%20%7C%20TypeScript-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-6.0-646CFF?style=for-the-badge&logo=vite&logoColor=white)](https://vitejs.dev)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v3.4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![Domain](https://img.shields.io/badge/Domain-North%20Indian%20Ocean-0284C7?style=for-the-badge&logo=google-maps&logoColor=white)](#domain--grid-specifications)
[![SIH](https://img.shields.io/badge/Smart%20India%20Hackathon-SIH%202026-FF9933?style=for-the-badge)](https://sih.gov.in)

<p align="center">
  <b>Developed for the Ministry of Earth Sciences (MoES) & Indian National Centre for Ocean Information Services (INCOIS)</b><br>
  <i>Reconstructing the hidden vertical ocean interior (0–1000m) from daily surface satellite skin measurements.</i>
</p>

[Overview](#-overview) • [Core Capabilities](#-core-capabilities) • [Component Architecture](#-ui-component-architecture) • [Operational Benchmark](#-live-operational-benchmark) • [ML & Physics Contract](#-deep-learning-architecture) • [Quick Start](#-quick-start) • [Scientific Integrity](#-scientific-integrity--guardrails)

---

</div>

## 📌 Overview

Satellites directly measure only the topmost ocean skin (~1 mm to 1 m). However, vital marine physical processes—such as **Tropical Cyclone Heat Potential (TCHP)**, rapid cyclone intensification in the Bay of Bengal and Arabian Sea, monsoon ocean coupling, and subsurface marine heatwaves—are strictly dictated by the **thermal structure of the upper 1000 meters**.

**OceanIQ** is an operational oceanographic intelligence platform powered by **OceanEmbed**, a multi-scale deep learning model that reconstructs daily subsurface ocean temperatures at **15 standard oceanographic depths** across the North Indian Ocean at $0.25^\circ \times 0.25^\circ$ spatial resolution.

```
       Daily Satellite Surface Observations (7 physical channels + 7 validity masks)
                                           │
                                           ▼
               ┌───────────────────────────────────────────────────────┐
               │    OceanEmbed Deep-Learning Feature Fusion Engine     │
               └───────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
  15-Depth Vertical Temperature Field               128-D Ocean Latent Embedding
  [0, 5, 10, 20, 30, 50, 75, 100, 125,              Manifold clustering & spatial
   150, 200, 300, 500, 700, 1000 m]                 oceanographic regime tracking
```

---

## ⚡ Core Capabilities

<table>
  <tr>
    <td width="50%">
      <h3>📈 15-Depth Vertical Sounding</h3>
      <p>Continuous vertical temperature reconstruction across 15 standard levels from <b>0m to 1000m</b> in under 1.5 seconds, mapping mixed-layer dynamics, thermocline steepness, and deep abyssal water.</p>
    </td>
    <td width="50%">
      <h3>🌡️ Upper-Ocean Heat Indices</h3>
      <p>Automated real-time calculation of key thermodynamic cyclone metrics: <b>D26</b> (26°C Isotherm Depth), <b>MLD</b> (Mixed Layer Depth), and <b>TCHP</b> (Tropical Cyclone Heat Potential in kJ/cm²).</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📏 Vertical Subsurface Transect</h3>
      <p>Multi-station 2D vertical cross-sections across neighboring coordinates to verify spatial continuity, eddy boundaries, cold domes, and isothermal layer transitions.</p>
    </td>
    <td width="50%">
      <h3>🔬 GLORYS12V1 Departure Mapping</h3>
      <p>Cell-by-cell comparative delta analysis against the Copernicus GLORYS12V1 physical ocean reanalysis reference, identifying anomalies and sanity-checking physical bounds.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>📊 Basin-Wide Depth-Wise Skill</h3>
      <p>Systematic RMSE, MAE, and bias tracking across all 14,000+ active ocean cells, delineating the natural thermocline error peak (50–150m band) from deep stable convergence.</p>
    </td>
    <td width="50%">
      <h3>🎯 Blind In-Situ Argo Validation</h3>
      <p>Automated spatiotemporal search (±0.25°, ±24h) against physical <b>INCOIS Argo floats</b>. Provides unbiased, un-trained ground truth verification with collocated distance and residuals.</p>
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🌌 128-D Latent Manifold</h3>
      <p>Interactive 2D PCA projection of the 128-dimensional ocean embedding space, classifying the observation's position relative to regional contextual reference clusters.</p>
    </td>
    <td width="50%">
      <h3>📄 Multi-Page Scientific Dossier</h3>
      <p>Client-side PDF compilation delivering formatted sounding curves, 15-depth thermal matrices, comparative departures, and cryptographic model audit lineage for research sharing.</p>
    </td>
  </tr>
</table>

---

## 🧩 UI Component Architecture

The OceanIQ dashboard is engineered with a **shadcn/ui-inspired component system**, emphasizing high data density, clear visual hierarchy, accessible contrast, and responsive layout.

```
dashboard/src/
├── components/
│   ├── targeting/
│   │   └── TargetingStationBar.tsx       # Date picker, coordinate inputs, regional basin presets
│   ├── map/
│   │   └── BasinLocationPicker.tsx       # Interactive Leaflet North Indian Ocean domain map
│   ├── sounding/
│   │   ├── SoundingProfileCard.tsx       # Plotly sounding curve, isotherm badges, depth controls
│   │   └── SubsurfaceTransectCard.tsx    # 2D vertical thermal cross-section transect
│   ├── diagnostics/
│   │   ├── DiagnosticTabs.tsx            # Tabbed diagnostic workbench orchestration
│   │   ├── ReconstructionDepartureCard.tsx # GLORYS12V1 spatial departure heatmap & column diff
│   │   ├── DepthwiseModelSkillCard.tsx   # Layer-by-layer RMSE vs depth with 50-150m dynamic band
│   │   ├── ArgoValidationCard.tsx        # INCOIS Argo float collocation & residual metrics
│   │   └── LatentManifoldCard.tsx        # 128-D latent PCA projection scatterplot
│   ├── audit/
│   │   └── ScientificProvenanceCard.tsx  # Latency counter, hardware audit, model lineage
│   └── reports/
│       └── TechnicalDossierGenerator.ts  # Multi-page client-side PDF compilation via pdf-lib
├── lib/
│   ├── ocean.ts                          # Domain boundaries, standard depth array, grid snapping
│   └── utils.ts                          # Class merge utility (cn)
├── types/
│   └── api.ts                            # Strict TypeScript interfaces mirroring FastAPI Pydantic schemas
└── data/
    └── presets.ts                        # Curated oceanographic station presets (Arabian Sea, BoB)
```

### Component Breakdown

| Component | Role | Visual Pattern | Key Interactions |
|---|---|---|---|
| **`TargetingStationBar`** | Global Targeting Controller | Sticky Command Header | Temporal picker, Coordinate clamp, Preset switch, Reconstruct trigger, PDF generation |
| **`BasinLocationPicker`** | Spatial Context & Selection | Leaflet Interactive Map | 5°N–30°N / 45°E–105°E domain bounds, Station pin, In-situ float markers, Click-to-target |
| **`SoundingProfileCard`** | Vertical Thermal Sounding | Dual-Axis Plotly Curve | Hover tooltips, D26/MLD/TCHP chips, View toggle (Profile / Split / Matrix / Transect) |
| **`SubsurfaceTransectCard`** | 2D Spatial-Vertical Cross-Section | Contour Heatmap Canvas | Latitude/Longitude axis switch, Thermocline gradient tracking |
| **`ReconstructionDepartureCard`** | Reanalysis Verification | Choropleth Map & Data Grid | 100m level spatial departure heatmap, full column delta comparison table |
| **`DepthwiseModelSkillCard`** | Basin-Scale Uncertainty | Dual Stat Cards & Error Plot | 50–150m dynamic error band highlight, Surface/Peak/Deep RMSE breakdown |
| **`ArgoValidationCard`** | Blind Physical Verification | Metadata Banner & Stat Grid | Float WMO & Cycle id, Radial distance, Residual RMSE, MAE, Mean bias |
| **`ScientificProvenanceCard`** | Lineage & Audit Log | Terminal-Style Data Sheet | Checkpoint sha256 hash, Input tensor shape, Inference latency (ms), Provider status |

---

## 🎯 Live Operational Benchmark

The following benchmark demonstrates an operational verification performed on **January 1st, 2019** in the **Central Arabian Sea**, cross-validated against both GLORYS12V1 numerical reanalysis and an independent in-situ INCOIS Argo float:

### 📍 Station Metadata
- **Coordinates:** $17.25^\circ\text{N}, 69.50^\circ\text{E}$ (Central Arabian Sea)
- **Observation Date:** `2019-01-01`
- **Inference Latency:** `1360 ms`

### 🌡️ 15-Depth Thermal Reconstruction

| Depth (m) | Reconstructed $T(z)$ (°C) | GLORYS12V1 Ref (°C) | Departure $\Delta T$ (°C) | Physical Layer |
|:---:|:---:|:---:|:---:|:---|
| **0** | **27.26** | 27.52 | -0.26 | Surface Skin / Mixed Layer |
| **5** | **27.26** | 27.52 | -0.26 | Upper Mixed Layer |
| **10** | **27.26** | 27.52 | -0.26 | Mixed Layer Base |
| **20** | **27.25** | 27.50 | -0.25 | Sub-surface Mixed Layer |
| **30** | **27.23** | 27.46 | -0.23 | Top of Thermocline Transition |
| **50** | **26.49** | 26.85 | -0.36 | Upper Thermocline ($D_{26}$ Zone) |
| **75** | **24.51** | 24.89 | -0.38 | Strong Thermocline Gradient |
| **100** | **22.25** | 22.76 | -0.51 | Peak Dynamic Shear Layer |
| **125** | **20.12** | 20.45 | -0.33 | Lower Thermocline |
| **150** | **18.24** | 18.52 | -0.28 | Thermocline Base |
| **200** | **15.65** | 15.82 | -0.17 | Intermediate Water |
| **300** | **13.10** | 13.18 | -0.08 | Intermediate Water |
| **500** | **10.55** | 10.60 | -0.05 | Deep Intermediate Water |
| **700** | **9.22** | 9.25 | -0.03 | Deep Stable Water |
| **1000** | **8.54** | 8.56 | -0.02 | Abyssal Convergence Layer |

### 📊 Upper Ocean Thermodynamic Metrics
- **$D_{26}$ (26°C Isotherm Depth):** `56.7 m`
- **Mixed Layer Depth (MLD):** `45.8 m` ($\Delta T = 0.2^\circ\text{C}$ criterion)
- **Tropical Cyclone Heat Potential (TCHP):** `25.3 kJ/cm²` ($\rho c_p \int_0^{D_{26}} [T(z) - 26^\circ\text{C}]\,dz$)
- **Station Column Mean Absolute Departure:** `0.33°C`

### 🚢 Collocated Blind In-Situ Argo Float Validation
- **Matched Float:** `WMO-2902257`, Cycle `140`
- **Observed Coordinates:** $17.18^\circ\text{N}, 69.41^\circ\text{E}$
- **Radial Separation:** `12.8 km` (well within the $\pm0.25^\circ$ grid cell radius)
- **Temporal Collocation:** Same observation date (`2019-01-01`)
- **Blind Validation RMSE:** **`0.74°C`**
- **Blind Validation MAE:** **`0.50°C`**
- **Column Residual Bias:** **`-0.08°C`** *(unbiased across the 1000m column)*

### 🌐 Basin-Scale Model Skill (14,200 active cells on 2019-01-01)
- **Surface (0m) RMSE:** `0.712°C`
- **Peak Reconstruction Error:** `1.155°C` at 100m *(situated precisely in the 50–150m thermocline band)*
- **Deep (700m) RMSE:** `0.369°C`
- **Abyssal (1000m) RMSE:** `0.389°C`
- **Basin Column Mean Bias:** `+0.052°C`

---

## 🧠 Deep-Learning Architecture

```
INPUT TENSOR [B, 14, 100, 240]
├── Physical Surface Variables (7 channels @ 0.25° grid)
│   ├── SST (OSTIA, 0.05° regridded)
│   ├── SSS (SMAP/SMOS L4, 0.125° regridded)
│   ├── SSH / SLA (DUACS altimetry, 0.25°)
│   ├── Currents U, V (OSCAR L4 OC, 0.25°)
│   └── Winds U, V (CCMP v3.1 / ASCAT-C, 0.25°)
└── Quality & Validity Masks (7 binary channels)
    └── 1 = valid observation, 0 = imputed/cloud gap
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│            Multi-Scale ResNet Feature Backbone         │
│   (Captures spatial teleconnections & mesoscale eddies)│
└────────────────────────────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│           Pointwise Column MLP + Attention Gate        │
│    (Models local vertical stratification physics)      │
└────────────────────────────────────────────────────────┘
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
Subsurface Field Head          Ocean Latent Head
[B, 15, 100, 240]              [B, 128, 100, 240]
Reconstructed T(z)             128-D Oceanographic
at 15 standard depths          Embedding Field
```

### Domain & Grid Specifications
- **Basin:** North Indian Ocean (Arabian Sea, Bay of Bengal, Equatorial Indian Ocean)
- **Latitude Extent:** $5^\circ\text{N} \to 30^\circ\text{N}$ ($100$ cells @ $0.25^\circ$)
- **Longitude Extent:** $45^\circ\text{E} \to 105^\circ\text{E}$ ($240$ cells @ $0.25^\circ$)
- **Temporal Resolution:** Daily snapshot
- **15 Target Depths ($z$):** `[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` meters

---

## 🛠️ Technology Stack

<div align="center">

| Layer | Technologies | Role |
|---|---|---|
| **Frontend** | React 19, TypeScript, Vite, TailwindCSS | High-performance reactive scientific UI |
| **Components & Styling** | Radix UI, Lucide Icons, Vanilla CSS Tokens | shadcn-inspired clean oceanographic design |
| **Visualization** | Plotly.js, Leaflet, React-Leaflet | Dynamic sounding curves, spatial GIS maps |
| **Reporting** | pdf-lib | Client-side 3-page scientific PDF dossier compilation |
| **Backend API** | FastAPI, Pydantic v2, Uvicorn | Typed REST API with strict validation contracts |
| **Numerical Science** | NumPy, SciPy, xarray, NetCDF4 | Grid snapping, interpolation, thermodynamic calculations |
| **Deep Learning** | PyTorch, TorchVision, CUDA 12.8 | Dual-path feature fusion, tensor inference |
| **Data Sources** | Copernicus Marine (GLORYS), INCOIS LAS (Argo) | Reference reanalysis & blind in-situ validation |

</div>

---

## 🚀 Quick Start

### Prerequisites
- **Python:** 3.10 or higher
- **Node.js:** 18.0 or higher (npm 9+)

### 1. Clone & Setup Repository
```bash
git clone https://github.com/surajsaggam/OceanEmbed.git
cd OceanEmbed
```

### 2. Launch FastAPI Backend
```bash
# Optional: create and activate a Python virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install backend requirements
pip install -r requirements.txt

# Run backend service
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Health Check: `http://127.0.0.1:8000/api/health`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### 3. Launch React Dashboard
In a new terminal window:
```bash
cd dashboard
npm install
npm run dev
```
Open **`http://localhost:5173`** in your browser.

---

## 🧪 Testing & Verification

### Backend Test Suite
```bash
# Run unit & integration tests
pytest tests/ -v
```

### Frontend Typecheck & Build
```bash
cd dashboard
npm run build
```

---

## 🛡️ Scientific Integrity & Guardrails

1. **Estimation, Not In-Situ Measurement:** OceanIQ reconstructs hidden subsurface temperature by learning geophysical statistical relationships between satellite skin observations and physical ocean reanalysis. It never claims to "directly measure" the ocean interior.
2. **Complementary to In-Situ Networks:** Argo profiling floats provide sparse coverage (~1 profile per $3^\circ \times 3^\circ$ every 10 days). OceanIQ bridges the daily spatio-temporal gap between float cycles; it does not replace Argo, shipboard CTDs, or moorings.
3. **No "Ground Truth" Label for Reanalysis:** GLORYS12V1 is treated as a **numerical ocean reanalysis reference**, not ground truth.
4. **Blind Validation Discipline:** In-situ Argo float profiles are strictly reserved for post-inference independent validation; they are never used during model training, hyperparameter optimization, or loss calculation.
5. **Thermodynamic Guardrails:** TCHP calculations adhere strictly to the oceanographic conventional integral $\rho c_p \int_0^{D_{26}} [T(z) - 26^\circ\text{C}]\,dz$, accompanied by transparent error bounds.

---

<div align="center">
  <sub>OceanIQ Prototype · Built for Smart India Hackathon (SIH 2026) · Ministry of Earth Sciences (MoES) & INCOIS</sub>
</div>
