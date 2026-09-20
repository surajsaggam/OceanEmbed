# OceanEmbed

**Satellite-Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Observations**

> **Ministry:** Ministry of Earth Sciences (MoES) · **Agency:** INCOIS · **Domain:** North Indian Ocean (5°N–30°N, 45°E–105°E)  
> **Authoritative Technical Source:** [`OceanEmbed_Final_Technical_Plan.md`](OceanEmbed_Final_Technical_Plan.md)  
> **Prototype Governance:** [`AGENTS.md`](AGENTS.md)

---

## 1. Scientific Problem & Purpose

Satellites directly observe only the top ocean skin (~1 m). However, critical processes such as Tropical Cyclone Heat Potential (TCHP), rapid cyclone intensification in the Bay of Bengal and Arabian Sea, monsoon coupling, and subsurface marine heatwaves are governed by the **subsurface thermal structure** down to 1000 m.

**OceanEmbed** reconstructs daily subsurface temperature across the North Indian Ocean on a $0.25^\circ \times 0.25^\circ$ grid at **15 authoritative standard depths**:
$$\text{Depths: } [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{ m}]$$

### Multi-Source Surface Inputs (14-Channel Tensor)
1. **SST:** Sea Surface Temperature (OSTIA, 0.05°)
2. **SSS:** Sea Surface Salinity (SMAP/SMOS L4 blended, 0.125°)
3. **SSH / SLA:** Sea Surface Height / SLA (DUACS altimetry, 0.25°)
4. **Currents ($U, V$):** OSCAR L4 OC (0.25°)
5. **Winds ($U, V$):** CCMP v3.1 / ASCAT-C (0.25°)
6. **7 Binary Validity Masks:** ($1 = \text{observed}, 0 = \text{imputed/climatology}$) to handle cloud gaps and sensor outages without naive imputation artifacts.

### Scientific Framing & Guardrails
- **Estimation, Not Measurement:** OceanEmbed learns statistical/physical relationships between satellite surface observations and subsurface reanalysis; it estimates hidden thermal structure and never claims to "measure" the deep ocean directly.
- **Complements In-Situ Networks:** Argo floats provide sparse coverage (~1 profile per $3^\circ$ per 10 days); OceanEmbed fills the continuous daily gaps between them and does not replace Argo floats, ships, or moorings.
- **Validation Discipline:** Trained against GLORYS12V1 ocean physical reanalysis and evaluated strictly out-of-time against independent, blind Argo float observations from INCOIS LAS.

---

## 2. Prototype Architecture & Separation of Concerns

This repository is the **OceanEmbed prototype/application**. It is architected so the separately developed ML/DL model can be plugged in without changing the user interface.

```text
                  React + TypeScript Dashboard
                  (Leaflet Map + Plotly Charts)
                               │
                               │ HTTP/JSON
                               ▼
                        FastAPI Backend
                               │
                               ▼
                     Inference Dispatcher
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
       MockInferenceProvider       RealOceanEmbedProvider
     (Synthetic Climatology)     (PyTorch Checkpoint Loader)
```

- **Prototype Scope:** Frontend UI, scientific visualization (15-depth vertical profiles, Argo comparison overlays, 2D PCA latent embedding scatter), RESTful API, mock data generator, typed contracts.
- **ML Teammate Scope:** Model architecture (dual-path CNN+MLP), reanalysis training, blind Argo evaluation, and checkpoint generation (`checkpoints/oceanembed_weights.pt`).

---

## 3. Project Structure

```text
OceanEmbed/
├── AGENTS.md                             # Agent governance and development rules
├── OceanEmbed_Final_Technical_Plan.md   # Authoritative scientific source of truth
├── README.md                             # This documentation
│
├── api/                                  # FastAPI Backend Application
│   ├── main.py                           # App entry, CORS, router mounting
│   ├── config.py                         # Settings, domain bounds (5-30N, 45-105E)
│   ├── routes/                           # REST endpoints (/reconstruct, /embedding, /argo, /health)
│   ├── schemas/                          # Typed Pydantic request/response models
│   └── services/                         # InferenceService, MockProvider, RealProvider
│
├── dashboard/                            # React + TypeScript Frontend (Vite)
│   ├── src/
│   │   ├── components/                   # Map, Controls, Profile Chart, Metrics Table, Embedding
│   │   ├── services/api.ts               # Typed HTTP client
│   │   ├── types/api.ts                  # TypeScript interfaces mirroring backend schemas
│   │   ├── data/presets.ts               # Oceanographic station presets
│   │   ├── App.tsx                       # Main application layout
│   │   └── index.css                     # Oceanographic design system tokens
│   └── package.json
│
├── ml_interface/                         # Boundary contract for the ML teammate
│   ├── base_provider.py                  # AbstractBaseClass for inference providers
│   ├── model_loader.py                   # PyTorch checkpoint loading utility
│   ├── tensor_contract.py                # Tensor dimension validators [B, 14, H, W] -> [B, 15, H, W]
│   └── README.md                         # Teammate integration instructions
│
├── mock_data/                            # Reference fixtures (Argo floats, embedding scatter)
├── tests/                                # Backend unit & integration test suite (pytest)
└── docs/                                 # API contracts & ML integration guides
```

---

## 4. Getting Started (Local Execution)

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 1. Launch Backend (FastAPI)
In the repository root:
```bash
# Run FastAPI server locally
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive API Swagger docs: `http://127.0.0.1:8000/docs`
- Service health endpoint: `http://127.0.0.1:8000/api/health`

### 2. Launch Dashboard (React + TypeScript)
In a separate terminal:
```bash
cd dashboard
npm install
npm run dev
```
Open `http://localhost:5173` in your browser.

---

## 5. Running Tests

Run the complete backend test suite:
```bash
python -m pytest tests/
```

Build and validate frontend TypeScript:
```bash
cd dashboard
npm run build
```

---

## 6. Scientific Provenance & Mock Data Policy

The application clearly demarcates synthetic/mock data from live model inference:
- When using the `MockInferenceProvider`, the dashboard displays a prominent **`DEMO / MOCK DATA`** badge.
- When the ML teammate supplies a checkpoint and `DEFAULT_PROVIDER=real` is set, the application automatically verifies the checkpoint and displays **`LIVE MODEL INFERENCE`**.
- In-situ Argo comparison metrics (RMSE, MAE, bias) are computed only against actual paired records and never fabricated.
