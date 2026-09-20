# OceanEmbed — Prototype Agent Instructions

## 1. Project Role and Scope

This repository is the **OceanEmbed prototype/application**.

The prototype owner is responsible for:
- frontend
- backend/API integration
- dashboard
- scientific visualization
- local application workflow
- integration with the separately developed ML/DL model

The ML/DL implementation itself is developed separately by another team member.

Do NOT independently redesign or reimplement the OceanEmbed scientific model unless explicitly requested.

---

## 2. Authoritative Technical Source

The file:

`OceanEmbed_Final_Technical_Plan.md`

is the single source of truth for:
- problem definition
- scientific terminology
- datasets
- model architecture
- 15 output depths
- validation methodology
- scientific limitations
- project roadmap

Frontend libraries and design references must never override this technical plan.

If a conflict is found, flag it instead of silently changing the scientific specification.

---

## 3. OceanEmbed Scientific Context

OceanEmbed reconstructs subsurface ocean temperature over the North Indian Ocean from daily surface observations.

Domain:
- 5°N–30°N
- 45°E–105°E
- 0.25° × 0.25° grid
- daily resolution

Output depths:

`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m`

Scientific framing:

Satellites directly observe the surface/ocean-skin layer. OceanEmbed estimates hidden subsurface temperature from multi-source surface observations using a deep-learning framework. The scientific model is trained against GLORYS reanalysis and independently evaluated against real Argo observations.

Never claim:
- satellites directly measure the reconstructed subsurface temperatures
- OceanEmbed replaces Argo, ships, or moorings
- uniform accuracy through 1000 m
- validation success before measured results exist

---

## 4. Prototype Architecture

The prototype should use:

```text
                 React + TypeScript
                         │
                         │ HTTP/JSON
                         ↓
                    FastAPI
                         │
                         ↓
              OceanEmbed inference
                         │
                         ↓
             ML model/checkpoint
```

The ML model is maintained separately.

The application must therefore be designed around a **stable inference interface** rather than embedding the entire training workflow inside the frontend.

The frontend must never execute Jupyter notebooks.

---

## 5. ML Integration Contract

Design the backend so the separately developed ML component can later be plugged in without redesigning the frontend.

At minimum, define an inference interface conceptually equivalent to:

### Request

```json
{
  "date": "YYYY-MM-DD",
  "latitude": 18.5,
  "longitude": 88.25
}
```

The exact request may evolve according to the ML implementation.

### Response

```json
{
  "date": "YYYY-MM-DD",
  "latitude": 18.5,
  "longitude": 88.25,
  "depths_m": [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000],
  "temperature_c": [],
  "model": {
    "name": "OceanEmbed",
    "version": "..."
  },
  "data_source": "...",
  "is_mock": false
}
```

The actual response schema must be documented and versioned.

For UI development, implement a mock inference provider that returns clearly labeled synthetic/demo data.

When the real ML model becomes available, replace the provider without changing the frontend contract.

---

## 6. Frontend Goal

Build a polished local-first scientific/oceanographic analysis dashboard.

It should feel like:

**a professional oceanographic research and analysis platform**

not:
- generic SaaS
- generic AI dashboard
- admin dashboard
- marketing landing page

The core user flow is:

```text
North Indian Ocean map
        ↓
Select date + location
        ↓
Run/request reconstruction
        ↓
Show 15-depth temperature profile
        ↓
Show Argo comparison when available
        ↓
Show Ocean Embedding visualization when available
        ↓
Show relevant metrics and provenance
```

---

## 7. Required Frontend Areas

Build reusable components for:

### Map
- North Indian Ocean view
- geographic boundaries/context
- selectable location
- selected point marker
- coordinate display

### Controls
- date selection
- latitude/longitude
- reconstruction trigger
- loading/error states

### Temperature Profile
- depth on vertical axis
- temperature on horizontal axis
- all 15 standard depths
- reconstructed profile
- clear units
- optional Argo overlay
- legend
- hover/tooltips

### Surface/Data Context
Support future visualization of:
- SST
- SSS
- SSH/SLA
- surface currents U/V
- surface winds U/V

Do not invent scientific values.

### Ocean Embedding
Provide a dedicated area for embedding visualization.

It should be designed so PCA/UMAP/t-SNE results from the ML pipeline can later be connected.

Do not fabricate scientific embedding interpretations.

### Metrics
Support:
- RMSE
- MAE
- bias
- correlation/R² where available
- depth-wise metrics

Only display actual supplied/measured values.

### Provenance
Clearly communicate:
- date
- location
- data/model source
- model version
- whether data is real or mock
- validation availability

---

## 8. Mock Data Policy

Mock data is allowed and encouraged for parallel UI development.

However:

- Clearly label it `DEMO`, `MOCK`, or `SYNTHETIC`.
- Never present mock values as scientific results.
- Keep mock data schemas identical to real API schemas.
- Make the mock provider replaceable.
- Do not hard-code fake validation claims.
- Do not show fake Argo agreement as if it were measured.

---

## 9. Frontend Design Reference

Primary design reference:

https://github.com/anthropics/skills/tree/main/skills/frontend-design

Read:

`skills/frontend-design/SKILL.md`

Use its principles for:
- intentional visual design
- typography
- hierarchy
- composition
- spacing
- interaction
- avoiding generic AI-generated interfaces

Do not clone the entire repository.

The design should be scientific, premium, restrained, precise, and data-focused.

Avoid:
- excessive gradients
- neon/glowing effects
- excessive glassmorphism
- repetitive card grids
- decorative elements that compete with data
- meaningless animations

---

## 10. Additional Frontend References

Introduce progressively, not all at once.

### UI/UX
https://github.com/nextlevelbuilder/ui-ux-pro-max-skill

Use for:
- information architecture
- user flows
- accessibility
- responsive behavior
- interaction patterns

### UI components
https://github.com/Jpisnice/shadcn-ui-mcp-server

Use for appropriate reusable UI components.

Do not let shadcn determine the overall visual identity.

### React engineering
https://github.com/vercel-labs/agent-skills

Use later for React performance and engineering review.

### Motion
https://github.com/freshtechbro/claudedesignskills

Use only when subtle interaction/transition animation genuinely improves the scientific workflow.

Do not add GSAP, Convex, React Native, authentication, cloud databases, or deployment infrastructure unless explicitly requested.

---

## 11. Technology

Preferred stack:

Frontend:
- React
- TypeScript
- Leaflet
- Plotly

Backend:
- FastAPI
- Python

ML:
- external/separately maintained PyTorch model

The prototype is **local-first**.

Do not introduce:
- Vercel deployment
- Convex
- cloud database
- authentication
- payment systems
- unnecessary infrastructure

---

## 12. ML/Frontend Separation

The ML developer will independently work on:
- data preprocessing
- model architecture
- training
- checkpoints
- GLORYS evaluation
- blind Argo validation
- embedding generation
- model experiments

The prototype developer should focus on:
- API contracts
- inference integration
- frontend
- visualization
- local application workflow

The two workstreams must meet through a clearly defined interface.

Do not duplicate ML logic in the frontend.

Do not place the ML training notebook inside the dashboard application.

---

## 13. Repository Structure

Use a clean structure such as:

```text
OceanEmbed/
├── AGENTS.md
├── OceanEmbed_Final_Technical_Plan.md
├── README.md
│
├── api/
│   ├── main.py
│   ├── routes/
│   ├── schemas/
│   └── services/
│
├── dashboard/
│   ├── src/
│   ├── public/
│   └── package.json
│
├── ml_interface/
│   ├── inference.py
│   └── README.md
│
├── mock_data/
│   └── ...
│
├── tests/
│
└── docs/
    └── api-contract.md
```

The ML developer may maintain their own separate repository/project structure if desired.

---

## 14. API Design

Use typed request/response schemas.

Separate:
- API routes
- validation schemas
- inference service
- mock inference service

Example conceptual architecture:

```text
/api/reconstruct
       │
       ↓
Request validation
       │
       ↓
Inference service
       │
       ├── Mock provider
       │
       └── Real OceanEmbed provider
       │
       ↓
Typed response
       │
       ↓
React dashboard
```

The frontend should not care whether the response came from mock data or the real model, except for displaying the appropriate status/provenance.

---

## 15. Development Order

### Phase 0 — Understand and plan

Before coding:
1. Read AGENTS.md.
2. Read OceanEmbed_Final_Technical_Plan.md.
3. Inspect frontend-design SKILL.md.
4. Inspect the current workspace.
5. Propose repository structure.
6. Propose frontend architecture.
7. Propose API contract.
8. Identify assumptions.
9. Wait for approval.

### Phase 1 — Foundation

Create:
- React + TypeScript application
- FastAPI application
- API schema
- mock inference provider
- basic frontend/backend connection
- project README
- tests

### Phase 2 — Scientific dashboard UI

Implement:
- map
- controls
- profile chart
- metrics area
- provenance
- loading/error/empty states

Use mock data.

### Phase 3 — Integration readiness

Implement:
- real inference provider interface
- checkpoint/model loading boundary
- embedding response schema
- Argo response schema
- model versioning/provenance

### Phase 4 — Real ML integration

When the ML developer supplies the model/API/checkpoint:
- integrate it
- do not rewrite the ML architecture
- verify tensor/input/output contracts
- test inference
- replace mock provider

### Phase 5 — Polish

Only after functionality works:
- responsive improvements
- accessibility
- performance
- subtle motion
- visual refinement

---

## 16. Agent Workflow

Before major implementation:

1. Inspect current files.
2. Read relevant documentation.
3. Check the technical plan.
4. State a concise implementation plan.
5. Implement incrementally.
6. Run tests/build checks.
7. Report actual results.
8. Stop before moving to the next major phase unless instructed.

Do not make large unrelated changes.

Do not silently install unnecessary technologies.

Do not claim a feature is working without testing it.

---

## 17. Scientific Integrity

Never fabricate:
- RMSE
- MAE
- bias
- R²/correlation
- Argo validation results
- model accuracy
- scientific citations
- dataset observations
- model performance

If a demo needs data before the ML model exists, use clearly labeled synthetic/mock data.

---

## 18. Definition of Done

The prototype should eventually allow a user to:

1. Launch the application locally.
2. View the North Indian Ocean.
3. Select date/location.
4. Request a reconstruction.
5. Receive a 15-depth temperature profile.
6. Visualize the profile clearly.
7. Overlay an Argo profile when a real validated observation is available.
8. View embedding information when supplied by the ML backend.
9. View measured metrics when available.
10. See clear provenance and mock/real status.

The prototype must remain usable even while the ML model is being developed separately.
