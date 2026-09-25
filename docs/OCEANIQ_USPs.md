# OceanIQ — Unique Selling Propositions (USPs) & Core Features

This document explains the key capabilities and unique features developed for the **OceanIQ** oceanographic reconstruction and analysis platform. It is written in simple, clear language so students, judges, and team members can quickly understand what OceanIQ does, why it matters, and how it works.

---

## 1. Vertical Subsurface Transect

### What it does
The Vertical Subsurface Transect feature allows a user to select two points across the North Indian Ocean map and draw a custom path. OceanIQ slices through the ocean along that line and displays a 2D vertical temperature cross-section from the surface down to 1000 meters depth. It shows how water temperature changes continuously with distance and depth.

### Why it is useful
In the real ocean, water temperature does not just vary vertically at one location; it forms fronts, eddies, and sloping layers across hundreds of kilometers. Instead of clicking one spot at a time, oceanographers and researchers can view an entire vertical curtain across basins like the Bay of Bengal or Arabian Sea to spot thermal boundaries and thermocline dips.

### How OceanIQ uses it
OceanIQ samples stations along the selected trajectory and extracts temperatures directly from the reconstructed 3D temperature field produced by the Phase-1 model. It overlays key physical boundaries including the 26°C isotherm (D26) and Mixed Layer Depth (MLD). No new ML model was added to create this cross-section.

---

## 2. Subsurface Reconstruction Departure

### What it does
Subsurface Reconstruction Departure shows the exact difference between the OceanIQ reconstructed temperature and the GLORYS12V1 numerical reanalysis reference temperature at any selected location or depth. It is calculated simply as:
`Reconstruction Departure = OceanIQ Reconstruction − GLORYS12V1 Reference`

### Why it is useful
Scientific AI systems should never operate as blind black boxes. By showing where OceanIQ runs slightly warmer (positive departure) or cooler (negative departure) compared to the GLORYS12V1 numerical ocean reanalysis reference dataset, users can immediately inspect reconstruction differences and assess agreement with the reference field.

### How OceanIQ uses it
OceanIQ queries the collocated GLORYS12V1 reference field for the identical date and coordinates, computes the difference across the 15 standard depths, and reports summary statistics including Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and mean bias. Crucially, OceanIQ clearly states that GLORYS12V1 is a numerical ocean reanalysis reference dataset, NOT direct physical ground truth.

---

## 3. Depth-Wise Model Skill

### What it does
Depth-Wise Model Skill shows how well the OceanIQ model performs across each of the 15 standard ocean depths: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, and 1000 meters. It displays standard statistical accuracy metrics—Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and bias—at each depth level.

### Why it is useful
Reconstructing subsurface ocean temperature is much harder in the thermocline (where temperatures drop abruptly) than in the uniform deep ocean or at the surface. This feature lets users see exactly where the model is strongest and where reconstruction error naturally increases, instead of hiding behind a single averaged number.

### How OceanIQ uses it
OceanIQ presents verified evaluation metrics derived from a basin-wide evaluation of 14,200 held-out test cells across the North Indian Ocean on 2019-01-01. These metrics evaluate architectural accuracy across depths over the whole basin: surface error is low (RMSE 0.71°C), error reaches an observed peak in the dynamic 50–150 m thermocline zone (~1.15°C), and then generally declines with depth, reaching 0.37°C at 700 m and 0.39°C at 1000 m.

---

## 4. TCHP — Tropical Cyclone Heat Potential

### What it does
Tropical Cyclone Heat Potential (TCHP) is a derived scientific indicator that measures the excess thermal heat stored in the upper ocean between the surface and the 26°C isotherm depth (D26). It integrates heat content only for water warmer than 26°C, which is the conventional 26°C threshold used in TCHP calculations.

### Why it is useful
Surface temperature alone can be misleading: a very thin layer of warm water can cool rapidly under strong storm winds. TCHP reveals how deep the heat reservoir actually goes, helping researchers understand how much thermal fuel is available in cyclone-prone areas such as the Bay of Bengal.

### How OceanIQ uses it
TCHP is calculated as a post-processing step directly from the OceanIQ reconstructed 15-depth temperature profile using standard oceanographic integration:
`TCHP = ρ × cp × integral of [T(z) − 26°C] from 0 to D26`
(using standard physical constants ρ = 1026 kg/m³ and cp = 3990 J/(kg·°C)). It is displayed prominently alongside D26 and MLD in physical units of kJ/cm². OceanIQ clearly states that TCHP is a physical heat-content indicator and does NOT predict cyclone tracks or guarantee rapid intensification.

---

## 5. Scientific PDF Report

### What it does
OceanIQ provides a one-click technical PDF report generator that exports a comprehensive, multi-page technical PDF briefing for any selected date and geographic coordinate. The report synthesizes all analysis views, charts, and metrics into a standardized document suitable for research, analysis and presentation.

### Why it is useful
Researchers, students, and technical teams need portable, reproducible documentation they can share, archive, or print. Instead of taking manual screenshots, OceanIQ formats the complete scientific analysis into an organized report with exact provenance and data-source information.

### How OceanIQ uses it
The PDF report is built dynamically from the active session data. It includes:
- **Reconstruction Summary:** Date, coordinates, and regional basin identification.
- **Key Physical Metrics:** D26, MLD, and TCHP estimates.
- **15-Depth Temperature Profile:** Clear vertical sounding chart with D26 and MLD markers.
- **15-Depth Discrete Thermal Matrix:** Depth-oriented color matrix showing discrete layer temperatures.
- **Vertical Subsurface Transect:** The 2D hydrographic cross-section whenever an active transect is selected.
- **Synoptic Surface Observations:** 7 surface physical variables and validity masks.
- **GLORYS12V1 Reconstruction Departure:** Local departure profile and depth-wise statistics.
- **Depth-Wise Model Skill:** Basin-wide RMSE, MAE, and bias across all 15 depths.
- **Independent In-Situ Argo Float Validation:** Collocated real float profile comparison when available (strictly avoiding synthetic float data).
- **128-D Latent Representation:** 2D PCA projection of the internal feature space against contextual reference points.
- **Methodology & Model Lineage:** Parameter counts, input tensor dimensions, depth configuration, and model lineage.
- **Scientific Disclaimers:** Explicit statements on physical boundaries and data sources.

---

## 6. 15-Depth Interactive Subsurface Profile

### What it does
This is the core visualization of OceanIQ. Given daily satellite surface observations at any point in the North Indian Ocean, OceanIQ reconstructs the subsurface water temperature down through 15 standardized depths: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, and 1000 meters.

### Why it is useful
Satellites can only see the "ocean skin" (the top millimeters of the sea). In-situ instruments like ships or mooring buoys are sparse and expensive. OceanIQ reconstructs the hidden vertical temperature profile below the surface so users can inspect the complete water column at daily synoptic resolution.

### How OceanIQ uses it
The interface plots depth downward on the vertical axis (from 0 m down to 1000 m) and temperature on the horizontal axis, following standard oceanographic scientific conventions. It automatically identifies and flags critical physical thresholds: the 26°C isotherm depth (D26) and the Mixed Layer Depth (MLD).

---

## 7. 128-D Ocean Embedding

### What it does
OceanEmbed compresses the 14-channel input tensor (7 physical surface variables plus 7 quality masks) into an internal 128-dimensional latent vector ($Z$). OceanIQ provides an interactive 2D Principal Component Analysis (PCA) projection of this embedding space to show where the current observation sits relative to contextual reference points across the basin.

### Why it is useful
Modern deep learning works by creating rich internal representations of complex data. Visualizing the latent space gives researchers a window into how the neural network groups different surface observation patterns, building intuition about how surface forcing relates to subsurface structure.

### How OceanIQ uses it
OceanIQ extracts the 128-dimensional embedding from the Phase-1 model and projects it onto two leading principal components. It displays the query reconstruction alongside contextual reference points across the North Indian Ocean. OceanIQ maintains scientific integrity by clearly labeling background clusters as contextual reference points without claiming that PCA clusters prove discovered water-mass classifications or physical causality.

---

## 8. Scientific & Validation Transparency

### What it does
Scientific and Validation Transparency is a built-in design principle that ensures OceanIQ never misleads the user about what is an AI estimate versus an observed reality. Every data source, reference dataset, and validation comparison is explicitly and accurately labeled throughout the user interface and reports.

### Why it is useful
Scientific AI tools must be trustworthy. Presenting synthetic numbers as real measurements, or calling numerical models "ground truth," damages scientific integrity and leads to incorrect research conclusions. Clear labeling ensures users understand the exact scope and limitations of the tool.

### How OceanIQ uses it
1. **GLORYS12V1 is Labeled as a Reanalysis Reference:** It is always identified as a numerical reanalysis model reference, never as physical "ground truth."
2. **Argo Float Profiles are Labeled as Independent In-Situ Validation:** Real Argo float data from the INCOIS Live Access Server is used strictly for independent blind validation.
3. **No Synthetic Argo Observations:** If no real Argo float was active within 50 km of the query location, OceanIQ honestly displays that no float was collocated rather than creating fake validation numbers.
4. **Estimates are Explicitly Identified:** All reconstructed temperatures and derived quantities (D26, MLD, TCHP) are clearly presented as model estimates.
5. **Clear Scientific Disclaimers:** The platform explicitly states that OceanIQ complements physical observations and numerical models rather than replacing ships, moorings, or Argo floats.

---

# Overall OceanIQ USP

OceanIQ delivers an end-to-end workflow for reconstructing, exploring, evaluating, and reporting subsurface ocean temperature:

```text
Surface Observations
       ↓
OceanEmbed Phase-1 Model
       ↓
15-Depth Subsurface Reconstruction (0 m – 1000 m)
       ↓
Interactive Exploration (Vertical Profiles & Custom Transects)
       ↓
Validation & Departure (GLORYS12V1 Reference & In-Situ Argo Floats)
       ↓
Model Skill Diagnostics (Basin-Wide Depth-Wise RMSE, MAE, Bias)
       ↓
Derived Physical Indicators (D26 Isotherm, MLD, and TCHP)
       ↓
Standardized Scientific PDF Reporting
```

This integrated pipeline allows oceanographers, meteorologists, and students to quickly investigate the hidden thermal state of the North Indian Ocean, evaluate reconstruction accuracy with complete scientific honesty, and export presentation-ready reports in a matter of seconds.

---

# Important Project Constraint

- **Frozen Model Architecture:** All these USPs and features were built entirely around the existing, frozen Phase-1 OceanEmbed deep-learning model.
- **No New ML Models:** No new neural networks, classifiers, or predictors were trained or introduced.
- **Unmodified Checkpoint:** The frozen Phase-1 checkpoint weights and architecture remain 100% untouched.
- **Scientific Post-Processing & Integration:** All newly added capabilities utilize existing model outputs, verified held-out evaluation datasets, real in-situ catalogs, and established, scientifically defined post-processing calculations.
