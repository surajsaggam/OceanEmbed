# OceanEmbed UI Implementation Inventory & Design Architecture

> **Status:** Research & Planning Inventory (Pre-Implementation)  
> **Source Documents:** `OceanEmbed_Final_Technical_Plan.md`, `AGENTS.md`, `design-system/DESIGN.md`, `design-system/apple-design/SKILL.md`  
> **Target Application:** OceanEmbed Subsurface Ocean Thermal Reconstruction Dashboard (North Indian Ocean)

---

## 1. Design Principles

OceanEmbed is a **scientific oceanographic analysis platform**, not a consumer SaaS dashboard, an admin template, or an AI showcase.

1. **Substance Over Spectacle:** The UI exists solely to illuminate ocean physics and subsurface thermodynamics. Every visual element, label, and control must serve scientific understanding.
2. **Direct Manipulation & Immediacy (Apple Design):** Input response must be instantaneous on pointer-down. Scrubbers, coordinate pins, and parameter changes should track continuously. Latency must be zeroed out on interactions that do not require network calls.
3. **Editorial Precision & Typographic Hierarchy (Stripe System):** Clear typographic contrast, restrained font weights (favoring light display weights and robust regular body), disciplined tracking (negative on headings, positive on micro-caps), and mandatory tabular figures (`tnum`) for physical coordinates, depths, and thermal readings.
4. **Physical Restraint:** No gratuitous neon glow, no pseudo-holographic cards, no decorative glassmorphism that obscures data, and no generic dark-mode "cockpit" clutter. Surfaces should be crisp, neutral, and high-contrast, letting bathymetry and thermal curves carry color.
5. **Radical Scientific Provenance:** Ground truth must never be blurred with synthetic or estimated data. Reconstructions, synthetic demo floats, and model metadata must always be unambiguously labeled.
6. **Interruptible, Purposeful Motion:** Animation is used only where it conveys physical continuity (e.g., station transitions, depth slicing, smooth spring-based layout toggles). No looping ambient animations.

---

## 2. Typography Direction

The typographic foundation pairs **Geist Variable** (standard clean grotesque loaded via `@fontsource-variable/geist`) with high-legibility monospace numerals for scientific metrics.

| Role | Size | Weight | Tracking | Line Height | Usage / Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Display Title** | 28–32px | 400 (Regular) / 300 (Light) | `-0.02em` (`-0.6px`) | 1.15 | Main view headers, station headline |
| **Section Heading** | 18–20px | 500 (Medium) | `-0.01em` (`-0.2px`) | 1.25 | Primary stage headers, panel titles |
| **Subheading** | 14–15px | 500 (Medium) | `0` | 1.35 | Diagnostic groups, parameter groups |
| **Body Regular** | 13–14px | 400 (Regular) | `0` | 1.45 | Descriptive text, scientific notes, provenance |
| **Scientific Numerics** | 12–14px | 500/600 | `-0.01em` + `tnum` | 1.2 | Temperatures (°C), depths (m), SLA, currents |
| **Coordinate / Lat-Lon**| 11–12px | 500 (Mono) | `0` + `tnum` | 1.0 | Station coordinates (`18.50°N, 88.25°E`) |
| **Micro Caption / Eyebrow**| 10–11px | 600 (Semibold) | `+0.04em` (Uppercase)| 1.2 | Channel badges, source labels, dates |

### Key Typographic Rules
- **Mandatory Tabular Numerics (`tnum`):** All numbers (temperature in °C, depth in meters, velocity in m/s, coordinates in degrees, timestamps) must use `font-feature-settings: "tnum" 1` to prevent jitter during updates.
- **Controlled Optical Sizing:** Headings scale down tracking as font size increases; captions and micro-labels use positive tracking for legibility.
- **Monospace Isolation:** Monospace font (`JetBrains Mono` or system mono) is restricted to numerical values, coordinates, and raw identifiers. General UI copy must remain in sans-serif.

---

## 3. Color Strategy

The palette is derived from **marine physical oceanography** and **scientific charting standards**, rejecting arbitrary neon styling.

### 3.1 Base Surfaces & Chrome
- **Canvas / Background:** Deep oceanic slate-navy (`#060c16` to `#09121f`), calibrated for high visual dynamic range with colorful thermal traces.
- **Card / Surface Layers:** Neutral, low-reflectance charcoal-slate (`#0e1826` to `#132032`) providing separation via subtle luminance steps rather than harsh borders.
- **Hairlines & Dividers:** Crisp 1px borders (`#1e2f47` / `rgba(255, 255, 255, 0.08)`), following Stripe's hairline precision.
- **Text Tiers:**
  - High-emphasis: `#f8fafc` (slate-50)
  - Medium-emphasis: `#94a3b8` (slate-400)
  - Subdued/Captions: `#64748b` (slate-500)

### 3.2 Semantic Oceanographic Colors
- **Reconstructed Profile (OceanEmbed):** Vivid Cyan-Teal (`#00d4b8` / `#06b6d4`) — primary reconstruction trace.
- **Independent Validation (Argo Float):** Coral Rose (`#f43f5e` / `#fb7185`) — dashed in-situ observation trace.
- **Isotherm D26 / Thermocline:** Warm Solar Amber (`#f59e0b`) — cyclone heat potential threshold line.
- **Mixed Layer Depth (MLD):** Light Cerulean (`#38bdf8`) — physical mixed layer marker.
- **Dynamic Topography (SSH/SLA):** Royal Violet (`#8b5cf6`) — altimetry anomaly indicator.
- **Halocline / Salinity (SSS):** Deep Aqua (`#0ea5e9`) — salinity state.
- **Status / Verification:**
  - Operational / Live: Emerald Green (`#10b981`)
  - Synthetic / Mock Demo: Amber Warning (`#f59e0b`)
  - Out of Domain / Error: Crimson (`#ef4444`)

---

## 4. Spacing & Radius Strategy

### 4.1 Spacing Scale (8pt Baseline with 4pt Sub-grid)
- `4px` (`xs`): Tight chip padding, input inner icons.
- `8px` (`sm`): Element gaps, button padding vertical.
- `12px` (`md`): Form field padding, component margins.
- `16px` (`lg`): Card inner padding, section separations.
- `24px` (`xl`): Major module gutters, stage margins.
- `32px` (`xxl`): Workspace section bounds.

### 4.2 Border Radius Scale
- `4px` (`rounded-xs`): Tags, micro-badges, tabular chips.
- `6px` (`rounded-sm`): Inputs, icon buttons, dropdown items.
- `8px` (`rounded-md`): Standard action buttons, small dialogs.
- `12px` (`rounded-lg`): Primary cards, chart containers, map viewport.
- `9999px` (`rounded-pill`): Filter chips, mode toggles, status pills.

---

## 5. Component Hierarchy

The interface will be organized into four clear tiers of visual prominence:

```text
┌─────────────────────────────────────────────────────────────┐
│ 1. Header Utility Strip (Global Context, Health, Audit)     │
├─────────────────────────────────────────────────────────────┤
│ 2. Spatial & Temporal Anchor (Targeting Bar / Date / NIO)   │
├──────────────────────────────┬──────────────────────────────┤
│ 3. Primary Analytical Stage  │ 4. Secondary Diagnostic Deck │
│    • 15-Depth Profile Chart  │    • Basin Bathymetry Map    │
│    • Depth Metric Matrix     │    • Satellite Surface Drivers│
│    • Argo Validation Overlay │    • 128-D Latent Manifold   │
│                              │    • Physical Provenance     │
└──────────────────────────────┴──────────────────────────────┘
```

- **Tier 1 (Anchor):** Date, coordinates, and execute trigger must be globally visible and accessible via keyboard shortcuts (`Cmd/Ctrl+Enter`).
- **Tier 2 (Hero Analytics):** The 15-depth vertical temperature sounding is the centerpiece of the application. It receives primary viewport priority and highest resolution.
- **Tier 3 (Context & Drivers):** Surface boundary observations (SST, SSS, SSH, currents, winds) contextualize the neural network inputs.
- **Tier 4 (Audit & Lineage):** Model version, inference latency, GLORYS training bounds, and synthetic/real provenance remain visible but unobtrusive.

---

## 6. Navigation Strategy

- **Single-Screen Deep Analysis Workstation:** OceanEmbed is an exploratory scientific instrument, not a multi-page marketing site. The core workflow happens without full page refreshes.
- **Mode Switching:** Clean segmented tabs (Profile vs. Table vs. Dual View) using spring-animated indicators.
- **Flexible Inspector Layout:** Collapsible side drawer / diagnostic deck allowing the primary profile to expand to 100% width for detailed examination of deep thermoclines.
- **Direct Preset Teleportation:** Dropdown/drawer for pre-calibrated oceanographic regimes (Bay of Bengal Freshwater Plume, Somali Upwelling, Arabian Sea High-Salinity, Equatorial Warm Pool).

---

## 7. Responsive Strategy

| Breakpoint | Width | Layout Adaptation |
| :--- | :--- | :--- |
| **Ultra-Wide** | ≥ 1600px | Side-by-side: 65% Primary Sounding Stage + 35% Diagnostic Deck with multi-metric grids. |
| **Desktop** | 1200–1599px | Side-by-side with tabbed diagnostic views; chart height 480px minimum. |
| **Laptop / Tablet Landscape** | 900–1199px | Stacked or collapsible inspector; sounding stage full-width, diagnostic deck accessible via sliding sheet. |
| **Mobile / Compact** | < 900px | Single-column stacked; sticky compact targeting bar; profile chart scrolls smoothly; touch targets ≥ 44px. |

---

## 8. Motion & Interaction Strategy (Apple Design Principles)

1. **Pointer-Down Responsiveness:** All clickable controls, chips, and preset items must trigger visual feedback immediately on pointer-down (scale `0.98`, subtle highlight), not on release.
2. **Critically Damped Springs:** Use Motion (`motion/react`) with critically damped springs (`damping: 1.0`, `duration: 0.35s`, `bounce: 0`) for layout transitions, tab sliders, and drawer sliding. Never use bouncy or cartoonish easing.
3. **Interruptible Drawer / Sheet:** If the user collapses or expands the diagnostic inspector, the animation must be interruptible mid-flight without locking input.
4. **Data Continuity:** When coordinates change, chart transitions must smoothly re-plot rather than flashing white or collapsing to zero.
5. **Respect `prefers-reduced-motion`:** Fallback instantly to simple opacity fades or zero-duration transitions when reduced motion is requested.

---

## 9. Visualization-to-Library Mapping

| Visualization Need | Recommended Library | Rationale | What It Must NOT Do |
| :--- | :--- | :--- | :--- |
| **North Indian Ocean Basin Map** | **Leaflet** | Direct geographic coordinate picking, 0.25° bounding rectangle, custom bathymetric tile layer (ESRI Ocean Base), crisp marker placement. | No heavy 3D globe animations or unprojected distortions. |
| **15-Depth Subsurface Temperature Profile** | **Plotly** | Native reversed Y-axis (`autorange: 'reversed'`, 0 to 1000m), continuous spline interpolation, multi-trace overlays (Recon vs. Argo), horizontal isotherm markers (D26), scientific hover tooltips. | No financial candlesticks or commercial market chart templates. |
| **Observed vs. Reconstructed Deltas (Table/Matrix)** | **shadcn / React Table** | Clean tabular figure display, depth-wise temperature differences (`ΔT = Recon - Argo`), sorting by depth. | No heavy animated chart wrappers where a table is clearer. |
| **128-D Latent Embedding Manifold (PCA/UMAP)** | **Apache ECharts** | Canvas-based rendering capable of handling hundreds of regime scatter points at 60fps with real-time target station crosshairs and regime highlighting. | No slow SVG rendering or decorative 3D particle clouds. |
| **Surface Variable Telemetry (SST, SSS, SSH, Currents, Wind)** | **shadcn Cards + Lucide** | Direct numerical cards with vector magnitudes, direction angles, and data source badges. | No faux speedometer dials or uncalibrated radial gauges. |

---

## 10. shadcn/ui Component Allocation

The following core shadcn/ui components will serve as the structural primitives:

- `Button`: Primary reconstruction action, preset buttons, icon toggles.
- `Dialog`: Scientific architecture inspection, technical plan reference, data product lineage.
- `Popover`: Provenance & audit popover, preset station selector, coordinate fine-tuning.
- `Tabs`: Segmented view mode switches (Chart / Split / Table), diagnostic deck tabs (Basin / Drivers / Latent / Validation).
- `Badge`: Data provenance indicators (`DEMO / MOCK` vs `LIVE INFERENCE`), depth count pills, channel badges.
- `Separator`: Clean hairline dividers between metric blocks.
- `Tooltip`: Oceanographic glossary terms (D26, MLD, TCHP, SLA, OSTIA, SMAP).
- `Slider` *(to be added if needed)*: Interactive depth slicing or threshold adjustment.

---

## 11. Componentry / 21st.dev Component Candidates

Selected high-craft components from Componentry / 21st.dev to evaluate for specific scientific workflows:
- **Interactive Coordinate Stepper / Input:** Precise 0.25° increment buttons with keyboard arrow support.
- **Glass / Blurred Command Strip:** Crisp translucent floating command strip for date and station selection.
- **Smooth Segmented Pill Switcher:** Polished spring-based active tab indicator.

---

## 12. HeroUI Evaluation

- **Status:** **Secondary / Minimal Usage.**
- **Verdict:** Do NOT introduce HeroUI as a primary styling system. Having two conflicting CSS frameworks (shadcn tailwind vs HeroUI components) causes bundle bloat and class conflicts.
- **Allowed Usage:** Only if a specialized headless component (such as an advanced accessible date-range calendar or complex slider) is required and not cleanly provided by shadcn. Otherwise, default to shadcn.

---

## 13. Scientific Interaction Patterns

1. **Interactive Station Targeting:**
   - Click anywhere in the North Indian Ocean basin on the Leaflet map -> automatically clamps to 5°N–30°N, 45°E–105°E -> snaps to 0.25° grid -> updates coordinate fields -> triggers reconstruction.
2. **Keyboard Accelerator:**
   - `Cmd + Enter` or `Ctrl + Enter` triggers profile reconstruction from any field.
3. **Direct Sounding Inspection:**
   - Hovering over any point on the Plotly temperature curve reveals depth (m), reconstructed temperature (°C), and delta vs collocated Argo float if present.
4. **Isotherm Tracking:**
   - Visual indicator for the 26°C isotherm (D26), indicating the depth in meters as a proxy for Tropical Cyclone Heat Potential.
5. **Manifold Cross-Referencing:**
   - Selecting a station updates the active star in the 128-D latent manifold, highlighting which oceanographic regime the surface conditions project into.

---

## 14. Accessibility Requirements (WCAG 2.1 AA)

- **Color Contrast:** All text must maintain a minimum contrast ratio of 4.5:1 against surfaces; micro-text and labels must meet 3:1.
- **Color-Independent Data Visualization:** Lines must be differentiated by markers and dashes (e.g., solid teal line for reconstruction, dashed coral line with diamond markers for Argo float), not color alone.
- **Full Keyboard Operability:** All inputs, tabs, popovers, and triggers must be reachable and operable via `Tab`, `Arrow` keys, `Enter`, and `Escape`.
- **Screen Reader Semantics:** Table elements must use appropriate `role="table"`, `<th scope="col">`, and descriptive `aria-label` attributes.
- **Focus Indicators:** Unambiguous, high-contrast focus rings (`focus-visible:ring-2 focus-visible:ring-cyan-500`).

---

## 15. Explicit "DO NOT" Rules (Anti-AI-Slop Guardrails)

1. **DO NOT** use generic AI-dashboard tropes: no pulsating purple glow, no floating neon particles, no gratuitous dark cockpit borders.
2. **DO NOT** use TradingView or crypto trading aesthetic: no candlestick charts, no buy/sell telemetry, no ticker tapes.
3. **DO NOT** fake scientific accuracy: never display fabricated RMSE or MAE as measured truth; synthetic demo data must be prominently flagged as `DEMO / MOCK`.
4. **DO NOT** make un-interruptible animations: no fixed 2-second CSS transitions on panel toggles.
5. **DO NOT** mix ad-hoc inline color styles: all colors must adhere strictly to the oceanographic semantic palette.
6. **DO NOT** bury key scientific metrics under multiple nested clicks: depth, temperature, coordinates, and latency must be immediately readable.
7. **DO NOT** recreate the deleted UI: do not replicate the old 3-column fixed workstation layout or old header command bar.

---

## 16. Proposed OceanEmbed Information Architecture

The proposed architecture organizes the user's cognitive workflow into three logical layers:

### Layer A: Context & Station Specification (Where & When)
- Geographic Basin View (Leaflet map focused on 5°–30°N, 45°–105°E)
- Temporal Selector (Observation date within 2015–2024 archive)
- Coordinate Targeting (Latitude / Longitude with 0.25° grid snap)
- Oceanographic Regime Quick Presets (Bay of Bengal, Arabian Sea, Somali Upwelling, Equatorial Warm Pool)

### Layer B: Core Reconstruction & Thermal Sounding (What is Happening Subsurface)
- Primary Sounding Stage: Plotly vertical temperature profile (0–1000m across all 15 authoritative depths)
- Collocated In-Situ Argo Comparison: Independent validation overlay when floats exist in spatial/temporal neighborhood
- Diagnostic Oceanographic Indicators: D26 Isotherm Depth (m), Mixed Layer Depth (MLD)
- Quantitative Residuals: Depth-wise temperature table (`Recon °C`, `Argo °C`, `ΔT °C`)

### Layer C: Boundary Drivers, Latent Manifold & Provenance (Why & How)
- Multi-Source Satellite Boundary Drivers: 7 surface channels (OSTIA SST, SMAP SSS, DUACS SSH, OSCAR currents, ASCAT winds)
- 128-D Latent Manifold Space: High-performance ECharts projection of the neural bottleneck vector $Z \in \mathbb{R}^{128}$
- Technical Provenance & Scientific Audit: Model version, active inference provider, latency in ms, GLORYS baseline notice
