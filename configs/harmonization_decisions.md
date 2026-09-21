# OceanEmbed — Temporal Harmonization & Physical Variable Decisions

**Domain:** North Indian Ocean ($5^\circ\text{N}–30^\circ\text{N}, 45^\circ\text{E}–105^\circ\text{E}$)  
**Target Cadence:** Daily ($1\text{ sample / day}$)  
**Model Input Contract:** 14 channels (7 physical variables + 7 binary validity masks)  
**Model Target Contract:** 15 channels (15 standard depth levels: $0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000\text{ m}$)  

---

## 1. Variable Semantics & Physical Units

| Physical Variable | Source Product | Variable Name | Source Units | Target Units | Conversion / Formula | Semantics & Oceanographic Role |
|---|---|---|---|---|---|---|
| **SST** | OSTIA L4 | `analysed_sst` | Kelvin ($K$) | Celsius ($^\circ\text{C}$) | $T_{^\circ\text{C}} = T_K - 273.15$ | Foundation Sea Surface Temperature (free of diurnal warming). Primary indicator of surface heat content. |
| **SSS** | SMAP/SMOS L4 | `sos` | Practical Salinity ($10^{-3}$) | $\text{PSU}$ | Direct passthrough | Sea Surface Salinity. Governs upper-ocean stratification and barrier layer formation in the Bay of Bengal. |
| **SSH** | DUACS Multi-Mission | `sla` | meters ($m$) | meters ($m$) | Direct passthrough | **Sea Level Anomaly (SLA)**. Selected over Absolute Dynamic Topography (`adt`). Zero-centered dynamic anomaly reflecting baroclinic thermocline heave. |
| **Currents (U)** | OSCAR L4 OC | `u` | $m\cdot s^{-1}$ | $m\cdot s^{-1}$ | Direct passthrough | Zonal surface current (positive eastward). Combines geostrophic balance and wind-driven Ekman drift. |
| **Currents (V)** | OSCAR L4 OC | `v` | $m\cdot s^{-1}$ | $m\cdot s^{-1}$ | Direct passthrough | Meridional surface current (positive northward). Captures Somali current, East India Coastal Current, and mesoscale eddies. |
| **Winds (U)** | CCMP v3.1 | `uwnd` | $m\cdot s^{-1}$ | $m\cdot s^{-1}$ | 4-step daily mean | Zonal 10-meter wind speed. Primary driver of equatorial Kelvin waves and Ekman pumping. |
| **Winds (V)** | CCMP v3.1 | `vwnd` | $m\cdot s^{-1}$ | $m\cdot s^{-1}$ | 4-step daily mean | Meridional 10-meter wind speed. Driver of coastal upwelling along the Oman and Somali coasts during Southwest Monsoon. |
| **Target (thetao)** | GLORYS12V1 | `thetao` | $^\circ\text{C}$ | $^\circ\text{C}$ | Vertical interpolation | 3D Potential Temperature at 15 standard depths down to $1000\text{ m}$. Physical target for reconstruction. |

---

## 2. DUACS SSH: SLA Selection Over ADT

The DUACS multi-mission altimetry file provides both:
1. `sla` (Sea Level Anomaly): Range $\approx [-0.30\text{ m}, +0.39\text{ m}]$; Basin Mean $\approx 0.066\text{ m}$.
2. `adt` (Absolute Dynamic Topography): Range $\approx [0.45\text{ m}, 1.52\text{ m}]$; Basin Mean $\approx 0.848\text{ m}$.

### Decision & Justification:
- **Decision:** The OceanEmbed pipeline **strictly extracts and uses `sla`**, never `adt`.
- **Oceanographic Justification:**
  - `adt` is defined as $\text{ADT} = \text{MDT} + \text{SLA}$, where MDT is the Mean Dynamic Topography (geoid + time-mean circulation). In the North Indian Ocean, MDT is dominated by permanent large-scale geopotential topography ($\sim 0.8\text{–}1.2\text{ m}$ elevation relative to the geoid).
  - Using `adt` would force the neural network to memorize static geographic geoid offsets rather than learning dynamical subsurface baroclinic fluctuations.
  - `sla` directly correlates with thermocline displacement via the two-layer reduced gravity model ($\eta' \approx -\frac{\Delta \rho}{\rho_0} h'$). A positive SLA anomaly indicates a depression of the $20^\circ\text{C}$ isotherm (downwelling / anticyclonic eddy), while a negative SLA anomaly indicates thermocline shoaling (upwelling / cyclonic eddy).

---

## 3. CCMP Wind Temporal Harmonization

### Cadence of Source:
CCMP v3.1 is supplied as daily files containing 4 synoptic intervals:
- `00:00:00 UTC`
- `06:00:00 UTC`
- `12:00:00 UTC`
- `18:00:00 UTC`

### Harmonization Method:
- **Decision:** The pipeline computes the **daily arithmetic mean** across all 4 synoptic slices:
  $$\overline{u}_{\text{wind}}(\text{day}) = \frac{1}{4} \sum_{t \in \{00, 06, 12, 18\}} u_{\text{wind}}(t), \quad \overline{v}_{\text{wind}}(\text{day}) = \frac{1}{4} \sum_{t \in \{00, 06, 12, 18\}} v_{\text{wind}}(t)$$
- **Physical Justification:**
  1. **Momentum Flux Integration:** Subsurface ocean mixed-layer depth and thermocline response operate on inertial and sub-inertial timescales ($> 1\text{–}2\text{ days}$). Daily mean wind captures the net momentum flux and Ekman pumping velocity ($\mathbf{w}_E = \frac{1}{\rho_0 f} \nabla \times \boldsymbol{\tau}$) driving the mixed layer.
  2. **Diurnal Cycle Smoothing:** High-frequency diurnal sea breeze / land breeze oscillations along the Indian subcontinent coastline are integrated out, preventing aliasing into the daily reconstruction.
  3. **No Arbitrary Forward-Filling:** Each day's wind is derived strictly from that day's 4 synoptic observations. No cross-day forward-filling or backward-filling is introduced.

---

## 4. GLORYS Depth Interpolation & Shallow-Water Target Masking

### Target Depth Levels:
`[0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]` m (15 levels).

### Source Native Depths:
GLORYS12V1 has 36 vertical levels in our regional extraction:
- Level 0: $0.494\text{ m}$
- Level 33: $763.333\text{ m}$
- Level 34: $902.339\text{ m}$
- Level 35: $1062.440\text{ m}$

### Interpolation & Masking Rules:
1. **1000 m Bracketing:** Level 34 ($902.339\text{ m}$) and Level 35 ($1062.440\text{ m}$) strictly bracket the deepest target depth ($1000\text{ m}$). Linear interpolation is performed strictly between these two physical levels.
2. **Zero Below-Deepest Extrapolation:**
   - Interpolator is configured with `bounds_error=False, fill_value=np.nan`.
   - At no ocean column is temperature extrapolated below the deepest valid native level.
3. **Surface Extension:**
   - The topmost GLORYS level is at $0.494\text{ m}$ depth. Because this is within the upper $0.5\text{ m}$ mixed layer, it is safely extended to $0.0\text{ m}$ surface depth.
4. **Shallow-Water Bathymetric Masking:**
   - Over continental shelves, marginal seas (Persian Gulf, Gulf of Oman), and shallow coastal waters where water depth $H_{\text{bottom}} < 1000\text{ m}$, any target depth deeper than $H_{\text{bottom}}$ evaluates to `NaN`.
   - In `DataHarmonizer`, `target_mask` is set to `0.0` for all levels exceeding bathymetric depth, and the target value carries a finite `0.0` fill.
   - Analysis on January 2020 verifies that valid ocean points decrease monotonically from $11,802$ at $0\text{ m}$ to $8,911$ at $1000\text{ m}$, correctly masking $2,891$ shallow-water columns ($24.5\%$ of the basin).
