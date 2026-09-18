# OceanEmbed — Horizontal Regridding Decisions

**Target Domain:** North Indian Ocean (NIO)  
**Bounding Box:** $5.0^\circ\text{N}$ to $30.0^\circ\text{N}$, $45.0^\circ\text{E}$ to $105.0^\circ\text{E}$  
**Resolution:** $0.25^\circ \times 0.25^\circ$  
**Grid Dimensions:** $H = 101$ (latitude), $W = 241$ (longitude)  
**Total Spatial Grid Points:** $24,341$  

---

## 1. Common Target Grid Definition

The North Indian Ocean domain is discretized into a regular rectilinear grid with identical coordinate arrays across all 7 surface physical variables and 15 subsurface depth levels:

- **Latitude array:**
  $$\text{lat} = [5.00, 5.25, 5.50, \dots, 29.75, 30.00]^\circ\text{N} \quad (N_{\text{lat}} = 101)$$
  *Strictly monotonically increasing ($+0.25^\circ$ step).*
- **Longitude array:**
  $$\text{lon} = [45.00, 45.25, 45.50, \dots, 104.75, 105.00]^\circ\text{E} \quad (N_{\text{lon}} = 241)$$
  *Strictly monotonically increasing ($+0.25^\circ$ step).*

All data arrays entering the model or baseline algorithms are aligned with these exact coordinate arrays. **No arrays are ever aligned by shape or raw array index alone.**

---

## 2. Source Products & Regridding Methods

| Variable | Raw Product | Native Grid Spacing | Native Dimensions over NIO | Interpolation Method | Regridding Notes |
|---|---|---|---|---|---|
| **SST** | OSTIA L4 | $0.05^\circ \times 0.05^\circ$ | $501 \times 1201$ | Bilinear (`scipy.interpolate.RegularGridInterpolator`) | Downsampling from high-resolution foundation SST. Area-mean preserved. |
| **SSS** | SMAP/SMOS L4 Blended | $0.125^\circ \times 0.125^\circ$ | $201 \times 481$ | Bilinear (`RegularGridInterpolator`) | Downsampling from blended microwave salinity. |
| **SSH (SLA)** | DUACS Multi-Mission | $0.25^\circ \times 0.25^\circ$ | $101 \times 241$ | Bilinear / Exact Coordinate Alignment | Native grid already $0.25^\circ$; interpolated to align exact half-grid cell centers. |
| **Currents (U, V)** | OSCAR L4 OC | $0.25^\circ \times 0.25^\circ$ | $101 \times 241$ (sub-region) | Bilinear / Exact Coordinate Alignment | Non-standard native dimension order `('time', 'longitude', 'latitude')` transposed to `('latitude', 'longitude')`; coordinates extracted from `ds.lat` and `ds.lon`. |
| **Winds (U, V)** | CCMP v3.1 | $0.25^\circ \times 0.25^\circ$ | $101 \times 241$ (sub-region) | Bilinear / Exact Coordinate Alignment | 4 synoptic times averaged first, then regridded to target coordinates. |
| **Target (thetao)** | GLORYS12V1 Reanalysis | $0.0833^\circ \times 0.0833^\circ$ | $301 \times 721$ (at 36 depth levels) | Bilinear 2D per horizontal slice + 1D vertical interpolation | Horizontal regridding applied to each native vertical level prior to column-wise vertical interpolation. |

---

## 3. Boundary & Land Handling Rules

1. **Strict Coordinate Bounding:**
   - Interpolation query points lie strictly inside the bounding box of the source datasets.
   - Extrapolation outside source coordinates is strictly prohibited (`bounds_error=False, fill_value=np.nan`).
2. **Land Preservation:**
   - Land boundaries and coastal masks from source datasets are preserved by assigning `NaN` to unobserved/land pixels during interpolation.
   - During subsequent mask processing (`pipeline/mask.py` and `pipeline/qc.py`), land pixels receive their deterministic regional climatological fill, while the associated channel validity mask is set to `0.0`.
3. **No Array Inversion:**
   - Source datasets with descending latitude arrays (e.g. $30^\circ\text{N} \to 5^\circ\text{N}$) are checked and sorted to ascending order prior to interpolation to ensure strictly monotonic grid inputs for interpolation routines.
