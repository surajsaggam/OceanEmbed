# OceanEmbed — ML Workspace

Deep learning framework for subsurface ocean temperature reconstruction from surface observations over the North Indian Ocean.

**Scientific source of truth:** `OceanEmbed_Final_Technical_Plan.md`  
**Agent procedural rules:** `AGENTS_ML.md`

---

## Architecture

Single-snapshot, single-resolution model that predicts a 15-depth temperature profile from 7 surface variables:

```
Input:  [B, 14, H, W]   (7 physical vars + 7 validity masks)
Output: {
  "temperature": [B, 15, H, W]   ← subsurface temperature at 15 depths
  "embedding":   [B, 128, H, W]  ← Ocean Embedding (first-class output)
}
```

**Domain:** North Indian Ocean — 5–30°N, 45–105°E, 0.25° resolution  
**Depths (m):** 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000

---

## Setup

### 1. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies (staged — see implementation plan for rationale)

```powershell
# Stage 1: Core scientific stack
pip install numpy scipy pandas xarray zarr netcdf4 h5netcdf dask

# Stage 2: PyTorch (Blackwell / RTX 5060 — check official matrix for current stable build)
# https://pytorch.org/get-started/locally/
# Expected: torch 2.7+ with CUDA 12.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Stage 3: Data access
pip install copernicusmarine earthaccess requests

# Stage 4: Visualisation (cartopy may require conda on Windows)
pip install matplotlib cmocean seaborn
# pip install cartopy   ← attempt; use conda if this fails

# Stage 5: Analysis and notebooks
pip install scikit-learn pyyaml tqdm jupyterlab ipywidgets

# Stage 6: Testing
pip install pytest pytest-cov
```

### 3. Editable install (required for identical imports in Jupyter + PyCharm)

```powershell
pip install -e .
```

### 4. Verify GPU

```powershell
python scripts/verify_gpu.py
# Results saved to environment/env_verification.json
```

### 5. Run tests

```powershell
pytest tests/ -v --tb=short
```

---

## Development Workflow

This workspace supports two complementary modes against the same `.py` modules:

| Mode | Tool | Purpose |
|---|---|---|
| Interactive | JupyterLab | Data exploration, training, evaluation, embedding visualisation |
| Project IDE | PyCharm | Module development, static analysis, running scripts and tests |

**Rule:** All reusable implementation lives in `.py` modules. Notebooks are a thin import-and-call layer.

All notebooks start with:
```python
%load_ext autoreload
%autoreload 2
from utils.config import load_config
```

---

## Module Structure

| Package | Purpose |
|---|---|
| `pipeline/` | QC, regridding, alignment, masking, normalisation, Dataset |
| `models/` | Multi-scale CNN, pointwise MLP, fusion/embedding, attention decoder |
| `baselines/` | Climatology and Ridge regression |
| `evaluation/` | Metrics (RMSE, MAE, bias, Pearson r, R²), GLORYS eval, Argo eval, ablations |
| `inference/` | `OceanEmbedPredictor` — clean Python interface for integration |
| `utils/` | `load_config()` and shared utilities |

---

## Argo Blind Validation — Hard Rule

Argo data is **never used** for training, validation, early stopping, hyperparameter tuning, or model selection.  
The blind evaluation runs **exactly once**, after all model choices are locked.  
The guard is enforced in code: `evaluation/argo_eval.py` asserts `config.argo_blind_locked == True`.

---

## Git Discipline

- Source code, configs, tests, `requirements.txt`, `environment/env_verification.json` → **committed**
- `data/`, `checkpoints/`, `results/` → **gitignored** (large or derived)
