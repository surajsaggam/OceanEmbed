# PS 26066 — OceanEmbed
## Final Technical Plan (Merged, Corrected, Implementation-Ready)
### Satellite-Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Observations

**Ministry:** Ministry of Earth Sciences (MoES) · **Agency:** INCOIS · **Theme:** Disaster Management / Oceanography · **Domain:** North Indian Ocean (5°N–30°N, 45°E–105°E)

---

## How this document was produced

You gave me two prior drafts — an *Improved Plan* (disciplined, cautious, no hard numeric claims) and a *Unified Plan* (more ambitious, adds multi-scale CNN + temporal deltas + a 4-term physics loss, but bakes in several numbers — exact RMSE targets, exact VRAM/training-time figures, deterministic "3×3=coastal / 5×5=eddy" framing — that aren't yet earned by any experiment). Both are scientifically sound in structure. This document is **not** a 50/50 average of the two; it's a single opinionated recommendation: take the Unified Plan's architecture and roadmap almost wholesale (it's genuinely stronger), but strip out every claim that sounds like a measured result before you've measured it, and reorder the build so a defensible, demo-able Phase-1 model exists before any of the "impressive" extras are attempted. That reordering — not the architecture — is the single most important decision in this document, because in a hackathon the team that fails is almost always the one that built the ambitious version first and never got a validated baseline running.

---

## 1. Problem Framing

**OceanEmbed** reconstructs the depth-wise subsurface temperature structure of the North Indian Ocean using only daily surface satellite observations, at **15 standard depths**:

> 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m

**Core idea (use this exact framing with judges):**
> Satellites directly observe only the top ~1 m of the ocean. OceanEmbed learns the statistical/physical relationship between what satellites *can* see (temperature, salinity, height, currents, winds) and what they *can't* (the temperature profile down to 1000 m), by training against a physical ocean reanalysis and validating independently against real Argo float profiles.

Always position this as an **estimation/reconstruction framework that complements sparse in-situ observations** — never as a replacement for Argo floats, ships, or moorings, and never as "measuring" the deep ocean. This single framing choice does more for scientific credibility with judges than any architectural detail.

**Why it matters (keep this short in the pitch — one line per bullet, don't over-promise causal claims):**
- Tropical Cyclone Heat Potential (TCHP) and D26 (depth of the 26°C isotherm) relate to cyclone rapid intensification in the Bay of Bengal / Arabian Sea.
- Subsurface heat content matters for monsoon coupling, marine heatwaves invisible at the surface, and barrier-layer effects from Ganges–Brahmaputra freshwater.
- Argo floats are sparse (~one profile per ~3° per ~10 days in this basin); OceanEmbed offers a **continuous daily 0.25° gridded field** to fill the gaps between them — it does not replace them.

---

## 2. Inputs and Datasets

Seven surface variables, all daily, all with credible public sources:

| Variable | Product | Native resolution | Source / DOI |
|---|---|---|---|
| SST | OSTIA | 0.05° | CMEMS/UK Met Office, `10.48670/moi-00168` |
| SSS | SMAP/SMOS L4 blended | 0.125° | NASA JPL/CMEMS, `10.48670/moi-00051` |
| SSH/SLA | DUACS multi-mission altimetry | 0.25° | CMEMS, `10.48670/moi-00145` |
| U, V currents | OSCAR L4 OC | 0.25° | PO.DAAC, `OSCAR_L4_OC_FINAL_V2.0` |
| Wind U, V | CCMP v3.1 (or ASCAT-C) | 0.25° | PO.DAAC/EUMETSAT |

**Training target:** GLORYS12V1 reanalysis, native 0.083°/50 levels → resampled to 0.25° and interpolated to the 15 standard depths. This is a data-assimilative model output, **not ground truth** — say this explicitly to judges; it's why independent Argo validation is non-negotiable, not optional polish.

**Independent validation:** Gridded/QC'd Argo profiles via INCOIS LAS. **Hard rule, stated plainly:** Argo data is never seen during training, hyperparameter tuning, or early stopping. It is touched exactly once, at final blind evaluation.

**Practical fallback (not in either original draft, but you'll need it):** Before committing to the full 2015–2024 span, do a dry run of every download pipeline on a **1-month subset** first. Satellite data portals (CMEMS especially) have registration delays, rate limits, and intermittent outages that are the single most common cause of lost hackathon time. If any one source is unavailable close to the deadline, have a documented fallback (e.g., drop that variable and note it as a limitation, or substitute a reanalysis-derived proxy) rather than blocking on it.

---

## 3. Data Harmonization Pipeline

```
Raw multi-source data
   → Quality control (physical range checks, e.g. SST in [-2, 36]°C, SSS in [0, 45] PSU)
   → Regrid to common 0.25° × 0.25° grid (xESMF, conservative/bilinear)
   → Daily timestamp alignment (fix to one UTC reference hour, e.g. 12:00 UTC)
   → Static land mask (e.g. GEBCO bathymetry)
   → Missing-data handling: fill with regional climatology + attach a validity mask
   → Training-only z-score normalization
   → Model-ready tensor
```

**Missing data — do this, not naive imputation:** cloud gaps in SST/SSS are real and frequent. Fill missing pixels with the regional climatological value, but **concatenate a binary validity mask per variable** (1 = observed, 0 = imputed) so the network can learn to trust or discount a pixel. This is one of the more defensible technical choices in both source drafts — keep it.

**Normalization:** `x_norm = (x − μ_train) / σ_train`, with μ and σ computed **only** on the training years. Never let validation, test, or Argo data leak into these statistics.

**Resulting input tensor (Phase 1, single-day):** 7 physical variables + 7 validity masks = **14 channels** at each (lat, lon). (This is a correction from the Unified Plan's 16-channel tensor, which pre-bundles the 2 temporal-delta channels into the base input — see §6 on why temporal channels should be added later, not from day one.)

---

## 4. Train / Validation / Test Split — Temporal, Not Random

Ocean state is highly autocorrelated day-to-day. A random split leaks near-duplicate states across train/val and will produce misleadingly good numbers. Use a strict **temporal holdout**:

```
2015 ───────────────── 2021   |   2022   |   2023 ────── 2024
        TRAINING              VALIDATION      BLIND TEST
  (targets + norm stats)     (early stop,     (Argo-compared,
                              LR schedule)      reported once)
```

Exact year boundaries should flex to whatever span each dataset actually has available — don't force these specific years if the real data coverage differs; the *structure* (train early years, validate on a middle year, held-out blind test on the most recent years, Argo compared only at the end) is what matters.

---

## 5. Core Model Architecture — Final Recommendation

This is the Unified Plan's architecture, kept nearly as-is because it's genuinely well designed, with two framing corrections (marked below) and one build-order correction (§8).

```
                 14-CHANNEL SURFACE TENSOR  [B, 14, H, W]
                                    |
          +-------------------------+-------------------------+
          |                                                    |
          v                                                    v
 PATH A — SPATIAL CNN                                PATH B — POINTWISE MLP
 parallel 3×3 (local) + dilated 3×3                  1×1 conv / per-pixel MLP
 (~5×5 effective receptive field)                     on the 14-channel vector
 branches → concat → residual block                   at each (lat, lon)
 → F_spatial ∈ R^(B×128×H×W)                          → F_local ∈ R^(B×64×H×W)
          |                                                    |
          +-------------------------+-------------------------+
                                    |
                         FEATURE FUSION
                (concat → 1×1 projection → LayerNorm → GELU)
                                    |
                     EXPLICIT OCEAN EMBEDDING
                          Z ∈ R^(B×128×H×W)
                (bottleneck; PCA/t-SNE-able; reusable)
                                    |
                    ATTENTION-GUIDED DECODER
             (skip connections from encoder, attention
              gates over the reconstruction, NOT hard-
              coded to any single depth band a priori)
                                    |
                  15 DEPTH-CHANNEL OUTPUT  [B, 15, H, W]
                    T(0m) ... T(1000m)
                                    |
                 ┌──────────────────┴──────────────────┐
                 ↓                                      ↓
           GLORYS target                          Blind ARGO
         (training / metrics)                 (independent validation)
```

**Why dual-path at all:** the CNN branch answers "what's happening around this location" (fronts, eddies, gradients); the pointwise MLP answers "what does the local surface column look like" (preserves direct air-sea/column relationships that spatial pooling would blur). Both matter, so both stay.

**Correction #1 — don't over-claim what the two CNN branches "mean."** The Unified Plan describes the 3×3 branch as detecting "coastal plumes" and the dilated branch as detecting "mesoscale eddies." Don't say this to judges as a design guarantee — a CNN with two receptive-field scales gets multi-scale spatial context, and *some* of that will likely correlate with plume- and eddy-scale structure, but the network isn't constrained to specialize that way. Frame it correctly: "two receptive-field scales let the network combine fine-grained local gradients with broader regional context" — that's true and still sounds good.

**Correction #2 — don't hard-code a "thermocline attention gate at 50–200 m" as an architectural given.** Let the attention decoder learn where to focus; if you want to *check* that it learns to attend more around the thermocline, that's a great interpretability plot to show judges after training — as a finding, not a designed-in assumption.

**Embedding dimension (Z ∈ R^128):** keep this — it's the namesake feature of the project and gives you a strong, cheap demo (2D PCA/UMAP scatter of ocean regimes, colorable by season / Arabian Sea vs Bay of Bengal / reconstruction error). 128 is a reasonable starting point; treat it as a hyperparameter you can justify ("we tested 64/128/256 and 128 balanced reconstruction accuracy against a still-interpretable, still-fast embedding"), not a fixed physical constant.

---

## 6. Temporal Context — Phase 2, Not Phase 1

Both drafts agree temporal information (is a front strengthening or weakening?) is useful, and agree a heavy 7-day ConvLSTM is a bad idea for a hackathon (OOM risk, slow convergence, hard to debug under time pressure). The Unified Plan's answer — a lightweight 3-day delta (ΔSST, ΔSSH) as two extra input channels — is the right lightweight approach, **but it should not be baked into the Phase-1 tensor.**

Reason: if you add temporal deltas from the very first training run, you can never cleanly show *whether they helped*, and if the first end-to-end run has a bug, you won't know if it's in the new channels or the core architecture. Sequence it instead:

1. **Phase 1:** single-day, 14-channel input, full architecture above, trained and validated end-to-end. This is your working, defensible core result.
2. **Phase 2:** add ΔSST/ΔSSH as 2 extra channels (14 → 16), retrain, and report the delta in validation RMSE against Phase 1. Keep the addition **only if it measurably helps.**

Also correct the Unified Plan's causal language here: `ΔSSH > 0` with rising `ΔSST` doesn't *uniquely* prove "an intensifying anticyclonic warm-core eddy" — several physical situations could produce that signature. Present temporal-delta channels as *"trend features giving the model additional dynamical context,"* not as hand-built detectors of named phenomena.

---

## 7. Loss Function — Build Additively, Don't Pre-Commit Weights

Both drafts land on the same good principle (stated explicitly in the Improved Plan): don't hard-code loss weights without testing them. Follow that discipline literally:

**Step 1 (Phase 1, ship this first):**
```
L = mean over 15 depths of MSE(T_pred, T_GLORYS)
```
Uniform weighting. No thermocline emphasis, no smoothness term, no inversion term. Get this training stably and evaluate it — this is your baseline "does the architecture even work" result.

**Step 2 (only after Step 1 works, run as an ablation, not a default):**
```
L_total = L_depth_mse
        + λ_smooth · L_smooth        (penalizes jagged depth-to-depth curvature)
        + λ_inv    · L_inversion     (small penalty, NOT a monotonicity constraint —
                                       real profiles have inversions; this term should
                                       only discourage physically implausible spikes,
                                       never forbid non-monotonic profiles)
```
Test depth-weighting (e.g. up-weighting 50–200 m) **as an experiment with a reported before/after number**, not as a pre-decided `w=1.6`. If it improves blind-ARGO RMSE, keep it and report the value you landed on, with the justification being the measured improvement — not "because the thermocline is important."

**Do not publish exact target RMSE numbers per depth band before you have them.** The Unified Plan's table (e.g. "<0.38°C at 0–30m, <0.18°C at 700–1000m") looks precise but has no stated derivation — remove it. Instead, define success **relatively**: *"OceanEmbed must outperform a climatological-mean baseline and a linear/ridge regression baseline at every depth, with degrading-but-still-informative skill at greater depths where surface coupling weakens."* Report your actual achieved numbers once you have them; that's more credible to technical judges than a suspiciously specific pre-registered target anyway.

---

## 8. Baselines and Ablations (must-run, not optional)

| Model | Purpose |
|---|---|
| Climatological mean profile | Floor — is the model learning anything beyond seasonal average? |
| Linear / ridge regression (per depth) | Is nonlinearity from deep learning actually earning its complexity? |
| CNN-only (no MLP path) | Value of spatial context alone |
| MLP-only (no CNN path) | Value of pointwise/local signal alone |
| Full dual-path (Phase 1) | Main result |
| Dual-path + attention decoder vs. plain decoder | Value of attention specifically |
| Single-day vs. + temporal deltas (Phase 2) | Value of temporal context |
| Leave-one-variable-out (SSS, SSH, currents, winds) | Variable importance / sanity check |

Report depth-wise **RMSE, MAE, bias, and correlation** against both GLORYS (in-distribution check) and blind Argo (the number that actually matters), plus spatial error maps and a seasonal/regional breakdown (Arabian Sea vs. Bay of Bengal at minimum).

---

## 9. Uncertainty, Interpretability, Extensions (nice-to-have, in priority order)

- **Embedding visualization (do this — cheap, high demo value):** PCA/t-SNE/UMAP of the 128-D embedding, colored by region/season/error. Present as an exploration tool, not proof that every visual cluster maps to one named physical mechanism.
- **Uncertainty estimation (stretch goal):** MC-dropout or a small ensemble is the cheapest credible option under hackathon time; heteroscedastic/quantile regression is more principled but costs more implementation time — only attempt if Phase 1+2 are solid with time to spare.
- **TCHP / D26 / marine-heatwave indicators:** genuinely excellent for the judge pitch (ties directly to the PS's cyclone/MoES relevance), but position it explicitly as a **downstream application computed from OceanEmbed's output**, not as part of the core model or training objective. Frame any specific threshold (e.g., "D26 > X predicts intensification") as illustrative, not as a validated forecasting claim.

---

## 10. What Not to Claim (say this out loud in the pitch — it builds credibility, not weakness)

- ❌ "This measures the deep ocean directly." → ✅ "This reconstructs a physically plausible estimate, trained against reanalysis and checked against real floats."
- ❌ "This replaces Argo floats / research vessels." → ✅ "This complements sparse in-situ networks with continuous gridded coverage between them."
- ❌ "Uniform accuracy to 1000 m." → ✅ "Skill is expected to degrade with depth as surface–subsurface coupling weakens — we report this explicitly rather than hide it."
- ❌ Specific pre-registered RMSE targets or exact VRAM/training-time numbers before you've measured them. → ✅ "Designed to be trainable on a single consumer/Colab-class GPU (e.g. RTX 3060 / T4); actual footprint reported after implementation."

---

## 11. System, Stack, and Repo Layout

**Stack:** Python/xarray/xESMF for the data pipeline; PyTorch (+ Lightning if useful) for the model; scikit-learn for PCA/t-SNE; FastAPI backend; a React + Leaflet (map) + Plotly (depth-profile curve) dashboard; Docker for packaging; optional ONNX export for lighter-weight serving during the demo.

```
OceanEmbed/
├── pipeline/          # download, regrid, mask, normalize, temporal deltas (Phase 2)
├── models/
│   ├── dual_path.py        # CNN + MLP + fusion + embedding bottleneck
│   ├── attention_decoder.py
│   └── losses.py            # MSE first; smoothness/inversion added behind flags
├── baselines/          # climatology, ridge regression
├── eval/               # GLORYS metrics, blind ARGO comparison, ablation runner
├── api/                # FastAPI: /reconstruct, /embedding, /tchp (extension)
├── dashboard/           # React + Leaflet + Plotly
├── notebooks/           # embedding visualization, error maps, ablation plots
├── Dockerfile / docker-compose.yml
└── README.md
```

**Data-leakage checklist (must all be true before you report a single number):**
1. Normalization stats computed only on training years.
2. Argo never touched until the single final blind evaluation.
3. Any temporal-delta features only look backward in time.
4. Test period is strictly out-of-time relative to training.

---

## 12. Build Order (the part that actually decides whether you finish)

```
1. Data acquisition + 1-month dry-run of full pipeline
2. Full harmonization pipeline (all years)
3. Temporal train/val/test split + leakage checklist
4. Baselines (climatology, ridge regression) — cheap, do first, gives you a floor
5. Phase-1 model: dual-path CNN+MLP, explicit embedding, attention decoder,
   single-day input, plain MSE loss
6. Evaluate Phase 1 against GLORYS + blind Argo — THIS IS YOUR MINIMUM VIABLE RESULT
7. Ablations (CNN-only, MLP-only, no-attention) — cheap once Phase 1 exists
8. Phase 2: temporal deltas — keep only if they measurably help
9. Loss additions: smoothness, then inversion-tolerant term — keep only if they help
10. Embedding visualization (PCA/UMAP) — high demo value, do this even under time pressure
11. Uncertainty estimation (if time remains)
12. TCHP/D26 extension demo (if time remains)
13. Dashboard + API integration
14. Pitch deck + recorded demo
```

If you run out of time, **stop after step 6 or 7** and present that honestly — a working, validated Phase-1 model with real Argo comparison numbers is a stronger submission than an ambitious but broken/untested Phase-2+ system.

**Must-have vs. nice-to-have, for triage under time pressure:**

| Must have (this is the submission floor) | Should have | Nice to have |
|---|---|---|
| End-to-end harmonized pipeline | Ablation studies | Temporal deltas |
| Working dual-path model, single-day | Embedding PCA/UMAP viz | Uncertainty estimation |
| Explicit embedding + attention decoder | Spatial/seasonal error maps | Depth-conditioned decoder |
| GLORYS training + blind Argo validation | Mask-aware missing data | TCHP/D26 dashboard extension |
| Baseline comparison + depth-wise metrics | | Near-real-time inference |
| Working demo (even a notebook is fine) | | |

---

## 13. Demo Design (my own addition — neither draft specified this, and it's what judges actually watch)

The strongest live demo for this PS is a single interaction, not a slide of metrics:

1. User picks a date + lat/lon on the dashboard map (or a pre-picked "interesting" location — an eddy, an upwelling zone).
2. Model returns the reconstructed 15-depth profile as a curve (Plotly).
3. Overlay the nearest **actual blind Argo profile** for that date/location on the same chart, if one exists nearby in time/space.
4. Show the 128-D embedding's 2D PCA position for that sample, colored against the full embedding scatter, so judges see "this profile came from a location the model has learned looks like an upwelling regime / eddy regime / open-ocean regime."

This single view demonstrates the reconstruction, the validation, and the "embedding" concept in one screen, which is exactly what a judge needs in a 5-minute pitch.

---

## 14. One-Line Description and Judge Pitch

> **OceanEmbed is a dual-path deep-learning framework that fuses spatial-CNN and pointwise-column encodings of multi-source surface ocean observations into an explicit 128-dimensional Ocean Embedding, decoded through an attention-guided head into a 15-depth subsurface temperature profile — trained against GLORYS reanalysis and independently validated against real Argo float measurements.**

**Problem → Solution → Innovation → Validation, in judge language:**
- *Problem:* Satellites see the surface; the subsurface thermal structure that drives cyclone intensification, monsoon coupling, and marine heatwaves is invisible to them.
- *Solution:* Learn the surface-to-subsurface relationship from physics-rich reanalysis data, using surface temperature, salinity, height, currents, and winds together.
- *Innovation:* An explicit, visualizable 128-D "Ocean Embedding" bottleneck — not just a black-box mapping — that can be inspected, clustered, and reused for downstream products like TCHP.
- *Validation:* Every reported number is checked against Argo floats the model never saw during training — with the leakage-prevention rules stated up front, unprompted, before a judge has to ask.

---

### Bottom line

Build the architecture from the Unified Plan. Build it in the order from the Improved Plan's discipline (baseline → simple working core → validate → then, and only then, add complexity). Never state a number — RMSE target, VRAM figure, training time — you haven't actually measured. That combination is the version of this project most likely to both work by the deadline and survive tough technical questioning from judges.
