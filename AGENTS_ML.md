# OceanEmbed — ML/DL Agent Instructions

## 1. Role

This workspace is dedicated ONLY to the OceanEmbed ML/DL and scientific-data pipeline.

The prototype/frontend is being developed separately by another team member.

Your responsibilities:
- scientific data acquisition support
- data QC and harmonization
- preprocessing
- dataset construction
- baselines
- PyTorch model implementation
- training
- evaluation against GLORYS
- blind Argo validation
- ablations
- Ocean Embedding extraction/visualization
- model checkpointing
- inference/export interface for later prototype integration

Do NOT build the React dashboard, frontend, FastAPI application, authentication, deployment infrastructure, or unrelated product features.

---

## 2. Source of Truth

The authoritative file is:

`OceanEmbed_Final_Technical_Plan.md`

Treat it as the single source of truth for the scientific problem, datasets, terminology, model architecture, evaluation methodology, and roadmap.

Do not silently modify the scientific architecture because of implementation convenience.

If an implementation detail is unclear, identify it and ask before changing the scientific specification.

---

## 3. Hardware

Development/training machine:

- GPU: NVIDIA GeForce RTX 5060
- VRAM: approximately 8 GB (8151 MiB reported)
- OS: Windows
- NVIDIA driver: 616.92
- CUDA UMD: 13.4

Use this information for practical training configuration only.

Use:
- configurable batch size
- mixed precision where supported
- memory-conscious data loading
- gradient accumulation if necessary
- checkpointing

Do not redesign the scientific model because of the GPU.

Never hard-code claims about training time, VRAM requirements, or expected accuracy.

---

## 4. Scientific Problem

OceanEmbed reconstructs subsurface ocean temperature over:

- North Indian Ocean
- 5°N–30°N
- 45°E–105°E
- 0.25° × 0.25°
- daily

Output depths:

`0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m`

Scientific framing:

Satellites observe the surface/ocean-skin layer. OceanEmbed estimates the hidden subsurface thermal structure from multi-source surface observations. The model is trained against GLORYS reanalysis and independently evaluated against real Argo observations.

Never describe the model as directly measuring the deep ocean or replacing Argo/ships/moorings.

---

## 5. Phase-1 Input

Use exactly:

7 physical surface variables:
1. SST
2. SSS
3. SSH/SLA
4. surface current U
5. surface current V
6. surface wind U
7. surface wind V

plus 7 validity masks.

Therefore:

`input = [B, 14, H, W]`

Do NOT add ΔSST or ΔSSH in Phase 1.

Those are Phase-2 experimental features.

---

## 6. Phase-1 Model

Implement exactly this conceptual architecture:

```text
14-channel input
       |
       +------------------------------+
       |                              |
       v                              v
Multi-scale CNN                 Pointwise MLP
local 3×3                       / 1×1 encoding
+ dilated 3×3
       |                              |
       v                              v
128 spatial features            64 local features
       |                              |
       +--------------+---------------+
                      |
                      v
                Feature Fusion
                      |
                      v
             Explicit 128-D
             Ocean Embedding
                      |
                      v
           Attention-guided Decoder
                      |
                      v
             15 depth outputs
```

Important:
- Ocean Embedding is explicitly 128-dimensional.
- Keep the embedding accessible from the model for later PCA/UMAP/t-SNE.
- Attention is learned.
- Do not hard-code attention to 50–200 m.
- Do not claim that individual CNN branches specifically detect named ocean phenomena without experimental evidence.
- Do not add ConvLSTM or heavy temporal architecture to Phase 1.

---

## 7. Phase-1 Loss

Start with:

**plain uniform-depth MSE**

Do not initially add:
- thermocline weighting
- smoothness loss
- inversion penalty
- complex physics loss

Only after Phase 1 is stable and evaluated should these become controlled experiments.

Every experimental loss should be configurable and its effect measured against the Phase-1 baseline.

---

## 8. Data Sources

Surface inputs follow the technical plan:

- OSTIA SST
- SMAP/SMOS SSS
- DUACS SSH/SLA
- OSCAR L4 currents
- CCMP v3.1 as primary winds
- ASCAT-C as possible alternative

Target:
- GLORYS12V1

Independent validation:
- QC/gridded Argo through INCOIS LAS

Use a 1-month dry run before attempting multi-year processing.

Data acquisition must be configurable and must clearly document credential requirements.

Do not fabricate access credentials or bypass registration/rate limits.

---

## 9. Preprocessing

Implement and test:

1. quality control
2. common 0.25° target grid
3. daily alignment
4. land/invalid mask
5. missing-value handling using the plan's regional-climatology + validity-mask strategy
6. training-only z-score normalization
7. strict temporal train/validation/test split

Example split:

- 2015–2021 train
- 2022 validation
- 2023–2024 blind test

Exact years may change only if dataset availability requires it.

Never randomly mix dates between train and test for the primary scientific evaluation.

Never compute normalization statistics from validation/test/Argo data.

---

## 10. GLORYS

GLORYS is a data-assimilative reanalysis target/reference, not observational ground truth.

Implement:
- acquisition
- resampling to 0.25°
- interpolation to the 15 required depths
- alignment with surface inputs
- target masks where appropriate

Document all interpolation/resampling assumptions.

Do not describe GLORYS as absolute ground truth.

---

## 11. Argo Rule

Argo is independent validation.

Do NOT use Argo for:
- training
- hyperparameter tuning
- early stopping
- model selection

Do not inspect Argo results while choosing between model variants.

Prepare the validation code/interface, but keep the actual blind evaluation isolated until the model and experimental choices are locked.

When blind evaluation is performed, record the exact protocol and results.

Never fabricate Argo agreement or performance.

---

## 12. Baselines

Implement before interpreting the deep model:

1. climatological baseline
2. linear/ridge regression baseline

Record:
- RMSE by depth
- MAE by depth
- bias by depth
- correlation/R² where appropriate

Store measured results as JSON/CSV.

Never hard-code expected values.

---

## 13. Evaluation

For the model and baselines, support:

- depth-wise RMSE
- depth-wise MAE
- depth-wise bias
- correlation/R² where appropriate
- spatial error maps
- seasonal breakdown
- regional breakdown

Primary results should show how skill changes with depth.

Do not collapse the entire profile into one misleading accuracy number.

---

## 14. Ablations

After a stable Phase-1 model:

- CNN-only
- MLP-only
- full dual-path
- attention vs non-attention
- leave-one-variable-out
- Phase-2 temporal features if implemented

Each ablation must have:
- fixed evaluation protocol
- recorded configuration
- measured result
- comparison against the Phase-1 reference

Do not select a winner before running the experiment.

---

## 15. Notebook + Python Workflow

The ML work must support BOTH:

### Jupyter notebooks

Use notebooks for:
- exploration
- visualization
- experiments
- debugging
- training runs
- evaluation
- embedding analysis

Recommended notebooks:

```text
notebooks/
├── 01_data_dry_run.ipynb
├── 02_data_exploration.ipynb
├── 03_pipeline_validation.ipynb
├── 04_baselines.ipynb
├── 05_phase1_training.ipynb
├── 06_phase1_evaluation.ipynb
├── 07_embedding_visualization.ipynb
├── 08_argo_blind_evaluation.ipynb
└── 09_ablation_analysis.ipynb
```

### Python modules

Reusable implementation MUST live in `.py` modules.

Do not put the only copy of:
- model architecture
- preprocessing
- dataset logic
- evaluation metrics

inside notebooks.

Notebooks should import reusable Python modules.

Example:

```python
from models.ocean_embed_net import OceanEmbedNet
```

This allows the same implementation to be used by the future inference service.

---

## 16. Recommended ML Repository

```text
OceanEmbed-ML/
├── AGENTS.md
├── OceanEmbed_Final_Technical_Plan.md
├── README.md
│
├── configs/
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   └── eval.yaml
│
├── pipeline/
│   ├── qc.py
│   ├── regrid.py
│   ├── align.py
│   ├── mask.py
│   ├── normalize.py
│   ├── datasets.py
│   ├── glorys.py
│   └── build_dataset.py
│
├── models/
│   ├── multi_scale_cnn.py
│   ├── pointwise_mlp.py
│   ├── embedding.py
│   ├── attention_decoder.py
│   └── ocean_embed_net.py
│
├── baselines/
│   ├── climatology.py
│   └── ridge.py
│
├── evaluation/
│   ├── metrics.py
│   ├── glorys_eval.py
│   ├── argo_eval.py
│   └── ablations.py
│
├── notebooks/
├── checkpoints/
├── results/
├── tests/
└── scripts/
```

Large raw/processed datasets should not be committed to Git.

---

## 17. Checkpoints and Integration

Save trained models to:

`checkpoints/`

Make checkpoint paths configurable.

A future prototype backend will need a clean inference interface.

Therefore provide a simple Python inference wrapper conceptually like:

```python
prediction = model.predict(surface_input)
```

or an equivalent typed interface.

The interface should return:
- 15 depths
- 15 temperatures
- 128-dimensional Ocean Embedding
- model/version metadata

The exact integration contract will be documented separately with the prototype developer.

Do not build FastAPI in this ML workspace unless explicitly requested.

---

## 18. Reproducibility

Every experiment should record:

- configuration
- random seed
- dataset split
- model configuration
- normalization statistics/version
- checkpoint
- metrics
- software environment where practical

Use deterministic settings where practical, while documenting any performance trade-offs.

---

## 19. Testing

At minimum test:

- QC behavior
- regridding shape
- mask behavior
- normalization behavior
- dataset tensor shapes
- model forward-pass shapes
- 15-depth output
- 128-D embedding output
- NaN/Inf handling
- temporal split leakage

Run tests before expensive training.

---

## 20. Scientific Integrity

Never invent:
- accuracy
- RMSE
- MAE
- R²
- Argo results
- dataset observations
- training time
- VRAM usage claims
- scientific citations
- performance targets

Clearly distinguish:
- measured results
- planned experiments
- hypotheses
- engineering estimates
- synthetic/mock data

---

## 21. Development Order

Follow this order:

### Phase 0
Understand plan → inspect environment → create structure → configure GPU/environment.

### Phase 1
1-month data dry run → QC → regrid → alignment → masks → normalization → validation.

### Phase 2
Climatology + ridge baselines.

### Phase 3
Phase-1 dual-path model + 128-D embedding + attention decoder + MSE.

### Phase 4
GLORYS evaluation.

### Phase 5
Controlled ablations.

### Phase 6
Blind Argo evaluation after model/experiment choices are locked.

### Phase 7
Embedding visualization.

### Phase 8
Phase-2 temporal features if justified.

### Phase 9
Loss extensions and other optional experiments.

Do not jump ahead because a later feature is visually interesting.

---

## 22. Agent Behaviour

Before writing substantial code:

1. Read AGENTS.md completely.
2. Read OceanEmbed_Final_Technical_Plan.md completely.
3. Inspect the current workspace.
4. Verify the environment and GPU.
5. Identify missing credentials/access requirements.
6. Produce a Phase-0 implementation plan.
7. Identify assumptions and potential scientific risks.
8. Wait for approval.

After approval:
- implement incrementally
- test each stage
- show actual outputs
- stop at phase boundaries
- do not silently start the next major phase

The objective is a reproducible, scientifically defensible ML pipeline, not merely code that runs.
