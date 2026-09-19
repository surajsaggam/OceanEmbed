"""
examples/inference_example.py
-----------------------------
Reference demonstration of the OceanEmbed ML inference pipeline.
Designed for the frontend/backend engineering team to understand the complete
public ML interface without needing any training or scientific pipeline knowledge.

Workflow Demonstrated:
  1. Initialize the frozen Phase-1 predictor.
  2. Prepare 7 surface observation fields (SST, SSS, SSH, U_curr, V_curr, WindU, WindV).
  3. Execute deterministic inference.
  4. Inspect 3D temperature fields, latent embedding, and spatial attention maps.
  5. Extract a 15-depth vertical temperature profile at an arbitrary coordinate.
  6. Print structured output metadata and physical predictions.

Usage:
  python examples/inference_example.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

# ── Import the public ML interface ───────────────────────────────────────────
from inference import OceanEmbedPredictor, extract_profile


def main() -> None:
    print("=" * 80)
    print("OCEANEMBED -- PRODUCTION ML INFERENCE DEMONSTRATION")
    print("=" * 80)

    # 1. Initialize the predictor
    # Automatically verifies SHA256 against frozen Phase-1 checkpoint and sets eval mode
    print("\n[Step 1] Initializing OceanEmbedPredictor...")
    predictor = OceanEmbedPredictor(device=None)  # Auto-detects CUDA GPU or CPU

    meta_init = predictor.predict(
        np.zeros((14, 101, 241), dtype=np.float32)
    )["metadata"]

    print(f"  Model Identifier:  {meta_init['model_name']}")
    print(f"  Checkpoint Path:   {meta_init['checkpoint_path']}")
    print(f"  Checkpoint SHA256: {meta_init['checkpoint_sha256']}")
    print(f"  Inference Device:  {meta_init['device']}")
    print(f"  Evaluation Mode:   {meta_init['eval_mode']} (requires_grad=False)")

    # 2. Prepare 7 surface observation variables
    # Format: Dict[str, np.ndarray] with each array having shape [101, 241]
    # Grid: North Indian Ocean 0.25° grid (lat: 5.0–30.0°N, lon: 45.0–105.0°E)
    print("\n[Step 2] Preparing 7 surface observation fields...")
    H, W = 101, 241

    # Check if a real processed test sample is available, else generate realistic physical fields
    test_sample_path = PROJECT_ROOT / "data" / "processed" / "test" / "oceanembed_2019-01-01.npz"
    if test_sample_path.exists():
        print(f"  Loading real satellite surface observations from: {test_sample_path.name}")
        with np.load(test_sample_path) as data:
            raw_input = data["input"]  # [14, 101, 241]
        
        # De-normalize channels 0..6 to represent the raw physical inputs
        # (This demonstrates how raw physical fields from NetCDF/Zarr are passed)
        stats = predictor.norm_stats
        surface_observations = {
            "SST": raw_input[0] * stats["SST"]["std"] + stats["SST"]["mean"],
            "SSS": raw_input[1] * stats["SSS"]["std"] + stats["SSS"]["mean"],
            "SSH": raw_input[2] * stats["SSH"]["std"] + stats["SSH"]["mean"],
            "U_curr": raw_input[3] * stats["U_curr"]["std"] + stats["U_curr"]["mean"],
            "V_curr": raw_input[4] * stats["V_curr"]["std"] + stats["V_curr"]["mean"],
            "WindU": raw_input[5] * stats["WindU"]["std"] + stats["WindU"]["mean"],
            "WindV": raw_input[6] * stats["WindV"]["std"] + stats["WindV"]["mean"],
        }
        sample_date = "2019-01-01"
    else:
        print("  Generating synthetic surface observation fields...")
        rng = np.random.default_rng(42)
        surface_observations = {
            "SST": rng.uniform(24.0, 31.0, size=(H, W)).astype(np.float32),
            "SSS": rng.uniform(32.0, 37.0, size=(H, W)).astype(np.float32),
            "SSH": rng.uniform(-0.3, 0.3, size=(H, W)).astype(np.float32),
            "U_curr": rng.uniform(-0.5, 0.5, size=(H, W)).astype(np.float32),
            "V_curr": rng.uniform(-0.5, 0.5, size=(H, W)).astype(np.float32),
            "WindU": rng.uniform(-10.0, 10.0, size=(H, W)).astype(np.float32),
            "WindV": rng.uniform(-10.0, 10.0, size=(H, W)).astype(np.float32),
        }
        sample_date = "2020-01-01"

    for k, v in surface_observations.items():
        print(f"    {k:8s} -> shape: {v.shape}, min: {v.min():.2f}, max: {v.max():.2f}")

    # 3. Execute inference
    print("\n[Step 3] Running deterministic forward inference...")
    result = predictor.predict(surface_observations, date=sample_date)

    # 4. Inspect outputs
    temp = result["temperature"]       # [15, 101, 241] in °C
    emb = result["embedding"]          # [128, 101, 241]
    attn = result["attention"]         # [1, 101, 241]
    v_mask = result["validity_mask"]   # [7, 101, 241]
    o_mask = result["ocean_mask"]      # [101, 241]
    meta = result["metadata"]

    print("\n[Step 4] Inference Output Tensor Contract:")
    print(f"  Temperature Reconstruction: shape = {temp.shape} | dtype = {temp.dtype} | units = {meta['units']['temperature']}")
    print(f"  Ocean Embedding (Z):        shape = {emb.shape} | dtype = {emb.dtype} | units = {meta['units']['embedding']}")
    print(f"  Spatial Attention Map:      shape = {attn.shape} | dtype = {attn.dtype} | min = {attn.min():.4f}, max = {attn.max():.4f}")
    print(f"  Validity Masks (7 vars):    shape = {v_mask.shape} | binary values = {np.unique(v_mask).tolist()}")
    print(f"  Ocean Grid Coverage:        {int(o_mask.sum())} / {H * W} ocean cells ({o_mask.mean() * 100:.1f}%)")
    print(f"  All outputs 100% finite:    {np.all(np.isfinite(temp)) and np.all(np.isfinite(emb)) and np.all(np.isfinite(attn))}")

    # 5. Extract a 15-depth vertical temperature profile
    # Query central Arabian Sea (15.0°N, 65.0°E)
    query_lat, query_lon = 15.0, 65.0
    print(f"\n[Step 5] Extracting 15-depth vertical profile at ({query_lat}N, {query_lon}E)...")
    profile = extract_profile(result, lat=query_lat, lon=query_lon)

    print(f"  Matched Grid Location: ({profile['lat_grid']}N, {profile['lon_grid']}E)")
    print(f"  Grid Cell Index:       (row={profile['grid_idx'][0]}, col={profile['grid_idx'][1]})")
    print(f"  Valid Ocean Pixel:     {profile['is_valid_ocean']}")
    print(f"  Local Attention Gate:  {profile.get('attention_weight', 0.0):.4f}")

    print("\n  +---------------+------------------------+")
    print("  |   Depth (m)   |   Temperature (deg C)  |")
    print("  +---------------+------------------------+")
    for depth, t_val in zip(profile["depth_m"], profile["temperature_C"]):
        print(f"  |   {depth:6d} m    |       {t_val:6.2f} C         |")
    print("  +---------------+------------------------+")

    print("\n[Done] ML inference workflow executed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
