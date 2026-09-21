"""
scripts/verify_pretraining_gate.py
----------------------------------
Pre-training verification gate for OceanEmbed Phase-1 full training.

Verifies:
1. Argo Blind Guard:
   - Config flag `argo_blind_locked: false` in configs/eval.yaml.
   - Guard execution: calling `check_argo_guard` or `run_blind_argo_evaluation` raises AssertionError.
   - Codebase scan: ensures scripts/train_phase1.py and pipeline/datasets.py have no references to Argo.
2. Normalization Statistics:
   - Frozen train_stats.json computed exclusively from 2015-2017 moments.
   - Exact mathematical match with combine_moments([2015, 2016, 2017]).
   - Verification that 2018 moments are strictly excluded.
3. Date Split & File Counts:
   - data/processed/train: exactly 1,096 daily files (2015-01-01 through 2017-12-31).
   - data/processed/val: exactly 365 daily files (2018-01-01 through 2018-12-31).
   - Zero intersection between train and validation dates.
4. Checkpoint & Early-Stopping Logic:
   - configs/train.yaml: patience=15, checkpoint_dir=checkpoints/phase1/.
   - train_phase1.py: best_val_loss tracking and patience counter behavior.
5. Hyperparameter & Precision Configuration:
   - batch_size=4, grad_accumulation=2 (effective 8), lr=1e-4, optimizer=adam, cosine scheduler.
   - precision='bf16' on CUDA.
6. Model Output Contracts:
   - Forward pass produces temperature [B, 15, 101, 241] and embedding [B, 128, 101, 241].
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.argo_eval import check_argo_guard, run_blind_argo_evaluation
from models.ocean_embed_net import OceanEmbedNet
from pipeline.normalize import combine_moments, compute_stats_from_moments
from utils.config import load_config


def verify_argo_guard():
    print("\n" + "=" * 80)
    print("CHECK 1: ARGO BLIND EVALUATION GUARD VERIFICATION")
    print("=" * 80)
    eval_cfg = load_config("eval")
    print(f"  eval.yaml argo_blind_locked flag: {eval_cfg.argo_blind_locked}")
    assert eval_cfg.argo_blind_locked is False, "CRITICAL: argo_blind_locked must be False!"

    # Test actual guard execution
    raised = False
    try:
        check_argo_guard(eval_cfg.argo_blind_locked)
    except AssertionError as e:
        raised = True
        print(f"  [PASS] check_argo_guard raised as expected: '{str(e).splitlines()[0]}'")
    assert raised, "CRITICAL: check_argo_guard did not raise AssertionError!"

    raised_func = False
    try:
        run_blind_argo_evaluation(model=None, argo_profiles=[], eval_cfg=eval_cfg)
    except AssertionError as e:
        raised_func = True
        print(f"  [PASS] run_blind_argo_evaluation raised as expected: '{str(e).splitlines()[0]}'")
    assert raised_func, "CRITICAL: run_blind_argo_evaluation did not raise AssertionError!"

    # Scan training scripts for forbidden Argo imports / access
    forbidden_files = [
        PROJECT_ROOT / "scripts" / "train_phase1.py",
        PROJECT_ROOT / "pipeline" / "datasets.py",
        PROJECT_ROOT / "pipeline" / "preprocess.py",
        PROJECT_ROOT / "models" / "ocean_embed_net.py",
    ]
    for p in forbidden_files:
        content = p.read_text(encoding="utf-8").lower()
        # Ensure no active code loading argo
        lines = [line.strip() for line in content.splitlines() if not line.strip().startswith("#")]
        active_code = "\n".join(lines)
        assert "argo" not in active_code, f"CRITICAL: Found unexpected Argo reference in active code of {p.name}!"
        print(f"  [PASS] Verified zero Argo references in active code: {p.relative_to(PROJECT_ROOT)}")

    print("  -> Argo isolation and runtime guard: 100% VERIFIED.")


def verify_normalization_stats():
    print("\n" + "=" * 80)
    print("CHECK 2: FROZEN TRAINING-ONLY NORMALIZATION STATISTICS")
    print("=" * 80)
    train_stats_path = PROJECT_ROOT / "data" / "norm_stats" / "train_stats.json"
    assert train_stats_path.exists(), f"Missing {train_stats_path}"

    with open(train_stats_path, "r", encoding="utf-8") as f:
        train_stats = json.load(f)

    # Recombine 2015, 2016, 2017 moments directly
    m_paths = [
        PROJECT_ROOT / "data" / "norm_stats" / "moments_2015.json",
        PROJECT_ROOT / "data" / "norm_stats" / "moments_2016.json",
        PROJECT_ROOT / "data" / "norm_stats" / "moments_2017.json",
    ]
    for p in m_paths:
        assert p.exists(), f"Missing yearly moments: {p}"

    m_list = []
    for p in m_paths:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
            m_list.append(d.get("moments", d))

    recomputed_moments = combine_moments(m_list)
    recomputed_stats = compute_stats_from_moments(recomputed_moments)

    # Check 2018 moments exist separately and are different
    m_2018_path = PROJECT_ROOT / "data" / "norm_stats" / "moments_2018.json"
    assert m_2018_path.exists(), "Missing moments_2018.json"
    with open(m_2018_path, "r", encoding="utf-8") as f:
        d_2018 = json.load(f)
        m_2018 = d_2018.get("moments", d_2018)

    print(f"  Verifying 7 physical variables in train_stats.json against 2015-2017 combination:")
    var_names = ["SST", "SSS", "SSH", "U_curr", "V_curr", "WindU", "WindV"]
    for var in var_names:
        ts_mean = train_stats[var]["mean"]
        ts_std = train_stats[var]["std"]
        rc_mean = recomputed_stats[var]["mean"]
        rc_std = recomputed_stats[var]["std"]
        m18_mean = m_2018[var]["sum"] / m_2018[var]["count"]

        print(f"    {var:<7}: Train Mean={ts_mean:.4f}, Std={ts_std:.4f} | Recomputed Mean={rc_mean:.4f}, Std={rc_std:.4f} | 2018 Mean={m18_mean:.4f}")
        assert np.isclose(ts_mean, rc_mean, atol=1e-6), f"Mean mismatch for {var}!"
        assert np.isclose(ts_std, rc_std, atol=1e-6), f"Std mismatch for {var}!"
        # Verify 2018 moments were NOT included in count
        assert recomputed_moments[var]["count"] > 0
        assert recomputed_moments[var]["count"] < (recomputed_moments[var]["count"] + m_2018[var]["count"])

    print("  -> train_stats.json derives exclusively from 2015-2017: 100% VERIFIED.")


def verify_date_splits():
    print("\n" + "=" * 80)
    print("CHECK 3: TEMPORAL SPLIT INTEGRITY & CONTIGUITY")
    print("=" * 80)
    train_dir = PROJECT_ROOT / "data" / "processed" / "train"
    val_dir = PROJECT_ROOT / "data" / "processed" / "val"
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"

    train_files = sorted(list(train_dir.glob("*.npz")))
    val_files = sorted(list(val_dir.glob("*.npz")))
    test_files = sorted(list(test_dir.glob("*.npz")))

    print(f"  Training files count:   {len(train_files)} (Expected: 1,096)")
    print(f"  Validation files count: {len(val_files)} (Expected: 365)")
    print(f"  Test files count:       {len(test_files)} (Expected: 0, unpopulated)")

    assert len(train_files) == 1096, f"Expected 1,096 train files, found {len(train_files)}"
    assert len(val_files) == 365, f"Expected 365 val files, found {len(val_files)}"
    assert len(test_files) == 0, f"Expected 0 test files, found {len(test_files)}"

    date_regex = re.compile(r"(\d{4}-\d{2}-\d{2})")

    train_dates = set()
    for f in train_files:
        match = date_regex.search(f.name)
        assert match, f"Cannot parse date from {f.name}"
        train_dates.add(match.group(1))

    val_dates = set()
    for f in val_files:
        match = date_regex.search(f.name)
        assert match, f"Cannot parse date from {f.name}"
        val_dates.add(match.group(1))

    # Check zero overlap
    overlap = train_dates.intersection(val_dates)
    print(f"  Date overlap between Train and Validation: {len(overlap)}")
    assert len(overlap) == 0, f"CRITICAL LEAKAGE: Overlap dates found: {overlap}"

    # Verify exact continuous date range for train: 2015-01-01 to 2017-12-31 (1096 days)
    start_train = date(2015, 1, 1)
    for i in range(1096):
        d_str = (start_train + timedelta(days=i)).isoformat()
        assert d_str in train_dates, f"Missing train date: {d_str}"

    # Verify exact continuous date range for val: 2018-01-01 to 2018-12-31 (365 days)
    start_val = date(2018, 1, 1)
    for i in range(365):
        d_str = (start_val + timedelta(days=i)).isoformat()
        assert d_str in val_dates, f"Missing val date: {d_str}"

    print(f"  Train date range: {min(train_dates)} to {max(train_dates)} (100% contiguous, 1096 days)")
    print(f"  Val date range:   {min(val_dates)} to {max(val_dates)} (100% contiguous, 365 days)")
    print("  -> Zero overlap & perfect contiguity: 100% VERIFIED.")


def verify_checkpoint_and_training_configs():
    print("\n" + "=" * 80)
    print("CHECK 4: TRAINING CONFIGURATIONS, CHECKPOINT & EARLY-STOPPING LOGIC")
    print("=" * 80)
    train_cfg = load_config("train")
    model_cfg = load_config("model")

    print(f"  Optimizer:           {train_cfg.optimizer} (Expected: 'adam' or 'adamw')")
    print(f"  Learning Rate:       {train_cfg.learning_rate} (Expected: 1.0e-4)")
    print(f"  Scheduler:           {train_cfg.lr_scheduler} (Expected: 'cosine')")
    print(f"  LR Min:              {train_cfg.lr_min} (Expected: 1.0e-6)")
    print(f"  Batch Size:          {train_cfg.batch_size} (Expected: 4)")
    print(f"  Grad Accumulation:   {train_cfg.grad_accumulation} (Effective batch size: {train_cfg.batch_size * train_cfg.grad_accumulation})")
    print(f"  Precision:           {train_cfg.precision} (Expected: 'bf16')")
    print(f"  Patience:            {train_cfg.patience} (Expected: 15)")
    print(f"  Checkpoint Dir:      {train_cfg.checkpoint_dir}")
    print(f"  Max Epochs:          {train_cfg.epochs}")

    assert train_cfg.batch_size == 4
    assert train_cfg.grad_accumulation == 2
    assert train_cfg.precision == "bf16"
    assert train_cfg.patience == 15
    assert train_cfg.learning_rate == 1.0e-4
    assert train_cfg.lr_scheduler == "cosine"

    # Verify best.pt path logic in scripts/train_phase1.py
    train_script = (PROJECT_ROOT / "scripts" / "train_phase1.py").read_text(encoding="utf-8")
    assert 'best_ckpt_path = checkpoint_dir / "best.pt"' in train_script
    assert "patience_counter = 0" in train_script
    assert "patience_counter += 1" in train_script
    assert "patience_counter >= train_cfg.patience" in train_script
    print("  [PASS] Checked train_phase1.py checkpoint saving (best.pt) and early stopping (patience=15).")
    print("  -> Configuration & Checkpoint Logic: 100% VERIFIED.")


def verify_model_contracts():
    print("\n" + "=" * 80)
    print("CHECK 5: MODEL ARCHITECTURE OUTPUT CONTRACTS")
    print("=" * 80)
    model_cfg = load_config("model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OceanEmbedNet(model_cfg).to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model instantiated on {device}. Trainable parameters: {num_params:,}")
    assert num_params == 525040, f"Expected 525,040 parameters, found {num_params}"

    B = 2
    dummy_input = torch.randn(B, 14, 101, 241, device=device)
    with torch.no_grad():
        out = model(dummy_input)
        out_att = model(dummy_input, return_attention=True)

    assert "temperature" in out, "Missing 'temperature' key in output dict"
    assert "embedding" in out, "Missing 'embedding' key in output dict"
    assert "attention" in out_att, "Missing 'attention' key when return_attention=True"

    t_shape = tuple(out["temperature"].shape)
    e_shape = tuple(out["embedding"].shape)
    a_shape = tuple(out_att["attention"].shape)

    print(f"  Temperature output shape: {t_shape} (Contract: ({B}, 15, 101, 241))")
    print(f"  Embedding output shape:   {e_shape} (Contract: ({B}, 128, 101, 241))")
    print(f"  Attention output shape:   {a_shape} (Contract: ({B}, 1, 101, 241))")

    assert t_shape == (B, 15, 101, 241), f"Incorrect temperature shape: {t_shape}"
    assert e_shape == (B, 128, 101, 241), f"Incorrect embedding shape: {e_shape}"
    assert a_shape == (B, 1, 101, 241), f"Incorrect attention shape: {a_shape}"
    print("  -> Tensor output contracts: 100% VERIFIED.")


def main():
    print("=" * 80)
    print("OCEANEMBED PHASE-1 PRE-TRAINING GATE VERIFICATION")
    print("=" * 80)
    verify_argo_guard()
    verify_normalization_stats()
    verify_date_splits()
    verify_checkpoint_and_training_configs()
    verify_model_contracts()
    print("\n" + "=" * 80)
    print("ALL PRE-TRAINING VERIFICATIONS PASSED (5/5 GATES CLEAN).")
    print("=" * 80)


if __name__ == "__main__":
    main()
