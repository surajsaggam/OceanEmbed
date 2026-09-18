"""
scripts/verify_gpu.py
---------------------
GPU / CUDA / AMP verification sequence.

Run this immediately after PyTorch installation:
    python scripts/verify_gpu.py

All checks must pass before any training attempt.
Results are saved to environment/env_verification.json (committed to git).
The precision field in configs/train.yaml is then updated based on the results.

Usage
-----
    python scripts/verify_gpu.py
    python scripts/verify_gpu.py --update-config   # also patches train.yaml
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_verification(update_config: bool = False) -> dict:
    results = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "checks": {},
    }

    # ── Check 1: PyTorch importable ───────────────────────────────────────
    try:
        import torch
        results["torch_version"] = torch.__version__
        results["checks"]["torch_import"] = "PASS"
        print(f"[1] PyTorch import: PASS  (version: {torch.__version__})")
    except ImportError as e:
        results["checks"]["torch_import"] = f"FAIL: {e}"
        print(f"[1] PyTorch import: FAIL  ({e})")
        print("\nInstall PyTorch before running this script.")
        print("See README.md for the correct CUDA 12.8 wheel command.")
        _save_results(results)
        sys.exit(1)

    # ── Check 2: CUDA availability ────────────────────────────────────────
    cuda_available = torch.cuda.is_available()
    results["cuda_available"] = cuda_available
    results["checks"]["cuda_available"] = "PASS" if cuda_available else "FAIL"
    print(f"[2] CUDA available:   {'PASS' if cuda_available else 'FAIL'}")

    if not cuda_available:
        results["cuda_version"] = None
        results["gpu_name"] = None
        results["checks"]["gpu_tensor_op"] = "SKIP (no CUDA)"
        results["checks"]["bf16_amp"] = "SKIP (no CUDA)"
        results["checks"]["fp16_amp"] = "SKIP (no CUDA)"
        results["vram_total_gb"] = None
        results["vram_free_gb"] = None
        _save_results(results)
        print("\n[WARNING] CUDA not available. PyTorch cannot use the GPU.")
        print("Possible causes:")
        print("  - Wrong PyTorch wheel (need cu128 for RTX 5060 Blackwell)")
        print("  - Driver issue")
        print("  - See: https://pytorch.org/get-started/locally/")
        sys.exit(1)

    # ── Check 3: GPU identity and CUDA version ───────────────────────────
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    results["gpu_name"] = gpu_name
    results["cuda_version"] = cuda_version
    results["checks"]["gpu_identity"] = "PASS"
    print(f"[3] GPU name:         {gpu_name}")
    print(f"    CUDA version:      {cuda_version}")

    # ── Check 4: Basic CUDA tensor operation ─────────────────────────────
    try:
        x = torch.randn(4, 14, 32, 32, device="cuda")
        y = x * 2.0 + 1.0
        assert y.device.type == "cuda"
        assert y.shape == x.shape
        del x, y
        torch.cuda.empty_cache()
        results["checks"]["gpu_tensor_op"] = "PASS"
        print("[4] Basic CUDA tensor op: PASS")
    except Exception as e:
        results["checks"]["gpu_tensor_op"] = f"FAIL: {e}"
        print(f"[4] Basic CUDA tensor op: FAIL  ({e})")

    # ── Check 5: bf16 AMP support ─────────────────────────────────────────
    bf16_ok = False
    try:
        x = torch.randn(4, 14, 32, 32, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            z = torch.matmul(x, x.transpose(-1, -2))
        assert z.dtype == torch.bfloat16
        del x, z
        torch.cuda.empty_cache()
        bf16_ok = True
        results["checks"]["bf16_amp"] = "PASS"
        print("[5] bf16 AMP:         PASS")
    except Exception as e:
        results["checks"]["bf16_amp"] = f"FAIL: {e}"
        print(f"[5] bf16 AMP:         FAIL  ({e})")

    # ── Check 6: fp16 AMP support ─────────────────────────────────────────
    fp16_ok = False
    try:
        x = torch.randn(4, 14, 32, 32, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            z = torch.matmul(x, x.transpose(-1, -2))
        assert z.dtype == torch.float16
        del x, z
        torch.cuda.empty_cache()
        fp16_ok = True
        results["checks"]["fp16_amp"] = "PASS"
        print("[6] fp16 AMP:         PASS")
    except Exception as e:
        results["checks"]["fp16_amp"] = f"FAIL: {e}"
        print(f"[6] fp16 AMP:         FAIL  ({e})")

    # ── Check 7: VRAM measurement ─────────────────────────────────────────
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_vram = torch.cuda.mem_get_info()[0] / 1e9
    results["vram_total_gb"] = round(total_vram, 2)
    results["vram_free_gb"] = round(free_vram, 2)
    results["checks"]["vram_measured"] = "PASS"
    print(f"[7] VRAM total:       {total_vram:.2f} GB")
    print(f"    VRAM free:        {free_vram:.2f} GB")

    # ── Recommended precision ─────────────────────────────────────────────
    if bf16_ok:
        recommended_precision = "bf16"
    elif fp16_ok:
        recommended_precision = "fp16"
    else:
        recommended_precision = "fp32"
    results["recommended_precision"] = recommended_precision
    print(f"\nRecommended precision for train.yaml: {recommended_precision}")

    # ── Update train.yaml if requested ───────────────────────────────────
    if update_config:
        _update_train_precision(recommended_precision)

    _save_results(results)

    # ── Summary ───────────────────────────────────────────────────────────
    all_critical = all(
        "PASS" in str(results["checks"].get(k, ""))
        for k in ["torch_import", "cuda_available", "gpu_tensor_op"]
    )

    print("\n" + "=" * 60)
    if all_critical:
        print("GPU VERIFICATION: ALL CRITICAL CHECKS PASSED")
        print(f"Results saved to: environment/env_verification.json")
        print(f"\nNext step: update configs/train.yaml")
        print(f"  precision: \"{recommended_precision}\"")
        if not update_config:
            print("  (or re-run with --update-config to do this automatically)")
    else:
        print("GPU VERIFICATION: ONE OR MORE CRITICAL CHECKS FAILED")
        print("Do not start training until CUDA is working.")
    print("=" * 60)

    return results


def _save_results(results: dict) -> None:
    out_path = PROJECT_ROOT / "environment" / "env_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def _update_train_precision(precision: str) -> None:
    import re
    train_yaml_path = PROJECT_ROOT / "configs" / "train.yaml"
    with open(train_yaml_path, "r", encoding="utf-8") as f:
        content = f.read()
    updated = re.sub(
        r'^(precision:\s*)".+"',
        f'precision: "{precision}"',
        content,
        flags=re.MULTILINE,
    )
    with open(train_yaml_path, "w", encoding="utf-8") as f:
        f.write(updated)
    print(f"  configs/train.yaml updated: precision = \"{precision}\"")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed GPU/CUDA/AMP verification")
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Automatically update configs/train.yaml with the recommended precision",
    )
    args = parser.parse_args()
    run_verification(update_config=args.update_config)
