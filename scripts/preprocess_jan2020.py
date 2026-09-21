"""
scripts/preprocess_jan2020.py
------------------------------
Runs batch preprocessing for January 2020 dry run.
Outputs normalized [14, 101, 241] input tensors and [15, 101, 241] target tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preprocess import ProductionPreprocessor


def main() -> None:
    print("=" * 70)
    print("OceanEmbed — Batch Preprocessing for January 2020 Dry Run")
    print("=" * 70)

    preprocessor = ProductionPreprocessor(
        raw_dir="data/raw",
        processed_dir="data/processed",
        norm_stats_path="data/norm_stats/train_stats.json",
    )

    counts = preprocessor.run_preprocessing(
        train_days=21,
        val_days=5,
        test_days=5,
    )

    print("\nSummary of Generated Samples:")
    print(f"  Train: {counts['train']} days (2020-01-01 to 2020-01-21)")
    print(f"  Val:   {counts['val']} days (2020-01-22 to 2020-01-26)")
    print(f"  Test:  {counts['test']} days (2020-01-27 to 2020-01-31)")
    print("=" * 70)


if __name__ == "__main__":
    main()
