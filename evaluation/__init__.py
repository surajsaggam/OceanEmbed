"""
evaluation package for OceanEmbed.
"""

from evaluation.ablations import AblationTracker
from evaluation.argo_eval import check_argo_guard, run_blind_argo_evaluation
from evaluation.glorys_eval import GlorysEvaluator
from evaluation.metrics import (
    compute_all_depth_metrics,
    depth_wise_bias,
    depth_wise_mae,
    depth_wise_r_and_r2,
    depth_wise_rmse,
)

__all__ = [
    "depth_wise_rmse",
    "depth_wise_mae",
    "depth_wise_bias",
    "depth_wise_r_and_r2",
    "compute_all_depth_metrics",
    "check_argo_guard",
    "run_blind_argo_evaluation",
    "GlorysEvaluator",
    "AblationTracker",
]
