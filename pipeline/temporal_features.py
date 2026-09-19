"""
pipeline/temporal_features.py
-----------------------------
Phase-2 Temporal Trend Feature Engineering: Delta-SST and Delta-SSH/SLA.

Scientific Rules:
1. Feature Definitions:
     Delta_SST(t) = SST(t) - SST(t-1)
     Delta_SSH(t) = SSH(t) - SSH(t-1)
   These are dynamic trend features giving the model temporal rate-of-change context.
   They are NOT hand-built detectors of named physical phenomena.

2. Validity and Masking:
     M_delta_SST = M_SST(t) * M_SST(t-1)
     M_delta_SSH = M_SSH(t) * M_SSH(t-1)
   A temporal difference is valid if and only if observations at BOTH day t and day t-1
   are valid ocean pixels.
   If either day is missing, land-masked, cloud-gapped, or unobserved:
     - M_delta is set to 0.0.
     - The normalized feature value is set to 0.0 (normalized mean fill).
     - An invalid delta is NEVER silently interpreted as a physically observed zero change.

3. Boundary Handling:
   - 2015-01-01 (sequence start): t-1 is unavailable -> M_delta = 0.0, fill = 0.0.
   - 2018-01-01 (val start): t-1 is 2017-12-31 (train end) -> strictly historical past data.
   - 2019-01-01 (test start): t-1 is 2018-12-31 (val end) -> strictly historical past data.
   - Never use t+1 or any future observation (strict temporal causality).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch


def compute_temporal_deltas(
    sst_t: np.ndarray,
    mask_sst_t: np.ndarray,
    ssh_t: np.ndarray,
    mask_ssh_t: np.ndarray,
    sst_t_minus_1: Optional[np.ndarray] = None,
    mask_sst_t_minus_1: Optional[np.ndarray] = None,
    ssh_t_minus_1: Optional[np.ndarray] = None,
    mask_ssh_t_minus_1: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes physical Delta_SST and Delta_SSH along with joint validity masks.

    Args:
        sst_t: Sea Surface Temperature at day t [H, W] in deg C.
        mask_sst_t: SST validity mask at day t [H, W] (1.0 = valid ocean, 0.0 = land/invalid).
        ssh_t: Sea Surface Height / SLA at day t [H, W] in m.
        mask_ssh_t: SSH validity mask at day t [H, W] (1.0 = valid ocean, 0.0 = land/invalid).
        sst_t_minus_1: Optional SST at day t-1 [H, W] in deg C.
        mask_sst_t_minus_1: Optional SST validity mask at day t-1 [H, W].
        ssh_t_minus_1: Optional SSH at day t-1 [H, W] in m.
        mask_ssh_t_minus_1: Optional SSH validity mask at day t-1 [H, W].

    Returns:
        delta_sst: [H, W] array of raw physical SST differences.
        mask_delta_sst: [H, W] binary mask where 1.0 indicates a valid temporal difference.
        delta_ssh: [H, W] array of raw physical SSH differences.
        mask_delta_ssh: [H, W] binary mask where 1.0 indicates a valid temporal difference.
    """
    H, W = sst_t.shape

    # Boundary case: previous day is completely unavailable (e.g. 2015-01-01)
    if (
        sst_t_minus_1 is None
        or mask_sst_t_minus_1 is None
        or ssh_t_minus_1 is None
        or mask_ssh_t_minus_1 is None
    ):
        delta_sst = np.zeros((H, W), dtype=np.float32)
        mask_delta_sst = np.zeros((H, W), dtype=np.float32)
        delta_ssh = np.zeros((H, W), dtype=np.float32)
        mask_delta_ssh = np.zeros((H, W), dtype=np.float32)
        return delta_sst, mask_delta_sst, delta_ssh, mask_delta_ssh

    # Compute raw differences
    raw_delta_sst = (sst_t - sst_t_minus_1).astype(np.float32)
    raw_delta_ssh = (ssh_t - ssh_t_minus_1).astype(np.float32)

    # Compute joint validity masks: valid iff BOTH days are valid and difference is finite
    m_delta_sst = (
        (mask_sst_t == 1.0)
        & (mask_sst_t_minus_1 == 1.0)
        & np.isfinite(raw_delta_sst)
    ).astype(np.float32)

    m_delta_ssh = (
        (mask_ssh_t == 1.0)
        & (mask_ssh_t_minus_1 == 1.0)
        & np.isfinite(raw_delta_ssh)
    ).astype(np.float32)

    # Where invalid, zero out raw values to guarantee finite arrays
    raw_delta_sst[m_delta_sst == 0.0] = 0.0
    raw_delta_ssh[m_delta_ssh == 0.0] = 0.0

    return raw_delta_sst, m_delta_sst, raw_delta_ssh, m_delta_ssh


def normalize_temporal_deltas(
    delta_sst: np.ndarray,
    mask_delta_sst: np.ndarray,
    delta_ssh: np.ndarray,
    mask_delta_ssh: np.ndarray,
    norm_stats: Dict[str, Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies frozen z-score normalization to Delta_SST and Delta_SSH.

    Scientific Rule:
      Where mask == 1.0: z = (delta - mean) / std
      Where mask == 0.0: z = 0.0 (normalized mean fill)
    """
    stats_sst = norm_stats.get("delta_SST", {"mean": 0.0, "std": 1.0})
    stats_ssh = norm_stats.get("delta_SSH", {"mean": 0.0, "std": 1.0})

    mu_sst, sigma_sst = float(stats_sst["mean"]), float(stats_sst["std"])
    mu_ssh, sigma_ssh = float(stats_ssh["mean"]), float(stats_ssh["std"])

    if sigma_sst < 1e-6:
        sigma_sst = 1.0
    if sigma_ssh < 1e-6:
        sigma_ssh = 1.0

    # Normalize valid pixels; invalid pixels receive 0.0 fill
    norm_sst = np.zeros_like(delta_sst, dtype=np.float32)
    valid_sst = (mask_delta_sst == 1.0) & np.isfinite(delta_sst)
    norm_sst[valid_sst] = (delta_sst[valid_sst] - mu_sst) / sigma_sst

    norm_ssh = np.zeros_like(delta_ssh, dtype=np.float32)
    valid_ssh = (mask_delta_ssh == 1.0) & np.isfinite(delta_ssh)
    norm_ssh[valid_ssh] = (delta_ssh[valid_ssh] - mu_ssh) / sigma_ssh

    return norm_sst, norm_ssh
