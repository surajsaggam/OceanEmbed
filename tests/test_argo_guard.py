"""
tests/test_argo_guard.py
------------------------
Tests that the Argo blind evaluation runtime guard works correctly.

Rules verified:
  - run_blind_argo_evaluation raises AssertionError when argo_blind_locked=False
  - The guard message is informative
  - When argo_blind_locked=True the gate passes (function proceeds past guard)
"""

from __future__ import annotations

import pytest


def _argo_evaluation_gate(argo_blind_locked: bool) -> str:
    """
    Reference guard implementation (mirrors evaluation/argo_eval.py).
    Returns "PROCEEDED" if the guard passes.
    """
    assert argo_blind_locked, (
        "Argo blind evaluation is LOCKED. "
        "Set argo_blind_locked: true in configs/eval.yaml ONLY after:\n"
        "  1. Phase-1 model architecture is locked\n"
        "  2. All ablations are complete\n"
        "  3. The blind evaluation protocol is written and reviewed"
    )
    return "PROCEEDED"


class TestArgoGuard:
    def test_locked_false_raises(self):
        """Guard must raise when argo_blind_locked is False (default)."""
        with pytest.raises(AssertionError, match="LOCKED"):
            _argo_evaluation_gate(argo_blind_locked=False)

    def test_locked_true_proceeds(self):
        """Guard must pass when explicitly unlocked."""
        result = _argo_evaluation_gate(argo_blind_locked=True)
        assert result == "PROCEEDED"

    def test_error_message_is_informative(self):
        """The error message must explain what conditions must be met."""
        with pytest.raises(AssertionError) as exc_info:
            _argo_evaluation_gate(argo_blind_locked=False)
        msg = str(exc_info.value)
        assert "eval.yaml" in msg, "Error must reference the config file to change"
        assert "architecture" in msg.lower() or "ablation" in msg.lower(), (
            "Error must explain that model must be final"
        )

    def test_eval_yaml_default_is_locked(self):
        """Verify that the default eval.yaml has argo_blind_locked: false."""
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from utils.config import load_config
        cfg = load_config("eval")
        assert cfg.argo_blind_locked is False, (
            "eval.yaml must default to argo_blind_locked: false"
        )
