"""
tests/test_credentials_and_inspection.py
----------------------------------------
Tests for safe credential loading, Git security, and raw inspection helpers.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.credentials import check_credentials_status, load_env


class TestCredentialsSecurity:
    def test_credential_status_returns_booleans(self):
        status = check_credentials_status()
        assert isinstance(status, dict)
        assert "copernicus_configured" in status
        assert "earthdata_configured" in status
        assert "env_file_exists" in status
        assert isinstance(status["copernicus_configured"], bool)
        assert isinstance(status["earthdata_configured"], bool)

    def test_dotenv_is_gitignored(self):
        """Verifies that Git actively ignores .env, .netrc, and credential files."""
        cmd = ["git", "check-ignore", ".env", ".env.local", ".netrc", "_netrc", "credentials.json"]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        assert result.returncode == 0, "All credential files must be ignored by Git"
        ignored_files = result.stdout.strip().splitlines()
        assert len(ignored_files) >= 5, f"Expected 5 ignored files, got {ignored_files}"
