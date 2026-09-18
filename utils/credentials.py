"""
utils/credentials.py
--------------------
Safe credential loading and validation for Copernicus Marine and NASA Earthdata.

SECURITY RULES:
  - NEVER log, print, or expose usernames, passwords, or tokens.
  - Automatically loads from .env in the project root if present (strictly gitignored).
  - Also checks system environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"


def load_env(env_path: Optional[Path] = None) -> Dict[str, str]:
    """
    Loads key-value pairs from a .env file into os.environ.
    Does not overwrite existing environment variables.
    """
    path = env_path or ENV_FILE
    loaded = {}
    if not path.exists():
        return loaded

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
                    loaded[key] = "***"  # Obscured value for safe status reporting
    return loaded


def get_copernicus_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (username, password) for Copernicus Marine from environment.
    """
    load_env()
    user = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") or os.environ.get("CMEMS_USERNAME")
    pwd = os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD") or os.environ.get("CMEMS_PASSWORD")
    return user, pwd


def get_earthdata_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (username, password) for NASA Earthdata from environment.
    """
    load_env()
    user = os.environ.get("EARTHDATA_USERNAME")
    pwd = os.environ.get("EARTHDATA_PASSWORD")
    return user, pwd


def check_credentials_status() -> Dict[str, bool]:
    """
    Checks whether credentials are configured, returning a boolean status dictionary.
    Does NOT return or print any secret strings.
    """
    cm_user, cm_pwd = get_copernicus_credentials()
    ed_user, ed_pwd = get_earthdata_credentials()

    return {
        "copernicus_configured": bool(cm_user and cm_pwd),
        "earthdata_configured": bool(ed_user and ed_pwd),
        "env_file_exists": ENV_FILE.exists(),
    }
