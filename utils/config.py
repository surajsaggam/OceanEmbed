"""
utils/config.py
---------------
Configuration loader for OceanEmbed.

All Python modules and notebooks import configs via:
    from utils.config import load_config

This is the single path for reading YAML configs.
No module should call yaml.safe_load() directly on a config path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


# Project root = directory containing this utils/ package
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs"


class Config:
    """
    Lightweight config object wrapping a dict.

    Supports attribute-style access (cfg.batch_size) as well as
    dict-style access (cfg['batch_size']) for compatibility.
    Nested dicts are also wrapped recursively.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
        self._data = data

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return the raw dict (for serialisation into run manifests)."""
        return self._data

    def __repr__(self) -> str:
        keys = list(self._data.keys())
        return f"Config(keys={keys})"


def load_config(config_name_or_path: str | Path) -> Config:
    """
    Load a YAML config file and return a Config object.

    Parameters
    ----------
    config_name_or_path : str or Path
        Either a bare name (e.g. "train") which resolves to
        configs/train.yaml relative to the project root, or
        an absolute/relative path to a YAML file.

    Returns
    -------
    Config
        Config object with attribute-style access.

    Examples
    --------
    >>> cfg = load_config("train")
    >>> cfg.batch_size
    4

    >>> cfg = load_config("configs/data.yaml")
    >>> cfg.domain.lat_min
    5.0
    """
    path = Path(config_name_or_path)

    # If it's a bare name (no suffix, no separator), resolve to configs/
    if not path.suffix and not path.is_absolute() and os.sep not in str(path) and "/" not in str(path):
        path = _CONFIGS_DIR / f"{path}.yaml"
    elif not path.is_absolute():
        # Relative path — resolve from project root
        path = _PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            f"Project root: {_PROJECT_ROOT}\n"
            f"Configs directory: {_CONFIGS_DIR}"
        )

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} did not parse to a dict. Got: {type(raw)}")

    return Config(raw)


def load_all_configs() -> dict[str, Config]:
    """
    Load all four standard configs and return them as a dict.

    Returns
    -------
    dict with keys: "data", "model", "train", "eval"
    """
    return {
        "data":  load_config("data"),
        "model": load_config("model"),
        "train": load_config("train"),
        "eval":  load_config("eval"),
    }
