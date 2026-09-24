"""Configuration loading and repository paths."""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("traffictrak")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_CONFIG = REPO_ROOT / "config" / "pipeline.yaml"
DEFAULT_GEOMETRY_CONFIG = REPO_ROOT / "config" / "camera_geometry.yaml"


def weights_dir() -> Path:
    """Directory holding model weights (override with TRAFFICTRAK_WEIGHTS_DIR)."""
    env = os.environ.get("TRAFFICTRAK_WEIGHTS_DIR")
    return Path(env) if env else REPO_ROOT / "weights"


def resolve_path(p: str | os.PathLike | None) -> Path | None:
    """Resolve a config path relative to the repository root."""
    if p is None or str(p) == "":
        return None
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


def pipeline_config_path() -> Path:
    env = os.environ.get("TRAFFICTRAK_CONFIG")
    return Path(env) if env else DEFAULT_PIPELINE_CONFIG


def geometry_config_path() -> Path:
    env = os.environ.get("TRAFFICTRAK_GEOMETRY")
    return Path(env) if env else DEFAULT_GEOMETRY_CONFIG


def deep_update(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


_CONFIG_CACHE: dict[str, dict] = {}


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict[str, Any]:
    """Load the pipeline YAML (cached) and apply optional overrides."""
    cfg_path = Path(path) if path else pipeline_config_path()
    key = str(cfg_path.resolve())
    if key not in _CONFIG_CACHE:
        with open(cfg_path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            raise ValueError(f"{cfg_path}: top level must be a mapping")
        _CONFIG_CACHE[key] = data
    cfg = copy.deepcopy(_CONFIG_CACHE[key])
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def class_cfg(cfg: dict, name: str) -> dict:
    return (cfg.get("classes") or {}).get(name) or {}


def class_enabled(cfg: dict, name: str) -> bool:
    return bool(class_cfg(cfg, name).get("enabled", False))
