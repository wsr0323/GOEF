"""Minimal YAML configuration loader with ``$VAR`` expansion."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def load_config(path: Path | str | None) -> dict:
    """Read a YAML file and expand environment variables in every string."""
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as error:  # pragma: no cover
        raise SystemExit("pyyaml is required to read a config file") from error
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: expected a mapping at the top level")
    return _expand(data)


def expand_path(value: Any) -> Any:
    if isinstance(value, (str, Path)):
        return Path(os.path.expandvars(str(value))).expanduser()
    return value
