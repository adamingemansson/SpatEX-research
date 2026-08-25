"""Load YAML configurations with recursive inheritance."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge an override without mutating either input."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Resolve a configuration and any parent named by ``inherits``."""
    path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if path in seen:
        raise ValueError(f"cyclic config inheritance involving {path}")
    seen.add(path)
    payload = yaml.safe_load(path.read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: configuration must be a mapping")
    parent = payload.pop("inherits", None)
    if parent is None:
        return payload
    parent_path = Path(parent)
    if not parent_path.is_absolute():
        parent_path = path.parent / parent_path
    return _merge(load_config(parent_path, seen), payload)
