"""Strict loading and merging of release configuration files."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load(path: str | Path, base: str | Path | None = None) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        overlay = yaml.safe_load(stream) or {}
    if base is None:
        return overlay
    with Path(base).open(encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}
    return merge(root, overlay)
