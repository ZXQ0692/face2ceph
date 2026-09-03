"""Read-only input resolution and isolated non-overwriting outputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

WORKSPACE_ENVIRONMENT_VARIABLE = "FACE2CEPH_WORKSPACE"


def _marked_workspace(start: Path) -> Path | None:
    current = start.resolve(strict=False)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "pipeline.yaml").is_file() and (
            candidate / "reference" / "SHA256SUMS"
        ).is_file():
            return candidate
    return None


def _resolve_workspace(
    environment: Mapping[str, str] | None = None,
    working_directory: str | os.PathLike[str] | None = None,
    source_file: str | os.PathLike[str] | None = None,
) -> Path:
    values = os.environ if environment is None else environment
    configured = values.get(WORKSPACE_ENVIRONMENT_VARIABLE, "").strip()
    if configured:
        workspace = Path(configured).expanduser().resolve(strict=True)
        if not workspace.is_dir():
            raise NotADirectoryError(f"workspace is not a directory: {workspace}")
        if _marked_workspace(workspace) != workspace:
            raise ValueError("workspace must contain configs/pipeline.yaml and reference/SHA256SUMS")
        return workspace

    source = Path(__file__ if source_file is None else source_file).expanduser().resolve(strict=False)
    source_workspace = _marked_workspace(source.parent)
    if source_workspace is not None:
        return source_workspace

    current = Path.cwd() if working_directory is None else Path(working_directory)
    current = current.expanduser().resolve(strict=True)
    current_workspace = _marked_workspace(current)
    if current_workspace is None:
        raise FileNotFoundError(
            f"No release workspace found; set {WORKSPACE_ENVIRONMENT_VARIABLE} to the release directory"
        )
    return current_workspace


RELEASE_ROOT = _resolve_workspace()
GENERATED_ROOT = (RELEASE_ROOT / "generated").resolve(strict=False)


def input_path(value: str | os.PathLike[str], kind: str | None = None) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if kind == "file" and not path.is_file():
        raise ValueError(f"expected a file: {path}")
    if kind == "dir" and not path.is_dir():
        raise ValueError(f"expected a directory: {path}")
    return path


def generated_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = GENERATED_ROOT / path
    path = path.resolve(strict=False)
    try:
        path.relative_to(GENERATED_ROOT)
    except ValueError as exc:
        raise ValueError(f"outputs must be inside {GENERATED_ROOT}") from exc
    return path


def output_path(value: str | os.PathLike[str]) -> Path:
    path = generated_path(value)
    if path.exists():
        raise FileExistsError(f"destination already exists: {path}")
    return path


def create_directory(value: str | os.PathLike[str]) -> Path:
    path = output_path(value)
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(value: str | os.PathLike[str], payload: Any) -> Path:
    path = output_path(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return path
