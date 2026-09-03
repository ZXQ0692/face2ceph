"""Reject generated, sensitive, local, or unverifiable release artifacts."""

from __future__ import annotations

import hashlib
import ast
import json
import re
import sys
import tomllib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".cff", ".csv", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
FORBIDDEN_SUFFIXES = {
    ".bmp",
    ".ckpt",
    ".dcm",
    ".dicom",
    ".docx",
    ".feather",
    ".gz",
    ".jpeg",
    ".jpg",
    ".log",
    ".nii",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".pyc",
    ".pyo",
    ".safetensors",
    ".tif",
    ".tiff",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}
FORBIDDEN_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "authorized_data",
    "checkpoints",
    "controlled_data",
    "images",
    "models",
    "photos",
    "raw_data",
    "raw_images",
    "restricted_data",
    "weights",
}
FORBIDDEN_FILENAMES = {
    ".env",
    "cohort.csv",
    "credentials.json",
    "dicomdir",
    "measurements.csv",
    "operator_experience.json",
    "partition.csv",
    "frozen_partition.csv",
    "split.csv",
}
FORBIDDEN_NAME_FRAGMENT = re.compile(
    r"(?:code[_-]?map|crosswalk|id[_-]?map|linkage)", re.IGNORECASE
)
LOCAL_PATH = re.compile(r"(?:[A-Za-z]:(?:\\|/(?!/))|/(?:home|Users)/)")
EAST_ASIAN_OR_FULLWIDTH_TEXT = re.compile(
    "["
    "\u1100-\u11ff"
    "\u2e80-\u2fdf"
    "\u3000-\u303f"
    "\u3040-\u30ff"
    "\u3100-\u312f"
    "\u3130-\u318f"
    "\u31a0-\u31ef"
    "\u3400-\u4dbf"
    "\u4e00-\u9fff"
    "\ua960-\ua97f"
    "\uac00-\ud7af"
    "\ud7b0-\ud7ff"
    "\uf900-\ufaff"
    "\ufe10-\ufe1f"
    "\ufe30-\ufe4f"
    "\uff00-\uffef"
    "\U00020000-\U0002fa1f"
    "\U00030000-\U000323af"
    "]"
)
FORBIDDEN_IMPORTS = {"logging", "mlflow", "tensorboard", "wandb"}
REQUIRED_GITIGNORE_RULES = {
    "*partition.csv",
    "split.csv",
    "**/*[Cc]rosswalk*",
    "**/*id[_-]map*",
    "**/*[Ll]inkage*",
    "**/*code[_-]map*",
}


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _files() -> list[Path]:
    return sorted(path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.parts)


def _manifest_errors() -> list[str]:
    reference = ROOT / "reference"
    manifest = reference / "SHA256SUMS"
    errors: list[str] = []
    declared: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = (reference / name).resolve(strict=False)
        try:
            path.relative_to(reference.resolve())
        except ValueError:
            errors.append(f"reference manifest escapes its directory: {name}")
            continue
        declared.add(name)
        if not path.is_file():
            errors.append(f"reference file is missing: {name}")
            continue
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            errors.append(f"reference checksum mismatch: {name}")
    covered: set[str] = set()
    for path in reference.rglob("*"):
        relative = path.relative_to(reference).as_posix()
        if path.is_file() and relative not in {"README.md", "SHA256SUMS"}:
            covered.add(relative)
    missing = sorted(covered - declared)
    if missing:
        errors.append(f"reference files are absent from SHA256SUMS: {missing}")
    return errors


def _gitattributes_errors() -> list[str]:
    path = ROOT / ".gitattributes"
    if not path.is_file():
        return []
    rules = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if rules != ["reference/** -text"]:
        return [".gitattributes must contain only: reference/** -text"]
    return []


def _gitignore_errors() -> list[str]:
    path = ROOT / ".gitignore"
    if not path.is_file():
        return []
    rules = {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing = sorted(REQUIRED_GITIGNORE_RULES - rules)
    return [f".gitignore is missing controlled-file rules: {missing}"] if missing else []


def _uses_logging(text: str) -> bool:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            alias.name.split(".", 1)[0] in FORBIDDEN_IMPORTS for alias in node.names
        ):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".", 1)[0] in FORBIDDEN_IMPORTS:
            return True
    return False


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def audit() -> list[str]:
    errors = _manifest_errors() + _gitattributes_errors() + _gitignore_errors()
    for directory in ROOT.rglob("*"):
        if directory.is_dir() and directory.name in FORBIDDEN_DIRECTORIES:
            errors.append(f"cache directory is present: {_relative(directory)}")
    for path in _files():
        relative = _relative(path)
        if relative.startswith("generated/") and relative != "generated/.gitignore":
            errors.append(f"generated output is present: {relative}")
        suffix = path.suffix.lower()
        if (
            suffix in FORBIDDEN_SUFFIXES
            or path.name.lower() in FORBIDDEN_FILENAMES
            or FORBIDDEN_NAME_FRAGMENT.search(path.stem)
        ):
            errors.append(f"forbidden release artifact is present: {relative}")
        if suffix in TEXT_SUFFIXES or path.name in {".gitattributes", ".gitignore"}:
            text = path.read_text(encoding="utf-8-sig")
            if LOCAL_PATH.search(text):
                errors.append(f"local absolute path is present: {relative}")
            if EAST_ASIAN_OR_FULLWIDTH_TEXT.search(text):
                errors.append(f"non-English text is present: {relative}")
            if suffix == ".py" and _uses_logging(text):
                errors.append(f"logging or telemetry code is present: {relative}")
            if suffix == ".json":
                json.loads(text, parse_constant=_reject_constant)
            elif suffix in {".yaml", ".yml"}:
                yaml.safe_load(text)
            elif suffix == ".toml":
                tomllib.loads(text)
    required = {
        ".gitattributes",
        ".gitignore",
        "README.md",
        "CITATION.cff",
        "LICENSE",
        "DATA_LICENSE.md",
        "THIRD_PARTY_NOTICES.md",
    }
    missing = sorted(name for name in required if not (ROOT / name).is_file())
    if missing:
        errors.append(f"required release files are missing: {missing}")
    return sorted(set(errors))


def main() -> int:
    try:
        errors = audit()
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1
    print("Release audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
