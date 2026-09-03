"""Checksum verification for immutable release assets."""

from __future__ import annotations

import hashlib
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: str | Path, manifest: str = "SHA256SUMS") -> list[str]:
    directory = Path(root).resolve(strict=True)
    errors: list[str] = []
    for line in (directory / manifest).read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        path = (directory / name).resolve(strict=False)
        try:
            path.relative_to(directory)
        except ValueError:
            errors.append(f"invalid manifest path: {name}")
            continue
        if not path.is_file():
            errors.append(f"missing: {name}")
        elif file_sha256(path) != expected:
            errors.append(f"checksum mismatch: {name}")
    return errors
