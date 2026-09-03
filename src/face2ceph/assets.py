"""Verified retrieval of declared third-party model assets."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from .workspace import GENERATED_ROOT, output_path


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(filename: str, expected_sha256: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError("asset filename must not contain a directory")
    path = GENERATED_ROOT / "assets" / filename
    if not path.is_file():
        raise FileNotFoundError(f"required asset is missing: {filename}")
    if sha256(path) != expected_sha256.lower():
        raise ValueError(f"checksum mismatch: {filename}")
    return path


def fetch(url: str, filename: str, expected_sha256: str) -> Path:
    expected_sha256 = expected_sha256.lower()
    if Path(filename).name != filename:
        raise ValueError("asset filename must not contain a directory")
    destination = GENERATED_ROOT / "assets" / filename
    if destination.exists():
        actual = sha256(destination)
        if actual != expected_sha256:
            raise ValueError(f"checksum mismatch: {destination}")
        return destination
    destination = output_path(Path("assets") / filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with urllib.request.urlopen(url, timeout=120) as response, destination.open("xb") as stream:
            created = True
            while chunk := response.read(1 << 20):
                stream.write(chunk)
        actual = sha256(destination)
        if actual != expected_sha256:
            raise ValueError(f"checksum mismatch for {filename}")
    except BaseException:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return destination
