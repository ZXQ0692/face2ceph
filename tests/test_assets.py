import hashlib
import io
from pathlib import Path

import pytest

from face2ceph import assets


def _local_output(root: Path, value: str | Path) -> Path:
    return root / Path(value)


def test_failed_concurrent_create_preserves_existing_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "generated"
    destination = root / "assets" / "model.bin"
    content = b"verified"

    class ConcurrentResponse(io.BytesIO):
        def __enter__(self) -> "ConcurrentResponse":
            destination.write_bytes(content)
            return super().__enter__()

    monkeypatch.setattr(assets, "GENERATED_ROOT", root)
    monkeypatch.setattr(assets, "output_path", lambda value: _local_output(root, value))
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *args, **kwargs: ConcurrentResponse())

    with pytest.raises(FileExistsError):
        assets.fetch("https://example.invalid/model", "model.bin", hashlib.sha256(content).hexdigest())

    assert destination.read_bytes() == content


def test_failed_download_removes_its_own_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "generated"
    destination = root / "assets" / "model.bin"
    monkeypatch.setattr(assets, "GENERATED_ROOT", root)
    monkeypatch.setattr(assets, "output_path", lambda value: _local_output(root, value))
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(b"invalid"))

    with pytest.raises(ValueError, match="checksum mismatch"):
        assets.fetch("https://example.invalid/model", "model.bin", hashlib.sha256(b"valid").hexdigest())

    assert not destination.exists()
