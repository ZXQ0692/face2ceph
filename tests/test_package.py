from pathlib import Path

import pytest

from face2ceph.configuration import merge
from face2ceph.integrity import verify_manifest
from face2ceph.targets import age_band, age_stratum
from face2ceph.workspace import GENERATED_ROOT, RELEASE_ROOT, _resolve_workspace, output_path


def _mark_workspace(path: Path) -> None:
    (path / "configs").mkdir(parents=True)
    (path / "configs" / "pipeline.yaml").touch()
    (path / "reference").mkdir()
    (path / "reference" / "SHA256SUMS").touch()


def test_age_groups() -> None:
    assert age_band(7) == "7-9"
    assert age_band(18) == ">=18"
    assert age_stratum(10) == "7-10"
    assert age_stratum(30) == "11-30"
    assert age_stratum(31) == ">30"


def test_recursive_configuration_merge() -> None:
    base = {"model": {"width": 1, "dropout": 0.2}, "seed": 42}
    overlay = {"model": {"width": 2}}
    assert merge(base, overlay) == {
        "model": {"width": 2, "dropout": 0.2},
        "seed": 42,
    }
    assert base["model"]["width"] == 1


def test_outputs_are_confined() -> None:
    assert GENERATED_ROOT == RELEASE_ROOT / "generated"
    assert output_path("unit/result.json") == GENERATED_ROOT / "unit" / "result.json"
    with pytest.raises(ValueError):
        output_path(Path("..") / "outside.json")


def test_workspace_environment_has_priority(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    _mark_workspace(configured)
    discovered = tmp_path / "discovered"
    _mark_workspace(discovered)
    assert _resolve_workspace(
        {"FACE2CEPH_WORKSPACE": str(configured)},
        working_directory=discovered,
        source_file=discovered / "src" / "face2ceph" / "workspace.py",
    ) == configured.resolve()


def test_workspace_is_discovered_from_source_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    _mark_workspace(checkout)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    assert _resolve_workspace(
        {},
        working_directory=elsewhere,
        source_file=checkout / "src" / "face2ceph" / "workspace.py",
    ) == checkout.resolve()


def test_installed_package_uses_marked_working_tree(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _mark_workspace(workspace)
    nested = workspace / "run" / "nested"
    nested.mkdir(parents=True)
    installed = tmp_path / "venv" / "Lib" / "site-packages" / "face2ceph" / "workspace.py"
    assert _resolve_workspace({}, working_directory=nested, source_file=installed) == workspace.resolve()


def test_installed_package_requires_an_explicit_workspace(tmp_path: Path) -> None:
    working = tmp_path / "run"
    working.mkdir()
    installed = tmp_path / "venv" / "Lib" / "site-packages" / "face2ceph" / "workspace.py"
    with pytest.raises(FileNotFoundError, match="FACE2CEPH_WORKSPACE"):
        _resolve_workspace({}, working_directory=working, source_file=installed)


def test_reference_manifest() -> None:
    root = Path(__file__).resolve().parents[1] / "reference"
    assert verify_manifest(root) == []
