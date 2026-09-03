from importlib.util import module_from_spec, spec_from_file_location
import hashlib
from pathlib import Path


def _guard():
    path = Path(__file__).resolve().parents[1] / "verify_release.py"
    spec = spec_from_file_location("face2ceph_release_guard", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_path_pattern_covers_both_windows_separators_without_matching_urls() -> None:
    pattern = _guard().LOCAL_PATH
    assert pattern.search("E" + ":/private/project")
    assert pattern.search("E" + ":\\private\\project")
    assert pattern.search("/" + "Users/name/project")
    assert pattern.search("/" + "home/name/project")
    assert pattern.search("https://example.org/resource") is None


def test_release_metadata_preserves_reference_bytes_and_ignores_controlled_partitions() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / ".gitattributes").read_text(encoding="utf-8").splitlines() == [
        "reference/** -text"
    ]
    ignore_rules = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "*partition.csv" in ignore_rules
    assert "split.csv" in ignore_rules


def test_release_guard_rejects_controlled_partition_and_mapping_names() -> None:
    guard = _guard()
    assert {"partition.csv", "frozen_partition.csv", "split.csv"} <= guard.FORBIDDEN_FILENAMES
    for name in ("code_map", "code-map", "id_map", "crosswalk", "linkage"):
        assert guard.FORBIDDEN_NAME_FRAGMENT.search(name)


def test_non_english_pattern_covers_east_asian_scripts_and_fullwidth_forms() -> None:
    pattern = _guard().EAST_ASIAN_OR_FULLWIDTH_TEXT
    for text in ("Chinese: \u6d4b\u8bd5", "Japanese: \u30c6\u30b9\u30c8", "Korean: \ud14c\uc2a4\ud2b8", "fullwidth\uff0c"):
        assert pattern.search(text)
    assert pattern.search("Acceptable × ° ‘Western quotes’") is None


def test_manifest_requires_every_reference_file_recursively(tmp_path: Path) -> None:
    guard = _guard()
    guard.ROOT = tmp_path
    reference = tmp_path / "reference"
    nested = reference / "nested"
    nested.mkdir(parents=True)
    root_file = reference / "thresholds.yaml"
    nested_file = nested / "summary.json"
    nested_readme = nested / "README.md"
    root_file.write_bytes(b"threshold: 1\r\n")
    nested_file.write_bytes(b"{}\n")
    nested_readme.write_bytes(b"Nested documentation\n")
    (reference / "README.md").write_text("Documentation\n", encoding="utf-8")
    (reference / "SHA256SUMS").write_text(
        f"{hashlib.sha256(root_file.read_bytes()).hexdigest()}  thresholds.yaml\n",
        encoding="ascii",
    )
    errors = guard._manifest_errors()
    assert errors == [
        "reference files are absent from SHA256SUMS: "
        "['nested/README.md', 'nested/summary.json']"
    ]

    with (reference / "SHA256SUMS").open("a", encoding="ascii", newline="\n") as stream:
        stream.write(f"{hashlib.sha256(nested_readme.read_bytes()).hexdigest()}  nested/README.md\n")
        stream.write(f"{hashlib.sha256(nested_file.read_bytes()).hexdigest()}  nested/summary.json\n")
    assert guard._manifest_errors() == []


def test_gitattributes_guard_rejects_broad_or_additional_rules(tmp_path: Path) -> None:
    guard = _guard()
    guard.ROOT = tmp_path
    attributes = tmp_path / ".gitattributes"
    attributes.write_text("reference/** -text\n", encoding="utf-8")
    assert guard._gitattributes_errors() == []
    attributes.write_text("* -text\n", encoding="utf-8")
    assert guard._gitattributes_errors()


def test_gitignore_guard_requires_controlled_file_rules(tmp_path: Path) -> None:
    guard = _guard()
    guard.ROOT = tmp_path
    ignore = tmp_path / ".gitignore"
    ignore.write_text("\n".join(sorted(guard.REQUIRED_GITIGNORE_RULES)) + "\n", encoding="utf-8")
    assert guard._gitignore_errors() == []
    ignore.write_text("split.csv\n", encoding="utf-8")
    assert guard._gitignore_errors()
