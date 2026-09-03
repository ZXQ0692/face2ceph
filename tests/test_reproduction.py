import numpy as np
import pytest

from face2ceph import reproduction


def _values() -> dict[str, float]:
    return {name: float(index) for index, name in enumerate(reproduction.QUANTITY_KEYS)}


def test_quantity_contract_contains_46_unique_values() -> None:
    keys = reproduction.QUANTITY_KEYS
    assert len(keys) == len(set(keys)) == 46
    assert sum(key.startswith("reliability.") for key in keys) == 18
    assert sum(key.startswith("ceiling.") for key in keys) == 2
    assert sum(key.startswith("performance.") for key in keys) == 18
    assert sum(key.startswith("conformal.") for key in keys) == 8


def test_comparison_reports_each_quantity_and_applies_absolute_tolerance() -> None:
    reference = _values()
    computed = reference.copy()
    computed[reproduction.QUANTITY_KEYS[0]] += 0.001
    report = reproduction.compare_quantities(computed, reference, tolerance=0.0005)

    assert report["quantity_count"] == 46
    assert report["passed_count"] == 45
    assert report["passed"] is False
    assert [check["name"] for check in report["checks"]] == list(reproduction.QUANTITY_KEYS)


def test_cli_writes_only_when_output_is_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    report = {
        "passed": True,
        "passed_count": 46,
        "quantity_count": 46,
        "checks": [],
    }
    written = []
    monkeypatch.setattr(reproduction, "reproduce", lambda *args: report)
    monkeypatch.setattr(
        reproduction, "write_json", lambda path, payload: written.append((path, payload))
    )

    assert reproduction.main(["--data-dir", "controlled"]) == 0
    assert written == []
    assert reproduction.main(["--data-dir", "controlled", "--output", "verification.json"]) == 0
    assert written == [("verification.json", report)]


def test_cli_can_print_every_value(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    values = _values()
    report = reproduction.compare_quantities(values, values)
    monkeypatch.setattr(reproduction, "reproduce", lambda *args: report)

    assert reproduction.main(["--data-dir", "controlled", "--show-values"]) == 0
    output = capsys.readouterr().out
    assert output.count(" computed=") == 46
    assert "46/46 quantities matched." in output


def test_prediction_archive_must_cover_the_analyzed_split() -> None:
    def row(case_id: str) -> dict[str, str]:
        values = {
            "case_id": case_id,
            "split": "internal_test",
            "analyzed": "1",
        }
        values.update({target: "0" for target in reproduction.TARGETS})
        return values

    measurement_rows = [row("a"), row("b")]
    archive = {"case_id": np.asarray(["a"]), "y_raw": np.zeros((1, 8))}
    with pytest.raises(ValueError, match="every analyzed"):
        reproduction._validate_prediction_rows(
            [measurement_rows[0]], archive, "internal_test", measurement_rows
        )
