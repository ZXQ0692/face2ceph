"""Bounded numerical verification from controlled frozen inputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .statistics import (
    balanced_accuracy,
    conformal_quantile,
    icc_1_1,
    regression_metrics,
    reliability_ceiling,
    single_tracing_error,
    stratum_offset,
)
from .integrity import verify_manifest
from .targets import CLASS_NAMES, TARGETS
from .workspace import RELEASE_ROOT, input_path, write_json

DEFAULT_TOLERANCE = 5e-6
LOWER_EXPERIENCE_STRATUM = "5-10y"

QUANTITY_KEYS = tuple(
    [f"reliability.icc_1_1.{target}" for target in TARGETS]
    + [f"reliability.single_tracing_error.{target}" for target in TARGETS]
    + [f"reliability.stratum_offset.{target}" for target in ("ANB", "SN_MP")]
    + [f"ceiling.{axis}" for axis in ("sagittal", "vertical")]
    + [f"performance.mae.{target}" for target in TARGETS]
    + [f"performance.r2.{target}" for target in TARGETS]
    + [f"performance.balanced_accuracy.{axis}" for axis in ("sagittal", "vertical")]
    + [f"conformal.q_hat.{target}" for target in TARGETS]
)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _reference_file(root: Path, name: str) -> Path:
    candidates = (root / name, root / "results" / name)
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"reference file is missing: {name}")


def _read_measurements(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "case_id",
            "age",
            "sex",
            "split",
            "sagittal",
            "vertical",
            "analyzed",
            "tracer_1",
            "tracer_2",
            *(target for target in TARGETS),
            *(f"{target}_t2" for target in TARGETS),
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError("measurement table does not match the required schema")
        rows = list(reader)
    if not rows:
        raise ValueError("measurement table is empty")
    return rows


def _read_operator_experience(path: Path) -> dict[str, str]:
    payload = _read_json(path)
    mapping = payload.get("operator_experience") if isinstance(payload, dict) else None
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("operator experience file does not match the required schema")
    values = {str(key): str(value) for key, value in mapping.items()}
    if not set(values.values()).issubset({LOWER_EXPERIENCE_STRATUM, ">10y"}):
        raise ValueError("operator experience file contains an unsupported stratum")
    return values


def _read_predictions(path: Path) -> dict[str, np.ndarray]:
    required = ("case_id", "y_raw", "mu", "sigma", "prob_sag", "prob_vert")
    with np.load(path, allow_pickle=False) as archive:
        missing = set(required).difference(archive.files)
        if missing:
            raise ValueError(f"prediction archive does not match the required schema: {path.name}")
        arrays = {name: np.asarray(archive[name]) for name in required}
    n = arrays["case_id"].shape[0] if arrays["case_id"].ndim == 1 else -1
    expected = {
        "y_raw": (n, len(TARGETS)),
        "mu": (n, len(TARGETS)),
        "sigma": (n, len(TARGETS)),
        "prob_sag": (n, len(CLASS_NAMES["sagittal"])),
        "prob_vert": (n, len(CLASS_NAMES["vertical"])),
    }
    if n < 1 or any(arrays[name].shape != shape for name, shape in expected.items()):
        raise ValueError(f"prediction archive has invalid array shapes: {path.name}")
    case_ids = [_case_id(value) for value in arrays["case_id"]]
    numeric = tuple(name for name in required if name != "case_id")
    if len(case_ids) != len(set(case_ids)) or any(not value for value in case_ids):
        raise ValueError(f"prediction archive has invalid case codes: {path.name}")
    if any(not np.isfinite(arrays[name]).all() for name in numeric):
        raise ValueError(f"prediction archive contains non-finite values: {path.name}")
    if (arrays["sigma"] < 0).any():
        raise ValueError(f"prediction archive contains negative uncertainty: {path.name}")
    for name in ("prob_sag", "prob_vert"):
        if (arrays[name] < 0).any() or not np.allclose(
            arrays[name].sum(axis=1), 1.0, rtol=0.0, atol=1e-5
        ):
            raise ValueError(f"prediction archive contains invalid probabilities: {path.name}")
    return arrays


def _number(value: str | None) -> float | None:
    text = "" if value is None else value.strip()
    return None if text.lower() in {"", "na", "nan"} else float(text)


def _paired_rows(
    rows: list[dict[str, str]], target: str
) -> list[tuple[dict[str, str], float, float]]:
    paired = []
    for row in rows:
        first = _number(row[target])
        second = _number(row[f"{target}_t2"])
        if second is None:
            continue
        if first is None:
            raise ValueError("a repeat tracing has no corresponding reference value")
        paired.append((row, first, second))
    if len(paired) < 2:
        raise ValueError(f"insufficient paired tracings for {target}")
    return paired


def _case_id(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _is_analyzed(row: Mapping[str, str]) -> bool:
    return row["analyzed"].strip().lower() in {"1", "true", "yes"}


def _age(value: str) -> float:
    return float(value.strip().removesuffix("+"))


def _thresholds(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    try:
        scheme = payload["schemes"][payload["primary"]]
        for axis in ("sagittal", "vertical"):
            if "strata" not in scheme[axis] or "bands" not in scheme[axis]:
                raise KeyError(axis)
    except (KeyError, TypeError) as exc:
        raise ValueError("threshold file does not define a valid primary scheme") from exc
    return scheme


def _band(axis: Mapping[str, Any], row: Mapping[str, str]) -> tuple[float, float]:
    strata = set(axis["strata"] or ())
    adult = _age(row["age"]) >= 18.0
    if not strata:
        key = "all"
    elif strata == {"adult"}:
        key = "adult" if adult else "minor"
    elif strata == {"sex"}:
        key = row["sex"]
    elif strata == {"adult", "sex"}:
        key = "adult" if adult else f"minor_{row['sex']}"
    else:
        raise ValueError("threshold strata are unsupported")
    try:
        values = axis["bands"][key]
        return float(values["lower"]), float(values["upper"])
    except (KeyError, TypeError) as exc:
        raise ValueError("no threshold band matches a measurement row") from exc


def _index_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed = {row["case_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("measurement table contains duplicate case codes")
    return indexed


def _prediction_rows(
    indexed: Mapping[str, dict[str, str]], case_ids: np.ndarray
) -> list[dict[str, str]]:
    try:
        return [indexed[_case_id(value)] for value in case_ids]
    except KeyError as exc:
        raise ValueError("prediction archive contains an unknown case code") from exc


def _validate_prediction_rows(
    rows: list[dict[str, str]],
    archive: Mapping[str, np.ndarray],
    expected_split: str,
    measurement_rows: list[dict[str, str]],
) -> None:
    expected_ids = {
        row["case_id"]
        for row in measurement_rows
        if row["split"] == expected_split and _is_analyzed(row)
    }
    observed_ids = {_case_id(value) for value in archive["case_id"]}
    if observed_ids != expected_ids:
        raise ValueError(f"prediction archive does not contain every analyzed {expected_split} case")
    if any(row["split"] != expected_split for row in rows):
        raise ValueError(f"prediction archive contains cases outside {expected_split}")
    if any(not _is_analyzed(row) for row in rows):
        raise ValueError("prediction archive contains a case excluded from analysis")
    measurements = np.asarray(
        [[_number(row[target]) for target in TARGETS] for row in rows], dtype=float
    )
    if not np.isfinite(measurements).all() or not np.allclose(
        measurements, archive["y_raw"], rtol=0.0, atol=1e-5
    ):
        raise ValueError("prediction references do not match the measurement table")


def compute_quantities(
    data_dir: str | Path,
    reference_dir: str | Path,
    operator_map: str | Path | None = None,
) -> dict[str, float]:
    data_root = input_path(data_dir, "dir")
    reference_root = input_path(reference_dir, "dir")
    operator_path = (
        input_path(operator_map, "file")
        if operator_map is not None
        else input_path(data_root / "operator_experience.json", "file")
    )
    rows = _read_measurements(input_path(data_root / "measurements.csv", "file"))
    experience = _read_operator_experience(operator_path)
    thresholds = _thresholds(_reference_file(reference_root, "thresholds.yaml"))
    values: dict[str, float] = {}
    errors: dict[str, float] = {}

    for target in TARGETS:
        paired = _paired_rows(rows, target)
        first = np.asarray([pair[1] for pair in paired], dtype=float)
        second = np.asarray([pair[2] for pair in paired], dtype=float)
        values[f"reliability.icc_1_1.{target}"] = icc_1_1(first, second)
        errors[target] = single_tracing_error(first, second)
        values[f"reliability.single_tracing_error.{target}"] = errors[target]

    for target in ("ANB", "SN_MP"):
        paired = _paired_rows(rows, target)
        try:
            first_strata = [experience[row["tracer_1"]] for row, _, _ in paired]
            second_strata = [experience[row["tracer_2"]] for row, _, _ in paired]
        except KeyError as exc:
            raise ValueError("operator experience mapping is incomplete") from exc
        differences = np.asarray([first - second for _, first, second in paired], dtype=float)
        values[f"reliability.stratum_offset.{target}"] = stratum_offset(
            first_strata,
            second_strata,
            differences,
            LOWER_EXPERIENCE_STRATUM,
        )

    analyzed = [
        row for row in rows if row["split"] == "internal_test" and _is_analyzed(row)
    ]
    for axis, target in (("sagittal", "ANB"), ("vertical", "SN_MP")):
        distances = []
        for row in analyzed:
            lower, upper = _band(thresholds[axis], row)
            measurement = _number(row[target])
            if measurement is None:
                raise ValueError("an analyzed case has a missing reference measurement")
            distances.append(min(abs(measurement - lower), abs(measurement - upper)))
        values[f"ceiling.{axis}"] = reliability_ceiling(distances, round(errors[target], 4))

    test = _read_predictions(
        input_path(data_root / "predictions" / "c4b_internal_test.npz", "file")
    )
    indexed = _index_rows(rows)
    ordered = _prediction_rows(indexed, test["case_id"])
    _validate_prediction_rows(ordered, test, "internal_test", rows)
    keep = np.asarray([_is_analyzed(row) for row in ordered], dtype=bool)
    if not keep.any():
        raise ValueError("prediction archive contains no analyzed cases")
    reference = test["y_raw"][keep]
    prediction = test["mu"][keep]
    for index, target in enumerate(TARGETS):
        metrics = regression_metrics(reference[:, index], prediction[:, index])
        values[f"performance.mae.{target}"] = metrics["mae"]
        values[f"performance.r2.{target}"] = metrics["r2"]

    for axis, probability_key in (("sagittal", "prob_sag"), ("vertical", "prob_vert")):
        class_index = {name: index for index, name in enumerate(CLASS_NAMES[axis])}
        try:
            truth = np.asarray([class_index[row[axis]] for row in ordered], dtype=int)[keep]
        except KeyError as exc:
            raise ValueError("measurement table contains an unsupported class label") from exc
        predicted = test[probability_key][keep].argmax(axis=1)
        values[f"performance.balanced_accuracy.{axis}"] = balanced_accuracy(
            truth, predicted, range(len(class_index))
        )

    calibration = _read_predictions(
        input_path(data_root / "predictions" / "c4b_calibration.npz", "file")
    )
    calibration_rows = _prediction_rows(indexed, calibration["case_id"])
    _validate_prediction_rows(calibration_rows, calibration, "calibration", rows)
    if set(map(_case_id, test["case_id"])) & set(map(_case_id, calibration["case_id"])):
        raise ValueError("calibration and internal-test prediction archives overlap")
    alpha = float(_read_json(_reference_file(reference_root, "conformal_quantiles.json"))["alpha"])
    for index, target in enumerate(TARGETS):
        values[f"conformal.q_hat.{target}"] = conformal_quantile(
            calibration["y_raw"][:, index],
            calibration["mu"][:, index],
            calibration["sigma"][:, index],
            alpha,
        )

    if set(values) != set(QUANTITY_KEYS):
        raise RuntimeError("reproduction quantity contract is incomplete")
    return {key: values[key] for key in QUANTITY_KEYS}


def load_reference_quantities(reference_dir: str | Path) -> dict[str, float]:
    root = input_path(reference_dir, "dir")
    reliability = _read_json(_reference_file(root, "reliability_summary.json"))
    boundary = _read_json(_reference_file(root, "boundary_analysis.json"))
    evaluation = _read_json(_reference_file(root, "evaluation_c4b.json"))
    conformal = _read_json(_reference_file(root, "conformal_quantiles.json"))
    try:
        internal = next(item for item in evaluation["results"] if item["split"] == "internal_test")
    except StopIteration as exc:
        raise ValueError("evaluation reference has no internal-test result") from exc
    values: dict[str, float] = {}
    for target in TARGETS:
        values[f"reliability.icc_1_1.{target}"] = float(
            reliability["measures"][target]["icc_inter"]
        )
        values[f"reliability.single_tracing_error.{target}"] = float(
            reliability["measures"][target]["sem_inter"]
        )
    for target in ("ANB", "SN_MP"):
        values[f"reliability.stratum_offset.{target}"] = float(
            reliability["measures"][target]["stratum_offset"]
        )
    for axis in ("sagittal", "vertical"):
        values[f"ceiling.{axis}"] = float(
            boundary["axes"][axis]["label_noise_accuracy_ceiling"]
        )
    for target in TARGETS:
        values[f"performance.mae.{target}"] = float(internal["regression"][target]["MAE"])
        values[f"performance.r2.{target}"] = float(internal["regression"][target]["R2"])
    for axis in ("sagittal", "vertical"):
        values[f"performance.balanced_accuracy.{axis}"] = float(
            internal["classification"][axis]["balanced_accuracy"]
        )
    q_hat = conformal["q_hat"]
    if isinstance(q_hat, dict):
        quantiles = {target: float(q_hat[target]) for target in TARGETS}
    else:
        quantiles = {
            target: float(value) for target, value in zip(conformal["targets"], q_hat, strict=True)
        }
    for target in TARGETS:
        values[f"conformal.q_hat.{target}"] = quantiles[target]
    if set(values) != set(QUANTITY_KEYS):
        raise ValueError("reference files do not define all reproduction quantities")
    return {key: values[key] for key in QUANTITY_KEYS}


def compare_quantities(
    computed: Mapping[str, float],
    reference: Mapping[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite non-negative value")
    if set(computed) != set(QUANTITY_KEYS) or set(reference) != set(QUANTITY_KEYS):
        raise ValueError("quantity mappings do not match the reproduction contract")
    checks = []
    for name in QUANTITY_KEYS:
        observed = float(computed[name])
        expected = float(reference[name])
        difference = observed - expected
        passed = bool(
            np.isfinite(observed) and np.isfinite(expected) and abs(difference) <= tolerance
        )
        checks.append(
            {
                "name": name,
                "computed": observed,
                "reference": expected,
                "difference": difference,
                "passed": passed,
            }
        )
    passed_count = sum(check["passed"] for check in checks)
    return {
        "passed": passed_count == len(checks),
        "passed_count": passed_count,
        "quantity_count": len(checks),
        "absolute_tolerance": tolerance,
        "checks": checks,
    }


def reproduce(
    data_dir: str | Path,
    reference_dir: str | Path = RELEASE_ROOT / "reference",
    operator_map: str | Path | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, Any]:
    integrity_errors = verify_manifest(reference_dir)
    if integrity_errors:
        raise ValueError("reference integrity check failed: " + "; ".join(integrity_errors))
    computed = compute_quantities(data_dir, reference_dir, operator_map)
    reference = load_reference_quantities(reference_dir)
    return compare_quantities(computed, reference, tolerance)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recompute and verify the 46 reported quantities.")
    parser.add_argument("--data-dir", required=True, help="controlled data bundle")
    parser.add_argument("--reference-dir", default=str(RELEASE_ROOT / "reference"))
    parser.add_argument("--operator-map", help="optional controlled operator-experience file")
    parser.add_argument("--atol", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--output", help="new JSON path inside generated/")
    parser.add_argument("--show-values", action="store_true", help="print every computed and reference value")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = reproduce(args.data_dir, args.reference_dir, args.operator_map, args.atol)
        if args.output:
            write_json(args.output, report)
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.show_values:
        for check in report["checks"]:
            print(
                f"{check['name']} computed={check['computed']:.10g} "
                f"reference={check['reference']:.10g} difference={check['difference']:.3g}"
            )
    print(f"{report['passed_count']}/{report['quantity_count']} quantities matched.")
    if not report["passed"]:
        for check in report["checks"]:
            if not check["passed"]:
                print(
                    f"{check['name']}: difference={check['difference']:.8g}",
                    file=sys.stderr,
                )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
