"""Deterministic eligible-cohort split and fold assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PartitionConfig:
    test_fraction: float = 0.20
    calibration_fraction: float = 0.10
    folds: int = 5
    seed: int = 42
    strata: tuple[str, ...] = ("sex", "age_band", "sagittal", "vertical")
    minimum_stratum_size: int = 20
    degrade_order: tuple[str, ...] = ("age_band", "vertical", "sex")
    case_id_column: str = "case_id"

    def __post_init__(self) -> None:
        if not 0 < self.test_fraction < 1:
            raise ValueError("test_fraction must lie between zero and one")
        if not 0 < self.calibration_fraction < 1:
            raise ValueError("calibration_fraction must lie between zero and one")
        if self.folds < 2 or self.minimum_stratum_size < 1:
            raise ValueError("Invalid fold or stratum configuration")


@dataclass(frozen=True)
class PartitionResult:
    assignments: pd.DataFrame
    requested_strata: tuple[str, ...]
    used_strata: tuple[str, ...]
    dropped_strata: tuple[str, ...]


def _stratum_key(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if not columns:
        return pd.Series("ALL", index=frame.index, dtype=object)
    return frame[list(columns)].astype(str).agg("|".join, axis=1)


def select_strata(frame: pd.DataFrame, config: PartitionConfig) -> tuple[pd.Series, tuple[str, ...], tuple[str, ...]]:
    missing = [column for column in config.strata if column not in frame]
    if missing:
        raise KeyError(f"Missing stratification columns: {missing}")
    columns = list(config.strata)
    dropped: list[str] = []
    while columns:
        key = _stratum_key(frame, columns)
        if int(key.value_counts().min()) >= config.minimum_stratum_size:
            return key, tuple(columns), tuple(dropped)
        candidate = next((column for column in config.degrade_order if column in columns), None)
        if candidate is None:
            break
        columns.remove(candidate)
        dropped.append(candidate)
    return _stratum_key(frame, columns), tuple(columns), tuple(dropped)


def _assign_groups(
    key: pd.Series,
    fractions: Sequence[tuple[str, float]],
    rng: np.random.Generator,
) -> pd.Series:
    assigned = pd.Series(index=key.index, dtype=object)
    for indices in key.groupby(key, sort=True).groups.values():
        shuffled = np.asarray(list(indices))
        rng.shuffle(shuffled)
        start = 0
        for name, fraction in fractions[:-1]:
            count = int(round(len(shuffled) * fraction))
            assigned.loc[shuffled[start : start + count]] = name
            start += count
        assigned.loc[shuffled[start:]] = fractions[-1][0]
    return assigned


def _assign_folds(key: pd.Series, folds: int, rng: np.random.Generator) -> pd.Series:
    assigned = pd.Series(pd.array([pd.NA] * len(key), dtype="Int64"), index=key.index)
    for indices in key.groupby(key, sort=True).groups.values():
        shuffled = np.asarray(list(indices))
        rng.shuffle(shuffled)
        assigned.loc[shuffled] = pd.array(np.arange(len(shuffled)) % folds, dtype="Int64")
    return assigned


def make_partition(frame: pd.DataFrame, config: PartitionConfig = PartitionConfig()) -> PartitionResult:
    required = {config.case_id_column, *config.strata}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing partition columns: {missing}")
    if frame[config.case_id_column].isna().any() or frame[config.case_id_column].duplicated().any():
        raise ValueError("Case identifiers must be complete and unique")
    cohort = frame.reset_index(drop=True).copy()
    key, used, dropped = select_strata(cohort, config)
    rng = np.random.default_rng(config.seed)

    first = _assign_groups(
        key,
        (("internal_test", config.test_fraction), ("remainder", 1.0 - config.test_fraction)),
        rng,
    )
    split = pd.Series(index=cohort.index, dtype=object)
    split.loc[first == "internal_test"] = "internal_test"
    remainder = first.index[first == "remainder"]
    second = _assign_groups(
        key.loc[remainder],
        (("calibration", config.calibration_fraction), ("train_cv", 1.0 - config.calibration_fraction)),
        rng,
    )
    split.loc[remainder] = second
    train_indices = split.index[split == "train_cv"]
    fold = pd.Series(pd.array([pd.NA] * len(cohort), dtype="Int64"), index=cohort.index)
    fold.loc[train_indices] = _assign_folds(key.loc[train_indices], config.folds, rng)

    assignments = pd.DataFrame(
        {
            config.case_id_column: cohort[config.case_id_column].to_numpy(),
            "split": split.to_numpy(),
            "fold": fold.array,
        }
    )
    validate_partition(assignments, cohort[config.case_id_column], config)
    return PartitionResult(assignments, tuple(config.strata), used, dropped)


def validate_partition(
    assignments: pd.DataFrame,
    expected_case_ids: Sequence[object] | pd.Series | None = None,
    config: PartitionConfig = PartitionConfig(),
) -> None:
    required = {config.case_id_column, "split", "fold"}
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise KeyError(f"Missing assignment columns: {missing}")
    identifiers = assignments[config.case_id_column]
    if identifiers.isna().any() or identifiers.duplicated().any():
        raise ValueError("Every case must occur exactly once")
    allowed = {"train_cv", "calibration", "internal_test"}
    observed = set(assignments["split"].dropna().astype(str))
    if observed != allowed or assignments["split"].isna().any():
        raise ValueError("Partition must contain train_cv, calibration, and internal_test")
    training = assignments["split"] == "train_cv"
    train_folds = pd.to_numeric(assignments.loc[training, "fold"], errors="coerce")
    if (
        train_folds.isna().any()
        or not train_folds.between(0, config.folds - 1).all()
        or not np.equal(train_folds, np.floor(train_folds)).all()
        or set(train_folds.astype(int)) != set(range(config.folds))
    ):
        raise ValueError("Training cases require valid fold indices")
    if assignments.loc[~training, "fold"].notna().any():
        raise ValueError("Calibration and test cases cannot have fold indices")
    if expected_case_ids is not None:
        expected = pd.Index(expected_case_ids)
        if expected.has_duplicates or set(expected) != set(identifiers):
            raise ValueError("The partition does not match the eligible cohort")


def attach_frozen_partition(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    case_id_column: str = "case_id",
) -> pd.DataFrame:
    config = PartitionConfig(case_id_column=case_id_column)
    validate_partition(assignments, frame[case_id_column], config)
    merged = frame.merge(
        assignments[[case_id_column, "split", "fold"]],
        on=case_id_column,
        how="left",
        validate="one_to_one",
    )
    if merged[["split"]].isna().any().any():
        raise ValueError("Some cases are absent from the frozen partition")
    return merged
