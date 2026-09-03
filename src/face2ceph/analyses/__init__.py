"""Aggregate publication analyses for authorized study inputs."""

from .confound import analyze_confound_probes
from .geometry import evaluate_geometry_baseline
from .learning import (
    PUBLICATION_ARM_LABELS,
    aggregate_learning_curve,
    compare_arm_histories,
    fit_power_curve,
    publication_compare_arms_summary,
    publication_learning_curve_fit,
    publication_learning_curve_summary,
)
from .perturbation import (
    DEFAULT_PERTURBATIONS,
    PerturbationSpec,
    PerturbationTransform,
    apply_perturbation,
    score_perturbation_grid,
)

__all__ = [
    "DEFAULT_PERTURBATIONS",
    "PerturbationSpec",
    "PerturbationTransform",
    "PUBLICATION_ARM_LABELS",
    "aggregate_learning_curve",
    "analyze_confound_probes",
    "apply_perturbation",
    "compare_arm_histories",
    "evaluate_geometry_baseline",
    "fit_power_curve",
    "publication_compare_arms_summary",
    "publication_learning_curve_fit",
    "publication_learning_curve_summary",
    "score_perturbation_grid",
]
