# Aggregate references

This directory contains only public, aggregate values used as immutable
comparison targets. The reproduction command never writes here.

`results/c0a_geometry_summary.json` is a checksummed provenance record from the
historical 320-feature LightGBM geometry pipeline. That feature-construction and
evaluation pipeline is not included in this release. The public Extra Trees
geometry utility is independent and does not reproduce this file.

In `results/compare_arms_summary.json`, `cls_oracle` is a diagnostic per-fold
maximum of mean sagittal and vertical balanced accuracy over epochs on the
held-out `train_cv` validation fold. It is not an internal-test result or an
additional arm-selection rule. For MAE-selected arms it does not correspond to
the saved checkpoint; for C1/C2 it coincides with their balanced-accuracy-selected
checkpoint. It was not used in the manuscript results.

Fold dispersion follows the convention stored in each frozen artifact.
`learning_curve.json` and `compare_arms_summary.json` use population SD
(`ddof=0`); `learning_curve_fit.json` uses sample SD (`ddof=1`). The public
artifact-specific generators retain these definitions without altering the
reference values.

In `results/confound_probe_c4b.json`, `P4_grl` is a frozen diagnostic showing
that no domain-adversarial objective contributed to training. The disclosed
final method has no gradient-reversal branch. Because the cohort is
single-centre and retains no second domain label, `P2_domain.applicable=false`
means that domain separability cannot be tested; it does not mean that the
representation was shown to be domain-invariant.

The confidence interval for the boundary analysis's last-segment contrast
includes zero. This is a non-detection, not evidence of equivalence or evidence
against a boundary effect. The frontal-only and profile-only comparisons are
post hoc. Perturbation results describe deterministic deviations from one
standardized acquisition setting and do not constitute external validation.

The frozen boundary artifact's `sd_cohort` values were calculated from a
17,610-row full-source label table that included 129 pre-eligibility rows. The
controlled release bundle begins at the 17,481-case eligible cohort and cannot
reconstruct that descriptive SD exactly. The measured single-tracing error,
test-set distance bins, and reported label-noise ceiling do not use the
full-source SD and remain recomputable. New analysis output therefore reports
`sd_eligible_cohort` and marks the frozen full-source estimand unavailable
rather than synthesizing absent records. Exact frozen Monte Carlo percentile
endpoints likewise require the unavailable historical case order; point
estimands remain reproducible from the controlled bundle.

The minimum controlled bundle for numerical verification and the core aggregate
analysis must contain:

```text
measurements.csv
predictions/
  c4b_calibration.npz
  c4b_internal_test.npz
```

Place `operator_experience.json` in the bundle or pass its controlled path with
`--operator-map`. The files are read by name and are never modified. Their
schemas are documented under `schemas/`.

This minimum bundle does not include trained weights, historical training
histories, or prediction archives for the learning arms. These materials may be
considered only when separately requested and approved, so code must not infer
their presence from access to the minimum bundle. When historical weights are
supplied, the release supports only the validated five-member `c4b` checkpoint
layout in addition to its native checkpoint format; it makes no compatibility
claim for other historical layouts.
