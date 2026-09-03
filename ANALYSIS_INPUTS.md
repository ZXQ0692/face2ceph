# Analysis inputs and output boundary

This document separates the core aggregate-analysis command from experimental
utilities that require additional controlled intermediates. Public code
availability does not mean that the historical inputs or frozen outputs were
produced by this reimplementation.

## Core controlled-bundle analysis

The `face2ceph analyze` command accepts:

- an authorized measurement/cohort CSV with opaque `case_id`, age, sex, the
  eight measurements, the canonical `analyzed` flag, paired `*_t2`
  measurements, and distinct `tracer_1` and `tracer_2` values for observed
  repeat tracings;
- either `split` and `fold` columns embedded in the authorized measurement
  table or the matching complete frozen eligible-cohort partition;
- a complete aligned calibration prediction archive;
- a complete aligned internal-test prediction archive.

For the heteroscedastic main arm, both prediction archives must contain
`case_id`, `mu`, `sigma`, `prob_sag`, and `prob_vert`. Case membership and truth
are derived from the authorized cohort and checked against the frozen
partition. Inputs are read-only.

```text
face2ceph analyze --cohort /path/to/authorized/measurements.csv --partition /path/to/authorized/frozen_partition.csv --calibration-predictions /path/to/authorized/c4b_calibration.npz --test-predictions /path/to/authorized/c4b_internal_test.npz --output-dir analysis/main --arm main
```

Omit `--partition` when the controlled measurement table already carries the
validated frozen `split` and `fold` columns.

The command writes nine aggregate JSON reports:

| Output | Aggregate content |
| --- | --- |
| `bland_altman.json` | Per-target agreement summaries |
| `age_strata_<arm>.json` | Age-stratified regression and classification summaries |
| `shrinkage_<arm>.json` | Calibration-to-test shrinkage summaries |
| `conformal_adaptivity_<arm>.json` | Aggregate interval adaptivity summaries |
| `sigma_patient_level_<arm>.json` | Aggregate uncertainty-stratified summaries |
| `threshold_sensitivity_<arm>.json` | Declared threshold-scheme sensitivity summaries |
| `posthoc_<arm>.json` | Declared post hoc route summaries |
| `cost_sensitive_<arm>.json` | Calibration-selected cost-sensitive summaries |
| `boundary_analysis.json` | Aggregate boundary and reliability-ceiling summaries |

`analysis_status.json` is written alongside them. It identifies the reports
generated from the supplied bundle and separately names unavailable controlled
intermediates and frozen artifacts whose generators are not included in this
release. The latter include `c0a_geometry_summary.json`, the complete
`reliability_summary.json`, and the learning-curve portions of
`boundary_analysis.json`. The minimum controlled bundle defined for the core
command does not include prediction archives for the learning arms. Such
archives may be present only when separately requested and approved. The core
analyzer does not accept them or regenerate the learning-curve boundary fields;
when an authorized holder has those additional archives, this remains an
interface and code-coverage limitation rather than evidence that the data are
absent. The reproduction checker verifies selected reliability quantities but
does not generate the complete reliability report, its bootstrap intervals, or
its reference ICC calculations. The status file also records that exact
historical Monte Carlo interval draws require the unavailable original case
order and that the frozen boundary-analysis full-source SD requires unavailable
pre-eligibility rows. The new boundary report labels its bundle-derived value as
`sd_eligible_cohort` and does not present it as the same estimand. The command
does not write case identifiers, per-case predictions, features, images, or
training histories into these aggregate JSON files.

Relevant machine-readable contracts are:

- [`measurements.schema.json`](schemas/measurements.schema.json);
- [`partition.schema.json`](schemas/partition.schema.json);
- [`predictions.schema.json`](schemas/predictions.schema.json);
- [`analysis_status.schema.json`](schemas/analysis_status.schema.json).

## Experimental analysis APIs

These APIs are importable from `face2ceph.analyses` and are tested, but they are
not additional `face2ceph analyze` modes. The caller must align controlled
arrays, retain the frozen split/fold semantics, and serialize only aggregate
results to a new path under `generated/`.

Inference accepts native checkpoints written by this release and the validated
historical five-member `c4b` layout. No compatibility claim is made for another
historical arm or checkpoint format. Native and historical envelopes cannot be
mixed within one ensemble directory. The historical `c4b` support does not make
those weights public; they remain separately controlled.

Native fold count is configurable but must agree between the selected pipeline
and every checkpoint. Historical `c4b` ensembles require exactly five members.
Batch size, worker count, mixed precision, and channels-last execution may be
changed for inference without changing checkpoint identity; the remaining
declared training-identity fields stay checked, and native members must retain
one identical complete stored training configuration.

| Capability and API | Additional controlled inputs | Output boundary |
| --- | --- | --- |
| Perturbation inference: `predict --perturbation <tag>`, `DEFAULT_PERTURBATIONS`, `PerturbationTransform`, and `apply_perturbation` | Authorized normalized images, SDF inputs where applicable, either a native checkpoint ensemble for the selected arm or the validated historical `c4b` ensemble, cohort, and frozen partition | Each prediction NPZ is case-level controlled data. It must not be published. Other historical checkpoint formats are not supported. |
| Perturbation scoring: `score_perturbation_grid` | One unperturbed archive, all 23 registered perturbation archives, aligned regression truth, and both class-label arrays | Returns a condition-level table and aggregate summary; no case identifier or per-case vector is returned. A partial or unregistered grid is rejected. |
| Frozen-feature confound probes: `analyze_confound_probes` | Train and inference feature matrices, acquisition-batch labels, inference-domain labels, and optional demographics and phenotype labels | Returns aggregate probe, shuffled-control, and silhouette summaries. Feature matrices and labels remain controlled. |
| Independent Extra Trees geometry utility: `evaluate_geometry_baseline` | Authorized handcrafted geometry feature matrix, aligned regression targets and class labels, and frozen outer-fold identifiers | Returns aggregate cross-fold regression and classification metrics without per-case predictions. It is not the historical 320-feature LightGBM `c0a` pipeline and cannot reproduce `c0a_geometry_summary.json`. |
| Arm-history comparison: `compare_arm_histories` and `publication_compare_arms_summary` | Per-fold epoch histories for every compared arm, each arm's declared checkpoint criterion, and public arm metadata for the publication-shaped output | Return arm and fold-level metric summaries, not case-level data. Each new `train` run writes a compatible `validation_history.json`; historical fold histories remain controlled. |
| Learning utilities: `publication_learning_curve_summary`, `publication_learning_curve_fit`, `aggregate_learning_curve`, and `fit_power_curve` | Per-fold epoch histories and actual fold training counts for at least four declared training sizes | Return the two publication-shaped learning artifacts or an expanded exploratory aggregate. New histories are produced by `train`; the utilities do not require prediction archives. Exact historical internal-test and boundary summaries separately require the corresponding learning-arm prediction archives and are not generated by the core analyzer. |

The structure accepted by the history utilities is documented in
[`analysis_arm_histories.schema.json`](schemas/analysis_arm_histories.schema.json).
The publication-shaped generators preserve the frozen artifact conventions:
`learning_curve.json` and `compare_arms_summary.json` use population SD
(`ddof=0`), whereas `learning_curve_fit.json` uses sample SD (`ddof=1`).
They also apply `PUBLICATION_ARM_LABELS`, the declared semantic-to-frozen arm
crosswalk; an explicit `arm_labels` override is accepted only when the resulting
labels remain unique.
The `features` array optionally emitted by `predict --include-features` is
documented in [`predictions.schema.json`](schemas/predictions.schema.json).

## Controlled-output warning

`predict --include-features` and `predict --perturbation` always create
case-aligned NPZ archives. They can reveal individual representation or
prediction information even when case codes are pseudonymous. Keep them inside
the access-restricted `generated/` tree, apply the data-use agreement, and do
not copy them into `reference/` or a public release.

The `predict` command is a cohort-aligned research batch path for declared study
splits. It is not a standalone new-patient interface, clinical device, or
deployment API, and no clinical or deployment claim follows from code
availability.

The aggregate utility outputs do not make the underlying measurements,
predictions, features, images, acquisition labels, checkpoints, or training
histories public. Complete article layout, typeset tables, and figure rendering
remain outside the release scope.
