# Model-arm registry

This registry is the authoritative translation between the semantic names used
by the public command line and the compact identifiers used in frozen study
artifacts. Compact identifiers are provenance labels; they are not undocumented
configuration switches. The differing labels used by publication-shaped JSON
generators are exposed as `face2ceph.analyses.PUBLICATION_ARM_LABELS`.

## Availability populations

The frozen partition is assigned to every eligible case before image-QC
selection. Training, prediction, and evaluation then select the availability
population required by the arm without changing any split assignment.

| Selector | Definition | Train/validation | Calibration | Internal test | Total |
| --- | --- | ---: | ---: | ---: | ---: |
| `eligible` | Every eligible cohort row | 12,583 | 1,399 | 3,499 | 17,481 |
| `usable` | Both normalized RGB views succeeded | 12,526 | 1,389 | 3,477 | 17,392 |
| `analyzed` | `usable` plus a valid profile signed-distance field | 12,510 | 1,388 | 3,470 | 17,368 |

The 24-case difference between `usable` and `analyzed` is caused solely by the
fixed silhouette/SDF quality-control rules. Outcomes do not enter that decision.

## Neural arms

| Semantic CLI/config name | Frozen identifier | Imaging and model distinction | Availability | Checkpoint criterion | Study status | Public-source status |
| --- | --- | --- | --- | --- | --- | --- |
| `classification_rgb` | `c1` | Frontal and profile RGB; concatenated metadata; classification only | `usable` | Maximize the mean sagittal/vertical balanced accuracy | Pre-specified model ladder | Training, prediction, evaluation, and arm-history comparison code are provided; the historical checkpoint, case-level predictions, and fold histories are controlled |
| `classification_shape` | `c2` | Adds profile SDF; concatenated metadata; classification only | `analyzed` | Maximize the mean sagittal/vertical balanced accuracy | Pre-specified model ladder | Training, prediction, evaluation, and arm-history comparison code are provided; the historical checkpoint, case-level predictions, and fold histories are controlled |
| `multitask` | `c3` | RGB plus SDF; FiLM metadata; homoscedastic regression | `analyzed` | Minimize the mean validation MAE across eight targets | Pre-specified model ladder | Training, prediction, evaluation, and arm-history comparison code are provided; the historical checkpoint, case-level predictions, and fold histories are controlled |
| `main` | `c4b` | RGB plus SDF; FiLM metadata; heteroscedastic regression | `analyzed` | Minimize the mean validation MAE across eight targets | Pre-specified primary model | Training, prediction, evaluation, and core-bundle aggregate analysis code are provided; the historical checkpoint, case-level predictions, and fold histories are controlled |
| `silhouette_only` | `silhouette_only` | Profile SDF imaging input | `analyzed` | Minimize the mean validation MAE across eight targets | Pre-specified input sensitivity arm | Training, prediction, evaluation, and arm-history comparison code are provided; the historical checkpoint, case-level predictions, and fold histories are controlled |
| `stronger_backbone` | `stronger_backbone` | Main architecture with `convnext_tiny.fb_in22k_ft_in1k_384` | `analyzed` | Minimize the mean validation MAE across eight targets | Pre-specified backbone sensitivity arm | Training, prediction, evaluation, and arm-history comparison code are provided; the historical checkpoint, case-level predictions, and fold histories are controlled |
| `profile_only` | `profile_only` | Profile RGB and SDF only | `analyzed` | Minimize the mean validation MAE across eight targets | Post hoc single-view analysis | Training, prediction, evaluation, and arm-history comparison code are provided; the historical checkpoint, case-level predictions, and fold histories are controlled |
| `frontal_only` | `frontal_only` | Frontal RGB only; the frozen study selection still applies silhouette QC | `analyzed` | Minimize the mean validation MAE across eight targets | Post hoc single-view analysis | Training, prediction, evaluation, and arm-history comparison code are provided; the historical checkpoint, case-level predictions, and fold histories are controlled |
| `learning_10` | `learning_curve_10pct` | Main architecture; 10% fold-training subset | `analyzed` | Minimize the mean validation MAE across eight targets | Pre-listed secondary learning-curve analysis | Training, prediction, evaluation, and learning/history aggregation code are provided; the historical checkpoint, case-level test predictions, and fold history are controlled |
| `learning_25` | `learning_curve_25pct` | Main architecture; 25% fold-training subset | `analyzed` | Minimize the mean validation MAE across eight targets | Pre-listed secondary learning-curve analysis | Training, prediction, evaluation, and learning/history aggregation code are provided; the historical checkpoint, case-level test predictions, and fold history are controlled |
| `learning_50` | `learning_curve_50pct` | Main architecture; 50% fold-training subset | `analyzed` | Minimize the mean validation MAE across eight targets | Pre-listed secondary learning-curve analysis | Training, prediction, evaluation, and learning/history aggregation code are provided; the historical checkpoint, case-level test predictions, and fold history are controlled |
| `learning_75` | `learning_curve_75pct` | Main architecture; 75% fold-training subset | `analyzed` | Minimize the mean validation MAE across eight targets | Pre-listed secondary learning-curve analysis | Training, prediction, evaluation, and learning/history aggregation code are provided; the historical checkpoint, case-level test predictions, and fold history are controlled |

The mean effective fold-training sample sizes recorded in the frozen
learning-curve artifact are 1,003.6, 2,503.6, 5,005.0, and 7,504.4 for the 10%,
25%, 50%, and 75% arms, respectively; the corresponding main-arm value is
10,008.0. These are means across five folds, not additional cohort counts.

The frozen `c0a` identifier denotes the historical handcrafted-geometry
baseline. Its 320-feature construction and LightGBM outer-fold evaluation
pipeline are not included in this release; `reference/results/c0a_geometry_summary.json`
is retained only as a checksummed provenance record. The public
`evaluate_geometry_baseline` utility is an independent Extra Trees baseline and
does not reproduce `c0a`. Labels such as `T0` and `P1` in frozen result
inventories identify analysis stages or artifact groups, not model
configurations.

All runnable arms create new outputs only under `generated/`. The aggregate
analysis and experimental utility inputs are detailed in
[`ANALYSIS_INPUTS.md`](ANALYSIS_INPUTS.md). No public arm contains the
historical study checkpoint, case-level prediction archive, or fold training
history, and no arm definition implies that a frozen aggregate under
`reference/results/` was produced by this reimplementation.
