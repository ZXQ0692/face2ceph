# Configuration rationale

`pipeline.yaml` contains the fixed settings shared by every arm. Files under
`arms/` are minimal overlays, and `ARM_REGISTRY.md` records their study meaning,
availability population, and checkpoint criterion.

Images are normalized geometrically before training: frontal views use the eye
landmarks, and profile views use the segmented facial extent. The 384 by 384
model input is therefore a standardized crop, not a direct aspect-distorting
resize of the source photograph. The profile signed-distance field stays
spatially aligned with its RGB view.

Training augmentation is restricted to small affine and photometric changes.
Horizontal reflection is excluded because it would reverse the standardized
profile orientation and change laterality semantics. The 5-degree rotation and
10% translation limits preserve the normalized clinical view while testing
minor positioning variation.

All neural arms share the 30-epoch cap, 3-epoch warmup, 10-epoch early-stopping
patience, optimizer settings, and fold definitions. This keeps arm comparisons
on one training budget. Regression arms select the lowest mean validation MAE;
classification-only arms select the highest mean validation balanced accuracy.
The internal-test split is not used for checkpoint, threshold, conformal, or
referral selection.

The main arm uses heteroscedastic regression so ensemble uncertainty combines
the predicted aleatoric variance and between-fold epistemic variance. The
classification, homoscedastic, input, backbone, and learning-fraction overlays
change only the fields shown in their YAML files.
