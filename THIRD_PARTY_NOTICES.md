# Third-party notices

The repository does not redistribute pretrained model binaries. The explicit
`face2ceph assets` command downloads the configured files from their upstream
locations into the controlled `generated/` tree and verifies the exact
SHA-256 values below. Training and inference do not download assets implicitly.

The project's MIT License does not relicense any third-party model, library, or
dataset. Apache-2.0 designations below are those stated in the linked upstream
model cards. They do not grant rights in training datasets or in photographs
processed by this project. Anyone redistributing a downloaded asset must review
and preserve the upstream license and notice obligations.

## MediaPipe vision assets

The following files are configured from the official MediaPipe model storage:

| Asset | Configured source | SHA-256 |
| --- | --- | --- |
| `face_landmarker.task` | [MediaPipe Face Landmarker bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task) | `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff` |
| `selfie_multiclass_256x256.tflite` | [MediaPipe SelfieMulticlass segmenter](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite) | `c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0` |

The official [Face Landmarker task
guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
identifies the models packaged in the landmarker bundle. Their official model
cards state the Apache License 2.0:

- [BlazeFace short-range model
  card](https://storage.googleapis.com/mediapipe-assets/MediaPipe%20BlazeFace%20Model%20Card%20%28Short%20Range%29.pdf);
- [Face Mesh V2 model
  card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf);
- [Blendshape V2 model
  card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf).

The segmenter is documented by the official [Image Segmenter task
guide](https://developers.google.com/edge/mediapipe/solutions/vision/image_segmenter).
Its [Multiclass Segmentation model
card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf)
states the Apache License 2.0.

## timm ConvNeXt checkpoints

| Asset | Pinned source | SHA-256 | Upstream license record |
| --- | --- | --- | --- |
| `convnext_tiny.in12k_ft_in1k.safetensors` | [Pinned Hugging Face revision](https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k/resolve/aa096f03029c7f0ec052013f64c819b34f8ad790/model.safetensors) | `a1aefa409b513cf209b085424eb3efffe4e4a9f511491bc2c12ea35209e6bb95` | [Model card: Apache-2.0](https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k) |
| `convnext_tiny.fb_in22k_ft_in1k_384.safetensors` | [Pinned Hugging Face revision](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k_384/resolve/32c741ba106f9e062174ef829354a16719c3dcac/model.safetensors) | `a5cbe10cfa2e90ec787a082253033ac0a61de0264b68210815a988e53a1a5d7c` | [Model card: Apache-2.0](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k_384) |

Relevant upstream citations are:

- Ross Wightman, `PyTorch Image Models`, Zenodo,
  [doi:10.5281/zenodo.4414861](https://doi.org/10.5281/zenodo.4414861).
- Zhuang Liu et al., “A ConvNet for the 2020s,”
  [arXiv:2201.03545](https://arxiv.org/abs/2201.03545).

The full Apache License 2.0 text is available from the
[Apache Software Foundation](https://www.apache.org/licenses/LICENSE-2.0).

## Python dependencies

Packages listed in `requirements.txt` and `requirements-reproduce.txt` are
installed from their normal package sources and are not vendored in this
repository. Each package remains governed by its upstream license. Version pins
record the study environment; they do not change those licenses or imply
redistribution permission.
