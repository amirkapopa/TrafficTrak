# Third-party notices

## Model weights (downloaded, not redistributed in git)
* **YOLOX-S / YOLOX-M ONNX** - Megvii-BaseDetection/YOLOX, release 0.1.1rc0.
  Licence: Apache License 2.0. https://github.com/Megvii-BaseDetection/YOLOX
  Trained by the YOLOX authors on MS COCO 2017.  The YOLOX pre/post-processing
  in `src/detection.py` (letterbox with pad value 114, grid decoding) follows the
  reference ONNX demo of that repository.

## Algorithms re-implemented (no code copied)
* **ByteTrack** association scheme - Zhang et al., "ByteTrack: Multi-Object
  Tracking by Associating Every Detection Box", ECCV 2022.  Reference code MIT
  licence: https://github.com/ifzhang/ByteTrack.  `src/tracking.py` is an
  independent implementation.
* Constant-velocity Kalman filter - textbook formulation.

## Python dependencies (installed via pip, not vendored)
NumPy (BSD-3), OpenCV (Apache-2.0), ONNX Runtime (MIT), SciPy (BSD-3), PyYAML (MIT);
analysis/demo only: Matplotlib (PSF-based), imageio-ffmpeg (BSD-2; bundles an
FFmpeg binary, LGPL/GPL), Streamlit (Apache-2.0), Altair (BSD-3), pandas (BSD-3);
development: pytest (MIT), ruff (MIT).

## Data
No external dataset was used to train or tune anything in this repository.
OpenCV's public sample clip `vtest.avi` was used only for local smoke tests and
is not included.
