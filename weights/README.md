# Model weights

| file | model | size | licence | source |
|---|---|---|---|---|
| `yolox_m.onnx` | YOLOX-M, COCO-pretrained, 640x640 | 101 MB | Apache-2.0 | [YOLOX 0.1.1rc0 release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) |
| `yolox_s.onnx` | YOLOX-S, COCO-pretrained, 640x640 | 36 MB | Apache-2.0 | same |
| `fire_smoke.onnx` *(optional, not shipped)* | any image classifier you have the right to use | - | - | see README "fire_smoke" |

Total shipped weights: **137 MB** (limit 5 GB).  Fetch them before the offline run:

```bash
bash weights/download.sh
```

The script pins SHA-256 checksums.  `yolox_m.onnx` exceeds GitHub's 100 MB
file limit, so weights are not committed to git.  `detector.model: auto`
uses YOLOX-M when a CUDA GPU is available and YOLOX-S on CPU.
