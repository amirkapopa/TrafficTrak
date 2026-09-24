# samples/

Put the organisers' sample videos (`*.mp4`) and `camera.md` here.  Videos are
git-ignored.  At the time this repository was built **neither the videos nor
`camera.md` were available**, so `config/camera_geometry.yaml` ships empty and
uncalibrated.  Once the files are here:

```bash
make eda prior          # EDA + learned scene prior from all samples
make calibrate          # draw lanes / stop lines / crossings / signal ROI
make visualize validate # predictions_samples.json + annotated videos + checks
```
