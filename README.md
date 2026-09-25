# TrafficTrak: offline traffic-event detection for a fixed CCTV camera

TrafficTrak detects traffic events in a fixed-camera road video. It returns
tight time segments per event (Part A). It also produces a causal score
predicting whether an accident will start within the next 5 seconds (Part B).
Everything runs offline on local open weights: a YOLOX detector, a
ByteTrack-style tracker, a scene model and transparent per-class rules.

```python
from solution import detect_events, RiskEstimator, CLASSES
detect_events("clip.mp4")      # [[12.4, 19.8, "stopped_vehicle"], ...]
```

> **Status of the challenge inputs.** The organisers' `run_submission.py`
> and `evaluate.py` are in the repository root, **unmodified**. The starter
> README, `solution.py` template and `examples/` are kept for reference. The
> sample videos are 4K clips (3840×2160, 29.97 fps) of a signal-controlled
> intersection in Tashkent. They are too large for git (about 2.4 GB each); a
> 2-minute sample (`clip_C3905.mp4`) is attached to the
> [`samples` release](https://github.com/amirkapopa/TrafficTrak/releases/tag/samples).
> From it:
> * the camera was verified to be **fixed**: 0 px shift, 0° rotation and
>   scale 1.000 across the clip;
> * `config/camera_geometry.yaml` was **calibrated**: 3 zebra crossings,
>   pedestrian islands, and the carriageway where vehicles actually drive;
> * `config/scene_prior.npz` holds the learned traffic directions, the road
>   area and the normal stopping places;
> * `config/background_reference.jpg` is the empty-road background;
> * `predictions_samples.json` is the official harness output on that clip.
>
> `camera.md` and the task description were not available, so stop lines and
> signal phases stay uncalibrated. `red_light` and `stop_line` therefore stay
> silent.

## How the organisers run this submission

```bash
pip install -r requirements.txt          # or: docker build -t traffictrak .
bash weights/download.sh                 # once, before going offline (137 MB, checksummed)
python run_submission.py --videos /data/test --out predictions.json
python evaluate.py --pred predictions.json --gt ground_truth.json
```

**Submission checklist:**
* `solution.py` exposes `CLASSES`, `detect_events` and `RiskEstimator` exactly as in the template.
* `run_submission.py` and `evaluate.py` are unmodified.
* The official harness produced VALID output on real clips, within budget even on CPU.
* **Time budget (3× duration for A + B).** Two emergency limiters, one per part (1.4× and 1.3× real time), cut work before the budget runs out. If CUDA is advertised but unusable, the pipeline switches to the small model automatically. Neither triggers on the target GPU.
* No network access at run time. Weights come from `weights/`. Crashes, missing weights and corrupt videos degrade to `[]` and never lose the video.

## Contents
1. [Quick start](#quick-start)
2. [Repository layout](#repository-layout)
3. [Architecture](#architecture)
4. [Event classes: method and prerequisites](#event-classes-method-and-prerequisites)
5. [Part B: causal accident risk](#part-b-causal-accident-risk)
6. [Calibrating the scene](#calibrating-the-scene)
7. [Models, weights, licences](#models-weights-licences)
8. [Determinism](#determinism)
9. [Runtime](#runtime)
10. [Rule-based vs learned components](#rule-based-vs-learned-components)
11. [Known failure cases and limitations](#known-failure-cases-and-limitations)
12. [Testing and quality](#testing-and-quality)
13. [Local labels and evaluation](#local-labels-and-evaluation)
14. [Upload demo](#upload-demo)
15. [Team](#team)
16. [Project website](#project-website)

---

## Quick start

```bash
# Python 3.10+ ; GPU box with CUDA 12 + cuDNN 9 (onnxruntime-gpu)
pip install -r requirements.txt -r requirements-dev.txt     # CPU only: requirements-cpu.txt
bash weights/download.sh                                      # 137 MB, SHA-256 verified; do this BEFORE going offline
make check                                                    # ruff + 70 unit tests

# with the sample videos + camera.md in samples/
make prior          # learn the scene prior (traffic directions, road area) from the samples
make predict        # = python run_submission.py --videos samples --out predictions_samples.json --team TrafficTrak
make validate       # = python evaluate.py --pred predictions_samples.json --validate-only
make dev-eval       # = python evaluate.py --pred predictions_samples.json --gt labels/dev_labels.json --per-video
```

**macOS (Apple Silicon or Intel, CPU only)**: use Python 3.11 or 3.12. The
pinned NumPy and SciPy have no Python 3.13 wheels.

```bash
brew install python@3.11 git
git clone https://github.com/amirkapopa/TrafficTrak && cd TrafficTrak
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-cpu.txt -r requirements-dev.txt -r requirements-demo.txt
bash weights/download.sh                 # works with macOS bash 3.2 + shasum
python solution.py path/to/video.mp4     # prints the detected events as JSON
streamlit run demo/app.py                # upload page at http://localhost:8501
```

Visual outputs, EDA, calibration and the demo:

```bash
pip install -r requirements-demo.txt
make visualize      # outputs/vis/<video>_annotated.mp4, _timeline.png, _events.json, outputs/risk/<video>.csv
make eda prior      # outputs/eda/* and the learned scene prior config/scene_prior.npz
make calibrate      # outputs/calibration/calibrate.html (click-to-draw geometry editor)
make candidates     # candidate clips + review.csv for manual annotation
make demo           # Streamlit upload demo
```

Docker (offline evaluation image, weights baked in at build time):

```bash
docker build -t traffictrak .
docker run --gpus all --network none -v $PWD/samples:/data/test -v $PWD/out:/out traffictrak   # -> out/predictions.json
```

## Repository layout

| path | responsibility |
|---|---|
| `solution.py` | required Part A / Part B interfaces; never raises |
| `config/camera_geometry.yaml` | **the one human-edited scene file** (normalised coordinates) |
| `config/pipeline.yaml` | all thresholds and switches, documented inline |
| `src/video.py` | metadata probe, fps fallback, threaded strided decoding, timestamps (`idx / fps`) |
| `src/detection.py` | YOLOX ONNX loading (cached), letterbox, decoding, group-wise NMS, class filtering |
| `src/tracking.py` | ByteTrack-style two-stage association, variable-dt Kalman filter, stationary-fragment stitching |
| `src/geometry.py` | polygons, polylines, lanes with direction paths, stop lines, crossings, signals, config validation |
| `src/features.py` | uniform-grid track series, smoothing, velocity/acceleration/heading, pairwise gap / closing speed / TTC / contact |
| `src/flow.py` | learned traffic-direction field and carriageway occupancy (per video + optional prior) |
| `src/signal_state.py` | traffic-light state from a configured ROI, temporal smoothing, reliability |
| `src/monitors.py` | static-obstacle (debris) monitor; optional fire/smoke classifier |
| `src/rules.py` | class-specific evidence and state machines |
| `src/segments.py` | gap merging, duration filters, clamping, same-class overlap resolution, schema validator |
| `src/pipeline.py` | Part A orchestration |
| `src/risk.py` | causal `RiskEstimator` |
| `src/visualize.py` | geometry / flow overlays, annotated video, timeline and risk-curve charts |
| `run_submission.py`, `evaluate.py` | organisers' harness and official metric (unmodified, excluded from linting) |
| `scripts/` | `run_local` (visual outputs), `analyze_samples` (EDA + scene prior), `calibrate_geometry`, `annotate_candidates` (review → official ground truth) |
| `demo/app.py` | Streamlit upload demo |
| `tests/` | geometry, rules (one synthetic scenario per class), segments, schema, causal risk, components, demo |
| `docs/technical_report.md` | one-page public technical report |
| `docs/index.html` | project website (GitHub Pages) |

## Architecture

```
video ──► FrameReader (thread, stride 2 on GPU / 3 on CPU, t = idx/fps)
            │
            ├─► YOLOX-M/S (ONNX Runtime, deterministic) ──► ByteTracker ──► tracks
            ├─► signal ROI classifier ──► SignalTimeline (red/amber/green/unknown)
            └─► ObstacleMonitor / FireSmokeMonitor (sparse, downscaled)
tracks ──► stitch stationary fragments ──► TrackSeries on a uniform grid
            (smoothed ground point, scale-normalised speed, heading, accel)
         ──► FlowField (leave-one-out direction field + learned carriageway, + prior)
         ──► rules.py (14 classes, each gated on its prerequisites)
         ──► segments.py (merge gaps, min duration, clamp, no same-class overlap)
         ──► [[start, end, label], ...]  sorted by (start, label)
```

**Scale units.** Every speed and distance threshold is expressed in the object's
own size (`sqrt(w*h)` of its box, about 2–3 m for a car). This makes the rules
roughly independent of perspective depth and resolution without a
homography.

**Timing convention.** Each processed sample covers ±dt/2. A run of
samples `i0..i1` becomes `[t(i0) − dt/2, t(i1) + dt/2]`, clamped to
`[0, duration]`. Duration is the minimum of container metadata and decoded
frames, so `end ≤ video_duration` always holds.

## Event classes: method and prerequisites

"Uncalibrated" means the shipped empty `camera_geometry.yaml`.

| class | method (see `src/rules.py`) | needs | active uncalibrated? |
|---|---|---|---|
| `stopped_vehicle` | vehicle stationary (speed + displacement) ≥ 10 s on the carriageway; suppressed if queued behind/ahead of other stationary vehicles or waiting at a red stop line; requires moving traffic passing it, or ≥ 45 s isolated | carriageway (manual, else learned from traffic) | **yes** (learned carriageway) |
| `wrong_way` | sustained heading ≥ 135° from lane direction, ≥ 1.5 s, ≥ 3 scale units; start at lane entry | lanes, else learned flow (≥ 6 supporting tracks, ≥ 85 % unidirectional) | **yes** (learned flow) |
| `congestion` | per direction group: ≥ max(6, 50 % capacity) vehicles, ≥ 70 % crawling, median speed ≤ crawl, ≥ 30 s | lane groups, else learned direction groups | **yes** |
| `accident` | raw-footprint contact at compatible depth + prior closing speed + abrupt deceleration + (struck-party velocity jolt, heading jolt or pedestrian fall) + both slow afterwards; end when all stop | nothing | **yes** |
| `near_miss` | TTC < 1 s with closing ≥ 1.5 units/s + hard braking or swerve onset; no contact; end when separated and not closing | nothing | **yes** |
| `road_obstacle` | (a) animal tracks on the carriageway; (b) persistent, static, unexplained foreground vs. empty-road background | (a) carriageway (learned OK); (b) calibrated carriageway or `obstacle_regions` | animals only |
| `jaywalking` | pedestrian ground point inside carriageway (with margin), outside crossings (+1.5 body-scale slack) and sidewalks, not a rider, occupant or standing rider with an undetected scooter, ≥ 1 s | calibrated carriageway | **calibrated for the challenge camera** |
| `failure_to_yield` | moving vehicle inside a crossing while a pedestrian is on or entering it nearby | crossings | no |
| `red_light` | front point crosses the stop line along the approach while the signal has been red for ≥ 0.4 s, then enters the intersection; end on exit | stop line + signal ROI (+ intersection) | no |
| `stop_line` | vehicle crosses the stop line and stops before the intersection while red; end at green | stop line + signal ROI | no |
| `solid_line_crossing` | inset bottom corners (wheel proxies) change side of a solid polyline; end when all points are across | solid lines | no |
| `illegal_turn` | track seen in a prohibited `from` zone then in its `to` zone; onset = heading deviation ≥ 15°, end = heading stable | prohibited_turns | no |
| `illegal_u_turn` | heading reversal ≥ 150° within 15 s while moving (reversing excluded), where prohibited | `u_turn_prohibited` or zones | no |
| `fire_smoke` | optional ONNX classifier (`weights/fire_smoke.onnx`) sampled at 2 Hz; strict colour+flicker heuristic behind `mode: heuristic` | classifier weights | no (off without weights) |

Signal rules are disabled per video when fewer than 60 % of signal samples
have a confident state. They never guess the light's colour.

## Part B: causal accident risk

`RiskEstimator.step(frame, t)` uses only the current and past frames. It never
opens the file and never reads Part A output. On a GPU it runs the shared
detector on every frame (every 3rd on CPU) with an online tracker and 2-second
kinematic histories. For every nearby road-user pair it computes:

* **F1 conflict**: predicted closest approach within 4 s. The score grows with
  shorter TTC, higher closing speed and smaller miss distance.
* **F2 evasive**: speed drop over the last second, or heading change over 0.7 s.
* **F3 violation**: wrong-way motion (lanes or scene prior), a pedestrian on the
  carriageway, or a vehicle approaching a red stop line at speed.

`p = sigmoid(−4 + 3·F1 + 2·F2 + 1.5·F3 + 1.5·F1·F2)`. A single cue stays
below 0.5, for example a pure maximal conflict gives about 0.27. Only agreeing
cues, such as a conflict plus evasive action, exceed 0.5. The output follows
the hazard with fast attack (τ = 0.3 s) and slow decay (τ = 2 s). Values ≥ 0.5
are released only after the raw hazard has stayed ≥ 0.5 for 0.5 s
(hysteresis), then held until the smoothed value drops below 0.3. Pairs whose
box sizes differ by more than 2.5× are skipped: they are at clearly different
depths, and their convergence in the image is a perspective effect. Both
guards were added after they removed false alarms on real footage.

**Calibration status:** the weights are hand-set and not fitted, because no
labelled data exists. On the synthetic head-on test the alarm is raised
0.4 s before contact; parallel traffic stays below 0.02. On the real test
clips the official alarm count is 0 (peaks 0.12–0.49). `make dev-eval`
reports the official Part B AP, alarm F1 and mTTA against human-reviewed
labels, so the weights can be refitted once clips are annotated. The metric
rewards precision: each false alarm lowers alarm F1.

## Calibrating the scene

All geometry lives in `config/camera_geometry.yaml`, in normalised
coordinates (`x/width`, `y/height`). The file is documented inline and loads
safely when empty. Invalid entries are skipped with a warning and never
crash the pipeline.

1. `make eda`: writes `outputs/eda/reference_median.jpg` (an empty-road median),
   vehicle and pedestrian heat maps, the learned flow field, and traffic-light
   candidates with suggested normalised ROIs (`eda.md`).
2. `make calibrate`: open `outputs/calibration/calibrate.html` in any browser.
   It is a self-contained page with no server and no network. Pick a shape
   type, click points, press *Finish*, and copy the generated YAML. Existing
   geometry is preloaded. Use `calibrate_geometry.py grid` for a
   coordinate-ruled frame instead.
3. Paste into `config/camera_geometry.yaml`, set lane `group`s and stop-line
   `signal`/`lanes`, then run `python scripts/calibrate_geometry.py render ...`
   and review `outputs/calibration/geometry_overlay.jpg`.
4. Set `calibrated: true`. Run `pytest tests/test_geometry.py` and `make visualize`.

Only stable scene facts belong in this file: lanes, legal directions, stop
lines, crossings, the signal head, solid markings and prohibited manoeuvres
(from `samples/camera.md`). Answers for particular videos never belong there.

**Learned scene prior.** `make prior` accumulates the direction field and
carriageway occupancy of all sample videos into `config/scene_prior.npz`. It
also saves an empty-road background, `config/background_reference.jpg`. Both
describe where and in which direction traffic normally moves. They store no
events, and are merged into each test video's own flow field.

## Models, weights, licences

| component | model / code | licence | notes |
|---|---|---|---|
| detector | YOLOX-M (GPU) / YOLOX-S (CPU), COCO, ONNX, 640×640 ([Megvii YOLOX 0.1.1rc0](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0)) | Apache-2.0 | 101 MB + 36 MB, `bash weights/download.sh` |
| tracker | clean-room reimplementation of ByteTrack's association ([ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack), MIT) | own code | no code copied |
| everything else | this repository | Apache-2.0 | |
| optional fire/smoke | any image classifier the team may legally use, as `weights/fire_smoke.onnx` | user-supplied | disabled when absent |

COCO classes used: person, bicycle, car, motorcycle, bus, truck, cat, dog,
horse, sheep, cow and traffic light. No external datasets were used for
training; no model was trained or fine-tuned. Weights total 137 MB, under the
5 GB limit. No paid API, hosted inference or online service is used at any
time. See `THIRD_PARTY_NOTICES.md`.

## Determinism

* `seed: 0` → `random`, NumPy and OpenCV RNGs are seeded (`src/determinism.py`).
  The pipeline has no stochastic step.
* ONNX Runtime: `use_deterministic_compute = True`, cuDNN algorithm search
  `DEFAULT` (no benchmarking).
* Hungarian assignment (SciPy), stable sorts, deterministic NMS ordering, and
  track ids assigned in detection order.
* Output sorted by `(start, label)`, rounded inward to 3 decimals.
* The runtime governor raises the stride only if processing becomes slower
  than 2.4× real time (logged as a warning), which should not happen on the
  target GPU. Tests assert identical outputs across repeated runs.

## Runtime

| measurement | value |
|---|---|
| YOLOX-S, one 1080p frame, 4-core CPU (this dev box) | 90 ms |
| YOLOX-M, one 1080p frame, 4-core CPU | 268 ms |
| 768×576 @ 10 fps, 79.5 s clip, CPU, Part A (stride 3) + Part B (every 3rd frame) | 29 s + 33 s = **0.78× duration** (measured) |
| 1080p @ 30 fps on a T4, Part A (stride 2, YOLOX-M) + Part B (every frame) | **≈ 1–1.5× duration (estimate; no GPU was available to measure)** |

Weights load once per process and are cached across videos. Decoding runs in
a background thread. If a machine turns out slower than budget, set
`part_a.stride_gpu: 3` and/or `risk.detect_every_gpu: 2` in `config/pipeline.yaml`.

## Rule-based vs learned components

* **Learned (pretrained, frozen):** YOLOX detector (COCO).
* **Estimated from data at run time, no training:** Kalman tracking, the
  per-video flow field and carriageway, signal state, background model.
* **Hand-written rules:** all 14 event decisions, segment boundaries and the
  risk combination. Every threshold is in `config/pipeline.yaml`.
* **Optional learned add-on:** a fire/smoke classifier (not shipped).

## Known failure cases and limitations

* **No calibration yet.** Seven classes are inactive until
  `camera_geometry.yaml` is filled in, and precision on the others is untuned
  because no sample footage was available. Every threshold is a documented
  starting point, not a tuned value.
* **Image-plane kinematics.** Scale-normalised speeds approximate but do not
  replace a ground-plane homography. Vehicles moving along the optical axis
  change scale fast, which can inflate speed noise far from the camera.
* **Occlusion.** Stationary vehicles hidden behind trucks are re-linked for
  gaps ≤ 8 s only. Long occlusions split `stopped_vehicle` events, which the
  3 s merge gap only partly repairs.
* **Accidents** need visible contact of raw footprints plus impact dynamics.
  Low-speed bumps with little velocity change, and single-vehicle crashes
  into fixed objects, are missed; the latter are disabled by default.
  Queues bumper-to-bumper under strong perspective overlap are guarded by the
  deceleration and jolt requirements, but remain the main false-positive risk.
* **Near misses** without visible braking or swerving (for example, the other
  party accelerates away) are missed.
* **Signals:** night glare, LED flicker aliasing or an ROI covering several
  heads can make the state `unknown`, which disables red-light rules for that
  video.
* **Congestion** in a video that is jammed from start to finish has no
  free-flow reference in its own tracks. It still fires, because thresholds
  are absolute plus relative, but the start is the video start.
* **Overhead cameras.** COCO-trained YOLOX rarely detects vehicles seen from
  directly above. If the challenge camera looks straight down, the detector
  must be swapped for one trained on aerial or top-down traffic data.
  `src/detection.py` accepts any YOLOX-format ONNX model via `detector.model`.
* **Learned direction needs a majority.** Without calibrated lanes,
  `wrong_way` abstains wherever less than 85 % of a cell's traffic agrees on a
  direction. Building the scene prior from the samples (`make prior`) fixes
  this in practice.
* **Small or distant objects** are missed at 640×640 input. Pedestrians far
  from the camera and debris are the weakest cases.
* **Same-class merging:** simultaneous events of one class, for example two
  vehicles stopped at once, become one segment, as the output format requires.

## Testing and quality

```bash
make check     # ruff (pyflakes, pycodestyle, bugbear, isort, pyupgrade) + pytest
```

The 70 tests cover:
* geometry: normalised conversion, containment, crossings, config validation
  and safe defaults;
* at least one synthetic trajectory scenario per event class, including
  negatives such as signal queues, gentle braking, allowed U-turns, crossings
  used legally and unreliable signals, with boundary-time assertions;
* segment merging, clamping and rounding safety;
* the output schema, including corrupt, missing or weightless inputs;
* runtime safety: CPU fallback to the small model and the Part B governor;
* Part B range, reset, determinism and **causality** (identical outputs when
  only future frames differ);
* signal classification, tracker identity and low-score rescue, YOLOX
  decoding, strided decoding, local metrics and the demo.

**The organisers' sample camera** (`clip_C3905.mp4`, 127.6 s, 4K). The first
run produced 6 events, which on inspection were all false:
* cars waiting at red lights;
* queued cars pulling away on green;
* two distant cars overlapping in the image;
* a 60 s red-light queue reported as congestion.

After calibrating the crossings, a first version of `failure_to_yield` fired
about 70 times. The fixes, each covered by a regression test:
* **near_miss:** braking only (turning is not swerving), plus a collision-course check;
* **accident:** the vehicles must stop together after contact;
* **stopped_vehicle:** "normal stopping places" learned from other vehicles are excluded;
* **congestion:** must last at least 90 s;
* **accident / near_miss:** ignore far-field vehicles;
* **failure_to_yield:** only a walking pedestrian well onto the zebra, within
  1.5 vehicle lengths in front of a moving vehicle;
* **jaywalking:** 1.5 body-scale slack around crossings. A standing "person"
  with a squat box, or one waiting in a vehicle stopping place, is treated as
  a rider whose scooter the detector missed. The last false alarm was a
  delivery rider waiting in the queue.

The final output is 4 `failure_to_yield` events: cars turning through the
lower crossing while pedestrians walk on it. Each was checked on the frames.

Official harness: 250 s against a 383 s budget on a 4-core CPU, VALID, no
false risk alarms.

**Real-footage checks** (clips used locally only, not redistributed):

| clip | result |
|---|---|
| OpenCV `vtest.avi`: fixed camera, pedestrian plaza, 79.5 s | no events, maximum risk 0.09 |
| `ahmetozlu/vehicle_counting_tensorflow` `input_video.mp4` (MIT): angled street camera, parked cars, 37.7 s | 20 tracks, no events; parked kerb-side cars correctly *not* reported as `stopped_vehicle` |
| same clip with seconds 8–20 appended **time-reversed** after 25 s, plus a scene prior learned from the original clip | exactly one `wrong_way` at 25.17–36.92 s (true start 24.99 s) |
| all three run through the **official** `run_submission.py` + `evaluate.py` (CPU) | every video within the 3× time budget (≤ 1.8×), output VALID, Score A = 1.0 on the hand-labelled set, 0 false risk alarms |
| `andrewssobral/simple_vehicle_counting` `video.avi`: near-overhead highway view, 320×176 | the COCO detector barely sees cars from directly above (see limitations) |

These checks found and fixed four real false positives:
* a car stopping beside a kerb-side parked car was flagged as a `near_miss`, because perspective made them look like they were closing on each other;
* a fixed learned-road threshold that failed once a prior inflated the counts;
* two risk alarms: one from a near and a far car converging only in the image, one from a 0.3 s spike.

Corrupt AVIs and missing files are handled gracefully.

## Local labels and evaluation

Sample videos are unlabeled, and this repository never treats them as
ground truth. To tune on them, a person must review candidates:

```bash
make candidates                     # loose thresholds -> outputs/candidates/*.mp4 + review.csv
# fill in accept / corrected label / start / end; add `manual` rows for missed events
python scripts/annotate_candidates.py labels --review outputs/candidates/review.csv --videos samples/
                                    # -> labels/dev_labels.json in the official ground-truth format
make predict dev-eval               # official Part A F1 @ tIoU 0.3/0.5/0.7 and Part B AP / alarm F1 / mTTA
```

**What the official metric implies** (`evaluate.py`):
* Part A averages F1 over every class in the ground truth **or in the
  predictions**. A single false event of a class that never occurs adds that
  class with F1 = 0, so precision-first rules are the right design.
* Part B scores the risk curve only if the test set contains accidents.
* Model score = 0.7 × A + 0.3 × B. The hackathon elimination score is
  0.6 × model + 0.25 × website + 0.15 × code.

## Upload demo

`make demo` (Streamlit) opens a page to upload an `.mp4`. It shows:
* a progress bar;
* event metrics and a segment table;
* **annotated playback** (H.264 via imageio-ffmpeg);
* a **clickable timeline** (click a bar to jump the video to it);
* the risk curve with a hover tooltip;
* JSON and CSV downloads.

The demo runs on CPU and imports the same code as `solution.py`, but the
evaluated entry points do not depend on it.

## Team

| member | role | links |
|---|---|---|
| **Amir Pulatov** | Project lead · UI/UX: direction, repository and starter-kit integration, testing on real footage (macOS) | [LinkedIn](https://www.linkedin.com/in/amir-pulatov-0ba608401) |
| **Roman Kim** | Computer Science, INHA University in Tashkent | [LinkedIn](https://www.linkedin.com/in/roman-kim-3054613a9) |
| **Madina Karimova** | Computer & Information Engineering, INHA University in Tashkent | [LinkedIn](https://www.linkedin.com/in/madinahon-karimova-409a952a0) |

Much of the implementation was written with Claude Code, an AI coding assistant, under the team's direction.

## Project website

`docs/index.html` is a self-contained static site covering the problem, method, results, engineering and team.
It has light and dark themes and works on phones. To publish it for free with GitHub Pages: open
**Settings → Pages**, set *Source* to **Deploy from a branch**, choose branch **main** and folder **/docs**, then
**Save**. The site appears at `https://amirkapopa.github.io/TrafficTrak/` within a minute or two.
