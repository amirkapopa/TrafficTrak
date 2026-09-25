# TrafficTrak: transparent traffic-event detection for a fixed CCTV camera

**Task.** Given a video from one fixed road camera, report every traffic
event of 14 classes as a precise time segment (Part A), and, frame by
frame, the probability that an accident starts within the next 5 seconds
(Part B). The system runs fully offline on one T4-class GPU, within 3× the
video duration, using only open weights.

**Approach.** One learned component does the perceiving; everything that
decides is explicit and auditable.

1. **Perception.** YOLOX-M (Apache-2.0, COCO-pretrained, ONNX Runtime with
   deterministic kernels) detects road users on every second frame. A
   ByteTrack-style tracker links them. It matches high-confidence boxes first
   and uses low-confidence boxes to keep tracks alive through partial
   occlusion. Stationary objects that lose their track behind passing trucks
   are re-linked afterwards.
2. **Kinematics without calibration.** Tracks are resampled onto a common
   time grid and smoothed. Speed, acceleration and heading are expressed in
   *scale units*, the object's own box size. One threshold therefore means
   the same physical thing near and far from the camera, without a
   ground-plane homography.
3. **Scene model.** The camera never moves, so where traffic flows and in
   which direction are stable facts. They are learned per video from its
   own tracks, leave-one-out so a wrong-way driver cannot vote for itself.
   They can be reinforced with a prior accumulated over all sample videos.
   Humans can add exact structure (lanes, stop lines, crossings, the signal
   head, solid markings, prohibited turns) in one YAML file with a
   click-to-draw browser tool.
4. **Rules as state machines.** Each class has a documented rule with a
   start/end convention matching the official definitions. Examples:
   * a *stopped vehicle* must stand still for ≥ 10 s on the road, while
     traffic keeps passing it, and not as part of a queue or at a red light;
   * an *accident* needs visible contact, prior closing speed, an abrupt
     deceleration and a second impact cue, and it ends when everyone
     involved has stopped.

   A rule whose prerequisites are missing stays silent instead of guessing.
5. **Segments.** Fragments are merged across short gaps and blips are
   dropped. Same-class overlaps are merged and times are rounded inward, so
   every output satisfies `0 ≤ start < end ≤ duration`.
6. **Causal risk.** Every frame, each nearby pair of road users is scored on:
   * predicted conflict: time and distance of closest approach, and closing speed;
   * evasive action: hard braking or a swerve;
   * rule violations: wrong-way driving, a pedestrian on the road, running a red.

   A logistic combination is built so that no single cue crosses 0.5. Fast
   attack, slow decay and a 0.5 s hysteresis keep one noisy frame from
   raising an alarm.

**Engineering.**
* Fixed seeds and deterministic ordering throughout; the same input always
  gives the same output.
* Corrupt, missing or empty videos and missing weights degrade to an empty
  result and never crash the harness.
* 70 unit tests, including one synthetic scenario per event class and a
  causality test for Part B, plus checks on real street footage.
* An upload demo with annotated playback, a clickable timeline and the risk curve.
* Weights: 137 MB.

**Honest status.** The sample videos and camera description were not
available during development. The scene file therefore ships uncalibrated,
and seven geometry-dependent classes (red light, stop line, crossing,
marking and turn rules) activate only after calibration. All thresholds
are principled defaults awaiting tuning on human-reviewed clips; tooling for
that review loop is included. The main open risks:
* perspective effects on image-plane kinematics;
* long occlusions;
* accidents without a clear change in velocity.
