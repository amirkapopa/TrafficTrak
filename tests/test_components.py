"""Signal-state classifier, tracker identity stability, video reader robustness."""

import cv2
import numpy as np

from src.config import load_config
from src.detection import COCO_NAMES, Detections, decode_yolox, letterbox, postprocess
from src.geometry import SignalSpec
from src.signal_state import SignalTimeline, classify_signal
from src.tracking import ByteTracker
from src.video import FrameReader, effective_duration, probe


# ------------------------------------------------------------------ signal
def signal_frame(lit):
    img = np.full((200, 200, 3), 40, np.uint8)
    cv2.rectangle(img, (90, 20), (110, 110), (20, 20, 20), -1)
    colours = {"red": (0, 0, 255), "amber": (0, 190, 255), "green": (60, 255, 0)}
    for k, name in enumerate(("red", "amber", "green")):
        c = colours[name] if name == lit else (35, 35, 35)
        cv2.circle(img, (100, 35 + 30 * k), 9, c, -1)
    return img


SPEC = SignalSpec("main", (90, 20, 111, 111), "vertical", ["red", "amber", "green"])


def test_signal_classifier_reads_lit_lamp():
    for state in ("red", "amber", "green"):
        assert classify_signal(signal_frame(state), SPEC) == state
    assert classify_signal(signal_frame(None), SPEC) == "unknown"


def test_signal_timeline_smoothing_and_queries():
    tl = SignalTimeline("s", smooth_window=1.0)
    seq = ["red"] * 20 + ["unknown"] + ["red"] * 5 + ["green"] * 30
    for i, s in enumerate(seq):
        tl.add(i * 0.2, s)
    assert tl.state_at(4.0) == "red"  # single unknown sample filtered
    assert tl.reliability > 0.9
    assert abs(tl.next_change(0.0, "green") - 26 * 0.2) < 0.41
    assert tl.is_state_throughout("red", 1.0, 3.0)
    assert not tl.is_state_throughout("red", 4.0, 7.0)
    assert [iv[2] for iv in tl.intervals()] == ["red", "green"]


# ------------------------------------------------------------------ tracking
def dets(boxes, cls="car", score=0.9):
    b = np.array(boxes, np.float32).reshape(-1, 4)
    return Detections(b, np.full(len(b), score, np.float32), np.full(len(b), COCO_NAMES.index(cls), np.int64))


def test_tracker_keeps_identities_and_survives_misses():
    tr = ByteTracker(load_config(), fps=15.0)
    for i in range(30):
        x = 10 + 8 * i
        boxes = [[x, 100, x + 60, 140], [500 - 6 * i, 200, 560 - 6 * i, 240]]
        if i in (12, 13):  # detector misses car 1 for two frames
            boxes = boxes[1:]
        tr.update(dets(boxes), i / 15.0, i)
    tracks = tr.all_tracks()
    assert len(tracks) == 2
    assert all(len(t.t_hist) >= 28 for t in tracks)


def test_tracker_low_score_rescue_and_no_cross_group_match():
    tr = ByteTracker(load_config(), fps=15.0)
    for i in range(10):
        tr.update(dets([[100 + 2 * i, 100, 160 + 2 * i, 140]], score=0.9 if i < 5 else 0.3), i / 15.0, i)
    assert len(tr.all_tracks()) == 1 and len(tr.all_tracks()[0].t_hist) == 10
    tr2 = ByteTracker(load_config(), fps=15.0)
    tr2.update(dets([[100, 100, 160, 140]], "car"), 0.0, 0)
    tr2.update(dets([[100, 100, 160, 140]], "car"), 0.07, 1)
    tr2.update(dets([[100, 100, 160, 140]], "person"), 0.13, 2)
    assert all(t.cls_name == "car" for t in tr2.all_tracks() if t.activated)


def test_yolox_decoding_shapes():
    img, r = letterbox(np.zeros((360, 640, 3), np.uint8), (640, 640))
    assert img.shape == (640, 640, 3) and r == 1.0
    raw = np.zeros((8400, 85), np.float32)
    raw[0, :4] = [0.5, 0.5, np.log(4), np.log(4)]  # stride-8 cell (0,0): 32x32 box centred at (4, 4)
    raw[0, 4] = 1.0
    raw[0, 5 + COCO_NAMES.index("car")] = 0.95
    d = postprocess(decode_yolox(raw, (640, 640)), 1.0, (360, 640), 0.1, 0.5, 1.0)
    assert len(d) == 1 and d.names() == ["car"]
    assert np.allclose(d.boxes[0], [0, 0, 20, 20])  # [-12, 20] clipped to the frame


# ------------------------------------------------------------------ video
def test_video_reader_stride_and_duration(tmp_path):
    path = str(tmp_path / "v.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (64, 48))
    for i in range(50):
        vw.write(np.full((48, 64, 3), i * 4, np.uint8))
    vw.release()
    meta = probe(path)
    assert meta.ok and abs(meta.fps - 20.0) < 1e-6 and meta.width == 64
    reader = FrameReader(path, meta.fps, stride=3, n_frames_hint=meta.n_frames)
    got = [(i, t) for i, t, _ in reader]
    assert [i for i, _ in got] == list(range(0, 50, 3))
    assert all(abs(t - i / 20.0) < 1e-9 for i, t in got)
    assert reader.read_failures == 0
    assert abs(effective_duration(meta, reader.frames_decoded) - 2.5) < 1e-6


def test_probe_handles_garbage(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    assert not probe(str(bad)).ok
    assert not probe(str(tmp_path / "missing.mp4")).ok
