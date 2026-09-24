"""End-to-end output contract of solution.detect_events."""

import json
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

import solution
from src.config import weights_dir
from src.segments import validate_segments

HAVE_WEIGHTS = any((weights_dir() / f).is_file() for f in ("yolox_s.onnx", "yolox_m.onnx"))


def write_video(path, n=40, fps=10.0, size=(320, 240)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        img = np.full((size[1], size[0], 3), 90, np.uint8)
        cv2.rectangle(img, (10 + 5 * i, 120), (60 + 5 * i, 150), (30, 30, 200), -1)
        vw.write(img)
    vw.release()
    return path


def test_interface_constants():
    assert solution.CLASSES[0] == "accident" and len(solution.CLASSES) == 14
    assert callable(solution.detect_events)
    assert hasattr(solution.RiskEstimator, "reset") and hasattr(solution.RiskEstimator, "step")


@pytest.mark.parametrize("path", ["/definitely/missing.mp4", ""])
def test_missing_video_returns_empty(path):
    assert solution.detect_events(path) == []


def test_corrupt_video_returns_empty(tmp_path):
    bad = tmp_path / "corrupt.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(2048))
    assert solution.detect_events(str(bad)) == []


def test_missing_weights_degrade_gracefully(tmp_path, monkeypatch):
    video = write_video(tmp_path / "v.mp4")
    empty = tmp_path / "noweights"
    empty.mkdir()
    code = f"import json, solution; print(json.dumps(solution.detect_events({str(video)!r})))"
    env = dict(os.environ, TRAFFICTRAK_WEIGHTS_DIR=str(empty))
    out = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(solution.__file__), env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0
    assert json.loads(out.stdout.strip().splitlines()[-1]) == []


@pytest.mark.skipif(not HAVE_WEIGHTS, reason="detector weights not downloaded")
def test_real_pipeline_output_is_valid(tmp_path):
    video = write_video(tmp_path / "v.mp4")
    segs = solution.detect_events(str(video))
    assert isinstance(segs, list)
    assert validate_segments(segs, 4.0) == []
    for s in segs:
        assert isinstance(s[0], float) and isinstance(s[1], float) and s[2] in solution.CLASSES
