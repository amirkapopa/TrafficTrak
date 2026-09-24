import math

import numpy as np
import pytest
import yaml

from src.config import DEFAULT_GEOMETRY_CONFIG
from src.geometry import Polygon, Polyline, angle_diff, build_geometry, load_geometry, segments_intersect, to_norm, to_px


def test_normalised_round_trip():
    pts = [[0.25, 0.5], [1.0, 0.0]]
    px = to_px(pts, 1920, 1080)
    assert np.allclose(px, [[480, 540], [1920, 0]])
    assert np.allclose(to_norm(px, 1920, 1080), pts)


def test_point_in_polygon_vectorised():
    sq = Polygon(np.array([[0, 0], [10, 0], [10, 10], [0, 10]]))
    inside = sq.contains(np.array([[5, 5], [15, 5], [-1, -1], [9.9, 0.1]]))
    assert inside.tolist() == [True, False, False, True]
    assert sq.signed_distance((5, 5)) == pytest.approx(5.0)
    assert sq.signed_distance((12, 5)) == pytest.approx(-2.0)
    assert sq.area() == pytest.approx(100.0)


def test_segment_intersection():
    assert segments_intersect((0, 0), (10, 10), (0, 10), (10, 0))
    assert not segments_intersect((0, 0), (1, 1), (2, 2), (3, 5))
    assert segments_intersect((0, 0), (2, 0), (1, 0), (3, 0))  # collinear overlap
    assert not segments_intersect((0, 0), (1, 0), (0, 1), (1, 1))  # parallel


def test_polyline_side_and_extent():
    line = Polyline(np.array([[0, 0], [10, 0], [20, 10]]))
    assert line.side((5, 5)) == -line.side((5, -5))
    sides, within = line.side_many(np.array([[5, 5], [5, -5], [-5, 1], [-0.5, 1]]), tol=1.0)
    assert sides[0] == -sides[1]
    assert within.tolist() == [True, True, False, True]
    s, d, _, _ = line.project((10, 5))
    assert d <= 5.0 and 0 <= s <= line.length
    assert line.crossed_by((5, -1), (5, 1))
    assert np.allclose(line.tangent_at((2, 1)), [1, 0])


def test_angle_diff_wraps():
    assert angle_diff(math.pi - 0.1, -math.pi + 0.1) == pytest.approx(0.2)
    assert angle_diff(0.0, math.pi) == pytest.approx(math.pi)


def test_shipped_geometry_is_safe_empty_default():
    with open(DEFAULT_GEOMETRY_CONFIG) as fh:
        data = yaml.safe_load(fh)
    g = build_geometry(data, 1920, 1080)
    assert g.warnings == []
    if not data.get("calibrated"):
        s = g.summary()
        assert s["lanes"] == 0 and s["stop_lines"] == 0 and s["signals"] == 0


def test_full_geometry_parses_and_scales():
    data = {
        "calibrated": True,
        "carriageway": [[[0, 0.5], [1, 0.5], [1, 1], [0, 1]]],
        "lanes": [{"id": "nb", "group": "north", "polygon": [[0.4, 1], [0.5, 1], [0.5, 0.5], [0.4, 0.5]],
                   "direction": [[0.45, 1.0], [0.45, 0.5]]}],
        "stop_lines": [{"id": "sl", "line": [[0.4, 0.6], [0.5, 0.6]], "lanes": ["nb"], "signal": "main"}],
        "signals": [{"id": "main", "roi": [0.6, 0.1, 0.62, 0.16], "layout": "vertical"}],
        "crossings": [{"id": "cw", "polygon": [[0.3, 0.52], [0.7, 0.52], [0.7, 0.56], [0.3, 0.56]]}],
        "intersection": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.5], [0.3, 0.5]],
        "solid_lines": [{"id": "c", "points": [[0.5, 1], [0.5, 0.5]]}],
        "prohibited_turns": [{"id": "t", "label": "illegal_turn", "from": [[0, 0], [0.1, 0], [0.1, 0.1]],
                              "to": [[0.5, 0], [0.6, 0], [0.6, 0.1]]}],
    }
    for w, h in ((1920, 1080), (1280, 720)):
        g = build_geometry(data, w, h)
        assert g.warnings == []
        lane = g.lane_of((0.45 * w, 0.8 * h))
        assert lane is not None and lane.id == "nb"
        assert lane.direction_at((0.45 * w, 0.8 * h))[1] < 0  # travelling up the image
        sl = g.stop_lines[0]
        assert sl.approach[1] < 0  # derived from lane direction
        assert sl.side((0.45 * w, 0.7 * h)) == -1 and sl.side((0.45 * w, 0.55 * h)) == 1
        assert g.on_carriageway(np.array([[0.1 * w, 0.9 * h], [0.1 * w, 0.1 * h]])).tolist() == [True, False]
        assert g.signals[0].roi[0] == int(0.6 * w)


def test_invalid_entries_are_skipped_not_fatal():
    data = {"lanes": [{"id": "bad", "polygon": [[0, 0], [1, 1]], "direction": [[0, 0], [1, 1]]}],
            "carriageway": [[[0, 0], [2.0, 0], [0, 1]]],
            "stop_lines": [{"id": "x", "line": [[0, 0], [1, 0]]}]}
    g = build_geometry(data, 100, 100)
    assert g.lanes == [] and g.carriageway == [] and g.stop_lines == []
    assert len(g.warnings) == 3


def test_missing_or_corrupt_file(tmp_path):
    assert load_geometry(tmp_path / "missing.yaml", 640, 480).summary()["lanes"] == 0
    bad = tmp_path / "bad.yaml"
    bad.write_text("- just\n- a list\n")
    assert load_geometry(bad, 640, 480).summary()["carriageway"] == 0
