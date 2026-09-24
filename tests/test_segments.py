import math

from src.segments import CLASSES, RawEvent, finalize, merge_same_label, to_output, validate_segments


def ev(s, e, label="wrong_way"):
    return RawEvent(label, s, e)


def test_gap_merging_and_overlap():
    out = merge_same_label([ev(0, 2), ev(2.5, 4), ev(3.5, 6), ev(10, 11)], gap=1.0)
    assert [(e.start, e.end) for e in out] == [(0, 6), (10, 11)]


def test_finalize_clamps_filters_sorts():
    cfg = {"segments": {"merge_gap_sec": {"default": 0.5}, "min_duration_sec": {"default": 0.5, "stopped_vehicle": 10}}}
    raw = [ev(-1, 1.2), ev(1.5, 2.0), ev(5, 5.2), ev(20, 99), RawEvent("stopped_vehicle", 3, 9),
           RawEvent("stopped_vehicle", 30, 45), RawEvent("not_a_class", 1, 2), ev(float("nan"), 3)]
    out = finalize(raw, 30.0, cfg)
    assert to_output(out) == [[0.0, 2.0, "wrong_way"], [20.0, 30.0, "wrong_way"]]
    assert validate_segments(to_output(out), 30.0) == []


def test_same_class_never_overlaps_after_rounding():
    raw = [ev(1.00049, 2.00049), ev(2.0004, 3.0), RawEvent("accident", 1.5, 2.5)]
    cfg = {"segments": {"merge_gap_sec": {"default": 0.0}, "min_duration_sec": {"default": 0.1}}}
    out = to_output(finalize(raw, 10.0, cfg))
    assert validate_segments(out, 10.0) == []
    assert sum(1 for s in out if s[2] == "wrong_way") == 1
    assert out == sorted(out, key=lambda s: (s[0], s[2], s[1]))


def test_multiple_classes_may_overlap():
    out = to_output(finalize([ev(0, 5), RawEvent("accident", 1, 3)], 10.0, {}))
    assert len(out) == 2 and validate_segments(out, 10.0) == []


def test_empty_and_degenerate():
    assert finalize([], 10.0, {}) == []
    assert finalize([ev(0, 5)], 0.0, {}) == []
    assert finalize([ev(0, 5)], math.inf, {}) == []
    assert finalize([ev(5, 5)], 10.0, {}) == []


def test_end_never_exceeds_duration():
    out = to_output(finalize([ev(1, 12.3456789)], 12.3456789, {"segments": {"decimals": 3}}))
    assert out[0][1] <= 12.3456789


def test_validator_catches_problems():
    bad = [[2, 1, "accident"], [0, 1, "foo"], [0, 3, "near_miss"], [2, 4, "near_miss"], [0, 100, "congestion"], "x"]
    probs = validate_segments(bad, 50.0)
    text = " ".join(probs)
    assert "start < end" in text and "unknown label" in text and "overlaps" in text and "> duration" in text
    assert "expected [start_sec, end_sec, label]" in text
    assert validate_segments([], 10.0) == []


def test_class_list_matches_spec():
    assert CLASSES == ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn", "stopped_vehicle",
                       "jaywalking", "failure_to_yield", "illegal_turn", "solid_line_crossing", "stop_line",
                       "congestion", "road_obstacle", "fire_smoke"]
