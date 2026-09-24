"""Event-rule state machines on synthetic trajectories (1280x720, 15 Hz).

A car box is 80x50 px, i.e. one scale unit ~ 63 px.
"""

import math

import numpy as np
import pytest

from src.signal_state import SignalTimeline

from .synth import H, W, labels, linear, make_track, norm, of, piecewise, run

FULL_ROAD = {"carriageway": [norm([(0, 0), (W, 0), (W, H), (0, H)])]}


def lane_traffic(start_tid, y, times, speed=300.0, x0=0.0):
    return [make_track(start_tid + i, "car", linear((x0, y), (speed, 0), t), t, t + (W - x0) / speed)
            for i, t in enumerate(times)]


def near(value, target, tol):
    return abs(value - target) <= tol


# --------------------------------------------------------------------------- normal traffic
def test_free_flowing_traffic_has_no_events():
    tracks = lane_traffic(1, 400, np.arange(0, 40, 2.0)) + lane_traffic(100, 480, np.arange(1, 40, 2.5))
    tracks.append(make_track(500, "person", linear((50, 700), (40, 0), 0), 0, 25, size=(30, 80)))
    events, _, _ = run(tracks, 50.0)
    assert events == []


# --------------------------------------------------------------------------- stopped_vehicle
def test_stopped_vehicle_with_passing_traffic():
    a = make_track(1, "car", piecewise([(10, 100, 500), (15, 600, 500), (45, 600, 500), (50, 1100, 500)]), 10, 50)
    early = lane_traffic(10, 500, [0, 2, 4, 6, 8])
    passers = lane_traffic(100, 560, np.arange(0, 50, 3.0))
    events, _, _ = run([a] + early + passers, 55.0)
    sv = of(events, "stopped_vehicle")
    assert len(sv) == 1
    assert near(sv[0].start, 15.0, 1.0) and near(sv[0].end, 45.0, 1.0)


def test_signal_queue_is_not_a_stopped_vehicle():
    a = make_track(1, "car", piecewise([(10, 100, 500), (15, 600, 500), (45, 600, 500), (50, 1100, 500)]), 10, 50)
    b = make_track(2, "car", piecewise([(10, 0, 500), (15, 510, 500), (45, 510, 500), (50, 1000, 500)]), 10, 50)
    passers = lane_traffic(100, 600, np.arange(0, 50, 3.0))
    events, _, _ = run([a, b] + passers, 55.0, geometry=FULL_ROAD)
    assert "stopped_vehicle" not in labels(events)


def test_short_stop_is_ignored():
    a = make_track(1, "car", piecewise([(10, 100, 500), (15, 600, 500), (22, 600, 500), (27, 1100, 500)]), 10, 27)
    passers = lane_traffic(100, 560, np.arange(0, 30, 3.0))
    events, _, _ = run([a] + passers, 30.0, geometry=FULL_ROAD)
    assert "stopped_vehicle" not in labels(events)


# --------------------------------------------------------------------------- wrong_way
def test_wrong_way_from_learned_flow():
    normal = lane_traffic(1, 400, np.arange(0, 30, 3.0))
    wrong = make_track(99, "car", linear((W, 400), (-300, 0), 31), 31, 31 + W / 300)
    events, _, _ = run(normal + [wrong], 40.0)
    ww = of(events, "wrong_way")
    assert len(ww) == 1
    assert ww[0].tracks == (99,) or 99 in ww[0].tracks
    assert near(ww[0].start, 31.0, 0.7) and near(ww[0].end, 31 + W / 300, 0.7)


def test_wrong_way_needs_support_without_lanes():
    wrong = make_track(99, "car", linear((W, 400), (-300, 0), 1), 1, 1 + W / 300)
    events, _, _ = run([wrong], 10.0)
    assert "wrong_way" not in labels(events)


def test_wrong_way_with_calibrated_lanes():
    geo = {"lanes": [{"id": "eb", "group": "eastbound", "polygon": norm([(0, 370), (W, 370), (W, 430), (0, 430)]),
                      "direction": norm([(0, 400), (W, 400)])}]}
    wrong = make_track(7, "car", linear((W, 400), (-300, 0), 2), 2, 2 + W / 300)
    ok = make_track(8, "car", linear((0, 400), (300, 0), 2), 2, 2 + W / 300)
    events, _, _ = run([wrong, ok], 10.0, geometry=geo)
    ww = of(events, "wrong_way")
    assert len(ww) == 1 and ww[0].tracks == (7,)
    assert near(ww[0].start, 2.0, 0.5)


# --------------------------------------------------------------------------- congestion
def test_congestion_from_standing_queue():
    movers = lane_traffic(1, 400, np.arange(0, 28, 2.0)) + lane_traffic(50, 450, np.arange(1, 28, 2.0))
    jam = []
    for i, x in enumerate(range(100, 900, 100)):
        jam.append(make_track(200 + i, "car", piecewise([(30, x, 420), (100, x + 350, 420), (105, x + 1500, 420)]), 30, 105))
    events, _, _ = run(movers + jam, 110.0)
    cg = of(events, "congestion")
    assert len(cg) == 1
    assert near(cg[0].start, 30.0, 2.0) and near(cg[0].end, 100.0, 3.0)
    assert "stopped_vehicle" not in labels(events)


# --------------------------------------------------------------------------- pedestrians
PED_GEO = {
    "carriageway": [norm([(0, 300), (W, 300), (W, 600), (0, 600)])],
    "crossings": [{"id": "cw", "polygon": norm([(900, 290), (1000, 290), (1000, 610), (900, 610)])}],
}


def test_jaywalking_outside_crossing_only():
    jay = make_track(1, "person", linear((400, 250), (0, 50), 5), 5, 13, size=(30, 80))
    legal = make_track(2, "person", linear((950, 250), (0, 50), 5), 5, 13, size=(30, 80))
    events, _, _ = run([jay, legal], 20.0, geometry=PED_GEO)
    jw = of(events, "jaywalking")
    assert len(jw) == 1 and jw[0].tracks == (1,)
    assert near(jw[0].start, 6.1, 0.5) and near(jw[0].end, 12.0, 0.5)


def test_jaywalking_disabled_without_calibrated_carriageway():
    jay = make_track(1, "person", linear((400, 250), (0, 50), 5), 5, 13, size=(30, 80))
    events, _, _ = run([jay], 20.0)
    assert "jaywalking" not in labels(events)


def test_failure_to_yield():
    geo = {"carriageway": [norm([(0, 300), (W, 300), (W, 600), (0, 600)])],
           "crossings": [{"id": "cw", "polygon": norm([(600, 300), (700, 300), (700, 600), (600, 600)])}]}
    ped = make_track(1, "person", linear((680, 280), (0, 30), 0), 0, 10, size=(30, 80))
    car = make_track(2, "car", linear((0, 520), (150, 0), 0), 0, 8)
    late_car = make_track(3, "car", linear((0, 520), (150, 0), 12), 12, 20)  # crossing empty by then
    events, _, _ = run([ped, car, late_car], 25.0, geometry=geo)
    fy = of(events, "failure_to_yield")
    assert len(fy) == 1 and fy[0].tracks == (2,)
    assert near(fy[0].start, 3.8, 0.4) and near(fy[0].end, 4.85, 0.4)


# --------------------------------------------------------------------------- signals
def signal_timeline(red_until=20.0, total=60.0):
    tl = SignalTimeline("main")
    for t in np.arange(0, total, 0.2):
        tl.add(float(t), "red" if t < red_until else "green")
    return tl


SIG_GEO = {
    "carriageway": [norm([(0, 350), (W, 350), (W, 550), (0, 550)])],
    "stop_lines": [{"id": "sl", "line": norm([(600, 350), (600, 550)]), "approach": norm([(500, 450), (600, 450)]),
                    "signal": "main"}],
    "intersection": norm([(620, 350), (900, 350), (900, 550), (620, 550)]),
}


def test_red_light_violation():
    runner = make_track(1, "car", linear((100, 450), (150, 0), 5), 5, 12)
    on_green = make_track(2, "car", linear((100, 450), (150, 0), 25), 25, 32)
    events, _, _ = run([runner, on_green], 40.0, geometry=SIG_GEO, signals={"main": signal_timeline()})
    rl = of(events, "red_light")
    assert len(rl) == 1 and rl[0].tracks == (1,)
    assert near(rl[0].start, 8.07, 0.3) and near(rl[0].end, 10.33, 0.4)


def test_stop_line_violation():
    car = make_track(1, "car", piecewise([(5, 100, 450), (8.1, 565, 450), (8.3, 575, 450), (25, 575, 450), (28, 1000, 450)]), 5, 28)
    events, _, _ = run([car], 40.0, geometry=SIG_GEO, signals={"main": signal_timeline()})
    sl = of(events, "stop_line")
    assert len(sl) == 1
    assert near(sl[0].start, 8.4, 0.7) and near(sl[0].end, 20.0, 0.3)
    assert "red_light" not in labels(events)


def test_signal_rules_disabled_when_signal_unreliable():
    tl = SignalTimeline("main")
    for t in np.arange(0, 40, 0.2):
        tl.add(float(t), "unknown")
    runner = make_track(1, "car", linear((100, 450), (150, 0), 5), 5, 12)
    events, _, _ = run([runner], 40.0, geometry=SIG_GEO, signals={"main": tl})
    assert "red_light" not in labels(events)


# --------------------------------------------------------------------------- markings & turns
def test_solid_line_crossing():
    geo = {"solid_lines": [{"id": "centre", "points": norm([(640, 0), (640, H)])}]}
    car = make_track(1, "car", piecewise([(0, 600, 700), (4, 600, 400), (6, 700, 200), (9, 700, -100)]), 0, 8.5)
    events, _, _ = run([car], 12.0, geometry=geo)
    sl = of(events, "solid_line_crossing")
    assert len(sl) == 1
    assert 4.0 <= sl[0].start <= 5.2 and 4.8 <= sl[0].end <= 6.3 and sl[0].end > sl[0].start


def u_turn_path(t):
    if t <= 3:
        return 200 + 200 * t, 500.0
    if t <= 6:
        a = -math.pi / 2 + (t - 3) / 3 * math.pi  # from bottom (y=500) around the right side to top (y=340)
        return 800 + 80 * math.cos(a), 420 - 80 * math.sin(a)
    return 800 - 200 * (t - 6), 340.0


def test_illegal_u_turn_when_prohibited():
    geo = dict(FULL_ROAD, u_turn_prohibited=True)
    car = make_track(1, "car", u_turn_path, 0, 9)
    events, _, _ = run([car], 12.0, geometry=geo)
    ut = of(events, "illegal_u_turn")
    assert len(ut) == 1
    assert 2.5 <= ut[0].start <= 4.0 and 5.5 <= ut[0].end <= 7.5


def test_u_turn_not_reported_when_allowed():
    car = make_track(1, "car", u_turn_path, 0, 9)
    events, _, _ = run([car], 12.0, geometry=FULL_ROAD)
    assert "illegal_u_turn" not in labels(events)


def test_prohibited_turn():
    geo = {"prohibited_turns": [{"id": "no_left", "label": "illegal_turn",
                                 "from": norm([(0, 450), (520, 450), (520, 550), (0, 550)]),
                                 "to": norm([(550, 0), (650, 0), (650, 300), (550, 300)])}]}
    turner = make_track(1, "car", piecewise([(0, 0, 500), (4, 600, 500), (8, 600, 100)]), 0, 8)
    straight = make_track(2, "car", linear((0, 500), (150, 0), 0), 0, 8)
    events, _, _ = run([turner, straight], 10.0, geometry=geo)
    it = of(events, "illegal_turn")
    assert len(it) == 1 and it[0].tracks == (1,)
    assert 3.0 <= it[0].start <= 4.5 and 5.8 <= it[0].end <= 8.0


# --------------------------------------------------------------------------- collisions
def test_rear_end_accident():
    a = make_track(1, "car", piecewise([(0, 100, 450), (2.6, 620, 450), (2.8, 630, 450), (20, 630, 450)]), 0, 20)
    b = make_track(2, "car", piecewise([(0, 700, 450), (2.6, 700, 450), (2.9, 730, 450), (20, 730, 450)]), 0, 20)
    events, _, _ = run([a, b], 22.0)
    acc = of(events, "accident")
    assert len(acc) == 1
    assert near(acc[0].start, 2.6, 0.5)
    assert 3.0 <= acc[0].end <= 5.0
    assert "near_miss" not in labels(events)


def test_near_miss_hard_braking():
    a = make_track(1, "car", piecewise([(0, 0, 450), (2.0, 500, 450), (2.7, 575, 450), (10, 575, 450)]), 0, 10)
    b = make_track(2, "car", piecewise([(0, 700, 450), (6, 700, 450), (9, 1200, 450)]), 0, 9)
    events, _, _ = run([a, b], 12.0)
    nm = of(events, "near_miss")
    assert len(nm) == 1
    assert 1.2 <= nm[0].start <= 2.3 and 2.4 <= nm[0].end <= 4.0
    assert "accident" not in labels(events)


def test_gentle_queue_formation_is_not_a_conflict():
    def gentle(t):  # 150 px/s, then decelerate uniformly over 4 s to stop 60 px behind the leader
        if t <= 2:
            return 100 + 150 * t, 450.0
        tt = min(t - 2, 4.0)
        return 400 + 150 * tt - 150 / 8 * tt * tt, 450.0
    a = make_track(1, "car", gentle, 0, 12)
    b = make_track(2, "car", linear((760, 450), (0, 0), 0), 0, 12)
    events, _, _ = run([a, b], 14.0)
    assert "accident" not in labels(events) and "near_miss" not in labels(events)


# --------------------------------------------------------------------------- obstacles
def test_animal_on_road():
    dog = make_track(1, "dog", linear((300, 350), (40, 0), 5), 5, 10, size=(60, 40))
    events, _, _ = run([dog], 15.0, geometry=FULL_ROAD)
    ro = of(events, "road_obstacle")
    assert len(ro) == 1 and near(ro[0].start, 5.0, 0.3) and near(ro[0].end, 10.0, 0.3)


# --------------------------------------------------------------------------- determinism / schema
def test_rules_are_deterministic():
    tracks = lane_traffic(1, 400, np.arange(0, 30, 3.0)) + [
        make_track(99, "car", linear((W, 400), (-300, 0), 31), 31, 31 + W / 300)]
    e1, _, _ = run(tracks, 40.0)
    tracks = lane_traffic(1, 400, np.arange(0, 30, 3.0)) + [
        make_track(99, "car", linear((W, 400), (-300, 0), 31), 31, 31 + W / 300)]
    e2, _, _ = run(tracks, 40.0)
    assert [(e.start, e.end, e.label) for e in e1] == [(e.start, e.end, e.label) for e in e2]


@pytest.mark.parametrize("label", ["wrong_way", "stopped_vehicle"])
def test_disabled_class_is_never_emitted(label):
    normal = lane_traffic(1, 400, np.arange(0, 30, 3.0))
    wrong = make_track(99, "car", linear((W, 400), (-300, 0), 31), 31, 31 + W / 300)
    events, _, _ = run(normal + [wrong], 40.0, overrides={"classes": {label: {"enabled": False}}})
    assert label not in labels(events)
