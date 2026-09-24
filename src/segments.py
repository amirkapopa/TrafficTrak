"""Event segments: gap merging, duration filtering, clamping, same-class
overlap resolution and output-schema validation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

CLASSES = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle",
    "fire_smoke",
]


@dataclass
class RawEvent:
    label: str
    start: float
    end: float
    score: float = 1.0
    tracks: tuple = ()
    note: str = ""
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"label": self.label, "start": round(self.start, 3), "end": round(self.end, 3),
                "score": round(float(self.score), 3), "tracks": [int(t) for t in self.tracks], "note": self.note}


def _per_class(table: dict, label: str, default: float) -> float:
    if not isinstance(table, dict):
        return default
    return float(table.get(label, table.get("default", default)))


def merge_same_label(events: list[RawEvent], gap: float) -> list[RawEvent]:
    """Merge events of one label that overlap or are separated by <= gap."""
    out: list[RawEvent] = []
    for ev in sorted(events, key=lambda e: (e.start, e.end)):
        if out and ev.start <= out[-1].end + gap:
            last = out[-1]
            last.end = max(last.end, ev.end)
            last.score = max(last.score, ev.score)
            last.tracks = tuple(sorted(set(last.tracks) | set(ev.tracks)))
            if ev.note and ev.note not in last.note:
                last.note = (last.note + "; " + ev.note).strip("; ")
        else:
            out.append(RawEvent(ev.label, ev.start, ev.end, ev.score, tuple(ev.tracks), ev.note, dict(ev.extra)))
    return out


def finalize(events: list[RawEvent], duration: float, cfg: dict | None = None) -> list[RawEvent]:
    """Clamp, merge, filter and sort raw events into non-overlapping segments."""
    scfg = (cfg or {}).get("segments", {})
    gaps = scfg.get("merge_gap_sec", {"default": 1.0})
    mins = scfg.get("min_duration_sec", {"default": 0.5})
    decimals = int(scfg.get("decimals", 3))
    if not (duration > 0 and math.isfinite(duration)):
        return []
    q = 10 ** decimals
    by_label: dict[str, list[RawEvent]] = {}
    for ev in events:
        if ev.label not in CLASSES:
            continue
        if not (math.isfinite(ev.start) and math.isfinite(ev.end)):
            continue
        s, e = max(0.0, ev.start), min(duration, ev.end)
        if e <= s:
            continue
        by_label.setdefault(ev.label, []).append(RawEvent(ev.label, s, e, ev.score, ev.tracks, ev.note, ev.extra))
    final: list[RawEvent] = []
    for label in sorted(by_label):
        merged = merge_same_label(by_label[label], _per_class(gaps, label, 1.0))
        min_d = _per_class(mins, label, 0.5)
        kept = []
        for ev in merged:
            if ev.end - ev.start < min_d:
                continue
            # quantise inward so rounding can never create overlap / out-of-range
            ev.start = math.ceil(ev.start * q - 1e-6) / q
            ev.end = math.floor(ev.end * q + 1e-6) / q
            ev.end = min(ev.end, math.floor(duration * q) / q)
            if ev.end <= ev.start:
                continue
            if kept and ev.start < kept[-1].end:
                kept[-1].end = max(kept[-1].end, ev.end)
                continue
            kept.append(ev)
        final.extend(kept)
    final.sort(key=lambda e: (e.start, e.label, e.end))
    return final


def to_output(events: list[RawEvent]) -> list[list]:
    return [[float(e.start), float(e.end), e.label] for e in events]


def validate_segments(segs, duration: float | None = None) -> list[str]:
    """Return a list of schema problems (empty list = valid)."""
    problems = []
    if not isinstance(segs, list):
        return ["prediction must be a list"]
    last_end: dict[str, float] = {}
    for i, seg in enumerate(sorted(
            [s for s in segs if isinstance(s, (list, tuple)) and len(s) == 3 and isinstance(s[0], (int, float))
             and isinstance(s[1], (int, float))],
            key=lambda s: (s[2] if isinstance(s[2], str) else "", s[0]))):
        s, e, label = seg
        if label not in CLASSES:
            problems.append(f"segment {i}: unknown label {label!r}")
        if not (math.isfinite(s) and math.isfinite(e)):
            problems.append(f"segment {i}: non-finite time")
            continue
        if not (0 <= s < e):
            problems.append(f"segment {i}: need 0 <= start < end, got {s}, {e}")
        if duration is not None and e > duration + 1e-6:
            problems.append(f"segment {i}: end {e} > duration {duration}")
        if label in last_end and s < last_end[label]:
            problems.append(f"segment {i}: overlaps previous {label} segment")
        last_end[label] = max(last_end.get(label, 0.0), e)
    for i, seg in enumerate(segs):
        if not (isinstance(seg, (list, tuple)) and len(seg) == 3):
            problems.append(f"entry {i}: expected [start_sec, end_sec, label]")
        elif not (isinstance(seg[0], (int, float)) and isinstance(seg[1], (int, float))) \
                or isinstance(seg[0], bool) or isinstance(seg[1], bool):
            problems.append(f"entry {i}: start/end must be numbers")
    return problems
