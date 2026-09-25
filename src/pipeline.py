"""Part A orchestration: decode -> detect -> track -> monitors -> features ->
rules -> segments.  One decoding pass per video; weights are cached."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import geometry_config_path, load_config, resolve_path
from .detection import VEHICLE_CLASSES, get_detector
from .determinism import set_determinism
from .features import TrackSeries, build_all_series
from .flow import FlowField
from .geometry import Geometry, load_geometry
from .monitors import FireSmokeMonitor, ObstacleMonitor
from .rules import SceneContext, run_rules
from .segments import RawEvent, finalize
from .signal_state import SignalTimeline, classify_signal
from .tracking import ByteTracker, stitch_stationary
from .video import FrameReader, VideoMeta, effective_duration, probe

log = logging.getLogger("traffictrak.pipeline")

ProgressFn = Callable[[float, str], None]


@dataclass
class Analysis:
    meta: VideoMeta
    duration: float = 0.0
    dt: float = 0.0
    stride: int = 1
    events: list[RawEvent] = field(default_factory=list)
    raw_events: list[RawEvent] = field(default_factory=list)
    series: list[TrackSeries] = field(default_factory=list)
    frame_tracks: dict[int, list[tuple[int, str, list[float]]]] = field(default_factory=dict)
    signals: dict[str, SignalTimeline] = field(default_factory=dict)
    flow: FlowField | None = None
    geometry: Geometry | None = None
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def segments(self) -> list[list]:
        return [[float(e.start), float(e.end), e.label] for e in self.events]


_PRIOR_CACHE: dict[str, tuple[FlowField | None, np.ndarray | None]] = {}


def load_scene_prior(cfg: dict) -> tuple[FlowField | None, np.ndarray | None]:
    """Optional learned scene prior (flow field + empty-road background)."""
    sc = cfg.get("scene", {})
    key = f"{sc.get('prior_path')}|{sc.get('background_path')}"
    if key not in _PRIOR_CACHE:
        flow, bg = None, None
        p = resolve_path(sc.get("prior_path"))
        if p is not None and p.is_file():
            flow = FlowField.load(p)
        b = resolve_path(sc.get("background_path"))
        if b is not None and b.is_file():
            bg = cv2.imread(str(b), cv2.IMREAD_COLOR)
        _PRIOR_CACHE[key] = (flow, bg)
    return _PRIOR_CACHE[key]


def choose_stride(cfg: dict, on_gpu: bool) -> int:
    pa = cfg.get("part_a", {})
    return max(1, int(pa.get("stride_gpu", 2) if on_gpu else pa.get("stride_cpu", 3)))


def analyze_video(video_path: str, cfg: dict | None = None, geometry_path: str | None = None,
                  progress: ProgressFn | None = None, keep_frame_tracks: bool = False,
                  detector=None) -> Analysis:
    """Run the full Part A pipeline on one video.  Never raises for bad input."""
    t_start = time.perf_counter()
    cfg = cfg or load_config()
    set_determinism(int(cfg.get("seed", 0)))
    meta = probe(video_path, float(cfg.get("video", {}).get("default_fps", 25.0)))
    result = Analysis(meta=meta)
    if not meta.ok:
        result.warnings.append(f"unreadable video: {meta.error}")
        log.warning("%s: %s", video_path, meta.error)
        return result
    geometry = load_geometry(geometry_path or geometry_config_path(), meta.width, meta.height)
    result.geometry = geometry
    if detector is None:
        detector = get_detector(cfg)
    on_gpu = bool(getattr(detector, "on_gpu", False))
    stride = choose_stride(cfg, on_gpu)
    dt = stride / meta.fps
    result.stride, result.dt = stride, dt

    tracker = ByteTracker(cfg, meta.fps)
    prior_flow, background = load_scene_prior(cfg)
    obstacle = ObstacleMonitor(cfg, geometry, meta.width, meta.height, background)
    fire = FireSmokeMonitor(cfg)
    sig_cfg = cfg.get("signal", {})
    signals = {s.id: SignalTimeline(s.id, smooth_window=float(sig_cfg.get("smooth_window_sec", 1.0)))
               for s in geometry.signals}
    sig_every = float(sig_cfg.get("sample_every_sec", 0.2))
    next_sig_t = 0.0

    reader = FrameReader(video_path, meta.fps, stride, meta.n_frames,
                         int(cfg.get("video", {}).get("max_consecutive_read_failures", 25)),
                         int(cfg.get("runtime", {}).get("prefetch_frames", 48)))
    budget = float(cfg.get("runtime", {}).get("budget_factor", 1.4))
    expected = meta.duration if meta.duration > 0 else None
    t_loop = time.perf_counter()
    n_proc = 0
    last_progress = -1.0
    for frame_idx, t, frame in reader:
        dets = detector(frame)
        updated = tracker.update(dets, t, frame_idx)
        n_proc += 1
        if keep_frame_tracks:
            result.frame_tracks[frame_idx] = [(tr.track_id, tr.cls_name, [float(x) for x in tr.last_box]) for tr in updated]
        if signals and t + 1e-9 >= next_sig_t:
            next_sig_t = t + sig_every
            for spec in geometry.signals:
                signals[spec.id].add(t, classify_signal(frame, spec, float(sig_cfg.get("dominance_ratio", 1.35)),
                                                        float(sig_cfg.get("min_lamp_brightness", 120))))
        if obstacle.enabled:
            obstacle.update(frame, t, dets.boxes[dets.scores >= float(cfg["tracker"].get("low_thresh", 0.1))])
        if fire.enabled:
            moving = [tr.last_box for tr in updated if tr.cls_name in VEHICLE_CLASSES]
            fire.update(frame, t, np.array(moving) if moving else np.zeros((0, 4)))
        # runtime governor (emergency only)
        if expected and n_proc % 50 == 0 and t > 5.0:
            elapsed = time.perf_counter() - t_start
            if elapsed > budget * t and reader.stride < 8:  # slower than budget x real time
                reader.stride += 1
                msg = f"runtime governor: {elapsed:.0f}s spent for {t:.0f}s of video, stride -> {reader.stride}"
                result.warnings.append(msg)
                log.warning(msg)
        if progress and expected:
            frac = min(t / expected, 1.0) * 0.9
            if frac - last_progress >= 0.01:
                last_progress = frac
                progress(frac, "detecting and tracking")
    result.timings["decode_detect_track"] = time.perf_counter() - t_loop
    duration = effective_duration(meta, reader.frames_decoded)
    result.duration = duration
    if duration <= 0:
        result.warnings.append("no decodable frames")
        return result
    if reader.read_failures:
        result.warnings.append(f"{reader.read_failures} unreadable frames skipped")

    t_rules = time.perf_counter()
    ctx = analyze_tracks(tracker.all_tracks(), cfg, geometry, meta.width, meta.height, dt, duration,
                         signals=signals, prior_flow=prior_flow,
                         obstacle_flags=obstacle.flags if obstacle.enabled else None, obstacle_every=obstacle.every,
                         fire_flags=fire.flags if fire.enabled else None, fire_every=fire.every)
    result.series, result.flow, result.signals = ctx.series, ctx.flow, signals
    result.raw_events = run_rules(ctx)
    result.events = finalize(result.raw_events, duration, cfg)
    result.timings["rules"] = time.perf_counter() - t_rules
    result.timings["total"] = time.perf_counter() - t_start
    if progress:
        progress(1.0, "done")
    log.info("%s: %d frames processed, %d tracks, %d events in %.1fs (video %.1fs)", video_path, n_proc,
             len(result.series), len(result.events), result.timings["total"], duration)
    return result



def analyze_tracks(tracks, cfg: dict, geometry: Geometry, width: int, height: int, dt: float, duration: float,
                   signals: dict | None = None, prior_flow: FlowField | None = None,
                   obstacle_flags=None, obstacle_every: float = 0.5, fire_flags=None, fire_every: float = 0.5) -> SceneContext:
    """Tracks -> smoothed series + learned flow field -> rule context.

    Separated from the decoding loop so rules can be unit-tested with
    synthetic trajectories."""
    tc = cfg.get("tracker", {})
    tracks = stitch_stationary(list(tracks), float(tc.get("stitch_max_gap_sec", 8.0)), float(tc.get("stitch_min_iou", 0.45)))
    series = build_all_series(tracks, dt, cfg)
    flow = FlowField.from_config(cfg)
    flow.rel_frac = float(cfg.get("scene", {}).get("carriageway_rel_frac", 0.25))
    moving = float(cfg.get("features", {}).get("moving_speed", 0.4))
    stationary = float(cfg.get("features", {}).get("stationary_speed", 0.12))
    min_stop = float(cfg.get("scene", {}).get("stop_zone_min_sec", 3.0))
    for s in series:
        flow.add_series(s, width, height, moving)
        flow.add_stops(s, width, height, stationary, min_stop)
    if prior_flow is not None:
        flow.add_prior(prior_flow, float(cfg.get("scene", {}).get("prior_weight", 1.0)))
    sc = cfg.get("scene", {})
    return SceneContext(
        cfg=cfg, geometry=geometry, flow=flow, series=series, width=width, height=height, dt=dt,
        duration=duration, signals=signals or {}, obstacle_flags=obstacle_flags, obstacle_every=obstacle_every,
        fire_flags=fire_flags, fire_every=fire_every,
        direction_groups=flow.direction_groups(float(sc.get("direction_group_min_share", 0.12)),
                                               float(sc.get("direction_group_min_separation_deg", 60))))
