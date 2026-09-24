"""Visual outputs: geometry overlays, flow-field render, annotated video,
event timeline and risk curve.  Not used by the evaluation entry points;
matplotlib / imageio-ffmpeg are imported lazily."""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .flow import FlowField
from .geometry import Geometry
from .segments import CLASSES, RawEvent

log = logging.getLogger("traffictrak.visualize")

# Reference data-viz palette (light mode) - slot 1 for series, muted inks for text/grid.
SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

# BGR colours for overlays on video frames
C_ROAD = (90, 170, 60)
C_LANE = (230, 170, 40)
C_STOP = (40, 40, 230)
C_CROSS = (240, 240, 240)
C_INTER = (0, 200, 240)
C_SIGNAL = (200, 60, 200)
C_SOLID = (0, 230, 255)
C_TURN = (0, 140, 255)
C_TRACK = (220, 200, 60)
C_ALERT = (40, 40, 235)


def _poly(img, pts, color, thickness=2, fill_alpha=0.0):
    p = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
    if fill_alpha > 0:
        over = img.copy()
        cv2.fillPoly(over, [p], color)
        cv2.addWeighted(over, fill_alpha, img, 1 - fill_alpha, 0, dst=img)
    cv2.polylines(img, [p], True, color, thickness, cv2.LINE_AA)


def _arrow_path(img, pts, color, thickness=2):
    p = np.round(pts).astype(np.int32)
    for a, b in zip(p[:-2], p[1:-1]):
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), color, thickness, cv2.LINE_AA)
    cv2.arrowedLine(img, tuple(int(v) for v in p[-2]), tuple(int(v) for v in p[-1]), color, thickness, cv2.LINE_AA,
                    tipLength=0.08)


def _label(img, text, org, color=(255, 255, 255), scale=0.5, bg=(0, 0, 0)):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + base), bg, -1)
    cv2.putText(img, text, (x + 2, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_geometry(img: np.ndarray, g: Geometry, alpha: float = 0.25, labels: bool = True) -> np.ndarray:
    out = img.copy()
    for poly in g.carriageway:
        _poly(out, poly.pts, C_ROAD, 2, alpha)
    for poly in g.exclusion_zones + g.sidewalks:
        _poly(out, poly.pts, (120, 120, 120), 1, alpha)
    for lane in g.lanes:
        _poly(out, lane.polygon.pts, C_LANE, 1)
        _arrow_path(out, lane.path.pts, C_LANE, 2)
        if labels:
            _label(out, f"{lane.id} ({lane.group})", lane.path.pts[0])
    for poly in g.crossings:
        _poly(out, poly.pts, C_CROSS, 2, alpha * 0.6)
    if g.intersection is not None:
        _poly(out, g.intersection.pts, C_INTER, 2)
    for sl in g.stop_lines:
        p = np.round(sl.line.pts).astype(np.int32)
        cv2.line(out, tuple(p[0]), tuple(p[-1]), C_STOP, 3, cv2.LINE_AA)
        mid = sl.line.pts.mean(axis=0)
        cv2.arrowedLine(out, tuple(np.round(mid - 30 * sl.approach).astype(int)), tuple(np.round(mid + 10 * sl.approach).astype(int)),
                        C_STOP, 2, cv2.LINE_AA, tipLength=0.3)
        if labels:
            _label(out, f"stop {sl.id} -> {sl.signal}", mid)
    for s in g.signals:
        x0, y0, x1, y1 = s.roi
        cv2.rectangle(out, (x0, y0), (x1, y1), C_SIGNAL, 2)
        if labels:
            _label(out, f"signal {s.id}", (x0, y0 - 2))
    for line in g.solid_lines:
        p = np.round(line.pts).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [p], False, C_SOLID, 2, cv2.LINE_AA)
    for t in g.prohibited_turns:
        _poly(out, t.from_zone.pts, C_TURN, 2)
        _poly(out, t.to_zone.pts, C_TURN, 2, alpha * 0.5)
        if labels:
            _label(out, f"{t.label}: {t.id}", t.to_zone.pts[0])
    for poly in g.u_turn_zones + g.obstacle_regions:
        _poly(out, poly.pts, C_TURN, 1)
    if labels and not g.calibrated:
        _label(out, "geometry: UNCALIBRATED", (10, 24), (0, 0, 0), 0.6, (0, 200, 255))
    return out


def draw_flow(img: np.ndarray, flow: FlowField, min_support: float = 3, min_conc: float = 0.6) -> np.ndarray:
    """Arrows of the dominant traffic direction per flow-grid cell."""
    out = img.copy()
    h, w = out.shape[:2]
    cw, ch = w / flow.gw, h / flow.gh
    occ = flow.carriageway_mask(3)
    over = out.copy()
    for oy in range(flow.oh):
        for ox in range(flow.ow):
            if occ[oy, ox]:
                cv2.rectangle(over, (int(ox * w / flow.ow), int(oy * h / flow.oh)),
                              (int((ox + 1) * w / flow.ow), int((oy + 1) * h / flow.oh)), C_ROAD, -1)
    cv2.addWeighted(over, 0.25, out, 0.75, 0, dst=out)
    for cy in range(flow.gh):
        for cx in range(flow.gw):
            x, y = (cx + 0.5) * cw, (cy + 0.5) * ch
            ang, conc, sup = flow.dominant(x, y, w, h)
            if sup < min_support or conc < min_conc:
                continue
            L = 0.45 * min(cw, ch)
            p0 = (int(x - L * math.cos(ang)), int(y - L * math.sin(ang)))
            p1 = (int(x + L * math.cos(ang)), int(y + L * math.sin(ang)))
            color = (40, 220, 255) if conc >= 0.85 else (200, 200, 200)
            cv2.arrowedLine(out, p0, p1, color, 2, cv2.LINE_AA, tipLength=0.35)
    return out


def heatmap_overlay(img: np.ndarray, points: np.ndarray, sigma: float = 12.0) -> np.ndarray:
    h, w = img.shape[:2]
    acc = np.zeros((h, w), np.float32)
    if len(points):
        p = np.round(points).astype(int)
        ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
        np.add.at(acc, (p[ok, 1], p[ok, 0]), 1.0)
    acc = cv2.GaussianBlur(acc, (0, 0), sigma)
    if acc.max() > 0:
        acc = (255 * np.sqrt(acc / acc.max())).astype(np.uint8)
    heat = cv2.applyColorMap(acc.astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    mask = (acc > 8)[..., None]
    return np.where(mask, cv2.addWeighted(img, 0.45, heat, 0.55, 0), img)


# ----------------------------------------------------------------------------- annotated video
def _series_index(series, dt):
    idx: dict[int, list] = {}
    for s in series:
        for i in range(s.n):
            idx.setdefault(s.k0 + i, []).append((s, i))
    return idx


def to_browser_mp4(path: str) -> str:
    """Re-encode to H.264 (yuv420p) so browsers can play it; returns the path used."""
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
    if not exe:
        return path
    tmp = path + ".h264.mp4"
    cmd = [exe, "-y", "-loglevel", "error", "-i", path, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", tmp]
    try:
        subprocess.run(cmd, check=True, timeout=1800)
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001
        log.warning("H.264 transcode failed (%s); keeping mp4v file", exc)
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def render_annotated_video(video_path: str, analysis, out_path: str, risk: tuple | None = None,
                           max_width: int = 960, progress=None, browser: bool = True) -> str | None:
    """Draw geometry, tracks, active events and the risk bar on every frame."""
    meta = analysis.meta
    if not meta.ok:
        return None
    scale = min(1.0, max_width / float(meta.width))
    size = (int(round(meta.width * scale)), int(round(meta.height * scale)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, size)
    cap = cv2.VideoCapture(video_path)
    index = _series_index(analysis.series, analysis.dt) if analysis.dt > 0 else {}
    events: list[RawEvent] = analysis.events
    raw_tracks: dict[str, set] = {}
    for e in analysis.raw_events:
        raw_tracks.setdefault(e.label, set()).update(e.tracks)
    base = draw_geometry(np.zeros((meta.height, meta.width, 3), np.uint8), analysis.geometry, 0.0, labels=False) \
        if analysis.geometry is not None else None
    geo_mask = base.any(axis=2) if base is not None else None
    rt, rv = (risk if risk is not None else (np.zeros(0), np.zeros(0)))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = i / meta.fps
            if geo_mask is not None:
                frame[geo_mask] = cv2.addWeighted(frame, 0.4, base, 0.6, 0)[geo_mask]
            active = [e for e in events if e.start <= t <= e.end]
            hot = set()
            for e in analysis.raw_events:
                if e.start <= t <= e.end:
                    hot.update(e.tracks)
            k = int(round(t / analysis.dt)) if analysis.dt > 0 else -1
            for s, j in index.get(k, []):
                x0, y0, x1, y1 = s.box[j]
                col = C_ALERT if s.tid in hot else C_TRACK
                cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
                _label(frame, f"{s.tid} {s.cls}", (x0, y0), (255, 255, 255), 0.45, col)
            y = 30
            for e in active:
                _label(frame, f"{e.label}  {e.start:.1f}-{e.end:.1f}s", (10, y), (255, 255, 255), 0.7, C_ALERT)
                y += 28
            _label(frame, f"t={t:6.2f}s", (meta.width - 150, 30), (255, 255, 255), 0.6)
            if len(rt):
                r = float(rv[min(np.searchsorted(rt, t + 1e-9, side="right") - 1, len(rv) - 1)]) if t >= rt[0] else 0.0
                bw = int(0.3 * meta.width)
                x0, y0 = meta.width - bw - 20, meta.height - 40
                cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + 18), (60, 60, 60), -1)
                cv2.rectangle(frame, (x0, y0), (x0 + int(bw * r), y0 + 18), C_ALERT if r >= 0.5 else (60, 180, 240), -1)
                _label(frame, f"accident risk (5 s) {r:.2f}", (x0, y0 - 4), (255, 255, 255), 0.5)
            if scale < 1.0:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            writer.write(frame)
            i += 1
            if progress and meta.n_frames and i % 50 == 0:
                progress(min(i / meta.n_frames, 1.0), "rendering")
    finally:
        cap.release()
        writer.release()
    return to_browser_mp4(out_path) if browser else out_path


# ----------------------------------------------------------------------------- charts
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_timeline(events: list, duration: float, out_png: str, title: str = "", risk: tuple | None = None) -> str:
    """One row per class (identity by row label, single hue); risk curve below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = 2 if risk is not None and len(risk[0]) else 1
    fig, axes = plt.subplots(rows, 1, figsize=(11, 5.2 if rows == 2 else 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2][:rows]}, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    ax = axes[0]
    _style(ax)
    for e in events:
        s, en, lab = (e.start, e.end, e.label) if isinstance(e, RawEvent) else (e[0], e[1], e[2])
        y = CLASSES.index(lab)
        ax.barh(y, en - s, left=s, height=0.62, color=SERIES, edgecolor=SURFACE, linewidth=1)
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES)
    ax.invert_yaxis()
    ax.set_xlim(0, max(duration, 1e-3))
    ax.set_title(title or "Detected events", loc="left", color=INK, fontsize=11)
    if not events:
        ax.text(0.5, 0.5, "no events detected", transform=ax.transAxes, ha="center", va="center", color=INK_2)
    if rows == 2:
        _plot_risk_axis(axes[1], risk, events, duration)
    axes[-1].set_xlabel("time (s)", color=INK_2)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


def _plot_risk_axis(ax, risk, events, duration):
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ts, rs = risk
    for e in events:
        s, en, lab = (e.start, e.end, e.label) if isinstance(e, RawEvent) else (e[0], e[1], e[2])
        if lab == "accident":
            ax.axvspan(s, en, color=GRID, alpha=0.9, linewidth=0)
    ax.axhline(0.5, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.plot(ts, rs, color=SERIES, linewidth=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(0, max(duration, 1e-3))
    ax.set_ylabel("P(accident\nwithin 5 s)", color=INK_2, fontsize=9)


def plot_risk(ts, rs, out_png: str, events=None, duration: float | None = None, title: str = "") -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 2.8), facecolor=SURFACE)
    _plot_risk_axis(ax, (ts, rs), events or [], duration or (float(ts[-1]) if len(ts) else 1.0))
    ax.set_title(title or "Causal accident-risk score (grey = detected accident, dashed = 0.5)", loc="left",
                 color=INK, fontsize=11)
    ax.set_xlabel("time (s)", color=INK_2)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png
