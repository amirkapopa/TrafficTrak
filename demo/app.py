"""TrafficTrak upload demo (Streamlit, CPU is fine).

    pip install -r requirements.txt -r requirements-demo.txt
    streamlit run demo/app.py

Upload an .mp4 -> Part A event segments, causal risk curve, annotated
playback with a clickable timeline.  The demo imports the same code as
solution.py but is not part of the evaluated submission.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import analyze_video  # noqa: E402
from src.risk import RiskEstimator, risk_curve  # noqa: E402
from src.segments import CLASSES  # noqa: E402
from src.visualize import render_annotated_video  # noqa: E402

SERIES = "#2a78d6"
MUTED = "#52514e"

st.set_page_config(page_title="TrafficTrak demo", page_icon="🚦", layout="wide")
st.title("TrafficTrak - traffic-event detection")
st.caption("Fixed-camera CCTV analysis: detector + tracker + transparent rules (Part A) and a causal "
           "accident-risk score (Part B). Runs locally on CPU; nothing is sent to any online service.")

with st.sidebar:
    st.header("Options")
    do_risk = st.checkbox("Compute causal risk curve (Part B)", value=True)
    do_video = st.checkbox("Render annotated video", value=True)
    st.markdown("Scene geometry: `config/camera_geometry.yaml`  \nThresholds: `config/pipeline.yaml`")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "m4v"])


def run_analysis(data: bytes, name: str, do_risk: bool, do_video: bool) -> dict:
    key = hashlib.sha1(data).hexdigest()[:16]
    work = Path(tempfile.gettempdir()) / "traffictrak_demo" / key
    work.mkdir(parents=True, exist_ok=True)
    video = work / Path(name).name
    if not video.exists():
        video.write_bytes(data)
    bar = st.progress(0.0, text="starting")
    stages = [("analysis", 0.0, 0.55)] + ([("risk", 0.55, 0.8)] if do_risk else []) + ([("render", 0.8, 1.0)] if do_video else [])
    spans = {s: (a, b) for s, a, b in stages}

    def cb(stage):
        a, b = spans[stage]
        return lambda f, msg="": bar.progress(min(a + (b - a) * float(f), 1.0), text=f"{stage}: {msg}")

    t0 = time.perf_counter()
    analysis = analyze_video(str(video), progress=cb("analysis"))
    t_a = time.perf_counter() - t0
    risk = None
    if do_risk and analysis.meta.ok:
        risk = risk_curve(str(video), RiskEstimator(), progress=cb("risk"))
    out_video = None
    if do_video and analysis.meta.ok:
        out_video = render_annotated_video(str(video), analysis, str(work / "annotated.mp4"), risk, progress=cb("render"))
    bar.progress(1.0, text="done")
    return {"name": name, "analysis": analysis, "risk": risk, "video": out_video, "t_part_a": t_a,
            "t_total": time.perf_counter() - t0}


if uploaded is not None and st.button("Analyze", type="primary"):
    try:
        st.session_state["result"] = run_analysis(uploaded.getvalue(), uploaded.name, do_risk, do_video)
        st.session_state["seek"] = 0
    except Exception as exc:  # noqa: BLE001
        st.error(f"analysis failed: {exc}")

res = st.session_state.get("result")
if res:
    a = res["analysis"]
    if not a.meta.ok:
        st.error(f"Could not read the video: {a.meta.error}")
        st.stop()
    segs = a.segments()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duration", f"{a.duration:.1f} s")
    c2.metric("Events", len(segs))
    c3.metric("Max risk", f"{res['risk'][1].max():.2f}" if res["risk"] is not None and len(res["risk"][1]) else "-")
    c4.metric("Part A runtime", f"{res['t_part_a']:.1f} s ({res['t_part_a'] / max(a.duration, 1e-6):.2f}x)")
    for w in a.warnings:
        st.warning(w)

    events_df = pd.DataFrame([{"label": e.label, "start": e.start, "end": e.end, "duration": round(e.end - e.start, 2),
                               "tracks": ", ".join(map(str, e.tracks)), "evidence": e.note} for e in a.events])

    left, right = st.columns([3, 2])
    with right:
        st.subheader("Timeline")
        st.caption("Click a bar to jump the video to that event.")
        if len(events_df):
            sel = alt.selection_point(name="ev", fields=["start"], on="click")
            chart = (alt.Chart(events_df).mark_bar(color=SERIES, cornerRadius=4, height=12)
                     .encode(x=alt.X("start:Q", title="time (s)", scale=alt.Scale(domain=[0, a.duration])), x2="end:Q",
                             y=alt.Y("label:N", sort=CLASSES, scale=alt.Scale(domain=CLASSES), title=None),
                             tooltip=["label", alt.Tooltip("start:Q", format=".2f"), alt.Tooltip("end:Q", format=".2f"),
                                      "evidence"],
                             opacity=alt.condition(sel, alt.value(1.0), alt.value(0.55)))
                     .add_params(sel).properties(height=340))
            ev = st.altair_chart(chart, use_container_width=True, on_select="rerun", key="timeline")
            picked = (ev or {}).get("selection", {}).get("ev") or []
            if picked:
                st.session_state["seek"] = int(max(0, float(picked[0]["start"]) - 1))
            options = ["-"] + [f"{r.label} @ {r.start:.1f}s" for r in events_df.itertuples()]
            choice = st.selectbox("or jump to", options)
            if choice != "-":
                st.session_state["seek"] = int(max(0, float(choice.split("@ ")[1][:-1]) - 1))
        else:
            st.info("No events detected in this video.")
    with left:
        st.subheader("Annotated playback")
        if res["video"] and Path(res["video"]).is_file():
            st.video(Path(res["video"]).read_bytes(), start_time=int(st.session_state.get("seek", 0)))
        else:
            st.info("Enable 'Render annotated video' to see playback.")

    if res["risk"] is not None and len(res["risk"][0]):
        st.subheader("Causal accident-risk score")
        ts, rs = res["risk"]
        step = max(1, len(ts) // 3000)
        rdf = pd.DataFrame({"t": ts[::step], "risk": rs[::step]})
        base = alt.Chart(rdf).encode(x=alt.X("t:Q", title="time (s)"))
        hover = alt.selection_point(fields=["t"], nearest=True, on="pointerover", empty=False)
        line = base.mark_line(color=SERIES, strokeWidth=2).encode(y=alt.Y("risk:Q", scale=alt.Scale(domain=[0, 1]),
                                                                          title="P(accident within 5 s)"))
        points = base.mark_point(color=SERIES, size=60, filled=True).encode(
            y="risk:Q", opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("t:Q", format=".2f", title="t (s)"), alt.Tooltip("risk:Q", format=".3f")]).add_params(hover)
        rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(color=MUTED, strokeDash=[4, 3]).encode(y="y:Q")
        st.altair_chart((line + points + rule).properties(height=220), use_container_width=True)

    st.subheader("Segments")
    st.dataframe(events_df, use_container_width=True, hide_index=True)
    st.download_button("Download predictions JSON", json.dumps({res["name"]: segs}, indent=2),
                       file_name=f"{Path(res['name']).stem}_events.json", mime="application/json")
    if res["risk"] is not None:
        csv = "t_sec,risk\n" + "\n".join(f"{t:.3f},{r:.4f}" for t, r in zip(*res["risk"]))
        st.download_button("Download risk CSV", csv, file_name=f"{Path(res['name']).stem}_risk.csv", mime="text/csv")
