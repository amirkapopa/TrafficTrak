"""The Streamlit demo renders results with events (skipped without streamlit)."""

import numpy as np
import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")

from src.pipeline import Analysis  # noqa: E402
from src.video import VideoMeta  # noqa: E402

from .synth import make_track, piecewise, run  # noqa: E402


def test_demo_renders_timeline_and_risk():
    a = make_track(1, "car", piecewise([(0, 100, 450), (2.6, 620, 450), (2.8, 630, 450), (20, 630, 450)]), 0, 20)
    b = make_track(2, "car", piecewise([(0, 700, 450), (2.6, 700, 450), (2.9, 730, 450), (20, 730, 450)]), 0, 20)
    events, raw, ctx = run([a, b], 22.0)
    assert events
    analysis = Analysis(meta=VideoMeta("x.mp4", 15.0, 330, 1280, 720, True, True), duration=22.0, dt=ctx.dt,
                        events=events, raw_events=raw, series=ctx.series)
    ts = np.arange(0, 22, 1 / 15)
    app = st_testing.AppTest.from_file("demo/app.py", default_timeout=60)
    app.session_state["result"] = {"name": "x.mp4", "analysis": analysis, "risk": (ts, np.clip(np.sin(ts) ** 2, 0, 1)),
                                   "video": None, "t_part_a": 1.0, "t_total": 2.0}
    app.run()
    assert not app.exception
    assert any("accident" in str(df.value.values) for df in app.dataframe)
    assert [m.value for m in app.metric][1] == str(len(events))
