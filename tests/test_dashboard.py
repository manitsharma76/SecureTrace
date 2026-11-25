"""Streamlit AppTest smoke tests for the dashboard. They use the real saved sessions and the served
model bundle, so they are skipped when `scripts/build_dashboard_bundle.py` has not been run."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "data" / "processed" / "models" / "dashboard" / "bundle.joblib"
SESSIONS = ROOT / "data" / "raw" / "lab" / "sessions"
pytestmark = pytest.mark.skipif(not BUNDLE.exists() or not SESSIONS.exists(), reason="dashboard bundle / lab sessions not built")

from streamlit.testing.v1 import AppTest  # noqa: E402


def _app(collector="http://127.0.0.1:9"):
    import os
    os.environ["SECURETRACE_COLLECTOR"] = collector  # unreachable by default: offline mode, deterministic
    return AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=180)


def _plan(run_id):
    return json.loads((SESSIONS / f"{run_id}.json").read_text())["plan"]


def _pick(at, wanted_tamper):
    from securetrace.data.lab import load_session_manifest
    box = at.sidebar.selectbox(key="run_pick")
    for rid in box.options:
        loc = load_session_manifest(SESSIONS / f"{rid}.json")["localizations"].values()
        if any(v.get("status") == "tamper" for v in loc) == wanted_tamper:
            return rid
    pytest.skip("no session of that kind among the held-out test sessions")


def _metric(at, label):
    return next(m.value for m in at.metric if m.label.startswith(label))


def test_tampered_session_renders_all_panels_with_one_localized_segment():
    at = _app().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Session explorer", "Live demo", "Trust & controls", "How it works", "Limitations"]
    rid = _pick(at, True)
    at.sidebar.selectbox(key="run_pick").set_value(rid).run()
    assert not at.exception
    segment = _metric(at, "3. Tampered segment")
    assert segment in ("A", "B", "C") and segment == _plan(rid)["tamper_segment"]
    assert "failed" in _metric(at, "1. Integrity")


def test_clean_session_raises_no_fused_alert():
    at = _app().run()
    rid = _pick(at, False)
    at.sidebar.selectbox(key="run_pick").set_value(rid).run()
    assert not at.exception
    assert _metric(at, "3. Tampered segment") == "none" and _metric(at, "4. Fused alert").startswith("0 alerts")


def test_threshold_slider_changes_counts_without_a_reload():
    at = _app().run()
    rid = _pick(at, True)
    at.sidebar.selectbox(key="run_pick").set_value(rid).run()
    low = _metric(at, "2. ML risk")
    at.sidebar.slider[0].set_value(0.95).run()
    assert not at.exception
    assert _metric(at, "2. ML risk") != low or int(low.split()[0]) == 0


def test_window_slider_and_card_buttons_work_and_flow_detail_has_counterfactuals():
    at = _app().run()
    rid = _pick(at, True)
    at.sidebar.selectbox(key="run_pick").set_value(rid).run()
    at.sidebar.select_slider[0].set_value(5.0).run()
    assert not at.exception
    next(b for b in at.button if b.key == "card_seg").click().run()
    assert at.radio(key="detail_view").value == "Tampered segment"
    next(b for b in at.button if b.key == "card_fused").click().run()
    assert at.radio(key="detail_view").value == "Fused alert" and not at.exception
    assert any(h.value.startswith("### Why flow") for h in at.markdown)
    cf = [s for s in at.slider if s.key and s.key.startswith("cf_")]
    assert cf, "counterfactual sliders should be present"
    before = [m.value for m in at.metric if m.label == "With your edits"][0]
    cf[0].set_value(cf[0].max).run()
    assert not at.exception
    assert [m.value for m in at.metric if m.label == "With your edits"]  # re-scored without error


def test_simulator_tab_names_the_attacked_segment():
    at = _app().run()
    assert at.radio(key="demo_mode").value.startswith("Simulator")  # lab unreachable -> simulator by default
    at.selectbox(key="sim_seg").set_value("C").run()
    assert not at.exception
    assert any("segment **C**" in e.value for e in at.error)
    at.selectbox(key="sim_seg").set_value("none").run()
    assert any("No tamper detected" in s.value for s in at.success)


def test_trust_tab_and_static_tabs_render():
    at = _app().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "ECE, raw" in labels and "ECE, calibrated" in labels
