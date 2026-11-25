"""SecureTrace dashboard.

    python scripts/build_dashboard_bundle.py     # once: trains the served model
    streamlit run dashboard/app.py

Tabs: Session explorer (four clickable cards -> hop-chain view, per-flow SHAP, counterfactuals),
Live demo (real attack in the Docker lab, or an offline simulator), Trust & controls, How it works,
Limitations. Reads the collector when reachable and saved sessions otherwise. Lab traffic is
synthetic; nothing here shows how the system behaves on a real network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import about, data, demo, explorer, trust  # noqa: E402
from dashboard.theme import HELP, caveat  # noqa: E402
from securetrace.fusion.live import lab_available  # noqa: E402

st.set_page_config(page_title="SecureTrace", layout="wide")


def _log_check(lab_up: bool) -> dict | None:
    if not lab_up:
        return None
    try:
        r = requests.get(f"{data.COLLECTOR}/log/verify", timeout=2)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


st.title("SecureTrace")
st.caption("Hop-chained integrity checks locate the tampered segment and focus an ML intrusion model.")
caveat()

if not data.BUNDLE.exists():
    st.error(f"No model bundle at {data.BUNDLE}. Run `python scripts/build_dashboard_bundle.py` first.")
    st.stop()
bundle = data.bundle()
lab_up = lab_available(data.COLLECTOR)

# A finished live demo asks to be opened: apply it before the sidebar widgets are created.
for _key, _default in (("group_filter", ["held-out test"]), ("verdict_filter", "all"), ("segment_filter", []), ("mutator_filter", "all")):
    st.session_state.setdefault(_key, _default)
pending = st.session_state.pop("pending_run", None)
if pending:
    st.session_state["group_filter"] = sorted(set(st.session_state["group_filter"]) | {"live demo"})
    st.session_state["verdict_filter"], st.session_state["segment_filter"], st.session_state["mutator_filter"] = "all", [], "all"
    st.session_state["run_pick"] = pending

refs, catalog = data.session_catalog(data.demo_stamp(), tuple(bundle["test_run_ids"]))
ref_by_id = {r.run_id: r for r in refs}

with st.sidebar:
    st.header("Controls")
    threshold = st.slider("Alert threshold", 0.05, 0.95, 0.5, 0.05, help=HELP["threshold"])
    window = st.select_slider("Weak-label window (s)", options=data.WINDOWS, value=bundle["window_seconds"], help=HELP["window"])
    st.header("Session")
    groups = st.multiselect("Sessions from", ["held-out test", "live demo", "trained on"], key="group_filter",
                            help="'held-out test' sessions were never seen by the served model.")
    verdict = st.radio("Integrity verdict", ["all", "tamper detected", "no tamper detected"], key="verdict_filter",
                       help="Observed by the integrity layer, not ground truth.")
    segs = st.multiselect("Located segment", ["A", "B", "C"], key="segment_filter", help="Segment the integrity layer blamed.")
    mutator = st.selectbox("Attack type (attacker plan, evaluation only)", ["all", "flip_byte", "rewrite_size"], key="mutator_filter",
                           help=HELP["eval_only"])
    sub = catalog[catalog.group.isin(groups)] if len(catalog) else catalog
    if verdict != "all":
        sub = sub[sub.integrity_verdict == verdict]
    if segs:
        sub = sub[sub.integrity_segment.isin(segs)]
    if mutator != "all":
        sub = sub[sub["plan_mutator (evaluation only)"] == mutator]
    ids = list(sub.run_id)
    if not ids:
        st.warning("No sessions match these filters.")
    run_id = st.selectbox(f"Run ({len(ids)} match)", ids, key="run_pick") if ids else None
    st.caption(f"Model: XGBoost adapted with weak labels; held-out attack type: {bundle['held_out_mutator']}.")
    st.write("Lab collector:", "**live**" if lab_up else "**offline** (saved sessions only)")

tab_explore, tab_demo, tab_trust, tab_how, tab_limits = st.tabs(
    ["Session explorer", "Live demo", "Trust & controls", "How it works", "Limitations"])

ctx = {"bundle": bundle, "threshold": threshold, "window": window, "lab_up": lab_up}
with tab_explore:
    if run_id is None:
        st.info("Choose a session in the sidebar, or run one in the Live demo tab.")
    else:
        ref = ref_by_id[run_id]
        flows, X = data.scored(str(ref.manifest_path), str(ref.extracted_dir), window, bundle)
        ctx.update(ref=ref, manifest=data.manifest(str(ref.manifest_path)), flows=flows, X=X, log_check=_log_check(lab_up))
        explorer.render(ctx)
with tab_demo:
    demo.render(ctx)
with tab_trust:
    if "flows" in ctx:
        trust.render(ctx)
    else:
        st.info("Select a session to see the trust panels.")
with tab_how:
    about.render_how()
with tab_limits:
    about.render_limits()
