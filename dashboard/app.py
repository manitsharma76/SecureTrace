"""SecureTrace dashboard: Integrity | ML risk | Tampered segment | Fused alert.

    python scripts/build_dashboard_bundle.py          # once: trains the served model
    streamlit run dashboard/app.py

Reads the collector (http://127.0.0.1:9000, `docker compose -f lab/docker-compose.yml up`) when it
is reachable and otherwise the saved session manifests, so it also works offline. Lab traffic is
synthetic; nothing here shows how the system behaves on a real network. True labels appear only in
columns marked "evaluation only" and never feed a score.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.data.lab import load_session_manifest  # noqa: E402
from securetrace.fusion.service import (  # noqa: E402
    fused_alert, integrity_summary, load_bundle, score_session, session_localization,
)

BUNDLE = Path(os.environ.get("SECURETRACE_BUNDLE", ROOT / "data/processed/models/dashboard/bundle.joblib"))
SESSIONS = Path(os.environ.get("SECURETRACE_SESSIONS", ROOT / "data/raw/lab/sessions"))
EXTRACTED = Path(os.environ.get("SECURETRACE_EXTRACTED", ROOT / "data/raw/lab/pcap/extracted"))
COLLECTOR = os.environ.get("SECURETRACE_COLLECTOR", "http://127.0.0.1:9000")
THRESHOLD = 0.5

st.set_page_config(page_title="SecureTrace", layout="wide")


@st.cache_resource
def _bundle():
    return load_bundle(BUNDLE)


def _collector_log_check() -> dict | None:
    try:
        r = requests.get(f"{COLLECTOR}/log/verify", timeout=2)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def _manifest_path(run_id: str) -> Path:
    return next(p for p in SESSIONS.glob("*.json") if p.stem == run_id)


st.title("SecureTrace")
st.caption("Hop-chained integrity checks locate the tampered segment and focus an ML intrusion model. "
           "Synthetic lab traffic only; results do not carry over to a real network.")

if not BUNDLE.exists():
    st.error(f"No model bundle at {BUNDLE}. Run `python scripts/build_dashboard_bundle.py` first.")
    st.stop()
bundle = _bundle()

with st.sidebar:
    st.header("Session")
    default_ids = [r for r in bundle["test_run_ids"] if (SESSIONS / f"{r}.json").exists()]
    show_all = st.checkbox("Include sessions the model trained on", value=False)
    ids = sorted(p.stem for p in SESSIONS.glob("*.json")) if show_all else sorted(default_ids)
    run_id = st.selectbox("Run", ids)
    st.caption(f"Model: XGBoost adapted with weak labels; held-out attack: {bundle['held_out_mutator']}.")
    live = _collector_log_check()
    st.write("Collector:", "live" if live is not None else "offline (using saved sessions)")

manifest = load_session_manifest(_manifest_path(run_id))
flows, X = score_session(_manifest_path(run_id), EXTRACTED, bundle, bundle["window_seconds"])
summary = integrity_summary(manifest)
loc = session_localization(manifest)
alert = fused_alert(manifest, flows, X, bundle, THRESHOLD)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.subheader("1. Integrity")
    counts = summary["message_status"]
    st.metric("Messages verified", summary["n_messages"])
    st.dataframe(pd.DataFrame({"status": list(counts), "messages": list(counts.values())}), hide_index=True)
    chain = live if live is not None else summary["log_integrity"]
    st.write("Event-log hash chain:", "intact" if chain.get("ok") else "BROKEN")
with c2:
    st.subheader("2. ML risk")
    per_seg = flows.groupby("segment").agg(flows=("risk", "size"), max_risk=("risk", "max"),
                                           flows_above_threshold=("risk", lambda s: int((s >= THRESHOLD).sum())))
    st.dataframe(per_seg.round(3))
    st.bar_chart(per_seg["flows_above_threshold"])
    st.caption(f"ML-only alerts in this session: {alert['ml_only_alerts']} (threshold {THRESHOLD}).")
with c3:
    st.subheader("3. Tampered segment")
    if loc is None:
        st.success("No tamper detected by the integrity layer.")
    else:
        st.error(f"Tamper localized to segment {loc['segment']} ({loc['n_failed_messages']} failed messages)")
        ev = pd.DataFrame(loc["evidence"]).T
        st.dataframe(ev)
    truth = manifest["plan"]["tamper_segment"]
    st.caption(f"Attacker ground truth (evaluation only): {truth or 'no tamper'}")
with c4:
    st.subheader("4. Fused alert")
    if alert["localization"] is None:
        st.info("No integrity failure, so no fused alert is raised.")
    else:
        st.write(f"Segment **{alert['localization']['segment']}**, ranked by calibrated risk. "
                 f"Fused alerts above threshold: {alert['fused_alerts_above_threshold']} "
                 f"(ML-only: {alert['ml_only_alerts']}, any-segment time window: {alert['window_alerts']}).")

if alert["alerts"]:
    st.subheader("Top flows on the tampered segment")
    st.dataframe(pd.DataFrame([{
        "flow": a["flow_row"], "segment": a["segment"], "risk": round(a["risk"], 3),
        "top features (TreeSHAP)": ", ".join(f"{n} ({v:+.2f})" for n, v in a["top_features"]),
        "malicious (evaluation only)": a["true_malicious_eval_only"],
    } for a in alert["alerts"]]), hide_index=True)
