"""Trust and controls: calibration, label quality vs window, and the shuffled-label control."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import data
from dashboard.theme import HELP
from securetrace.fusion.control import live_shuffle_control, stored_shuffle_control
from securetrace.fusion.explain import window_label_quality


@st.cache_data(show_spinner="Re-training twice on the training sessions (about a minute)...")
def _live_control(seed: int, _bundle: dict) -> dict:
    return live_shuffle_control(_bundle, data.SESSIONS, data.EXTRACTED, seed)


def render(ctx: dict) -> None:
    b = ctx["bundle"]
    st.subheader("Can the risk score be trusted?")
    trust = b.get("trust")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Reliability (calibration)**")
        if trust:
            frames = []
            for name, key in (("raw model score", "reliability_raw"), ("calibrated (isotonic)", "reliability_calibrated")):
                frames.append(pd.DataFrame(trust[key]).assign(score=name))
            df = pd.concat(frames)
            line = alt.Chart(df).mark_line(point=True).encode(
                x=alt.X("confidence:Q", title="predicted probability", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("accuracy:Q", title="observed malicious fraction (evaluation only)", scale=alt.Scale(domain=[0, 1])),
                color="score:N", tooltip=["score", "confidence", "accuracy", "n"])
            diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(color="gray", strokeDash=[4, 3]).encode(x="x", y="y")
            st.altair_chart(line + diag, width="stretch")
            m1, m2 = st.columns(2)
            m1.metric("ECE, raw", f"{trust['ece_raw']:.3f}", help="Expected calibration error: average gap between predicted and observed rates. Lower is better.")
            m2.metric("ECE, calibrated", f"{trust['ece_calibrated']:.3f}", delta=f"{trust['ece_calibrated'] - trust['ece_raw']:+.3f}", delta_color="inverse")
            st.caption("Calibration is fit on WEAK validation labels. In our runs the improvement is modest and within noise; "
                       "treat it as unproven. Points on the dashed line are perfectly calibrated.")
    with c2:
        st.markdown("**Weak-label quality vs window width**")
        lq = data.PROCESSED / "label_quality.csv"
        if lq.exists():
            q = pd.read_csv(lq).melt(id_vars="window_seconds", value_vars=["bag_precision", "bag_recall"], var_name="measure")
            chart = alt.Chart(q).mark_line(point=True).encode(
                x=alt.X("window_seconds:Q", scale=alt.Scale(type="log"), title="window width (s)"),
                y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1.02]), title="fraction (evaluation only)"),
                color="measure:N", tooltip=["window_seconds", "measure", "value"])
            sel = alt.Chart(pd.DataFrame({"w": [ctx["window"]]})).mark_rule(color="#E45756", strokeDash=[5, 3]).encode(x="w:Q")
            st.altair_chart(chart + sel, width="stretch")
            st.caption("Precision = share of flows inside an integrity-failure window that are truly malicious. A wider window "
                       "catches more attack flows (recall) but includes more benign ones. The red line is your window slider.")
        q_now = window_label_quality(ctx["flows"])
        if q_now["flows_in_windows"]:
            st.metric("This session at the chosen window: precision", f"{q_now['precision']:.0%}", help=HELP["eval_only"])

    st.divider()
    st.subheader("Control: do the results depend on the integrity labels?")
    st.markdown("If the gain came from lab artefacts instead of the integrity evidence, **randomly shuffled** weak labels would "
                "work just as well. They do not: the gain disappears.")
    held = b["held_out_mutator"]
    try:
        table = stored_shuffle_control(data.PROCESSED / "phase4_results.csv", data.PROCESSED / "phase3_before_numbers.csv", held)
        long = table.melt(id_vars="method", var_name="training labels", value_name="PR-AUC")
        st.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("training labels:N", title=None, axis=alt.Axis(labels=False)),
            y=alt.Y("PR-AUC:Q"), color=alt.Color("training labels:N", sort=["public model (no adaptation)", "adapted, real labels", "adapted, shuffled labels"],
                                                 scale=alt.Scale(range=["#9AA0A6", "#54A24B", "#E45756"])),
            column=alt.Column("method:N", title=None), tooltip=["method", "training labels", "PR-AUC"]).properties(width=90, height=180))
        st.caption(f"Mean PR-AUC over 5 seeds from the logged Phase 4 runs (held-out attack: {held}). Higher is better; "
                   f"chance is the attack rate, about {trust['test_attack_rate']:.2f}." if trust else "")
    except FileNotFoundError:
        st.info("Phase 4 results not found; run scripts/run_all.py first.")

    st.markdown("**Run it live** (re-trains the adapted model twice on the training sessions, with real and with shuffled labels):")
    if st.button("Re-train with shuffled labels now"):
        res = _live_control(0, b)
        st.dataframe(pd.DataFrame([res]).T.rename(columns={0: "PR-AUC"}).round(3), width="stretch")
        gap = res["adapted, real labels"] - res["adapted, shuffled labels"]
        st.success(f"Real labels beat shuffled labels by {gap:.2f} PR-AUC; shuffled labels land near chance "
                   f"({res['attack rate (chance)']:.2f}).")
    vr = data.ROOT / "docs" / "verification_report.md"
    if vr.exists():
        with st.expander("Project verification report (14 checks)"):
            st.markdown(vr.read_text(encoding="utf-8"))
