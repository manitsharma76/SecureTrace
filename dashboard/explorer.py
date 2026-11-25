"""Session explorer: four summary cards (clickable) -> detail views, plus the alerting-mode comparison."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import data
from dashboard.theme import BROKEN_COLOR, HELP, SEGMENT_COLORS, badge
from securetrace.fusion.alerts import candidate_mask
from securetrace.fusion.explain import (
    WHY_NOT_ANOTHER_SEGMENT, compare_modes, counterfactual_risk, fetch_message_events, flow_contributions,
    hop_chain_dot, hop_table, segment_risk_table, segment_verdicts, window_label_quality,
)
from securetrace.fusion.service import integrity_summary, session_localization

VIEWS = ["Integrity", "ML risk", "Tampered segment", "Fused alert"]
SEG_SCALE = alt.Scale(domain=list(SEGMENT_COLORS), range=list(SEGMENT_COLORS.values()))


def _open_view(name: str) -> None:
    st.session_state["detail_view"] = name


def render(ctx: dict) -> None:
    m, flows, thr = ctx["manifest"], ctx["flows"], ctx["threshold"]
    loc, summary = session_localization(m), integrity_summary(m)
    counts = summary["message_status"]
    modes = compare_modes(flows, loc, thr)
    n_above = int((flows["risk"] >= thr).sum())

    if ctx["ref"].group == "trained on":
        st.warning("The served model was trained on this session, so its scores here are optimistic. "
                   "Pick a 'held-out test' or 'live demo' session for an honest picture.")
    st.markdown(badge(ctx["ref"].group, "#6C757D") + "  " + badge(f"window {ctx['window']:g} s") + "  "
                + badge(f"threshold {thr:.2f}", "#8E6BBF"), unsafe_allow_html=True)

    cards = st.columns(4)
    with cards[0]:
        st.metric("1. Integrity", f"{counts['tamper']} / {summary['n_messages']} failed", help=HELP["tamper_msgs"])
        st.button("Explain", key="card_int", on_click=_open_view, args=("Integrity",), width="stretch")
    with cards[1]:
        st.metric("2. ML risk", f"{n_above} of {len(flows)} flows", help=HELP["flows_above"])
        st.button("Explain", key="card_ml", on_click=_open_view, args=("ML risk",), width="stretch")
    with cards[2]:
        st.metric("3. Tampered segment", loc["segment"] if loc else "none", help=HELP["segment"])
        st.button("Explain", key="card_seg", on_click=_open_view, args=("Tampered segment",), width="stretch")
    with cards[3]:
        st.metric("4. Fused alert", f"{int(modes.iloc[2]['alerts'])} alerts (ML-only {int(modes.iloc[0]['alerts'])})",
                  help=HELP["fused"])
        st.button("Explain", key="card_fused", on_click=_open_view, args=("Fused alert",), width="stretch")

    st.subheader("Why localization helps: the three alerting modes on this session")
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(modes, hide_index=True, width="stretch", column_config={
            "irrelevant alerts (evaluation only)": st.column_config.NumberColumn(help=HELP["eval_only"]),
            "recall (evaluation only)": st.column_config.NumberColumn(help=HELP["eval_only"])})
    with right:
        chart = alt.Chart(modes.rename(columns={"irrelevant alerts (evaluation only)": "irrelevant"})).mark_bar().encode(
            x=alt.X("irrelevant:Q", title="irrelevant alerts (evaluation only)"),
            y=alt.Y("mode:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("mode:N", legend=None, scale=alt.Scale(range=["#9AA0A6", "#B8860B", "#54A24B"])),
            tooltip=["mode", "alerts", "irrelevant"]).properties(height=130)
        st.altair_chart(chart, width="stretch")
    if loc is None:
        st.info("The integrity layer found no tamper, so time-window and localized modes raise nothing. "
                "An ML-only system still alerts on benign-looking flows.")

    st.divider()
    view = st.radio("Detail", VIEWS, horizontal=True, key="detail_view", label_visibility="collapsed")
    {"Integrity": _integrity, "ML risk": _ml_risk, "Tampered segment": _segment, "Fused alert": _fused}[view](ctx, loc, summary)


# ---------------------------------------------------------------------------------------------


def _integrity(ctx, loc, summary):
    m = ctx["manifest"]
    st.markdown("**What the integrity layer checked.** Every message carries a hop-chained HMAC: each node "
                "verifies the previous tag, then adds its own over `digest || previous tag`. Nodes send signed "
                "events (digests and tags only, never payloads) to a collector with a hash-chained log.")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(pd.DataFrame({"verdict": list(summary["message_status"]), "messages": list(summary["message_status"].values())}),
                     hide_index=True)
        chain = ctx.get("log_check") or summary["log_integrity"]
        st.write("Event-log hash chain:", "**intact**" if chain.get("ok") else "**BROKEN**")
    with c2:
        rows = [{"message": k, "verdict": v["status"], "seconds": v.get("first_ts")} for k, v in m["localizations"].items()]
        df = pd.DataFrame(rows)
        if not df.empty and df["seconds"].notna().any():
            df["seconds"] = df["seconds"] - df["seconds"].min()
            st.altair_chart(alt.Chart(df).mark_circle(size=70).encode(
                x=alt.X("seconds:Q", title="seconds into the session"), y=alt.Y("verdict:N", title=None),
                color=alt.Color("verdict:N", legend=None,
                                scale=alt.Scale(domain=["ok", "tamper", "config_error", "incomplete"],
                                                range=["#9AA0A6", BROKEN_COLOR, "#B8860B", "#8E6BBF"])),
                tooltip=["message", "verdict", "seconds"]).properties(height=170), width="stretch")
    n_inc = summary["message_status"].get("incomplete", 0)
    if n_inc:
        st.caption(f"{n_inc} message(s) are 'incomplete' (missing events): not evidence either way, so they never become weak labels.")


def _ml_risk(ctx, loc, summary):
    flows, thr = ctx["flows"], ctx["threshold"]
    st.markdown("**What the ML model says.** An XGBoost model, first trained on public data and then adapted with "
                "integrity-derived weak labels, scores every flow on every segment. Risk is calibrated on weak validation labels.")
    st.dataframe(segment_risk_table(flows, thr), width="stretch")
    hist = alt.Chart(flows).mark_bar(opacity=0.8).encode(
        x=alt.X("risk:Q", bin=alt.Bin(maxbins=20), title="calibrated risk"), y=alt.Y("count()", title="flows"),
        color=alt.Color("segment:N", scale=SEG_SCALE), tooltip=["segment", "count()"]).properties(height=200)
    rule = alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(color="#E45756", strokeDash=[5, 3]).encode(x="t:Q")
    st.altair_chart(hist + rule, width="stretch")
    st.caption("ML alone sees flows on all three segments, so it cannot say where the problem is. That is what the "
               "integrity layer adds.")


def _segment(ctx, loc, summary):
    m = ctx["manifest"]
    msgs = m["localizations"]
    tamper_ids = [k for k, v in msgs.items() if v.get("status") == "tamper"]
    options = tamper_ids + [k for k in msgs if k not in tamper_ids]
    msg_id = st.selectbox("Message to explain", options, help="Messages the integrity layer flagged come first.") if options else None
    if msg_id is None:
        st.info("This session has no messages.")
        return
    res = msgs[msg_id]
    evidence = res.get("evidence") or {}
    st.markdown(f"**Verdict for this message: `{res['status']}`**" + (f", segment **{res['segment']}**" if res.get("segment") else ""))
    if evidence:
        st.graphviz_chart(hop_chain_dot(evidence), width="stretch")
        st.dataframe(hop_table(evidence), hide_index=True, width="stretch")
        st.info(WHY_NOT_ANOTHER_SEGMENT)
    else:
        st.info("No per-segment evidence was recorded for this message (the collector had incomplete events).")

    live = fetch_message_events(data.COLLECTOR, ctx["ref"].run_id, msg_id) if ctx["lab_up"] else None
    if live:
        st.markdown("**What each node reported (live from the collector, digests and tags only).** The segment where "
                    "'digest out' of the emitter differs from 'digest in' of the verifier is where the data changed.")
        st.dataframe(pd.DataFrame(live), hide_index=True, width="stretch")
    with st.expander("Why a key or configuration error is NOT called a tamper"):
        example = {s: {"digest_changed": False, "tag_changed": False, "verify_failed": s == "B",
                       "emitter_tag_matches_collector_key": s != "B"} for s in "ABC"}
        st.write("If the digest and tag arrive exactly as they were sent but verification still fails (for example a node "
                 "holds the wrong key), nothing was altered on the wire. Such cases are labelled `config_error` and are "
                 "excluded from weak labels, so a misconfiguration cannot masquerade as an attack.")
        st.dataframe(hop_table(example)[["segment", "digest changed", "tag changed", "verification failed", "verdict"]],
                     hide_index=True)
    plan = m["plan"].get("tamper_segment")
    agrees = (loc["segment"] if loc else None) == plan
    st.caption(f"Attacker plan (evaluation only): {plan or 'no tamper'}; the integrity layer {'agrees' if agrees else 'DISAGREES'}.")


def _candidates(ctx, loc):
    flows = ctx["flows"]
    if loc is None:
        return flows.sort_values("raw_risk", ascending=False).head(25), False
    mask = candidate_mask("localized", flows, loc, flows["in_failure_window"])
    return flows[mask].sort_values("raw_risk", ascending=False).head(25), True


def _fused(ctx, loc, summary):
    flows, thr = ctx["flows"], ctx["threshold"]
    cands, fused = _candidates(ctx, loc)
    if fused:
        st.markdown(f"**Fused alert.** Integrity evidence says segment **{loc['segment']}** "
                    f"({loc['n_failed_messages']} failed messages). Only flows on that segment inside the failure window "
                    f"are candidates, ranked by calibrated risk.")
    else:
        st.info("No integrity failure, so no fused alert is raised. Below, for comparison, are the highest-risk flows an "
                "ML-only system would look at.")
    window_q = window_label_quality(flows)
    if loc is not None and window_q["flows_in_windows"]:
        st.caption(f"At this window the weak labels cover {window_q['flows_in_windows']} flows; "
                   f"{window_q['precision']:.0%} are truly malicious and {window_q['recall']:.0%} of the true attack flows are "
                   f"covered (evaluation only).")
    if cands.empty:
        st.warning("No candidate flows at this window; widen the window in the sidebar.")
        return
    table = cands.assign(**{"alert": cands["risk"] >= thr}).rename(columns={"true_malicious_eval_only": "malicious (evaluation only)"})
    show = table[["segment", "risk", "raw_risk", "alert", "malicious (evaluation only)"]].reset_index().rename(columns={"index": "flow"})
    st.caption("Click a row (or use the picker) to open the flow explanation.")
    ev = st.dataframe(show, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
                      key=f"alert_table_{ctx['ref'].run_id}", column_config={
                          "risk": st.column_config.ProgressColumn("calibrated risk", min_value=0.0, max_value=1.0, format="%.2f"),
                          "malicious (evaluation only)": st.column_config.CheckboxColumn(help=HELP["eval_only"])})
    clicked = ev.selection.rows[0] if ev.selection.rows else None
    flow_ids = list(show["flow"])
    pick = st.selectbox("Flow", flow_ids, index=clicked if clicked is not None else 0, key=f"flow_pick_{ctx['ref'].run_id}_{clicked}")
    _flow_detail(ctx, int(pick))


def _flow_detail(ctx, flow_idx: int) -> None:
    b, X, flows, thr = ctx["bundle"], ctx["X"], ctx["flows"], ctx["threshold"]
    row = X.loc[flow_idx]
    st.markdown(f"### Why flow {flow_idx}?")
    contrib = flow_contributions(b, row, n_features=10)
    contrib["label"] = [f"{f} = {v:.3g}" for f, v in zip(contrib.feature, contrib.value)]
    c1, c2 = st.columns([3, 2])
    with c1:
        chart = alt.Chart(contrib).mark_bar().encode(
            x=alt.X("contribution:Q", title="pushes risk down  <  log-odds contribution (TreeSHAP)  >  pushes risk up"),
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=300)),
            color=alt.condition(alt.datum.contribution > 0, alt.value(BROKEN_COLOR), alt.value("#4C78A8")),
            tooltip=["feature", "meaning", "value", "contribution"]).properties(height=300)
        st.altair_chart(chart, width="stretch")
    with c2:
        st.dataframe(contrib[["feature", "meaning"]], hide_index=True, width="stretch")
        st.caption(f"Flow is on segment {flows.loc[flow_idx, 'segment']}; malicious per attacker log (evaluation only): "
                   f"{bool(flows.loc[flow_idx, 'true_malicious_eval_only'])}.")

    st.markdown("### What changes if...?")
    st.caption("Move a feature and the model re-scores this flow immediately. This shows which behaviours the model "
               "treats as suspicious (it is an explanation tool, not a statement about real traffic).")
    ranges = b.get("feature_ranges", {})
    edits, cols = {}, st.columns(3)
    for col, (_, r) in zip(cols, contrib.head(3).iterrows()):
        lo, mid, hi = ranges.get(r.feature, (r.value, r.value, r.value))
        lo, hi = min(lo, r.value), max(hi, r.value)
        if hi <= lo:
            col.write(f"{r.feature}: constant")
            continue
        edits[r.feature] = col.slider(r.feature, float(lo), float(hi), float(r.value), key=f"cf_{ctx['ref'].run_id}_{flow_idx}_{r.feature}",
                                      help=r.meaning)
    base, new = counterfactual_risk(b, row), counterfactual_risk(b, row, edits)
    m1, m2, m3 = st.columns(3)
    m1.metric("Original calibrated risk", f"{base['calibrated']:.2f}")
    m2.metric("With your edits", f"{new['calibrated']:.2f}", delta=f"{new['calibrated'] - base['calibrated']:+.2f}")
    m3.metric("Alert at threshold?", "yes" if new["calibrated"] >= thr else "no")
