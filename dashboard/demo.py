"""Live demo: run a real attack in the Docker lab, or an offline simulator on the real integrity code."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.theme import badge
from securetrace.fusion.explain import hop_chain_dot, segment_verdicts
from securetrace.fusion.live import run_live_session
from securetrace.fusion.simulator import MUTATIONS, run_simulation

LAB_MUTATORS = ["flip_byte", "rewrite_size"]


def render(ctx: dict) -> None:
    lab_up = ctx["lab_up"]
    default = 0 if lab_up else 1
    mode = st.radio("Demo mode", ["Lab: run a real attack in the Docker lab", "Simulator: no Docker needed"],
                    index=default, horizontal=True, key="demo_mode")
    if mode.startswith("Lab"):
        st.markdown(badge("LAB MODE: real traffic", "#54A24B" if lab_up else "#E45756"), unsafe_allow_html=True)
        _lab(ctx) if lab_up else st.error(
            "The lab collector is not reachable. Start it with `docker compose --env-file .env -f lab/docker-compose.yml up -d` "
            "or switch to the simulator, which runs the same integrity code without Docker.")
    else:
        st.markdown(badge("SIMULATOR MODE: in-process", "#4C78A8"), unsafe_allow_html=True)
        _simulator()


def _lab(ctx: dict) -> None:
    st.markdown("**Run a real session.** A sender pushes hop-chained messages through two relays to a receiver while an "
                "on-path attacker (inside the isolated Docker network only) tampers with one segment. Packets are "
                "captured, flows extracted, integrity events verified and the fused alert built. It takes about 1 to 2 minutes.")
    c1, c2, c3 = st.columns([1, 1, 1])
    segment = c1.selectbox("Segment to tamper", ["A", "B", "C", "none (clean session)"], index=1)
    mutator = c2.selectbox("Attack", LAB_MUTATORS, help="flip_byte changes one payload byte; rewrite_size truncates or pads the message.",
                           disabled=segment.startswith("none"))
    go = c3.button("Run live attack", type="primary", width="stretch")
    if go:
        seg = None if segment.startswith("none") else segment
        with st.status("Running a real session in the lab...", expanded=True) as status:
            log = st.empty()
            lines: list[str] = []

            def on_step(text: str) -> None:
                lines.append(text)
                log.markdown("\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines)))

            result = run_live_session(seg, mutator, on_step)
            if result["ok"]:
                status.update(label=f"Finished: session {result['run_id']}", state="complete")
            else:
                status.update(label="The session failed", state="error")
        if result["ok"]:
            st.session_state["last_demo"] = {"run_id": result["run_id"], "planned": seg, "mutator": mutator}
            st.cache_data.clear()  # the catalog must pick up the new session
        else:
            st.error(result["error"])
    last = st.session_state.get("last_demo")
    if last:
        st.success(f"Session `{last['run_id']}` is ready. Planned tamper (evaluation only): {last['planned'] or 'none'}.")
        st.button("Open it in the Session explorer", on_click=lambda: st.session_state.update(pending_run=last["run_id"]))
        st.caption("Then use the Session explorer tab: the cards, the hop-chain view and the flow explanations all use this new session.")


def _simulator() -> None:
    st.markdown("**Tamper simulator.** This pushes one message through the project's real hop-chained HMAC code "
                "(sender, relay1, relay2, receiver) and lets an attacker change it on one segment. Keys are generated "
                "fresh at random on every run and are never shown. Change anything below and the result recomputes.")
    c1, c2, c3 = st.columns([3, 1, 1])
    message = c1.text_area("Message", "Wire 100 EUR to account 1234", height=90, key="sim_msg")
    segment = c2.selectbox("Tamper on segment", ["none", "A", "B", "C"], index=2, key="sim_seg",
                           help="Where the on-path attacker changes the data.")
    mutation = c3.selectbox("Attacker action", MUTATIONS, key="sim_mut", disabled=segment == "none",
                            help="flip_byte changes one byte; truncate removes the end; pad appends bytes.")
    result = run_simulation(message, None if segment == "none" else segment, mutation)
    loc = result["localization"]

    if loc["status"] == "tamper":
        st.error(f"Verification FAILED. Tamper localized to segment **{loc['segment']}** "
                 f"(attacker acted on segment {segment}: {'correct' if loc['segment'] == segment else 'WRONG'}).")
    elif loc["status"] == "ok":
        st.success("All hops verified. No tamper detected.")
    else:
        st.warning(f"Status: {loc['status']}")
    st.graphviz_chart(hop_chain_dot(loc["evidence"]), width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("**What happens, step by step**")
        for i, (who, text) in enumerate(result["steps"], 1):
            st.markdown(f"{i}. **{who}**: {text}")
    with right:
        st.markdown("**What each node saw** (digests and tags shortened)")
        st.dataframe(pd.DataFrame(result["hops"]), hide_index=True, width="stretch")
        st.markdown("**What the attacker did on each segment**")
        st.dataframe(pd.DataFrame(result["wire"]), hide_index=True, width="stretch")
    st.caption("Try it: change one letter, move the attacker to another segment, or choose 'none'. "
               "The blamed segment always follows the attacker, and never a later one.")
