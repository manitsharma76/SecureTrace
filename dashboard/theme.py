"""Shared look: one colour per segment, the standing caveat, and hover help for every metric."""

from __future__ import annotations

import streamlit as st

from securetrace.fusion.explain import BROKEN_COLOR, SEGMENT_COLORS  # noqa: F401  (re-exported)

HELP = {
    "threshold": "A flow raises an alert when its calibrated risk (estimated probability of being malicious) "
                 "is at least this value. Lower = more alerts, higher = fewer.",
    "window": "Width of the time window around each integrity failure. Flows inside it on the located "
              "segment become weak positive labels / fused-alert candidates. Wider windows catch more "
              "true attack flows but also more benign ones.",
    "tamper_msgs": "Messages whose hop-chained HMAC check failed and whose digest/tag changed on a segment.",
    "flows_above": "Flows (all segments) whose calibrated risk reaches the threshold: what an ML-only system would alert on.",
    "segment": "The segment where the emitted digest/tag differs from the received one. Chosen by cryptography, not by ML.",
    "fused": "Alerts that survive when only the located segment's flows inside the failure window are considered.",
    "eval_only": "Uses the attacker's own per-flow log. Shown for evaluation; never an input to any score.",
}


def caveat() -> None:
    st.caption("Synthetic lab traffic only: these results do not carry over to a real network without validation "
               "there. Columns marked 'evaluation only' use the attacker's own ground truth and never feed a score.")


def badge(text: str, color: str = "#4C78A8") -> str:
    return (f"<span style='background:{color};color:white;padding:2px 10px;border-radius:12px;"
            f"font-size:0.8rem;font-weight:600'>{text}</span>")
