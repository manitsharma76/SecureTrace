"""'How it works' and 'Limitations' tabs."""

from __future__ import annotations

import re

import streamlit as st

from dashboard import data

PIPELINE_DOT = """
digraph G { rankdir=LR; bgcolor=transparent;
 node [shape=box, style="rounded,filled", fillcolor="#F3F4F6", fontname="Helvetica", color="#9AA0A6"];
 edge [fontname="Helvetica", color="#6C757D"];
 sender -> relay1 [label="seg A", color="#4C78A8", penwidth=3];
 relay1 -> relay2 [label="seg B", color="#F58518", penwidth=3];
 relay2 -> receiver [label="seg C", color="#54A24B", penwidth=3];
 attacker [label="on-path attacker\\n(isolated Docker net)", fillcolor="#FDE2E1", color="#E45756"];
 attacker -> relay1 [style=dashed, color="#E45756"];
 collector [label="collector\\nhash-chained log", fillcolor="#E8F0FE"];
 sender -> collector [style=dotted, label="signed events"]; relay1 -> collector [style=dotted]; relay2 -> collector [style=dotted]; receiver -> collector [style=dotted];
 sensor [label="sensors\\npcap -> CICFlowMeter", fillcolor="#E8F0FE"];
 collector -> loc [label="HMAC check"]; loc [label="localize segment\\n+ weak labels", fillcolor="#E6F4EA"];
 sensor -> model [label="flows"]; loc -> model [label="weak labels"]; model [label="ML model\\n(public data -> adapted)", fillcolor="#E6F4EA"];
 model -> alert; loc -> alert; alert [label="fused alert\\nsegment + ranked flows + evidence", fillcolor="#FFF4CE"];
}
"""


def render_how() -> None:
    st.subheader("How SecureTrace works")
    st.markdown("**Cryptography answers: did the data change, and on which segment? The ML model answers: what suspicious "
                "network activity happened there?** Each message carries a hop-chained HMAC; a failed check gives a "
                "tampered-segment location and a weak, time-window label. Those labels adapt and focus an intrusion model.")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    st.markdown("""
**Three research questions, and what we found** (see the Trust tab and FINAL_REPORT.md):

1. **Q2: Do integrity-derived weak labels close the gap to a new network?** Yes in this lab: PR-AUC rises from about 0.2 to 0.8 to 0.9, and shuffled labels remove the gain.
2. **Q3: How do they compare with label-free adaptation?** Anomaly detection, self-supervised pretraining and domain adaptation stay near chance or are unstable; weak labels beat all of them.
3. **Q4: Does localizing to a segment and fusing with the ML score give better alerts?** It cuts irrelevant alerts per incident about 10x at the same recall.

Weak-label adaptation and localization-plus-fusion are **variants or combinations of prior work** (positive-unlabelled and
multiple-instance learning, alert correlation), not claimed as first.
""")


def render_limits() -> None:
    st.subheader("Limitations")
    report = data.ROOT / "FINAL_REPORT.md"
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    m = re.search(r"## Limitations\n(.*?)(\n## |\Z)", text, re.S)
    st.markdown(m.group(1).strip() if m else
                "- Synthetic lab: results do not carry over to a real network.\n- The attack footprint was enlarged on purpose.\n- Small data: 100 sessions.")
