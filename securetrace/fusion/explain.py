"""Explainability helpers for the dashboard (no Streamlit imports, fully testable).

* Hop-chain view: why the integrity layer blames one segment (and never another).
* Feature glossary: plain-language names for CICFlowMeter features.
* Per-flow TreeSHAP contributions and counterfactual re-scoring.
* The three alerting modes compared on one session.

True labels (the columns ending in ``_eval_only``) are only ever summarised for evaluation; no
function here feeds them into a score.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests

from securetrace.fusion.alerts import candidate_mask, incident_metrics
from securetrace.integrity.chain import NODE_ORDER, SEGMENT_EMITTER, SEGMENT_ORDER, SEGMENT_VERIFIER

SEGMENT_COLORS = {"A": "#4C78A8", "B": "#F58518", "C": "#54A24B"}
BROKEN_COLOR, OK_COLOR = "#E45756", "#9AA0A6"

# --------------------------------------------------------------------------------------------
# Hop chain
# --------------------------------------------------------------------------------------------


def segment_verdicts(evidence: dict) -> dict[str, dict]:
    """Per-segment verdict from the collector's evidence, using the same rule as
    securetrace.integrity.localize: a changed digest or tag is a TAMPER on that segment; a failed
    verification (or an emitter tag the collector's key does not reproduce) with an unchanged
    hand-off is a KEY/CONFIG problem, never a tamper."""
    out = {}
    for segment in SEGMENT_ORDER:
        ev = evidence.get(segment)
        if ev is None:
            out[segment] = {"verdict": "unknown", "reason": "no evidence for this segment"}
            continue
        if ev["digest_changed"] or ev["tag_changed"]:
            what = "the data digest" if ev["digest_changed"] else "the HMAC tag"
            out[segment] = {"verdict": "tamper",
                            "reason": f"{what} the verifier received differs from what the emitter sent"}
        elif ev["verify_failed"] or not ev["emitter_tag_matches_collector_key"]:
            out[segment] = {"verdict": "config_error",
                            "reason": "hand-off is identical but the tag does not verify: keys/configuration "
                                      "disagree, which is not evidence of tampering"}
        else:
            out[segment] = {"verdict": "ok", "reason": "digest and tag arrived exactly as sent and verify"}
    return out


def hop_table(evidence: dict) -> pd.DataFrame:
    verdicts = segment_verdicts(evidence)
    rows = []
    for segment in SEGMENT_ORDER:
        ev = evidence.get(segment, {})
        rows.append({
            "segment": segment,
            "emitter -> verifier": f"{SEGMENT_EMITTER[segment]} -> {SEGMENT_VERIFIER[segment]}",
            "digest changed": ev.get("digest_changed"), "tag changed": ev.get("tag_changed"),
            "verification failed": ev.get("verify_failed"),
            "emitter tag matches collector key": ev.get("emitter_tag_matches_collector_key"),
            "verdict": verdicts[segment]["verdict"], "why": verdicts[segment]["reason"],
        })
    return pd.DataFrame(rows)


def first_broken_segment(evidence: dict) -> str | None:
    for segment, v in segment_verdicts(evidence).items():
        if v["verdict"] == "tamper":
            return segment
    return None


WHY_NOT_ANOTHER_SEGMENT = (
    "Each relay tags what it actually received and forwards that, so after a tamper on segment k the "
    "next hop's hand-off is internally consistent and verifies. Only the segment where the emitted "
    "digest/tag differs from the received one is blamed, never a later segment."
)


def hop_chain_dot(evidence: dict) -> str:
    """Graphviz source for sender -> relay1 -> relay2 -> receiver, segment edges coloured by verdict."""
    verdicts = segment_verdicts(evidence)
    lines = ["digraph G {", "rankdir=LR; bgcolor=transparent;",
             'node [shape=box, style="rounded,filled", fillcolor="#F3F4F6", fontname="Helvetica", color="#9AA0A6"];',
             'edge [fontname="Helvetica", penwidth=3];']
    for node in NODE_ORDER:
        lines.append(f'{node} [label="{node}"];')
    for segment in SEGMENT_ORDER:
        v = verdicts[segment]["verdict"]
        color = {"tamper": BROKEN_COLOR, "config_error": "#B8860B"}.get(v, SEGMENT_COLORS[segment])
        mark = {"tamper": "TAMPER", "config_error": "key/config", "ok": "verified", "unknown": "?"}[v]
        lines.append(f'{SEGMENT_EMITTER[segment]} -> {SEGMENT_VERIFIER[segment]} '
                     f'[label="segment {segment}\\n{mark}", color="{color}", fontcolor="{color}"];')
    lines.append("}")
    return "\n".join(lines)


def fetch_message_events(collector_url: str, run_id: str, msg_id: str, timeout: float = 3.0) -> list[dict] | None:
    """Per-node digests/tags for one message from the live collector (None if unreachable). Values are
    shortened for display; no payload and no key material is ever present in these events."""
    try:
        r = requests.get(f"{collector_url}/runs/{run_id}/events", timeout=timeout)
        if not r.ok:
            return None
        events = [e for e in r.json() if e.get("msg_id") == msg_id]
    except requests.RequestException:
        return None
    order = {n: i for i, n in enumerate(NODE_ORDER)}
    rows = []
    for e in sorted(events, key=lambda e: order.get(e["node"], 9)):
        short = lambda h: (h[:10] + "...") if h else "-"
        rows.append({
            "node": e["node"], "digest in": short(e.get("digest_in")), "digest out": short(e.get("digest_out")),
            "newest tag in": short((e.get("chain_in") or [None])[-1]), "newest tag out": short((e.get("chain_out") or [None])[-1]),
            "tag verified": e.get("verify_ok"), "error kind": e.get("error_kind") or "-",
        })
    return rows


# --------------------------------------------------------------------------------------------
# Feature glossary
# --------------------------------------------------------------------------------------------

_DIRECTION = [("Fwd", "forward-direction (sender to receiver)"), ("FWD", "forward-direction (sender to receiver)"),
              ("Bwd", "reply-direction (receiver to sender)")]
_QUANTITY = [
    ("Packet Length", "packet length"), ("Segment Payload Length", "TCP payload length"),
    ("Segment Header Length", "TCP header length"), ("Header Length", "header length"),
    ("IAT", "time between packets"), ("Init Win Bytes", "initial TCP window size"),
    ("Act Data Pkts", "packets carrying data"), ("Bulk Rate Avg", "bulk-transfer rate"),
    ("Bytes/Bulk Avg", "bytes per bulk transfer"), ("Packet/Bulk Avg", "packets per bulk transfer"),
    ("PSH Flags", "PSH flag count"), ("RST Flags", "RST flag count"), ("URG Flags", "URG flag count"),
    ("TCP Retrans. Count", "TCP retransmissions"), ("Packets/s", "packet rate"),
    ("Seg Size Min", "minimum segment size"), ("Segment Size Avg", "average segment size"),
    ("Packet", "packet count"), ("packets", "packet count"),
]
_STAT = [("Std", "variation (std)"), ("Mean", "average"), ("Max", "maximum"), ("Min", "minimum"),
         ("Total", "total"), ("Avg", "average")]
_EXACT = {
    "Down/Up Ratio": "ratio of reply bytes to forward bytes",
    "Flow Duration": "how long the flow lasted", "Flow Bytes/s": "bytes per second over the whole flow",
    "Flow Packets/s": "packets per second over the whole flow",
    "Active Std": "variation in active periods", "Active Mean": "average active period", "Active Max": "longest active period",
    "Active Min": "shortest active period", "Idle Max": "longest idle gap", "Idle Mean": "average idle gap",
    "Idle Min": "shortest idle gap", "Idle Std": "variation in idle gaps",
    "Total Fwd Packet": "number of forward packets", "Total Bwd packets": "number of reply packets",
    "Total Length of Fwd Packet": "total forward bytes", "Total Length of Bwd Packet": "total reply bytes",
    "ACK Flag Count": "ACK flag count", "SYN Flag Count": "SYN flag count", "FIN Flag Count": "FIN flag count",
    "PSH Flag Count": "PSH flag count", "RST Flag Count": "RST flag count", "URG Flag Count": "URG flag count",
    "Subflow Fwd Bytes": "forward bytes in the sub-flow", "Subflow Bwd Bytes": "reply bytes in the sub-flow",
}


def describe_feature(name: str) -> str:
    """Plain-language description of a CICFlowMeter feature, e.g.
    'Bwd Packet Length Std' -> 'reply-direction packet length: variation (std)'."""
    if name in _EXACT:
        return _EXACT[name]
    rest, direction = name, ""
    for token, text in _DIRECTION:
        if rest.startswith(token + " "):
            direction, rest = text + " ", rest[len(token) + 1:]
            break
    stat = ""
    for token, text in _STAT:
        if rest.endswith(" " + token):
            stat, rest = text, rest[: -len(token) - 1]
            break
    for token, text in _QUANTITY:
        if rest == token or rest.startswith(token):
            qty = text
            break
    else:
        qty = rest.lower()
    out = f"{direction}{qty}".strip()
    return f"{out}: {stat}" if stat else out


# --------------------------------------------------------------------------------------------
# Per-flow explanation
# --------------------------------------------------------------------------------------------


def flow_contributions(bundle: dict, x_row: pd.Series, n_features: int = 10) -> pd.DataFrame:
    """Exact TreeSHAP contributions (XGBoost pred_contribs, log-odds scale) for one flow, largest
    absolute effect first, with the feature value and a plain-language description."""
    import xgboost as xgb

    X = x_row.reindex(bundle["columns"]).to_frame().T.astype(float)
    contrib = bundle["model"].get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[0]
    bias, values = float(contrib[-1]), contrib[:-1]
    top = np.argsort(-np.abs(values))[:n_features]
    df = pd.DataFrame({
        "feature": [bundle["columns"][i] for i in top], "value": [float(X.iloc[0, i]) for i in top],
        "contribution": [float(values[i]) for i in top],
    })
    df["meaning"] = df["feature"].map(describe_feature)
    df.attrs["baseline_log_odds"] = bias
    return df


def counterfactual_risk(bundle: dict, x_row: pd.Series, edits: dict[str, float] | None = None) -> dict:
    """Re-score one flow with some features changed. Returns raw model probability and the
    calibrated probability; with no edits it reproduces the flow's original score."""
    row = x_row.reindex(bundle["columns"]).astype(float).copy()
    for feature, value in (edits or {}).items():
        if feature not in row.index:
            raise KeyError(f"unknown feature {feature!r}")
        row[feature] = float(value)
    raw = float(bundle["model"].predict_proba(row.to_frame().T)[0, 1])
    return {"raw": raw, "calibrated": float(bundle["calibrator"].predict(np.array([raw]))[0])}


# --------------------------------------------------------------------------------------------
# Alerting modes compared
# --------------------------------------------------------------------------------------------


def compare_modes(flows: pd.DataFrame, localization: dict | None, threshold: float) -> pd.DataFrame:
    """The three alerting modes on one session. 'irrelevant' and 'recall' use the attacker's true
    flow labels and are for evaluation only."""
    y = flows["true_malicious_eval_only"].to_numpy().astype(int)
    rank, prob = flows["raw_risk"].to_numpy(), flows["risk"].to_numpy()
    rows = []
    for mode, label in (("ml_only", "ML only (all segments)"),
                        ("time_window", "Time-window correlation (any segment)"),
                        ("localized", "Localized fusion (integrity-located segment)")):
        if mode == "ml_only":
            mask = np.ones(len(flows), dtype=bool)
        elif localization is None:
            mask = np.zeros(len(flows), dtype=bool)  # no integrity failure -> no correlated alert
        else:
            window = flows["in_any_segment_window"] if mode == "time_window" else flows["in_failure_window"]
            mask = candidate_mask(mode, flows, localization, window)
        m = incident_metrics(y, rank, prob, mask, threshold)
        rows.append({"mode": label, "candidate flows": m["n_candidates"], "alerts": m["n_alerts"],
                     "irrelevant alerts (evaluation only)": m["irrelevant_alerts"],
                     "recall (evaluation only)": round(m["recall"], 3)})
    return pd.DataFrame(rows)


def window_label_quality(flows: pd.DataFrame) -> dict:
    """How good the weak labels are for this session at the chosen window (evaluation only):
    precision = share of flows inside an integrity-failure window that are truly malicious."""
    inside = flows["in_failure_window"].to_numpy()
    y = flows["true_malicious_eval_only"].to_numpy()
    n_in, n_true = int(inside.sum()), int(y.sum())
    return {"flows_in_windows": n_in, "truly_malicious": n_true,
            "precision": float(y[inside].mean()) if n_in else float("nan"),
            "recall": float(inside[y].mean()) if n_true else float("nan")}


def segment_risk_table(flows: pd.DataFrame, threshold: float) -> pd.DataFrame:
    return flows.groupby("segment").agg(
        flows=("risk", "size"), max_risk=("risk", "max"),
        flows_above_threshold=("risk", lambda s: int((s >= threshold).sum())),
    ).round(3)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
