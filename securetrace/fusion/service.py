"""Everything the dashboard shows for one lab session, computed outside Streamlit so it is testable.

A "bundle" is the Phase 4 weak-label-adapted XGBoost plus its isotonic calibrator and feature
columns (see scripts/build_dashboard_bundle.py). Integrity evidence comes from the session manifest
(offline) or the live collector (online); true labels appear only as a clearly labelled evaluation
column, never as an input to any score.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from securetrace.data.lab import load_session_manifest, load_session_true_labels
from securetrace.data.weak_labels import build_bags, build_session_weak_labels
from securetrace.fusion.alerts import top_feature_contributions

STATUSES = ("ok", "tamper", "config_error", "incomplete")


def load_bundle(path: str | Path) -> dict:
    return joblib.load(path)


def integrity_summary(manifest: dict) -> dict:
    """Per-message verdict counts plus the hash-chained log's verification result."""
    counts = {s: 0 for s in STATUSES}
    for loc in manifest.get("localizations", {}).values():
        counts[loc.get("status", "incomplete")] = counts.get(loc.get("status", "incomplete"), 0) + 1
    return {"message_status": counts, "log_integrity": manifest.get("log_integrity", {}),
            "n_messages": sum(counts.values())}


def session_localization(manifest: dict) -> dict | None:
    """The segment named by most 'tamper' messages, with one message's evidence. None if no tamper."""
    hits = [v for v in manifest.get("localizations", {}).values() if v.get("status") == "tamper" and v.get("segment")]
    if not hits:
        return None
    seg = pd.Series([h["segment"] for h in hits]).mode().iloc[0]
    sample = next(h for h in hits if h["segment"] == seg)
    return {"segment": seg, "n_failed_messages": len(hits), "evidence": sample.get("evidence", {})}


def score_session(manifest_path: str | Path, extracted_dir: str | Path, bundle: dict,
                  window_seconds: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (flows, X). One row per flow in `flows`: segment, start time, raw and calibrated risk, whether it lies inside an
    integrity-failure window on its own segment, any-segment time overlap, and the TRUE label
    (evaluation only)."""
    X, in_window, meta = build_session_weak_labels(manifest_path, extracted_dir, window_seconds=window_seconds)
    _, y_true, _ = load_session_true_labels(manifest_path, extracted_dir)
    X = X.reindex(columns=bundle["columns"])
    raw = bundle["model"].predict_proba(X)[:, 1]
    manifest = load_session_manifest(manifest_path)
    bags = build_bags(manifest, window_seconds)
    start, end = meta["flow_start"].to_numpy(), meta["flow_end"].to_numpy()
    any_window = np.zeros(len(meta), dtype=bool)
    for bag in bags:
        any_window |= (start <= bag.window_end) & (end >= bag.window_start)
    flows = pd.DataFrame({
        "segment": meta["segment"].to_numpy(), "start": start, "raw_risk": raw,
        "risk": bundle["calibrator"].predict(raw), "in_failure_window": in_window.to_numpy().astype(bool),
        "in_any_segment_window": any_window, "true_malicious_eval_only": y_true.to_numpy().astype(bool),
    })
    return flows, X


def fused_alert(manifest: dict, flows: pd.DataFrame, X: pd.DataFrame, bundle: dict, threshold: float = 0.5,
                top: int = 10) -> dict:
    """Novelty B alert: only flows on the localized segment inside its failure window, ranked by
    risk, with the integrity evidence and TreeSHAP features attached. No localization -> no alert."""
    loc = session_localization(manifest)
    ml_only_alerts = int((flows["risk"] >= threshold).sum())
    if loc is None:
        return {"localization": None, "alerts": [], "ml_only_alerts": ml_only_alerts, "window_alerts": 0}
    cand = flows[(flows["segment"] == loc["segment"]) & flows["in_failure_window"]]
    cand = cand.sort_values("raw_risk", ascending=False)
    shown = cand.head(top)
    feats = top_feature_contributions(bundle["model"].get_booster(), X.reindex(columns=bundle["columns"]).loc[shown.index])
    alerts = [{"flow_row": int(i), "segment": r.segment, "risk": float(r.risk),
               "top_features": [(n, round(v, 3)) for n, v in f],
               "true_malicious_eval_only": bool(r.true_malicious_eval_only)} for (i, r), f in zip(shown.iterrows(), feats)]
    return {"localization": loc, "alerts": alerts, "ml_only_alerts": ml_only_alerts,
            "window_alerts": int((flows.loc[flows["in_any_segment_window"], "risk"] >= threshold).sum()),
            "fused_alerts_above_threshold": int((cand["risk"] >= threshold).sum())}
