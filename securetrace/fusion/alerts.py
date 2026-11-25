"""Alerting modes for one incident (a test session in which the integrity layer localized a tamper).

* ``ml_only``     - rank every flow of the session by model score; no integrity input.
* ``time_window`` - how a SIEM correlates: take flows (on ANY segment) that overlap the time window
                    of an integrity alert, rank them by model score. Time says "when", not "where".
* ``localized``   - Novelty B: take only the flows on the segment the HMAC chain localized, inside
                    its window, rank them by calibrated score and attach the integrity evidence.

Ground truth is the attacker's own true per-flow label, used for scoring only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MODES = ("ml_only", "time_window", "localized")


def candidate_mask(mode: str, flows: pd.DataFrame, localization: dict | None, window_flows: pd.Series) -> np.ndarray:
    """`flows` has columns segment; `window_flows` marks flows inside any integrity-failure window
    (any segment); the localized segment comes from the collector's localization."""
    if mode == "ml_only":
        return np.ones(len(flows), dtype=bool)
    if mode == "time_window":
        return window_flows.to_numpy(dtype=bool)
    if mode == "localized":
        seg = localization["segment"] if localization else None
        return (window_flows.to_numpy(dtype=bool)) & (flows["segment"].to_numpy() == seg)
    raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    """Precision among the k highest-scored candidates; fewer than k candidates -> over what exists
    (0 when there are none, i.e. the incident was missed entirely)."""
    if len(score) == 0:
        return 0.0
    top = np.argsort(-score, kind="stable")[:k]
    return float(np.mean(y_true[top]))


def incident_metrics(y_true: np.ndarray, rank_score: np.ndarray, prob: np.ndarray, mask: np.ndarray,
                     threshold: float, ks: tuple[int, ...] = (5, 10)) -> dict:
    """Candidates are ranked by `rank_score` (precision@k); an alert FIRES when the calibrated
    probability `prob` reaches `threshold` (alert counts, irrelevant alerts)."""
    yc, pc, rc = y_true[mask], prob[mask], rank_score[mask]
    fired = pc >= threshold
    out = {
        "n_candidates": int(mask.sum()),
        "n_alerts": int(fired.sum()),
        "irrelevant_alerts": int((fired & (yc == 0)).sum()),
        "true_alerts": int((fired & (yc == 1)).sum()),
        "recall": float((fired & (yc == 1)).sum() / max(1, y_true.sum())),
    }
    for k in ks:
        out[f"precision_at_{k}"] = precision_at_k(yc, rc, k)
    return out


def top_feature_contributions(booster, X: pd.DataFrame, n_features: int = 3) -> list[list[tuple[str, float]]]:
    """Exact TreeSHAP values from XGBoost (pred_contribs), top features per flow by |contribution|."""
    import xgboost as xgb

    contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]  # last column is the bias
    cols = list(X.columns)
    out = []
    for row in contrib:
        top = np.argsort(-np.abs(row))[:n_features]
        out.append([(cols[i], float(row[i])) for i in top])
    return out
