"""Load generated lab sessions (scripts/generate_lab_runs.py) and attach TRUE per-flow labels
from the attacker's own ground truth.

Per CLAUDE.md ("True lab labels are for evaluation only") and DEVELOPMENT_PLAN.md ("The lab
records true per-flow labels ... These are never used for training. They are used only to score
the weak-label methods"), everything in this module is for evaluation and the phase 2 go/no-go
check -- never for training the production model. The weak-label builder (integrity failures ->
bag labels) is a separate, deliberately coarser path.

CICFlowMeter writes pcap timestamps in UTC; the attacker's ground truth uses `time.time()`
(also UTC-based, epoch seconds). Both sides are compared in UTC here -- confirmed empirically by
diffing a session's recorded timestamps against its CICFlowMeter output, not assumed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from securetrace.features.schema import drop_identity_features, harmonize_columns

# Tolerance for the flow-vs-tamper-event window overlap test. CICFlowMeter's "Timestamp" is a
# flow's first-packet time, not the attacker's per-message relay time, so a small margin absorbs
# clock/parsing slack without blurring across unrelated messages (session gaps are >> this).
WINDOW_MARGIN_SECONDS = 1.0


def load_session_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flow_csv_path(extracted_dir: Path, run_id: str, segment: str) -> Path:
    return extracted_dir / f"{run_id}_seg{segment}.pcap_Flow.csv"


def _tamper_windows(ground_truth: list[dict], segment: str) -> list[float]:
    """Epoch-second timestamps of tampered messages the attacker relayed on this segment."""
    return [entry["ts"] for entry in ground_truth if entry["segment"] == segment and entry["tampered"]]


def _flow_is_near_any(flow_start: float, flow_end: float, event_times: list[float], margin: float) -> bool:
    return any(flow_start - margin <= t <= flow_end + margin for t in event_times)


def load_session_true_labels(
    manifest_path: str | Path, extracted_dir: str | Path, *, margin_seconds: float = WINDOW_MARGIN_SECONDS
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """One session's flows (all three segments) with TRUE labels, for evaluation only.

    Returns (X, y_true, meta) with meta carrying run_id, segment and the raw CICFlowMeter Label
    column (always "NeedManualLabel" for lab data -- the true label comes from ground truth, not
    from this column).
    """
    manifest = load_session_manifest(manifest_path)
    run_id = manifest["run_id"]
    extracted_dir = Path(extracted_dir)

    frames, metas = [], []
    for segment in ("A", "B", "C"):
        csv_path = flow_csv_path(extracted_dir, run_id, segment)
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, low_memory=False)
        if frame.empty:
            continue
        frame = harmonize_columns(frame)

        parsed = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True)
        epoch = pd.Timestamp("1970-01-01", tz="UTC")
        start = (parsed - epoch).dt.total_seconds()  # NaT rows -> NaN, handled below
        duration_s = pd.to_numeric(frame["Flow Duration"], errors="coerce").fillna(0) / 1e6
        end = start + duration_s

        tamper_times = _tamper_windows(manifest["ground_truth"], segment)
        label = [
            _flow_is_near_any(s, e, tamper_times, margin_seconds) if pd.notna(s) else False
            for s, e in zip(start, end)
        ]

        frames.append(frame)
        metas.append(pd.DataFrame({
            "run_id": run_id, "segment": segment, "label": frame.get("Label", "NeedManualLabel"),
            "true_malicious": pd.array(label, dtype="boolean"),
        }))

    if not frames:
        raise ValueError(f"No non-empty flow CSVs found for run {run_id} under {extracted_dir}")

    combined = pd.concat(frames, ignore_index=True)
    meta = pd.concat(metas, ignore_index=True)

    working = combined.drop(columns=["Label"], errors="ignore")
    working = drop_identity_features(working)
    for column in working.columns:
        if not pd.api.types.is_numeric_dtype(working[column]):
            working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.replace([np.inf, -np.inf], np.nan).astype(np.float32)

    y_true = meta["true_malicious"].astype(bool)
    return working.reset_index(drop=True), y_true.reset_index(drop=True), meta.reset_index(drop=True)


def load_sessions_true_labels(
    manifest_paths: Iterable[str | Path], extracted_dir: str | Path, *, margin_seconds: float = WINDOW_MARGIN_SECONDS
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Concatenate `load_session_true_labels` over several sessions (run_id stays in `meta`,
    so callers can still split by session -- see CLAUDE.md "split by session")."""
    X_parts, y_parts, meta_parts = [], [], []
    for manifest_path in manifest_paths:
        X, y, meta = load_session_true_labels(manifest_path, extracted_dir, margin_seconds=margin_seconds)
        X_parts.append(X)
        y_parts.append(y)
        meta_parts.append(meta)
    X = pd.concat(X_parts, ignore_index=True)
    shared_columns = sorted(set.intersection(*(set(part.columns) for part in X_parts)))
    X = X[shared_columns]
    y = pd.concat(y_parts, ignore_index=True)
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, y, meta
