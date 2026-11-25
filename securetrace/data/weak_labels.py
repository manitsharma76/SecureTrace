"""Weak-label builder: integrity failure -> (segment, time window) -> positive bag.

Per CLAUDE.md "Weak labels are bag-level": (segment, time window) is positive, everything else is
unlabelled (not negative). Key and config errors are excluded from labels -- they say the keys or
configuration disagree, not that anything was tampered (see securetrace.integrity.localize). Only
an integrity failure with status "tamper" produces a positive bag.

This builder consumes the SAME session manifests used for the true-label evaluation
(securetrace.data.lab), but it must never use the manifest's `ground_truth` field for anything
other than, later, scoring label quality -- the weak labels here come only from `localizations`
(what the collector observed), exactly what a real deployment would have.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from securetrace.data.lab import flow_csv_path, load_session_manifest
from securetrace.features.schema import drop_identity_features, harmonize_columns


@dataclass(frozen=True)
class Bag:
    run_id: str
    segment: str
    window_start: float
    window_end: float


def tamper_events(manifest: dict) -> list[tuple[str, float]]:
    """(segment, event_time) for every message the collector localized as a real tamper.

    Excludes "config_error" (key/config disagreement, not tampering) and "incomplete" (missing
    events, not evidence either way) -- see CLAUDE.md "Key and config errors are excluded".
    """
    events = []
    for result in manifest.get("localizations", {}).values():
        if result.get("status") == "tamper" and result.get("segment"):
            events.append((result["segment"], result["first_ts"]))
    return events


def build_bags(manifest: dict, window_seconds: float) -> list[Bag]:
    """Merge nearby same-segment tamper events into bags, so one burst of failures on the same
    segment becomes one wider bag instead of many overlapping ones."""
    events = sorted(tamper_events(manifest))
    bags: list[Bag] = []
    for segment in sorted({s for s, _ in events}):
        times = sorted(t for s, t in events if s == segment)
        start = end = times[0]
        for t in times[1:]:
            if t - end <= window_seconds:
                end = t
            else:
                bags.append(Bag(manifest["run_id"], segment, start - window_seconds / 2, end + window_seconds / 2))
                start = end = t
        bags.append(Bag(manifest["run_id"], segment, start - window_seconds / 2, end + window_seconds / 2))
    return bags


def _flow_window(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    parsed = pd.to_datetime(frame["Timestamp"], errors="coerce", utc=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    start = (parsed - epoch).dt.total_seconds()
    duration_s = pd.to_numeric(frame["Flow Duration"], errors="coerce").fillna(0) / 1e6
    return start, start + duration_s


def build_session_weak_labels(
    manifest_path: str | Path, extracted_dir: str | Path, *, window_seconds: float = 2.0
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """One session's flows with WEAK labels: 1 if the flow falls in a positive bag, 0 (unlabelled,
    not a confirmed negative) otherwise.

    Returns (X, y_weak, meta); meta carries run_id, segment and the bag key each positive flow
    was assigned to (None for unlabelled flows).
    """
    manifest = load_session_manifest(manifest_path)
    run_id = manifest["run_id"]
    extracted_dir = Path(extracted_dir)
    bags = build_bags(manifest, window_seconds)
    bags_by_segment: dict[str, list[Bag]] = {}
    for bag in bags:
        bags_by_segment.setdefault(bag.segment, []).append(bag)
    for segment_bags in bags_by_segment.values():
        segment_bags.sort(key=lambda b: b.window_start)

    frames, metas = [], []
    for segment in ("A", "B", "C"):
        csv_path = flow_csv_path(extracted_dir, run_id, segment)
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, low_memory=False)
        if frame.empty:
            continue
        frame = harmonize_columns(frame)
        start, end = _flow_window(frame)

        segment_bags = bags_by_segment.get(segment, [])
        starts = [b.window_start for b in segment_bags]
        bag_keys, weak = [], []
        for s, e in zip(start, end):
            key = None
            if pd.notna(s) and segment_bags:
                # Bags are sorted and (by construction, for window_seconds > 0) non-overlapping;
                # the candidate just before this flow's start is the only one that can overlap it.
                i = bisect.bisect_right(starts, s) - 1
                for j in (i, i + 1):
                    if 0 <= j < len(segment_bags):
                        bag = segment_bags[j]
                        if s <= bag.window_end and e >= bag.window_start:
                            key = (bag.run_id, bag.segment, bag.window_start, bag.window_end)
                            break
            bag_keys.append(key)
            weak.append(key is not None)

        frames.append(frame)
        metas.append(pd.DataFrame({"run_id": run_id, "segment": segment, "bag_key": bag_keys, "weak_positive": weak,
                                   "flow_start": start.to_numpy()}))

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

    y_weak = meta["weak_positive"].astype(bool)
    return working.reset_index(drop=True), y_weak.reset_index(drop=True), meta.reset_index(drop=True)


def build_sessions_weak_labels(
    manifest_paths: Iterable[str | Path], extracted_dir: str | Path, *, window_seconds: float = 2.0
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    X_parts, y_parts, meta_parts = [], [], []
    for manifest_path in manifest_paths:
        X, y, meta = build_session_weak_labels(manifest_path, extracted_dir, window_seconds=window_seconds)
        X_parts.append(X)
        y_parts.append(y)
        meta_parts.append(meta)
    shared_columns = sorted(set.intersection(*(set(part.columns) for part in X_parts)))
    X = pd.concat([part[shared_columns] for part in X_parts], ignore_index=True)
    y = pd.concat(y_parts, ignore_index=True)
    meta = pd.concat(meta_parts, ignore_index=True)
    return X, y, meta


def label_quality(
    manifest_paths: Iterable[str | Path], extracted_dir: str | Path, window_seconds: float
) -> dict:
    """What fraction of flows in a positive bag are actually malicious (per the TRUE labels), at
    this window width -- DEVELOPMENT_PLAN.md "Measure label quality against the true labels"."""
    from securetrace.data.lab import load_session_true_labels

    manifest_paths = list(manifest_paths)
    total_positive_bag_flows = total_truly_malicious_in_bags = 0
    total_truly_malicious = total_flows = 0
    for manifest_path in manifest_paths:
        _, y_weak, meta_weak = build_session_weak_labels(manifest_path, extracted_dir, window_seconds=window_seconds)
        _, y_true, meta_true = load_session_true_labels(manifest_path, extracted_dir)
        # Both loaders iterate segments A, B, C in the same order over the same CSVs, so rows align.
        assert len(y_weak) == len(y_true), "weak/true label row count mismatch -- loaders diverged"
        total_positive_bag_flows += int(y_weak.sum())
        total_truly_malicious_in_bags += int((y_weak & y_true).sum())
        total_truly_malicious += int(y_true.sum())
        total_flows += len(y_true)

    precision = total_truly_malicious_in_bags / total_positive_bag_flows if total_positive_bag_flows else float("nan")
    recall = total_truly_malicious_in_bags / total_truly_malicious if total_truly_malicious else float("nan")
    return {
        "window_seconds": window_seconds,
        "n_flows": total_flows,
        "n_positive_bag_flows": total_positive_bag_flows,
        "n_truly_malicious": total_truly_malicious,
        "bag_precision": precision,  # fraction of a positive bag that is actually malicious
        "bag_recall": recall,  # fraction of true malicious flows captured by some positive bag
    }
