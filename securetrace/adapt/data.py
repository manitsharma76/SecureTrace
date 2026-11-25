"""Assemble the adaptation data: weak-labelled TRAIN/VAL flows and true-labelled TEST flows.

Weak labels are bag-level (CLAUDE.md): a flow is positive if it falls in a (segment, window) bag
that an integrity failure opened; every other flow is UNLABELLED (not negative). True labels are
used for the test split only, for scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from securetrace.data.lab import load_sessions_true_labels, load_session_manifest
from securetrace.data.lab_split import SessionSplit
from securetrace.data.weak_labels import build_sessions_weak_labels


def load_saved_model(model_dir: Path, family: str, seed: int) -> tuple[object, dict]:
    stem = Path(model_dir) / f"{family}_seed{seed}"
    model = joblib.load(f"{stem}.joblib")
    meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
    return model, meta


@dataclass
class WeakSet:
    X: pd.DataFrame          # aligned to the public model's feature columns
    s: np.ndarray            # 1 = in a positive bag, 0 = UNLABELLED
    run_id: np.ndarray
    bag_id: np.ndarray       # str; positive bags carry their integrity bag key, others a time bucket


@dataclass
class AdaptData:
    train: WeakSet
    val: WeakSet
    X_test: pd.DataFrame
    y_test: np.ndarray       # TRUE labels, scoring only
    window_seconds: float


def _weak_set(paths: list[Path], extracted_dir: Path, columns: list[str], window: float) -> WeakSet:
    X, s, meta = build_sessions_weak_labels(paths, extracted_dir, window_seconds=window)
    X = X.reindex(columns=columns)
    bucket = np.floor(meta["flow_start"].fillna(-1).to_numpy() / max(window, 1e-9)).astype(int)
    bag = np.where(
        meta["bag_key"].notna(),
        meta["bag_key"].astype(str),
        meta["run_id"] + "|" + meta["segment"] + "|u" + pd.Series(bucket).astype(str),
    )
    return WeakSet(X=X, s=s.to_numpy().astype(int), run_id=meta["run_id"].to_numpy(), bag_id=bag)


def subsample_tamper_sessions(paths: list[Path], fraction: float, seed: int) -> list[Path]:
    """Keep all clean sessions and a `fraction` of the tampered ones (label-efficiency curve:
    performance vs number of tamper incidents)."""
    if fraction >= 1.0:
        return list(paths)
    tampered = [p for p in paths if load_session_manifest(p)["plan"]["tamper_segment"] is not None]
    clean = [p for p in paths if p not in set(tampered)]
    rng = np.random.default_rng(seed)
    keep = max(1, int(round(len(tampered) * fraction)))
    chosen = rng.choice(len(tampered), size=keep, replace=False) if tampered else []
    return clean + [tampered[i] for i in sorted(chosen)]


def build_adapt_data(
    split: SessionSplit, extracted_dir: Path, columns: list[str], *, window_seconds: float,
    tamper_fraction: float = 1.0, seed: int = 0,
) -> AdaptData:
    train_paths = subsample_tamper_sessions(split.train, tamper_fraction, seed)
    train = _weak_set(train_paths, extracted_dir, columns, window_seconds)
    val = _weak_set(split.val, extracted_dir, columns, window_seconds)
    X_test, y_test, _ = load_sessions_true_labels(split.test, extracted_dir)
    X_test = X_test.reindex(columns=columns)
    if set(train.run_id) & set(val.run_id):
        raise AssertionError("a session appears in both adaptation train and val")
    return AdaptData(train, val, X_test, y_test.to_numpy().astype(int), window_seconds)
