"""Shared preprocessing and the public-data sample used by the domain-adaptation comparators."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from securetrace.data.public import load_public_frame, split_by_day
from securetrace.models.factory import SignedLog1p


def make_prep():
    """Median-impute, signed log1p, standardise: the same recipe as the public MLP/logreg."""
    return make_pipeline(SimpleImputer(strategy="median"), SignedLog1p(), StandardScaler())


def shared_columns(columns: list[str], X_lab: pd.DataFrame) -> list[str]:
    """Public-model columns that the lab CICFlowMeter output actually populates. A column that is
    all-NaN in the lab would be a constant (imputed) in one domain and informative in the other, a
    pure artefact that domain adaptation would be asked to 'fix'."""
    return [c for c in columns if c in X_lab.columns and X_lab[c].notna().any()]


def load_public_sample(config_path: Path, columns: list[str], n_rows: int, seed: int,
                       cache_dir: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Labelled public training-day flows (the same days the public models trained on), subsampled
    with a fixed seed. Cached because cleaning the full CSVs is slow."""
    digest = hashlib.sha1("|".join(columns).encode("utf-8")).hexdigest()[:10]
    cache = cache_dir / f"public_train_all_{digest}.pkl"
    if cache.exists():
        X, y = pd.read_pickle(cache)
    else:
        cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["data"]
        files = [Path(p) for p in cfg["public_csv_paths"]]
        X_all, y_all, meta = load_public_frame(
            files, label_col=cfg.get("label_col", "Label"), day_col=cfg.get("day_col", "Day"),
            drop_attempted=cfg.get("drop_attempted", True), drop_features=cfg.get("drop_features", []),
        )
        X, _, _, y, _, _ = split_by_day(X_all, y_all, meta["day"], train_days=cfg["train_days"],
                                        val_days=cfg["val_days"], test_days=cfg["test_days"])
        X = X.reindex(columns=columns)
        cache_dir.mkdir(parents=True, exist_ok=True)
        pd.to_pickle((X, y), cache)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_rows, len(X)), replace=False)
    return X.iloc[idx].reset_index(drop=True), np.asarray(y.iloc[idx] if hasattr(y, "iloc") else y[idx]).astype(int)
