"""The shuffled-weak-label control, for the dashboard: does the gain depend on the integrity labels?"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from securetrace.adapt.data import build_adapt_data, load_saved_model
from securetrace.adapt.tree import adapt_xgboost
from securetrace.data.lab_split import split_lab_sessions


def stored_shuffle_control(phase4_csv: Path, before_csv: Path, held_out_mutator: str) -> pd.DataFrame:
    """Mean PR-AUC over seeds from the logged runs: the public model before adaptation (Phase 3),
    adapted with the real integrity-derived weak labels, adapted with the same labels shuffled."""
    df = pd.read_csv(phase4_csv)
    df = df[(df.study == "main") & (df.held_out_mutator == held_out_mutator) & df["mode"].isin(["naive", "elkan_noto", "mil", "nnpu"])]
    before = pd.read_csv(before_csv)
    before = before[before.held_out_mutator == held_out_mutator].groupby("family").pr_auc.mean()
    keys = ["family", "mode"]
    out = df[~df.shuffled].groupby(keys).pr_auc.mean().rename("adapted, real labels").to_frame()
    out["adapted, shuffled labels"] = df[df.shuffled].groupby(keys).pr_auc.mean()
    out = out.reset_index()
    out["public model (no adaptation)"] = out.family.map(before)
    out["method"] = out.family + " / " + out["mode"]
    return out[["method", "public model (no adaptation)", "adapted, real labels", "adapted, shuffled labels"]]


def live_shuffle_control(bundle: dict, sessions_dir: Path, extracted_dir: Path, seed: int = 0) -> dict:
    """Re-train the served XGBoost adaptation twice on the TRAIN sessions, once with the real weak
    labels and once with them randomly permuted, and score both on the held-out test split with true
    labels (evaluation only). Takes tens of seconds."""
    manifests = sorted(Path(sessions_dir).glob("*.json"))
    split = split_lab_sessions(manifests, held_out_mutator=bundle["held_out_mutator"], seed=0)
    public, meta = load_saved_model(Path(bundle.get("model_dir", "data/processed/models/phase1_public_baseline")), "xgboost", seed)
    data = build_adapt_data(split, extracted_dir, meta["feature_columns"], window_seconds=bundle["window_seconds"], seed=seed)
    before = average_precision_score(data.y_test, public.predict_proba(data.X_test)[:, 1])
    out = {"public model (no adaptation)": float(before), "attack rate (chance)": float(data.y_test.mean())}
    for name, shuffle in (("adapted, real labels", False), ("adapted, shuffled labels", True)):
        s_train = np.random.default_rng(10_000 + seed).permutation(data.train.s) if shuffle else data.train.s
        s_val = np.random.default_rng(10_000 + seed).permutation(data.val.s) if shuffle else data.val.s
        model, _ = adapt_xgboost(public, data.train.X, s_train, data.train.run_id, data.val.X, s_val, mode="naive", seed=seed)
        out[name] = float(average_precision_score(data.y_test, model.predict_proba(data.X_test)[:, 1]))
    return out
