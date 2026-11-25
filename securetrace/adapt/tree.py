"""Tree-model adaptation: continue boosting the public XGBoost model on weak lab labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from securetrace.adapt.pu import (
    elkan_noto_dataset, estimate_label_frequency, fit_labelled_vs_unlabelled, out_of_fold_g,
)

TREE_MODES = ("naive", "elkan_noto")


def adapt_xgboost(
    public: XGBClassifier, X: pd.DataFrame, s: np.ndarray, groups: np.ndarray,
    X_val: pd.DataFrame, s_val: np.ndarray, *, mode: str, seed: int, n_new_trees: int = 60,
    learning_rate: float = 0.05,
) -> tuple[XGBClassifier, dict]:
    """`naive`: unlabelled treated as negative (the baseline to beat). `elkan_noto`: unlabelled
    flows enter as fractional positives/negatives from the Elkan-Noto weights."""
    info: dict = {}
    if mode == "naive":
        X_fit, y_fit, w_fit = X, s.astype(float), None
    elif mode == "elkan_noto":
        g_oof = out_of_fold_g(X, s, groups, seed)
        g_full = fit_labelled_vs_unlabelled(X, s, seed)
        c = estimate_label_frequency(g_full, X_val, s_val)
        info["label_frequency_c"] = c
        X_fit, y_fit, w_fit = elkan_noto_dataset(X, s, g_oof, c)
    else:
        raise ValueError(f"unknown tree adaptation mode {mode!r}; choose from {TREE_MODES}")

    model = XGBClassifier(
        objective="binary:logistic", n_estimators=n_new_trees, max_depth=6, learning_rate=learning_rate,
        tree_method="hist", n_jobs=-1, random_state=seed,
    )
    model.fit(X_fit, y_fit, sample_weight=w_fit, xgb_model=public.get_booster())
    return model, info
