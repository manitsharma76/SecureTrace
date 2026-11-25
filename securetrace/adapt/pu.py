"""Positive-Unlabelled building blocks: Elkan-Noto label-frequency / class-prior estimation,
Elkan-Noto reweighting, and the non-negative PU (nnPU) risk."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from securetrace.models.factory import SignedLog1p


def _g_model(seed: int):
    return make_pipeline(
        SimpleImputer(strategy="median"), SignedLog1p(), StandardScaler(),
        LogisticRegression(C=0.1, max_iter=300, random_state=seed),
    )


def fit_labelled_vs_unlabelled(X: pd.DataFrame, s: np.ndarray, seed: int):
    """g(x) = P(s=1 | x): a classifier of 'in a positive bag' vs 'unlabelled'."""
    return _g_model(seed).fit(X, s)


def out_of_fold_g(X: pd.DataFrame, s: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 3) -> np.ndarray:
    """Out-of-fold g, grouped by session, so reweighting does not just echo g's own training fit."""
    g = np.zeros(len(s), dtype=float)
    n_groups = len(np.unique(groups))
    for train_idx, test_idx in GroupKFold(n_splits=min(n_splits, n_groups)).split(X, s, groups):
        if len(np.unique(s[train_idx])) < 2:
            g[test_idx] = float(s[train_idx].mean())
            continue
        model = fit_labelled_vs_unlabelled(X.iloc[train_idx], s[train_idx], seed)
        g[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    return g


def estimate_label_frequency(g_model, X_val: pd.DataFrame, s_val: np.ndarray) -> float:
    """Elkan-Noto estimator e1: c = mean g(x) over held-out labelled positives (never the
    training positives, which g has already fit)."""
    positives = X_val[s_val == 1]
    if len(positives) == 0:
        return 0.5
    return float(np.clip(g_model.predict_proba(positives)[:, 1].mean(), 0.05, 1.0))


def estimate_class_prior(s: np.ndarray, c: float) -> float:
    """pi = P(y=1) = P(s=1) / c (SCAR assumption: labelled positives are a random subset of positives)."""
    return float(np.clip(np.mean(s) / c, 0.01, 0.9))


def elkan_noto_dataset(X: pd.DataFrame, s: np.ndarray, g: np.ndarray, c: float):
    """Weighted (X, y, w): labelled -> positive w=1; each unlabelled flow appears twice, once as
    positive with weight w(x) = (1-c)/c * g/(1-g) (clipped to [0, 1]) and once as negative with 1-w."""
    labelled = s == 1
    g = np.clip(g, 1e-4, 1 - 1e-4)
    w = np.clip((1 - c) / c * g / (1 - g), 0.0, 1.0)
    Xu = X[~labelled]
    X_all = pd.concat([X[labelled], Xu, Xu], ignore_index=True)
    y_all = np.concatenate([np.ones(labelled.sum()), np.ones(len(Xu)), np.zeros(len(Xu))])
    w_all = np.concatenate([np.ones(labelled.sum()), w[~labelled], 1.0 - w[~labelled]])
    keep = w_all > 1e-6
    return X_all[keep].reset_index(drop=True), y_all[keep], w_all[keep]


def nnpu_loss(z_p: torch.Tensor, z_u: torch.Tensor, prior: float, *, beta: float = 0.0, gamma: float = 1.0) -> torch.Tensor:
    """Non-negative PU risk (Kiryo et al. 2017) with the sigmoid loss. If the estimated negative
    risk drops below -beta the step ascends on it instead, which stops the model overfitting by
    labelling the unlabelled set negative."""
    def loss(z, y):
        return torch.sigmoid(-y * z)

    pos_risk = prior * loss(z_p, 1.0).mean()
    neg_risk = loss(z_u, -1.0).mean() - prior * loss(z_p, -1.0).mean()
    if neg_risk.item() < -beta:
        return -gamma * neg_risk
    return pos_risk + neg_risk
