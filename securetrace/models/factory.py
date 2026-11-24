"""Phase 1 model families behind one interface: `build_model(family, params, seed)`
returns an object with `fit(X, y)` and `predict_proba(X)[:, 1]`."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAMILIES = ("logreg", "random_forest", "xgboost", "lightgbm", "mlp")


class SignedLog1p(BaseEstimator, TransformerMixin):
    """Flow statistics are heavy-tailed (and a few, like IAT, can be negative); compress them
    before scaling so linear/neural models are not dominated by outliers."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X))


def _neural_preprocess():
    return [SimpleImputer(strategy="median"), SignedLog1p(), StandardScaler()]


def build_model(family: str, params: dict[str, Any] | None, seed: int):
    p = dict(params or {})
    if family == "logreg":
        return make_pipeline(
            *_neural_preprocess(),
            LogisticRegression(C=p.get("C", 1.0), max_iter=p.get("max_iter", 500), random_state=seed),
        )
    if family == "random_forest":
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 150),
            max_depth=p.get("max_depth"),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            n_jobs=-1,
            random_state=seed,
        )
    if family == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 6),
            learning_rate=p.get("eta", 0.05),
            subsample=p.get("subsample", 1.0),
            colsample_bytree=p.get("colsample_bytree", 1.0),
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", -1),
            num_leaves=p.get("num_leaves", 63),
            learning_rate=p.get("eta", 0.05),
            subsample=p.get("subsample", 1.0),
            colsample_bytree=p.get("colsample_bytree", 1.0),
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    if family == "mlp":
        return make_pipeline(
            *_neural_preprocess(),
            MLPClassifier(
                hidden_layer_sizes=tuple(p.get("hidden", (128, 64))),
                early_stopping=True,
                validation_fraction=0.1,
                max_iter=p.get("max_iter", 60),
                random_state=seed,
            ),
        )
    raise ValueError(f"Unknown model family '{family}'. Choose from {FAMILIES}.")


def predict_attack_score(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
