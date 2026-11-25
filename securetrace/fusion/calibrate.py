"""Score calibration (isotonic regression, temperature/Platt scaling) and the reliability metrics
(ECE, reliability bins) used to judge it.

Calibrators are fit on WEAK validation labels only. ECE against the true labels is reported on the
lab test split for evaluation, never used to fit anything."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class IsotonicCalibrator:
    def fit(self, score: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(score, y)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        return self.iso.predict(score)


class TemperatureCalibrator:
    """p' = sigmoid((logit(p) + b) / T): temperature plus a bias, fit by minimising log-loss."""

    def fit(self, score: np.ndarray, y: np.ndarray) -> "TemperatureCalibrator":
        z, y = _logit(np.asarray(score)), np.asarray(y).astype(float)

        def nll(log_t: float) -> float:
            t = np.exp(log_t)
            # closed-form-free: optimise T for the best bias at each T via a 1-D inner search
            inner = minimize_scalar(lambda b: _nll(z, y, t, b), bounds=(-10, 10), method="bounded")
            return inner.fun

        res = minimize_scalar(nll, bounds=(-3, 3), method="bounded")
        self.t = float(np.exp(res.x))
        self.b = float(minimize_scalar(lambda b: _nll(z, y, self.t, b), bounds=(-10, 10), method="bounded").x)
        return self

    def predict(self, score: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(_logit(np.asarray(score)) + self.b) / self.t))


def _nll(z: np.ndarray, y: np.ndarray, t: float, b: float) -> float:
    p = np.clip(1.0 / (1.0 + np.exp(-(z + b) / t)), 1e-9, 1 - 1e-9)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> list[dict]:
    p, y = np.asarray(p, dtype=float), np.asarray(y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append({"bin": b, "n": int(m.sum()), "confidence": float(p[m].mean()), "accuracy": float(y[m].mean())})
    return out


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = reliability_bins(p, y, n_bins)
    n = sum(b["n"] for b in bins)
    return float(sum(b["n"] * abs(b["confidence"] - b["accuracy"]) for b in bins) / n) if n else float("nan")
