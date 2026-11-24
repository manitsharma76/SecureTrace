from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_curve


def fpr_at_recall(y_true: np.ndarray, score: np.ndarray, recall: float) -> float:
    """Lowest false-positive rate reachable while keeping TPR >= `recall` (oracle threshold)."""
    fpr, tpr, _ = roc_curve(y_true, score)
    reachable = fpr[tpr >= recall]
    return float(reachable.min()) if reachable.size else float("nan")


def select_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximising macro-F1 on the given (validation) data."""
    candidates = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197)))
    best, best_f1 = 0.5, -1.0
    for threshold in candidates:
        value = f1_score(y_true, score >= threshold, average="macro", zero_division=0)
        if value > best_f1:
            best, best_f1 = float(threshold), value
    return best


def evaluate_scores(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    threshold: float,
    recalls: tuple[float, ...] = (0.90, 0.95),
) -> dict[str, float]:
    """Metrics for one evaluation set. `threshold` must come from validation data, never the test set.

    Sets with a single class have no PR-AUC / FPR-at-recall; those come back as NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    out: dict[str, float] = {
        "n": float(len(y_true)),
        "attack_rate": float(y_true.mean()) if len(y_true) else float("nan"),
        "macro_f1": float(f1_score(y_true, score >= threshold, average="macro", zero_division=0)),
    }
    two_classes = len(np.unique(y_true)) == 2
    out["pr_auc"] = float(average_precision_score(y_true, score)) if two_classes else float("nan")
    for r in recalls:
        out[f"fpr_at_recall_{int(r * 100)}"] = fpr_at_recall(y_true, score, r) if two_classes else float("nan")
    return out
