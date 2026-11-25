import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from securetrace.fusion.alerts import candidate_mask, incident_metrics, precision_at_k, top_feature_contributions
from securetrace.fusion.calibrate import (
    IsotonicCalibrator, TemperatureCalibrator, expected_calibration_error, reliability_bins,
)


def _overconfident(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    z = 2.0 * (y - 0.5) + rng.normal(scale=1.5, size=n)  # noisy evidence
    p = 1 / (1 + np.exp(-4.0 * z))                        # far too confident
    return p, y


def test_calibrators_reduce_ece_of_an_overconfident_score():
    p, y = _overconfident()
    half = len(y) // 2
    raw = expected_calibration_error(p[half:], y[half:])
    for cal in (IsotonicCalibrator(), TemperatureCalibrator()):
        cal.fit(p[:half], y[:half])
        assert expected_calibration_error(cal.predict(p[half:]), y[half:]) < raw


def test_ece_is_zero_for_perfectly_calibrated_probabilities_and_bins_cover_all_points():
    p = np.array([0.0] * 50 + [1.0] * 50)
    y = np.array([0] * 50 + [1] * 50)
    assert expected_calibration_error(p, y) == 0.0
    assert sum(b["n"] for b in reliability_bins(p, y)) == 100


def test_precision_at_k_and_empty_candidate_set():
    assert precision_at_k(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]), 2) == 0.5
    assert precision_at_k(np.array([]), np.array([]), 5) == 0.0  # a missed incident scores 0, not NaN


def test_localized_mode_keeps_only_the_localized_segment_inside_the_window():
    flows = pd.DataFrame({"segment": ["A", "A", "B", "B", "C"]})
    in_window = pd.Series([True, False, True, True, True])
    loc = {"segment": "B"}
    assert candidate_mask("localized", flows, loc, in_window).tolist() == [False, False, True, True, False]
    assert candidate_mask("time_window", flows, loc, in_window).tolist() == [True, False, True, True, True]
    assert candidate_mask("ml_only", flows, loc, in_window).all()
    with pytest.raises(ValueError):
        candidate_mask("bogus", flows, loc, in_window)


def test_a_localization_to_the_wrong_segment_cannot_surface_the_true_attack():
    """Guards the claim behind Novelty B: a mis-localized incident finds nothing, it is not silently
    rescued by other segments' flows."""
    flows = pd.DataFrame({"segment": ["A", "B", "B"]})
    in_window = pd.Series([True, True, True])
    y = np.array([0, 1, 1])
    mask = candidate_mask("localized", flows, {"segment": "A"}, in_window)
    res = incident_metrics(y, np.array([0.1, 0.9, 0.8]), np.array([0.1, 0.9, 0.8]), mask, 0.5)
    assert res["true_alerts"] == 0 and res["recall"] == 0.0


def test_incident_metrics_counts_alerts_and_irrelevant_alerts():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.7, 0.2, 0.1])
    res = incident_metrics(y, p, p, np.ones(5, dtype=bool), 0.5)
    assert (res["n_alerts"], res["irrelevant_alerts"], res["true_alerts"]) == (3, 1, 2)
    assert res["recall"] == 1.0 and res["precision_at_5"] == pytest.approx(0.4)


def test_treeshap_contributions_name_the_informative_feature():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    y = (X["a"] > 0).astype(int)
    model = xgb.XGBClassifier(n_estimators=20, max_depth=2).fit(X, y)
    top = top_feature_contributions(model.get_booster(), X.iloc[:5], n_features=1)
    assert all(t[0][0] == "a" for t in top)
