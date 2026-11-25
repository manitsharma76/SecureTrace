import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from securetrace.fusion.calibrate import IsotonicCalibrator
from securetrace.fusion.explain import (
    compare_modes, counterfactual_risk, describe_feature, first_broken_segment, flow_contributions,
    hop_chain_dot, hop_table, segment_risk_table, segment_verdicts, window_label_quality,
)
from securetrace.integrity.localize import localize
from tests.test_integrity import SEG_KEYS, flip_byte, run_path, truncate


def _evidence(**changes):
    base = {s: {"digest_changed": False, "tag_changed": False, "verify_failed": False,
                "emitter_tag_matches_collector_key": True} for s in "ABC"}
    for seg, ev in changes.items():
        base[seg].update(ev)
    return base


def test_verdicts_match_the_real_localizer_for_every_segment_and_attack():
    for seg in "ABC":
        for attack in (flip_byte, truncate):
            loc = localize(run_path(tamper=(seg, attack)), SEG_KEYS)
            v = segment_verdicts(loc.evidence)
            assert [s for s in "ABC" if v[s]["verdict"] == "tamper"] == loc.tampered_segments == [seg]
            assert first_broken_segment(loc.evidence) == seg
    clean = localize(run_path(), SEG_KEYS)
    assert all(v["verdict"] == "ok" for v in segment_verdicts(clean.evidence).values())


def test_key_or_config_error_is_not_called_a_tamper():
    v = segment_verdicts(_evidence(B={"verify_failed": True, "emitter_tag_matches_collector_key": False}))
    assert v["B"]["verdict"] == "config_error" and v["A"]["verdict"] == "ok"
    assert first_broken_segment(_evidence(B={"verify_failed": True})) is None


def test_hop_table_and_graph_mark_only_the_broken_segment():
    ev = _evidence(B={"digest_changed": True, "verify_failed": True})
    table = hop_table(ev)
    assert list(table.segment) == list("ABC") and list(table.verdict) == ["ok", "tamper", "ok"]
    dot = hop_chain_dot(ev)
    assert dot.count("TAMPER") == 1 and dot.count("verified") == 2 and "relay1 -> relay2" in dot


def test_describe_feature_gives_plain_language():
    d = describe_feature("Bwd Packet Length Std")
    assert d.startswith("reply-direction") and d.endswith("packet length: variation (std)")
    assert "initial TCP window" in describe_feature("FWD Init Win Bytes")
    assert describe_feature("Down/Up Ratio").startswith("ratio of reply")
    assert describe_feature("Totally Unknown Thing") == "totally unknown thing"


def _bundle():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(600, 3)), columns=["a", "b", "c"])
    y = (X["a"] > 0.5).astype(int)
    model = xgb.XGBClassifier(n_estimators=30, max_depth=3).fit(X, y)
    cal = IsotonicCalibrator().fit(model.predict_proba(X)[:, 1], y)
    return {"model": model, "calibrator": cal, "columns": ["a", "b", "c"]}, X


def test_flow_contributions_name_the_informative_feature():
    bundle, X = _bundle()
    row = X.iloc[0].copy()
    row["a"] = 2.0
    c = flow_contributions(bundle, row, n_features=3)
    assert c.iloc[0].feature == "a" and c.iloc[0].contribution > 0
    assert {"feature", "value", "contribution", "meaning"} <= set(c.columns)


def test_counterfactual_with_no_edits_reproduces_the_score_and_edits_move_it():
    bundle, X = _bundle()
    row = X.iloc[3].copy()
    row["a"] = -2.0
    base = counterfactual_risk(bundle, row)
    assert base["raw"] == pytest.approx(float(bundle["model"].predict_proba(row.to_frame().T)[0, 1]))
    assert counterfactual_risk(bundle, row, {"a": 3.0})["raw"] > base["raw"] + 0.3
    with pytest.raises(KeyError):
        counterfactual_risk(bundle, row, {"nope": 1.0})


def _flows():
    return pd.DataFrame({
        "segment": ["A", "A", "B", "B", "B", "C"],
        "raw_risk": [0.9, 0.2, 0.95, 0.9, 0.1, 0.8], "risk": [0.9, 0.2, 0.95, 0.9, 0.1, 0.8],
        "in_failure_window": [False, False, True, True, False, False],
        "in_any_segment_window": [True, False, True, True, False, True],
        "true_malicious_eval_only": [False, False, True, True, False, False]})


def test_compare_modes_shows_localization_removing_irrelevant_alerts():
    t = compare_modes(_flows(), {"segment": "B"}, 0.5)
    ml, tw, lz = t.iloc[0], t.iloc[1], t.iloc[2]
    assert (ml["alerts"], tw["alerts"], lz["alerts"]) == (4, 4, 2)
    assert (ml["irrelevant alerts (evaluation only)"], lz["irrelevant alerts (evaluation only)"]) == (2, 0)
    assert lz["recall (evaluation only)"] == ml["recall (evaluation only)"] == 1.0


def test_compare_modes_without_integrity_failure_raises_no_correlated_alert():
    t = compare_modes(_flows(), None, 0.5)
    assert list(t["alerts"])[1:] == [0, 0] and t["alerts"][0] > 0


def test_window_quality_and_segment_table():
    q = window_label_quality(_flows())
    assert q["flows_in_windows"] == 2 and q["precision"] == 1.0 and q["recall"] == 1.0
    assert segment_risk_table(_flows(), 0.5).loc["B", "flows_above_threshold"] == 2
