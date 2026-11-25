import datetime
import json

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from securetrace.fusion.calibrate import IsotonicCalibrator
from securetrace.fusion.service import fused_alert, integrity_summary, score_session, session_localization
from securetrace.data.lab import load_session_manifest

T0 = 1_700_000_000.0


def _iso(epoch):
    return datetime.datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S.%f")


def _write_session(tmp_path, tamper):
    loc = {"m0": {"status": "tamper", "segment": "B", "first_ts": T0 + 10, "evidence": {"B": {"digest_changed": True}}}} if tamper else {
        "m0": {"status": "ok", "segment": None, "first_ts": T0 + 10}}
    gt = [{"run_id": "r1", "msg_id": "m0", "segment": "B", "tampered": tamper, "mutator": "flip_byte" if tamper else None, "ts": T0 + 10}]
    manifest = {"run_id": "r1", "plan": {"tamper_segment": "B" if tamper else None, "mutator": None},
                "localizations": loc, "ground_truth": gt, "log_integrity": {"ok": True}}
    path = tmp_path / "r1.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    for seg in ("A", "B"):
        rows = [{"Flow ID": "x", "Src IP": "1.1.1.1", "Dst IP": "2.2.2.2", "Src Port": 5000, "Dst Port": 9100,
                 "Timestamp": _iso(T0 + t), "Flow Duration": 400_000, "Total Fwd Packet": 50 if seg == "B" and t == 10 else 5,
                 "Label": "NeedManualLabel"} for t in (1, 10, 20)]
        pd.DataFrame(rows).to_csv(tmp_path / f"r1_seg{seg}.pcap_Flow.csv", index=False)
    return path


def _bundle():
    X = pd.DataFrame({"Total Fwd Packet": [5.0] * 40 + [50.0] * 10})
    y = np.r_[np.zeros(40), np.ones(10)]
    model = XGBClassifier(n_estimators=10, max_depth=2).fit(X, y)
    cal = IsotonicCalibrator().fit(model.predict_proba(X)[:, 1], y)
    return {"model": model, "calibrator": cal, "columns": ["Total Fwd Packet"], "window_seconds": 2.0}


def test_integrity_summary_counts_message_statuses(tmp_path):
    summary = integrity_summary(load_session_manifest(_write_session(tmp_path, True)))
    assert summary["message_status"]["tamper"] == 1 and summary["n_messages"] == 1


def test_fused_alert_points_at_the_localized_segment_only(tmp_path):
    path, bundle = _write_session(tmp_path, True), _bundle()
    flows, X = score_session(path, tmp_path, bundle)
    manifest = load_session_manifest(path)
    alert = fused_alert(manifest, flows, X, bundle)
    assert alert["localization"]["segment"] == "B"
    assert alert["alerts"] and all(a["segment"] == "B" for a in alert["alerts"])
    assert alert["alerts"][0]["true_malicious_eval_only"] is True      # the 50-packet flow near the tamper
    assert alert["alerts"][0]["top_features"][0][0] == "Total Fwd Packet"


def test_no_integrity_failure_means_no_fused_alert(tmp_path):
    path, bundle = _write_session(tmp_path, False), _bundle()
    flows, X = score_session(path, tmp_path, bundle)
    manifest = load_session_manifest(path)
    assert session_localization(manifest) is None
    alert = fused_alert(manifest, flows, X, bundle)
    assert alert["alerts"] == [] and alert["window_alerts"] == 0
