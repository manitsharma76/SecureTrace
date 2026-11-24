import json

import pandas as pd

from securetrace.data.lab import load_session_true_labels, load_sessions_true_labels


def _write_manifest(path, run_id, ground_truth):
    path.write_text(json.dumps({
        "run_id": run_id, "plan": {"tamper_segment": "B"},
        "n_messages_acked": len(ground_truth), "localizations": {}, "ground_truth": ground_truth,
        "log_integrity": {"ok": True},
    }), encoding="utf-8")


def _write_flow_csv(path, timestamps_and_durations, label="NeedManualLabel"):
    rows = []
    for ts, duration_us in timestamps_and_durations:
        rows.append({
            "Flow ID": "a-b-1-2-6", "Src IP": "1.1.1.1", "Dst IP": "2.2.2.2", "Src Port": 1,
            "Dst Port": 2, "Timestamp": ts, "Flow Duration": duration_us,
            "Total Fwd Packet": 10, "Label": label,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_flows_overlapping_a_tampered_window_are_labeled_true(tmp_path):
    manifest_path = tmp_path / "run1.json"
    _write_manifest(manifest_path, "run1", [
        {"run_id": "run1", "msg_id": "m0", "segment": "A", "tampered": False, "mutator": None,
         "ts": 1700000000.0},
        {"run_id": "run1", "msg_id": "m0", "segment": "B", "tampered": True, "mutator": "flip_byte",
         "ts": 1700000010.0},
    ])
    import datetime

    def iso(epoch):
        return datetime.datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S.%f")

    _write_flow_csv(tmp_path / "run1_segA.pcap_Flow.csv", [(iso(1700000000.0), 500_000)])  # near event A, not B
    _write_flow_csv(tmp_path / "run1_segB.pcap_Flow.csv", [(iso(1700000009.5), 1_000_000)])  # overlaps event B

    X, y, meta = load_session_true_labels(manifest_path, tmp_path)
    assert len(y) == 2
    by_segment = dict(zip(meta["segment"], y))
    assert by_segment["A"] is False
    assert by_segment["B"] is True


def test_flows_far_from_any_tamper_event_are_labeled_false(tmp_path):
    manifest_path = tmp_path / "run2.json"
    _write_manifest(manifest_path, "run2", [
        {"run_id": "run2", "msg_id": "m0", "segment": "B", "tampered": True, "mutator": "flip_byte",
         "ts": 1700000000.0},
    ])
    import datetime

    far_ts = datetime.datetime.utcfromtimestamp(1700000500.0).strftime("%Y-%m-%d %H:%M:%S.%f")
    _write_flow_csv(tmp_path / "run2_segB.pcap_Flow.csv", [(far_ts, 100_000)])

    X, y, meta = load_session_true_labels(manifest_path, tmp_path)
    assert y.tolist() == [False]


def test_identity_and_label_columns_are_dropped_from_features(tmp_path):
    manifest_path = tmp_path / "run3.json"
    _write_manifest(manifest_path, "run3", [])
    import datetime

    ts = datetime.datetime.utcfromtimestamp(1700000000.0).strftime("%Y-%m-%d %H:%M:%S.%f")
    _write_flow_csv(tmp_path / "run3_segA.pcap_Flow.csv", [(ts, 100_000)])

    X, y, meta = load_session_true_labels(manifest_path, tmp_path)
    assert "Src IP" not in X.columns
    assert "Dst Port" not in X.columns
    assert "Label" not in X.columns
    assert "Total Fwd Packet" in X.columns
    assert meta["label"].iloc[0] == "NeedManualLabel"


def test_empty_flow_csv_is_skipped_without_raising(tmp_path):
    manifest_path = tmp_path / "run4.json"
    _write_manifest(manifest_path, "run4", [])
    (tmp_path / "run4_segA.pcap_Flow.csv").write_text(
        "Flow ID,Src IP,Dst IP,Src Port,Dst Port,Timestamp,Flow Duration,Total Fwd Packet,Label\n",
        encoding="utf-8",
    )
    import datetime

    ts = datetime.datetime.utcfromtimestamp(1700000000.0).strftime("%Y-%m-%d %H:%M:%S.%f")
    _write_flow_csv(tmp_path / "run4_segB.pcap_Flow.csv", [(ts, 100_000)])

    X, y, meta = load_session_true_labels(manifest_path, tmp_path)
    assert len(y) == 1  # only segB's row; segA's empty CSV contributed nothing


def test_load_sessions_true_labels_concatenates_and_keeps_shared_columns(tmp_path):
    import datetime

    ts = datetime.datetime.utcfromtimestamp(1700000000.0).strftime("%Y-%m-%d %H:%M:%S.%f")

    m1 = tmp_path / "run5.json"
    _write_manifest(m1, "run5", [])
    _write_flow_csv(tmp_path / "run5_segA.pcap_Flow.csv", [(ts, 100_000)])

    m2 = tmp_path / "run6.json"
    _write_manifest(m2, "run6", [])
    _write_flow_csv(tmp_path / "run6_segA.pcap_Flow.csv", [(ts, 200_000)])

    X, y, meta = load_sessions_true_labels([m1, m2], tmp_path)
    assert len(X) == 2
    assert set(meta["run_id"]) == {"run5", "run6"}
