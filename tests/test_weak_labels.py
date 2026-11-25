import datetime
import json

import pandas as pd

from securetrace.data.weak_labels import (
    Bag,
    build_bags,
    build_session_weak_labels,
    build_sessions_weak_labels,
    label_quality,
    tamper_events,
)


def _iso(epoch):
    return datetime.datetime.utcfromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S.%f")


def _manifest(path, run_id, localizations):
    data = {"run_id": run_id, "plan": {"tamper_segment": "B"}, "ground_truth": [],
            "localizations": localizations, "log_integrity": {"ok": True}}
    path.write_text(json.dumps(data), encoding="utf-8")
    return data


def _flow_csv(path, rows, label="NeedManualLabel"):
    out = []
    for ts, duration_us in rows:
        out.append({"Flow ID": "a-b-1-2-6", "Src IP": "1.1.1.1", "Dst IP": "2.2.2.2", "Src Port": 1,
                    "Dst Port": 9100, "Timestamp": ts, "Flow Duration": duration_us,
                    "Total Fwd Packet": 10, "Label": label})
    pd.DataFrame(out).to_csv(path, index=False)


def test_tamper_events_excludes_config_error_and_incomplete():
    manifest = {"localizations": {
        "m0": {"status": "tamper", "segment": "A", "first_ts": 100.0},
        "m1": {"status": "config_error", "segment": None, "first_ts": 200.0},
        "m2": {"status": "incomplete", "segment": None, "first_ts": 300.0},
        "m3": {"status": "ok", "segment": None, "first_ts": 400.0},
    }}
    assert tamper_events(manifest) == [("A", 100.0)]


def test_build_bags_merges_nearby_events_on_the_same_segment():
    manifest = {"run_id": "r1", "localizations": {
        "m0": {"status": "tamper", "segment": "B", "first_ts": 100.0},
        "m1": {"status": "tamper", "segment": "B", "first_ts": 100.5},
        "m2": {"status": "tamper", "segment": "B", "first_ts": 110.0},  # far from the first two
        "m3": {"status": "tamper", "segment": "A", "first_ts": 100.2},  # different segment
    }}
    bags = build_bags(manifest, window_seconds=2.0)
    b_bags = sorted((b for b in bags if b.segment == "B"), key=lambda b: b.window_start)
    assert len(b_bags) == 2  # [100, 100.5] merge; 110.0 is its own bag
    assert any(b.segment == "A" for b in bags)


def test_flows_in_a_positive_bag_get_weak_label_one(tmp_path):
    manifest_path = tmp_path / "run1.json"
    _manifest(manifest_path, "run1", {
        "m0": {"status": "tamper", "segment": "B", "first_ts": 1700000010.0},
    })
    _flow_csv(tmp_path / "run1_segA.pcap_Flow.csv", [(_iso(1700000010.0), 500_000)])
    _flow_csv(tmp_path / "run1_segB.pcap_Flow.csv", [(_iso(1700000009.5), 1_000_000)])

    X, y, meta = build_session_weak_labels(manifest_path, tmp_path, window_seconds=2.0)
    by_segment = dict(zip(meta["segment"], y))
    assert by_segment["A"] is False  # no tamper event on A
    assert by_segment["B"] is True


def test_config_error_produces_no_positive_bag(tmp_path):
    manifest_path = tmp_path / "run2.json"
    _manifest(manifest_path, "run2", {
        "m0": {"status": "config_error", "segment": None, "first_ts": 1700000000.0},
    })
    _flow_csv(tmp_path / "run2_segB.pcap_Flow.csv", [(_iso(1700000000.0), 100_000)])

    X, y, meta = build_session_weak_labels(manifest_path, tmp_path, window_seconds=2.0)
    assert y.tolist() == [False]


def test_flows_far_from_any_bag_stay_unlabelled(tmp_path):
    manifest_path = tmp_path / "run3.json"
    _manifest(manifest_path, "run3", {
        "m0": {"status": "tamper", "segment": "B", "first_ts": 1700000000.0},
    })
    far_ts = _iso(1700001000.0)
    _flow_csv(tmp_path / "run3_segB.pcap_Flow.csv", [(far_ts, 100_000)])

    X, y, meta = build_session_weak_labels(manifest_path, tmp_path, window_seconds=2.0)
    assert y.tolist() == [False]


def test_label_quality_precision_and_recall_against_true_labels(tmp_path):
    manifest_path = tmp_path / "run4.json"
    _manifest(manifest_path, "run4", {
        "m0": {"status": "tamper", "segment": "B", "first_ts": 1700000010.0},
    })
    manifest_data = json.loads(manifest_path.read_text())
    manifest_data["ground_truth"] = [
        {"run_id": "run4", "msg_id": "m0", "segment": "B", "tampered": True, "ts": 1700000010.0},
    ]
    manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

    # One flow truly overlaps the tamper event (true positive), one flow is far away but still
    # falls in a wide bag window (false positive at a large window width).
    _flow_csv(tmp_path / "run4_segB.pcap_Flow.csv", [
        (_iso(1700000009.5), 1_000_000),   # overlaps the event: true malicious AND in the bag
        (_iso(1700000050.0), 100_000),     # far away: never truly malicious
    ])

    narrow = label_quality([manifest_path], tmp_path, window_seconds=2.0)
    assert narrow["bag_precision"] == 1.0
    assert narrow["bag_recall"] == 1.0
    assert narrow["n_positive_bag_flows"] == 1

    wide = label_quality([manifest_path], tmp_path, window_seconds=100.0)
    assert wide["n_positive_bag_flows"] == 2  # both flows now fall in the (very wide) bag
    assert wide["bag_precision"] == 0.5  # only one of the two is actually malicious
    assert wide["bag_recall"] == 1.0


def test_build_sessions_weak_labels_concatenates_sessions(tmp_path):
    p1 = tmp_path / "run5.json"
    _manifest(p1, "run5", {"m0": {"status": "tamper", "segment": "A", "first_ts": 1700000000.0}})
    _flow_csv(tmp_path / "run5_segA.pcap_Flow.csv", [(_iso(1700000000.0), 100_000)])

    p2 = tmp_path / "run6.json"
    _manifest(p2, "run6", {})
    _flow_csv(tmp_path / "run6_segA.pcap_Flow.csv", [(_iso(1700000500.0), 200_000)])

    X, y, meta = build_sessions_weak_labels([p1, p2], tmp_path)
    assert len(X) == 2
    assert set(meta["run_id"]) == {"run5", "run6"}
    assert y.tolist() == [True, False]
