import pandas as pd

from securetrace.data.public import split_by_day
from securetrace.features.schema import drop_identity_features, label_to_binary, prepare_features
from securetrace.data.public import load_and_clean_public_data
import tempfile
from pathlib import Path


def test_drop_identity_features_removes_ip_and_time_columns():
    frame = pd.DataFrame(
        {
            "Flow ID": [1, 2],
            "Source IP": ["10.0.0.1", "10.0.0.2"],
            "Timestamp": ["2024-01-01", "2024-01-02"],
            "Total Fwd Packets": [10, 20],
            "Label": ["BENIGN", "ATTACK"],
        }
    )

    trimmed = drop_identity_features(frame)
    assert "Source IP" not in trimmed.columns
    assert "Timestamp" not in trimmed.columns
    assert "Flow ID" not in trimmed.columns
    assert "Total Fwd Packets" in trimmed.columns
    assert "Label" in trimmed.columns


def test_prepare_features_binary_labels_and_cleaning():
    frame = pd.DataFrame(
        {
            "Flow ID": [1, 2],
            "Source IP": ["10.0.0.1", "10.0.0.2"],
            "Total Fwd Packets": [10, 20],
            "Label": ["BENIGN", "ATTACK"],
        }
    )

    X, y = prepare_features(frame, label_col="Label")
    assert list(X.columns) == ["Total Fwd Packets"]
    assert y.tolist() == ["BENIGN", "ATTACK"]
    assert label_to_binary(y).tolist() == [0, 1]


def test_split_by_day_has_no_overlap_between_sets():
    features = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [10, 20, 30, 40, 50, 60],
        },
        index=[0, 1, 2, 3, 4, 5],
    )
    labels = pd.Series([0, 1, 1, 0, 1, 0], index=[0, 1, 2, 3, 4, 5])
    days = pd.Series([1, 1, 2, 2, 3, 3], index=[0, 1, 2, 3, 4, 5])

    train_X, val_X, test_X, train_y, val_y, test_y = split_by_day(
        features,
        labels,
        days,
        train_days={1},
        val_days={2},
        test_days={3},
    )

    assert set(train_X.index) == {0, 1}
    assert set(val_X.index) == {2, 3}
    assert set(test_X.index) == {4, 5}
    assert set(train_y.index).isdisjoint(set(val_y.index))
    assert set(val_y.index).isdisjoint(set(test_y.index))


def test_public_loader_aligns_shared_features_across_datasets():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first.csv"
        second = root / "second.csv"
        pd.DataFrame(
            {
                "Src IP": ["10.0.0.1", "10.0.0.2"],
                "Flow Duration": [100, 200],
                "Shared Feature": [1, 2],
                "Label": ["BENIGN", "ATTACK"],
            }
        ).to_csv(first, index=False)
        pd.DataFrame(
            {
                "Dst IP": ["10.0.1.1", "10.0.1.2"],
                "Flow Duration": [300, 400],
                "Shared Feature": [3, 4],
                "Label": ["BENIGN", "ATTACK"],
            }
        ).to_csv(second, index=False)

        X, y, days = load_and_clean_public_data([first, second])

        assert list(X.columns) == ["Flow Duration", "Shared Feature"]
        assert y.tolist() == [0, 1, 0, 1]
        assert len(days) == 4


# ---- fixes found while starting the Phase 1 pipeline ----------------------------------------

import numpy as np
import pytest

from securetrace.data.public import load_public_frame
from securetrace.eval.metrics import evaluate_scores, fpr_at_recall, select_threshold
from securetrace.features.schema import IDS2018_TO_CANONICAL, harmonize_columns


def _write(path, columns, rows):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def test_2018_names_map_onto_2017_names_and_features_are_shared(tmp_path):
    f17, f18 = tmp_path / "a2017.csv", tmp_path / "b2018.csv"
    _write(f17, ["Src IP", "Timestamp", "Total Fwd Packet", "Flow Duration", "Label"],
           [["1.1.1.1", "2017-07-03 10:00:00.5", 5, 10, "BENIGN"], ["1.1.1.2", "2017-07-04 10:00:00.5", 7, 11, "DoS Hulk"]])
    _write(f18, ["Dst Port", "Timestamp", "Tot Fwd Pkts", "Flow Duration", "Label"],
           [[80, "14/02/2018 08:31:01", 3, 12, "Benign"], [80, "15/02/2018 08:31:01", 4, 13, "FTP-BruteForce"]])
    X, y, meta = load_public_frame([f17, f18])
    assert sorted(X.columns) == ["Flow Duration", "Total Fwd Packet"]
    assert meta["dataset"].tolist() == ["ids2017", "ids2017", "ids2018", "ids2018"]
    # dd/mm/YYYY 2018 timestamps must parse (day > 12 used to become NaT)
    assert meta["day"].tolist() == ["2017-07-03", "2017-07-04", "2018-02-14", "2018-02-15"]
    assert y.tolist() == [0, 1, 0, 1]


def test_loader_drops_bad_rows_attempted_labels_and_duplicates(tmp_path):
    path = tmp_path / "x2018.csv"
    _write(path, ["Tot Fwd Pkts", "Timestamp", "Label"], [
        [1, "14/02/2018 01:00:00", "Benign"],
        [1, "14/02/2018 01:00:00", "Benign"],          # duplicate
        [2, "14/02/2018 01:00:00", "Label"],           # stray repeated header row
        [3, "14/02/2018 01:00:00", None],              # missing label
        [4, "01/01/1970 00:00:01", "Benign"],          # garbage timestamp
        [5, "14/02/2018 01:00:00", "Web Attack - XSS - Attempted"],
        [6, "14/02/2018 01:00:00", "FTP-BruteForce"],
    ])
    X, y, meta = load_public_frame([path])
    assert X["Total Fwd Packet"].tolist() == [1.0, 6.0]
    assert y.tolist() == [0, 1]


def test_no_identity_or_time_columns_reach_the_feature_matrix(tmp_path):
    path = tmp_path / "f.csv"
    _write(path, ["Flow ID", "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol", "Timestamp",
                  "Flow Duration", "Idle Mean", "Total Connection Flow Time", "Label"],
           [["a", "1.1.1.1", 1, "2.2.2.2", 80, 6, "2017-07-03 10:00:00.5", 5, 6, 7, "BENIGN"]])
    X, _, _ = load_public_frame([path])
    assert set(X.columns) == {"Flow Duration", "Idle Mean"}


def test_label_to_binary_rejects_missing_and_maps_attack_families():
    assert label_to_binary(pd.Series(["Benign", "DoS attacks-GoldenEye", "Infiltration - Portscan"])).tolist() == [0, 1, 1]
    with pytest.raises(ValueError):
        label_to_binary(pd.Series(["Benign", None]))


def test_split_by_day_rejects_a_day_in_two_splits():
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    days = pd.Series(["d1", "d2", "d3"])
    with pytest.raises(ValueError):
        split_by_day(X, y, days, train_days=["d1", "d2"], val_days=["d2"], test_days=["d3"])


def test_split_by_day_accepts_dates_and_strings_equally():
    import datetime

    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 0])
    days = pd.Series(["2017-07-03", "2017-07-04", "2017-07-05"])
    tr, va, te, *_ = split_by_day(X, y, days, train_days=[datetime.date(2017, 7, 3)],
                                  val_days=["2017-07-04"], test_days=["2017-07-05"])
    assert (len(tr), len(va), len(te)) == (1, 1, 1)


def test_harmonize_map_targets_are_unique():
    assert len(set(IDS2018_TO_CANONICAL.values())) == len(IDS2018_TO_CANONICAL)
    assert harmonize_columns(pd.DataFrame({"Tot Fwd Pkts": [1]})).columns.tolist() == ["Total Fwd Packet"]


def test_metrics_perfect_and_single_class():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    t = select_threshold(y, s)
    r = evaluate_scores(y, s, threshold=t)
    assert r["macro_f1"] == 1.0 and r["pr_auc"] == 1.0 and r["fpr_at_recall_95"] == 0.0
    assert fpr_at_recall(y, np.array([0.9, 0.8, 0.2, 0.1]), 0.9) == 1.0   # inverted scorer is useless
    only_benign = evaluate_scores(np.zeros(4), s, threshold=0.5)
    assert np.isnan(only_benign["pr_auc"])


def test_insdn_is_detected_shares_all_columns_and_dedups_per_file(tmp_path):
    cols = ["Src IP", "Timestamp", "Flow Duration", "Total Connection Flow Time", "Label"]
    f17, sdn = tmp_path / "cic_ids_2017_corrected.csv", tmp_path / "insdn.csv"
    _write(f17, cols, [["a", "2017-07-03 10:00:00.5", 5, 1, "BENIGN"]])
    _write(sdn, cols, [
        ["b", "2020-02-04 10:00:00.1", 9, 1, "DDoS"],
        ["c", "2020-02-04 10:00:00.2", 9, 2, "DDoS"],   # differs only in a time-like column -> duplicate
        ["d", "2020-02-04 10:00:00.3", 9, 3, "Normal"],
    ])
    X, y, meta = load_public_frame([f17, sdn])
    assert list(X.columns) == ["Flow Duration"]
    assert meta["dataset"].tolist() == ["ids2017", "insdn", "insdn"]
    assert y.tolist() == [0, 1, 0]
    assert meta["day"].tolist() == ["2017-07-03", "2020-02-04", "2020-02-04"]
