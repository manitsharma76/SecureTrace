import json

import pytest

from securetrace.data.lab_split import split_lab_sessions


def _manifest(path, run_id, mutator):
    path.write_text(json.dumps({
        "run_id": run_id, "plan": {"tamper_segment": "B" if mutator else None, "mutator": mutator},
        "ground_truth": [], "localizations": {}, "log_integrity": {"ok": True},
    }), encoding="utf-8")


def _make_sessions(tmp_path, n_clean, n_flip, n_rewrite):
    paths = []
    for i in range(n_clean):
        p = tmp_path / f"clean{i}.json"
        _manifest(p, f"clean{i}", None)
        paths.append(p)
    for i in range(n_flip):
        p = tmp_path / f"flip{i}.json"
        _manifest(p, f"flip{i}", "flip_byte")
        paths.append(p)
    for i in range(n_rewrite):
        p = tmp_path / f"rewrite{i}.json"
        _manifest(p, f"rewrite{i}", "rewrite_size")
        paths.append(p)
    return paths


def test_held_out_mutator_sessions_are_all_in_test(tmp_path):
    paths = _make_sessions(tmp_path, n_clean=6, n_flip=6, n_rewrite=6)
    split = split_lab_sessions(paths, held_out_mutator="rewrite_size", seed=0)

    rewrite_paths = {p for p in paths if "rewrite" in p.stem}
    assert rewrite_paths <= set(split.test)
    assert not (rewrite_paths & set(split.train))
    assert not (rewrite_paths & set(split.val))


def test_no_session_appears_in_two_splits(tmp_path):
    paths = _make_sessions(tmp_path, n_clean=10, n_flip=10, n_rewrite=10)
    split = split_lab_sessions(paths, held_out_mutator="rewrite_size", seed=1)
    train, val, test = set(split.train), set(split.val), set(split.test)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(paths)


def test_held_out_mutator_is_absent_from_train_and_val(tmp_path):
    paths = _make_sessions(tmp_path, n_clean=8, n_flip=8, n_rewrite=8)
    split = split_lab_sessions(paths, held_out_mutator="flip_byte", seed=2)
    for path in split.train + split.val:
        assert "flip" not in path.stem


def test_non_held_out_classes_appear_in_both_train_and_val(tmp_path):
    paths = _make_sessions(tmp_path, n_clean=10, n_flip=10, n_rewrite=10)
    split = split_lab_sessions(paths, held_out_mutator="rewrite_size", val_fraction=0.2,
                               test_fraction=0.2, seed=3)
    train_classes = {p.stem.rstrip("0123456789") for p in split.train}
    val_classes = {p.stem.rstrip("0123456789") for p in split.val}
    assert "clean" in train_classes and "clean" in val_classes
    assert "flip" in train_classes and "flip" in val_classes


def test_split_is_deterministic_for_a_fixed_seed(tmp_path):
    paths = _make_sessions(tmp_path, n_clean=6, n_flip=6, n_rewrite=6)
    a = split_lab_sessions(paths, held_out_mutator="rewrite_size", seed=7)
    b = split_lab_sessions(paths, held_out_mutator="rewrite_size", seed=7)
    assert set(a.train) == set(b.train)
    assert set(a.val) == set(b.val)
    assert set(a.test) == set(b.test)
