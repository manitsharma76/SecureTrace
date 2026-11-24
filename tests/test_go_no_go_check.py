import json

from scripts.go_no_go_check import session_splits


def _manifest(path, run_id, tamper_segment):
    path.write_text(json.dumps({"run_id": run_id, "plan": {"tamper_segment": tamper_segment},
                                 "ground_truth": [], "localizations": {}, "log_integrity": {}}),
                     encoding="utf-8")


def test_session_split_test_fold_always_has_both_classes(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"t{i}.json"
        _manifest(p, f"t{i}", "B")
        paths.append(p)
    for i in range(4):
        p = tmp_path / f"c{i}.json"
        _manifest(p, f"c{i}", None)
        paths.append(p)

    for seed in range(5):
        train, test = session_splits(paths, seed)
        test_segments = [json.loads(p.read_text())["plan"]["tamper_segment"] for p in test]
        assert any(s for s in test_segments), f"seed {seed}: no tampered session in test fold"
        assert any(not s for s in test_segments), f"seed {seed}: no clean session in test fold"
        assert set(train).isdisjoint(set(test))
        assert set(train) | set(test) == set(paths)


def test_session_split_never_splits_within_a_session(tmp_path):
    p1, p2 = tmp_path / "a.json", tmp_path / "b.json"
    _manifest(p1, "a", "A")
    _manifest(p2, "b", None)
    train, test = session_splits([p1, p2], seed=0)
    for session_path in (p1, p2):
        assert (session_path in train) != (session_path in test)
