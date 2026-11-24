import json
import random

from scripts.generate_lab_runs import plan_session, read_ground_truth
from securetrace.integrity.chain import SEGMENT_ORDER


def test_plan_session_tamper_segment_is_always_a_real_segment_or_none():
    rng = random.Random(0)
    for _ in range(200):
        plan = plan_session(rng, hold_out_mutator=None)
        assert plan["tamper_segment"] in SEGMENT_ORDER or plan["tamper_segment"] is None
        if plan["tamper_segment"] is None:
            assert plan["mutator"] is None
            assert plan["recon"] is False
        else:
            assert plan["mutator"] is not None


def test_plan_session_never_uses_the_held_out_mutator():
    rng = random.Random(1)
    seen = set()
    for _ in range(300):
        plan = plan_session(rng, hold_out_mutator="flip_byte")
        if plan["mutator"]:
            seen.add(plan["mutator"])
    assert "flip_byte" not in seen
    assert seen  # some other mutator was actually used


def test_plan_session_produces_both_clean_and_tamper_sessions():
    rng = random.Random(2)
    plans = [plan_session(rng, None) for _ in range(100)]
    assert any(p["tamper_segment"] is None for p in plans)
    assert any(p["tamper_segment"] is not None for p in plans)


def test_read_ground_truth_parses_jsonl_and_handles_missing_file(tmp_path):
    assert read_ground_truth(tmp_path, "nope") == []

    path = tmp_path / "run1.jsonl"
    path.write_text(
        json.dumps({"run_id": "run1", "msg_id": "m0", "tampered": False}) + "\n"
        + json.dumps({"run_id": "run1", "msg_id": "m1", "tampered": True}) + "\n",
        encoding="utf-8",
    )
    entries = read_ground_truth(tmp_path, "run1")
    assert len(entries) == 2
    assert entries[1]["tampered"] is True
