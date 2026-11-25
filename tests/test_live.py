import json

from securetrace.fusion import live
from securetrace.fusion.sessions import catalog, discover


def test_parse_step_and_command_building():
    assert live.parse_step("STEP: Sender sends 12 messages\n") == "Sender sends 12 messages"
    assert live.parse_step("  ground truth: 3 tampered") is None
    cmd = live.build_command("B", "flip_byte", 5)
    assert cmd[cmd.index("--force-tamper") + 1] == "B" and cmd[cmd.index("--force-mutator") + 1] == "flip_byte"
    clean = live.build_command(None, "flip_byte", 5)
    assert "--force-clean" in clean and "--force-tamper" not in clean


def test_archive_moves_a_session_out_of_the_experiment_dataset(tmp_path):
    lab = tmp_path / "data" / "raw" / "lab"
    for sub in ("sessions", "attacker", "pcap", "pcap/extracted"):
        (lab / sub).mkdir(parents=True)
    rid = "gen99-0000-abc123"
    (lab / "sessions" / f"{rid}.json").write_text("{}")
    (lab / "sessions" / "keep-0001.json").write_text("{}")
    (lab / "attacker" / f"{rid}.jsonl").write_text("")
    (lab / "pcap" / f"{rid}_segA_0.pcap").write_text("")
    (lab / "pcap" / "extracted" / f"{rid}_segA.pcap_Flow.csv").write_text("")
    (lab / "runs_index.yaml").write_text(f"- run_id: {rid}\n- run_id: keep-0001\n")
    out = live.archive_session(rid, tmp_path)
    demo = tmp_path / "data" / "raw" / "lab_demo"
    assert out == demo / "sessions" / f"{rid}.json" and out.exists()
    assert (demo / "extracted" / f"{rid}_segA.pcap_Flow.csv").exists() and (demo / "attacker" / f"{rid}.jsonl").exists()
    assert not list((lab / "sessions").glob(f"*{rid}*")) and (lab / "sessions" / "keep-0001.json").exists()
    index = (lab / "runs_index.yaml").read_text()
    assert rid not in index and "keep-0001" in index


def test_discover_groups_sessions_and_describes_integrity_verdicts(tmp_path):
    sessions, demo = tmp_path / "s", tmp_path / "demo" / "sessions"
    sessions.mkdir()
    demo.mkdir(parents=True)

    def write(path, run_id, tamper):
        loc = {"m0": {"status": "tamper", "segment": "B", "first_ts": 1.0}} if tamper else {"m0": {"status": "ok", "segment": None}}
        path.write_text(json.dumps({"run_id": run_id, "plan": {"tamper_segment": "B" if tamper else None,
                                                               "mutator": "flip_byte" if tamper else None},
                                    "localizations": loc, "ground_truth": [], "log_integrity": {"ok": True}}))

    write(sessions / "t1.json", "t1", True)
    write(sessions / "tr1.json", "tr1", False)
    write(demo / "d1.json", "d1", True)
    refs = discover(sessions, tmp_path / "ex", tmp_path / "demo", ["t1"])
    assert {r.run_id: r.group for r in refs} == {"t1": "held-out test", "tr1": "trained on", "d1": "live demo"}
    cat = catalog(refs).set_index("run_id")
    assert cat.loc["t1", "integrity_verdict"] == "tamper detected" and cat.loc["t1", "integrity_segment"] == "B"
    assert cat.loc["tr1", "integrity_verdict"] == "no tamper detected"
