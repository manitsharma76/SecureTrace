from pathlib import Path

from lab.sensors.extract import SEGMENT_RE, find_segment_pcaps


def test_segment_pcap_name_pattern_matches_expected_names():
    for name, expected in [
        ("run001_segA.pcap", ("run001", "A")),
        ("tamperB2_segB.pcapng", ("tamperB2", "B")),
        ("my-run-7_segC.pcap", ("my-run-7", "C")),
    ]:
        match = SEGMENT_RE.match(name)
        assert match is not None
        assert (match["run_id"], match["segment"]) == expected


def test_segment_pcap_name_pattern_rejects_unrelated_files():
    for name in ("readme.txt", "segA.pcap", "run001_segD.pcap", "run001.pcap", "run001_seg.pcap"):
        assert SEGMENT_RE.match(name) is None


def test_find_segment_pcaps_filters_by_run_id_and_sorts(tmp_path: Path):
    for name in ["r2_segA.pcap", "r1_segB.pcap", "r1_segA.pcap", "notes.txt", "r1_segC.pcapng"]:
        (tmp_path / name).touch()

    all_found = find_segment_pcaps(tmp_path, None)
    assert {(r, s) for r, s, _ in all_found} == {("r1", "A"), ("r1", "B"), ("r1", "C"), ("r2", "A")}

    only_r1 = find_segment_pcaps(tmp_path, "r1")
    assert {(r, s) for r, s, _ in only_r1} == {("r1", "A"), ("r1", "B"), ("r1", "C")}
    assert all(r == "r1" for r, _, _ in only_r1)
