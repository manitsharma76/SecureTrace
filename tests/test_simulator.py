import pytest

from securetrace.fusion.simulator import MUTATIONS, mutate, run_simulation


def test_clean_message_verifies_everywhere():
    r = run_simulation("hello", None)
    assert r["localization"]["status"] == "ok" and all(h["tag verified"] in (None, True) for h in r["hops"])


@pytest.mark.parametrize("segment", "ABC")
@pytest.mark.parametrize("mutation", MUTATIONS)
def test_tamper_is_localized_to_the_attacked_segment_only(segment, mutation):
    loc = run_simulation("Wire 100 EUR to account 1234", segment, mutation)["localization"]
    assert loc["status"] == "tamper" and loc["segment"] == segment and loc["tampered_segments"] == [segment]


def test_flipping_one_byte_fails_verification_at_the_next_hop_only():
    r = run_simulation("abcdefgh", "B", "flip_byte")
    verified = {h["node"]: h["tag verified"] for h in r["hops"]}
    assert verified["relay2"] is False and verified["receiver"] is True and verified["relay1"] is True


def test_keys_are_fresh_each_run_and_tags_are_shortened():
    a, b = run_simulation("same", "A"), run_simulation("same", "A")
    assert a["hops"][0]["newest tag out"] != b["hops"][0]["newest tag out"]  # different random keys
    assert all(len(h["newest tag out"]) <= 13 for h in a["hops"])  # 10 hex chars + "...", never a full tag


def test_mutations_change_the_data_and_edge_cases_are_handled():
    for m in MUTATIONS:
        assert mutate(b"abcd", m) != b"abcd"
    assert run_simulation("", "A")["localization"]["segment"] == "A"
    assert run_simulation("x", "C", "truncate")["localization"]["segment"] == "C"
    with pytest.raises(ValueError):
        mutate(b"x", "bogus")
    with pytest.raises(ValueError):
        run_simulation("x", "Z")
