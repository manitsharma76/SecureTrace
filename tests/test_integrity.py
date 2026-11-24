import sqlite3

import pytest

from securetrace.integrity import (
    HashChainedLog,
    KeyConfigError,
    derive_event_key,
    extend_chain,
    load_node_keys,
    localize,
    originate,
    process_hop,
    sha256_hex,
    verify_chain_tail,
    verify_event_signature,
)
from securetrace.integrity.chain import NODE_ORDER, SEGMENT_ORDER
from securetrace.integrity.events import IntegrityEvent

SECRET = b"s" * 32
ENV = {"LAB_KEY_A": "a" * 32, "LAB_KEY_B": "b" * 32, "LAB_KEY_C": "c" * 32}
SEG_KEYS = {"A": ENV["LAB_KEY_A"].encode(), "B": ENV["LAB_KEY_B"].encode(), "C": ENV["LAB_KEY_C"].encode()}
DATA = bytes(range(256)) * 40


def run_path(data=DATA, tamper=None, node_env=None):
    """Push one message sender -> receiver. `tamper` = (segment, fn(data, chain) -> (data, chain)).
    `node_env` overrides a node's environment (to simulate key/config errors)."""
    node_env = node_env or {}
    events, chain = [], None
    for role in NODE_ORDER:
        keys = load_node_keys(role, node_env.get(role, ENV))
        ek = derive_event_key(SECRET, role)
        if role == "sender":
            ev, chain = originate(keys, ek, "run1", "m1", data, 1.0)
        else:
            ev, chain = process_hop(role, keys, ek, "run1", "m1", data, chain, 1.0)
        events.append(ev.to_dict())
        # The attacker sits on the wire after this node's output, i.e. on its outgoing segment.
        if tamper and ev.out_segment == tamper[0]:
            data, chain = tamper[1](data, chain)
    return events


def flip_byte(data, chain):
    return data[:100] + bytes([data[100] ^ 1]) + data[101:], chain


def append_bytes(data, chain):
    return data + b"extra", chain


def truncate(data, chain):
    return data[:-10], chain


def swap_tag(data, chain):
    return data, chain[:-1] + ["00" * 32]


def alter_older_tag(data, chain):
    return data, ["11" * 32] + chain[1:] if len(chain) > 1 else (data, chain)


def strip_chain(data, chain):
    return data, []


def test_one_byte_payload_flip_fails_verification():
    chain = extend_chain(SEG_KEYS["A"], sha256_hex(DATA), [])
    assert verify_chain_tail(SEG_KEYS["A"], sha256_hex(DATA), chain)
    flipped = DATA[:7] + bytes([DATA[7] ^ 0x01]) + DATA[8:]
    assert not verify_chain_tail(SEG_KEYS["A"], sha256_hex(flipped), chain)


def test_clean_transfer_is_ok():
    assert localize(run_path(), SEG_KEYS).status == "ok"


@pytest.mark.parametrize("segment", SEGMENT_ORDER)
@pytest.mark.parametrize("attack", [flip_byte, append_bytes, truncate, swap_tag, strip_chain])
def test_tamper_on_segment_k_is_attributed_to_k_only(segment, attack):
    result = localize(run_path(tamper=(segment, attack)), SEG_KEYS)
    assert result.status == "tamper"
    assert result.segment == segment
    assert result.tampered_segments == [segment]  # never another segment


@pytest.mark.parametrize("segment", ["B", "C"])
def test_altering_an_older_tag_is_a_tamper_not_a_config_error(segment):
    result = localize(run_path(tamper=(segment, alter_older_tag)), SEG_KEYS)
    assert (result.status, result.segment) == ("tamper", segment)


def test_wrong_key_on_a_node_is_config_error_not_tamper():
    bad = {**ENV, "LAB_KEY_B": "z" * 32}
    result = localize(run_path(node_env={"relay1": bad}), SEG_KEYS)  # relay1 tags segment B with a wrong key
    assert result.status == "config_error"
    assert result.tampered_segments == []
    assert result.config_segments == ["B"]


def test_missing_key_is_config_error_not_tamper():
    no_key = {k: v for k, v in ENV.items() if k != "LAB_KEY_A"}
    # relay1 cannot load its key at all -> it is built with keys None, reports key_error
    events = run_path()
    for e in events:
        if e["node"] == "relay1":
            e["verify_ok"], e["error_kind"] = None, "key_error"
    assert localize(events, SEG_KEYS).status == "config_error"
    with pytest.raises(KeyConfigError):
        load_node_keys("relay1", no_key)


def test_missing_events_are_reported_incomplete():
    events = [e for e in run_path() if e["node"] != "relay2"]
    result = localize(events, SEG_KEYS)
    assert result.status == "incomplete" and result.missing_nodes == ["relay2"]


def test_placeholder_and_short_keys_are_rejected_without_leaking_them():
    for bad in ("replace-me", "", "tinysecret-q7"):
        with pytest.raises(KeyConfigError) as err:
            load_node_keys("relay1", {**ENV, "LAB_KEY_A": bad})
        assert "LAB_KEY_A" in str(err.value)
        assert not bad or bad == "replace-me" or bad not in str(err.value)


def test_event_signature_detects_edit_and_cross_node_forgery():
    ev = run_path()[1]
    event = IntegrityEvent.from_dict(ev)
    assert verify_event_signature(SECRET, event)
    event.verify_ok = True if not event.verify_ok else event.verify_ok
    event.digest_in = "0" * 64
    assert not verify_event_signature(SECRET, event)
    # relay1's key cannot produce a valid signature for relay2
    forged = IntegrityEvent.from_dict(run_path()[2])
    forged.node = "relay1"
    assert not verify_event_signature(SECRET, forged)


def _log_with_rows(n=5, path=":memory:"):
    log = HashChainedLog(path)
    for i in range(n):
        log.append({"run_id": "r", "msg_id": f"m{i}", "value": i})
    return log


def test_log_verifies_when_untouched():
    log = _log_with_rows()
    assert log.verify()[0]
    assert log.verify(expected_head=log.head())[0]


def test_log_detects_a_modified_row(tmp_path):
    path = str(tmp_path / "log.db")
    log = _log_with_rows(path=path)
    db = sqlite3.connect(path)
    db.execute("UPDATE log SET payload = replace(payload, '\"value\":2', '\"value\":99') WHERE seq = 3")
    db.commit()
    ok, bad_seq, _ = log.verify()
    assert not ok and bad_seq == 3


def test_log_detects_deleted_inserted_and_reordered_rows(tmp_path):
    path = str(tmp_path / "log.db")
    log = _log_with_rows(path=path)
    db = sqlite3.connect(path)
    db.execute("DELETE FROM log WHERE seq = 2")
    db.commit()
    assert not log.verify()[0]

    path2 = str(tmp_path / "log2.db")
    log2 = _log_with_rows(path=path2)
    db2 = sqlite3.connect(path2)
    db2.execute("UPDATE log SET seq = 100 WHERE seq = 2")
    db2.execute("UPDATE log SET seq = 2 WHERE seq = 3")
    db2.execute("UPDATE log SET seq = 3 WHERE seq = 100")
    db2.commit()
    assert not log2.verify()[0]


def test_log_tail_truncation_needs_and_is_caught_by_anchored_head(tmp_path):
    path = str(tmp_path / "log.db")
    log = _log_with_rows(path=path)
    anchor = log.head()
    db = sqlite3.connect(path)
    db.execute("DELETE FROM log WHERE seq = 5")
    db.commit()
    assert log.verify()[0]  # internally still consistent...
    assert not log.verify(expected_head=anchor)[0]  # ...but the anchored head exposes it


def test_log_detects_index_column_edit(tmp_path):
    path = str(tmp_path / "log.db")
    log = _log_with_rows(path=path)
    db = sqlite3.connect(path)
    db.execute("UPDATE log SET msg_id = 'forged' WHERE seq = 4")
    db.commit()
    assert not log.verify()[0]
