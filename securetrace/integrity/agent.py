"""Per-hop integrity logic, independent of the transport (used by the lab agents and the tests)."""

from __future__ import annotations

from securetrace.integrity.chain import IN_SEGMENT, OUT_SEGMENT, extend_chain, sha256_hex, verify_chain_tail
from securetrace.integrity.events import ERR_KEY, ERR_MALFORMED, ERR_TAG_MISMATCH, IntegrityEvent, sign_event


def originate(
    keys: dict, event_key: bytes, run_id: str, msg_id: str, data: bytes, ts: float
) -> tuple[IntegrityEvent, list[str]]:
    """Sender: hash the data, start the tag chain, report. Returns (event, chain to send)."""
    digest = sha256_hex(data)
    chain = extend_chain(keys["out"], digest, [])
    event = IntegrityEvent(
        run_id=run_id, msg_id=msg_id, node="sender", out_segment=OUT_SEGMENT["sender"],
        digest_out=digest, chain_in=[], chain_out=chain, ts=ts,
    )
    return sign_event(event_key, event), chain


def process_hop(
    role: str,
    keys: dict,
    event_key: bytes,
    run_id: str,
    msg_id: str,
    data: bytes,
    chain_in: list[str] | None,
    ts: float,
) -> tuple[IntegrityEvent, list[str] | None]:
    """Relay or receiver: verify what arrived, then (for relays) tag what will be forwarded.

    A relay always tags and forwards the data it actually received, even when verification failed,
    so one tamper does not cascade into false failures on later segments.
    """
    digest_in = sha256_hex(data)
    verify_ok, error_kind = None, None
    if keys.get("in") is None:
        error_kind = ERR_KEY
    elif not chain_in:
        verify_ok, error_kind = False, ERR_MALFORMED
    else:
        verify_ok = verify_chain_tail(keys["in"], digest_in, chain_in)
        error_kind = None if verify_ok else ERR_TAG_MISMATCH

    chain_out = digest_out = None
    if role in OUT_SEGMENT:
        if keys.get("out") is None:
            error_kind = error_kind or ERR_KEY
        else:
            digest_out = digest_in
            chain_out = extend_chain(keys["out"], digest_in, chain_in or [])

    event = IntegrityEvent(
        run_id=run_id, msg_id=msg_id, node=role,
        in_segment=IN_SEGMENT[role], out_segment=OUT_SEGMENT.get(role),
        digest_in=digest_in, chain_in=list(chain_in or []), verify_ok=verify_ok, error_kind=error_kind,
        digest_out=digest_out, chain_out=chain_out, ts=ts,
    )
    return sign_event(event_key, event), chain_out
