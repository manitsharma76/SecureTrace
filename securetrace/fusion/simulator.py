"""In-process tamper simulator for the dashboard's offline demo mode.

Pushes one message sender -> relay1 -> relay2 -> receiver through the REAL integrity code
(securetrace.integrity: hop-chained HMAC, signed events, localization), with an attacker that
mutates the data on one chosen segment. No network, no Docker. Keys are generated fresh from the OS
random source on every call and never returned or logged, so there is nothing secret to display.
"""

from __future__ import annotations

import os
import time

from securetrace.integrity.agent import originate, process_hop
from securetrace.integrity.chain import NODE_ORDER, SEGMENT_ORDER, load_node_keys
from securetrace.integrity.events import derive_event_key
from securetrace.integrity.localize import localize

MUTATIONS = ("flip_byte", "truncate", "pad")


def mutate(data: bytes, mutation: str) -> bytes:
    if mutation == "flip_byte":
        i = len(data) // 2
        return data[:i] + bytes([data[i] ^ 0x01]) + data[i + 1:]
    if mutation == "truncate":
        return data[: len(data) - max(1, len(data) // 4)]  # always shorter, possibly empty
    if mutation == "pad":
        return data + b"\x00" * 16
    raise ValueError(f"unknown mutation {mutation!r}; choose from {MUTATIONS}")


def _short(value: str | None) -> str:
    return (value[:10] + "...") if value else "-"


def run_simulation(message: str | bytes, tamper_segment: str | None, mutation: str = "flip_byte") -> dict:
    """Returns {hops, wire, localization, steps}. `hops` is one row per node (digests/tags shortened),
    `wire` one row per segment saying whether the attacker changed the data there."""
    data = message.encode("utf-8") if isinstance(message, str) else bytes(message)
    data = data or b" "
    if tamper_segment is not None and tamper_segment not in SEGMENT_ORDER:
        raise ValueError(f"tamper_segment must be one of {SEGMENT_ORDER} or None")

    env = {f"LAB_KEY_{s}": os.urandom(16).hex() for s in SEGMENT_ORDER}  # 32 chars, fresh each run
    secret = os.urandom(32)
    segment_keys = {s: env[f"LAB_KEY_{s}"].encode() for s in SEGMENT_ORDER}

    events, wire, steps, chain = [], [], [], None
    now = time.time()
    for role in NODE_ORDER:
        keys = load_node_keys(role, env)
        event_key = derive_event_key(secret, role)
        if role == "sender":
            ev, chain = originate(keys, event_key, "sim", "m0", data, now)
            steps.append(("sender", f"Hashes the message (SHA-256 {_short(ev.digest_out)}) and starts the tag chain "
                                    f"with its segment-A key: tag = HMAC(k_A, digest)."))
        else:
            ev, chain = process_hop(role, keys, event_key, "sim", "m0", data, chain, now)
            verdict = "verifies" if ev.verify_ok else "FAILS verification"
            steps.append((role, f"Recomputes the digest of what arrived ({_short(ev.digest_in)}) and the newest tag "
                                f"{verdict} against segment {ev.in_segment}'s key."
                                + (" It still tags and forwards the data it actually received." if ev.out_segment else "")))
        events.append(ev.to_dict())
        if ev.out_segment:
            changed = tamper_segment == ev.out_segment
            before = data
            if changed:
                data = mutate(data, mutation)
            wire.append({"segment": ev.out_segment, "attacker_changed_data": changed,
                         "digest_before": _short(ev.digest_out), "digest_after": _short(_digest(data))})
            if changed:
                steps.append((f"wire {ev.out_segment}", f"On-path attacker applies '{mutation}': digest "
                                                         f"{_short(ev.digest_out)} -> {_short(_digest(data))}."))
    loc = localize(events, segment_keys)
    steps.append(("collector", f"Compares each emitter's output with its verifier's input: status = "
                               f"{loc.status}" + (f", tampered segment = {loc.segment}." if loc.segment else ".")))
    hops = [{"node": e["node"], "digest in": _short(e.get("digest_in")), "digest out": _short(e.get("digest_out")),
             "newest tag in": _short((e.get("chain_in") or [None])[-1]),
             "newest tag out": _short((e.get("chain_out") or [None])[-1]),
             "tag verified": e.get("verify_ok")} for e in events]
    return {"hops": hops, "wire": wire, "localization": loc.to_dict(), "steps": steps}


def _digest(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()
