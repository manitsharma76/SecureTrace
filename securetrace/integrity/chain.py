"""Hop-chained HMAC: tag_i = HMAC(k_i, SHA256(data) || tag_{i-1}).

Path:      sender --A--> relay1 --B--> relay2 --C--> receiver
Segment keys k_A, k_B, k_C are shared by the two nodes at the ends of that segment (and the
collector). Hashing the data first lets the collector re-verify a whole chain from digests alone,
so payloads never leave the lab nodes.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Mapping

NODE_ORDER = ("sender", "relay1", "relay2", "receiver")
SEGMENT_ORDER = ("A", "B", "C")
# Segment a node verifies on receipt / tags on send.
IN_SEGMENT = {"relay1": "A", "relay2": "B", "receiver": "C"}
OUT_SEGMENT = {"sender": "A", "relay1": "B", "relay2": "C"}
# The node that emits onto a segment and the node that verifies it.
SEGMENT_EMITTER = {"A": "sender", "B": "relay1", "C": "relay2"}
SEGMENT_VERIFIER = {"A": "relay1", "B": "relay2", "C": "receiver"}

MIN_KEY_BYTES = 16
_PLACEHOLDERS = {"", "replace-me", "changeme", "change-me"}


class KeyConfigError(RuntimeError):
    """A key is missing, a placeholder, or too short. The message never contains key material."""


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hop_tag(key: bytes, digest_hex: str, prev_tag_hex: str | None) -> str:
    """HMAC-SHA256 over digest || previous tag (empty for the first hop)."""
    message = bytes.fromhex(digest_hex) + bytes.fromhex(prev_tag_hex or "")
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify_tag(key: bytes, digest_hex: str, prev_tag_hex: str | None, tag_hex: str) -> bool:
    try:
        expected = hop_tag(key, digest_hex, prev_tag_hex)
    except ValueError:  # malformed hex from the wire
        return False
    return hmac.compare_digest(expected, tag_hex or "")


def extend_chain(key: bytes, digest_hex: str, chain: list[str] | None) -> list[str]:
    """Append this hop's tag, computed over the digest and the previous tag in the chain."""
    chain = list(chain or [])
    return chain + [hop_tag(key, digest_hex, chain[-1] if chain else "")]


def verify_chain_tail(key: bytes, digest_hex: str, chain: list[str] | None) -> bool:
    """Verify the newest tag; it binds the data digest and the previous tag. An altered earlier
    tag therefore fails here too (the previous tag is part of what was MACed)."""
    if not chain:
        return False
    return verify_tag(key, digest_hex, chain[-2] if len(chain) > 1 else "", chain[-1])


def read_key(environ: Mapping[str, str], name: str) -> bytes:
    value = environ.get(name, "")
    if value.strip().lower() in _PLACEHOLDERS or len(value.encode()) < MIN_KEY_BYTES:
        raise KeyConfigError(f"{name} is missing, a placeholder, or shorter than {MIN_KEY_BYTES} bytes")
    return value.encode()


def segment_key(environ: Mapping[str, str], segment: str) -> bytes:
    return read_key(environ, f"LAB_KEY_{segment}")


def load_node_keys(role: str, environ: Mapping[str, str]) -> dict[str, bytes | None]:
    """Keys a node needs: 'in' (verify the incoming segment) and 'out' (tag the outgoing one).

    A node is only ever given its own two segment keys; the attacker container gets none.
    """
    return {
        "in": segment_key(environ, IN_SEGMENT[role]) if role in IN_SEGMENT else None,
        "out": segment_key(environ, OUT_SEGMENT[role]) if role in OUT_SEGMENT else None,
    }
