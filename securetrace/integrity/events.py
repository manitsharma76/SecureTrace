"""Signed integrity events sent by each node to the collector (digests and tags only, no payload)."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field

_EVENT_DOMAIN = b"securetrace/event/"

# error_kind values. Only "tag_mismatch" can indicate tampering; the others are key/config
# problems and are excluded from weak labels.
ERR_TAG_MISMATCH = "tag_mismatch"
ERR_MALFORMED = "malformed"
ERR_KEY = "key_error"


@dataclass
class IntegrityEvent:
    run_id: str
    msg_id: str
    node: str
    in_segment: str | None = None
    out_segment: str | None = None
    digest_in: str | None = None
    chain_in: list[str] | None = None  # full tag chain as received
    verify_ok: bool | None = None  # None = nothing to verify (sender) or could not verify
    error_kind: str | None = None
    digest_out: str | None = None
    chain_out: list[str] | None = None  # full tag chain as sent on
    ts: float = 0.0
    sig: str = field(default="")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "IntegrityEvent":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def derive_event_key(secret: bytes, node: str) -> bytes:
    """Per-node event-signing key derived from SECURETRACE_SECRET_KEY, so a node holding only its
    own key cannot sign as another node."""
    return hmac.new(secret, _EVENT_DOMAIN + node.encode(), hashlib.sha256).digest()


def _canonical(event: IntegrityEvent) -> bytes:
    body = event.to_dict()
    body.pop("sig", None)
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def sign_event(node_event_key: bytes, event: IntegrityEvent) -> IntegrityEvent:
    """Sign with the node's own derived key (the node never needs the master secret)."""
    event.sig = hmac.new(node_event_key, _canonical(event), hashlib.sha256).hexdigest()
    return event


def verify_event_signature(secret: bytes, event: IntegrityEvent) -> bool:
    """Collector side: re-derive the claimed node's key from the master secret and compare."""
    expected = hmac.new(derive_event_key(secret, event.node), _canonical(event), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, event.sig or "")
