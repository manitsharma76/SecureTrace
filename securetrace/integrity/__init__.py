"""Integrity layer: hop-chained HMAC tags, signed events, localization, tamper-evident log."""

from securetrace.integrity.agent import originate, process_hop
from securetrace.integrity.chain import (
    NODE_ORDER,
    SEGMENT_ORDER,
    KeyConfigError,
    extend_chain,
    hop_tag,
    load_node_keys,
    sha256_hex,
    verify_chain_tail,
    verify_tag,
)
from securetrace.integrity.eventlog import HashChainedLog
from securetrace.integrity.events import IntegrityEvent, derive_event_key, sign_event, verify_event_signature
from securetrace.integrity.localize import Localization, localize
