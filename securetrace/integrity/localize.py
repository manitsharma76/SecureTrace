"""Attribute an integrity failure to a path segment (or to a key/config error).

For each segment the emitter's claimed output is compared with the verifier's observed input:

* digest or tag differs  -> something changed the message *on that segment*: TAMPER there.
* handoff is identical but verification failed, or the emitter's tag is not what the collector
  computes from its own key -> the keys or configuration disagree: CONFIG ERROR, not a tamper.

Relays tag what they actually received and forward, so a tamper on segment k does not make later
segments fail and is never attributed to another segment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

from securetrace.integrity.chain import (
    NODE_ORDER,
    SEGMENT_EMITTER,
    SEGMENT_ORDER,
    SEGMENT_VERIFIER,
    extend_chain,
)


@dataclass
class Localization:
    status: str  # "ok" | "tamper" | "config_error" | "incomplete"
    segment: str | None = None  # first tampered segment (status == "tamper")
    tampered_segments: list[str] = field(default_factory=list)
    config_segments: list[str] = field(default_factory=list)
    missing_nodes: list[str] = field(default_factory=list)
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def localize(events: list[dict], segment_keys: Mapping[str, bytes]) -> Localization:
    """`events` are the (already signature-checked) events for ONE message.

    `segment_keys` maps segment -> key and is held by the collector only.
    """
    by_node = {e["node"]: e for e in events}
    missing = [n for n in NODE_ORDER if n not in by_node]
    if missing:
        return Localization("incomplete", missing_nodes=missing)

    tampered, config, evidence = [], [], {}
    for segment in SEGMENT_ORDER:
        emitter, verifier = by_node[SEGMENT_EMITTER[segment]], by_node[SEGMENT_VERIFIER[segment]]
        digest_changed = emitter["digest_out"] != verifier["digest_in"]
        tag_changed = emitter["chain_out"] != verifier["chain_in"]
        verify_failed = verifier["verify_ok"] is not True
        try:
            expected = extend_chain(segment_keys[segment], emitter["digest_out"], emitter["chain_in"])
            emitter_consistent = expected == emitter["chain_out"]
        except (ValueError, KeyError, TypeError):
            emitter_consistent = False
        evidence[segment] = {
            "digest_changed": digest_changed,
            "tag_changed": tag_changed,
            "verify_failed": verify_failed,
            "emitter_tag_matches_collector_key": emitter_consistent,
        }
        if digest_changed or tag_changed:
            tampered.append(segment)
        elif verify_failed or not emitter_consistent:
            config.append(segment)

    if tampered:
        return Localization("tamper", tampered[0], tampered, config, [], evidence)
    if config:
        return Localization("config_error", None, [], config, [], evidence)
    return Localization("ok", evidence=evidence)
