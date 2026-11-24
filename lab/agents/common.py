"""Shared agent plumbing: config from environment, event posting, fault injection."""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from securetrace.integrity import load_node_keys
from securetrace.integrity.chain import KeyConfigError

FAULT_FILE = Path("/tmp/lab_fault")  # "bad_key" | "no_key": injected key/config errors for test sessions
_POOL = ThreadPoolExecutor(max_workers=4)


def event_key(role: str) -> bytes:
    name = f"EVENT_KEY_{role.upper()}"
    try:
        return bytes.fromhex(os.environ[name])
    except (KeyError, ValueError):
        raise KeyConfigError(f"{name} is missing or not hex") from None


def node_keys(role: str) -> dict:
    """Segment keys for this node, with an optional injected fault (never logs key material)."""
    fault = FAULT_FILE.read_text().strip() if FAULT_FILE.exists() else ""
    if fault == "no_key":
        return {"in": None, "out": None}
    keys = load_node_keys(role, os.environ)
    if fault == "bad_key":
        keys = {k: (b"wrong-key-" + b"x" * 24 if v is not None else None) for k, v in keys.items()}
    return keys


def post_event(event_dict: dict) -> None:
    """Fire-and-forget so reporting does not change the transfer's network timing."""
    url = os.environ.get("COLLECTOR_URL", "http://collector:9000") + "/events"

    def send() -> None:
        for attempt in range(5):
            try:
                if requests.post(url, json=event_dict, timeout=3).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.3 * (attempt + 1))
        print(f"WARN: could not deliver event {event_dict['msg_id']}/{event_dict['node']}", file=sys.stderr)

    _POOL.submit(send)


def flush_events() -> None:
    _POOL.shutdown(wait=True)
