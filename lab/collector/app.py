"""Collector: ingests signed integrity events into a hash-chained SQLite log and localizes failures.

Run:  uvicorn --factory lab.collector.app:app_from_env --host 0.0.0.0 --port 9000
Env:  SECURETRACE_SECRET_KEY (verifies event signatures), LAB_KEY_A/B/C (recomputes tag chains),
      COLLECTOR_DB (SQLite path, default /data/collector.db).
Only digests, tags and timestamps are stored. Payloads never reach the collector.
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import Body, FastAPI, HTTPException

from securetrace.integrity import HashChainedLog, IntegrityEvent, localize, verify_event_signature
from securetrace.integrity.chain import NODE_ORDER, SEGMENT_ORDER, read_key, segment_key

REQUIRED = {"run_id", "msg_id", "node", "ts"}


def create_app(log: HashChainedLog, secret: bytes, segment_keys: dict[str, bytes]) -> FastAPI:
    app = FastAPI(title="SecureTrace collector")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True}

    @app.post("/events")
    def post_event(body: dict = Body(...)) -> dict[str, Any]:
        if not REQUIRED <= body.keys() or body.get("node") not in NODE_ORDER:
            raise HTTPException(422, "malformed event")
        event = IntegrityEvent.from_dict(body)
        if not verify_event_signature(secret, event):
            raise HTTPException(401, "bad event signature")
        seq = log.append({"kind": "integrity_event", **event.to_dict()})
        return {"seq": seq}

    def _events(run_id: str, msg_id: str) -> list[dict]:
        return [r for r in log.records(run_id, msg_id) if r.get("kind") == "integrity_event"]

    @app.get("/runs/{run_id}/messages/{msg_id}/localization")
    def message_localization(run_id: str, msg_id: str) -> dict[str, Any]:
        events = _events(run_id, msg_id)
        if not events:
            raise HTTPException(404, "no events for message")
        return localize(events, segment_keys).to_dict()

    @app.get("/runs/{run_id}/localizations")
    def run_localizations(run_id: str) -> dict[str, Any]:
        by_msg: dict[str, list[dict]] = {}
        for r in log.records(run_id):
            if r.get("kind") == "integrity_event":
                by_msg.setdefault(r["msg_id"], []).append(r)
        out = {}
        for msg_id, events in by_msg.items():
            result = localize(events, segment_keys).to_dict()
            result["first_ts"] = min(e["ts"] for e in events)
            result["last_ts"] = max(e["ts"] for e in events)
            out[msg_id] = result
        return out

    @app.get("/runs/{run_id}/events")
    def run_events(run_id: str) -> list[dict]:
        return [r for r in log.records(run_id) if r.get("kind") == "integrity_event"]

    @app.get("/log/head")
    def log_head() -> dict[str, Any]:
        seq, row_hash = log.head()
        return {"seq": seq, "hash": row_hash}

    @app.get("/log/verify")
    def log_verify() -> dict[str, Any]:
        ok, bad_seq, reason = log.verify()
        return {"ok": ok, "first_bad_seq": bad_seq, "reason": reason}

    return app


def app_from_env() -> FastAPI:
    secret = read_key(os.environ, "SECURETRACE_SECRET_KEY")
    keys = {s: segment_key(os.environ, s) for s in SEGMENT_ORDER}
    return create_app(HashChainedLog(os.environ.get("COLLECTOR_DB", "/data/collector.db")), secret, keys)
