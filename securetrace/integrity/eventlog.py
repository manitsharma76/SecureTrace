"""Tamper-evident (hash-chained) event log on SQLite.

row_hash = SHA256(prev_hash | seq | payload). Editing, deleting, inserting or reordering a row
breaks the chain at that point. Truncating the *tail* is only detectable against an externally
anchored head, so `head()` is meant to be saved elsewhere (the run orchestrator does).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading

GENESIS = "0" * 64


def _row_hash(prev_hash: str, seq: int, payload: str) -> str:
    return hashlib.sha256(f"{prev_hash}|{seq}|{payload}".encode()).hexdigest()


class HashChainedLog:
    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS log ("
                " seq INTEGER PRIMARY KEY, run_id TEXT, msg_id TEXT,"
                " payload TEXT NOT NULL, prev_hash TEXT NOT NULL, row_hash TEXT NOT NULL)"
            )
            self._db.execute("CREATE INDEX IF NOT EXISTS ix_log_msg ON log(run_id, msg_id)")
            self._db.commit()

    def append(self, record: dict) -> int:
        payload = json.dumps(record, sort_keys=True, separators=(",", ":"))
        with self._lock:
            row = self._db.execute("SELECT seq, row_hash FROM log ORDER BY seq DESC LIMIT 1").fetchone()
            seq, prev = (row[0] + 1, row[1]) if row else (1, GENESIS)
            self._db.execute(
                "INSERT INTO log(seq, run_id, msg_id, payload, prev_hash, row_hash) VALUES (?,?,?,?,?,?)",
                (seq, record.get("run_id"), record.get("msg_id"), payload, prev, _row_hash(prev, seq, payload)),
            )
            self._db.commit()
        return seq

    def head(self) -> tuple[int, str]:
        with self._lock:
            row = self._db.execute("SELECT seq, row_hash FROM log ORDER BY seq DESC LIMIT 1").fetchone()
        return (row[0], row[1]) if row else (0, GENESIS)

    def records(self, run_id: str | None = None, msg_id: str | None = None) -> list[dict]:
        query, args, clauses = "SELECT payload FROM log", [], []
        if run_id is not None:
            clauses.append("run_id = ?")
            args.append(run_id)
        if msg_id is not None:
            clauses.append("msg_id = ?")
            args.append(msg_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._lock:
            return [json.loads(p) for (p,) in self._db.execute(query + " ORDER BY seq", args)]

    def verify(self, expected_head: tuple[int, str] | None = None) -> tuple[bool, int | None, str]:
        """Return (ok, first_bad_seq, reason). `expected_head` also catches tail truncation."""
        with self._lock:
            rows = self._db.execute(
                "SELECT seq, run_id, msg_id, payload, prev_hash, row_hash FROM log ORDER BY seq"
            ).fetchall()
        prev, expected_seq = GENESIS, 1
        for seq, run_id, msg_id, payload, prev_hash, row_hash in rows:
            if seq != expected_seq:
                return False, expected_seq, "sequence gap (row deleted or inserted)"
            if prev_hash != prev:
                return False, seq, "broken link to previous row"
            if row_hash != _row_hash(prev_hash, seq, payload):
                return False, seq, "row content does not match its hash"
            record = json.loads(payload)
            if record.get("run_id") != run_id or record.get("msg_id") != msg_id:
                return False, seq, "indexed columns do not match payload"
            prev, expected_seq = row_hash, seq + 1
        if expected_head is not None and (expected_seq - 1, prev) != tuple(expected_head):
            return False, expected_seq, "head does not match anchored head (tail truncated or rewritten)"
        return True, None, "ok"
