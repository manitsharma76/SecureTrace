#!/usr/bin/env python3
"""Generate random lab keys into the gitignored .env. Key values are never printed or logged.

Writes: SECURETRACE_SECRET_KEY, LAB_KEY_A/B/C, and one derived EVENT_KEY_<NODE> per node so a node
can sign only its own events. Refuses to overwrite an existing .env unless --force.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from securetrace.integrity import derive_event_key
from securetrace.integrity.chain import NODE_ORDER, SEGMENT_ORDER


def build_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    secret = secrets.token_hex(32)
    env = {"SECURETRACE_SECRET_KEY": secret}
    env.update({f"LAB_KEY_{s}": secrets.token_hex(32) for s in SEGMENT_ORDER})
    env.update({f"EVENT_KEY_{n.upper()}": derive_event_key(secret.encode(), n).hex() for n in NODE_ORDER})
    env["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns/mlflow.db"
    env.update(extra or {})
    return env


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=ROOT / ".env")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.path.exists() and not args.force:
        print(f"{args.path} exists; keeping it (use --force to regenerate).")
        return 0
    args.path.write_text("".join(f"{k}={v}\n" for k, v in build_env().items()), encoding="utf-8")
    print(f"Wrote {args.path} with {len(build_env())} entries (values not shown).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
