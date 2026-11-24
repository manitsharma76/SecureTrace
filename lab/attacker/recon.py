"""Optional preceding-recon step (DEVELOPMENT_PLAN.md: "each with optional preceding recon").

Its only purpose is to vary the attacker's footprint and timing before a tamper run, so the
weak-label builder sees some tamper sessions with a short burst of extra connections beforehand
and some without. It connects to a fixed, already-known list of lab addresses on a fixed, already-
known list of lab ports (the other containers' real service ports); it discovers nothing, since
the topology is static and defined in docker-compose.yml. This is plain TCP connect probing
(`socket.create_connection`), not packet crafting.
"""

from __future__ import annotations

import argparse
import random
import socket
import time

from lab.agents.noise import HTTP_PORT, INTERACTIVE_PORT, TRANSFER_PORT


def run(targets: list[str], seed: int, duration: float) -> int:
    rng = random.Random(seed)
    ports = [TRANSFER_PORT, HTTP_PORT, INTERACTIVE_PORT]
    end, attempts = time.time() + duration, 0
    while time.time() < end:
        host, port = rng.choice(targets), rng.choice(ports)
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass
        except OSError:
            pass
        attempts += 1
        time.sleep(rng.uniform(0.02, 0.3))
    return attempts


def main() -> int:
    parser = argparse.ArgumentParser(description="Benign-shaped connect probing before a tamper run.")
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()
    print(f"recon attempts: {run(args.targets, args.seed, args.duration)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
