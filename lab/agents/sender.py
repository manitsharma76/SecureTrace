"""Sender: sends N random payloads through the path, one TCP connection per message per hop.

  python -m lab.agents.sender --run-id r001 --n 30 --seed 7
Env: NEXT_HOP (relay1's address on segment A), LAB_KEY_A, EVENT_KEY_SENDER, COLLECTOR_URL.
"""

from __future__ import annotations

import argparse
import os
import random
import socket
import time

from lab.agents import noise
from lab.agents.common import event_key, flush_events, node_keys, post_event
from lab.agents.wire import TRANSFER_PORT, write_frame
from securetrace.integrity import originate


def send_messages(run_id: str, n: int, seed: int, size_min: int, size_max: int, mean_gap: float) -> dict:
    rng = random.Random(seed)
    target = os.environ["NEXT_HOP"]
    acks = {"1": 0, "0": 0, "": 0}
    for i in range(n):
        size = int(rng.uniform(size_min, size_max))
        data = rng.randbytes(size)
        msg_id = f"{run_id}-{i:04d}"
        event, chain = originate(node_keys("sender"), event_key("sender"), run_id, msg_id, data, time.time())
        post_event(event.to_dict())
        try:
            with socket.create_connection((target, TRANSFER_PORT), timeout=15) as s:
                write_frame(s, {"run_id": run_id, "msg_id": msg_id, "chain": chain}, data,
                            chunk=rng.choice([4096, 8192, 16384]), pace_s=rng.choice([0, 0, 0.001]))
                acks[s.recv(1).decode() or ""] += 1
        except OSError:
            acks[""] += 1
        time.sleep(rng.expovariate(1.0 / mean_gap))
    return acks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size-min", type=int, default=20_000)
    parser.add_argument("--size-max", type=int, default=800_000)
    parser.add_argument("--mean-gap", type=float, default=1.0)
    args = parser.parse_args()
    event_key("sender")
    acks = send_messages(args.run_id, args.n, args.seed, args.size_min, args.size_max, args.mean_gap)
    flush_events()
    print(f"sent {args.n} messages; end-to-end acks: {acks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
