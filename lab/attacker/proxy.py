"""In-path TCP-terminating relay for one lab segment, used to generate the tampered-session
training data described in DEVELOPMENT_PLAN.md ("biggest risk" / attacker variants).

How it fits the topology: each logical segment (A/B/C) is wired as two point-to-point Docker
networks, e.g. segB_in (relay1 <-> attacker) and segB_out (attacker <-> relay2), so the attacker
is the only reachable peer on the wire for either side -- a genuine on-path relay, set up by
ordinary network topology, not by ARP spoofing or packet injection (see lab/docker-compose.yml).
It is never attached to the control network and holds no HMAC keys. A `recon()` step can run
beforehand to vary the attacker's footprint (see recon.py).

This file contains no packet crafting or spoofing, only an ordinary socket relay bound to the
segment's "_in" address. For a session that doesn't target this segment, nothing is listening
there, so sender/relay traffic for that segment... is simply not usable -- the orchestrator must
always run this relay (in pass-through mode, i.e. `--mutator` omitted) on every segment during a
session, and only switch one segment's instance to a real mutator for the tamper case. In
pass-through mode the relay is functionally transparent: same bytes, same chain, forwarded as-is.
The relay:
  1. Accepts the redirected TCP connection from the upstream legitimate node.
  2. Reads the SecureTrace wire frame (lab/agents/wire.py): header + payload.
  3. Optionally mutates the payload bytes (the `mutate` callback -- this is what "tampering"
     means here: the integrity layer is expected to catch it).
  4. Opens its own new TCP connection to the real downstream node and forwards the frame.

Because the TCP connection is fully terminated and re-opened, this naturally reproduces the
"TCP-terminating proxy" footprint from the plan: added latency, a new TCP handshake and
congestion-control state, and re-segmentation of the payload into this process's own packets.
"""

from __future__ import annotations

import argparse
import random
import socket
import socketserver
import time
from typing import Callable

from lab.agents.wire import TRANSFER_PORT, read_frame, write_frame

Mutator = Callable[[bytes, random.Random], bytes]


def flip_byte(payload: bytes, rng: random.Random) -> bytes:
    """Smallest possible change: must fail hop-tag verification (securetrace/integrity)."""
    if not payload:
        return payload
    i = rng.randrange(len(payload))
    return payload[:i] + bytes([payload[i] ^ 0x01]) + payload[i + 1 :]


def rewrite_size(payload: bytes, rng: random.Random) -> bytes:
    """Truncate or pad: changes the digest and the length a downstream flow extractor sees."""
    if rng.random() < 0.5 and len(payload) > 16:
        return payload[: rng.randint(len(payload) // 2, len(payload) - 1)]
    return payload + rng.randbytes(rng.randint(16, 2048))


MUTATORS: dict[str, Mutator] = {"flip_byte": flip_byte, "rewrite_size": rewrite_size}


def make_handler(downstream_host: str, downstream_port: int, mutator: Mutator | None, seed: int, pace_s: float):
    rng = random.Random(seed)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                header, payload = read_frame(self.request)
            except (ConnectionError, ValueError, KeyError):
                return
            if mutator is not None:
                payload = mutator(payload, rng)
            try:
                with socket.create_connection((downstream_host, downstream_port), timeout=10) as out:
                    write_frame(out, header, payload, pace_s=pace_s)  # re-segmented here, by design
                    ack = out.recv(1)
            except OSError:
                ack = b"0"
            self.request.sendall(ack or b"0")

    return Handler


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def run(
    listen_host: str, downstream_host: str, mutator_name: str | None, seed: int, pace_s: float,
    port: int = TRANSFER_PORT,
) -> None:
    mutator = MUTATORS[mutator_name] if mutator_name else None
    handler = make_handler(downstream_host, TRANSFER_PORT, mutator, seed, pace_s)
    print(f"attacker relay on {listen_host}:{port} -> {downstream_host}:{TRANSFER_PORT}, mutator={mutator_name}",
          flush=True)
    Server((listen_host, port), handler).serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description="On-path TCP-terminating relay for one segment.")
    parser.add_argument("--listen", required=True, help="This container's address on the segment's _in network")
    parser.add_argument("--downstream", required=True, help="Real downstream node, on the segment's _out network")
    parser.add_argument("--mutator", choices=sorted(MUTATORS), default=None, help="Omit to relay untouched")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pace-s", type=float, default=0.0005, help="Per-chunk delay: added latency footprint")
    parser.add_argument("--listen-port", type=int, default=TRANSFER_PORT)
    args = parser.parse_args()
    run(args.listen, args.downstream, args.mutator, args.seed, args.pace_s, args.listen_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
