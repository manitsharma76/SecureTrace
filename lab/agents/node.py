"""Relay / receiver agent. Verifies the incoming hop tag, reports a signed event, forwards.

  python -m lab.agents.node --role relay1      (env NEXT_HOP=<ip of next node on the outgoing segment>)
  python -m lab.agents.node --role receiver

A relay forwards what it received even when verification failed (see securetrace.integrity.agent),
so tampering on one segment does not make later segments report false failures.
"""

from __future__ import annotations

import argparse
import os
import socket
import socketserver
import sys
import time

from lab.agents import noise
from lab.agents.common import event_key, node_keys, post_event
from lab.agents.wire import TRANSFER_PORT, read_frame, write_frame
from securetrace.integrity import process_hop


def make_handler(role: str):
    next_hop = os.environ.get("NEXT_HOP")

    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            try:
                header, payload = read_frame(self.request)
            except (ConnectionError, ValueError, KeyError):
                return
            event, chain_out = process_hop(
                role, node_keys(role), event_key(role), header["run_id"], header["msg_id"],
                payload, header.get("chain"), time.time(),
            )
            post_event(event.to_dict())
            ack = b"1" if event.verify_ok else b"0"
            if next_hop:
                try:
                    with socket.create_connection((next_hop, TRANSFER_PORT), timeout=10) as out:
                        write_frame(
                            out, {"run_id": header["run_id"], "msg_id": header["msg_id"], "chain": chain_out or []},
                            payload,
                        )
                        out.recv(1)
                except OSError as exc:
                    print(f"forward failed: {exc}", file=sys.stderr)
            self.request.sendall(ack)

    return Handler


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["relay1", "relay2", "receiver"])
    args = parser.parse_args()
    event_key(args.role)  # fail fast if misconfigured (message never contains key material)
    noise.start_servers()
    print(f"{args.role} listening on :{TRANSFER_PORT}, next hop {os.environ.get('NEXT_HOP')}", flush=True)
    Server(("0.0.0.0", TRANSFER_PORT), make_handler(args.role)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
