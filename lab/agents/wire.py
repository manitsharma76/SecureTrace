"""Framing for the transfer protocol: 4-byte big-endian header length, JSON header, payload bytes.

Header: {"run_id", "msg_id", "size", "chain": [hex tags]}. The protocol is deliberately plain; the
integrity comes from the hop-chained tags, not from the transport.
"""

from __future__ import annotations

import json
import socket
import struct
import time

TRANSFER_PORT = 9100
MAX_HEADER = 64 * 1024


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(65536, n - len(buf)))
        if not chunk:
            raise ConnectionError("peer closed before the frame was complete")
        buf.extend(chunk)
    return bytes(buf)


def read_frame(sock: socket.socket) -> tuple[dict, bytes]:
    (hlen,) = struct.unpack(">I", recv_exact(sock, 4))
    if hlen > MAX_HEADER:
        raise ValueError("header too large")
    header = json.loads(recv_exact(sock, hlen))
    return header, recv_exact(sock, int(header["size"]))


def write_frame(
    sock: socket.socket, header: dict, payload: bytes, chunk: int = 16384, pace_s: float = 0.0
) -> None:
    header = {**header, "size": len(payload)}
    raw = json.dumps(header, separators=(",", ":")).encode()
    sock.sendall(struct.pack(">I", len(raw)) + raw)
    for i in range(0, len(payload), chunk):
        sock.sendall(payload[i : i + chunk])
        if pace_s:
            time.sleep(pace_s)
