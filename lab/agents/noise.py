"""Benign background traffic: servers (run inside every node) and a randomized client.

Patterns are chosen to look unlike each other in flow statistics: short HTTP-style requests,
bulk downloads, interactive trickles with think times, periodic keepalives, UDP request/response.
Everything stays on the isolated lab segments.
"""

from __future__ import annotations

import argparse
import random
import socket
import socketserver
import threading
import time

HTTP_PORT, INTERACTIVE_PORT, UDP_PORT = 8080, 2222, 5353


class _Http(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        data = self.request.recv(4096)
        try:
            size = int(data.split(b" ")[1].strip(b"/").split(b"?")[0] or 1024)
        except (IndexError, ValueError):
            size = 1024
        body = b"x" * min(size, 4_000_000)
        self.request.sendall(b"HTTP/1.0 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body)


class _Interactive(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            while True:
                data = self.request.recv(1024)
                if not data:
                    return
                self.request.sendall(data[:64])
        except OSError:
            return


class _Udp(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        data, sock = self.request
        sock.sendto(data[:80] + b"\x00" * random.randint(8, 120), self.client_address)


class _TcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start_servers() -> None:
    for port, handler in ((HTTP_PORT, _Http), (INTERACTIVE_PORT, _Interactive)):
        server = _TcpServer(("0.0.0.0", port), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
    udp = socketserver.ThreadingUDPServer(("0.0.0.0", UDP_PORT), _Udp)
    udp.daemon_threads = True
    threading.Thread(target=udp.serve_forever, daemon=True).start()


def _http_get(target: str, size: int) -> None:
    with socket.create_connection((target, HTTP_PORT), timeout=5) as s:
        s.sendall(b"GET /%d HTTP/1.0\r\n\r\n" % size)
        while s.recv(65536):
            pass


def _interactive(target: str, rng: random.Random) -> None:
    with socket.create_connection((target, INTERACTIVE_PORT), timeout=5) as s:
        for _ in range(rng.randint(5, 40)):
            s.sendall(b"k" * rng.randint(1, 48))
            s.recv(256)
            time.sleep(rng.uniform(0.05, 0.6))


def _keepalive(target: str, rng: random.Random) -> None:
    with socket.create_connection((target, INTERACTIVE_PORT), timeout=5) as s:
        for _ in range(rng.randint(3, 8)):
            s.sendall(b"ping")
            s.recv(64)
            time.sleep(rng.uniform(0.8, 2.0))


def _udp(target: str, rng: random.Random) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.settimeout(2)
        for _ in range(rng.randint(1, 6)):
            s.sendto(b"q" * rng.randint(20, 90), (target, UDP_PORT))
            try:
                s.recvfrom(512)
            except socket.timeout:
                pass
            time.sleep(rng.uniform(0.05, 0.5))


def run_client(target: str, duration: float, seed: int, intensity: float = 1.0) -> int:
    rng = random.Random(seed)
    end, count = time.time() + duration, 0
    while time.time() < end:
        kind = rng.choices(["http_small", "http_bulk", "interactive", "keepalive", "udp"], [4, 2, 2, 1, 3])[0]
        try:
            if kind == "http_small":
                _http_get(target, rng.randint(200, 20_000))
            elif kind == "http_bulk":
                _http_get(target, rng.randint(300_000, 3_000_000))
            elif kind == "interactive":
                _interactive(target, rng)
            elif kind == "keepalive":
                _keepalive(target, rng)
            else:
                _udp(target, rng)
            count += 1
        except OSError:
            pass
        time.sleep(rng.expovariate(intensity * 2.0))
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Benign background traffic client.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--intensity", type=float, default=1.0)
    args = parser.parse_args()
    print(f"noise flows: {run_client(args.target, args.duration, args.seed, args.intensity)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
