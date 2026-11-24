import random
import socket
import threading
import time

from lab.agents.wire import read_frame, write_frame
from lab.attacker.proxy import MUTATORS, flip_byte, make_handler, rewrite_size

RNG_SEED = 0


def test_flip_byte_changes_exactly_one_bit_of_one_byte():
    rng = random.Random(RNG_SEED)
    payload = bytes(range(64)) * 4
    mutated = flip_byte(payload, rng)
    assert len(mutated) == len(payload)
    diff = [i for i in range(len(payload)) if payload[i] != mutated[i]]
    assert len(diff) == 1
    assert bin(payload[diff[0]] ^ mutated[diff[0]]).count("1") == 1


def test_flip_byte_on_empty_payload_is_a_noop():
    assert flip_byte(b"", random.Random(0)) == b""


def test_rewrite_size_always_changes_length():
    rng = random.Random(1)
    payload = b"x" * 500
    for _ in range(20):
        mutated = rewrite_size(payload, rng)
        assert len(mutated) != len(payload)


def test_mutators_registry_matches_documented_names():
    assert set(MUTATORS) == {"flip_byte", "rewrite_size"}


def _run_relay(handler_factory, port: int):
    import socketserver

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    server = Server(("127.0.0.1", port), handler_factory)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _downstream_echo(port: int) -> list[bytes]:
    import socketserver

    received: list[bytes] = []

    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            _, payload = read_frame(self.request)
            received.append(payload)
            self.request.sendall(b"1")

    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    server = Server(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return received


def test_relay_forwards_payload_to_downstream_untouched_when_no_mutator():
    downstream_port, relay_port = 18100, 18101
    received = _downstream_echo(downstream_port)
    time.sleep(0.05)
    handler = make_handler("127.0.0.1", downstream_port, mutator=None, seed=RNG_SEED, pace_s=0.0)
    _run_relay(handler, relay_port)
    time.sleep(0.05)

    with socket.create_connection(("127.0.0.1", relay_port), timeout=5) as s:
        write_frame(s, {"run_id": "r", "msg_id": "m", "chain": []}, b"hello world")
        ack = s.recv(1)
    time.sleep(0.1)
    assert ack == b"1"
    assert received == [b"hello world"]


def test_relay_applies_mutator_before_forwarding():
    downstream_port, relay_port = 18110, 18111
    received = _downstream_echo(downstream_port)
    time.sleep(0.05)
    handler = make_handler("127.0.0.1", downstream_port, mutator=flip_byte, seed=RNG_SEED, pace_s=0.0)
    _run_relay(handler, relay_port)
    time.sleep(0.05)

    original = bytes(range(32))
    with socket.create_connection(("127.0.0.1", relay_port), timeout=5) as s:
        write_frame(s, {"run_id": "r", "msg_id": "m", "chain": []}, original)
        s.recv(1)
    time.sleep(0.1)
    assert received[0] != original
    assert len(received[0]) == len(original)
