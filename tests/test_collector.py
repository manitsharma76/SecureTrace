import pytest
from fastapi.testclient import TestClient

from lab.collector.app import create_app
from securetrace.integrity import HashChainedLog
from tests.test_integrity import SECRET, SEG_KEYS, flip_byte, run_path


@pytest.fixture
def client():
    log = HashChainedLog()
    return TestClient(create_app(log, SECRET, SEG_KEYS)), log


def post_all(client, events):
    for e in events:
        assert client.post("/events", json=e).status_code == 200


def test_clean_message_localizes_ok(client):
    c, _ = client
    post_all(c, run_path())
    assert c.get("/runs/run1/messages/m1/localization").json()["status"] == "ok"


@pytest.mark.parametrize("segment", ["A", "B", "C"])
def test_tamper_is_localized_through_the_api(client, segment):
    c, _ = client
    post_all(c, run_path(tamper=(segment, flip_byte)))
    body = c.get("/runs/run1/messages/m1/localization").json()
    assert (body["status"], body["segment"]) == ("tamper", segment)
    assert c.get("/runs/run1/localizations").json()["m1"]["segment"] == segment


def test_forged_or_unsigned_events_are_rejected_and_not_logged(client):
    c, log = client
    event = run_path()[1]
    event["verify_ok"] = True
    event["digest_in"] = "0" * 64  # edited after signing
    assert c.post("/events", json=event).status_code == 401
    assert c.post("/events", json={"run_id": "r"}).status_code == 422
    assert log.head()[0] == 0


def test_log_endpoint_reports_integrity(client):
    c, _ = client
    post_all(c, run_path())
    assert c.get("/log/verify").json()["ok"] is True
    assert c.get("/log/head").json()["seq"] == 4


def test_unknown_message_is_404(client):
    c, _ = client
    assert c.get("/runs/x/messages/y/localization").status_code == 404
