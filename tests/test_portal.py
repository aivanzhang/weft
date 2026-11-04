from contextlib import contextmanager
from http.client import HTTPConnection
import io
import json
import threading

import pytest

from weft.io import digest, write_json
from weft.lab_worker import atomic_json
from weft.portal import Portal, PortalServer
from test_lab_http import make_ready_lab


@contextmanager
def serving(portal):
    server = PortalServer(("127.0.0.1", 0), portal)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)


def request(server, path, method="GET", body=None, *, auth=True, headers=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
    request_headers = {"Authorization": "Bearer " + server.portal.token} if auth else {}
    request_headers.update(headers or {})
    try:
        connection.request(method, path, body, request_headers)
        response = connection.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()


@pytest.fixture
def portal(tmp_path):
    value = Portal(tmp_path / "portal", project=tmp_path, start=False)
    yield value
    value.close()


def upload(portal, identity="a" * 32):
    return portal.upload(io.BytesIO(b"example video"), 13, name="input.webm", mode="exercise", ai=False, request_id=identity)


def test_upload_pins_original_and_no_duplicate_queue(portal):
    job = upload(portal)
    assert job["status"] == "queued"
    assert job["source_sha256"] == digest(portal.root / job["id"] / "source.webm")
    with pytest.raises(ValueError, match="already exists"):
        upload(portal)
    assert len(portal.jobs) == 1


def test_interrupted_upload_and_restart_preserve_evidence(tmp_path):
    portal = Portal(tmp_path / "portal", project=tmp_path, start=False)
    with pytest.raises(ValueError, match="interrupted"):
        portal.upload(io.BytesIO(b"short"), 13, name="input.mp4", mode="exercise", ai=False, request_id="a" * 32)
    job = upload(portal, "b" * 32)
    portal.close()
    reopened = Portal(tmp_path / "portal", project=tmp_path, start=False)
    try:
        assert reopened.summary(job["id"])["status"] == "interrupted"
        assert (reopened.root / job["id"] / "source.webm").read_bytes() == b"example video"
        assert len(reopened.jobs) == 2
    finally:
        reopened.close()


def test_directory_lease_prevents_duplicate_workers(portal):
    with pytest.raises(ValueError, match="Another portal"):
        Portal(portal.root, project=portal.project, start=False)


def test_cancel_does_not_relabel_ready_result(portal):
    job = upload(portal)
    assert portal.cancel(job["id"])["status"] == "cancelled"
    path = portal.root / job["id"] / "status.json"
    atomic_json(path, {"status": "ready", "stage": "ready", "events": []})
    before = path.read_bytes()
    assert portal.cancel(job["id"])["status"] == "ready"
    assert path.read_bytes() == before


def test_muse_review_auth_dedup_cancel_and_artifacts(portal, monkeypatch):
    import weft.portal as module
    monkeypatch.setattr(module, "muse_availability", lambda _: {"available": True})
    job = upload(portal)
    identity = job["id"]
    folder = portal.root / identity
    with pytest.raises(ValueError, match="Finish reconstruction"):
        portal.review(identity)
    atomic_json(folder / "status.json", {"status": "ready", "route": "exercise", "events": []})
    before = (folder / "status.json").read_bytes()
    with serving(portal) as server:
        path = f"/api/projects/{identity}/muse"
        assert request(server, path, "POST", auth=False)[0] == 401
        assert request(server, path, "POST", headers={"Origin": "https://attacker.invalid"})[0] == 403
        assert request(server, path, "POST", b"unexpected")[0] == 400
        assert request(server, path, "POST")[0] == 202
        assert request(server, path, "POST")[0] == 202
        assert portal.summary(identity)["muse"]["status"] == "queued"
        (folder / "muse/advice").mkdir()
        write_json(folder / "muse/advice/hypotheses.json", {"accepted": False, "advice": {}})
        assert request(server, f"/api/projects/{identity}/files/muse/advice/hypotheses.json")[0] == 200
        assert request(server, f"/api/projects/{identity}/files/muse/status.json")[0] == 400
        assert request(server, f"/api/projects/{identity}/cancel", "POST")[0] == 200
    assert portal.summary(identity)["muse"]["status"] == "cancelled"
    assert (folder / "status.json").read_bytes() == before
    assert portal.summary(identity)["sandbox"]
    assert portal.review(identity)["muse"]["status"] == "cancelled"


def test_muse_restart_preserves_ready_world_without_retry(tmp_path, monkeypatch):
    import weft.portal as module
    monkeypatch.setattr(module, "muse_availability", lambda _: {"available": True})
    portal = Portal(tmp_path / "portal", project=tmp_path, start=False)
    job = upload(portal)
    folder = portal.root / job["id"]
    atomic_json(folder / "status.json", {"status": "ready", "route": "objects", "events": []})
    portal.review(job["id"])
    portal.close()
    reopened = Portal(tmp_path / "portal", project=tmp_path, start=False)
    try:
        assert reopened.summary(job["id"])["status"] == "ready"
        assert reopened.summary(job["id"])["muse"]["status"] == "interrupted"
        assert reopened.review(job["id"])["muse"]["status"] == "interrupted"
    finally:
        reopened.close()


def test_muse_missing_budget_does_not_queue(portal):
    job = upload(portal)
    folder = portal.root / job["id"]
    atomic_json(folder / "status.json", {"status": "ready", "route": "exercise", "events": []})
    with pytest.raises(ValueError, match="allowance"):
        portal.review(job["id"])
    assert not (folder / "muse").exists()


def test_inflight_muse_receipt_write_does_not_break_project_list(portal):
    job = upload(portal)
    folder = portal.root / job["id"]
    (folder / "muse/advice").mkdir(parents=True)
    atomic_json(folder / "muse/status.json", {"status": "running"})
    (folder / "muse/advice/advisor-call.json").write_text('{"status":')
    assert "receipt" not in portal.summary(job["id"])["muse"]
    atomic_json(folder / "muse/status.json", {"status": "failed"})
    with pytest.raises(json.JSONDecodeError):
        portal.summary(job["id"])


def test_local_checks_need_review_but_no_key_and_preserve_ready_world(portal):
    job = upload(portal)
    identity = job["id"]
    folder = portal.root / identity
    with pytest.raises(ValueError, match="completed simulation"):
        portal.check(identity)
    atomic_json(folder / "status.json", {"status": "ready", "route": "exercise", "events": []})
    (folder / "muse").mkdir()
    atomic_json(folder / "muse/status.json", {"status": "proposed"})
    before = (folder / "status.json").read_bytes(), (folder / "muse/status.json").read_bytes()
    with serving(portal) as server:
        path = f"/api/projects/{identity}/muse/checks"
        assert request(server, path, "POST", auth=False)[0] == 401
        assert request(server, path, "POST", headers={"Origin": "https://attacker.invalid"})[0] == 403
        assert request(server, path, "POST", b"unexpected")[0] == 400
        assert request(server, path, "POST")[0] == 202
        assert request(server, path, "POST")[0] == 202
        assert portal.summary(identity)["muse"]["checks"]["status"] == "queued"
        assert request(server, f"/api/projects/{identity}/cancel", "POST")[0] == 200
    assert portal.summary(identity)["muse"]["checks"]["status"] == "cancelled"
    assert before == ((folder / "status.json").read_bytes(), (folder / "muse/status.json").read_bytes())
    assert portal.check(identity)["muse"]["checks"]["status"] == "cancelled"


def test_check_restart_keeps_original_ready_and_review_proposed(tmp_path):
    portal = Portal(tmp_path / "portal", project=tmp_path, start=False)
    job = upload(portal)
    folder = portal.root / job["id"]
    atomic_json(folder / "status.json", {"status": "ready", "route": "exercise", "events": []})
    (folder / "muse").mkdir()
    atomic_json(folder / "muse/status.json", {"status": "proposed"})
    portal.check(job["id"])
    portal.close()
    reopened = Portal(tmp_path / "portal", project=tmp_path, start=False)
    try:
        result = reopened.summary(job["id"])
        assert result["status"] == "ready" and result["muse"]["status"] == "proposed"
        assert result["muse"]["checks"]["status"] == "interrupted"
        assert reopened.check(job["id"])["muse"]["checks"]["status"] == "interrupted"
    finally:
        reopened.close()


def test_active_cancel_uses_independent_marker(portal):
    job = upload(portal)
    portal.active = job["id"]
    assert portal.cancel(job["id"])["status"] == "cancelling"
    assert (portal.root / job["id"] / "cancel-request.json").is_file()


def test_upload_rejects_unsafe_inputs_before_writing(portal):
    for values in ({"name": "../credentials.mp4"}, {"mode": "auto"}, {"ai": True}, {"size": 100_000_001}):
        args = {"name": "video.mp4", "mode": "exercise", "ai": False, "size": 13, "request_id": "a" * 32} | values
        with pytest.raises(ValueError):
            portal.upload(io.BytesIO(b"example video"), **args)
    assert not portal.jobs


def test_http_auth_upload_progress_artifacts_and_cancel(portal):
    with serving(portal) as server:
        assert request(server, "/api/portal", auth=False)[0] == 401
        assert request(server, "/api/portal", headers={"Origin": "https://attacker.invalid"})[0] == 403
        status, headers, body = request(server, "/", auth=False)
        assert status == 200 and b"Build simulation" in body and "Content-Disposition" not in headers
        path = "/api/projects?name=input.webm&mode=exercise&ai=false&request_id=" + "a" * 32
        status, _, raw = request(server, path, "POST", b"example video")
        assert status == 202
        job = json.loads(raw)
        status, _, raw = request(server, "/api/portal")
        assert status == 200 and len(json.loads(raw)["jobs"]) == 1
        assert request(server, f"/api/projects/{job['id']}/files/source.webm")[2] == b"example video"
        assert request(server, f"/api/projects/{job['id']}/files/job.json")[0] == 400
        assert request(server, f"/api/projects/{job['id']}/files/../../../credentials/openai-api-key")[0] == 400
        assert request(server, f"/api/projects/{job['id']}/cancel", "POST")[0] == 200


def test_two_sandboxes_route_state_without_changing_global_lab(portal):
    ids = []
    for marker in ("a", "b"):
        job = upload(portal, marker * 32)
        lab = make_ready_lab(portal.root / job["id"] / "lab")
        lab.close()
        atomic_json(portal.root / job["id"] / "status.json", {"status": "ready", "events": []})
        ids.append(job["id"])
    with serving(portal) as server:
        for identity in ids:
            code, headers, _ = request(server, f"/labs/{identity}/", auth=False)
            assert code == 200
            assert headers["X-Frame-Options"] == "SAMEORIGIN"
            assert "frame-ancestors 'self'" in headers["Content-Security-Policy"]
            assert request(server, f"/labs/{identity}/api/status", auth=False)[0] == 401
            assert request(server, f"/labs/{identity}/api/status", headers={"Origin": "https://attacker.invalid"})[0] == 403
            assert request(server, f"/labs/{identity}/api/status")[0] == 200
        _, headers, _ = request(server, "/demo", auth=False)
        assert headers["X-Frame-Options"] == "DENY"
        status, _, raw = request(server, f"/labs/{ids[0]}/api/step", "POST", b'{"action":1}', headers={"Content-Type": "application/json"})
        assert status == 200
        first = portal.labs[ids[0]].session.snapshot()
        second = portal.labs[ids[1]].session.snapshot()
        assert first != second
        assert server.lab is portal


def test_demo_code_exchange_connects_without_token_and_keeps_other_routes_private(portal):
    code = portal.demo.code()
    with serving(portal) as server:
        body = json.dumps({"code": code}).encode()
        assert request(server, "/api/demo/start", "POST", auth=False)[0] == 401
        assert request(server, "/api/portal", auth=False)[0] == 401
        assert request(server, "/api/demo/unlock", "POST", b'{"code":"wrong"}', auth=False)[0] == 400
        assert request(server, "/api/demo/unlock", "POST", b'[]', auth=False)[0] == 400
        assert request(server, "/api/demo/unlock", "POST", body, auth=False,
                       headers={"Origin": "https://attacker.invalid"})[0] == 403
        assert request(server, "/api/demo/unlock", "POST", body, auth=False,
                       headers={"Host": "attacker.invalid"})[0] == 403
        status, headers, raw = request(server, "/api/demo/unlock", "POST", body, auth=False,
                                      headers={"Origin": f"http://127.0.0.1:{server.server_port}"})
        result = json.loads(raw)
        assert status == 200 and result == {"ok": True, "token": portal.token}
        assert headers["Cache-Control"] == "no-store"
        assert request(server, "/api/demo/steps", auth=False,
                       headers={"Authorization": "Bearer " + result["token"]})[0] == 200
        assert not portal.jobs
