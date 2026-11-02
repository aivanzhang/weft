"""Portal repair requests share the existing authenticated, bounded worker queue."""

import io
import json

import pytest

from test_portal import serving, request
from weft import portal as module
from weft.lab_worker import atomic_json


@pytest.fixture
def ready(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "repair_availability", lambda root: {"available": True, "reason": "Bounded local comparison"})
    portal = module.Portal(tmp_path / "portal", project=tmp_path, start=False)
    job = portal.upload(io.BytesIO(b"analytical video placeholder"), 27, name="source.mp4",
                        mode="exercise", ai=False, request_id="b" * 32)
    identity = job["id"]
    root = portal.root / identity
    atomic_json(root / "status.json", {"status": "ready", "route": "exercise", "events": []})
    (root / "muse").mkdir()
    atomic_json(root / "muse/status.json", {"status": "proposed"})
    yield portal, identity, root
    portal.close()


def test_repair_auth_dedup_cancel_and_artifact_scope(ready):
    portal, identity, root = ready
    before = (root / "status.json").read_bytes()
    with serving(portal) as server:
        path = f"/api/projects/{identity}/muse/repairs"
        assert request(server, path, "POST", auth=False)[0] == 401
        assert request(server, path, "POST", headers={"Origin": "https://foreign.invalid"})[0] == 403
        assert request(server, path, "POST", b"{}")[0] == 400
        assert request(server, path, "POST")[0] == 202
        first = (root / "muse/repairs/status.json").read_bytes()
        assert request(server, path, "POST")[0] == 202
        assert (root / "muse/repairs/status.json").read_bytes() == first
        assert portal.cancel(identity)["muse"]["repairs"]["status"] == "cancelled"
    assert (root / "status.json").read_bytes() == before
    run = root / "muse/repairs/run"
    run.mkdir()
    (run / "report.json").write_text("{}")
    assert portal.artifact(identity, "muse/repairs/run/report.json") == run / "report.json"
    with pytest.raises(ValueError, match="public"):
        portal.artifact(identity, "muse/repairs/status.json")


def test_running_comparison_cancels_before_queued_check_and_never_cancels_completed_result(ready):
    portal, identity, root = ready
    portal.repair(identity)
    atomic_json(root / "muse/repairs/status.json", {"status": "running", "applied": False})
    checks = root / "muse/checks"
    checks.mkdir()
    atomic_json(checks / "status.json", {"status": "queued"})
    assert portal.cancel(identity)["muse"]["repairs"]["status"] == "cancelling"
    assert json.loads((checks / "status.json").read_bytes())["status"] == "queued"
    atomic_json(root / "muse/repairs/status.json", {"status": "failed", "applied": False})
    portal.cancel(identity)
    assert json.loads((checks / "status.json").read_bytes())["status"] == "cancelled"


def test_unavailable_repair_does_not_create_work(ready, monkeypatch):
    portal, identity, root = ready
    monkeypatch.setattr(module, "repair_availability", lambda _: {"available": False, "reason": "No supported hypothesis"})
    with pytest.raises(ValueError, match="No supported hypothesis"):
        portal.repair(identity)
    assert not (root / "muse/repairs").exists()


def test_queue_leaves_attempt_admission_to_the_repair_worker(ready, monkeypatch):
    portal, identity, root = ready
    portal.repair(identity)
    calls = []
    def run(selected, **modes):
        assert selected == identity and modes == {"review": False, "checks": False, "repairs": True}
        assert json.loads((root / "muse/repairs/status.json").read_bytes())["status"] == "queued"
        calls.append(selected)
        portal.stopping.set()
    monkeypatch.setattr(portal, "_run", run)
    portal._work()
    assert calls == [identity] and portal.active is None
