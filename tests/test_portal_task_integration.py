"""Portal task routing and lifecycle; native/source qualification has separate checks."""

from hashlib import sha256
from io import BytesIO
import json
import sys
from types import SimpleNamespace

import pytest

from weft.io import write_json
from weft.portal import Portal, PortalHandler


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.setattr("weft.portal.muse_availability", lambda _: {"available": False})
    portal = Portal(tmp_path / "portal", project=tmp_path, start=False)
    identity = "a" * 32
    folder = portal.root / identity
    folder.mkdir()
    job = {"id": identity, "name": "Task report", "created_at": "fixture", "ai": False,
        "mode": "exercise", "source_sha256": "b" * 64, "kind": "task_job",
        "task_operation": "train_evaluate", "parent_id": "c" * 32}
    portal.jobs[identity] = job
    write_json(folder / "job.json", job)
    write_json(folder / "status.json", {"status": "queued", "stage": "train_evaluate"})
    (folder / "task-job").mkdir()
    yield portal, identity, folder
    if portal.lease is not None:
        portal.close()


def test_report_child_has_no_source_sandbox_or_implicit_work(project):
    portal, identity, folder = project
    report = {"status": "complete", "task_success": False}
    write_json(folder / "task-job/report.json", report)
    write_json(folder / "status.json", {"status": "complete", "report_sha256": sha256((folder / "task-job/report.json").read_bytes()).hexdigest(),
        "artifacts": ["task-job/report.json"]})
    result = portal.summary(identity)
    assert not result["source_available"] and result["sandbox"] is None
    assert result["task_job"]["report"] == report
    assert result["task_tools"] == {"create_available": False, "learning_available": False}
    assert not result["perception"]["available"] and not result["support_experiment_available"]
    before = (folder / "status.json").read_bytes()
    assert portal.cancel(identity)["status"] == "complete"
    assert (folder / "status.json").read_bytes() == before
    assert portal._reserved_growth() == 0 and portal._pending_jobs() == 0
    for action in ("review", "checks", "repairs"):
        with pytest.raises(ValueError, match="Task-learning reports"):
            portal._require_review_body(identity, action=action)


def test_task_artifact_allowlist_and_link_refusal(project):
    portal, identity, folder = project
    write_json(folder / "task-job/report.json", {"status": "failed"})
    assert portal.artifact(identity, "task-job/report.json") == folder / "task-job/report.json"
    (folder / "worker.log").write_text("Local worker evidence")
    assert portal.artifact(identity, "worker.log") == folder / "worker.log"
    for name in ("task-job/parent/job.json", "task-job/../job.json", "task-job/implementation/portal_task_jobs.py"):
        with pytest.raises(ValueError): portal.artifact(identity, name)
    (folder / "task-job/request.json").symlink_to(folder / "job.json")
    with pytest.raises(ValueError): portal.artifact(identity, "task-job/request.json")


def test_worker_supervisor_is_keyless_and_preserves_complete(project, monkeypatch):
    portal, identity, folder = project
    seen = []
    for name in ("OPENAI_API_KEY", "META_API_KEY", "MODEL_API_KEY"):
        monkeypatch.setenv(name, "fixture-secret-never-forward")
    def launch(argv, **options):
        seen.append((argv, options["env"]))
        write_json(folder / "status.json", {"status": "complete"})
        return SimpleNamespace(poll=lambda: 0, wait=lambda **_: 0, returncode=0)
    monkeypatch.setattr("weft.portal.subprocess.Popen", launch)
    portal._run(identity)
    assert seen[0][0][-1] == "tasks"
    assert not ({"OPENAI_API_KEY", "META_API_KEY", "MODEL_API_KEY"} & seen[0][1].keys())
    assert json.loads((folder / "status.json").read_bytes())["status"] == "complete"


def test_queue_leaves_exclusive_start_to_task_worker(project, monkeypatch):
    portal, identity, folder = project
    seen = []
    def run(actual, **options):
        seen.append(portal._status(actual)["status"])
        portal.stopping.set()
    monkeypatch.setattr(portal, "_run", run)
    portal._work()
    assert seen == ["queued"]


def test_task_dispatch_uses_bounded_cpu_and_own_executor(project, monkeypatch):
    from weft import portal_worker
    portal, identity, folder = project
    calls, limits = [], []
    monkeypatch.setattr(sys, "argv", ["worker", str(folder), str(portal.project), "tasks"])
    monkeypatch.setattr(portal_worker, "_limit", lambda kind, value: limits.append((kind, value)))
    monkeypatch.setattr(portal_worker.threading.Thread, "start", lambda _: None)
    monkeypatch.setattr("weft.portal_task_jobs.execute_job", lambda child, **kw: calls.append((child, kw)) or 0)
    assert portal_worker.main() == 0
    assert calls == [(folder, {"project": portal.project})]
    assert (portal_worker.resource.RLIMIT_CPU, 185) in limits
    assert (folder / "worker-receipt.json").is_file()


@pytest.mark.parametrize("operation", ["create", "train"])
def test_task_http_parser_is_authenticated_bounded_and_strict(operation):
    seen = []
    handler = SimpleNamespace(path="/api/projects/" + "a" * 32 + "/tasks/" + operation,
        headers={"Content-Length": "2"}, rfile=BytesIO(b"{}"), connection=SimpleNamespace(settimeout=lambda _: None),
        server=SimpleNamespace(portal=SimpleNamespace(task_action=lambda *args: seen.append(args) or {"status": "queued"})),
        route_lab=lambda: None, authorized=lambda: True, reply=lambda result, status=200: (status, result))
    assert PortalHandler.do_POST(handler)[0] == 202
    assert seen == [("a" * 32, operation, {})]
    handler.authorized = lambda: False
    assert PortalHandler.do_POST(handler) is None and len(seen) == 1
    handler.authorized = lambda: True
    for body in (b'{"a":1,"a":2}', b'{"a":NaN}', b"{" * 4097):
        handler.rfile = BytesIO(body); handler.headers = {"Content-Length": str(len(body))}
        assert PortalHandler.do_POST(handler)[0] == 400
    assert len(seen) == 1


def test_action_blocks_active_parent_and_inflight_recording(project, monkeypatch):
    portal, identity, _ = project
    monkeypatch.setattr("weft.portal_task_jobs.create_task_child", lambda *_args, **_kw: pytest.fail("No competing job"))
    portal.active = identity
    with pytest.raises(ValueError, match="worker is finishing"): portal.task_action(identity, "create", {})
    portal.active = None
    portal.labs[identity] = SimpleNamespace(loading=False, recorder=object(), export_future=None, close=lambda: None)
    with pytest.raises(ValueError, match="recording"): portal.task_action(identity, "create", {})
    with pytest.raises(ValueError, match="Unknown"): portal.task_action(identity, "replace", {})


def test_completed_report_survives_restart(project):
    portal, identity, folder = project
    write_json(folder / "status.json", {"status": "complete"})
    portal.close()
    portal.lease = None
    reopened = Portal(portal.root, project=portal.project, start=False)
    try:
        assert reopened.summary(identity)["status"] == "complete"
        assert reopened._pending_jobs() == 0
    finally:
        reopened.close()
