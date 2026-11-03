"""Repair background jobs cannot duplicate physics or rewrite a completed comparison."""

import fcntl
import json
from pathlib import Path
import sys

import pytest

from weft import portal_repair_jobs as jobs
from weft.io import digest, write_json


def job(root, *, queue=True):
    (root / "muse/advice").mkdir(parents=True)
    write_json(root / "status.json", {"status": "ready", "route": "exercise", "lab": "lab"})
    write_json(root / "muse/status.json", {"status": "proposed"})
    write_json(root / "muse/advice/context.json", {"observations": [{"id": "source:frame:0"}], "objects": [{"id": "person"}]})
    write_json(root / "muse/advice/hypotheses.json", {"source": "model_proposal", "accepted": False,
        "advice": {"hypotheses": [{"status": "hypothesis", "concept": "articulation",
                     "evidence_ids": ["source:frame:0"], "object_ids": ["person"]}]}})
    if queue:
        (root / "muse/repairs").mkdir()
        write_json(root / "muse/repairs/status.json", {"status": "queued"})
    return root


def completed_run(out):
    out.mkdir()
    report = {"profile": "portal-controller-repair-comparison-v1", "status": "complete", "applied": False,
        "physical_validation": "unverified", "accepted_in_simulation": False, "selected_candidate": None,
        "candidates": [{"id": "feedback-1", "accepted_in_simulation": False, "reason": "insufficient improvement"}],
        "seconds": .1, "control_steps": 12}
    write_json(out / "report.json", report)
    write_json(out / "manifest.json", {"status": "complete", "files": {"report.json": digest(out / "report.json")}})
    return report


def test_adapter_runs_once_preserves_project_and_reads_only_small_report(tmp_path, monkeypatch):
    root = job(tmp_path / "project")
    original = {name: (root / name).read_bytes() for name in ("status.json", "muse/status.json", "muse/advice/hypotheses.json")}
    calls = []
    def compare(source, out, *, max_seconds):
        assert source == root and out == root / "muse/repairs/run" and 0 < max_seconds <= 60
        calls.append(out)
        return completed_run(out)
    monkeypatch.setattr("weft.portal_repairs.run_review_repairs", compare)
    assert jobs.execute_repair_job(root) == 0 and len(calls) == 1
    status = json.loads((root / "muse/repairs/status.json").read_bytes())
    assert status["status"] == "complete" and status["applied"] is False
    assert status["candidates_tested"] == 1 and status["accepted_in_simulation"] is False
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert jobs.execute_repair_job(root) == 0 and len(calls) == 1
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert all((root / name).read_bytes() == raw for name, raw in original.items())
    assert jobs.read_small_report(root)["selected_candidate"] is None
    assert jobs.availability(root)["available"] is False


@pytest.mark.parametrize("change", ["missing_review", "no_hypothesis", "unknown", "support", "objects", "foreign_evidence"])
def test_availability_is_bounded_no_compute_no_keys_and_has_reason(tmp_path, monkeypatch, change):
    root = job(tmp_path / "project", queue=False)
    monkeypatch.setattr("weft.portal_repairs.run_review_repairs", lambda *_a, **_k: pytest.fail("Availability cannot simulate"))
    if change == "missing_review": write_json(root / "muse/status.json", {"status": "failed"})
    elif change == "objects": write_json(root / "status.json", {"status": "ready", "route": "objects"})
    elif change == "support":
        context = json.loads((root / "muse/advice/context.json").read_bytes())
        context["simulation_world"] = {"scene_profile": "shared-world-v4"}
        write_json(root / "muse/advice/context.json", context)
    else:
        proposal = json.loads((root / "muse/advice/hypotheses.json").read_bytes())
        if change == "no_hypothesis": proposal["advice"]["hypotheses"] = []
        elif change == "unknown": proposal["advice"]["hypotheses"][0]["status"] = "unknown"
        else: proposal["advice"]["hypotheses"][0]["evidence_ids"] = ["foreign:frame:0"]
        write_json(root / "muse/advice/hypotheses.json", proposal)
    result = jobs.availability(root)
    assert result["available"] is False and result["reason"]
    assert result["estimated_external_api_usd"] == 0 and result["requires_new_ai_request"] is False


def test_eligible_saved_review_needs_no_new_provider_allowance(tmp_path):
    root = job(tmp_path / "project", queue=False)
    result = jobs.availability(root)
    assert result["available"] is True and result["maximum_candidates"] == 3 and result["maximum_seconds"] == 60


@pytest.mark.parametrize("failure", [TimeoutError, ValueError])
def test_failure_preserves_partial_run_and_never_retries(tmp_path, monkeypatch, failure):
    root = job(tmp_path / "project")
    calls = []
    def interrupted(source, out, **kwargs):
        calls.append(out)
        out.mkdir()
        (out / "partial.json").write_bytes(b"retained partial actions")
        raise failure("fixture stopped")
    monkeypatch.setattr("weft.portal_repairs.run_review_repairs", interrupted)
    assert jobs.execute_repair_job(root) == 1
    partial = root / "muse/repairs/run/partial.json"
    assert partial.read_bytes() == b"retained partial actions"
    assert jobs.execute_repair_job(root) == 1 and len(calls) == 1
    assert partial.read_bytes() == b"retained partial actions"
    with pytest.raises(ValueError, match="not complete"): jobs.read_small_report(root)


@pytest.mark.parametrize("has_run", [False, True])
def test_stale_unknown_attempt_is_interrupted_not_reexecuted(tmp_path, monkeypatch, has_run):
    root = job(tmp_path / "project")
    work = root / "muse/repairs"
    write_json(work / "attempt.json", {"pid": -1})
    if has_run:
        (work / "run").mkdir()
        (work / "run/partial.json").write_bytes(b"unknown termination")
    monkeypatch.setattr("weft.portal_repairs.run_review_repairs", lambda *_a, **_k: pytest.fail("No automatic retry"))
    assert jobs.execute_repair_job(root) == 1
    assert json.loads((work / "status.json").read_bytes())["status"] == "interrupted"


def test_sealed_terminal_run_recovers_only_lifecycle_metadata(tmp_path, monkeypatch):
    root = job(tmp_path / "project")
    work = root / "muse/repairs"
    completed_run(work / "run")
    before = {p.name: p.read_bytes() for p in (work / "run").iterdir()}
    monkeypatch.setattr("weft.portal_repairs.run_review_repairs", lambda *_a, **_k: pytest.fail("No repeated comparison"))
    assert jobs.execute_repair_job(root) == 0
    assert json.loads((work / "status.json").read_bytes())["lifecycle_recovered_from_sealed_run"] is True
    assert before == {p.name: p.read_bytes() for p in (work / "run").iterdir()}


def test_live_owner_prevents_duplicate_work_or_status_changes(tmp_path, monkeypatch):
    root = job(tmp_path / "project")
    work = root / "muse/repairs"
    previous = (work / "status.json").read_bytes()
    monkeypatch.setattr("weft.portal_repairs.run_review_repairs", lambda *_a, **_k: pytest.fail("Duplicate owner cannot simulate"))
    with (work / "worker.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert jobs.execute_repair_job(root) == 0
    assert (work / "status.json").read_bytes() == previous


def test_cancel_before_start_leaves_source_and_no_run(tmp_path, monkeypatch):
    root = job(tmp_path / "project")
    work = root / "muse/repairs"
    write_json(work / "cancel-request.json", {})
    monkeypatch.setattr("weft.portal_repairs.run_review_repairs", lambda *_a, **_k: pytest.fail("Cancelled job cannot simulate"))
    assert jobs.execute_repair_job(root) == 1
    assert json.loads((work / "status.json").read_bytes())["status"] == "cancelled"
    assert not (work / "run").exists()


def test_summary_rejects_report_tampering_without_loading_traces(tmp_path, monkeypatch):
    root = job(tmp_path / "project")
    monkeypatch.setattr("weft.portal_repairs.run_review_repairs", lambda source, out, **kwargs: completed_run(out))
    assert jobs.execute_repair_job(root) == 0
    report = root / "muse/repairs/run/report.json"
    value = json.loads(report.read_bytes());value["accepted_in_simulation"] = True
    write_json(report, value)
    with pytest.raises(ValueError, match="manifest"): jobs.read_small_report(root)


def test_worker_dispatches_repairs_with_own_receipt_and_cpu_allowance(tmp_path, monkeypatch):
    from weft import portal_worker
    root = job(tmp_path / "project")
    calls, limits = [], []
    monkeypatch.setattr(sys, "argv", ["worker", str(root), str(tmp_path), "repairs"])
    monkeypatch.setattr(portal_worker.resource, "setrlimit", lambda key, value: limits.append((key, value)))
    monkeypatch.setattr(portal_worker.threading.Thread, "start", lambda self: None)
    def execute(folder, *, project):
        assert project == tmp_path
        calls.append(folder)
        return 0
    monkeypatch.setattr(jobs, "execute_repair_job", execute)
    assert portal_worker.main() == 0 and calls == [root]
    assert (portal_worker.resource.RLIMIT_CPU, (65, 65)) in limits
    assert (root / "muse/repairs/worker-receipt.json").is_file()
    assert not (root / "worker-receipt.json").exists()
