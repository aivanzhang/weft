import json

import pytest

from weft.io import digest, write_json
from weft.meta_access import access_status, configure_access
from weft.portal_review import construction_input, execute_review
from weft import scene_advisor
from test_scene_advisor import run, reply


@pytest.fixture
def job(run, tmp_path, monkeypatch):
    root = tmp_path / "job"
    (root / "planar").mkdir(parents=True)
    source = root / "planar/run"
    run.rename(source)
    write_json(source / "manifest.json", {"files": {p.name: digest(p) for p in source.iterdir()}})
    write_json(root / "job.json", {"source_file": "source.mp4", "source_sha256": digest(source / "source.mp4")})
    write_json(root / "status.json", {"status": "ready", "route": "objects"})
    (root / "muse").mkdir()
    configure_access(tmp_path, total_budget_usd=5)
    monkeypatch.setenv("META_API_KEY", "fixture-secret")
    return root


def test_review_preserves_world_and_excludes_scorer(job, tmp_path, monkeypatch):
    source, binding = construction_input(job)
    before = {p.name: digest(p) for p in source.iterdir()}
    sent = []
    def respond(request):
        sent.append(request)
        assert b"DO_NOT_EXPOSE" not in request.data
        return reply(scene_advisor.prepare_context(source))
    monkeypatch.setattr(scene_advisor, "_open", respond)
    original_status = (job / "status.json").read_bytes()
    assert execute_review(job, project=tmp_path) == 0
    assert len(sent) == 1
    assert before == {p.name: digest(p) for p in source.iterdir()}
    assert (job / "status.json").read_bytes() == original_status
    assert json.loads((job / "muse/status.json").read_text())["status"] == "proposed"
    assert json.loads((job / "muse/advice/hypotheses.json").read_text())["accepted"] is False
    assert json.loads((job / "muse/binding.json").read_text()) == binding
    assert access_status(tmp_path)["unsettled_count"] == 0
    assert all("fixture-secret" not in p.read_text() for p in (job / "muse").rglob("*.json"))
    # A repeated worker cannot create a second request or overwrite prior advice.
    assert execute_review(job, project=tmp_path) == 0
    assert len(sent) == 1


@pytest.mark.parametrize("mutation", ["source", "evidence", "symlink", "status"])
def test_modified_input_never_calls_provider(job, tmp_path, monkeypatch, mutation):
    source = job / "planar/run"
    if mutation == "source":
        (source / "source.mp4").write_bytes(b"changed")
    elif mutation == "evidence":
        (source / "evidence.json").write_text("{}")
    elif mutation == "symlink":
        target = source / "manifest.json"
        target.rename(tmp_path / "manifest.json")
        target.symlink_to(tmp_path / "manifest.json")
    else:
        write_json(job / "status.json", {"status": "failed", "route": "objects"})
    monkeypatch.setattr(scene_advisor, "_open", lambda *_: pytest.fail("No request allowed"))
    assert execute_review(job, project=tmp_path) == 1
    assert access_status(tmp_path)["reservation_count"] == 0


def test_network_failure_keeps_unknown_charge_and_ready_world(job, tmp_path, monkeypatch):
    monkeypatch.setattr(scene_advisor, "_open", lambda *_: (_ for _ in ()).throw(TimeoutError()))
    assert execute_review(job, project=tmp_path) == 1
    assert access_status(tmp_path)["unsettled_count"] == 1
    assert json.loads((job / "status.json").read_text())["status"] == "ready"


def test_cancel_before_request_spends_nothing(job, tmp_path, monkeypatch):
    write_json(job / "muse/cancel-request.json", {})
    monkeypatch.setattr(scene_advisor, "_open", lambda *_: pytest.fail("No request allowed"))
    assert execute_review(job, project=tmp_path) == 0
    assert access_status(tmp_path)["reservation_count"] == 0
    assert json.loads((job / "muse/status.json").read_text())["status"] == "cancelled"


def test_wrong_provider_model_keeps_usage_but_rejects_proposal(job, tmp_path, monkeypatch):
    source, _ = construction_input(job)
    monkeypatch.setattr(scene_advisor, "_open", lambda _: reply(scene_advisor.prepare_context(source), model="unexpected"))
    assert execute_review(job, project=tmp_path) == 1
    assert not (job / "muse/advice/hypotheses.json").exists()
    budget = access_status(tmp_path)
    assert budget["committed_usd"] > 0 and budget["unsettled_count"] == 0
