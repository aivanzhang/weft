import json

import pytest

from weft.io import digest, write_json
from weft.meta_access import access_status, configure_access
from weft.portal_review import construction_input, execute_review
from weft import scene_advisor
from test_scene_advisor import run, reply, proposal


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


def test_support_child_review_is_explicitly_original_body_only(job, tmp_path, monkeypatch):
    source = job / "body_source"
    (job / "planar/run").rename(source)
    evidence = json.loads((source / "evidence.json").read_text())
    evidence["reference"] = evidence.pop("generation")
    for row in evidence["reference"]:
        row["object_id"] = "person"
    write_json(source / "evidence.json", evidence)
    write_json(source / "human.json", {"profile": "exercise-v3", "assumptions": ["Original body assumptions"]})
    write_json(source / "manifest.json", {"files": {
        p.name: digest(p) for p in source.iterdir() if p.name != "manifest.json"}})
    config = json.loads((job / "job.json").read_text())
    write_json(job / "job.json", {**config, "kind": "support_experiment", "body_source_directory": "body_source"})
    write_json(job / "status.json", {"status": "ready", "route": "exercise"})
    write_json(job / "support-priors.json", {"marker": "DO_NOT_EXPOSE_CHILD_SUPPORT"})
    _, binding = construction_input(job)
    assert binding["review_scope"] == "original_body_only"
    assert "Added supports, load windows and child-world control settings are not reviewed" in binding["scope"]
    assert set(binding["input_sha256"]) == {"evidence.json", "human.json", "source.mp4"}
    before = {p.name: digest(p) for p in source.iterdir()}
    sent = []
    def respond(request):
        sent.append(request)
        assert b"DO_NOT_EXPOSE" not in request.data
        assert "original body evidence only" in json.loads((job / "muse/status.json").read_text())["message"]
        context = scene_advisor.prepare_context(source)
        advice = proposal(context)
        advice["hypotheses"][0]["object_ids"] = ["person"]
        return reply(context, choices=[{"finish_reason": "stop", "message": {"content": json.dumps(advice)}}])
    monkeypatch.setattr(scene_advisor, "_open", respond)
    assert execute_review(job, project=tmp_path) == 0
    assert len(sent) == 1
    status = json.loads((job / "muse/status.json").read_text())
    assert "were not reviewed" in status["message"]
    assert "Original-body review ready" in status["message"]
    assert json.loads((job / "muse/binding.json").read_text()) == binding
    assert before == {p.name: digest(p) for p in source.iterdir()}
