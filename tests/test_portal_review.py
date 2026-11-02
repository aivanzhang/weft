import json

import pytest

from weft.io import digest, write_json
from weft.meta_access import access_status, configure_access
from weft.portal_review import construction_input, execute_review, review_input
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


def exercise_job(job):
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
    return job


def test_legacy_support_binding_remains_explicitly_original_body_only(job):
    exercise_job(job)
    _, binding = construction_input(job)
    assert binding["review_scope"] == "original_body_only"
    assert "Added supports, load windows and child-world control settings are not reviewed" in binding["scope"]
    assert set(binding["input_sha256"]) == {"evidence.json", "human.json", "source.mp4"}


def test_new_world_review_sends_authored_geometry_not_results(job, tmp_path, monkeypatch):
    from test_portal_scene_context import make_job
    from weft.portal_checks import inputs

    exercise_job(job)
    make_job(job, source_sha256=json.loads((job / "job.json").read_bytes())["source_sha256"])
    source, binding, world = review_input(job)
    before = {p.relative_to(job).as_posix(): digest(p) for p in job.rglob("*") if p.is_file()}
    sent = []
    def respond(request):
        payload = json.loads(request.data)
        context = json.loads(payload["messages"][1]["content"][0]["text"])
        assert context["simulation_world"] == world
        assert "DO_NOT_READ" not in request.data.decode() and "DO_NOT_EXPOSE" not in request.data.decode()
        assert {"world:human", "world:" + world["objects"][0]["id"]} <= {o["id"] for o in context["objects"]}
        advice = proposal(context)
        advice["hypotheses"][0]["object_ids"] = ["world:" + world["objects"][0]["id"]]
        sent.append(request)
        return reply(context, choices=[{"finish_reason": "stop", "message": {"content": json.dumps(advice)}}])
    monkeypatch.setattr(scene_advisor, "_open", respond)
    assert execute_review(job, project=tmp_path) == 0
    assert len(sent) == 1
    assert json.loads((job / "muse/status.json").read_bytes())["review_scope"] == "coupled_world_assumptions"
    assert json.loads((job / "muse/binding.json").read_bytes()) == binding
    assert before == {name: digest(job / name) for name in before}
    inputs(job)
    context_path = job / "muse/advice/context.json"
    context = json.loads(context_path.read_bytes())
    context["simulation_world"]["objects"][0]["position"][0] += .1
    write_json(context_path, context)
    with pytest.raises(ValueError, match="Reviewed world context"):
        inputs(job)


def test_world_changed_during_review_retains_charge_but_rejects_result(job, tmp_path, monkeypatch):
    from test_portal_scene_context import make_job

    exercise_job(job)
    make_job(job, source_sha256=json.loads((job / "job.json").read_bytes())["source_sha256"])
    def respond(request):
        context = json.loads(json.loads(request.data)["messages"][1]["content"][0]["text"])
        advice = proposal(context)
        advice["hypotheses"][0]["object_ids"] = ["person"]
        (job / "lab/run/scene.json").write_text('{}')
        return reply(context, choices=[{"finish_reason": "stop", "message": {"content": json.dumps(advice)}}])
    monkeypatch.setattr(scene_advisor, "_open", respond)
    assert execute_review(job, project=tmp_path) == 1
    assert access_status(tmp_path)["committed_usd"] > 0
    assert json.loads((job / "muse/status.json").read_bytes())["status"] == "failed"
