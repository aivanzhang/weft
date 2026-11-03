"""Saved moving lifecycle; analytic source-admission stubs are not real-video evidence."""

from copy import deepcopy
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from weft import portal_moving_equipment as module
from weft.io import canonical_hash, write_json
from weft.portal import Portal


def settings():
    return {"schema_version": 1, "profile": "moving-depth-family-priors-v1", "camera": {"distance_model_units": 2},
        "depth_domain_model_units": [.1, 4.], "shape": {"method": "sphere_from_median_image_box_diagonal"},
        "mass_kg": 1., "friction": [.5, .005, .0001], "initialization": {"method": "first_initialized_box_zero_velocity"}}


@pytest.fixture
def upload(tmp_path, monkeypatch):
    import weft.portal as portal_module
    monkeypatch.setattr(portal_module, "muse_availability", lambda _: {"available": False})
    portal = Portal(tmp_path / "portal", project=tmp_path, start=False)
    video = b"Analytical bytes; full body admission is stubbed in this fixture."
    job = portal.upload(BytesIO(video), len(video), name="fixture.mp4", mode="exercise", ai=False,
        request_id="1" * 32, exercise_inventory="equipment_or_obstacles", exercise_output="simulation")
    root = portal.root / job["id"]
    write_json(root / "status.json", {"status": "needs_input", "route": "exercise", "events": []})
    (root / "perception").mkdir()
    (root / "equipment").mkdir()
    write_json(root / "perception/status.json", {"status": "evidence_ready"})
    write_json(root / "perception/prepared-source.json", {"source_sha256": job["source_sha256"],
        "job_sha256": sha256((root / "job.json").read_bytes()).hexdigest()})
    from weft.portal_equipment_candidates import PROFILE as old_profile
    write_json(root / "equipment/candidates.json", {"schema_version": 1, "profile": old_profile, "candidates": []})
    write_json(root / "equipment/status.json", {"status": "complete", "message": "Earlier stationary construction refused",
        "report_sha256": sha256((root / "equipment/candidates.json").read_bytes()).hexdigest()})
    yield portal, root, video
    portal.close()


def fake_family():
    factor = {"object_id": "raw-0", "status": "feasible_intervals", "reasons": [],
        "depth_intervals": {"certified_intervals": [[.2, .4], [2., 3.]], "possible_intervals": [[.19, .41], [1.99, 3.01]], "boundary_ambiguity": True}}
    doc = {"schema_version": 1, "profile": "moving-depth-family-v1", "settings": settings(),
        "identity_family": {"assignment_count": 7}, "factors": [factor, {"object_id": "raw-1", "status": "unsupported_input", "reasons": ["missing"], "depth_intervals": None}],
        "status": "ready_for_explicit_selection"}
    return SimpleNamespace(document=doc, document_bytes=json.dumps(doc).encode())


def fake_preparation(upload, monkeypatch):
    portal, root, _ = upload
    import weft.moving_depth_family as family
    import weft.moving_equipment_inputs as inputs
    monkeypatch.setattr(module, "_source", lambda *a, **k: (object(), {"settings": settings()}, {"source": "analytic mocked admission"}))
    monkeypatch.setattr(family, "prepare_moving_depth_family", lambda *a, **k: fake_family())
    monkeypatch.setattr(inputs, "verify_moving_equipment_inputs_unchanged", lambda *a, **k: None)
    portal.moving_equipment(root.name, "prepare", settings())
    assert module.execute_job(root, project=portal.project) == 0
    return portal, root


def choice(root, depth=.3):
    return {"schema_version": 1, "profile": "moving-depth-selection-v1",
        "family_sha256": module.summary(root)["family"]["family_sha256"],
        "choices": [{"equipment": [{"object_id": "raw-0", "interval_index": 0, "depth_model_units": depth}]}]}


def test_prepare_retains_parent_full_family_and_explicit_no_default_depth(upload, monkeypatch):
    portal, root, _ = upload
    old = {p: (root / p).read_bytes() for p in ("job.json", "status.json", "perception/status.json", "equipment/status.json")}
    assert portal.summary(root.name)["moving_equipment"]["available"]
    fake_preparation(upload, monkeypatch)
    state = portal.summary(root.name)
    assert state["status"] == "needs_input" and state["sandbox"] is None
    moving = state["moving_equipment"]
    assert moving["status"] == "awaiting_choices" and moving["family"]["identity_family"]["assignment_count"] == 7
    assert len(moving["family"]["factors"][0]["certified_intervals"]) == 2
    assert moving["family"]["factors"][1]["status"] == "unsupported_input"
    assert not (root / "moving-equipment/build-request.json").exists()
    portal.moving_equipment(root.name, "prepare", settings())
    assert old == {p: (root / p).read_bytes() for p in old}
    assert portal.artifact(root.name, "moving-equipment/family.json").is_file()
    with pytest.raises(ValueError, match="approved receipt"):
        portal.artifact(root.name, "moving-equipment/implementation.py")


def test_choices_pin_complete_family_and_cannot_be_replaced(upload, monkeypatch):
    portal, root = fake_preparation(upload, monkeypatch)
    value = choice(root)
    changed = deepcopy(value); changed["family_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="differs"):
        portal.moving_equipment(root.name, "build", changed)
    many = deepcopy(value); many["choices"] *= 4
    with pytest.raises(ValueError, match="one to three"):
        portal.moving_equipment(root.name, "build", many)
    portal.moving_equipment(root.name, "build", value)
    raw = (root / "moving-equipment/build-request.json").read_bytes()
    portal.moving_equipment(root.name, "build", value)
    with pytest.raises(ValueError, match="immutable"):
        portal.moving_equipment(root.name, "build", choice(root, .35))
    assert (root / "moving-equipment/build-request.json").read_bytes() == raw
    assert portal._reserved_growth() == (256 << 20) + module.RECEIPT_BYTES


@pytest.mark.parametrize("damage", ["selected", "active", "descriptor", "parent"])
def test_ineligible_saved_sources_do_not_queue(upload, damage):
    portal, root, _ = upload
    if damage == "selected":
        write_json(root / "equipment/selection.json", {})
    elif damage == "active":
        write_json(root / "equipment/status.json", {"status": "running"})
    elif damage == "descriptor":
        write_json(root / "perception/prepared-source.json", {"source_sha256": "f" * 64})
    else:
        write_json(root / "status.json", {"status": "ready", "lab": "old", "route": "exercise"})
    assert not portal.summary(root.name)["moving_equipment"]["available"]
    with pytest.raises(ValueError):
        portal.moving_equipment(root.name, "prepare", settings())
    assert not (root / "moving-equipment").exists()


def test_restarts_cancel_and_source_failure_never_retry(upload, monkeypatch):
    portal, root, _ = upload
    original = (root / "status.json").read_bytes()
    portal.moving_equipment(root.name, "prepare", settings())
    portal.cancel(root.name)
    assert module.read_status(root)["status"] == "cancelled"
    assert module.execute_job(root, project=portal.project) == 1
    assert (root / "status.json").read_bytes() == original
    # A queued phase on restart is interrupted, not re-executed.
    write_json(root / "moving-equipment/status.json", {"status": "queued", "phase": "prepare"})
    other = portal.root.parent / "restart-copy"
    import shutil
    shutil.copytree(portal.root, other)
    resumed = Portal(other, project=portal.project, start=False)
    try:
        assert resumed.summary(root.name)["moving_equipment"]["status"] == "interrupted"
    finally:
        resumed.close()


def test_worker_starts_before_validation_and_does_not_fallback(upload, monkeypatch):
    portal, root, _ = upload
    portal.moving_equipment(root.name, "prepare", settings())
    def refuse(*args, **kwargs):
        assert (root / "moving-equipment/prepare-start.json").is_file()
        raise ValueError("changed exact source")
    monkeypatch.setattr(module, "_source", refuse)
    assert module.execute_job(root, project=portal.project) == 1
    assert "changed exact source" in module.read_status(root)["error"]
    assert not (root / "moving-equipment/family.json").exists()


def test_scoped_storage_plan_and_no_provider_environment(upload, monkeypatch):
    portal, root, _ = upload
    portal.moving_equipment(root.name, "prepare", settings())
    assert module.storage_limit(root) == 1536 << 20
    for key in ("OPENAI_API_KEY", "META_API_KEY", "MODEL_API_KEY", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(key, "must-not-enter-worker")
    captured = {}
    class Process:
        returncode = 0
        def __init__(self, command, **kwargs):
            self.stdout = BytesIO(b"analytic worker output")
            captured.update(command=command, **kwargs)
            write_json(root / "moving-equipment/status.json", {"status": "complete", "phase": "prepare"})
        def poll(self): return 0
        def wait(self, **kwargs): return 0
    import weft.portal as portal_module
    monkeypatch.setattr(portal_module.subprocess, "Popen", Process)
    portal._run(root.name, moving=True)
    assert captured["command"][-1] == "moving"
    assert not any("KEY" in name or name.startswith("AWS") for name in captured["env"])
    assert module.read_status(root)["status"] == "complete"
    write_json(root / "moving-equipment/storage-plan.json", {"max_job_bytes": 9 << 30})
    with pytest.raises(ValueError, match="closed workflow"):
        module.storage_limit(root)


def test_storage_growth_not_hidden_by_normal_job_reservation(upload, monkeypatch):
    portal, root, _ = upload
    import weft.portal as portal_module
    monkeypatch.setattr(portal, "_stored_bytes", lambda: portal_module.MAX_STORAGE - module.RECEIPT_BYTES + 1)
    with pytest.raises(ValueError, match="storage allowance"):
        portal.moving_equipment(root.name, "prepare", settings())
    assert not (root / "moving-equipment").exists()


def test_authenticated_moving_handler_and_closed_selected_identity(upload, monkeypatch):
    from weft.portal import PortalHandler
    calls, replies = [], []
    payload = json.dumps(settings()).encode()
    handler = SimpleNamespace(path="/api/projects/" + "a" * 32 + "/moving-equipment/prepare",
        headers={"Content-Length": str(len(payload))}, rfile=BytesIO(payload),
        connection=SimpleNamespace(settimeout=lambda _: None), route_lab=lambda: None, authorized=lambda: True,
        server=SimpleNamespace(portal=SimpleNamespace(moving_equipment=lambda *args: calls.append(args) or {"status": "queued"})),
        reply=lambda value, status=200: replies.append((value, status)))
    PortalHandler.do_POST(handler)
    assert calls == [("a" * 32, "prepare", settings())] and replies[-1][1] == 202
    handler.authorized = lambda: False
    PortalHandler.do_POST(handler)
    assert len(calls) == 1
    handler.authorized = lambda: True; handler.headers = {"Content-Length": str((16 << 10) + 1)}
    PortalHandler.do_POST(handler)
    assert replies[-1][1] == 400 and len(calls) == 1
    portal, root = fake_preparation(upload, monkeypatch)
    write_json(root / "moving-equipment/selection.json", {"candidate_id": "../../outside"})
    with pytest.raises(ValueError, match="closed, contained"):
        module.selected_profile(root, {"moving_candidate_id": "../../outside"})


@pytest.mark.parametrize("member", ["request.json", "build-request.json"])
def test_late_command_mutation_cannot_publish_terminal_success(upload, monkeypatch, member):
    portal, root = fake_preparation(upload, monkeypatch)
    selected = choice(root)
    portal.moving_equipment(root.name, "build", selected)
    import weft.moving_depth_family as family
    import weft.moving_equipment_inputs as inputs
    monkeypatch.setattr(family, "admit_moving_depth_family", lambda *a, **k: fake_family())
    monkeypatch.setattr(family, "materialize_moving_depth_selection", lambda *a, **k: {"candidates": []})
    def mutate(*args, **kwargs):
        path = root / "moving-equipment" / member
        value = json.loads(path.read_bytes())
        if member == "request.json":
            value["settings"]["camera"]["distance_model_units"] = 3
        else:
            value["choices"][0]["equipment"][0]["depth_model_units"] = .35
        write_json(path, value)
    monkeypatch.setattr(inputs, "verify_moving_equipment_inputs_unchanged", mutate)
    assert module.execute_job(root, project=portal.project) == 1
    assert "request or explicit selection changed" in module.read_status(root)["error"]
    assert (root / "status.json").read_bytes() == (root / "moving-equipment/parent-status.json").read_bytes()


def test_actual_analytic_family_constructor_selected_portal_recording(upload, monkeypatch):
    """Real geometry/world/lab/replay; only original video/body qualification is stubbed."""
    portal, root, video = upload
    import test_root_motion
    from test_moving_equipment_proposal import packet
    import weft.moving_equipment_inputs as admission
    monkeypatch.setattr(test_root_motion, "HASH", sha256(video).hexdigest())
    inputs, _ = packet(monkeypatch, raw_tracks=3, image_shift=(300, 190))
    monkeypatch.setattr(module, "_source", lambda *a, **k: (inputs, {"settings": settings()}, {"scope": "analytic source admission stub"}))
    monkeypatch.setattr(admission, "verify_moving_equipment_inputs_unchanged", lambda *a, **k: None)
    preserved = {name: (root / name).read_bytes() for name in ("job.json", "perception/status.json", "equipment/candidates.json", "equipment/status.json")}
    portal.moving_equipment(root.name, "prepare", settings())
    assert module.execute_job(root, project=portal.project) == 0, module.read_status(root)
    family = module.summary(root)["family"]
    assert family["identity_family"]["assignment_count"] == 7
    factor = next(f for f in family["factors"] if f["certified_intervals"])
    low, high = factor["certified_intervals"][0]
    # This test authors one midpoint explicitly; production never chooses a representative.
    selected = {"schema_version": 1, "profile": "moving-depth-selection-v1", "family_sha256": family["family_sha256"],
        "choices": [{"equipment": [{"object_id": factor["object_id"], "interval_index": 0, "depth_model_units": low + (high-low)/2}]}]}
    portal.moving_equipment(root.name, "build", selected)
    assert module.execute_job(root, project=portal.project) == 0, module.read_status(root)
    report = module.summary(root)["report"]
    assert report["candidates"][0]["selectable"], report
    chosen = portal.moving_equipment(root.name, "select", {"id": "candidate-1"})
    assert chosen["control_profile"] == "shared_world_exploration"
    assert not chosen["support_experiment_available"] and chosen["review_unavailable_reason"]
    for action in ("review", "checks", "repairs"):
        with pytest.raises(ValueError, match="Manual exploration"):
            portal._require_review_body(root.name, action=action)
    assert portal.moving_equipment(root.name, "select", {"id": "candidate-1"})["moving_candidate_id"] == "candidate-1"
    lab = portal.lab(root.name)
    lab._future.result(timeout=60)
    summary = lab.summary()
    assert summary["control_profile"] == "shared_world_exploration", summary
    saved_bundle = json.loads((root / chosen["lab"] / "run/moving-proposals.json").read_bytes())
    assert canonical_hash(saved_bundle["depth_family"]) == family["family_sha256"]
    assert saved_bundle["depth_selection"] == selected
    assert len(saved_bundle["depth_family"]["factors"]) == 3
    lab.control("/api/reset", {"seed": 1729})
    lab.control("/api/record/start", {})
    action = [0.] * len(summary["scene"]["action_contract"]["names"])
    for i in range(6):
        action[-3] = i * .01
        lab.control("/api/step", {"action": list(action), "steps": 1})
    lab.control("/api/record/stop", {})
    receipt = lab.export_future.result(timeout=30)
    assert receipt["action_replay"]["exact_state_replay"] and receipt["action_replay"]["steps"] == 6
    assert receipt["recording_evidence"]["status"] == "available", receipt
    assert preserved == {name: (root / name).read_bytes() for name in preserved}
    with pytest.raises(ValueError, match="opened sandbox"):
        portal.moving_equipment(root.name, "select", {"id": "candidate-2"})
