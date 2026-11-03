"""Metadata routing only; the task-child loader separately qualifies source and runtime."""

from io import BytesIO
import sys
from types import ModuleType, SimpleNamespace

import pytest

from weft.io import write_json
from weft.portal import Portal


@pytest.fixture
def placement(tmp_path, monkeypatch):
    import weft.portal as module
    monkeypatch.setattr(module, "muse_availability", lambda _: {"available": False})
    portal = Portal(tmp_path / "portal", project=tmp_path, start=False)
    video = b"Metadata fixture; not a qualified source video"
    job = portal.upload(BytesIO(video), len(video), name="Authored placement.mp4", mode="exercise", ai=False,
        request_id="1" * 32, exercise_inventory="equipment_or_obstacles", exercise_output="simulation")
    root = portal.root / job["id"]
    portal.jobs[job["id"]].update(kind="imported_lab", control_profile="moving_equipment_task")
    write_json(root / "job.json", portal.jobs[job["id"]])
    write_json(root / "status.json", {"status": "ready", "route": "exercise", "lab": "lab", "events": []})
    yield portal, job["id"]
    portal.close()


def test_placement_summary_and_server_guards_do_not_claim_exploration_or_muse(placement):
    portal, identity = placement
    summary = portal.summary(identity)
    assert summary["control_profile"] == "moving_equipment_task"
    assert "exploration" not in summary
    assert not summary["review_attachment"]["available"]
    assert not summary["support_experiment_available"]
    assert "placement" in summary["review_unavailable_reason"]
    for action in ("review", "checks", "repairs"):
        with pytest.raises(ValueError, match="placement tasks"):
            portal._require_review_body(identity, action=action)


def test_placement_uses_its_background_loader_not_legacy_lab(placement, monkeypatch):
    portal, identity = placement
    calls = []
    module = ModuleType("weft.moving_equipment_compatibility_lab")
    def load(root, *, project):
        calls.append((root, project))
        return SimpleNamespace(close=lambda: None)
    module.LoadingCompatibleTaskLab = load
    monkeypatch.setitem(sys.modules, module.__name__, module)
    first = portal.lab(identity)
    assert portal.lab(identity) is first
    assert calls == [(portal.root / identity / "lab", portal.project)]
    assert first.token == portal.token


def test_spatial_import_routes_off_thread_and_keeps_review_unavailable(placement, monkeypatch):
    portal, identity = placement
    portal.jobs[identity].update(control_profile="spatial_rigid", mode="objects", source_scope={"modeled_frames": 21})
    calls = []
    module = ModuleType("weft.portal_spatial_lab")
    def load(root, *, project):
        calls.append((root, project)); return SimpleNamespace(close=lambda: None)
    module.LoadingSpatialPortalLab = load
    monkeypatch.setitem(sys.modules, module.__name__, module)
    first = portal.lab(identity)
    assert portal.lab(identity) is first and len(calls) == 1
    summary = portal.summary(identity)
    assert summary["control_profile"] == "spatial_rigid" and summary["source_scope"] == {"modeled_frames": 21}
    assert "rigid worlds" in summary["review_unavailable_reason"]
    for action in ("review", "checks", "repairs"):
        with pytest.raises(ValueError, match="rigid worlds"):
            portal._require_review_body(identity, action=action)
