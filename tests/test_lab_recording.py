import json
from pathlib import Path

import pytest

from weft.io import canonical_hash, digest, write_json
from weft.lab import Lab
from weft.recording import export_recording, load_recording, verify_recording
from weft.scene import SceneEnv
from weft.scene_learning import ERROR_EDGES, VELOCITY_EDGES


@pytest.fixture
def ready_lab(tmp_path):
    root = tmp_path / "lab"
    run = root / "run"
    run.mkdir(parents=True)
    scene = {"schema_version": 1, "profile": "planar-rigid-v1", "width": 1.0, "height": 0.6,
             "dt": 1 / 30, "gravity": [0, 0], "damping": 0.9, "restitution": 0.65,
             "objects": [{"id": "actor", "label": "Synthetic actor", "role": "active", "shape": "circle",
                          "position": [0.2, 0.3], "velocity": [0, 0], "radius": 0.035,
                          "color": "#E0B620", "mass": 1, "provenance": {"geometry": "synthetic"}},
                         {"id": "obstacle", "label": "Synthetic obstacle", "role": "static", "shape": "box",
                          "position": [0.5, 0.45], "velocity": [0, 0], "size": [0.15, 0.08],
                          "color": "#567799", "mass": 1, "provenance": {"geometry": "synthetic"}}],
             "controlled_id": "actor", "target": [0.8, 0.3],
             "task": {"position_tolerance": 0.06, "speed_tolerance": 0.15, "max_steps": 150,
                      "acceleration": 2}, "relationships": [],
             "assumptions": ["Synthetic software fixture, not physical reconstruction"]}
    policy = {"schema_version": 1, "algorithm": "tabular_q_learning", "scene_sha256": canonical_hash(scene),
              "error_edges": ERROR_EDGES, "velocity_edges": VELOCITY_EDGES, "q_values": {}}
    write_json(run / "scene.json", scene)
    write_json(run / "policy.json", policy)
    write_json(run / "manifest.json", {"files": {p.name: digest(p) for p in run.iterdir()}})
    write_json(root / "lab-config.json", {"profile": "planar"})
    write_json(root / "status.json", {"status": "ready", "events": []})
    return root


def _run_hashes(root):
    return {p.name: digest(p) for p in (root / "run").iterdir()}


def test_lab_records_authoritative_actions_and_exports_without_mutating_run(ready_lab):
    before = _run_hashes(ready_lab)
    lab = Lab(ready_lab)
    try:
        # Beginning a recording returns to a reproducible reset, not a hidden mid-episode state.
        lab.control("/api/step", {"action": 4, "steps": 3})
        started = lab.control("/api/record/start", {})
        assert started["snapshot"]["steps"] == 0
        for action in (2, 4, 5):
            lab.control("/api/step", {"action": action, "steps": 4})
        reset = lab.control("/api/reset", {"seed": 18})
        assert reset["snapshot"]["steps"] == 0
        lab.control("/api/step", {"action": 3, "steps": 2})
        stopped = lab.control("/api/record/stop", {})
        assert stopped["export"] == "running" and not stopped["recording"]
        result = lab.export_future.result(timeout=10)
        session = ready_lab / "sessions" / stopped["directory"]
        data = load_recording(session)
        assert result["steps"] == 14 and result["resets"] == 1
        assert lab.summary()["export"]["status"] == "ready"
        env = SceneEnv(data["scene"])
        try:
            for event in data["events"]:
                if event["type"] in ("start", "reset"):
                    env.reset(seed=event["seed"], options={"jitter": False})
                else:
                    _, reward, terminated, truncated, _ = env.step(event["action"])
                    assert reward == event["reward"]
                    assert terminated == event["terminated"] and truncated == event["truncated"]
                assert env.snapshot() == event["snapshot"]
        finally:
            env.close()
        assert _run_hashes(ready_lab) == before
    finally:
        lab.close()


def test_lab_resume_preserves_old_sessions_and_can_record_again(ready_lab):
    lab = Lab(ready_lab)
    first = lab.control("/api/record/start", {})["directory"]
    lab.control("/api/step", {"action": 2, "steps": 2})
    lab.control("/api/record/stop", {})
    first_result = lab.export_future.result(timeout=10)
    lab.close()
    first_manifest = first_result["manifest_sha256"]
    resumed = Lab(ready_lab)
    try:
        assert {"name": first, "complete": True} in resumed.summary()["sessions"]
        second = resumed.control("/api/record/start", {})["directory"]
        assert second != first
        resumed.control("/api/step", {"action": 1, "steps": 2})
        resumed.control("/api/record/stop", {})
        resumed.export_future.result(timeout=10)
        assert verify_recording(ready_lab / "sessions" / first)["manifest_sha256"] == first_manifest
        assert len(resumed.summary()["sessions"]) == 2
    finally:
        resumed.close()


def test_lab_shutdown_preserves_active_journal_for_later_export(ready_lab):
    lab = Lab(ready_lab)
    session = lab.control("/api/record/start", {})["directory"]
    lab.control("/api/step", {"action": 2, "steps": 3})
    lab.close()
    root = ready_lab / "sessions" / session
    data = load_recording(root)
    assert data["metadata"]["status"] == "closed"
    assert len(data["events"]) == 4
    assert export_recording(root)["steps"] == 3


def test_recording_limit_stops_before_an_unrecorded_action(ready_lab, monkeypatch):
    monkeypatch.setattr("weft.recording.MAX_EVENTS", 3)
    lab = Lab(ready_lab)
    try:
        session_name = lab.control("/api/record/start", {})["directory"]
        lab.control("/api/step", {"action": 2, "steps": 2})
        before = lab.session.snapshot()
        with pytest.raises(ValueError, match="Recording stopped"):
            lab.control("/api/step", {"action": 2})
        assert lab.session.snapshot() == before
        assert lab.recorder is None
        data = load_recording(ready_lab / "sessions" / session_name)
        assert len(data["events"]) == 3
        assert data["events"][-1]["snapshot"] == before
    finally:
        lab.close()


def test_recording_limit_stops_before_reset_mutates_state(ready_lab, monkeypatch):
    monkeypatch.setattr("weft.recording.MAX_EVENTS", 2)
    lab = Lab(ready_lab)
    try:
        lab.control("/api/record/start", {})
        lab.control("/api/step", {"action": 2})
        before = lab.session.snapshot()
        with pytest.raises(ValueError, match="Recording stopped"):
            lab.control("/api/reset", {"seed": 19})
        assert lab.session.snapshot() == before
        assert lab.recorder is None
    finally:
        lab.close()
