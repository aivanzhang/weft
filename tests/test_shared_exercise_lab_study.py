"""Lab acceptance preserves complete task attempts, exact input custody and failed outcomes."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


path = Path(__file__).resolve().parents[1] / "scripts/studies/check_shared_exercise_lab.py"
spec = importlib.util.spec_from_file_location("exercise_lab_study", path)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def snapshot(*, terminal=False, completed=False, failure=None):
    return {"terminated": terminal, "truncated": False,
            "resolved_reset_options": {"jitter": False, "jitter_scale": 0.0},
            "task_metrics": {"completed_interval": completed, "failure": failure,
                             "success": completed and failure is None, "failed_gates": []},
            "task_sha256": "task", "engine_state": {"normalized_action": [0.] * 25}}


def baseline(steps, *, terminal=False, completed=False, failure=None):
    return {"scene": {"profile": "shared-exercise-recording-v1",
                      "exercise_task": {"interval": {"horizon_steps": 3}}},
            "events": [{"type": "start", "snapshot": snapshot()}] + [
                {"type": "step", "snapshot": snapshot(terminal=terminal and index == steps - 1,
                    completed=completed and index == steps - 1, failure=failure if index == steps - 1 else None)}
                for index in range(steps)]}


@pytest.mark.parametrize("recording", [baseline(0), baseline(2), baseline(2, terminal=True),
                                        baseline(2, terminal=True, completed=True)])
def test_baseline_requires_complete_interval_or_explicit_failure(recording):
    with pytest.raises(ValueError):
        study.validate_baseline_episode(recording)


@pytest.mark.parametrize("recording", [baseline(3, terminal=True, completed=True),
    baseline(1, terminal=True, failure="human_velocity_envelope_exceeded")])
def test_completed_and_failed_attempts_keep_their_actual_outcome(recording):
    original = deepcopy(recording)
    study.validate_baseline_episode(recording)
    assert recording == original


@pytest.mark.parametrize("mutation", ["reset", "jitter", "profile", "horizon"])
def test_baseline_requires_nominal_single_human_episode(mutation):
    recording = baseline(3, terminal=True, completed=True)
    if mutation == "reset":
        recording["events"][1]["type"] = "reset"
    elif mutation == "jitter":
        recording["events"][0]["snapshot"]["resolved_reset_options"]["jitter"] = True
    elif mutation == "profile":
        recording["scene"]["profile"] = "shared-world-recording-v1"
    else:
        recording["scene"]["exercise_task"]["interval"]["horizon_steps"] = 181
    with pytest.raises(ValueError):
        study.validate_baseline_episode(recording)


@pytest.mark.parametrize("sessions", [[], [{"name": "older", "complete": True}],
                                       [{"name": "new", "complete": False}]])
def test_resume_requires_the_exact_new_complete_recording(sessions):
    with pytest.raises(ValueError, match="newly completed"):
        study.validate_resumed_recording({"snapshot": {"task_sha256": "task"}, "sessions": sessions}, "task", "new")


def test_resume_preserves_task_identity():
    restored = {"snapshot": {"task_sha256": "task"}, "sessions": [{"name": "new", "complete": True}]}
    study.validate_resumed_recording(restored, "task", "new")
    with pytest.raises(ValueError, match="exact task"):
        study.validate_resumed_recording(restored, "other-task", "new")


def package(root, files):
    root.mkdir()
    hashes = {}
    for name, value in files.items():
        raw = json.dumps(value).encode()
        (root / name).write_bytes(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    study.write_json(root / "manifest.json", {"files": hashes})
    return hashes


@pytest.mark.parametrize("status", ["ready", "trained", "evaluated"])
def test_manifest_accepts_only_supported_explicit_terminal_status(tmp_path, status):
    root = tmp_path / "package"
    hashes = package(root, {"scene.json": {}})
    study.write_json(root / "manifest.json", {"files": hashes, "status": status})
    assert study._manifest(root) == (study.digest(root / "manifest.json"), hashes)


def test_manifest_preserves_historical_absent_status(tmp_path):
    root = tmp_path / "package"
    hashes = package(root, {"scene.json": {}})
    assert study._manifest(root) == (study.digest(root / "manifest.json"), hashes)


@pytest.mark.parametrize("status", ["running", "queued", "failed", "cancelled", "rejected",
                                     "unsupported_input", "unknown", None, True, {}])
def test_manifest_rejects_unfinished_failed_and_invalid_status(tmp_path, status):
    root = tmp_path / "package"
    hashes = package(root, {"scene.json": {}})
    study.write_json(root / "manifest.json", {"files": hashes, "status": status})
    with pytest.raises(ValueError, match="supported terminal status"):
        study._manifest(root)


def test_task_selection_preserves_sealed_default_and_explicit_override(tmp_path):
    source = tmp_path / "source"
    hashes = package(source, {"exercise-task.json": {"name": "sealed"}})
    assert study._selected_task(source, hashes, None) == ({"name": "sealed"}, hashes["exercise-task.json"], "sealed_input")
    explicit = tmp_path / "explicit.json"
    explicit.write_text('{"name": "explicit"}')
    assert study._selected_task(source, hashes, explicit) == ({"name": "explicit"}, study.digest(explicit), "explicit_file")
    assert study._selected_task(source, {}, None) == (None, None, "default_authored_task")
    (source / "exercise-task.json").write_text('{"name": "changed"}')
    with pytest.raises(ValueError, match="SHA256"):
        study._selected_task(source, hashes, None)


def test_changed_baseline_seal_is_rejected_before_preparing_lab(tmp_path, monkeypatch):
    source, previous, out = tmp_path / "source", tmp_path / "baseline", tmp_path / "out"
    package(source, {"scene.json": {}})
    package(previous, {"events.jsonl": {"old": True}})
    (previous / "events.jsonl").write_text("changed")
    monkeypatch.setattr(sys, "argv", [str(path), "--run", str(source), "--baseline-recording", str(previous), "--out", str(out)])
    monkeypatch.setattr(study, "prepare_shared_exercise_lab", lambda *args, **kwargs: pytest.fail("Prepared an unsealed baseline"))
    with pytest.raises(ValueError, match="SHA256"):
        study.main()
    assert not out.exists()


def test_elapsed_deadline_cannot_publish_verified_output(tmp_path):
    (tmp_path / "artifact.json").write_text("{}")
    with pytest.raises(TimeoutError):
        study._seal(tmp_path, study.time.monotonic() - 1)
    with pytest.raises(TimeoutError):
        study._remaining(study.time.monotonic() - 1)
    assert not (tmp_path / "manifest.json").exists()


@pytest.mark.parametrize("limit", ["bytes", "files"])
def test_output_bounds_publish_only_failed_receipt(tmp_path, monkeypatch, limit):
    (tmp_path / "artifact.json").write_text("{}")
    monkeypatch.setattr(study, "MAX_OUTPUT_BYTES" if limit == "bytes" else "MAX_OUTPUT_FILES", 1)
    with pytest.raises(ValueError, match="bound") as error:
        study._seal(tmp_path, study.time.monotonic() + 1)
    assert not (tmp_path / "manifest.json").exists()
    study._failure_receipt(tmp_path, error.value)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["custody"] == "failure_receipt_only"
    assert manifest["unsealed_outputs_preserved"] is True
    assert manifest["files"] == {"failure.json": study.digest(tmp_path / "failure.json")}
    assert (tmp_path / "artifact.json").read_text() == "{}"


@pytest.mark.parametrize("early_failure", [False, True])
def test_generic_driver_records_own_complete_or_failed_attempt(tmp_path, monkeypatch, early_failure):
    source, out = tmp_path / "source", tmp_path / "out"
    scene, task = {"objects": [{"id": "arbitrary-equipment"}]}, {"interval": {"horizon_steps": 6}}
    package(source, {"scene.json": scene, "exercise-task.json": task})
    events, completed = [], []
    limit = 1 if early_failure else 6

    class Lab:
        def __init__(self, root):
            self.root, self.steps, self.recording = root, 0, False
            self.env = SimpleNamespace(action_count=19)

        def summary(self):
            return {"control_profile": "shared_exercise", "learned_policy_available": False,
                    "scene": scene, "exercise_task": task, "snapshot": self.snapshot(), "sessions": completed}

        def snapshot(self):
            done = self.steps >= limit
            return snapshot(terminal=done, completed=done and not early_failure,
                            failure="human_velocity_envelope_exceeded" if done and early_failure else None)

        def load_session(self):
            return self

        def inspection(self):
            self.inspection_future = SimpleNamespace(result=lambda **kwargs: {})

        def close(self):
            pass

        def control(self, endpoint, payload):
            if endpoint == "/api/actor" or payload.get("action") == 0 or payload.get("policy"):
                raise ValueError("Human-only controls")
            if endpoint in ("/api/reset", "/api/record/start"):
                self.steps = 0
                self.recording = True
                events.append({"type": "start" if endpoint.endswith("start") else "reset", "snapshot": self.snapshot()})
                return {"snapshot": self.snapshot()}
            if endpoint == "/api/step":
                transitions = []
                for _ in range(payload["steps"]):
                    if self.steps >= limit:
                        break
                    self.steps += 1
                    state = self.snapshot()
                    controller = {"reference": "authored", "coast": "coast"}.get(payload.get("controller"), "manual")
                    event = {"type": "step", "controller": controller, "snapshot": state,
                             "terminated": state["terminated"], "truncated": state["truncated"]}
                    transitions.append(event)
                    events.append(event)
                return {"transitions": transitions}
            assert endpoint == "/api/record/stop"
            completed.append({"name": "exact-new-recording", "complete": True})
            directory = self.root / "sessions" / completed[0]["name"]
            self.export_future = SimpleNamespace(result=lambda **kwargs: {"directory": str(directory), "action_replay": {
                "exact_state_replay": True, "events": len(events), "steps": sum(e["type"] == "step" for e in events)}})
            return {"directory": completed[0]["name"]}

    def prepare(run, root, *, task_path, max_seconds):
        assert run == source and task_path is None and 0 < max_seconds <= 60
        root.mkdir()
        (root / "run").mkdir()
        (root / "run/scene.json").write_bytes((run / "scene.json").read_bytes())
        return {"task_source": "sealed_input"}

    monkeypatch.setattr(study, "Lab", Lab)
    monkeypatch.setattr(study, "prepare_shared_exercise_lab", prepare)
    monkeypatch.setattr(study, "load_recording", lambda *args, **kwargs: {"events": events})
    monkeypatch.setattr(study, "verify_recording", lambda *args, **kwargs: {"events": len(events),
        "steps": sum(e["type"] == "step" for e in events), "resets": sum(e["type"] == "reset" for e in events),
        "video_frames": len(events)})
    monkeypatch.setattr(study, "verify_artifacts", lambda *args: {"verified": True})
    monkeypatch.setattr(sys, "argv", [str(path), "--run", str(source), "--out", str(out)])
    study.main()
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "integration_verified"
    assert report["baseline_comparison"] == "not_requested"
    assert report["matched_baseline_steps"] is None and report["exact_baseline_state_outcomes"] is None
    assert report["post_reset_authored_steps"] == limit
    assert report["declared_horizon_steps"] == 6
    assert report["task_success"] is not early_failure
    assert report["completed_interval"] is not early_failure
    assert (report["task_failure"] is not None) is early_failure
    assert set(report["recorded_controllers"]) == {"manual", "coast", "authored"}
    assert report["recorded_resets"] >= 1
    assert json.loads((out / "settings.json").read_text())["task_source"] == "sealed_input"
    assert json.loads((out / "manifest.json").read_text())["status"] == "verified"
