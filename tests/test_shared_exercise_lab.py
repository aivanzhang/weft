"""Human-only live labs keep independent task meaning, provenance and recorded intent."""

import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

from weft.io import digest, write_json
from weft.capabilities import describe_run_capabilities
from weft.lab import Lab, PUBLIC_ASSETS
from weft.recording import load_recording
from weft.shared_exercise_lab import load_shared_exercise_lab, prepare_shared_exercise_lab
from weft.shared_exercise_task import build_shared_exercise_task
from test_lab_http import request, serving
from test_shared_exercise_run import source_fixture
from test_shared_world_packaging import candidate, seal


def prepared(tmp_path, *, root_assistance=False):
    source, scene = source_fixture(tmp_path)
    target = tmp_path / "lab"
    prepare_shared_exercise_lab(source, target, root_assistance=root_assistance)
    return target, source, scene


@pytest.mark.parametrize("root_assistance", [False, True])
def test_human_only_lab_modes_reset_record_export_and_resume(tmp_path, root_assistance):
    target, source, scene = prepared(tmp_path, root_assistance=root_assistance)
    source_before = {path.name: path.read_bytes() for path in source.iterdir()}
    run_before = {path.relative_to(target / "run").as_posix(): digest(path)
                  for path in (target / "run").rglob("*") if path.is_file()}
    lab = Lab(target)
    try:
        status = lab.summary()
        count = 22 if root_assistance else 19
        assert status["control_profile"] == "shared_exercise"
        assert status["scene"] == scene
        assert status["exercise_task"]["control"]["object_forces"] == "forbidden"
        assert len(status["action_contract"]["names"]) == count
        caps = status["capabilities"]
        assert caps["action_contract"]["dimension"] == count
        assert caps["action_contract"]["object_force_channels"] == []
        assert not caps["controllers"]["learned_policy_available"]
        assert caps["task_contract"]["task_type"] == "shared-exercise-task-v1"
        assert [item["id"] for item in caps["entities"]["controllable_actors"]] == ["human"]
        lab.control("/api/reset", {"seed": 19})
        started = lab.control("/api/record/start", {})
        assert started["snapshot"]["resolved_reset_options"] == {"jitter": False, "jitter_scale": 0.}
        manual = [0.] * count
        manual[0] = .25
        events = []
        for payload in ({"action": manual}, {"controller": "coast"}, {"controller": "reference"}):
            events.extend(lab.control("/api/step", payload)["transitions"])
        assert [event["controller"] for event in events] == ["manual", "coast", "authored"]
        assert events[1]["action"] == [0.] * count
        assert all(not any(event["snapshot"]["engine_state"]["normalized_action"][22:]) for event in events)
        assert lab.control("/api/step", {"controller": "coast"})["transitions"] == []
        reset = lab.control("/api/reset", {"seed": 29})["snapshot"]
        assert reset["engine_state"]["steps"] == 0
        assert reset["task_metrics"]["observed_substeps"] == 0
        lab.control("/api/step", {"controller": "reference", "steps": 3})
        lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=30)
        assert exported["action_replay"]["exact_state_replay"]
        assert exported["events"] == exported["video_frames"] == 8
        data = load_recording(Path(exported["directory"]))
        assert data["scene"]["profile"] == "shared-exercise-recording-v1"
        assert data["events"][4]["seed"] == 29
        for name in PUBLIC_ASSETS.values():
            assert (target / "run" / "lab-implementation/assets" / name).is_file()
    finally:
        lab.close()
    resumed = Lab(target)
    try:
        assert resumed.summary()["snapshot"]["engine_state"]["steps"] == 0
        assert len(resumed.summary()["sessions"]) == 1
    finally:
        resumed.close()
    assert {path.name: path.read_bytes() for path in source.iterdir()} == source_before
    assert {path.relative_to(target / "run").as_posix(): digest(path)
            for path in (target / "run").rglob("*") if path.is_file()} == run_before


def test_exact_scene_and_task_are_preserved_without_importing_legacy_policy(tmp_path):
    source, scene = source_fixture(tmp_path)
    scene["learned_policy_available"] = True
    write_json(source / "scene.json", scene)
    write_json(source / "policy.json", {"legacy_policy": "must never execute"})
    seal(source)
    task = build_shared_exercise_task(scene, root_assistance=True)
    task_path = tmp_path / "chosen-task.json"
    task_path.write_text(json.dumps(task, separators=(",", ":")))
    target = tmp_path / "lab"
    prepare_shared_exercise_lab(source, target, task_path=task_path)
    assert (target / "run/scene.json").read_bytes() == (source / "scene.json").read_bytes()
    assert (target / "run/exercise-task.json").read_bytes() == task_path.read_bytes()
    assert not (target / "run/policy.json").exists()
    assert describe_run_capabilities(target / "run")["task_contract"]["task_type"] == "shared-exercise-task-v1"
    lab = Lab(target)
    try:
        state = lab.summary()
        assert state["scene"]["learned_policy_available"] is True
        assert state["learned_policy_available"] is False
        with pytest.raises(ValueError, match="no learned"):
            lab.control("/api/step", {"policy": True})
        state["exercise_task"]["control"]["root_assistance"] = "disabled"
        state["scene"]["task"]["horizon_steps"] = 1
        assert lab.summary()["exercise_task"] == task
        assert lab.summary()["scene"] == scene
    finally:
        lab.close()


def test_sealed_input_task_preserves_custom_contacts_and_root_controls_by_default(tmp_path):
    source, scene = source_fixture(tmp_path)
    equipment = next(obj["id"] for obj in scene["objects"] if obj["role"] != "static")
    task = build_shared_exercise_task(scene, root_assistance=True, contact_requirements=[{
        "object_id": equipment, "wrist": "left", "kind": "wrist_region_contact", "start_step": 1, "end_step": 3,
    }])
    task_raw = json.dumps(task, indent=2).encode()
    (source / "exercise-task.json").write_bytes(task_raw)
    seal(source)
    target = tmp_path / "lab"
    prepare_shared_exercise_lab(source, target)
    loaded = load_shared_exercise_lab(target / "run")
    assert loaded["task"] == task
    assert (target / "run/exercise-task.json").read_bytes() == task_raw
    assert (target / "run/input-task.json").read_bytes() == task_raw
    assert not loaded["contract"]["learned_policy_available"]


@pytest.mark.parametrize("failure", ["stale_scene", "conflicting_root_assistance"])
def test_sealed_input_task_cannot_be_silently_replaced_by_defaults(tmp_path, failure):
    source, scene = source_fixture(tmp_path)
    task = build_shared_exercise_task(scene)
    if failure == "stale_scene":
        task["scene_sha256"] = "a" * 64
    write_json(source / "exercise-task.json", task)
    seal(source)
    with pytest.raises(ValueError, match="scene identity|conflicts with the sealed input task"):
        prepare_shared_exercise_lab(source, tmp_path / "unused", root_assistance=failure != "stale_scene")
    assert not (tmp_path / "unused").exists()


def test_explicit_task_is_still_an_explicit_override_of_the_saved_choice(tmp_path):
    source, scene = source_fixture(tmp_path)
    write_json(source / "exercise-task.json", build_shared_exercise_task(scene))
    seal(source)
    chosen = build_shared_exercise_task(scene, root_assistance=True)
    task_path = tmp_path / "chosen-task.json"
    write_json(task_path, chosen)
    prepare_shared_exercise_lab(source, tmp_path / "lab", task_path=task_path)
    assert load_shared_exercise_lab(tmp_path / "lab/run")["task"] == chosen


def test_source_equivalence_closure_is_standalone_and_tampering_blocks_resume(tmp_path):
    source = tmp_path / "source"
    scene, observation = candidate(source, equivalent=True)
    scene["task"]["horizon_steps"] = 3
    write_json(source / "scene.json", scene)
    write_json(source / "geometry-study.json", {"panels": ["initial.png"]})
    (source / "geometry-inspection").mkdir()
    (source / "geometry-inspection/initial.png").write_bytes(b"sealed analytic image placeholder")
    seal(source)
    target = tmp_path / "lab"
    prepare_shared_exercise_lab(source, target)
    loaded = load_shared_exercise_lab(target / "run")
    assert loaded["source_verification"]["source_evidence_verified"]
    assert "manifest.json" in {row["name"] for row in loaded["artifacts"]}
    assert all("/" not in row["name"] for row in loaded["artifacts"])
    bundle = scene["source_binding"]["source_equivalences"][0]["bundle_path"]
    assert (target / "run" / bundle / "source-frames.log").read_bytes() == (source / bundle / "source-frames.log").read_bytes()
    assert digest(target / "run/source.webm") == scene["source_binding"]["source_sha256"]
    assert (target / "run/geometry-study.json").read_bytes() == (source / "geometry-study.json").read_bytes()
    assert (target / "run/geometry-inspection/initial.png").read_bytes() == (source / "geometry-inspection/initial.png").read_bytes()
    (target / "run" / observation).write_bytes(b"changed")
    lab = Lab(target)
    try:
        with pytest.raises(ValueError, match="SHA256"):
            lab.summary()
        assert lab.session is None
    finally:
        lab.close()


@pytest.mark.parametrize("mutation", ["task", "scene", "runtime_snapshot", "config_profile", "config_manifest"])
def test_resume_rejects_task_scene_runtime_and_config_tampering(tmp_path, mutation):
    target, _, _ = prepared(tmp_path)
    names = {"task": "exercise-task.json", "scene": "scene.json", "runtime_snapshot": "lab-implementation/shared_world.py"}
    if mutation in names:
        with (target / "run" / names[mutation]).open("ab") as stream:
            stream.write(b" ")
    else:
        path = target / "lab-config.json"
        config = json.loads(path.read_text())
        if mutation == "config_profile":
            del config["control_profile"]
        else:
            config["manifest_sha256"] = "0" * 64
        write_json(path, config)
    lab = Lab(target)
    try:
        with pytest.raises(ValueError):
            lab.summary()
        assert lab.session is None
    finally:
        lab.close()


def test_current_physics_runtime_mismatch_requires_new_lab(tmp_path, monkeypatch):
    import weft.shared_exercise_lab as module

    target, _, _ = prepared(tmp_path)
    original = module.read_package_file

    def changed(directory, name, **kwargs):
        if Path(directory) == Path(module.__file__).parent and name == "shared_world.py":
            raise ValueError("Injected changed core runtime")
        return original(directory, name, **kwargs)

    monkeypatch.setattr(module, "read_package_file", changed)
    with pytest.raises(ValueError, match="physics/evaluator runtime differs"):
        load_shared_exercise_lab(target / "run")


def test_immutable_scene_and_capabilities_are_prepared_once_for_controls(tmp_path, monkeypatch):
    from weft.shared_exercise_lab import SharedExerciseSession

    target, _, _ = prepared(tmp_path)
    lab = Lab(target)
    try:
        session = lab.load_session()
        original = SharedExerciseSession.scene.fget
        copies = []

        def scene_copy(self):
            copies.append(True)
            return original(self)

        def revalidation(*args, **kwargs):
            raise AssertionError("Immutable capabilities must not be rebuilt per request")

        monkeypatch.setattr(SharedExerciseSession, "scene", property(scene_copy))
        monkeypatch.setattr("weft.capabilities.describe_capabilities", revalidation)
        capabilities = lab.capabilities(session)
        capabilities["action_contract"]["dimension"] = 99
        assert lab.capabilities(session)["action_contract"]["dimension"] == 19
        lab.control("/api/step", {"controller": "coast"})
        assert copies == []
        lab.summary()
        assert copies == [True]
    finally:
        lab.close()


def test_resealed_original_video_cannot_replace_canonical_source(tmp_path):
    source = tmp_path / "source"
    candidate(source, equivalent=True)
    target = tmp_path / "lab"
    prepare_shared_exercise_lab(source, target)
    media = target / "run/source.webm"
    media.write_bytes(b"a different video, despite a new outer manifest")
    manifest_path = target / "run/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["source.webm"] = digest(media)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="original video"):
        load_shared_exercise_lab(target / "run")


@pytest.mark.parametrize("bound", ["MAX_OUTPUT_BYTES", "MAX_OUTPUT_FILES"])
def test_complete_package_bound_rejects_before_output_creation(tmp_path, monkeypatch, bound):
    source, _ = source_fixture(tmp_path)
    monkeypatch.setattr("weft.shared_exercise_lab." + bound, 1)
    with pytest.raises(ValueError, match="output bound"):
        prepare_shared_exercise_lab(source, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_unsafe_source_paths_and_task_override_reject_before_output(tmp_path):
    source, scene = source_fixture(tmp_path)
    task = tmp_path / "task.json"
    write_json(task, build_shared_exercise_task(scene))
    with pytest.raises(ValueError, match="not both"):
        prepare_shared_exercise_lab(source, tmp_path / "unused", task_path=task, root_assistance=True)
    raw = (source / "scene.json").read_bytes()
    outside = tmp_path / "outside.json"
    outside.write_bytes(raw)
    (source / "scene.json").unlink()
    (source / "scene.json").symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe"):
        prepare_shared_exercise_lab(source, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_stale_failure_is_a_recordable_task_failure_not_current_geometry(tmp_path, monkeypatch):
    target, _, _ = prepared(tmp_path)
    lab = Lab(target)
    original = mujoco.mj_step

    def nonfinite_step(model, data):
        original(model, data)
        data.qpos[0] = np.nan

    try:
        initial = lab.control("/api/record/start", {})["snapshot"]
        monkeypatch.setattr(mujoco, "mj_step", nonfinite_step)
        result = lab.control("/api/step", {"controller": "coast", "steps": 3})
        state = result["snapshot"]
        assert len(result["transitions"]) == 1
        assert state["engine_state_status"] == "last_valid_control_boundary"
        assert state["engine_state"] == initial["engine_state"]
        assert state["terminated"] and state["task_metrics"]["simulated_duration_s"] > 0
        assert lab.recorder is not None
        lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=30)
        assert exported["action_replay"]["exact_state_replay"]
        assert load_recording(Path(exported["directory"]))["events"][-1]["snapshot"] == state
    finally:
        lab.close()


@pytest.mark.parametrize("root_assistance", [False, True])
def test_exercise_http_contract_auth_modes_and_mutation_rejection(tmp_path, root_assistance):
    target, _, _ = prepared(tmp_path, root_assistance=root_assistance)
    count = 22 if root_assistance else 19
    with serving(Lab(target)) as server:
        assert request(server, "/api/status", token=False)[0] == 401
        code, _, body = request(server, "/api/status")
        status = json.loads(body)
        assert code == 200 and status["control_profile"] == "shared_exercise"
        assert status["snapshot"]["profile"] == "shared-exercise-state-v1"
        for endpoint, payload in [
            ("/api/actor", {"actor": "ball"}), ("/api/joint", {"joint": "left_elbow"}),
            ("/api/step", {"action": 0}), ("/api/step", {"action": [0.] * (19 if root_assistance else 22)}),
            ("/api/step", {"action": [True] + [0.] * (count - 1)}), ("/api/step", {"policy": True}),
            ("/api/step", {"controller": "coast", "action": [0.] * count}),
            ("/api/reset", {"seed": 7, "root_assistance": True}),
            ("/api/record/start", {"task": {}}),
        ]:
            assert request(server, endpoint, method="POST", data=payload)[0] == 400
        state = json.loads(request(server, "/api/status")[2])["snapshot"]
        assert state["engine_state"]["steps"] == 0
        for payload, label in [({"action": [.25] + [0.] * (count - 1)}, "manual"),
                               ({"controller": "coast"}, "coast"), ({"controller": "reference"}, "authored")]:
            code, _, body = request(server, "/api/step", method="POST", data=payload)
            result = json.loads(body)
            assert code == 200 and result["transitions"][0]["controller"] == label
            assert result["snapshot"]["profile"] == "shared-exercise-state-v1"
        code, _, body = request(server, "/api/reset", method="POST", data={"seed": 7})
        assert code == 200 and json.loads(body)["snapshot"]["engine_state"]["steps"] == 0


def test_cli_prepares_explicit_task_without_starting_server(tmp_path, monkeypatch, capsys):
    from weft.cli import main

    source, _ = source_fixture(tmp_path)
    target = tmp_path / "lab"
    monkeypatch.setattr(sys, "argv", ["weft", "shared-exercise-lab", "--run", str(source), "--out", str(target),
                                    "--root-assistance"])
    main()
    assert json.loads(capsys.readouterr().out)["control_profile"] == "shared_exercise"
    assert load_shared_exercise_lab(target / "run")["task"]["control"]["root_assistance"] == "explicit_3_axis"
