"""Six bounded native recording cases: 48 controls including replay, one short export.

Synthetic fixture geometry and original task gates stay fixed. Early termination is
preserved and replayed before reporting a planned-control shortfall, never retuned.
"""

import builtins
from copy import deepcopy
from hashlib import sha256
import json
import os
import socket
import subprocess
import time

import mujoco
import numpy as np
from PIL import Image
import pytest

from test_shared_world_feet import fixture_scene
from weft.io import canonical_hash
from weft.recording import Recorder, export_recording, load_recording
from weft.shared_exercise_feet_projection import derive_base_scoring_scene
from weft.shared_exercise_feet_recording import (
    recording_scene, validate_action, validate_recording_scene, verify_action_replay,
)
from weft.shared_exercise_feet_state import prepare_shared_exercise_feet_snapshot_validator
from weft.shared_exercise_feet_task import SharedExerciseFeetTaskEnv, build_shared_exercise_feet_task
from weft.shared_exercise_reset import human_jitter_options
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_world_lab import authored_shared_world_scene


CASES = [(f"shared-world-v{version}", root) for version in (2, 3, 4) for root in (False, True)]
MAX_CONTROLS_PER_CASE = 8
MAX_EXPORT_BYTES = 16 << 20


@pytest.fixture(scope="module", autouse=True)
def local_native_only():
    def forbidden(*_args, **_kwargs):
        pytest.fail("Feet integration permits local physics/PIL/OpenCV export, not network, workers, inference or native GL")

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "tensorflow", "mediapipe", "OpenGL", "glfw", "requests", "httpx"}:
            forbidden()
        return original_import(name, *args, **kwargs)

    with pytest.MonkeyPatch.context() as guard:
        guard.setattr(builtins, "__import__", guarded_import)
        for owner, names in ((mujoco, ("Renderer", "GLContext")),
                             (socket, ("socket", "create_connection")),
                             (subprocess, ("Popen", "run", "check_call", "check_output")),
                             (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
            for name in names:
                if hasattr(owner, name):
                    guard.setattr(owner, name, forbidden)
        yield


@pytest.fixture(scope="module")
def analytical_template(local_native_only):
    return authored_shared_world_scene()


def check_initial_diagnostics(state):
    info = state["ankle_diagnostics"]
    assert info["observed_substeps"] == info["valid_substeps"] == info["excluded_foot_contact_records"] == 0
    assert info["valid_observed_duration_s"] == 0 and info["latest_substep"] is None
    assert info["ankle_effort_mean_squared"] is info["ankle_hold_error_rmse_rad"] is None
    for pair in info["contact_aggregates"]:
        assert pair["contact_records"] == 0
        assert pair["first_contact_time_s"] is pair["last_contact_time_s"] is pair["peak_normal_force"] is None


@pytest.mark.parametrize("profile,root_assistance", CASES,
                         ids=[f"v{version}-roots-{mode}" for version in (2, 3, 4) for mode in ("off", "on")])
def test_native_feet_task_record_load_replay(analytical_template, tmp_path, monkeypatch, profile, root_assistance):
    before_template = deepcopy(analytical_template)
    base, actual, original_reference = fixture_scene(
        analytical_template, profile, contact="static" if profile == "shared-world-v4" else "floor")
    assert analytical_template == before_template
    scoring = derive_base_scoring_scene(actual)
    assert scoring == {**base, "learned_policy_available": False} and actual["profile"] == "shared-world-v5"
    assert actual["foot_extension"]["reference_json"].encode() == original_reference
    before_scene = deepcopy(actual)
    options = {"root_assistance": root_assistance}
    if profile == "shared-world-v4":
        options["support_requirements"] = [{
            "object_id": "table", "human_segment": "left_forearm", "direction_world": [0., 0., 1.],
            "minimum_load_body_weight_fraction": .05, "start_step": 0,
            "end_step": scoring["task"]["horizon_steps"],
        }]
    original_task = build_shared_exercise_task(scoring, **options)
    task = build_shared_exercise_feet_task(actual, original_task)
    assert actual == before_scene and task["base_task"] == original_task
    assert task["interval"] == original_task["interval"]
    assert task["scope"]["foot_support_credit"] is False
    assert task["scope"]["ground_load_requirements"] == "unsupported"
    scene_sha, scoring_sha = canonical_hash(actual), canonical_hash(scoring)
    original_task_sha, task_sha = canonical_hash(original_task), canonical_hash(task)
    reference_sha = sha256(original_reference).hexdigest()
    validate = prepare_shared_exercise_feet_snapshot_validator(actual, task)
    bundle = recording_scene(actual, task)
    for old_count in (19, 22):
        with pytest.raises(ValueError, match="exactly"):
            validate_action([0.] * old_count, bundle)
    unsupported_policy = deepcopy(bundle)
    unsupported_policy["learned_policy"] = {"profile": "shared-exercise-policy-v1"}
    with pytest.raises(ValueError, match="policies and avatars"):
        validate_recording_scene(unsupported_policy)

    control_calls = 0
    real_step = SharedExerciseFeetTaskEnv.step

    def bounded_step(instance, action):
        nonlocal control_calls
        assert control_calls < MAX_CONTROLS_PER_CASE, "Native recording/replay control budget exceeded"
        control_calls += 1
        return real_step(instance, action)

    monkeypatch.setattr(SharedExerciseFeetTaskEnv, "step", bounded_step)
    env = SharedExerciseFeetTaskEnv(actual, task)
    recorder = None
    saved_states, phase_counts, outcomes = [], [], []
    contacts_seen = {"human_floor": 0, "human_static": 0}
    root = tmp_path / "recording"
    requirements = original_task.get("support_requirements", original_task.get("contact_requirements", []))
    equipment = original_task.get("equipment_ids", [])
    expected_dimension = 197 + 17 * len(equipment) + 6 * len(requirements) + 1

    def capture(observation):
        assert observation.shape == env.observation_space.shape == (expected_dimension,)
        assert env.observation_space.contains(observation)
        state = json.loads(json.dumps(env.snapshot(), allow_nan=False))
        validate(state)
        engine, metrics, ankles = state["engine_state"], state["task_metrics"], state["ankle_diagnostics"]
        assert engine["profile"] == "shared-world-v5" and canonical_hash(env._engine.scene) == scene_sha
        assert state["scene_sha256"] == task["scene_sha256"]
        assert metrics["scene_sha256"] == task["base_scene_sha256"] == scoring_sha
        assert metrics["task_sha256"] == task["base_task_sha256"] == original_task_sha
        assert metrics["gates_sha256"] == original_task["gates_sha256"]
        assert metrics["evaluator_sha256"] == env.evaluator_sha256
        assert metrics["source_scope"] == original_task["source_scope"]
        assert state["adapter_implementation_sha256"] == task["adapter_implementation_sha256"]
        assert state["foot_reference_sha256"] == reference_sha
        assert engine["physical_validation"] == "unverified"
        assert len(engine["human"]["action_names"]) == 28
        assert len(engine["human"]["angles"]) == 25 and len(state["last_action"]) == env.action_count
        assert {"left_foot", "right_foot"} <= {part["id"] for part in engine["human"]["segments"]}
        assert not any(engine["normalized_action"][28:])
        assert all(not any(item.get("virtual_force", [0., 0., 0.])) for item in engine["objects"])
        if not root_assistance:
            assert not any(engine["normalized_action"][19:22])
            assert engine["human"]["mobility"]["support_force"] == [0., 0., 0.]
            np.testing.assert_array_equal(env._engine.data.ctrl[19:22], np.zeros(3))
        assert ankles["source"] == "simulated_engine_substeps" and ankles["scored"] is False
        assert ankles["foot_support_credit"] is False
        assert ankles["observed_substeps"] == metrics["observed_substeps"]
        assert ankles["valid_substeps"] == metrics["valid_substeps"]
        if profile == "shared-world-v4":
            assert all(row["human_segment"] == "left_forearm" for row in metrics["support_requirements"])
            assert all(pair["human_segment"] in ("left_foot", "right_foot") for pair in ankles["contact_aggregates"])
        for contact in engine["contact_details"]:
            if contact["kind"] in contacts_seen and any(name in contact["objects"] for name in ("left_foot_geom", "right_foot_geom")):
                contacts_seen[contact["kind"]] += 1
        saved_states.append(state)
        return state

    try:
        assert env.action_space.shape == (28 if root_assistance else 25,)
        assert env.observation_contract()["fields"][0]["count"] == 197
        observation, _ = env.reset(seed=17)
        initial = capture(observation)
        check_initial_diagnostics(initial)
        recorder = Recorder(root, bundle, profile="shared-exercise-feet")
        recorder.start(initial, seed=17)
        for phase in ("manual", "authored"):
            if phase == "authored":
                reset_options = human_jitter_options(.02)
                observation, _ = env.reset(seed=29, options=reset_options)
                reset = capture(observation)
                check_initial_diagnostics(reset)
                assert reset["resolved_reset_options"] == reset_options
                recorder.append({"type": "reset", "seed": 29, "snapshot": reset})
            performed = 0
            for index in range(2):
                if phase == "manual":
                    action = np.zeros(env.action_count)
                    motor = "left_ankle_y" if index == 0 else "right_ankle_y"
                    action[env.action_names.index(motor)] = .05 if index == 0 else -.05
                else:
                    action = env.authored_action()
                observation, reward, terminated, truncated, _ = env.step(action)
                state = capture(observation)
                recorder.append({"type": "step", "controller": phase, "action": action.tolist(), "reward": reward,
                                 "terminated": terminated, "truncated": truncated, "snapshot": state})
                performed += 1
                outcomes.append({"phase": phase, "action_index": index, "reward": reward,
                                 "terminated": terminated, "truncated": truncated,
                                 "engine_state_status": state["engine_state_status"], "task_metrics": state["task_metrics"],
                                 "ankle_diagnostics": state["ankle_diagnostics"]})
                if terminated or truncated:
                    break
            phase_counts.append(performed)
        if profile == "shared-world-v3" and not root_assistance:
            before_render, native_time = env.snapshot(), float(env._engine.data.time)
            first, second = env.render(), env.render()
            assert first.dtype == np.uint8 and first.ndim == 3 and first.shape[1:] == (640, 3)
            assert 0 < first.shape[0] <= 1024
            np.testing.assert_array_equal(first, second)
            assert env.snapshot() == before_render and float(env._engine.data.time) == native_time
            assert control_calls == sum(phase_counts)
        recorder.close()
    finally:
        if recorder is not None and recorder._started is not None and not recorder._closed:
            recorder.close()
        env.close()

    data = load_recording(root)
    assert data["scene"] == bundle
    assert [event["snapshot"] for event in data["events"]] == saved_states
    assert [event["index"] for event in data["events"]] == list(range(len(saved_states)))
    assert data["metadata"]["limits"]["max_event_bytes"] == 512 << 10
    assert data["metadata"]["limits"]["max_journal_bytes"] == 128 << 20
    assert data["metadata"]["seed"] == 17 and data["metadata"]["status"] == "closed"
    reset_event = next(event for event in data["events"] if event["type"] == "reset")
    assert reset_event["seed"] == 29 and reset_event["snapshot"]["resolved_reset_options"] == human_jitter_options(.02)
    replay = verify_action_replay(root, max_seconds=20)
    recorded_steps = sum(phase_counts)
    assert replay["exact_state_replay"] is True and replay["events"] == len(saved_states)
    assert replay["steps"] == recorded_steps and control_calls == 2 * recorded_steps <= MAX_CONTROLS_PER_CASE
    assert replay["scene_sha256"] == task["scene_sha256"] and replay["task_sha256"] == task_sha
    for key in ("base_scene_sha256", "base_task_sha256", "projection_contract_sha256", "adapter_implementation_sha256"):
        assert replay[key] == task[key]

    # Save honest task failures before any export or planned-control assertion.
    with (tmp_path / "native-outcome.json").open("x") as stream:
        json.dump({"scope": "Analytical native software integration, not task success or physical validation",
                   "base_profile": profile, "root_assistance": root_assistance,
                   "planned_controls": 4, "recorded_controls": recorded_steps, "controls_including_replay": control_calls,
                   "phase_counts": phase_counts, "recorded_foot_contact_details": contacts_seen,
                   "scene_sha256": scene_sha, "task_sha256": task_sha,
                   "source_reference_sha256": reference_sha,
                   "outcomes": outcomes, "replay": replay}, stream, indent=2, allow_nan=False)

    if profile == "shared-world-v3" and not root_assistance:
        exported = export_recording(root, deadline=time.monotonic() + 20, max_output_bytes=MAX_EXPORT_BYTES)
        assert exported["video_frames"] == exported["events"] == len(saved_states)
        assert exported["steps"] == recorded_steps and exported["resets"] == 1 and exported["verified_files"] == 6
        assert control_calls == 2 * recorded_steps
        html = (root / "replay.html").read_text()
        assert '"feet_recording_scope"' in html and '"ankle_diagnostics"' in html
        assert "not physical validation" in html and "__WEFT_DATA__" not in html
        with Image.open(root / "preview.gif") as preview:
            assert preview.format == "GIF" and 1 <= preview.n_frames <= len(saved_states)
            assert preview.width <= 480 and preview.height <= 300
        assert (root / "video.mp4").stat().st_size > 0
    assert phase_counts == [2, 2], "Native termination shortened the declared control plan; inspect native-outcome.json"
