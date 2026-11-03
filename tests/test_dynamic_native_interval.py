"""Analytical free-body timing qualification; no source reconstruction claims."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_native_clock_runtime import root_fixture
from weft.dynamic_native_interval import dynamic_native_interval, validate_dynamic_interval
from weft.native_interval import elapsed_substeps
from weft.shared_world_lab import authored_shared_world_scene
from weft.source_clock import clock_summary


def dynamic_scene(endpoint=.075):
    scene = authored_shared_world_scene()
    _, _, binding = root_fixture()
    body = scene["human_spec"]
    sha = binding["source_sha256"]
    body.update(source_sha256=sha, native_clock=clock_summary(binding))
    for name in ("reference", "root_reference"):
        body[name] = deepcopy(body[name][:4])
        for index, row in enumerate(body[name]):
            row["time_s"] = index * endpoint / 3
        body[name][-1]["time_s"] = endpoint
    scene["profile"] = "shared-world-v2"
    scene["coordinate_frame"]["clock_binding"].update(
        native_clock=deepcopy(body["native_clock"]), alignment_method="native_pts_elapsed_time_v1",
        source_fps=binding["nominal_fps"], source_time_offset_s=0.)
    scene["source_binding"] = {
        "mode": "source_qualified", "source_sha256": sha, "total_source_frames": 10,
        "evidence_documents": {"d" * 64: "evidence/human.json"},
        "entity_bindings": [{"world_entity_id": "human", "entity_kind": "articulated_human",
                             "observation_entity_id": sha + ":human:0", "observation_document_sha256": "d" * 64,
                             "provenance_mode": "observed_track", "observed_slots": 4,
                             "missing_slots": 6, "pre_init_slots": 0}] + [
            {"world_entity_id": obj["id"], "entity_kind": "rigid_object", "observation_entity_id": None,
             "observation_document_sha256": None, "provenance_mode": "authored_addition",
             "observed_slots": 0, "missing_slots": 10, "pre_init_slots": 0} for obj in scene["objects"]],
    }
    marker = dynamic_native_interval(body, scene["dt"], dynamic_object_ids=["ball", "marker_box"])
    scene["task"].update(native_interval=marker, start_offset_s=0., horizon_steps=marker["full_control_steps"] + 1)
    return scene


def test_native_dynamic_contract_keeps_full_15233_endpoint_without_rounding():
    scene = dynamic_scene(15.233)
    original = deepcopy(scene)
    marker = validate_dynamic_interval(scene)
    assert marker["full_control_steps"] == 456
    assert marker["terminal_duration_s"] == pytest.approx(.033)
    assert marker["duration_s"] == 15.233
    assert scene["task"]["horizon_steps"] == 457
    assert elapsed_substeps(scene, 457 * 16) == 15.233
    assert marker["dynamic_object_ids"] == ["ball", "marker_box"]
    assert scene == original


@pytest.mark.parametrize("change", [
    lambda s: s.update(profile="shared-world-v1"),
    lambda s: s.update(profile="shared-world-v3"),
    lambda s: s["source_binding"].update(mode="analytic_fixture"),
    lambda s: s["source_binding"].update(source_sha256="f" * 64),
    lambda s: s["human_spec"].pop("native_clock"),
    lambda s: s["task"].update(start_offset_s=.001),
    lambda s: s["task"].update(horizon_steps=2),
    lambda s: s["task"].update(horizon_steps=True),
    lambda s: s["task"]["native_interval"].update(full_control_steps=True),
    lambda s: s["task"]["native_interval"].update(source_end_s=.1),
    lambda s: s["task"]["native_interval"].update(source_clock_binding_sha256="f" * 64),
    lambda s: s["task"]["native_interval"].update(dynamic_object_ids=["ball"]),
    lambda s: s["task"]["native_interval"].update(dynamic_object_ids=["marker_box", "ball"]),
    lambda s: s["task"]["native_interval"].update(extra="unsealed"),
    lambda s: s["objects"][0].update(role="static"),
])
def test_dynamic_contract_rejects_changed_span_inventory_or_source(change):
    scene = dynamic_scene()
    change(scene)
    with pytest.raises(ValueError):
        validate_dynamic_interval(scene)


@pytest.mark.parametrize("ids", [[], ["ball", "ball"], [False], [""], ["a" * 129], [str(i) for i in range(13)]])
def test_dynamic_inventory_is_bounded_and_unambiguous(ids):
    with pytest.raises(ValueError, match="inventory"):
        dynamic_native_interval(dynamic_scene()["human_spec"], 1 / 30, dynamic_object_ids=ids)


def test_grid_aligned_native_source_needs_no_partial_action_marker():
    body = dynamic_scene()["human_spec"]
    body["reference"][-1]["time_s"] = .1
    assert dynamic_native_interval(body, 1 / 30, dynamic_object_ids=["ball"]) is None


@pytest.mark.parametrize("floor_contact", [False, True])
def test_whole_model_partial_action_matches_independent_manual_timestep(floor_contact):
    from weft.shared_world import SharedWorldEnv, validate_shared_snapshot, validate_shared_world

    scene = dynamic_scene()
    ball = scene["objects"][0]
    ball.update(position=[.2, .2, ball["radius"] + .001 if floor_contact else .8],
                velocity=[0., 0., -.2], angular_velocity=[.4, -.3, .2])
    scene = validate_shared_world(scene)
    manual = deepcopy(scene)
    manual["task"].pop("native_interval")
    manual["profile"] = "shared-world-v1"
    for key in ("reference", "root_reference"):
        manual["human_spec"][key][-1]["time_s"] = .1
    samples = []
    env, reference = SharedWorldEnv(scene, substep_observer=samples.append), SharedWorldEnv(manual)
    try:
        options = {"jitter": False, "jitter_scale": 0.}
        env.reset(seed=19, options=options)
        reference.reset(seed=19, options=options)
        nominal = env.model.opt.timestep
        initial = env.snapshot()
        contacts = []
        for index in range(3):
            action = env.authored_action()
            action[22:25] = [.2, -.1, .05]
            if index == 2:
                reference.model.opt.timestep = scene["task"]["native_interval"]["terminal_duration_s"] / 16
            _, _, terminated, truncated, _ = env.step(action)
            reference.step(action)
            assert np.array_equal(env.data.qpos, reference.data.qpos)
            assert np.array_equal(env.data.qvel, reference.data.qvel)
            assert env._objects_state() == reference._objects_state()
            assert env._current_contacts() == reference._current_contacts()
            contacts.extend(env.snapshot()["contact_details"])
            assert env.model.opt.timestep == nominal
            assert not terminated and truncated == (index == 2)
            validate_shared_snapshot(env.snapshot(), scene)
        assert len(samples) == 48
        for index, sample in enumerate(samples):
            assert sample.interval_start_s == pytest.approx(elapsed_substeps(scene, index), abs=1e-12)
            assert sample.interval_end_s == pytest.approx(elapsed_substeps(scene, index + 1), abs=1e-12)
        assert env.data.time == pytest.approx(.075, abs=1e-12)
        assert env.snapshot()["time_s"] == pytest.approx(.075, abs=1e-12)
        assert env.snapshot()["source_time_s"] == pytest.approx(.075, abs=1e-12)
        assert env.observation()[154] == pytest.approx(1.)
        if floor_contact:
            assert any("ball" in str(row) and "floor" in str(row) for row in contacts)
        else:
            final_ball = env._objects_state()[0]
            assert final_ball["velocity"][2] < -.8
            assert final_ball["position"][2] < initial["objects"][0]["position"][2]
        env.reset(seed=19, options=options)
        assert env.snapshot() == initial
        assert env.model.opt.timestep == nominal
    finally:
        env.close()
        reference.close()


def test_early_terminal_failure_keeps_actual_partial_time_and_restores_timestep(monkeypatch):
    from weft.shared_world import SharedWorldEnv, validate_shared_snapshot

    scene = dynamic_scene()
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=19, options={"jitter": False})
        for _ in range(2):
            env.step(env.authored_action())
        nominal, original = env.model.opt.timestep, env._instability
        monkeypatch.setattr(env, "_instability", lambda: "object_left_bounds"
                            if env.data.time >= elapsed_substeps(scene, 35) - 1e-12 else original())
        _, _, terminated, truncated, _ = env.step(env.authored_action())
        assert terminated and not truncated
        assert env.data.time == pytest.approx(elapsed_substeps(scene, 35), abs=1e-12)
        assert env.model.opt.timestep == nominal
        snapshot = env.snapshot()
        assert snapshot["partial_action"] == {"completed_substeps": 3, "requested_substeps": 16}
        validate_shared_snapshot(snapshot, scene)
    finally:
        env.close()


def test_native_dynamic_journal_replays_exact_object_states_and_export_clock(tmp_path):
    from weft.recording import Recorder, _native_export_timing, _replay_html, load_recording
    from weft.shared_world import SharedWorldEnv

    scene = dynamic_scene()
    recorder = Recorder(tmp_path / "recording", scene, profile="shared-world")
    env, replay = SharedWorldEnv(scene), SharedWorldEnv(scene)
    try:
        env.reset(seed=19)
        recorder.start(env.snapshot(), seed=19)
        for _ in range(3):
            action = env.authored_action()
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.append({"type": "step", "action": action.tolist(), "reward": reward,
                             "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()})
        recorder.close()
        saved = load_recording(recorder.root)
        replay.reset(seed=19)
        assert replay.snapshot() == saved["events"][0]["snapshot"]
        for event in saved["events"][1:]:
            replay.step(event["action"])
            assert replay.snapshot() == event["snapshot"]
        timing = _native_export_timing(saved["scene"], saved["events"])
        assert timing["recorded_simulated_duration_s"] == pytest.approx(.075)
        assert timing["video_frame_timing"] == "constant_rate_event_frames"
        assert "native-dynamic-terminal-control-interval-v1" in _replay_html(saved["scene"], saved["events"])
    finally:
        if recorder._started is not None and not recorder._closed:
            recorder.close()
        env.close()
        replay.close()


def test_dynamic_native_learning_and_stationary_task_reject_before_work(tmp_path, monkeypatch):
    from weft import shared_world_learning as learning
    from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task, validate_shared_exercise_task

    scene = dynamic_scene()
    def forbidden(*args, **kwargs):
        raise AssertionError("Unqualified learning must not compile or evaluate dynamics")
    monkeypatch.setattr(learning, "SharedWorldEnv", forbidden)
    monkeypatch.setattr(learning, "audit_shared_world_feasibility", forbidden)
    for call in (lambda: build_shared_exercise_task(scene),
                 lambda: validate_shared_exercise_task({}, scene),
                 lambda: SharedExerciseTaskEnv(scene, {}),
                 lambda: learning.validate_shared_policy({}, scene),
                 lambda: learning.policy_action({}, SimpleNamespace(scene=scene)),
                 lambda: learning.train_shared_world_policy(scene, tmp_path / "forbidden")):
        with pytest.raises(ValueError, match="not qualified|separately qualified"):
            call()
    assert not (tmp_path / "forbidden").exists()


def test_runtime_closure_is_added_only_for_new_marker_and_old_marker_still_rejects():
    from weft.native_interval import native_interval, validate_interval
    from weft.shared_world_lab import _lab_runtime_modules

    old = authored_shared_world_scene()
    original = _lab_runtime_modules(old)
    assert not {"native_interval.py", "dynamic_native_interval.py", "source_clock.py"} & set(original)
    scene = dynamic_scene()
    assert _lab_runtime_modules(scene) == (*original, "source_clock.py", "native_interval.py", "dynamic_native_interval.py")
    scene["task"]["native_interval"] = native_interval(scene["human_spec"], scene["dt"])
    with pytest.raises(ValueError, match="person-only"):
        validate_interval(scene)


def test_full_15233_native_interval_advances_all_dynamic_substeps_without_hold():
    from weft.shared_world import SharedWorldEnv, validate_shared_snapshot

    scene = dynamic_scene(15.233)
    # Dense authored knots preserve the original root-gap gate in this timing fixture.
    for name in ("reference", "root_reference"):
        row = scene["human_spec"][name][0]
        scene["human_spec"][name] = [dict(deepcopy(row), time_s=min(index / 30, 15.233)) for index in range(458)]
    boundaries = []
    env = SharedWorldEnv(scene, substep_observer=lambda row: boundaries.append((row.interval_start_s, row.interval_end_s)))
    try:
        env.reset(seed=19, options={"jitter": False})
        for index in range(457):
            _, _, terminated, truncated, _ = env.step(env.authored_action())
            assert not terminated and truncated == (index == 456)
        assert len(boundaries) == 457 * 16
        for index, (start, end) in enumerate(boundaries):
            assert start == pytest.approx(elapsed_substeps(scene, index), abs=1e-9)
            assert end == pytest.approx(elapsed_substeps(scene, index + 1), abs=1e-9)
        assert boundaries[-1][1] == pytest.approx(15.233, abs=1e-9)
        assert env.data.time == pytest.approx(15.233, abs=1e-9)
        assert env.model.opt.timestep == scene["dt"] / 16
        assert env.observation()[154] == pytest.approx(1.)
        snapshot = env.snapshot()
        validate_shared_snapshot(snapshot, scene)
        assert len(snapshot["objects"]) == 3
        assert snapshot["source_time_s"] == pytest.approx(15.233, abs=1e-9)
        with pytest.raises(RuntimeError, match="Reset"):
            env.step(env.authored_action())
    finally:
        env.close()
