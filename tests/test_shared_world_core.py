"""Coupled-world state isolation, contact attribution and admission regressions."""

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from weft.io import canonical_hash, digest, write_json
from weft.lab import SharedWorldSession
from weft.observations import adapt_exercise
from weft.pipeline import verify_artifacts
from weft.recording import Recorder, load_recording
from weft.shared_world import SharedWorldEnv, audit_shared_world_feasibility, validate_shared_snapshot
from weft.shared_world_lab import authored_shared_world_scene
from weft.shared_world_learning import ALGORITHM, POLICY_PROFILE, policy_action, train_shared_world_policy


def test_reference_inspection_preserves_live_solver_state_and_next_trajectory():
    scene = authored_shared_world_scene()
    inspected, untouched = SharedWorldEnv(scene), SharedWorldEnv(scene)
    try:
        initial_reference = inspected.reference_snapshot(0)
        validate_shared_snapshot(initial_reference, inspected.scene)
        for env in (inspected, untouched):
            env.reset(seed=17)
            for _ in range(5):
                action = env.authored_action()
                action[-3:] = [0.2, -0.3, 0.1]
                env.step(action)
        arrays = ("qpos", "qvel", "qacc", "qacc_warmstart", "ctrl", "qfrc_applied", "xfrc_applied")
        before = {name: getattr(inspected.data, name).copy() for name in arrays}
        live_before = inspected.snapshot()
        for index in (5, 12, 0, 25):
            reference = inspected.reference_snapshot(index)
            validate_shared_snapshot(reference, inspected.scene)
            assert not any(reference["normalized_action"])
            assert not any(reference["human"]["motor_forces"])
            assert all(not any(obj.get("virtual_force", [])) for obj in reference["objects"])
        assert inspected.snapshot() == live_before
        for name in arrays:
            np.testing.assert_array_equal(getattr(inspected.data, name), before[name])
        for _ in range(5):
            first = inspected.step(inspected.authored_action())
            second = untouched.step(untouched.authored_action())
            np.testing.assert_array_equal(first[0], second[0])
            assert first[1:] == second[1:]
            assert inspected.snapshot() == untouched.snapshot()
    finally:
        inspected.close()
        untouched.close()


def test_contact_with_another_object_does_not_satisfy_goal_contact():
    scene = authored_shared_world_scene()
    scene["task"]["object_goal"]["target_object_id"] = "marker_box"
    scene["target"] = next(obj["position"][:] for obj in scene["objects"] if obj["id"] == "marker_box")
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0)
        for _ in range(env.horizon_steps):
            _, _, terminated, truncated, _ = env.step(env.authored_action())
            if terminated or truncated:
                break
        metrics = env.metrics()
        assert metrics["human_object_contacts_total"] > 0
        assert metrics["human_goal_object_contacts_total"] == 0
        assert metrics["human_object_contact_satisfied"] is False
        assert metrics["success"] is False
    finally:
        env.close()


def test_initial_human_static_intersection_rejected_without_moving_geometry():
    scene = authored_shared_world_scene()
    base = SharedWorldEnv(scene)
    try:
        base.reset(seed=0)
        pelvis = base.snapshot()["human"]["joints"]["pelvis"]
    finally:
        base.close()
    table = next(obj for obj in scene["objects"] if obj["id"] == "table")
    table["position"] = pelvis
    table["size"] = [0.1, 0.1, 0.1]
    original = deepcopy(scene)
    audit = audit_shared_world_feasibility(scene)
    assert audit["status"] == "infeasible"
    assert any(row["kind"] == "human_static" for row in audit["initial_intersections"])
    env = SharedWorldEnv(scene)
    try:
        with pytest.raises(ValueError, match="intersect"):
            env.reset(seed=0)
        np.testing.assert_array_equal(env.data.xpos[env.model.body("table").id], pelvis)
        with pytest.raises(RuntimeError, match="Reset before stepping"):
            env.step(np.zeros(len(env.action_names)))
        assert scene == original
    finally:
        env.close()


def test_policy_action_avoids_compilation_but_keeps_scene_and_action_binding(monkeypatch):
    env = SharedWorldEnv(authored_shared_world_scene())
    try:
        env.reset(seed=0)
        policy = {
            "schema_version": 1,
            "profile": POLICY_PROFILE,
            "algorithm": ALGORITHM,
            "scene_sha256": canonical_hash(env.scene),
            "action_names": list(env.action_names),
            "weights": [0.1] * 16,
        }
        def unexpected_compile(*args, **kwargs):
            raise AssertionError("Policy action recompiled the validated environment")
        monkeypatch.setattr("weft.shared_world.human_subtree_mass", unexpected_compile)
        assert env.action_space.contains(policy_action(policy, env).astype(np.float32))
        wrong_actions = deepcopy(policy)
        wrong_actions["action_names"][0] = "unbound_motor"
        with pytest.raises(ValueError, match="action_names differ"):
            policy_action(wrong_actions, env)
        env.scene["target"][0] += 0.01
        with pytest.raises(ValueError, match="scene_sha256 does not match"):
            policy_action(policy, env)
    finally:
        env.close()


def test_source_qualified_training_preserves_pinned_observations(tmp_path):
    from weft.shared_world_evidence import build_shared_world_binding, verify_shared_world_sources

    scene = authored_shared_world_scene()
    source_sha = "c" * 64
    scene["human_spec"]["source_sha256"] = source_sha
    observations = adapt_exercise({
        "profile": "exercise-pose-v1", "source_sha256": source_sha,
        "width": 640, "height": 480, "fps": 30, "frame_count": 61, "decoded_frames": 61,
        "frames": [{"id": f"{source_sha}:frame:{i}", "frame": i, "time_s": i / 30,
                    "status": "tracked", "landmarks": [{"x": 0.2, "y": 0.4, "z": 0.0}]}
                   for i in range(0, 61, 2)],
    })
    raw = json.dumps(observations, sort_keys=True).encode()
    sha = hashlib.sha256(raw).hexdigest()
    world_ids = ["human", *(obj["id"] for obj in scene["objects"])]
    refs = {name: None for name in world_ids}
    refs["human"] = (sha, observations["objects"][0]["id"])
    scene["source_binding"], _ = build_shared_world_binding(
        {sha: raw}, source_sha256=source_sha, world_ids=world_ids, entity_refs=refs)
    with pytest.raises(ValueError, match="sealed source directory"):
        train_shared_world_policy(scene, tmp_path / "unbound")
    assert not (tmp_path / "unbound").exists()

    source = tmp_path / "source"
    document_name = scene["source_binding"]["evidence_documents"][sha]
    document_path = source / document_name
    document_path.parent.mkdir(parents=True)
    document_path.write_bytes(raw)
    write_json(source / "scene.json", scene)
    write_json(source / "manifest.json", {"files": {"scene.json": digest(source / "scene.json"), document_name: sha}})
    original_manifest = digest(source / "manifest.json")
    out = tmp_path / "trained"
    train_shared_world_policy(source, out, seed=42, generations=1, population=4, eval_episodes=2, max_seconds=30)
    assert (out / document_name).read_bytes() == raw
    trained_scene = json.loads((out / "scene.json").read_text())
    assert verify_shared_world_sources(trained_scene, out)["source_evidence_verified"]
    verify_artifacts(out)
    assert digest(source / "manifest.json") == original_manifest


def _boundary_exit_scene():
    scene = authored_shared_world_scene()
    box = next(obj for obj in scene["objects"] if obj["id"] == "marker_box")
    box["position"] = [scene["bounds"][0] - 0.036, 0.15, 1.7]
    box["velocity"] = [3.0, 0.0, 0.0]
    return scene


def test_partial_failure_clock_survives_recording_and_exact_action_replay(tmp_path):
    scene = _boundary_exit_scene()
    env = SharedWorldEnv(scene)
    replay = SharedWorldEnv(scene)
    recorder = Recorder(tmp_path / "partial_failure", scene, profile="shared-world")
    try:
        env.reset(seed=5)
        recorder.start(env.snapshot(), seed=5)
        for _ in range(8):
            action = env.authored_action()
            _, reward, terminated, truncated, _ = env.step(action)
            state = env.snapshot()
            recorder.append({"type": "step", "action": action.tolist(), "reward": reward,
                             "terminated": terminated, "truncated": truncated, "snapshot": state})
            if terminated or truncated:
                break
            assert "partial_action" not in state
            assert state["time_s"] == state["steps"] * env.dt
        recorder.close()
        assert state["coupled_metrics"]["failure"] == "object_left_bounds"
        assert state["partial_action"] == {"completed_substeps": 15, "requested_substeps": 16}
        assert state["time_s"] == env.data.time
        assert state["time_s"] < state["steps"] * env.dt
        assert state["source_time_s"] == env.source_start + env.data.time
        validate_shared_snapshot(state, scene)

        journal = recorder.root / "events.jsonl"
        saved_bytes = journal.read_bytes()
        saved = load_recording(recorder.root)
        replay.reset(seed=saved["events"][0]["seed"])
        assert replay.snapshot() == saved["events"][0]["snapshot"]
        for event in saved["events"][1:]:
            replay.step(event["action"])
            assert replay.snapshot() == event["snapshot"]
        assert journal.read_bytes() == saved_bytes

        # Old failure traces retain their nominal control clock; never rewrite their evidence.
        legacy = deepcopy(state)
        legacy.pop("partial_action")
        legacy["time_s"] = legacy["steps"] * env.dt
        legacy["source_time_s"] = env.source_start + legacy["time_s"]
        validate_shared_snapshot(legacy, scene)
        env.reset(seed=5)
        assert "partial_action" not in env.snapshot()
    finally:
        env.close()
        replay.close()


@pytest.mark.parametrize("change", [
    "ongoing", "success", "truncated", "no_failure", "wrong_count", "wrong_requested",
    "full_duration", "wrong_source_clock",
])
def test_partial_clock_requires_consistent_terminal_failure(change):
    scene = _boundary_exit_scene()
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=5)
        for _ in range(8):
            _, _, terminated, truncated, _ = env.step(env.authored_action())
            if terminated or truncated:
                break
        state = env.snapshot()
        if change == "ongoing":
            state["terminated"] = False
        elif change == "success":
            state["success"] = True
        elif change == "truncated":
            state["terminated"], state["truncated"] = False, True
        elif change == "no_failure":
            state["coupled_metrics"]["failure"] = None
        elif change == "wrong_count":
            state["partial_action"]["completed_substeps"] = 0
        elif change == "wrong_requested":
            state["partial_action"]["requested_substeps"] = 8
        elif change == "full_duration":
            state["time_s"] = state["steps"] * env.dt
        elif change == "wrong_source_clock":
            state["source_time_s"] += env.dt
        with pytest.raises(ValueError, match="[Pp]artial|physics substeps"):
            validate_shared_snapshot(state, scene)
    finally:
        env.close()


def test_partial_step_exception_keeps_last_valid_session_snapshot(tmp_path, monkeypatch):
    import mujoco

    scene = _boundary_exit_scene()
    marker = next(obj for obj in scene["objects"] if obj["id"] == "marker_box")
    marker["position"][0] = 0.2
    marker["velocity"] = [5.99, 0.0, 0.0]
    write_json(tmp_path / "scene.json", scene)
    session = SharedWorldSession(tmp_path)
    try:
        completed = 0
        physics_step = mujoco.mj_step
        def counted_step(model, data):
            nonlocal completed
            physics_step(model, data)
            completed += 1
        monkeypatch.setattr(mujoco, "mj_step", counted_step)
        before = deepcopy(session.snapshot())
        action = [0.0] * len(session.env.action_names)
        action[-3] = 1.0
        with pytest.raises(RuntimeError, match="rigid_velocity_envelope_exceeded"):
            session.step(action)
        assert 0 < completed < session.env.substeps
        assert session.env._last_action_substeps == completed
        assert session.env.data.time == pytest.approx(completed * session.env.dt / session.env.substeps)
        preserved = session.snapshot()
        assert preserved["time_s"] == before["time_s"]
        assert preserved["objects"] == before["objects"]
        assert "partial_action" not in preserved
        assert "Last valid state" in preserved["simulation_failure"]["scope"]
        assert session.reset(5)["time_s"] == 0.0
    finally:
        session.env.close()
