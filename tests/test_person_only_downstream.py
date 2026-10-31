"""Person-only downstream composition; no video inference, training trials or dynamics steps."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from weft.capabilities import describe_capabilities
from weft.io import canonical_hash, digest, write_json
from weft import exercise_learning_pipeline as workflow
from weft import recording
from weft import shared_exercise_lab as lab
from weft import shared_exercise_learning as gains
from weft import shared_exercise_feasibility_learning as feasibility
from weft import shared_exercise_run as baselines
from weft.shared_exercise_policies import prepare_policy, validate_policy
from weft import shared_exercise_residual_learning as residual
from weft import shared_exercise_selection as selection
from weft import shared_exercise_task as tasks
from weft.shared_exercise_recording import recording_scene
from weft.shared_exercise_reset import describe_reset_protocol
from weft.shared_world import HUMAN_ONLY_PROFILE, SharedWorldEnv
from test_exercise_learning_pipeline import _stub_stages
from test_shared_exercise_selection import NoPhysicsEnv
from test_shared_exercise_run import source_fixture
from test_shared_world_human_only import person_scene
from test_shared_world_fullspan import fullspan_scene


@pytest.fixture(autouse=True)
def no_dynamics(monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda *_a, **_k: pytest.fail("Downstream tests must not advance physics"))


def source_package(tmp_path, *, steps=3):
    source = tmp_path / "source"
    source.mkdir()
    scene = person_scene(steps=steps)
    task = tasks.build_shared_exercise_task(scene)
    write_json(source / "scene.json", scene)
    write_json(source / "exercise-task.json", task)
    write_json(source / "inventory.json", scene["inventory"])
    write_json(source / "source-clock.json", {"scope": "Analytic fixture: no native video clock"})
    write_json(source / "manifest.json", {"status": "ready", "files": {
        path.name: digest(path) for path in source.iterdir()}})
    return source, scene, task


def policy_fixture(scene, task):
    env = NoPhysicsEnv(scene, task)
    return residual._policy(env, np.zeros((2, 19)), 1729, .5, residual._runtime_hashes(scene))


@pytest.mark.parametrize("assistance", [False, True])
def test_capabilities_are_one_human_with_ground_and_no_object_reward(assistance):
    scene = person_scene()
    task = tasks.build_shared_exercise_task(scene, root_assistance=assistance)
    base = describe_capabilities(scene)
    assert base["representation_family"] == "articulated_human_3d"
    assert base["observation_contract"]["dimension"] == 155
    assert base["task_contract"]["task_type"] == "human_motion_v1"
    assert base["task_contract"]["success_criteria"]["object_goal"] is None
    assert base["action_contract"]["dimension"] == 22
    doc = describe_capabilities(scene, exercise_task=task, learned_policy_available=True)
    assert doc["entities"]["total_count"] == doc["entities"]["dynamic_count"] == 1
    assert doc["entities"]["static_count"] == 0
    assert doc["entities"]["support"] == "implicit_ground_plane"
    assert doc["entities"]["rigid_objects_present"] is doc["entities"]["mixed_rigid_articulated"] is False
    assert doc["entities"]["multi_target_selection_endpoint"] is None
    assert [row["id"] for row in doc["entities"]["controllable_actors"]] == ["human"]
    assert doc["action_contract"]["dimension"] == (22 if assistance else 19)
    assert doc["action_contract"]["object_force_channels"] == []
    assert doc["observation_contract"]["dimension"] == 156
    assert doc["task_contract"]["success_criteria"]["gate_applicability"] == dict(tasks.HUMAN_ONLY_GATE_APPLICABILITY)
    assert set(doc["task_contract"]["success_criteria"]["frozen_gates"]) == set(selection.BODY_GATES)
    assert doc["controllers"]["trained_target_id"] == "human"
    assert doc["evidence_separation"]["object_absence_verified"] is False


def test_new_policy_pins_inventory_without_extending_legacy_closure():
    scene = person_scene()
    task = tasks.build_shared_exercise_task(scene)
    legacy = fullspan_scene(steps=3)
    assert len(lab.runtime_files(legacy)) == len(lab.RUNTIME_FILES) == 17
    assert len(residual._runtime_hashes()) == len(residual.POLICY_RUNTIME_FILES) == 21
    assert set(lab.runtime_files(scene)) - set(lab.RUNTIME_FILES) == {"person_inventory.py"}
    hashes = residual._runtime_hashes(scene)
    assert len(hashes) == 22 and hashes["person_inventory.py"] == digest(Path(lab.__file__).with_name("person_inventory.py"))
    policy = policy_fixture(scene, task)
    assert residual.validate_shared_exercise_residual_policy(policy, scene, task) == policy
    policy["runtime_sha256"].pop("person_inventory.py")
    with pytest.raises(ValueError, match="runtime differs"):
        residual.validate_shared_exercise_residual_policy(policy, scene, task)


@pytest.mark.parametrize("assistance", [False, True])
def test_existing_residual_actor_is_detached_bounded_and_keeps_root_exact(monkeypatch, assistance):
    scene = person_scene()
    task = tasks.build_shared_exercise_task(scene, root_assistance=assistance)
    meta = NoPhysicsEnv(scene, task)
    action = np.linspace(-.6, .6, meta.action_count)
    action[-1] = -0.
    env = SimpleNamespace(**vars(meta), observation_contract=lambda: {"profile": "shared-exercise-observation-v1"},
                          authored_action=lambda: action.copy(), observation=lambda: np.array([.5]))
    policy = policy_fixture(scene, task)
    actor = residual.prepare_shared_exercise_residual_policy(policy, env)
    monkeypatch.setattr(residual, "_runtime_hashes", lambda *_: pytest.fail("Actor rehashed runtime in hot path"))
    policy["residual_coefficients"][0][0] = .25
    assert actor().tobytes() == action.tobytes()
    changed = residual._action(np.full((2, 19), .25), env)
    assert np.all(np.abs(changed) <= 1)
    assert changed[19:].tobytes() == action[19:].tobytes()


def metric_fixture(*, complete):
    scene = person_scene()
    task = tasks.build_shared_exercise_task(scene)
    policy = policy_fixture(scene, task)
    reset = residual.TRAIN_RESETS[0]
    measured = tasks._ExerciseMetrics(scene, task, (), 0., 1.)
    if complete:
        measured.samples = measured.valid_samples = task["interval"]["horizon_steps"] * scene["substeps"]
        measured.elapsed = measured.valid_duration = task["interval"]["horizon_steps"] * scene["dt"]
        measured.angular_squared = .2 ** 2 * measured.valid_duration
        measured.root_squared = .03 ** 2 * measured.valid_duration
        measured.angular_peak, measured.root_peak = .2, .03
        measured.reward_integral = .7 * measured.valid_duration
    fixed = {name: policy[name] for name in ("scene_sha256", "task_sha256", "evaluator_sha256")}
    return scene, task, policy, reset, measured.metrics() | fixed | {"reset_protocol": describe_reset_protocol(reset)}


@pytest.mark.parametrize("complete", [False, True])
def test_selection_accepts_exact_v2_metrics_with_not_applicable_gates(complete):
    scene, task, policy, reset, metrics = metric_fixture(complete=complete)
    selection._metrics(metrics, scene, task, policy, reset, initial=not complete)
    assert metrics["equipment"] == metrics["contact_requirements"] == []
    assert metrics["success"] is complete
    assert metrics["gate_applicability"]["equipment_stability"] == "not_applicable"


@pytest.mark.parametrize("change", [
    lambda m: m.pop("gate_applicability"),
    lambda m: m["gate_applicability"].update(equipment_stability="passed"),
    lambda m: m.update(reward_profile="equipment_stability_bonus"),
    lambda m: m.update(profile="shared-exercise-task-v1"),
    lambda m: m["equipment"].append({"object_id": "invented"}),
    lambda m: m.update(reward_integral=1.1 * m["declared_duration_s"], mean_reward=1.1),
])
def test_selection_rejects_laundered_profile_equipment_credit_and_reward(change):
    scene, task, policy, reset, metrics = metric_fixture(complete=True)
    change(metrics)
    with pytest.raises(ValueError):
        selection._metrics(metrics, scene, task, policy, reset)


def test_person_views_are_carried_without_changing_legacy_view_selection(tmp_path):
    source, scene, _ = source_package(tmp_path)
    manifest = (source / "manifest.json").read_bytes()
    assert residual._source_views(source, manifest) == {}
    views = residual._source_views(source, manifest, scene)
    assert set(views) == {"inventory.json", "source-clock.json"}
    assert all(raw == (source / name).read_bytes() for name, raw in views.items())


def test_new_scene_and_task_flow_through_one_lab_and_long_fullspan_budget(tmp_path, monkeypatch):
    source, scene, task = source_package(tmp_path, steps=200)
    calls = _stub_stages(monkeypatch)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    out = tmp_path / "workflow"
    result = workflow.prepare_exercise_learning_pipeline(source, out, train_settings={
        "generations": 1, "population": 4, "trials": 3, "evaluation_seed": 924017})
    assert result["status"] == "ready"
    assert calls == ["training", "training_selection", "independent_evaluation", "interactive_lab"]
    plan = json.loads((out / "plan.json").read_bytes())
    assert plan["maximum_control_steps"] == 4000
    assert json.loads((out / "training/settings.json").read_bytes())["maximum_control_steps"] == 1600
    assert plan["maximum_evaluation_steps"] == 2400
    assert plan["policy_runtime_sha256"]["person_inventory.py"] == residual._runtime_hashes(scene)["person_inventory.py"]
    for child in ("training", "selected", "evaluation", "lab/run"):
        assert (out / child / "scene.json").read_bytes() == before["scene.json"]
        assert (out / child / "exercise-task.json").read_bytes() == before["exercise-task.json"]
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}
    assert len(list(out.glob("*/run/scene.json"))) == 1


def test_lab_package_retains_inventory_runtime_and_exact_source_task(tmp_path):
    source, scene, task = source_package(tmp_path)
    out = tmp_path / "lab"
    result = lab.prepare_shared_exercise_lab(source, out)
    loaded = lab.load_shared_exercise_lab(out / "run")
    assert result["scene_sha256"] == canonical_hash(scene)
    assert loaded["scene"] == scene and loaded["task"] == task
    contract = json.loads((out / "run/exercise-lab.json").read_bytes())
    assert set(contract["runtime_sha256"]) == set(lab.runtime_files(scene))
    for name in ("scene.json", "exercise-task.json", "inventory.json", "source-clock.json"):
        assert (out / "run" / name).read_bytes() == (source / name).read_bytes()
    contract["runtime_sha256"].pop("person_inventory.py")
    write_json(out / "run/exercise-lab.json", contract)
    manifest = json.loads((out / "run/manifest.json").read_bytes())
    manifest["files"]["exercise-lab.json"] = digest(out / "run/exercise-lab.json")
    write_json(out / "run/manifest.json", manifest)
    with pytest.raises(ValueError, match="runtime differs"):
        lab.load_shared_exercise_lab(out / "run")


@pytest.mark.parametrize("policy_profile", ["shared-exercise-policy-v1", "shared-exercise-policy-v3"])
def test_person_lab_does_not_admit_other_policy_runtime_closures(policy_profile):
    scene, task = person_scene(), None
    task = tasks.build_shared_exercise_task(scene)
    files = {"scene.json": lab._bytes(scene), "exercise-task.json": lab._bytes(task),
             "policy.json": lab._bytes({"profile": policy_profile}), "training.json": b"{}",
             "settings.json": b"{}", "input-manifest.json": b"{}"}
    manifest = lab._bytes({"status": "trained", "files": {name: lab._sha(raw) for name, raw in files.items()}})
    with pytest.raises(ValueError, match="newly pinned v2"):
        lab._validate_policy_package(manifest, files, scene, task)


@pytest.mark.parametrize("profile", ["planar-rigid-v1", "exercise-v1", "unknown"])
def test_unvalidated_nested_scene_cannot_authorize_empty_recording(tmp_path, profile):
    scene = {"profile": profile, "objects": [], "engine_scene": {"profile": HUMAN_ONLY_PROFILE}}
    with pytest.raises(ValueError, match="requires 1"):
        recording.Recorder(tmp_path / "recording", scene)


@pytest.mark.parametrize("profile", [gains.POLICY_PROFILE, feasibility.POLICY_PROFILE])
def test_unsupported_policy_dispatch_actor_and_recording_fail_before_authorization(profile):
    scene = person_scene()
    task = tasks.build_shared_exercise_task(scene)
    document = {"profile": profile}
    env = SimpleNamespace(scene=scene, task=task)
    for operation in (lambda: validate_policy(document, scene, task), lambda: prepare_policy(document, env),
                      lambda: recording_scene(scene, task, policy_document=lab._bytes(document))):
        with pytest.raises(ValueError, match="require the v2 residual learner"):
            operation()


@pytest.mark.parametrize("learner", ["gain_v1", "feasibility_v3"])
def test_unsupported_trainers_reject_before_output_or_environment(tmp_path, monkeypatch, learner):
    source, scene, task = source_package(tmp_path)
    for module in (gains, feasibility):
        monkeypatch.setattr(module, "SharedExerciseTaskEnv", lambda *_: pytest.fail("Unsupported trainer constructed physics"))
    if learner == "gain_v1":
        train = gains.train_shared_exercise_policy
    else:
        for name in ("policy.json", "training.json", "settings.json", "input-manifest.json"):
            write_json(source / name, {})
        write_json(source / "manifest.json", {"status": "trained", "files": {
            p.name: digest(p) for p in source.iterdir() if p.name != "manifest.json"}})
        train = feasibility.train_shared_exercise_feasibility_policy
    out = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="require the v2 residual learner"):
        train(source, out)
    assert not out.exists()


@pytest.mark.parametrize("person_only", [False, True])
def test_baseline_entrypoint_pins_inventory_only_for_new_profile_before_physics(tmp_path, monkeypatch, person_only):
    source = source_package(tmp_path)[0] if person_only else source_fixture(tmp_path)[0]
    def stop(*_args, **_kwargs):
        raise RuntimeError("Intentional stop before environment construction")
    monkeypatch.setattr(baselines, "SharedExerciseTaskEnv", stop)
    out = tmp_path / "baseline"
    with pytest.raises(RuntimeError, match="Intentional stop"):
        baselines.evaluate_shared_exercise_run(source, out)
    pins = json.loads((out / "settings.json").read_bytes())["implementation_sha256"]
    assert ("person_inventory.py" in pins) is person_only
    expected = set(baselines.IMPLEMENTATION_FILES) | {"project/pyproject.toml", "project/requirements-lab.lock"}
    assert set(pins) == expected | ({"person_inventory.py"} if person_only else set())
    if person_only:
        assert digest(out / "implementation/person_inventory.py") == pins["person_inventory.py"]
    assert json.loads((out / "manifest.json").read_bytes())["status"] == "failed"
    assert not (out / "recordings").exists()


@pytest.mark.parametrize("scene_profile", ["shared-world-v1", "shared-world-v2"])
@pytest.mark.parametrize("kind", ["gains", "feasibility"])
def test_new_policy_scope_restriction_preserves_legacy_profiles(scene_profile, kind):
    scene = fullspan_scene(steps=3)
    scene["profile"] = scene_profile
    task = tasks.build_shared_exercise_task(scene)
    env = NoPhysicsEnv(scene, task)
    if kind == "gains":
        policy = gains._policy(scene, task, env, np.zeros((3, 3)), 1729, .5, gains._runtime_hashes())
    else:
        rank = {"profile": feasibility.RANK_PROFILE, "complete": True, "feasible": True,
                "max_normalized_excess": 0., "mean_normalized_excess": 0., "violated_terms": 0,
                "gate_terms": 2 * (len(selection.BODY_GATES) + len(selection.EQUIPMENT_GATES) * len(task["equipment_ids"])),
                "reward_score": .5}
        selected = {"residual_coefficients": np.zeros((2, 19)).tolist(), "score": .5, "optimization_rank": rank}
        policy = feasibility._policy(env, selected, 1729, feasibility._runtime_hashes(),
                                     {"original_training_manifest_sha256": "a" * 64})
    assert "person_inventory.py" not in policy["runtime_sha256"]
    assert validate_policy(policy, scene, task) == policy


@pytest.mark.parametrize("wrapped", [False, True])
def test_empty_object_recording_roundtrip_render_and_replay_keep_only_body(tmp_path, wrapped):
    scene = person_scene()
    task = tasks.build_shared_exercise_task(scene)
    env = tasks.SharedExerciseTaskEnv(scene, task) if wrapped else SharedWorldEnv(scene)
    try:
        env.reset(seed=7)
        state = env.snapshot()
        doc = recording_scene(scene, task) if wrapped else scene
        recorder = recording.Recorder(tmp_path / "session", doc, profile="shared-exercise" if wrapped else "shared-world")
        recorder.start(state, seed=7)
        recorder.append({"type": "reset", "seed": 7, "snapshot": deepcopy(state)})
        recorder.close()
        loaded = recording.load_recording(tmp_path / "session")
        saved_scene, events = loaded["scene"], loaded["events"]
        assert saved_scene == doc and [row["snapshot"] for row in events] == [state, state]
        image = recording.render_snapshot(state, saved_scene, width=480)
        assert image.shape[1] == 480 and image.size > 0
        replay = recording._replay_html(saved_scene, events)
        assert '"target":null' in replay and HUMAN_ONLY_PROFILE in replay
        recording._action([0.] * (19 if wrapped else 22), saved_scene)
        with pytest.raises(ValueError):
            recording._action([0.] * (22 if wrapped else 25), saved_scene)
    finally:
        env.close()
