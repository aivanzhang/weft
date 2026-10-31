"""Analytical engine boundary plus real protected scoring; no native or source job."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json
import os
import socket
import subprocess
from types import MappingProxyType, SimpleNamespace

import mujoco
import numpy as np
import pytest

from weft import camera_shared_task as camera
from weft import shared_exercise_task as kernel
from weft import shared_world
from weft.camera_shared_contract import validate_camera_coupling
from weft.io import canonical_hash
from weft.shared_exercise_reset import HUMAN_JITTER_PROFILE
from weft.shared_world_observation import EquipmentState, HumanObjectContact, SubstepSample


def _deny(*_args, **_kwargs):
    raise AssertionError("Native/model/media/network/process execution or repeated preparation is forbidden")


class AnalyticalEngine:
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    instances = []

    def __init__(self, scene, render_mode, *, substep_observer, substep_observer_motor_count):
        assert scene["profile"] == "shared-world-v6" and scene["objects"] == []
        assert substep_observer_motor_count == 22
        self.scene = deepcopy(scene)
        self.observer = substep_observer
        self.human_action_names = self.action_names = tuple(scene["action_contract"]["names"])
        self.human_mass = 70.
        self.camera_pin = canonical_hash(scene["camera_coupling"])
        self.steps, self.time = 0, 0.
        self.actions, self.samples = [], []
        self.mode = None
        self.angular_error = self.root_error = self.penetration = 0.
        self.closed = False
        self.instances.append(self)

    def observation(self):
        if self.mode == "invalid_observation":
            return np.full(158, np.nan)
        return np.r_[np.arange(155) / 100 + self.steps / 100, [9999.] * 3].astype(np.float32)

    def reset(self, *, seed, options):
        if self.mode == "reset_failure":
            raise ValueError("analytical_reset_failure")
        self.reset_call = {"seed": seed, "options": deepcopy(options)}
        self.steps, self.time = 0, 0.
        self.samples.clear()
        return self.observation(), deepcopy(self.reset_call)

    def snapshot(self):
        state = {"profile": "shared-world-v6", "camera_coupling_sha256": self.camera_pin,
                 "time_s": self.time, "source_time_s": self.scene["human_spec"]["reference"][0]["time_s"] + self.time,
                 "steps": self.steps, "objects": [], "raw_engine_marker": "actual_v6_not_legacy",
                 "human": {"mobility": {"max_penetration": self.penetration}}}
        if self.mode == "finite_failure" and self.steps:
            state["partial_action"] = {"completed_substeps": 3, "requested_substeps": self.scene["substeps"]}
        return state

    def authored_human_action(self):
        return np.linspace(-.8, .8, 22)

    def feature_matrix(self):
        return np.arange(66).reshape(22, 3) / 100

    def step(self, action):
        self.actions.append(tuple(action))
        for substep in range(self.scene["substeps"]):
            index = self.steps * self.scene["substeps"] + substep
            tick = self.scene["dt"] / self.scene["substeps"]
            invalid = self.mode == "nonfinite" and substep == 3
            finite_failure = self.mode == "finite_failure" and substep == 2
            sample = SubstepSample(self.steps + 1, substep + 1, index * tick,
                None if invalid else (index + 1) * tick, index * tick, not invalid,
                "nonfinite_dynamics" if invalid else "human_position_envelope_exceeded" if finite_failure else None,
                () if invalid else (self.angular_error,) * 19 + (self.root_error,) * 3,
                () if invalid else (0.,) * 22, () if invalid else (0.,) * 22, tuple(action), (), (),
                MappingProxyType({"left": (0.,) * 3, "right": (0.,) * 3}), None if invalid else self.penetration)
            if self.mode == "bad_clock" and substep == 1:
                sample = replace(sample, contact_time_s=sample.contact_time_s + tick / 3)
            if self.mode == "unexpected_equipment":
                sample = replace(sample, equipment=(EquipmentState("invented", (0.,) * 3, (1., 0., 0., 0.),
                                                                      (0.,) * 3, (0.,) * 3, (0.,) * 3),))
            if self.mode == "unexpected_contact":
                sample = replace(sample, contacts=(HumanObjectContact("left_forearm_geom", "left_forearm", "invented",
                    (0.,) * 3, 0., 10., (0., 0., 10.), (0., 0., 1.)),))
            self.samples.append(sample)
            if not invalid:
                self.time = sample.interval_end_s
            if self.mode != "missing_sample" or substep != 1:
                try:
                    self.observer(sample)
                except Exception as error:
                    self.steps += 1
                    raise RuntimeError("analytical_observer_failure") from error
            if invalid:
                self.steps += 1
                raise RuntimeError("analytical_nonfinite_failure")
            if finite_failure:
                self.steps += 1
                return self.observation(), 12345., True, False, {}
        self.steps += 1
        return self.observation(), 12345., False, self.steps >= self.scene["task"]["horizon_steps"], {}

    def close(self):
        self.closed = True


def _analytical_validation(scene):
    # Only the native full-human boundary is substituted; source coupling is real.
    validate_camera_coupling(scene["human_spec"], scene["camera_coupling"], dt=scene["dt"],
                            coordinate_frame=scene["coordinate_frame"], source_binding=scene["source_binding"])
    return deepcopy(scene)


@pytest.fixture(autouse=True)
def analytical_only(monkeypatch):
    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(
        from_xml_string=_deny, from_xml_path=_deny, from_binary_path=_deny))
    for name in ("MjData", "Renderer", "mj_forward", "mj_step", "mj_resetData", "mj_contactForce", "mj_kinematics"):
        monkeypatch.setattr(mujoco, name, _deny)
    for owner, names in ((socket, ("socket", "create_connection")),
                         (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                         (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, _deny)
    for path, names in (("weft.multiview_pose", ("triangulate_pair", "_triangulate_rays")),
                        ("weft.camera_body", ("make_camera_body", "build_camera_reference")),
                        ("weft.multiview_root", ("make_two_view_root", "triangulate_pair")),
                        ("weft.camera_ground", ("apply_camera_ground", "audit_ground_clearance"))):
        for name in names:
            monkeypatch.setattr(importlib.import_module(path), name, _deny)
    monkeypatch.setattr(shared_world, "SharedWorldEnv", _deny)
    monkeypatch.setattr(kernel, "SharedWorldEnv", _deny)
    monkeypatch.setattr(shared_world, "human_subtree_mass", _deny)
    monkeypatch.setattr(camera, "validate_shared_world", _analytical_validation)
    monkeypatch.setattr(camera, "SharedWorldEnv", AnalyticalEngine)
    AnalyticalEngine.instances.clear()


def scene_fixture():
    tools = importlib.import_module("test_camera_shared_contract")
    coupling_api = importlib.import_module("weft.camera_shared_contract")
    body = tools._fixture(sampled=(1, 2, 3, 4), fps=30000 / 1001)["body"]
    arguments = tools._scene_arguments(coupling_api, body)
    names = list(shared_world.joint_contract(body)[0]) + ["root_x", "root_y", "root_z"]
    return {"schema_version": 1, "profile": "shared-world-v6", "physical_validation": "unverified",
            "human_spec": body, "camera_coupling": coupling_api.build_camera_coupling(body),
            "dt": arguments["dt"], "substeps": 8, "gravity": [0., 0., -9.81],
            "coordinate_frame": arguments["coordinate_frame"], "source_binding": arguments["source_binding"],
            "objects": [], "target": None, "learned_policy_available": False,
            "task": {"start_offset_s": 0., "horizon_steps": 3, "object_goal": None},
            "action_contract": {"names": names, "human_motor_count": 22, "object_force_channels": 0,
                                "dynamic_object_ids": [], "units": ["authored_test_units"] * 22,
                                "physical_limits": [30.] * 19 + [180.] * 3}}


def setup(*, root=False):
    scene = scene_fixture()
    task = camera.build_camera_shared_task(scene, root_assistance=root)
    return scene, task


def environment(*, root=False):
    scene, task = setup(root=root)
    return camera.CameraSharedTaskEnv(scene, task)


def test_context_is_not_legacy_task_and_actual_v6_source_is_unchanged():
    scene, task = setup()
    before = deepcopy((scene, task))
    actual, detached = camera.prepare_camera_shared_task(task, scene)
    assert actual == scene and actual is not scene and detached == task
    context = task["metric_context"]
    arguments = context["kernel_arguments"]
    assert context["profile"] == camera.CONTEXT_PROFILE and context["valid_legacy_task"] is False
    assert context["kernel_branch"] == arguments["profile"] == kernel.HUMAN_ONLY_TASK_PROFILE
    assert context["scene_sha256"] == task["scene_sha256"] == canonical_hash(scene)
    assert not {"scene_sha256", "gates_sha256", "schema_version"} & arguments.keys()
    with pytest.raises(ValueError, match="fields"):
        # Validate only its shape: the original task validator must not accept selector arguments.
        original_validator = kernel.validate_shared_world
        try:
            kernel.validate_shared_world = _analytical_validation
            kernel.validate_shared_exercise_task(arguments, scene)
        finally:
            kernel.validate_shared_world = original_validator
    assert context["source_body_dt_s"] == 1 / 30 and context["control_dt_s"] == 1001 / 30000
    assert task["body_sha256"] == canonical_hash(scene["human_spec"])
    assert task["camera_coupling_sha256"] == canonical_hash(scene["camera_coupling"])
    assert task["source_clock_sha256"] == scene["camera_coupling"]["source_clock_sha256"]
    assert task["metric_context_sha256"] == canonical_hash(context)
    assert task["kernel_evaluator_sha256"] == camera.implementation_identity()["files"]["shared_exercise_task.py"]
    assert task["gates_sha256"] == kernel.GATES_SHA256 and task["gate_profile"] == kernel.GATE_PROFILE
    assert task["scope"]["learning"] == "unqualified" and task["scope"]["ground_support_verified"] is False
    detached["metric_context"]["kernel_arguments"]["interval"]["horizon_steps"] = 1
    assert (scene, task) == before and AnalyticalEngine.instances == []


@pytest.mark.parametrize("field", ["scene_sha256", "body_sha256", "camera_coupling_sha256", "source_clock_sha256",
                                  "metric_context_sha256", "kernel_evaluator_sha256", "adapter_implementation_sha256", "evaluator_sha256"])
def test_pins_reject_before_engine_creation(field):
    scene, task = setup()
    task[field] = "0" * 64
    with pytest.raises(ValueError, match="identities"):
        camera.CameraSharedTaskEnv(scene, task)
    assert AnalyticalEngine.instances == []


@pytest.mark.parametrize("change", ["profile", "interval", "control", "kernel_branch", "arguments", "gate", "scope"])
def test_task_cannot_relabel_or_redefine_scoring(change):
    scene, task = setup()
    if change == "profile":
        task["profile"] = kernel.HUMAN_ONLY_TASK_PROFILE
    elif change == "interval":
        task["interval"]["horizon_steps"] = 1
    elif change == "control":
        task["control"]["root_assistance"] = "explicit_3_axis"
    elif change == "kernel_branch":
        task["metric_context"]["kernel_branch"] = kernel.PROFILE
    elif change == "arguments":
        task["metric_context"]["kernel_arguments"]["equipment_ids"] = ["invented"]
    elif change == "gate":
        task["gates_sha256"] = "0" * 64
    else:
        task["scope"]["ground_support_verified"] = True
    with pytest.raises(ValueError):
        camera.validate_camera_shared_task(task, scene)
    assert AnalyticalEngine.instances == []


@pytest.mark.parametrize("change", ["legacy", "static_object", "dynamic_object", "policy", "root_type", "zero_gravity"])
def test_unsupported_admission_has_no_engine_or_inferred_equipment(change):
    scene = scene_fixture()
    kwargs = {}
    if change == "legacy":
        scene["profile"] = "shared-world-v3"
    elif change.endswith("object"):
        scene["objects"] = [{"id": "equipment", "role": "static" if change == "static_object" else "passive"}]
    elif change == "policy":
        scene["learned_policy_available"] = True
    elif change == "root_type":
        kwargs["root_assistance"] = 1
    else:
        scene["gravity"] = [0., 0., 0.]
    with pytest.raises(ValueError):
        camera.build_camera_shared_task(scene, **kwargs)
    assert AnalyticalEngine.instances == []


@pytest.mark.parametrize("root", [False, True])
def test_one_actual_engine_actions_and_observations_keep_real_layout(root):
    env = environment(root=root)
    assert len(AnalyticalEngine.instances) == 1
    engine = AnalyticalEngine.instances[0]
    observation, info = env.reset(seed=17, options={"jitter": True, "jitter_scale": .5})
    assert env.scene["profile"] == engine.scene["profile"] == "shared-world-v6"
    assert env.task["profile"] == camera.PROFILE
    assert engine.reset_call["options"] == {"jitter": True, "jitter_scale": .5, "position_jitter": 0., "velocity_jitter": 0.}
    assert info["reset_protocol"]["profile"] == camera.RESET_PROFILE
    assert info["reset_protocol"]["equipment_stability_reference"] == "not_applicable"
    assert len(observation) == 156 and 9999. not in observation and observation[-1] == 0.
    np.testing.assert_array_equal(observation[:155], engine.observation()[:155])
    np.testing.assert_array_equal(env.authored_action(), engine.authored_human_action()[:env.action_count])
    np.testing.assert_array_equal(env.feature_matrix(), engine.feature_matrix()[:env.action_count])
    action = np.linspace(-.5, .5, 22 if root else 19)
    _, reward, _, _, info = env.step(action)
    assert engine.actions[-1][:env.action_count] == tuple(action)
    assert engine.actions[-1][19:] == (tuple(action[19:]) if root else (0.,) * 3)
    assert reward != 12345. and info["legacy_coupled_score_used"] is False
    contract = env.action_contract()
    assert contract["joint_channels"] == list(range(19)) and contract["object_force_channels"] == []
    assert contract["root_assistance_channels"] == (list(range(19, 22)) if root else [])
    assert env.observation_contract()["profile"] == camera.OBSERVATION_PROFILE
    assert env.metadata["render_fps"] == pytest.approx(30000 / 1001)
    env.close()
    assert engine.closed


@pytest.mark.parametrize("root", [False, True])
def test_unchanged_kernel_arithmetic_matches_every_scored_value(root):
    env = environment(root=root)
    env.reset(seed=2)
    engine = AnalyticalEngine.instances[0]
    engine.angular_error, engine.root_error = .12, .08
    action = np.linspace(-.2, .2, env.action_count)
    returns = []
    for _ in range(env.horizon_steps):
        _, reward, terminated, truncated, _ = env.step(action)
        returns.append(reward)
    reference = kernel._ExerciseMetrics(env.scene, env.task["metric_context"]["kernel_arguments"], (), 0., engine.human_mass)
    for sample in engine.samples:
        reference.consume(sample)
    expected, actual = reference.metrics(), env.metrics()
    assert expected["profile"] == kernel.HUMAN_ONLY_TASK_PROFILE and actual["profile"] == camera.METRICS_PROFILE
    assert {key: value for key, value in actual.items() if key in expected and key != "profile"} == {
        key: value for key, value in expected.items() if key != "profile"}
    assert actual["observed_substeps"] == actual["valid_substeps"] == 24
    assert actual["equipment"] == actual["contact_requirements"] == []
    assert actual["reward_profile"] == kernel.HUMAN_ONLY_REWARD_PROFILE
    assert actual["gate_applicability"] == dict(kernel.HUMAN_ONLY_GATE_APPLICABILITY)
    assert actual["completed_interval"] and actual["success"] and terminated and not truncated
    assert sum(returns) * env.dt == pytest.approx(actual["reward_integral"])
    assert actual["observed_source_time_s"] == pytest.approx(env.scene["human_spec"]["reference"][-1]["time_s"])
    snapshot = env.snapshot()
    assert snapshot["engine_state"] == engine.snapshot()
    assert snapshot["engine_state"]["raw_engine_marker"] == "actual_v6_not_legacy"
    assert snapshot["profile"] == camera.STATE_PROFILE and snapshot["task_metrics"]["scene_sha256"] == snapshot["scene_sha256"]
    assert env._accumulator.scene["profile"] == "shared-world-v6"


@pytest.mark.parametrize("mode", ["finite_failure", "nonfinite", "missing_sample", "bad_clock", "unexpected_equipment", "unexpected_contact"])
def test_partial_clock_and_failure_evidence_are_retained(mode):
    env = environment()
    env.reset(seed=3)
    initial = env.snapshot()
    engine = AnalyticalEngine.instances[0]
    engine.mode = mode
    _, _, terminated, truncated, _ = env.step(np.zeros(19))
    final = env.snapshot()
    metrics = final["task_metrics"]
    assert terminated and not truncated and not metrics["success"] and not metrics["completed_interval"]
    assert metrics["failure"] and metrics["failure_reward_penalty"] == -1.
    assert metrics["observed_substeps"] <= env.substeps and metrics["valid_substeps"] <= metrics["observed_substeps"]
    if mode == "finite_failure":
        assert final["engine_state_status"] == "current"
        assert final["engine_state"]["partial_action"]["completed_substeps"] == 3
        assert metrics["simulated_duration_s"] == pytest.approx(3 * env.dt / env.substeps)
        assert final["engine_state"]["time_s"] == metrics["simulated_duration_s"]
    if mode == "nonfinite":
        assert final["engine_state_status"] == "last_valid_control_boundary"
        assert final["engine_state"] == initial["engine_state"]
        assert metrics["observed_substeps"] == 4 and metrics["valid_substeps"] == 3
        assert metrics["clock_status"] == "unavailable_after_last_observed_time"
        assert metrics["simulated_duration_s"] == pytest.approx(3 * env.dt / env.substeps)
    assert metrics["observed_source_time_s"] == pytest.approx(
        env.scene["human_spec"]["reference"][0]["time_s"] + metrics["simulated_duration_s"])
    assert env.cancel() == final
    with pytest.raises(RuntimeError, match="Reset before stepping"):
        env.step(np.zeros(19))


def test_failed_gate_is_not_early_termination_or_replaced_by_late_recovery():
    env = environment()
    env.reset(seed=5)
    engine = AnalyticalEngine.instances[0]
    engine.penetration = .03
    _, _, terminated, truncated, _ = env.step(np.zeros(19))
    assert not terminated and not truncated
    engine.penetration = 0.
    for _ in range(env.horizon_steps - 1):
        _, _, terminated, truncated, _ = env.step(np.zeros(19))
    snapshot = env.snapshot()
    assert not terminated and truncated and snapshot["task_metrics"]["completed_interval"]
    assert not snapshot["task_metrics"]["success"]
    assert "max_floor_penetration_model_units" in snapshot["task_metrics"]["failed_gates"]
    assert env.cancel() == snapshot


@pytest.mark.parametrize("steps", [0, 1])
def test_cancel_is_idempotent_no_hidden_action_or_time_and_terminal_success_is_preserved(steps):
    env = environment()
    with pytest.raises(RuntimeError, match="Reset before cancelling"):
        env.cancel()
    env.reset(seed=4)
    for _ in range(steps):
        env.step(np.zeros(19))
    engine = AnalyticalEngine.instances[0]
    before, action_count = env.snapshot(), len(engine.actions)
    cancelled = env.cancel()
    assert cancelled == env.cancel()
    assert len(engine.actions) == action_count
    assert cancelled["engine_state"] == before["engine_state"]
    assert cancelled["task_metrics"]["observed_substeps"] == before["task_metrics"]["observed_substeps"]
    assert cancelled["task_metrics"]["observed_source_time_s"] == before["task_metrics"]["observed_source_time_s"]
    assert cancelled["task_metrics"]["reward_integral"] == pytest.approx(before["task_metrics"]["reward_integral"] - env.dt)
    assert cancelled["task_metrics"]["failure"] == "cancelled" and cancelled["task_metrics"]["failure_reward_penalty"] == -1.
    env.reset(seed=4)
    for _ in range(env.horizon_steps):
        env.step(np.zeros(19))
    completed = env.snapshot()
    assert completed["task_metrics"]["success"] and env.cancel() == completed
    assert cancelled["task_metrics"]["failure"] == "cancelled"


@pytest.mark.parametrize("mode", ["reset_failure", "invalid_observation"])
def test_failed_reset_leaves_no_reusable_task_state(mode):
    env = environment()
    env.reset(seed=2)
    engine = AnalyticalEngine.instances[0]
    engine.mode = mode
    with pytest.raises((ValueError, RuntimeError)):
        env.reset(seed=7)
    for query in (env.snapshot, env.metrics, env.observation, env.cancel):
        with pytest.raises(RuntimeError, match="Reset before"):
            query()
    with pytest.raises(RuntimeError, match="Reset before stepping"):
        env.step(np.zeros(19))


@pytest.mark.parametrize("action", [[True] * 19, [0.] * 22, [float("nan")] * 19, [1.01] * 19])
def test_invalid_actions_never_reach_engine(action):
    env = environment()
    env.reset(seed=1)
    with pytest.raises(ValueError, match="exactly 19"):
        env.step(action)
    assert AnalyticalEngine.instances[0].actions == []


@pytest.mark.parametrize("root", [False, True])
def test_extreme_integer_and_unsupported_dtypes_preserve_active_state(root, monkeypatch):
    env = environment(root=root)
    env.reset(seed=19)
    engine = AnalyticalEngine.instances[0]
    before, calls = env.snapshot(), []

    def engine_rejects(action):
        calls.append(tuple(action))
        raise ValueError("Engine rejects an out-of-range action before physics")

    monkeypatch.setattr(engine, "step", engine_rejects)
    count = env.action_count
    for action in (np.full(count, -(2**63), dtype=np.int64), [-(2**63)] * count,
                   np.full(count, 2**64 - 1, dtype=np.uint64), np.zeros(count, dtype=bool),
                   [True] + [0] * (count - 1), np.ones(count, dtype=complex), np.full(count, np.nan)):
        with pytest.raises(ValueError):
            env.step(action)
        assert env.snapshot() == before
        assert engine.time == 0. and engine.actions == [] and calls == []
        assert not env._done


def test_render_cached_and_hot_paths_never_rehash_reprepare_or_mutate_engine(monkeypatch):
    env = environment()
    env.reset(seed=8)
    engine = AnalyticalEngine.instances[0]
    preparations, renders = [], []

    def prepare(scene):
        assert scene["profile"] == "shared-world-v6" and "camera_reference" in scene["human_spec"]
        preparations.append(scene)

        def render(state):
            renders.append(deepcopy(state))
            state["mutated_renderer_copy"] = True
            return np.zeros((2, 2, 3), dtype=np.uint8)

        return render

    monkeypatch.setattr(camera, "prepare_shared_renderer", prepare)
    for name in ("implementation_identity", "canonical_hash", "validate_shared_world", "prepare_camera_shared_task"):
        monkeypatch.setattr(camera, name, _deny)
    before = env.snapshot()
    assert np.array_equal(env.render(), env.render())
    assert len(preparations) == 1 and len(renders) == 2 and env.snapshot() == before
    env.step(np.zeros(19))
    env.metrics()
    env.observation()
    env.authored_action()
    env.feature_matrix()
    env.snapshot()
    env.render()
    env.cancel()
    env.reset(seed=8)
    assert len(engine.actions) == 1 and len(preparations) == 1


def test_oversized_task_rejects_before_native_scene_preparation(monkeypatch):
    scene, task = setup()
    task["extra"] = "x" * camera.MAX_TASK_BYTES
    monkeypatch.setattr(camera, "validate_shared_world", _deny)
    with pytest.raises(ValueError, match="64 KiB"):
        camera.prepare_camera_shared_task(task, scene)
