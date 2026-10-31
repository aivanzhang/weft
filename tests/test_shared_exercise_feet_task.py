"""Pure task orchestration: one analytical engine stand-in, unchanged real scorers."""

from copy import deepcopy
import json
from pathlib import Path
import socket
import subprocess
from time import perf_counter, process_time
from types import MappingProxyType, SimpleNamespace

import mujoco
import numpy as np
import pytest

from test_shared_exercise_feet_projection import scene_fixture
from test_shared_support_task import contact
from weft import shared_exercise_feet_task as feet
from weft import shared_exercise_task as original
from weft import shared_world
from weft.io import canonical_hash
from weft.shared_exercise_feet_projection import project_substep
from weft.shared_exercise_reset import HUMAN_JITTER_PROFILE
from weft.shared_world_observation import EquipmentState, SubstepSample


class AnalyticalEngine:
    instances = []

    def __init__(self, scene, render_mode, *, substep_observer, substep_observer_motor_count):
        assert scene["profile"] == "shared-world-v5" and substep_observer_motor_count == 28
        self.scene = deepcopy(scene)
        self.observer = substep_observer
        self.human_action_names = tuple(scene["action_contract"]["names"][:28])
        self.action_names = tuple(scene["action_contract"]["names"])
        self.human_mass = 71.
        self.steps, self.time = 0, 0.
        self.actions, self.samples = [], []
        self.mode, self.contacts_at = None, lambda _: ()
        self.ankle_error, self.penetration = .3, 0.
        self.closed = False
        self.instances.append(self)

    def observation(self):
        if self.mode == "invalid_observation":
            return np.full(217, np.nan)
        return np.r_[np.arange(197) / 100 + self.steps / 100, [9999.] * 20].astype(np.float32)

    def reset(self, *, seed, options):
        if self.mode == "reset_failure":
            raise ValueError("authored_reset_failure")
        self.reset_call = {"seed": seed, "options": deepcopy(options)}
        self.steps, self.time = 0, 0.
        self.samples.clear()
        return self.observation(), deepcopy(self.reset_call)

    def snapshot(self):
        return {"profile": "shared-world-state-v1", "scene_profile": "shared-world-v5",
                "scene_sha256": canonical_hash(self.scene), "time_s": self.time, "steps": self.steps,
                "human": {"action_names": list(self.human_action_names),
                          "mobility": {"max_penetration": self.penetration},
                          "feet": {"reference_sha256": self.scene["foot_extension"]["reference_sha256"],
                                   "initial_ankle_targets_rad": [0.] * 6}},
                "objects": deepcopy(self.scene["objects"])}

    def authored_human_action(self):
        return np.linspace(-.8, .8, 28)

    def feature_matrix(self):
        return np.arange(84).reshape(28, 3) / 100

    def step(self, action):
        self.actions.append(tuple(action))
        equipment = tuple(EquipmentState(obj["id"], tuple(obj["position"]), tuple(obj["quaternion"]),
                                         tuple(obj["velocity"]), tuple(obj["angular_velocity"]), (0.,) * 3)
                          for obj in self.scene["objects"])
        for substep in range(self.scene["substeps"]):
            index = self.steps * self.scene["substeps"] + substep
            tick = self.scene["dt"] / self.scene["substeps"]
            invalid = self.mode == "nonfinite" and substep == 3
            finite_failure = self.mode == "finite_failure" and substep == 1
            sample = SubstepSample(self.steps + 1, substep + 1, index * tick,
                None if invalid else (index + 1) * tick, index * tick, not invalid,
                "nonfinite_dynamics" if invalid else "finite_dynamics_failure" if finite_failure else None,
                () if invalid else (0.,) * 22 + (self.ankle_error,) * 6,
                () if invalid else (0.,) * 28, () if invalid else (0.,) * 28,
                tuple(action[:28]), () if invalid else equipment, () if invalid else self.contacts_at(index),
                MappingProxyType({"left": (0.,) * 3, "right": (0.,) * 3}),
                None if invalid else self.penetration)
            self.samples.append(sample)
            if not invalid:
                self.time = sample.interval_end_s
            try:
                self.observer(sample)
            except Exception as error:
                raise RuntimeError("analytical_observer_failure") from error
            if invalid:
                raise RuntimeError("analytical_nonfinite_failure")
            if finite_failure:
                self.steps += 1
                return self.observation(), 12345., True, False, {}
        self.steps += 1
        return self.observation(), 12345., False, self.steps >= self.scene["task"]["horizon_steps"], {}

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def pure_orchestration_only(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Feet task tests cannot compile models, run physics, use network or start workers")

    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(
        from_xml_string=forbidden, from_xml_path=forbidden, from_binary_path=forbidden))
    for name in ("MjData", "Renderer", "mj_forward", "mj_step", "mj_resetData", "mj_contactForce", "mj_kinematics"):
        monkeypatch.setattr(mujoco, name, forbidden)
    monkeypatch.setattr(shared_world, "SharedWorldEnv", forbidden)
    monkeypatch.setattr(original, "SharedWorldEnv", forbidden)
    monkeypatch.setattr(shared_world, "human_subtree_mass", lambda _spec, scale=1.: 70. * scale**3)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(feet, "SharedWorldEnv", AnalyticalEngine)
    AnalyticalEngine.instances.clear()


def setup(profile="shared-world-v3", *, root=False, substeps=8):
    base, scene = scene_fixture(profile)
    if substeps != 8:
        base["substeps"] = scene["substeps"] = substeps
        base["coordinate_frame"]["clock_binding"]["physics_substeps"] = substeps
        scene["coordinate_frame"]["clock_binding"]["physics_substeps"] = substeps
        for value in (base, scene):
            value.pop("action_contract")
        base, scene = shared_world.validate_shared_world(base), shared_world.validate_shared_world(scene)
    kwargs = {"root_assistance": root}
    if profile == "shared-world-v4":
        kwargs["support_requirements"] = [{"object_id": "support_a", "human_segment": "left_forearm",
            "direction_world": [0., 0., 1.], "minimum_load_body_weight_fraction": .05,
            "start_step": 0, "end_step": 2}]
    base_task = original.build_shared_exercise_task(base, **kwargs)
    task = feet.build_shared_exercise_feet_task(scene, base_task)
    return base, scene, base_task, task


def foot_contact(**changes):
    return contact(human_geom="left_foot_geom", human_segment="left_foot", normal_force=100.,
                   force_on_human_world=(0., 0., 100.), **changes)


@pytest.mark.parametrize("profile", ["shared-world-v2", "shared-world-v3", "shared-world-v4"])
def test_task_binds_exact_actual_scene_original_task_and_derived_scoring_identity(profile):
    base, scene, base_task, task = setup(profile)
    before = deepcopy((scene, base_task, task))
    actual, scoring, validated = feet.prepare_shared_exercise_feet_task(task, scene)
    assert actual == scene and scoring == base and validated == task
    assert task["scene_sha256"] == canonical_hash(scene) != task["base_scene_sha256"]
    assert task["base_scene_sha256"] == base_task["scene_sha256"] == canonical_hash(base)
    assert task["base_task"] == base_task and task["base_task_sha256"] == canonical_hash(base_task)
    assert task["adapter_implementation_sha256"] == feet.adapter_implementation_identity()["sha256"]
    assert task["foot_reference_sha256"] == scene["foot_extension"]["reference_sha256"]
    assert task["control"]["root_assistance"] == "disabled"
    assert task["interval"] == base_task["interval"] and task["scope"]["foot_support_credit"] is False
    validated["base_task"]["interval"]["horizon_steps"] = 1
    assert (scene, base_task, task) == before and AnalyticalEngine.instances == []


@pytest.mark.parametrize("field", ["scene_sha256", "base_scene_sha256", "base_task_sha256",
                                  "foot_reference_sha256", "projection_contract_sha256", "adapter_implementation_sha256"])
def test_identity_changes_reject_before_any_engine_is_created(field):
    _, scene, _, task = setup()
    task[field] = "0" * 64
    with pytest.raises(ValueError, match="identities"):
        feet.SharedExerciseFeetTaskEnv(scene, task)
    assert AnalyticalEngine.instances == []


def test_task_cannot_change_interval_controls_original_gates_or_semantically_relabel_legacy():
    base, scene, base_task, task = setup()
    changes = [lambda t: t["interval"].update(horizon_steps=1),
               lambda t: t["control"].update(root_assistance="explicit_3_axis"),
               lambda t: t["base_task"].update(gates_sha256="0" * 64),
               lambda t: t["scope"].update(foot_support_credit=0),
               lambda t: t.update(profile=original.HUMAN_ONLY_TASK_PROFILE)]
    for change in changes:
        altered = deepcopy(task)
        change(altered)
        with pytest.raises(ValueError):
            feet.validate_shared_exercise_feet_task(altered, scene)
    with pytest.raises(ValueError, match="shared-world-v5"):
        feet.build_shared_exercise_feet_task(base, base_task)
    assert AnalyticalEngine.instances == []


@pytest.mark.parametrize("root", [False, True])
def test_one_actual_engine_routes_every_public_action_feature_and_authored_channel(root):
    _, scene, _, task = setup("shared-world-v2", root=root)
    env = feet.SharedExerciseFeetTaskEnv(scene, task)
    assert len(AnalyticalEngine.instances) == 1 and env.scene == scene and env.task == task
    engine = AnalyticalEngine.instances[0]
    env.reset(seed=17)
    expected_indices = tuple(range(19)) + tuple(range(22, 28)) + ((19, 20, 21) if root else ())
    np.testing.assert_array_equal(env.authored_action(), engine.authored_human_action()[list(expected_indices)])
    np.testing.assert_array_equal(env.feature_matrix(), engine.feature_matrix()[list(expected_indices)])
    values = np.linspace(-.5, .5, 28 if root else 25)
    _, reward, _, _, _ = env.step(values)
    full = engine.actions[-1]
    assert tuple(full[i] for i in expected_indices) == tuple(values)
    assert full[28:] == (0.,) * 3 and full[19:22] == (tuple(values[25:]) if root else (0.,) * 3)
    contract = env.action_contract()
    assert contract["engine_motor_indices"] == list(expected_indices)
    assert contract["root_assistance_channels"] == (list(range(25, 28)) if root else [])
    assert contract["ankle_channels"] == list(range(19, 25)) and contract["object_force_channels"] == []
    assert contract["names"] == [scene["action_contract"]["names"][i] for i in expected_indices]
    assert reward != 12345.
    env.close()
    assert engine.closed


@pytest.mark.parametrize("profile", ["shared-world-v2", "shared-world-v3", "shared-world-v4"])
def test_full_197_value_human_block_and_exact_base_metrics_provenance(profile):
    base, scene, base_task, task = setup(profile)
    env = feet.SharedExerciseFeetTaskEnv(scene, task)
    observation, _ = env.reset(seed=9)
    engine = AnalyticalEngine.instances[0]
    np.testing.assert_array_equal(observation[:197], engine.observation()[:197])
    assert 9999. not in observation
    assert env.observation_contract()["profile"] == feet.OBSERVATION_PROFILE
    assert env.observation_contract()["fields"][0]["count"] == 197
    assert len(observation) == 198 + (17 if profile == "shared-world-v2" else 6 if profile == "shared-world-v4" else 0)
    snapshot = env.snapshot()
    assert snapshot["profile"] == feet.STATE_PROFILE != original.STATE_PROFILE
    assert snapshot["engine_state"] == engine.snapshot() and snapshot["scene_sha256"] == canonical_hash(scene)
    assert snapshot["task_metrics"]["scene_sha256"] == canonical_hash(base)
    assert snapshot["task_metrics"]["task_sha256"] == canonical_hash(base_task)
    assert snapshot["task_metrics"]["gates_sha256"] == base_task["gates_sha256"]
    assert snapshot["task_metrics"]["evaluator_sha256"] == env.evaluator_sha256
    assert snapshot["ankle_diagnostics"]["latest_substep"] is None
    assert snapshot["ankle_diagnostics"]["source"] == "simulated_engine_substeps"
    assert snapshot["ankle_diagnostics"]["ankle_effort_mean_squared"] is None
    snapshot["engine_state"]["human"]["feet"]["initial_ankle_targets_rad"][0] = 1.
    assert env.snapshot()["engine_state"] == engine.snapshot()


@pytest.mark.parametrize("retained_forearm", [False, True])
def test_original_support_scorer_gets_no_foot_credit_and_earlier_diagnostics_survive(retained_forearm):
    base, scene, base_task, task = setup("shared-world-v4")
    env = feet.SharedExerciseFeetTaskEnv(scene, task)
    env.reset(seed=2)
    engine = AnalyticalEngine.instances[0]
    forearm = (contact(normal_force=100., force_on_human_world=(0., 0., 100.)),) if retained_forearm else ()
    engine.contacts_at = lambda index: forearm + ((foot_contact(),) if index < 3 else ())
    expected = original._support_module().SupportMetrics(base, base_task,
        tuple(EquipmentState(o["id"], tuple(o["position"]), tuple(o["quaternion"]), (0.,) * 3, (0.,) * 3, (0.,) * 3)
              for o in scene["objects"]), 0., engine.human_mass)
    action = np.r_[(0.,) * 19, (.75,) * 6]
    env.step(action)
    for sample in engine.samples:
        expected.consume(project_substep(sample)[0])
    metrics = env.metrics()
    assert all(metrics[key] == value for key, value in expected.metrics().items())
    assert env._accumulator.weight == pytest.approx(71. * 9.81)
    assert metrics["support_requirements"][0]["qualifying_now"] is retained_forearm
    diagnostics = env.ankle_diagnostics()
    assert diagnostics["latest_substep"]["excluded_foot_contacts"] == []
    row = diagnostics["contact_aggregates"][0]
    assert row["contact_records"] == diagnostics["excluded_foot_contact_records"] == 3
    assert row["first_contact_time_s"] == 0. and row["last_contact_time_s"] == pytest.approx(2 / 240)
    assert row["peak_normal_force"] == 100.
    assert diagnostics["contact_aggregates"][1]["peak_normal_force"] is None
    assert diagnostics["ankle_effort_mean_squared"] == pytest.approx(.75**2)
    assert diagnostics["ankle_hold_error_rmse_rad"] == pytest.approx(.3)
    assert metrics["joint_effort_mean_squared"] == metrics["root_assistance_mean_squared"] == 0.


@pytest.mark.parametrize("mode", ["nonfinite", "finite_failure"])
def test_partial_failure_retains_actual_clocks_actions_and_stale_or_current_engine_state(mode):
    _, scene, _, task = setup()
    env = feet.SharedExerciseFeetTaskEnv(scene, task)
    before, _ = env.reset(seed=3)
    engine = AnalyticalEngine.instances[0]
    engine.mode = mode
    observation, reward, terminated, truncated, _ = env.step(np.zeros(25))
    snapshot = env.snapshot()
    metrics, diag = snapshot["task_metrics"], snapshot["ankle_diagnostics"]
    assert terminated and not truncated and reward < 0. and metrics["failure"]
    assert diag["observed_substeps"] == metrics["observed_substeps"] == (4 if mode == "nonfinite" else 2)
    assert diag["valid_substeps"] == metrics["valid_substeps"] == (3 if mode == "nonfinite" else 2)
    assert diag["valid_observed_duration_s"] == metrics["valid_observed_duration_s"]
    assert snapshot["last_action"] == [0.] * 25
    assert metrics["completed_interval"] is False
    if mode == "nonfinite":
        assert snapshot["engine_state_status"] == "last_valid_control_boundary"
        assert snapshot["engine_state"]["time_s"] == 0. < metrics["simulated_duration_s"]
        assert diag["latest_substep"]["interval_end_s"] is None and diag["latest_substep"]["valid"] is False
        np.testing.assert_array_equal(observation, before)
    else:
        assert snapshot["engine_state_status"] == "current"
        assert snapshot["engine_state"]["time_s"] == metrics["simulated_duration_s"]
        assert diag["latest_substep"]["failure"] == "finite_dynamics_failure"
    with pytest.raises(RuntimeError, match="after completion"):
        env.step(np.zeros(25))


@pytest.mark.parametrize("failure", ["reset_failure", "invalid_observation"])
def test_reset_distribution_failure_and_unobserved_diagnostics_remain_explicit(failure):
    _, scene, _, task = setup("shared-world-v4")
    env = feet.SharedExerciseFeetTaskEnv(scene, task)
    engine = AnalyticalEngine.instances[0]
    _, info = env.reset(seed=31, options={"profile": HUMAN_JITTER_PROFILE, "jitter": True, "jitter_scale": .2})
    assert engine.reset_call == {"seed": 31, "options": {"jitter": True, "jitter_scale": .2,
                                                           "position_jitter": 0., "velocity_jitter": 0.}}
    assert info["reset_protocol"]["static_support_reset_state"] == "unchanged_scene_pose_and_velocity"
    assert all(row["first_contact_time_s"] is row["last_contact_time_s"] is row["peak_normal_force"] is None
               for row in env.ankle_diagnostics()["contact_aggregates"])
    engine.mode = failure
    with pytest.raises((ValueError, RuntimeError)):
        env.reset(seed=32)
    for method in (env.snapshot, env.authored_action, env.metrics):
        with pytest.raises(RuntimeError, match="Reset"):
            method()


def test_step_snapshot_and_cached_feet_renderer_never_repeat_preparation(monkeypatch):
    from weft import shared_exercise_feet_recording as recording

    _, scene, _, task = setup()
    env = feet.SharedExerciseFeetTaskEnv(scene, task)
    env.reset(seed=4)
    calls = {"scene": 0, "prepare": 0, "frames": []}
    frozen_recording = {"engine_scene": scene, "exercise_task": task}
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def recording_scene(actual, outer):
        calls["scene"] += 1
        assert actual == scene and outer == task
        return frozen_recording

    def prepare_renderer(value):
        calls["prepare"] += 1
        assert value is frozen_recording

        def render(snapshot):
            calls["frames"].append(snapshot)
            return frame

        return render

    def forbidden(*_args, **_kwargs):
        pytest.fail("Hot paths cannot validate scenes, prepare models or read implementation bytes")

    monkeypatch.setattr(feet, "prepare_shared_exercise_feet_task", forbidden)
    monkeypatch.setattr(feet, "adapter_implementation_identity", forbidden)
    monkeypatch.setattr(feet, "validate_shared_world", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(recording, "recording_scene", recording_scene)
    monkeypatch.setattr(recording, "prepare_renderer", prepare_renderer)
    before = env.snapshot()
    assert env.render() is frame and env.snapshot() == before
    for _ in range(2):
        env.step(np.zeros(25))
        state = env.snapshot()
        assert state["adapter_implementation_sha256"] == task["adapter_implementation_sha256"]
        assert env.render() is frame and env.snapshot() == state
    assert state["task_metrics"]["completed_interval"]
    assert calls["scene"] == calls["prepare"] == 1
    assert [value["task_metrics"]["observed_substeps"] for value in calls["frames"]] == [0, 8, 16]
    assert all(value["profile"] == feet.STATE_PROFILE and value["scene_sha256"] == task["scene_sha256"]
               and value["engine_state"]["scene_profile"] == "shared-world-v5" for value in calls["frames"])


def test_maximum_contact_diagnostics_are_bounded_not_an_episode_trace(capsys):
    _, scene, _, task = setup("shared-world-v4", substeps=64)
    env = feet.SharedExerciseFeetTaskEnv(scene, task)
    env.reset(seed=5)
    engine = AnalyticalEngine.instances[0]
    engine.contacts_at = lambda _: (foot_contact(),) * 256
    start, cpu = perf_counter(), process_time()
    for _ in range(2):
        env.step(np.zeros(25))
    snapshot = env.snapshot()
    elapsed, used_cpu = perf_counter() - start, process_time() - cpu
    diag = snapshot["ankle_diagnostics"]
    assert diag["observed_substeps"] == 128 and diag["excluded_foot_contact_records"] == 128 * 256
    assert len(diag["latest_substep"]["excluded_foot_contacts"]) == 256
    assert len(diag["contact_aggregates"]) == 2 and len(json.dumps(diag).encode()) < 128 << 10
    assert diag["contact_aggregates"][0]["contact_records"] == 32768
    with capsys.disabled():
        print(json.dumps({"pure_adapter_measurement": {"controls": 2, "substeps": 128,
              "contacts_per_substep": 256, "wall_seconds": elapsed, "cpu_seconds": used_cpu,
              "diagnostics_bytes": len(json.dumps(diag).encode()), "native_execution": False}}))
