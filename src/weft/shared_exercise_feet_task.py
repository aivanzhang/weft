"""A versioned feet adapter around one shared-world engine and unchanged exercise scorers."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from weft import shared_exercise_task as base_tasks
from weft.io import canonical_hash
from weft.shared_exercise_feet_projection import (
    FOOT_SEGMENTS, HUMAN_OBSERVATION_VALUES, PROFILE as PROJECTION_PROFILE,
    PROJECTION_CONTRACT_SHA256, derive_base_scoring_scene, project_substep,
    scatter_task_action, task_motor_indices,
)
from weft.shared_exercise_reset import engine_reset_options, resolve_reset_options
from weft.shared_world import FEET_PROFILE, SharedWorldEnv, validate_shared_world
from weft.shared_world_feet import TARGET_MODE
from weft.shared_world_observation import EquipmentState


PROFILE = "shared-exercise-feet-task-v1"
STATE_PROFILE = "shared-exercise-feet-state-v1"
OBSERVATION_PROFILE = "shared-exercise-feet-observation-v1"
DIAGNOSTICS_PROFILE = "shared-exercise-feet-diagnostics-v1"
MAX_TASK_BYTES = 128 << 10
ADAPTER_FILES = ("shared_exercise_feet_task.py", "shared_exercise_feet_projection.py", "shared_exercise_feet_state.py")
SCOPE = MappingProxyType({
    "success_scope": "unchanged_base_task_simulated_criteria",
    "ankle_metrics": "unscored_diagnostics",
    "foot_support_credit": False,
    "ground_load_requirements": "unsupported",
    "learning": "unsupported",
    "physical_validation": "unverified",
})


def adapter_implementation_identity():
    """Identify only this adapter, not the complete physics/runtime dependency closure."""
    files = {}
    for name in ADAPTER_FILES:
        path = Path(__file__).with_name(name)
        if not path.is_file() or path.stat().st_size > 1 << 20:
            raise ValueError("Feet adapter implementation files must exist within the bounded preparation closure")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": files, "sha256": canonical_hash(files)}


def _scenes(scene):
    # Preparation may compile mass-check models; never call this on a step/snapshot path.
    if not isinstance(scene, dict) or scene.get("profile") != FEET_PROFILE:
        raise ValueError("Feet exercise tasks require an explicit shared-world-v5 scene")
    actual = validate_shared_world(scene)
    return actual, derive_base_scoring_scene(actual)


def _envelope(scene, base_scene, base_task):
    return {
        "schema_version": 1, "profile": PROFILE,
        "scene_sha256": canonical_hash(scene), "base_scene_sha256": canonical_hash(base_scene),
        "base_task_sha256": canonical_hash(base_task), "base_task": deepcopy(base_task),
        "projection_profile": PROJECTION_PROFILE,
        "projection_contract_sha256": PROJECTION_CONTRACT_SHA256,
        "adapter_implementation_sha256": adapter_implementation_identity()["sha256"],
        "foot_reference_sha256": scene["foot_extension"]["reference_sha256"],
        "interval": deepcopy(base_task["interval"]),
        "control": {**base_task["control"], "ankle_target_mode": TARGET_MODE},
        "scope": dict(SCOPE),
    }


def build_shared_exercise_feet_task(scene, base_task):
    """Bind an explicit original task; its root-assistance choice remains authoritative."""
    actual, base_scene = _scenes(scene)
    original = base_tasks.validate_shared_exercise_task(base_task, base_scene)
    return _envelope(actual, base_scene, original)


def prepare_shared_exercise_feet_task(task, scene):
    """Prepare detached engine/scoring scenes and task once, outside observation paths."""
    try:
        encoded = json.dumps(task, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Feet task must be finite bounded JSON") from error
    if (len(encoded) > MAX_TASK_BYTES or not isinstance(task, dict)
            or type(task.get("schema_version")) is not int or task.get("schema_version") != 1
            or task.get("profile") != PROFILE or not isinstance(task.get("base_task"), dict)):
        raise ValueError("Feet task profile, original task or size differs from contract")
    actual, base_scene = _scenes(scene)
    original = base_tasks.validate_shared_exercise_task(task["base_task"], base_scene)
    expected = _envelope(actual, base_scene, original)
    if canonical_hash(task) != canonical_hash(expected):
        raise ValueError("Feet task fields, source identities, original task or projection contract differ")
    return actual, base_scene, expected


def validate_shared_exercise_feet_task(task, scene):
    """Validate the actual engine and derived scoring identities without relabeling either."""
    return prepare_shared_exercise_feet_task(task, scene)[2]


class SharedExerciseFeetTaskEnv(base_tasks.SharedExerciseTaskEnv):
    """Expose ankles in the existing task family; no second simulator or new reward."""

    def __init__(self, scene, task, render_mode=None):
        self._scene, self._base_scene, self._task = prepare_shared_exercise_feet_task(task, scene)
        self._base_task = self._task["base_task"]
        self._static_support = self._base_task["profile"] == base_tasks.SUPPORT_TASK_PROFILE
        self._equipment_ids = () if self._static_support else tuple(self._base_task["equipment_ids"])
        self._requirement_key = "support_requirements" if self._static_support else "contact_requirements"
        self.scene_sha256, self.task_sha256 = self._task["scene_sha256"], canonical_hash(self._task)
        self.base_scene_sha256 = self._task["base_scene_sha256"]
        self.base_task_sha256 = self._task["base_task_sha256"]
        self.evaluator_sha256 = (base_tasks._support_evaluator_sha256() if self._static_support else
                                hashlib.sha256(Path(base_tasks.__file__).read_bytes()).hexdigest())
        self._root_assistance = self._base_task["control"]["root_assistance"] == "explicit_3_axis"
        self._motor_indices = task_motor_indices(root_assistance=self._root_assistance)
        self.render_mode = render_mode
        self.dt, self.substeps = self._scene["dt"], self._scene["substeps"]
        self.horizon_steps = self._base_task["interval"]["horizon_steps"]
        self.action_count = len(self._motor_indices)
        self.action_names = tuple(self._scene["action_contract"]["names"][i] for i in self._motor_indices)
        self.action_space = spaces.Box(-1., 1., shape=(self.action_count,), dtype=np.float32)
        self.observation_size = (HUMAN_OBSERVATION_VALUES + 1 + 17 * len(self._equipment_ids)
                                 + 6 * len(self._base_task[self._requirement_key]))
        self.observation_space = spaces.Box(-1e6, 1e6, shape=(self.observation_size,), dtype=np.float32)
        self._accumulator = None
        self._done = True
        self._last_action = np.zeros(self.action_count)
        self._last_observation = self._last_engine_state = None
        self._state_status = "current"
        self._renderer = None
        self._reset_options = None
        self.terminated = self.truncated = False
        self._clear_diagnostics()
        self._engine = SharedWorldEnv(self._scene, render_mode, substep_observer=self._observe_substep,
                                      substep_observer_motor_count=28)

    @property
    def base_scene(self):
        return deepcopy(self._base_scene)

    @property
    def base_task(self):
        return deepcopy(self._base_task)

    def action_contract(self):
        contract = self._scene["action_contract"]
        return {"names": list(self.action_names), "normalized_bounds": [-1., 1.],
                "joint_channels": list(range(25)), "body_joint_channels": list(range(19)),
                "ankle_channels": list(range(19, 25)),
                "root_assistance_channels": list(range(25, self.action_count)),
                "engine_motor_indices": list(self._motor_indices),
                "units": [contract["units"][i] for i in self._motor_indices],
                "physical_limits": [contract["physical_limits"][i] for i in self._motor_indices],
                "root_assistance": self._base_task["control"]["root_assistance"],
                "ankle_target_mode": TARGET_MODE, "object_force_channels": []}

    def observation_contract(self):
        fields = [{"name": "human_reference_conditioned_state", "count": HUMAN_OBSERVATION_VALUES}]
        if not self._static_support:
            fields.append({"name": "equipment", "ids": list(self._equipment_ids), "count_each": 17,
                           "fields": ["position_model_units:3", "quaternion:4", "velocity_model_units_s:3",
                                      "angular_velocity_rad_s:3", "displacement_model_units:3", "orientation_error_rad:1"]})
        fields.append({"name": self._requirement_key, "count_each": 6,
                       "fields": ["active", "qualifying", "load_fraction_clipped_10_unobserved_zero",
                                  "observed_coverage_fraction", "current_gap_s", "max_gap_s"]})
        fields.append({"name": "elapsed_fraction", "count": 1})
        return {"profile": OBSERVATION_PROFILE, "size": self.observation_size, "fields": fields,
                "base_task_profile": self._base_task["profile"], "ankle_target_mode": TARGET_MODE}

    def _clear_diagnostics(self):
        self._foot_observed = self._foot_valid = self._foot_contact_records = 0
        self._foot_duration = self._ankle_squared = self._ankle_effort = self._ankle_peak = 0.
        self._latest_foot_sample = None
        self._foot_pairs = {(segment, obj["id"]): {
            "human_segment": segment, "object_id": obj["id"], "contact_records": 0,
            "first_contact_time_s": None, "last_contact_time_s": None, "peak_normal_force": None,
        } for obj in self._scene["objects"] for segment in FOOT_SEGMENTS}

    def _observe_substep(self, sample):
        if self._accumulator is None or self._foot_observed >= self.horizon_steps * self.substeps:
            raise ValueError("Feet observations require a reset and the unchanged bounded source interval")
        projected, excluded = project_substep(sample)
        if any((contact.human_segment, contact.object_id) not in self._foot_pairs for contact in excluded):
            raise ValueError("Excluded contact does not belong to the pinned scene equipment")
        self._foot_observed += 1
        self._latest_foot_sample = (sample, excluded)
        for contact in excluded:
            key = (contact.human_segment, contact.object_id)
            state = self._foot_pairs[key]
            state["contact_records"] += 1
            self._foot_contact_records += 1
            if state["first_contact_time_s"] is None:
                state["first_contact_time_s"] = sample.contact_time_s
            state["last_contact_time_s"] = sample.contact_time_s
            previous = state["peak_normal_force"]
            state["peak_normal_force"] = contact.normal_force if previous is None else max(previous, contact.normal_force)
        before = self._accumulator.valid_samples
        try:
            self._accumulator.consume(projected)
        finally:
            # Preserve the scorer's accepted durations even if later equipment validation fails.
            if self._accumulator.valid_samples == before + 1:
                elapsed = sample.interval_end_s - sample.interval_start_s
                squared = sum((q - target) ** 2 for q, target in zip(sample.human_qpos[22:], sample.target_qpos[22:])) / 6
                self._foot_valid += 1
                self._foot_duration += elapsed
                self._ankle_squared += squared * elapsed
                self._ankle_effort += sum(value * value for value in sample.human_action[22:]) / 6 * elapsed
                self._ankle_peak = max(self._ankle_peak, math.sqrt(squared))

    def ankle_diagnostics(self):
        latest = None
        if self._latest_foot_sample is not None:
            sample, contacts = self._latest_foot_sample
            latest = {key: getattr(sample, key) for key in (
                "step_index", "substep_index", "interval_start_s", "interval_end_s", "contact_time_s", "valid", "failure")}
            latest["excluded_foot_contacts"] = [{
                "human_geom": item.human_geom, "human_segment": item.human_segment, "object_id": item.object_id,
                "position": list(item.position), "distance": item.distance, "normal_force": item.normal_force,
                "force_on_human_world": list(item.force_on_human_world),
                "normal_on_human_world": list(item.normal_on_human_world),
            } for item in contacts]
        duration = self._foot_duration
        return {"profile": DIAGNOSTICS_PROFILE, "source": "simulated_engine_substeps",
                "target_mode": TARGET_MODE, "scored": False,
                "foot_support_credit": False, "ground_load_requirements": "unsupported",
                "retention": "latest_observed_substep_plus_episode_aggregates",
                "observed_substeps": self._foot_observed, "valid_substeps": self._foot_valid,
                "valid_observed_duration_s": duration,
                "ankle_hold_error_rmse_rad": math.sqrt(self._ankle_squared / duration) if duration else None,
                "peak_ankle_hold_error_rmse_rad": self._ankle_peak if duration else None,
                "ankle_effort_mean_squared": self._ankle_effort / duration if duration else None,
                "excluded_foot_contact_records": self._foot_contact_records,
                "contact_aggregates": deepcopy(list(self._foot_pairs.values())), "latest_substep": latest}

    def _observation(self, engine_observation):
        acc = self._accumulator
        values = list(engine_observation[:HUMAN_OBSERVATION_VALUES])
        for name in self._equipment_ids:
            item, initial = acc.latest[name], acc.initial[name]
            values.extend(np.asarray(item.position) / acc.scale)
            values.extend(item.quaternion)
            values.extend(np.asarray(item.velocity) / acc.scale)
            values.extend(item.angular_velocity)
            values.extend((np.asarray(item.position) - initial.position) / acc.scale)
            values.append(base_tasks._quat_angle(item.quaternion, initial.quaternion))
        states = acc.requirements if self._static_support else acc.contacts
        for state in states:
            observed, load = state["observed_duration_s"], state["load_body_weight_fraction"]
            values.extend((state["active_now"], state["qualifying_now"], min(10., load) if load is not None else 0.,
                           state["qualifying_duration_s"] / observed if observed else 0.,
                           state["current_gap_s"], state["max_gap_s"]))
        values.append(acc.elapsed / acc.duration)
        observation = np.asarray(values, dtype=np.float32)
        if not self.observation_space.contains(observation):
            raise RuntimeError("Feet exercise observation exceeds its finite declared bounds")
        return observation

    def reset(self, *, seed=None, options=None):
        resolved = resolve_reset_options(options)
        self._done, self._accumulator = True, None
        self._clear_diagnostics()
        gym.Env.reset(self, seed=seed)
        try:
            observation, reset_info = self._engine.reset(seed=seed, options=engine_reset_options(resolved))
            state = self._engine.snapshot()
            initial = tuple(EquipmentState(obj["id"], tuple(obj["position"]), tuple(obj["quaternion"]), tuple(obj["velocity"]),
                                           tuple(obj["angular_velocity"]), tuple(obj.get("virtual_force", [0., 0., 0.])))
                            for obj in state["objects"])
            accumulator = base_tasks._support_module().SupportMetrics if self._static_support else base_tasks._ExerciseMetrics
            self._accumulator = accumulator(self._base_scene, self._base_task, initial,
                state["human"]["mobility"]["max_penetration"], self._engine.human_mass)
            self._last_observation = self._observation(observation)
            self._last_engine_state = state
        except Exception:
            self._accumulator = None
            raise
        self._done = False
        self.terminated = self.truncated = False
        self._state_status = "current"
        self._reset_options = resolved
        self._last_action = np.zeros(self.action_count)
        return self._last_observation.copy(), {**self.metrics(), "reset": deepcopy(reset_info),
                                              "action_contract": self.action_contract()}

    def authored_action(self):
        if self._accumulator is None or self._done:
            raise RuntimeError("Reset before requesting feet exercise actions")
        return self._engine.authored_human_action()[list(self._motor_indices)].copy()

    def feature_matrix(self):
        if self._accumulator is None or self._done:
            raise RuntimeError("Reset before reading feet exercise features")
        return self._engine.feature_matrix()[list(self._motor_indices)].copy()

    def step(self, action):
        if self._accumulator is None or self._done:
            raise RuntimeError("Reset before stepping the feet task and after completion")
        full_action = scatter_task_action(action, engine_action_count=len(self._engine.action_names),
                                          root_assistance=self._root_assistance)
        values = np.asarray(action, dtype=float).copy()
        before = self._accumulator.reward_integral
        self._last_action = values
        try:
            engine_observation, _, terminated, truncated, _ = self._engine.step(full_action)
            self._last_observation = self._observation(engine_observation)
            self._last_engine_state = self._engine.snapshot()
        except RuntimeError as error:
            self._accumulator.fail("engine_failure:" + str(error))
            self._accumulator.penalize_failure()
            self._done = True
            self.terminated, self.truncated = True, False
            self._state_status = "last_valid_control_boundary"
            reward = (self._accumulator.reward_integral - before) / self._scene["dt"]
            return self._last_observation.copy(), float(reward), True, False, {
                **self.metrics(), "observation_status": "last_valid_control_boundary"}
        metrics = self.metrics()
        if (terminated or truncated) and not metrics["completed_interval"] and self._accumulator.failure is None:
            self._accumulator.fail("engine_stopped_before_observed_interval_completed")
            metrics = self.metrics()
        self._done = bool(terminated or truncated or self._accumulator.failure)
        if self._accumulator.failure:
            self._accumulator.penalize_failure()
            metrics = self.metrics()
        self.terminated = bool(self._accumulator.failure or metrics["success"])
        self.truncated = bool(self._done and not self.terminated)
        reward = (self._accumulator.reward_integral - before) / self._scene["dt"]
        return self._last_observation.copy(), float(reward), self.terminated, self.truncated, {
            **metrics, "observation_status": "current", "legacy_coupled_score_used": False}

    def metrics(self):
        if self._accumulator is None:
            raise RuntimeError("Reset before reading feet exercise metrics")
        reset = (base_tasks._support_reset_protocol(self._reset_options) if self._static_support else
                 base_tasks._reset_protocol(self._reset_options))
        return {**self._accumulator.metrics(), "scene_sha256": self.base_scene_sha256,
                "task_sha256": self.base_task_sha256, "evaluator_sha256": self.evaluator_sha256,
                "reset_protocol": reset}

    def snapshot(self):
        if self._accumulator is None:
            raise RuntimeError("Reset before reading feet exercise state")
        return {"schema_version": 1, "profile": STATE_PROFILE, "scene_sha256": self.scene_sha256,
                "task_sha256": self.task_sha256, "base_scene_sha256": self.base_scene_sha256,
                "base_task_sha256": self.base_task_sha256, "projection_profile": PROJECTION_PROFILE,
                "projection_contract_sha256": PROJECTION_CONTRACT_SHA256,
                "adapter_implementation_sha256": self._task["adapter_implementation_sha256"],
                "foot_reference_sha256": self._task["foot_reference_sha256"],
                "engine_state": deepcopy(self._last_engine_state), "task_metrics": self.metrics(),
                "ankle_diagnostics": self.ankle_diagnostics(), "last_action": self._last_action.tolist(),
                "engine_state_status": self._state_status, "resolved_reset_options": dict(self._reset_options),
                "terminated": self.terminated, "truncated": self.truncated}

    def render(self):
        if self._renderer is None:
            from weft.shared_exercise_feet_recording import prepare_renderer, recording_scene

            self._renderer = prepare_renderer(recording_scene(self._scene, self._task))
        return self._renderer(self.snapshot())
