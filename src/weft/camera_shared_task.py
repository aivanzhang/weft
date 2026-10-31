"""Camera-specific task binding around one real shared world and unchanged scoring arithmetic."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from weft import shared_exercise_task as kernel
from weft.io import canonical_hash
from weft.shared_exercise_reset import (
    HUMAN_JITTER_PROFILE, describe_reset_protocol, engine_reset_options, resolve_reset_options,
)
from weft.shared_world import CAMERA_PROFILE, SharedWorldEnv, prepare_shared_renderer, validate_shared_world


PROFILE = "camera-shared-task-v1"
STATE_PROFILE = "camera-shared-state-v1"
METRICS_PROFILE = "camera-shared-metrics-v1"
OBSERVATION_PROFILE = "camera-shared-observation-v1"
CONTEXT_PROFILE = "camera-shared-metric-context-v1"
RESET_PROFILE = "camera-shared-reset-v1"
HUMAN_OBSERVATION_VALUES = 155
MAX_TASK_BYTES = 64 << 10
IMPLEMENTATION_FILES = ("camera_shared_task.py", "shared_exercise_task.py")
SCOPE = MappingProxyType({
    "success_scope": "authored_simulated_camera_tracking_criteria",
    "physical_validation": "unverified",
    "source_contact_verified": False,
    "ground_support_verified": False,
    "equipment_scoring": "unsupported",
    "foot_controls": "unsupported",
    "learning": "unqualified",
    "body_scale": "supplied_camera_root_scale_unchanged_monocular_shape",
    "body_mass": "unchanged_authored_model",
    "clock_alignment": "original_source_grid_with_unchanged_body_period",
})


def implementation_identity():
    """Pin this adapter and unchanged arithmetic, not the complete physics runtime."""
    files = {}
    for name in IMPLEMENTATION_FILES:
        path = Path(__file__).with_name(name)
        if not path.is_file() or path.stat().st_size > 1 << 20:
            raise ValueError("Camera task implementation must exist within its bounded preparation closure")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"files": files, "sha256": canonical_hash(files)}


def _task_bytes(task):
    parts, count = [], 0
    try:
        for part in json.JSONEncoder(sort_keys=True, allow_nan=False).iterencode(task):
            encoded = part.encode("utf-8")
            count += len(encoded)
            if count > MAX_TASK_BYTES:
                raise ValueError("Camera task exceeds its 64 KiB bound")
            parts.append(encoded)
    except (TypeError, RecursionError, UnicodeError) as error:
        raise ValueError("Camera task requires bounded finite JSON") from error
    return b"".join(parts)


def _scene(scene):
    if not isinstance(scene, dict) or scene.get("profile") != CAMERA_PROFILE:
        raise ValueError("Camera tasks require an actual shared-world-v6 scene")
    if scene.get("objects") != []:
        raise ValueError("Camera task v1 requires objects=[]; static equipment scoring is unqualified")
    if scene.get("learned_policy_available") is not False:
        raise ValueError("Camera tasks have no qualified learned policy")
    actual = validate_shared_world(scene)
    contract = actual["action_contract"]
    if (contract["human_motor_count"] != 22 or len(contract["names"]) != 22
            or contract["object_force_channels"] != 0 or contract["dynamic_object_ids"]
            or actual["target"] is not None or actual["task"]["object_goal"] is not None):
        raise ValueError("Camera task requires exactly 22 human motors and no object channels or goal")
    if np.linalg.norm(actual["gravity"]) <= 1e-9:
        raise ValueError("Camera task arithmetic requires nonzero declared gravity and body mass")
    return actual


def _envelope(scene, root_assistance):
    implementation = implementation_identity()
    scene_sha = canonical_hash(scene)
    coupling = scene["camera_coupling"]
    interval = {key: scene["task"][key] for key in ("start_offset_s", "horizon_steps")}
    control = {"root_assistance": "explicit_3_axis" if root_assistance else "disabled", "object_forces": "forbidden"}
    # These selector arguments lack legacy task identity/gates and are never legacy-validated.
    arguments = {"profile": kernel.HUMAN_ONLY_TASK_PROFILE, "interval": deepcopy(interval),
                 "control": dict(control), "equipment_ids": [], "contact_requirements": [], "source_scope": dict(SCOPE)}
    context = {"schema_version": 1, "profile": CONTEXT_PROFILE, "scene_sha256": scene_sha,
               "kernel": "weft.shared_exercise_task._ExerciseMetrics", "kernel_branch": kernel.HUMAN_ONLY_TASK_PROFILE,
               "valid_legacy_task": False, "projection": "identity_22_motor_samples_without_equipment",
               "control_dt_s": scene["dt"], "source_body_dt_s": scene["human_spec"]["dt"],
               "substeps": scene["substeps"], "kernel_arguments": arguments,
               "kernel_arguments_sha256": canonical_hash(arguments)}
    return {
        "schema_version": 1, "profile": PROFILE, "scene_sha256": scene_sha,
        "camera_coupling_sha256": canonical_hash(coupling), "source_sha256": coupling["source_sha256"],
        "body_sha256": coupling["body_sha256"], "source_clock_sha256": coupling["source_clock_sha256"],
        "metric_context": context, "metric_context_sha256": canonical_hash(context),
        "kernel_evaluator_sha256": implementation["files"]["shared_exercise_task.py"],
        "adapter_implementation_sha256": implementation["files"]["camera_shared_task.py"],
        "evaluator_sha256": implementation["sha256"],
        "gate_profile": kernel.GATE_PROFILE, "gates_sha256": kernel.GATES_SHA256,
        "interval": interval, "control": control, "scope": dict(SCOPE),
    }


def build_camera_shared_task(scene, *, root_assistance=False):
    """Select explicit root authority without changing the source scene or scoring gates."""
    if type(root_assistance) is not bool:
        raise ValueError("Camera root assistance must be an explicit boolean")
    actual = _scene(scene)
    task = _envelope(actual, root_assistance)
    _task_bytes(task)
    return task


def prepare_camera_shared_task(task, scene):
    """Prepare detached actual scene/context once, never on a control or snapshot path."""
    encoded = _task_bytes(task)
    if (not isinstance(task, dict) or type(task.get("schema_version")) is not int or task["schema_version"] != 1
            or task.get("profile") != PROFILE or not isinstance(task.get("control"), dict)
            or task["control"].get("root_assistance") not in ("disabled", "explicit_3_axis")):
        raise ValueError("Camera task profile or explicit controls differ from contract")
    actual = _scene(scene)
    expected = _envelope(actual, task["control"]["root_assistance"] == "explicit_3_axis")
    if encoded != _task_bytes(expected):
        raise ValueError("Camera task identities, context, interval, controls or original gates differ")
    return actual, expected


def validate_camera_shared_task(task, scene):
    return prepare_camera_shared_task(task, scene)[1]


def _reset_options(options):
    resolved = resolve_reset_options(options)
    resolved["profile"] = HUMAN_JITTER_PROFILE
    return resolved


def _reset_protocol(options):
    result = describe_reset_protocol(options)
    result.update(profile=RESET_PROFILE, distribution_profile=HUMAN_JITTER_PROFILE,
                  equipment_stability_reference="not_applicable", equipment_reset_state="no_equipment")
    return result


class CameraSharedTaskEnv(kernel.SharedExerciseTaskEnv):
    """Use the existing task lifecycle and scorer with one unchanged v6 engine."""

    def __init__(self, scene, task, render_mode=None):
        self._scene, self._task = prepare_camera_shared_task(task, scene)
        self._kernel_arguments = self._task["metric_context"]["kernel_arguments"]
        self._static_support = False
        self._equipment_ids = ()
        self._requirement_key = "contact_requirements"
        self.scene_sha256, self.task_sha256 = self._task["scene_sha256"], canonical_hash(self._task)
        self.evaluator_sha256 = self._task["evaluator_sha256"]
        self.render_mode = render_mode
        self.dt, self.substeps = self._scene["dt"], self._scene["substeps"]
        self.horizon_steps = self._task["interval"]["horizon_steps"]
        self._source_start = self._scene["camera_coupling"]["clock"]["source_time_offset_s"]
        self.action_count = 22 if self._task["control"]["root_assistance"] == "explicit_3_axis" else 19
        self.action_names = tuple(self._scene["action_contract"]["names"][:self.action_count])
        self.action_space = spaces.Box(-1., 1., shape=(self.action_count,), dtype=np.float32)
        self.observation_size = HUMAN_OBSERVATION_VALUES + 1
        self.observation_space = spaces.Box(-1e6, 1e6, shape=(self.observation_size,), dtype=np.float32)
        self._accumulator = None
        self._done = True
        self._last_action = np.zeros(self.action_count)
        self._last_observation = self._last_engine_state = None
        self._state_status = "current"
        self._renderer = None
        self._reset_options = None
        self.terminated = self.truncated = False
        self.metadata = {**SharedWorldEnv.metadata, "render_fps": 1.0 / self.dt}
        self._engine = SharedWorldEnv(self._scene, render_mode, substep_observer=self._observe_substep,
                                      substep_observer_motor_count=22)

    def observation_contract(self):
        return {"profile": OBSERVATION_PROFILE, "size": self.observation_size,
                "fields": [{"name": "human_reference_conditioned_state", "count": HUMAN_OBSERVATION_VALUES},
                           {"name": "elapsed_fraction", "count": 1}],
                "source_clock_sha256": self._task["source_clock_sha256"]}

    def _observation(self, engine_observation):
        values = np.asarray(engine_observation)
        if values.ndim != 1 or len(values) < HUMAN_OBSERVATION_VALUES:
            raise RuntimeError("Camera observation must retain the full 155-value human block")
        observation = np.r_[values[:HUMAN_OBSERVATION_VALUES], self._accumulator.elapsed / self._accumulator.duration].astype(np.float32)
        if not self.observation_space.contains(observation):
            raise RuntimeError("Camera task observation exceeds its finite declared bounds")
        return observation

    def reset(self, *, seed=None, options=None):
        resolved = _reset_options(options)
        self._done, self._accumulator = True, None
        gym.Env.reset(self, seed=seed)
        try:
            observation, reset_info = self._engine.reset(seed=seed, options=engine_reset_options(resolved))
            state = self._engine.snapshot()
            if state["objects"]:
                raise ValueError("Camera task reset cannot introduce equipment")
            self._accumulator = kernel._ExerciseMetrics(self._scene, self._kernel_arguments, (),
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

    def step(self, action):
        if self._accumulator is not None and not self._done:
            values = np.asarray(action)
            # Signed integer minima overflow abs(); reject before the inherited action is stored.
            if values.dtype.kind in "iu" and (np.any(values < -1) or np.any(values > 1)):
                raise ValueError(f"Exercise action requires exactly {self.action_count} numeric values in [-1,1]")
        return super().step(action)

    def metrics(self):
        if self._accumulator is None:
            raise RuntimeError("Reset before reading camera task metrics")
        result = self._accumulator.metrics()
        result.update(profile=METRICS_PROFILE, kernel_branch=kernel.HUMAN_ONLY_TASK_PROFILE,
                      scene_sha256=self.scene_sha256, task_sha256=self.task_sha256,
                      evaluator_sha256=self.evaluator_sha256, reset_protocol=_reset_protocol(self._reset_options),
                      observed_source_time_s=self._source_start + result["simulated_duration_s"])
        for key in ("metric_context_sha256", "kernel_evaluator_sha256", "adapter_implementation_sha256",
                    "camera_coupling_sha256", "source_clock_sha256", "body_sha256"):
            result[key] = self._task[key]
        return result

    def snapshot(self):
        if self._accumulator is None:
            raise RuntimeError("Reset before reading camera task state")
        return {"schema_version": 1, "profile": STATE_PROFILE, "scene_sha256": self.scene_sha256,
                "task_sha256": self.task_sha256,
                **{key: self._task[key] for key in ("camera_coupling_sha256", "source_sha256", "body_sha256",
                    "source_clock_sha256", "metric_context_sha256", "kernel_evaluator_sha256",
                    "adapter_implementation_sha256", "evaluator_sha256")},
                "engine_state": deepcopy(self._last_engine_state), "task_metrics": self.metrics(),
                "last_action": self._last_action.tolist(), "engine_state_status": self._state_status,
                "resolved_reset_options": dict(self._reset_options),
                "terminated": self.terminated, "truncated": self.truncated}

    def cancel(self):
        """Finalize only an active observed boundary; completed results are immutable."""
        if self._accumulator is None:
            raise RuntimeError("Reset before cancelling a camera task")
        if not self._done:
            if self._state_status != "current":
                raise RuntimeError("Camera cancellation requires the current observed boundary")
            self._accumulator.fail("cancelled")
            self._accumulator.penalize_failure()
            self._done = self.terminated = True
            self.truncated = False
        return self.snapshot()

    def render(self):
        if self._accumulator is None:
            raise RuntimeError("Reset before rendering a camera task")
        if self._renderer is None:
            self._renderer = prepare_shared_renderer(self._scene)
        return self._renderer(deepcopy(self._last_engine_state))
