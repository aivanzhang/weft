"""Human-only exercise tasks; simulator success does not establish grasp or physical transfer."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from types import MappingProxyType

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from weft.io import canonical_hash
from weft.shared_exercise_reset import (
    HUMAN_JITTER_PROFILE, describe_reset_protocol as _reset_protocol, engine_reset_options,
    resolve_reset_options as _reset_options,
)
from weft.shared_world import (
    HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE, SharedWorldEnv, prepare_shared_snapshot_validator, validate_shared_world,
)
from weft.shared_world_observation import EquipmentState, SubstepSample


PROFILE = "shared-exercise-task-v1"
HUMAN_ONLY_TASK_PROFILE = "shared-exercise-task-v2"
SUPPORT_TASK_PROFILE = "shared-support-task-v1"
TASK_PROFILES = (PROFILE, HUMAN_ONLY_TASK_PROFILE, SUPPORT_TASK_PROFILE)
HUMAN_ONLY_REWARD_PROFILE = "human_tracking_effort_v1"
HUMAN_ONLY_GATE_APPLICABILITY = MappingProxyType({
    "human_tracking": "applied", "equipment_stability": "not_applicable",
    "contact_requirements": "not_applicable",
})
GATE_PROFILE = "shared-exercise-gates-v1"
STATE_PROFILE = "shared-exercise-state-v1"
MAX_TASK_BYTES = 64 * 1024
MAX_REQUIREMENTS = 24
GATES = MappingProxyType({
    "angular_rmse_rad": .35, "root_rmse_model_units": .15,
    "max_floor_penetration_model_units": .025,
    "peak_angular_rmse_rad": .70, "peak_root_rmse_model_units": .30,
    "equipment_displacement_model_units": .03, "equipment_orientation_rad": .15,
    "equipment_speed_model_units_s": .15, "equipment_angular_speed_rad_s": .75,
    "wrist_region_radius_model_units": .06, "contact_distance_model_units": .001,
    "contact_normal_body_weight_fraction": .001, "support_upward_body_weight_fraction": .05,
    "contact_duration_fraction": .95, "max_contact_gap_s": .10,
})
GATES_SHA256 = canonical_hash(dict(GATES))
GATES_PROVENANCE = "Frozen authored simulator criteria; equipment stability is reset-relative. Not measured biomechanics, recovered grasp or physical transfer."


def _support_module():
    # Legacy profiles retain their original pinned dependency closure.
    from weft import shared_support_task
    return shared_support_task


def _support_evaluator_sha256():
    support_task = _support_module()
    return support_task.evaluator_identity({
        "shared_exercise_task.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_support_task.py": hashlib.sha256(Path(support_task.__file__).read_bytes()).hexdigest(),
    })


def _support_reset_protocol(options):
    result = _reset_protocol(options)
    result.update(profile="shared-support-reset-v1", object_position_jitter_world=0.,
                  object_velocity_jitter_world_s=0., equipment_stability_reference="not_applicable",
                  static_support_reset_state="unchanged_scene_pose_and_velocity")
    return result


def _prefix_scope(scene, identifiers):
    refs = scene["human_spec"].get("reference_evidence", [])
    match = re.search(r":frame:(\d+)$", refs[0].get("evidence_id", "")) if refs else None
    body_frame = int(match[1]) if match else None
    rows = []
    for obj in scene["objects"]:
        if obj["id"] not in identifiers:
            continue
        prefix = obj.get("provenance", {}).get("source_prefix", {})
        first = prefix.get("equipment_observation_frame")
        stated_body = prefix.get("body_start_frame")
        if any(value is not None and (type(value) is not int or value < 0) for value in (first, stated_body)):
            raise ValueError("Equipment source-prefix frames must be nonnegative integers")
        if stated_body is not None and stated_body != body_frame:
            raise ValueError("Equipment source-prefix body frame differs from the preserved human reference")
        gap = max(0, first - body_frame) if first is not None and body_frame is not None else None
        analytic = scene["source_binding"]["mode"] == "analytic_fixture"
        rows.append({"object_id": obj["id"], "body_start_frame": body_frame,
                     "equipment_observation_frame": first, "unobserved_prefix_frames": gap,
                     "initial_state_scope": "analytic_fixture" if analytic else
                     "authored_static_extrapolation" if gap else "source_linked_prior" if gap == 0 else "unknown"})
    return {"success_scope": "authored_simulated_proxy_task", "physical_validation": "unverified",
            "clock_alignment": "preserved_scene_clocks_not_retimed", "equipment_initialization": rows,
            "grasp_supported": False, "source_contact_verified": False}


def _interval(scene):
    return {key: deepcopy(scene["task"][key]) for key in ("start_offset_s", "horizon_steps", "native_interval") if key in scene["task"]}


def build_shared_exercise_task(scene, *, root_assistance=False, equipment_ids=None, contact_requirements=(), support_requirements=None):
    scene = validate_shared_world(scene)
    if scene["profile"] == STATIC_SUPPORT_PROFILE:
        if equipment_ids is not None or contact_requirements != ():
            raise ValueError("Static-support tasks require explicit support_requirements, not equipment/contact criteria")
        return _support_module().build_shared_support_task(scene, support_requirements=support_requirements,
                                                     root_assistance=root_assistance)
    if support_requirements is not None:
        raise ValueError("Support requirements require an explicit static-support scene")
    if type(root_assistance) is not bool:
        raise ValueError("root_assistance must be an explicit boolean")
    if (equipment_ids is not None and (not isinstance(equipment_ids, (list, tuple)) or len(equipment_ids) > 12)
            or not isinstance(contact_requirements, (list, tuple)) or len(contact_requirements) > MAX_REQUIREMENTS):
        raise ValueError("Equipment and contact requirements must be bounded lists")
    identifiers = [obj["id"] for obj in scene["objects"] if obj["role"] != "static"]
    selected = identifiers if equipment_ids is None else list(equipment_ids)
    profile = HUMAN_ONLY_TASK_PROFILE if scene["profile"] == HUMAN_ONLY_PROFILE else PROFILE
    task = {"schema_version": 1, "profile": profile, "scene_sha256": canonical_hash(scene),
            "gate_profile": GATE_PROFILE, "gates_sha256": GATES_SHA256,
            "interval": _interval(scene),
            "control": {"root_assistance": "explicit_3_axis" if root_assistance else "disabled",
                        "object_forces": "forbidden"},
            "equipment_ids": selected, "contact_requirements": deepcopy(list(contact_requirements)),
            "source_scope": _prefix_scope(scene, selected)}
    return validate_shared_exercise_task(task, scene)


def validate_shared_exercise_task(task, scene):
    scene = validate_shared_world(scene)
    if (scene["task"].get("native_interval", {}).get("profile") == "native-dynamic-terminal-control-interval-v1"
            or any(obj["provenance"].get("profile") == "moving-equipment-free-body-v1" for obj in scene["objects"])):
        raise ValueError("Dynamic native intervals require a separately qualified task; stationary exercise scoring is unsupported")
    if scene["profile"] == STATIC_SUPPORT_PROFILE:
        return _support_module().validate_shared_support_task(task, scene)
    try:
        raw = json.dumps(task, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Exercise task must be finite bounded JSON") from error
    if len(raw.encode()) > MAX_TASK_BYTES or not isinstance(task, dict) or set(task) != {
        "schema_version", "profile", "scene_sha256", "gate_profile", "gates_sha256", "interval", "control",
        "equipment_ids", "contact_requirements", "source_scope",
    }:
        raise ValueError("Exercise task fields or size differ from contract")
    human_only = scene["profile"] == HUMAN_ONLY_PROFILE
    profile = HUMAN_ONLY_TASK_PROFILE if human_only else PROFILE
    if (task["schema_version"] != 1 or type(task["schema_version"]) is not int or task["profile"] != profile
            or task["scene_sha256"] != canonical_hash(scene) or task["gate_profile"] != GATE_PROFILE
            or task["gates_sha256"] != GATES_SHA256):
        raise ValueError("Exercise scene identity or frozen gate profile differs")
    if scene["controlled_id"] != "human" or any(obj["role"] == "active" for obj in scene["objects"]):
        raise ValueError("Exercise tasks require human control with passive dynamic equipment")
    if np.linalg.norm(scene["gravity"]) <= 1e-9:
        raise ValueError("Exercise force criteria require nonzero declared gravity")
    expected_interval = _interval(scene)
    interval = task["interval"]
    if (not isinstance(interval, dict) or set(interval) != set(expected_interval)
            or type(interval["horizon_steps"]) is not int or type(interval["start_offset_s"]) not in (int, float)
            or interval != expected_interval):
        raise ValueError("Exercise interval must preserve the complete declared scene interval")
    control = task["control"]
    if (not isinstance(control, dict) or set(control) != {"root_assistance", "object_forces"}
            or not isinstance(control["root_assistance"], str) or control["root_assistance"] not in {"disabled", "explicit_3_axis"}
            or control["object_forces"] != "forbidden"):
        raise ValueError("Exercise controls must exclude object forces and explicitly declare root assistance")
    identifiers = task["equipment_ids"]
    dynamic = {obj["id"] for obj in scene["objects"] if obj["role"] != "static"}
    minimum = 0 if human_only else 1
    if (not isinstance(identifiers, list) or not minimum <= len(identifiers) <= 12
            or any(not isinstance(name, str) for name in identifiers)
            or len(set(identifiers)) != len(identifiers) or set(identifiers) != dynamic):
        raise ValueError("Exercise stability must cover every unique dynamic equipment object")
    requirements = task["contact_requirements"]
    if not isinstance(requirements, list) or len(requirements) > MAX_REQUIREMENTS or human_only and requirements:
        raise ValueError("Exercise tasks permit at most 24 contact requirements")
    seen = {}
    for row in requirements:
        if not isinstance(row, dict) or set(row) != {"object_id", "wrist", "kind", "start_step", "end_step"}:
            raise ValueError("Contact requirement fields differ from contract; grasp is unsupported")
        if (any(not isinstance(row[key], str) for key in ("object_id", "wrist", "kind"))
                or row["object_id"] not in dynamic or row["wrist"] not in {"left", "right"}
                or row["kind"] not in {"wrist_region_contact", "upward_support"}):
            raise ValueError("Only designated wrist-region contact/upward support is supported; not grasp")
        start, end = row["start_step"], row["end_step"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= expected_interval["horizon_steps"]:
            raise ValueError("Contact windows must be bounded half-open control-step intervals")
        key = (row["wrist"], row["object_id"])
        if any(start < previous_end and previous_start < end for previous_start, previous_end in seen.get(key, [])):
            raise ValueError("Repeated wrist/object contact windows cannot overlap")
        seen.setdefault(key, []).append((start, end))
    if task["source_scope"] != _prefix_scope(scene, identifiers):
        raise ValueError("Exercise task must preserve its source-prefix uncertainty and simulation-only scope")
    return deepcopy(task)


def _quat_angle(first, second):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    return 2 * math.acos(float(np.clip(abs(np.dot(a, b)), 0, 1)))


class _ExerciseMetrics:
    """Accumulate actual substep durations without retaining a trajectory or accepting terminal-only recovery."""

    def __init__(self, scene, task, initial_equipment, initial_penetration, human_mass):
        self.scene, self.task = scene, task
        self.scale = scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"]
        self.dt, self.substeps = scene["dt"], scene["substeps"]
        self.horizon = task["interval"]["horizon_steps"]
        self.duration = self.dt * self.horizon
        self.terminal_duration = self.dt
        if "native_interval" in scene["task"]:
            self.duration = scene["task"]["native_interval"]["duration_s"]
            self.terminal_duration = scene["task"]["native_interval"]["terminal_duration_s"]
        gravity = np.asarray(scene["gravity"], dtype=float)
        self.weight = float(human_mass * np.linalg.norm(gravity))
        if self.weight <= 1e-9:
            raise ValueError("Exercise force criteria require nonzero declared gravity and body mass")
        self.up = -gravity / np.linalg.norm(gravity)
        if task["profile"] == HUMAN_ONLY_TASK_PROFILE and initial_equipment:
            raise ValueError("Person-only reset cannot contain equipment states")
        self.initial = {item.object_id: item for item in initial_equipment if item.object_id in task["equipment_ids"]}
        if set(self.initial) != set(task["equipment_ids"]):
            raise ValueError("Reset must expose all designated equipment states")
        self.latest = dict(self.initial)
        self.equipment = {name: {"displacement_model_units": 0., "orientation_rad": 0.,
                                "speed_model_units_s": 0., "angular_speed_rad_s": 0.} for name in self.initial}
        self.contacts = [{"required_duration_s": (row["end_step"] - row["start_step"]) * self.dt,
                          "observed_duration_s": 0., "qualifying_duration_s": 0., "current_gap_s": 0.,
                          "max_gap_s": 0., "qualifying_now": False, "active_now": False,
                          "load_body_weight_fraction": 0.} for row in task["contact_requirements"]]
        self.samples = self.valid_samples = 0
        self.elapsed = self.valid_duration = self.angular_squared = self.root_squared = 0.
        self.angular_peak = self.root_peak = 0.
        self.max_penetration = float(initial_penetration)
        self.joint_effort = self.root_effort = self.reward_integral = 0.
        self.failure_penalty = 0.
        self.failure = None
        self.clock_status = "observed"
        self.contact_pairs = {(row["object_id"], row["wrist"]) for row in task["contact_requirements"]}
        self._update_equipment(self.initial.values())

    def fail(self, reason):
        self.failure = self.failure or str(reason)[:240]

    def penalize_failure(self):
        if not self.failure_penalty:
            self.failure_penalty = -1.
            self.reward_integral -= self.dt

    def _update_equipment(self, items):
        by_id = {item.object_id: item for item in items}
        if not set(self.initial) <= set(by_id):
            raise ValueError("Substep equipment coverage is incomplete")
        for name, initial in self.initial.items():
            current, maximum = by_id[name], self.equipment[name]
            if any(value != 0 for value in current.virtual_force):
                self.fail("forbidden_object_actuation")
            values = {"displacement_model_units": math.dist(current.position, initial.position) / self.scale,
                      "orientation_rad": _quat_angle(current.quaternion, initial.quaternion),
                      "speed_model_units_s": float(np.linalg.norm(current.velocity)) / self.scale,
                      "angular_speed_rad_s": float(np.linalg.norm(current.angular_velocity))}
            for key, value in values.items():
                maximum[key] = max(maximum[key], value)
            self.latest[name] = current

    def consume(self, sample):
        if not isinstance(sample, SubstepSample):
            raise ValueError("Exercise scoring requires the immutable engine substep contract")
        if self.task["profile"] == HUMAN_ONLY_TASK_PROFILE and (sample.equipment or sample.contacts):
            raise ValueError("Person-only substeps cannot contain equipment or human-object contacts")
        expected_step, expected_substep = divmod(self.samples, self.substeps)
        self.samples += 1
        aligned = not (sample.step_index != expected_step + 1 or sample.substep_index != expected_substep + 1
                or sample.step_index > self.horizon or not math.isclose(sample.interval_start_s, self.elapsed, rel_tol=0, abs_tol=1e-9)
                or not math.isclose(sample.contact_time_s, sample.interval_start_s, rel_tol=0, abs_tol=1e-9))
        if not aligned:
            self.fail("incomplete_or_misaligned_substep_observation")
        if sample.interval_end_s is not None:
            expected_duration = (self.terminal_duration if expected_step == self.horizon - 1 else self.dt) / self.substeps
            if (not math.isfinite(sample.interval_end_s) or
                    not math.isclose(sample.interval_end_s - sample.interval_start_s, expected_duration, rel_tol=0, abs_tol=1e-9)):
                self.fail("invalid_substep_duration")
                self.clock_status = "invalid_substep_clock"
                return
            else:
                self.elapsed = sample.interval_end_s
        else:
            self.clock_status = "unavailable_after_last_observed_time"
        if not aligned:
            return
        if not sample.valid:
            self.fail(sample.failure or "invalid_substep_observation")
            return
        if sample.interval_end_s is None:
            self.fail("missing_substep_clock")
            return
        if sample.failure is not None:
            self.fail(sample.failure)
        arrays = [np.asarray(getattr(sample, name), dtype=float) for name in
                  ("human_qpos", "human_qvel", "target_qpos", "human_action")]
        if any(value.shape != (22,) or not np.isfinite(value).all() for value in arrays):
            raise ValueError("Exercise substep human state must contain 22 finite values")
        q, _, target, action = arrays
        if np.any(abs(action) > 1) or (self.task["control"]["root_assistance"] == "disabled" and np.any(action[19:] != 0)):
            self.fail("forbidden_human_actuation")
        dt = sample.interval_end_s - sample.interval_start_s
        angular = float(np.mean((q[:19] - target[:19]) ** 2))
        root = float(np.mean(((q[19:] - target[19:]) / self.scale) ** 2))
        self.valid_samples += 1
        self.valid_duration += dt
        self.angular_squared += angular * dt
        self.root_squared += root * dt
        self.angular_peak = max(self.angular_peak, math.sqrt(angular))
        self.root_peak = max(self.root_peak, math.sqrt(root))
        self.max_penetration = max(self.max_penetration, sample.max_human_floor_penetration_model_units)
        self._update_equipment(sample.equipment)
        # A contact manifold is scanned once, even when several declared intervals use the same pair.
        loads = {pair: [0., 0.] for pair in self.contact_pairs}
        for contact in sample.contacts:
            wrist_name = "left" if contact.human_segment == "left_forearm" else "right" if contact.human_segment == "right_forearm" else None
            pair = (contact.object_id, wrist_name)
            if (pair not in loads or contact.distance > GATES["contact_distance_model_units"] * self.scale
                    or math.dist(contact.position, sample.wrist_positions[wrist_name]) > GATES["wrist_region_radius_model_units"] * self.scale):
                continue
            loads[pair][0] += max(0., contact.normal_force)
            loads[pair][1] += float(np.dot(contact.force_on_human_world, self.up))
        contact_scores = []
        for requirement, state in zip(self.task["contact_requirements"], self.contacts):
            active = requirement["start_step"] <= sample.step_index - 1 < requirement["end_step"]
            state.update(active_now=active, qualifying_now=False, load_body_weight_fraction=0.)
            if not active:
                continue
            normal, upward = loads[(requirement["object_id"], requirement["wrist"])]
            qualifying = normal / self.weight >= GATES["contact_normal_body_weight_fraction"]
            load, threshold = normal / self.weight, GATES["contact_normal_body_weight_fraction"]
            if requirement["kind"] == "upward_support":
                load, threshold = max(0., upward / self.weight), GATES["support_upward_body_weight_fraction"]
                qualifying = qualifying and load >= threshold
            state["load_body_weight_fraction"] = load
            state["observed_duration_s"] += dt
            state["qualifying_now"] = bool(qualifying)
            state["qualifying_duration_s"] += dt if qualifying else 0.
            state["current_gap_s"] = 0. if qualifying else state["current_gap_s"] + dt
            state["max_gap_s"] = max(state["max_gap_s"], state["current_gap_s"])
            contact_scores.append(min(1., load / threshold))
        joint_effort, root_effort = float(np.mean(action[:19] ** 2)), float(np.mean(action[19:] ** 2))
        self.joint_effort += joint_effort * dt
        self.root_effort += root_effort * dt
        tracking = math.exp(-angular / GATES["angular_rmse_rad"] ** 2 - root / GATES["root_rmse_model_units"] ** 2)
        if self.task["profile"] == HUMAN_ONLY_TASK_PROFILE:
            # Absent equipment earns no stability credit; only human tracking is scored.
            reward = tracking
        else:
            stability = np.mean([math.exp(-sum((value / GATES["equipment_" + key]) ** 2 for key, value in maximum.items()))
                                 for maximum in self.equipment.values()])
            reward = (.5 * tracking + .3 * stability + .2 * np.mean(contact_scores)) if contact_scores else (.625 * tracking + .375 * stability)
        self.reward_integral += float(reward - .01 * joint_effort - .02 * root_effort) * dt

    def metrics(self):
        angular = math.sqrt(self.angular_squared / self.valid_duration) if self.valid_duration else None
        root = math.sqrt(self.root_squared / self.valid_duration) if self.valid_duration else None
        completed = (self.samples == self.valid_samples == self.horizon * self.substeps
                     and math.isclose(self.elapsed, self.duration, abs_tol=1e-8) and self.failure is None)
        failures = []
        for name, value in (("angular_rmse_rad", angular), ("root_rmse_model_units", root),
                            ("peak_angular_rmse_rad", self.angular_peak), ("peak_root_rmse_model_units", self.root_peak),
                            ("max_floor_penetration_model_units", self.max_penetration)):
            if value is None or value > GATES[name] + 1e-12:
                failures.append(name)
        equipment = []
        for name, maximum in self.equipment.items():
            passed = all(value <= GATES["equipment_" + key] + 1e-12 for key, value in maximum.items())
            equipment.append({"object_id": name, **maximum, "within_limits": passed})
            if not passed:
                failures.append("equipment_stability:" + name)
        contacts = []
        for index, (requirement, state) in enumerate(zip(self.task["contact_requirements"], self.contacts)):
            fraction = state["qualifying_duration_s"] / state["required_duration_s"]
            passed = (math.isclose(state["observed_duration_s"], state["required_duration_s"], abs_tol=1e-8)
                      and fraction >= GATES["contact_duration_fraction"] - 1e-12
                      and state["max_gap_s"] <= GATES["max_contact_gap_s"] + 1e-12)
            contacts.append({**requirement, **state, "qualifying_fraction": fraction, "satisfied": passed})
            if not passed:
                failures.append("contact_requirement:" + str(index))
        if self.failure:
            failures.insert(0, self.failure)
        return {"profile": self.task["profile"], "completed_interval": completed, "success": completed and not failures,
                "failure": self.failure, "failed_gates": failures, "observed_substeps": self.samples,
                "valid_substeps": self.valid_samples, "simulated_duration_s": self.elapsed,
                "valid_observed_duration_s": self.valid_duration, "declared_duration_s": self.duration,
                "clock_status": self.clock_status,
                "angular_rmse_rad": angular, "root_rmse_model_units": root,
                "peak_angular_rmse_rad": self.angular_peak, "peak_root_rmse_model_units": self.root_peak,
                "max_floor_penetration_model_units": self.max_penetration,
                "equipment": equipment, "contact_requirements": contacts,
                "joint_effort_mean_squared": self.joint_effort / self.valid_duration if self.valid_duration else 0.,
                "root_assistance_mean_squared": self.root_effort / self.valid_duration if self.valid_duration else 0.,
                "reward_integral": self.reward_integral, "mean_reward": self.reward_integral / self.duration,
                "failure_reward_penalty": self.failure_penalty,
                "gates_sha256": GATES_SHA256, "source_scope": deepcopy(self.task["source_scope"]),
                "gate_profile": GATE_PROFILE, "gates_provenance": GATES_PROVENANCE,
                "root_assistance": self.task["control"]["root_assistance"], "virtual_object_forces": "forbidden",
                **({"gate_applicability": dict(HUMAN_ONLY_GATE_APPLICABILITY),
                    "reward_profile": HUMAN_ONLY_REWARD_PROFILE}
                   if self.task["profile"] == HUMAN_ONLY_TASK_PROFILE else {})}


class SharedExerciseTaskEnv(gym.Env):
    """A distinct Gym task around unchanged shared-world dynamics and legacy scores."""

    metadata = SharedWorldEnv.metadata

    def __init__(self, scene, task, render_mode=None):
        self._scene = validate_shared_world(scene)
        self._task = validate_shared_exercise_task(task, self._scene)
        self._static_support = self._task["profile"] == SUPPORT_TASK_PROFILE
        self._equipment_ids = () if self._static_support else tuple(self._task["equipment_ids"])
        self._requirement_key = "support_requirements" if self._static_support else "contact_requirements"
        self.scene_sha256, self.task_sha256 = canonical_hash(self._scene), canonical_hash(self._task)
        self.evaluator_sha256 = (_support_evaluator_sha256() if self._static_support else
                                 hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self._engine = SharedWorldEnv(self._scene, render_mode, substep_observer=self._observe_substep)
        self.render_mode = render_mode
        self.dt, self.substeps = self._scene["dt"], self._scene["substeps"]
        self.horizon_steps = self._task["interval"]["horizon_steps"]
        self.action_count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
        self.action_names = tuple(self._engine.human_action_names[:self.action_count])
        self.action_space = spaces.Box(-1., 1., shape=(self.action_count,), dtype=np.float32)
        self.observation_size = 156 + 17 * len(self._equipment_ids) + 6 * len(task[self._requirement_key])
        self.observation_space = spaces.Box(-1e6, 1e6, shape=(self.observation_size,), dtype=np.float32)
        self._accumulator = None
        self._done = True
        self._last_action = np.zeros(self.action_count)
        self._last_observation = self._last_engine_state = None
        self._state_status = "current"
        self._renderer = None
        self._reset_options = None
        self.terminated = self.truncated = False

    @property
    def scene(self):
        return deepcopy(self._scene)

    @property
    def task(self):
        return deepcopy(self._task)

    def action_contract(self):
        return {"names": list(self.action_names), "normalized_bounds": [-1., 1.],
                "joint_channels": list(range(19)), "root_assistance_channels": list(range(19, self.action_count)),
                "units": list(self._scene["action_contract"]["units"][:self.action_count]),
                "physical_limits": list(self._scene["action_contract"]["physical_limits"][:self.action_count]),
                "root_assistance": self._task["control"]["root_assistance"], "object_force_channels": []}

    def observation_contract(self):
        if self._static_support:
            return {"profile": "shared-support-observation-v1", "size": self.observation_size,
                    "fields": [{"name": "human_reference_conditioned_state", "count": 155},
                               {"name": "support_requirements", "count_each": 6,
                                "fields": ["active", "qualifying", "load_fraction_clipped_10_unobserved_zero",
                                           "observed_coverage_fraction", "current_gap_s", "max_gap_s"]},
                               {"name": "elapsed_fraction", "count": 1}]}
        return {"profile": "shared-exercise-observation-v1", "size": self.observation_size,
                "fields": [{"name": "human_reference_conditioned_state", "count": 155},
                           {"name": "equipment", "ids": list(self._task["equipment_ids"]), "count_each": 17,
                            "fields": ["position_model_units:3", "quaternion:4", "velocity_model_units_s:3",
                                       "angular_velocity_rad_s:3", "displacement_model_units:3", "orientation_error_rad:1"]},
                           {"name": "contact_requirements", "count_each": 6,
                            "fields": ["active", "qualifying", "load_fraction_clipped_10", "observed_coverage_fraction",
                                       "current_gap_s", "max_gap_s"]}, {"name": "elapsed_fraction", "count": 1}]}

    def _observe_substep(self, sample):
        self._accumulator.consume(sample)

    def _observation(self, engine_observation):
        acc = self._accumulator
        values = list(engine_observation[:155])
        for name in self._equipment_ids:
            item, initial = acc.latest[name], acc.initial[name]
            values.extend(np.asarray(item.position) / acc.scale)
            values.extend(item.quaternion)
            values.extend(np.asarray(item.velocity) / acc.scale)
            values.extend(item.angular_velocity)
            values.extend((np.asarray(item.position) - initial.position) / acc.scale)
            values.append(_quat_angle(item.quaternion, initial.quaternion))
        states = acc.requirements if self._static_support else acc.contacts
        for state in states:
            observed = state["observed_duration_s"]
            load = state["load_body_weight_fraction"]
            values.extend((state["active_now"], state["qualifying_now"], min(10., load) if load is not None else 0.,
                           state["qualifying_duration_s"] / observed if observed else 0.,
                           state["current_gap_s"], state["max_gap_s"]))
        values.append(acc.elapsed / acc.duration)
        observation = np.asarray(values, dtype=np.float32)
        if not self.observation_space.contains(observation):
            raise RuntimeError("Exercise observation exceeds its finite declared bounds")
        return observation

    def reset(self, *, seed=None, options=None):
        resolved = _reset_options(options)
        self._done, self._accumulator = True, None
        super().reset(seed=seed)
        observation, reset_info = self._engine.reset(seed=seed, options=engine_reset_options(resolved))
        state = self._engine.snapshot()
        initial = tuple(EquipmentState(obj["id"], tuple(obj["position"]), tuple(obj["quaternion"]), tuple(obj["velocity"]),
                                       tuple(obj["angular_velocity"]), tuple(obj.get("virtual_force", [0., 0., 0.])))
                        for obj in state["objects"])
        accumulator = _support_module().SupportMetrics if self._static_support else _ExerciseMetrics
        self._accumulator = accumulator(self._scene, self._task, initial,
            state["human"]["mobility"]["max_penetration"], self._engine.human_mass)
        self._done = False
        self.terminated = self.truncated = False
        self._state_status = "current"
        self._reset_options = resolved
        self._last_action = np.zeros(self.action_count)
        self._last_observation = self._observation(observation)
        self._last_engine_state = state
        return self._last_observation.copy(), {**self.metrics(), "reset": deepcopy(reset_info),
                                              "action_contract": self.action_contract()}

    def authored_action(self):
        if self._accumulator is None or self._done:
            raise RuntimeError("Reset before requesting exercise actions")
        return self._engine.authored_human_action()[:self.action_count].copy()

    def feature_matrix(self):
        if self._accumulator is None or self._done:
            raise RuntimeError("Reset before reading exercise policy features")
        return self._engine.feature_matrix()[:self.action_count].copy()

    def step(self, action):
        if self._accumulator is None or self._done:
            raise RuntimeError("Reset before stepping the exercise task and after completion")
        values = np.asarray(action)
        if (values.shape != (self.action_count,) or values.dtype.kind not in "fiu" or not np.isfinite(values).all()
                or np.any(abs(values) > 1) or isinstance(action, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in action)):
            raise ValueError(f"Exercise action requires exactly {self.action_count} numeric values in [-1,1]")
        values = values.astype(float, copy=True)
        full_action = np.zeros(len(self._engine.action_names))
        full_action[:self.action_count] = values
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
            raise RuntimeError("Reset before reading exercise metrics")
        reset = _support_reset_protocol(self._reset_options) if self._static_support else _reset_protocol(self._reset_options)
        return {**self._accumulator.metrics(), "scene_sha256": self.scene_sha256, "task_sha256": self.task_sha256,
                "evaluator_sha256": self.evaluator_sha256, "reset_protocol": reset}

    def snapshot(self):
        if self._accumulator is None:
            raise RuntimeError("Reset before reading exercise state")
        return {"schema_version": 1, "profile": STATE_PROFILE, "scene_sha256": self.scene_sha256,
                "task_sha256": self.task_sha256, "engine_state": deepcopy(self._last_engine_state),
                "task_metrics": self.metrics(), "last_action": self._last_action.tolist(),
                "engine_state_status": self._state_status, "resolved_reset_options": dict(self._reset_options),
                "terminated": self.terminated, "truncated": self.truncated}

    def observation(self):
        if self._accumulator is None:
            raise RuntimeError("Reset before reading exercise observations")
        return self._last_observation.copy()

    def render(self):
        if self._renderer is None:
            from weft.shared_exercise_recording import prepare_renderer, recording_scene

            self._renderer = prepare_renderer(recording_scene(self._scene, self._task))
        return self._renderer(self.snapshot())

    def close(self):
        self._engine.close()


def validate_shared_exercise_snapshot(snapshot, scene, task):
    """Strict one-shot validation; use the prepared validator only for the same pinned scene/task."""
    return prepare_shared_exercise_snapshot_validator(scene, task)(snapshot)


def prepare_shared_exercise_snapshot_validator(scene, task):
    """Pin private immutable inputs once; every call still checks the complete dynamic snapshot."""
    scene = validate_shared_world(scene)
    task = validate_shared_exercise_task(task, scene)
    if task["profile"] == SUPPORT_TASK_PROFILE:
        return _prepare_support_snapshot_validator(scene, task)
    scene_sha256, task_sha256 = canonical_hash(scene), canonical_hash(task)
    validate_engine = prepare_shared_snapshot_validator(scene)
    gates = dict(GATES)
    initial_equipment = {obj["id"]: obj for obj in scene["objects"]}
    equipment_order = [obj["id"] for obj in scene["objects"] if obj["role"] != "static"]
    contact_windows = tuple((row, (row["end_step"] - row["start_step"]) * scene["dt"])
                            for row in task["contact_requirements"])
    root_assistance_disabled = task["control"]["root_assistance"] == "disabled"
    expected = {"schema_version", "profile", "scene_sha256", "task_sha256", "engine_state", "task_metrics", "last_action",
                "engine_state_status", "resolved_reset_options", "terminated", "truncated"}
    action_count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    numeric = ("simulated_duration_s", "valid_observed_duration_s", "declared_duration_s", "peak_angular_rmse_rad",
               "peak_root_rmse_model_units", "max_floor_penetration_model_units", "joint_effort_mean_squared",
               "root_assistance_mean_squared", "reward_integral", "mean_reward", "failure_reward_penalty")
    fixed = {"profile": task["profile"], "gates_sha256": GATES_SHA256, "gate_profile": GATE_PROFILE,
             "gates_provenance": GATES_PROVENANCE, "source_scope": task["source_scope"],
             "root_assistance": task["control"]["root_assistance"], "virtual_object_forces": "forbidden",
             "scene_sha256": scene_sha256, "task_sha256": task_sha256}
    if task["profile"] == HUMAN_ONLY_TASK_PROFILE:
        fixed.update(gate_applicability=dict(HUMAN_ONLY_GATE_APPLICABILITY), reward_profile=HUMAN_ONLY_REWARD_PROFILE)
    metric_keys = {*fixed, *numeric, "angular_rmse_rad", "root_rmse_model_units", "completed_interval", "success", "failure",
            "failed_gates", "observed_substeps", "valid_substeps", "equipment", "contact_requirements", "evaluator_sha256", "clock_status", "reset_protocol"}
    duration = task["interval"]["horizon_steps"] * scene["dt"]
    tick = scene["dt"] / scene["substeps"]
    elapsed = lambda count: count * tick
    if "native_interval" in scene["task"]:
        from weft.native_interval import elapsed_substeps
        duration = scene["task"]["native_interval"]["duration_s"]
        elapsed = lambda count: elapsed_substeps(scene, count)
    maximum = task["interval"]["horizon_steps"] * scene["substeps"]

    def validate(snapshot):
        try:
            raw = json.dumps(snapshot, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("Exercise snapshot must be finite JSON") from error
        if (len(raw.encode()) > 4 * 1024 * 1024 or not isinstance(snapshot, dict) or set(snapshot) != expected
                or type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1
                or snapshot["profile"] != STATE_PROFILE or snapshot["scene_sha256"] != scene_sha256
                or snapshot["task_sha256"] != task_sha256):
            raise ValueError("Exercise snapshot contract or pinned identities differ")
        if snapshot["resolved_reset_options"] != _reset_options(snapshot["resolved_reset_options"]):
            raise ValueError("Exercise reset options must be resolved explicitly")
        state = snapshot["engine_state"]
        validate_engine(state)
        if state["controlled_id"] != "human":
            raise ValueError("Exercise snapshots must retain the human as their only controlled actor")
        if (state["steps"] == 0
                and snapshot["resolved_reset_options"].get("profile") == HUMAN_JITTER_PROFILE):
            for actual in state["objects"]:
                nominal = initial_equipment[actual["id"]]
                if (any(not np.allclose(actual[key], nominal[key], rtol=0, atol=1e-12)
                        for key in ("position", "velocity", "angular_velocity"))
                        or not math.isclose(abs(float(np.dot(actual["quaternion"], nominal["quaternion"]))),
                                            1., rel_tol=0, abs_tol=1e-12)):
                    raise ValueError("Human-only equipment reset must retain the exact scene pose and velocity")
        status = snapshot["engine_state_status"]
        if status not in {"current", "last_valid_control_boundary"}:
            raise ValueError("Exercise engine-state status must distinguish stale failure geometry")
        action = np.asarray(snapshot["last_action"])
        if (action.shape != (action_count,) or action.dtype.kind not in "fiu" or np.any(abs(action) > 1)
                or any(isinstance(value, bool) for value in snapshot["last_action"])):
            raise ValueError("Exercise snapshot action differs from its human-only control contract")
        if any(value != 0 for value in state["normalized_action"][action_count:]):
            raise ValueError("Exercise engine state contains forbidden object/root actuation")
        if status == "current" and state["normalized_action"][:action_count] != snapshot["last_action"]:
            raise ValueError("Exercise and engine actions disagree")
        metrics = snapshot["task_metrics"]
        if (not isinstance(metrics, dict) or set(metrics) != metric_keys or any(metrics[key] != value for key, value in fixed.items())
                or metrics["reset_protocol"] != _reset_protocol(snapshot["resolved_reset_options"])
                or not isinstance(metrics["evaluator_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", metrics["evaluator_sha256"])):
            raise ValueError("Exercise metrics fields, gates or source scope differ")
        if any(type(metrics[key]) not in (int, float) or not math.isfinite(metrics[key]) for key in numeric):
            raise ValueError("Exercise metrics require finite numeric values")
        observed, valid = metrics["observed_substeps"], metrics["valid_substeps"]
        if (type(observed) is not int or type(valid) is not int or not 0 <= valid <= observed <= maximum
                or not math.isclose(metrics["declared_duration_s"], duration, rel_tol=0, abs_tol=1e-9)
                or not math.isclose(metrics["valid_observed_duration_s"], elapsed(valid), rel_tol=0, abs_tol=1e-8)
                or not elapsed(valid) - 1e-8 <= metrics["simulated_duration_s"] <= elapsed(observed) + 1e-8):
            raise ValueError("Exercise duration/substep accounting differs")
        failure = metrics["failure"]
        if failure is not None and (not isinstance(failure, str) or not 1 <= len(failure) <= 240):
            raise ValueError("Exercise failure must be a bounded explicit reason")
        if valid < observed and failure is None:
            raise ValueError("Missing or invalid substeps cannot masquerade as an ongoing valid task")
        if (metrics["clock_status"] not in {"observed", "invalid_substep_clock", "unavailable_after_last_observed_time"}
                or metrics["clock_status"] != "observed" and failure is None):
            raise ValueError("Unavailable exercise clocks must remain explicit failures")
        if status == "last_valid_control_boundary" and (failure is None or state["time_s"] > metrics["simulated_duration_s"] + 1e-8):
            raise ValueError("Stale exercise geometry requires a later explicit failure")
        if status == "current" and not math.isclose(state["time_s"], metrics["simulated_duration_s"], rel_tol=0, abs_tol=1e-8):
            raise ValueError("Current engine geometry must match actual exercise time")
        failures = [failure] if failure else []
        for key in ("angular_rmse_rad", "root_rmse_model_units", "peak_angular_rmse_rad", "peak_root_rmse_model_units",
                    "max_floor_penetration_model_units"):
            value = metrics[key]
            if value is None:
                if valid or key not in {"angular_rmse_rad", "root_rmse_model_units"}:
                    raise ValueError("Exercise tracking values cannot be missing after valid substeps")
            elif type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Exercise tracking values must be finite and nonnegative")
            if value is None or value > gates[key] + 1e-12:
                failures.append(key)
        equipment = metrics["equipment"]
        if (not isinstance(equipment, list) or any(not isinstance(row, dict) for row in equipment)
                or [row.get("object_id") for row in equipment] !=
                equipment_order):
            raise ValueError("Exercise metrics must retain every equipment object in scene order")
        for row in equipment:
            keys = {"displacement_model_units", "orientation_rad", "speed_model_units_s", "angular_speed_rad_s"}
            if (set(row) != keys | {"object_id", "within_limits"}
                    or any(type(row[key]) not in (int, float) or row[key] < 0 or not math.isfinite(row[key]) for key in keys)):
                raise ValueError("Equipment stability metrics must be bounded nonnegative values")
            passed = all(row[key] <= gates["equipment_" + key] + 1e-12 for key in keys)
            if type(row["within_limits"]) is not bool or row["within_limits"] != passed:
                raise ValueError("Equipment stability cannot hide a whole-interval gate failure")
            if not passed:
                failures.append("equipment_stability:" + row["object_id"])
        contacts = metrics["contact_requirements"]
        if not isinstance(contacts, list) or len(contacts) != len(contact_windows):
            raise ValueError("Exercise contact requirement metrics are incomplete")
        for index, (row, (required, required_duration)) in enumerate(zip(contacts, contact_windows)):
            measures = {"required_duration_s", "observed_duration_s", "qualifying_duration_s", "current_gap_s", "max_gap_s",
                        "load_body_weight_fraction", "qualifying_fraction"}
            if (not isinstance(row, dict) or set(row) != set(required) | measures | {"qualifying_now", "active_now", "satisfied"}
                    or any(row[key] != value for key, value in required.items())
                    or any(type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] < 0 for key in measures)
                    or any(type(row[key]) is not bool for key in ("qualifying_now", "active_now", "satisfied"))):
                raise ValueError("Exercise contact metrics differ from the declared requirements")
            qualifying = row["qualifying_duration_s"] / required_duration
            if (not math.isclose(row["required_duration_s"], required_duration, rel_tol=0, abs_tol=1e-9)
                    or not 0 <= row["qualifying_duration_s"] <= row["observed_duration_s"] + 1e-8 <= required_duration + 2e-8
                    or not row["current_gap_s"] <= row["max_gap_s"] + 1e-8 <= row["observed_duration_s"] + 2e-8
                    or not math.isclose(row["qualifying_fraction"], qualifying, rel_tol=0, abs_tol=1e-8)):
                raise ValueError("Exercise contact duration accounting differs")
            passed = (math.isclose(row["observed_duration_s"], required_duration, abs_tol=1e-8)
                      and qualifying >= gates["contact_duration_fraction"] - 1e-12
                      and row["max_gap_s"] <= gates["max_contact_gap_s"] + 1e-12)
            if row["satisfied"] != passed:
                raise ValueError("Exercise contact success cannot ignore interval coverage or gaps")
            if not passed:
                failures.append("contact_requirement:" + str(index))
        completed = observed == valid == maximum and failure is None
        success = completed and not failures
        if (type(metrics["completed_interval"]) is not bool or metrics["completed_interval"] != completed
                or type(metrics["success"]) is not bool or metrics["success"] != success or metrics["failed_gates"] != failures):
            raise ValueError("Exercise success differs from complete independent task gates")
        if (metrics["failure_reward_penalty"] not in (0., -1.)
                or not math.isclose(metrics["mean_reward"], metrics["reward_integral"] / duration, abs_tol=1e-9)
                or any(not 0 <= metrics[key] <= 1 + 1e-9 for key in ("joint_effort_mean_squared", "root_assistance_mean_squared"))
                or root_assistance_disabled and metrics["root_assistance_mean_squared"] != 0):
            raise ValueError("Exercise effort/reward accounting differs from the action contract")
        expected_terminated = bool(failure or success)
        expected_truncated = bool(completed and not expected_terminated)
        if (type(snapshot["terminated"]) is not bool or type(snapshot["truncated"]) is not bool
                or snapshot["terminated"] != expected_terminated or snapshot["truncated"] != expected_truncated):
            raise ValueError("Exercise completion flags differ from task completion/failure")
        return deepcopy(snapshot)

    return validate


def _prepare_support_snapshot_validator(scene, task):
    """Validate support snapshots without broadening any legacy task/state contract."""
    support_task = _support_module()
    validate_engine = prepare_shared_snapshot_validator(scene)
    scene_sha, task_sha = canonical_hash(scene), canonical_hash(task)
    evaluator_sha = _support_evaluator_sha256()
    supports = {obj["id"]: obj for obj in scene["objects"]}
    action_count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    horizon, tick = task["interval"]["horizon_steps"], scene["dt"] / scene["substeps"]
    duration, maximum = horizon * scene["dt"], horizon * scene["substeps"]
    expected = {"schema_version", "profile", "scene_sha256", "task_sha256", "engine_state", "task_metrics", "last_action",
                "engine_state_status", "resolved_reset_options", "terminated", "truncated"}
    fixed = {"profile": support_task.PROFILE, "scene_sha256": scene_sha, "task_sha256": task_sha,
             "gate_profile": support_task.GATE_PROFILE, "gates_sha256": support_task.GATES_SHA256,
             "gates_provenance": support_task.GATES_PROVENANCE, "source_scope": task["source_scope"],
             "reward_profile": support_task.REWARD_PROFILE, "evaluator_sha256": evaluator_sha,
             "root_assistance": task["control"]["root_assistance"], "virtual_object_forces": "forbidden", "equipment": [],
             "gate_applicability": {"equipment_stability": "not_applicable", "declared_support": "required"}}
    tracking = ("angular_rmse_rad", "root_rmse_model_units", "peak_angular_rmse_rad", "peak_root_rmse_model_units")
    efforts = ("joint_effort_mean_squared", "root_assistance_mean_squared")
    finite = ("simulated_duration_s", "valid_observed_duration_s", "declared_duration_s",
              "max_floor_penetration_model_units", "reward_integral", "mean_reward", "failure_reward_penalty")
    metric_keys = {*fixed, *tracking, *efforts, *finite, "support_requirements", "completed_interval", "success", "failure",
                   "failed_gates", "observed_substeps", "valid_substeps", "clock_status", "reset_protocol"}
    gates = dict(support_task.GATES)
    windows = [(row, (row["end_step"] - row["start_step"]) * scene["dt"])
               for row in task["support_requirements"]]

    def numeric(value, *, lower=0., upper=1e12):
        return type(value) in (int, float) and math.isfinite(value) and lower <= value <= upper

    def validate(snapshot):
        try:
            raw = json.dumps(snapshot, allow_nan=False, separators=(",", ":"))
        except (TypeError, ValueError, RecursionError) as error:
            raise ValueError("Support snapshot must be finite JSON") from error
        if (len(raw.encode()) > 4 << 20 or not isinstance(snapshot, dict) or set(snapshot) != expected
                or type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1
                or snapshot["profile"] != STATE_PROFILE or snapshot["scene_sha256"] != scene_sha
                or snapshot["task_sha256"] != task_sha):
            raise ValueError("Support snapshot fields or pinned identities differ")
        options = snapshot["resolved_reset_options"]
        if options != _reset_options(options):
            raise ValueError("Support reset options must be resolved explicitly")
        state = snapshot["engine_state"]
        validate_engine(state)
        if state["controlled_id"] != "human" or state["target"] is not None:
            raise ValueError("Support snapshots require human control and no movable-object target")
        if [obj["id"] for obj in state["objects"]] != list(supports):
            raise ValueError("Support snapshot must retain all declared static objects")
        for obj in state["objects"]:
            original = supports[obj["id"]]
            if (obj["position"] != original["position"] or obj["quaternion"] != original["quaternion"]
                    or any(any(obj[key]) for key in ("velocity", "angular_velocity"))
                    or any(obj.get("virtual_force", (0., 0., 0.)))):
                raise ValueError("Static support pose, motion or force differs from its declared scene")
        status = snapshot["engine_state_status"]
        if status not in {"current", "last_valid_control_boundary"}:
            raise ValueError("Support geometry must distinguish a stale failure boundary")
        action = snapshot["last_action"]
        if (not isinstance(action, list) or len(action) != action_count
                or any(not numeric(value, lower=-1., upper=1.) for value in action)
                or any(state["normalized_action"][action_count:])
                or status == "current" and state["normalized_action"][:action_count] != action):
            raise ValueError("Support snapshot contains inconsistent or forbidden actions")
        metrics = snapshot["task_metrics"]
        if (not isinstance(metrics, dict) or set(metrics) != metric_keys
                or any(metrics[key] != value for key, value in fixed.items())
                or metrics["reset_protocol"] != _support_reset_protocol(options)
                or any(not numeric(metrics[key], lower=-1e12) for key in finite)):
            raise ValueError("Support metric fields, evaluator identity, criteria or source scope differ")
        observed, valid = metrics["observed_substeps"], metrics["valid_substeps"]
        if (type(observed) is not int or type(valid) is not int or not 0 <= valid <= observed <= maximum
                or not math.isclose(metrics["declared_duration_s"], duration, rel_tol=0, abs_tol=1e-9)
                or not math.isclose(metrics["valid_observed_duration_s"], valid * tick, rel_tol=0, abs_tol=1e-8)
                or not valid * tick - 1e-8 <= metrics["simulated_duration_s"] <= observed * tick + 1e-8):
            raise ValueError("Support duration/substep accounting differs")
        failure, clock = metrics["failure"], metrics["clock_status"]
        if failure is not None and (not isinstance(failure, str) or not 1 <= len(failure) <= 240):
            raise ValueError("Support failures must be bounded explicit reasons")
        if valid < observed and failure is None:
            raise ValueError("Missing support substeps require an explicit failure")
        if (clock not in {"not_observed", "observed", "invalid_substep_clock", "unavailable_after_last_observed_time"}
                or clock == "not_observed" and observed != 0
                or clock in {"invalid_substep_clock", "unavailable_after_last_observed_time"} and failure is None
                or observed == 0 and clock != "not_observed"):
            raise ValueError("Support clock availability contradicts observed samples")
        if (status == "current" and not math.isclose(state["time_s"], metrics["simulated_duration_s"], rel_tol=0, abs_tol=1e-8)
                or status == "last_valid_control_boundary" and (failure is None or state["time_s"] > metrics["simulated_duration_s"] + 1e-8)):
            raise ValueError("Support geometry clock contradicts current/failure state")
        failures = [failure] if failure else []
        for key in (*tracking, "max_floor_penetration_model_units"):
            value = metrics[key]
            if not valid and key in tracking and value is not None:
                raise ValueError("Unobserved support tracking must remain unavailable, not a fabricated zero")
            if value is None:
                if valid or key not in tracking:
                    raise ValueError("Support tracking cannot be missing after valid observations")
            elif not numeric(value):
                raise ValueError("Support tracking and penetration must be nonnegative finite values")
            if (value is None and key in tracking[:2]) or (value is not None and value > gates[key] + 1e-12):
                failures.append(key)
        if valid and any(metrics[mean] > metrics[peak] + 1e-12 for mean, peak in zip(tracking[:2], tracking[2:])):
            raise ValueError("Support mean tracking errors cannot exceed their observed peaks")
        for key in efforts:
            if (not valid and metrics[key] is not None) or (valid and not numeric(metrics[key], upper=1. + 1e-9)):
                raise ValueError("Support effort is unavailable until observed, then bounded by public actions")
        if valid and action_count == 19 and metrics["root_assistance_mean_squared"] != 0:
            raise ValueError("Disabled support root assistance cannot have nonzero effort")
        rows = metrics["support_requirements"]
        if not isinstance(rows, list) or len(rows) != len(windows):
            raise ValueError("Support requirement metric coverage differs")
        for index, (row, (requirement, required_duration)) in enumerate(zip(rows, windows)):
            values = {"required_duration_s", "observed_duration_s", "qualifying_duration_s", "current_gap_s", "max_gap_s"}
            optional = {"load_body_weight_fraction", "peak_load_body_weight_fraction", "qualifying_fraction"}
            flags = {"active_now", "qualifying_now", "satisfied"}
            if (not isinstance(row, dict) or set(row) != set(requirement) | values | optional | flags
                    or any(row[key] != value for key, value in requirement.items())
                    or any(not numeric(row[key], upper=60.) for key in values)
                    or any(row[key] is not None and not numeric(row[key], upper=1e30) for key in optional)
                    or any(type(row[key]) is not bool for key in flags)):
                raise ValueError("Support requirement values or authored identities differ")
            available = max(0., min(observed * tick, requirement["end_step"] * scene["dt"])
                            - requirement["start_step"] * scene["dt"])
            if (not math.isclose(row["required_duration_s"], required_duration, rel_tol=0, abs_tol=1e-9)
                    or not 0 <= row["qualifying_duration_s"] <= row["observed_duration_s"] + 1e-8 <= required_duration + 2e-8
                    or not max(0., available - (observed - valid) * tick) - 1e-8 <= row["observed_duration_s"] <= available + 1e-8
                    or not row["current_gap_s"] <= row["max_gap_s"] + 1e-8 <= row["observed_duration_s"] + 2e-8
                    or row["max_gap_s"] > row["observed_duration_s"] - row["qualifying_duration_s"] + 1e-8):
                raise ValueError("Support interval duration/coverage/gap accounting differs")
            fraction = row["qualifying_duration_s"] / required_duration if row["observed_duration_s"] else None
            if ((fraction is None) != (row["qualifying_fraction"] is None)
                    or fraction is not None and not math.isclose(row["qualifying_fraction"], fraction, rel_tol=0, abs_tol=1e-8)
                    or (row["peak_load_body_weight_fraction"] is None) != (row["observed_duration_s"] == 0)
                    or row["active_now"] != (row["load_body_weight_fraction"] is not None)
                    or row["active_now"] and row["observed_duration_s"] == 0):
                raise ValueError("Missing support loads and measured coverage must remain explicit")
            if valid == observed:
                active = observed > 0 and requirement["start_step"] <= (observed - 1) // scene["substeps"] < requirement["end_step"]
                if row["active_now"] != active:
                    raise ValueError("Support active window differs from its observed substep")
            if (row["load_body_weight_fraction"] is not None and
                    row["load_body_weight_fraction"] > row["peak_load_body_weight_fraction"] + 1e-9
                    or row["qualifying_now"] and (not row["active_now"] or row["current_gap_s"] != 0
                        or row["load_body_weight_fraction"] < requirement["minimum_load_body_weight_fraction"])
                    or row["active_now"] and not row["qualifying_now"] and row["current_gap_s"] < tick - 1e-9
                    or row["qualifying_duration_s"] > 0 and row["peak_load_body_weight_fraction"] < requirement["minimum_load_body_weight_fraction"]):
                raise ValueError("Support current load/qualification contradicts its observed peak or gap")
            passed = (math.isclose(row["observed_duration_s"], required_duration, rel_tol=0, abs_tol=1e-8)
                      and fraction is not None and fraction >= gates["contact_duration_fraction"] - 1e-12
                      and row["max_gap_s"] <= gates["max_contact_gap_s"] + 1e-12)
            if row["satisfied"] != passed:
                raise ValueError("Support success cannot bypass coverage or maximum-gap gates")
            if not passed:
                failures.append("support_requirement:" + str(index))
        completed = observed == valid == maximum and failure is None
        success = completed and not failures
        if (type(metrics["completed_interval"]) is not bool or metrics["completed_interval"] != completed
                or type(metrics["success"]) is not bool or metrics["success"] != success or metrics["failed_gates"] != failures):
            raise ValueError("Support success differs from complete independent task gates")
        penalty = metrics["failure_reward_penalty"]
        reward_before_effort = metrics["reward_integral"] - penalty * scene["dt"]
        if valid:
            reward_before_effort += (.01 * metrics[efforts[0]] + .02 * metrics[efforts[1]]) * valid * tick
        if (penalty not in (0., -1.) or penalty and failure is None
                or not -1e-8 <= reward_before_effort <= valid * tick + 1e-8
                or not math.isclose(metrics["mean_reward"], metrics["reward_integral"] / duration, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Support reward must reflect observed tracking/load and explicit action effort")
        terminated = bool(failure or success)
        truncated = bool(completed and not terminated)
        if (type(snapshot["terminated"]) is not bool or snapshot["terminated"] != terminated
                or type(snapshot["truncated"]) is not bool or snapshot["truncated"] != truncated):
            raise ValueError("Support completion flags differ from interval success/failure")
        return deepcopy(snapshot)

    return validate
