"""Authored virtual-force placement scoring; sealed source admission remains external."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
from types import MappingProxyType

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from weft.io import canonical_hash
from weft.mobile_human import ROOT_AXES
from weft.shared_exercise_task import GATES
from weft.shared_world import SharedWorldEnv, validate_shared_world
from weft.shared_world_observation import SubstepSample

PROFILE = "moving-equipment-placement-task-v1"
SCOPE = ("Authored simulated object placement with explicit human/root motors and virtual object forces; "
         "not recovered exercise intent, grasp, measured physical motion or physical validation. "
         "Exact original start only; changed-start policy evaluation is not qualified.")
BODY_GATES = MappingProxyType({key: GATES[key] for key in ("angular_rmse_rad", "root_rmse_model_units",
    "peak_angular_rmse_rad", "peak_root_rmse_model_units", "max_floor_penetration_model_units")})
MAX_TASK_BYTES = 64 << 10


def _bounded(value, maximum):
    size = 0
    try:
        for chunk in json.JSONEncoder(allow_nan=False).iterencode(value):
            size += len(chunk.encode())
            if size > maximum:
                raise ValueError("Placement document exceeds its byte bound")
    except (TypeError, RecursionError) as error:
        raise ValueError("Placement requires bounded finite JSON") from error


def _number(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid placement {name}")
    return value


def _scorer_identity():
    return {name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("moving_equipment_task.py", "shared_exercise_task.py", "whole_body.py")}


def _binding(scene, bundle, candidate_id):
    # These joins pin an already admitted world; hashes alone never authenticate source evidence.
    _bounded(bundle, 32 << 20)
    if not isinstance(bundle, dict) or bundle.get("profile") != "moving-depth-selected-proposals-v1":
        raise ValueError("Placement requires an explicit selected depth family")
    from weft.moving_depth_family import validate_depth_selection
    validate_depth_selection(bundle["depth_selection"], bundle["depth_family"])
    candidates = [row for row in bundle["candidates"] if row["id"] == candidate_id]
    if len(candidates) != 1 or candidates[0]["status"] != "conditional_free_body_hypothesis":
        raise ValueError("Placement candidate is absent or unsupported")
    ids = sorted(row["object_id"] for row in candidates[0]["objects"])
    if ids != sorted(obj["id"] for obj in scene["objects"]):
        raise ValueError("Placement candidate object inventory differs")
    digest = canonical_hash(bundle)
    for obj in scene["objects"]:
        provenance = obj["provenance"]
        if (provenance.get("profile") != "moving-equipment-free-body-v1"
                or provenance.get("proposal_sha256") != digest or provenance.get("candidate_id") != candidate_id
                or provenance.get("trajectory_driving") is not False or provenance.get("hidden_assistance") is not False):
            raise ValueError("Placement scene differs from its explicit moving proposal")
    return {"scene_sha256": canonical_hash(scene), "bundle_sha256": digest, "candidate_id": candidate_id,
            "family_sha256": canonical_hash(bundle["depth_family"]),
            "selection_sha256": canonical_hash(bundle["depth_selection"]),
            "source_binding_sha256": canonical_hash(scene["source_binding"]),
            "native_clock_sha256": canonical_hash(scene["human_spec"].get("native_clock")),
            "source_admission": "external_sealed_exploration_loader_required"}


def build_moving_equipment_task(scene, selected_bundle, *, candidate_id, target_object_id,
        target_position_world, position_tolerance_world, speed_tolerance_world_s, hold_duration_s):
    scene = validate_shared_world(scene)
    task = {"schema_version": 1, "profile": PROFILE, "binding": _binding(scene, selected_bundle, candidate_id),
        "interval": {key: deepcopy(scene["task"][key]) for key in ("start_offset_s", "horizon_steps", "native_interval") if key in scene["task"]},
        "action_contract": deepcopy(scene["action_contract"]),
        "control": "explicit_human_root_and_virtual_object_forces", "reset": {"jitter": False},
        "goal": {"object_id": target_object_id, "position_world": deepcopy(target_position_world),
                 "position_tolerance_world": position_tolerance_world, "speed_tolerance_world_s": speed_tolerance_world_s,
                 "terminal_consecutive_hold_s": hold_duration_s},
        "body_gates": dict(BODY_GATES), "scorer_sha256": _scorer_identity(), "scope": SCOPE}
    return validate_moving_equipment_task(task, scene, selected_bundle)


def validate_moving_equipment_task(task, scene, selected_bundle):
    _bounded(task, MAX_TASK_BYTES)
    scene = validate_shared_world(scene)
    fields = {"schema_version", "profile", "binding", "interval", "action_contract", "control", "reset", "goal",
              "body_gates", "scorer_sha256", "scope"}
    if (not isinstance(task, dict) or set(task) != fields or type(task["schema_version"]) is not int
            or task["schema_version"] != 1 or task["profile"] != PROFILE or not isinstance(task["binding"], dict)):
        raise ValueError("Placement task has an unsupported closed contract")
    expected = _binding(scene, selected_bundle, task["binding"].get("candidate_id"))
    interval = {key: deepcopy(scene["task"][key]) for key in ("start_offset_s", "horizon_steps", "native_interval") if key in scene["task"]}
    if (task["binding"] != expected or canonical_hash(task["interval"]) != canonical_hash(interval)
            or task["action_contract"] != scene["action_contract"] or task["body_gates"] != BODY_GATES
            or task["scorer_sha256"] != _scorer_identity() or task["scope"] != SCOPE
            or task["control"] != "explicit_human_root_and_virtual_object_forces"
            or task["reset"] != {"jitter": False} or type(task["reset"].get("jitter")) is not bool):
        raise ValueError("Placement source, interval, controls or frozen scorer differs")
    if scene["task"]["start_offset_s"] != 0:
        raise ValueError("Placement v1 requires the complete original interval")
    goal = task["goal"]
    if not isinstance(goal, dict) or set(goal) != {"object_id", "position_world", "position_tolerance_world",
            "speed_tolerance_world_s", "terminal_consecutive_hold_s"}:
        raise ValueError("Placement goal fields differ")
    if goal["object_id"] not in scene["action_contract"]["dynamic_object_ids"]:
        raise ValueError("Placement target must be an admitted dynamic object")
    pos = goal["position_world"]
    if not isinstance(pos, list) or len(pos) != 3:
        raise ValueError("Placement goal requires three world coordinates")
    for value, bound in zip(pos, scene["bounds"]):
        _number(value, "target coordinate", 0, bound)
    _number(goal["position_tolerance_world"], "position tolerance", 1e-5, .5)
    _number(goal["speed_tolerance_world_s"], "speed tolerance", 1e-5, 1.)
    duration = interval.get("native_interval", {}).get("duration_s", scene["dt"] * interval["horizon_steps"])
    _number(goal["terminal_consecutive_hold_s"], "hold duration", min(scene["dt"], duration), duration)
    return deepcopy(task)


class PlacementMetrics:
    """Constant-memory scorer consuming every actual engine substep, including the partial last action."""

    def __init__(self, scene, task, initial_penetration=0.):
        self.task = deepcopy(task)
        self.task_sha256 = canonical_hash(self.task)
        self.goal = self.task["goal"]
        contract = scene["action_contract"]
        names = contract["names"][:contract["human_motor_count"]]
        self.roots = [names.index(name) for name in ROOT_AXES]
        self.angular = [i for i in range(len(names)) if i not in self.roots]
        self.count = len(names)
        self.ids = set(contract["dynamic_object_ids"])
        self.scale = scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"]
        self.dt, self.substeps = scene["dt"], scene["substeps"]
        self.horizon = task["interval"]["horizon_steps"]
        native = task["interval"].get("native_interval", {})
        self.duration = native.get("duration_s", self.horizon * self.dt)
        self.last_dt = native.get("terminal_duration_s", self.dt)
        self.samples = self.valid_samples = 0
        self.elapsed = self.angular_integral = self.root_integral = self.reward_integral = 0.
        self.angular_peak = self.root_peak = self.hold = 0.
        self.penetration = _number(initial_penetration, "initial penetration", 0., 100.)
        self.distance = self.speed = None
        self.failure = None

    def fail(self, reason):
        self.failure = self.failure or str(reason)[:300]

    def consume_boundary(self, diagnostics):
        # Contact telemetry precedes integration; retain the engine's final forward-check floor measurement too.
        penetration = diagnostics.get("max_floor_penetration_model_units")
        if type(penetration) not in (int, float) or not math.isfinite(penetration) or penetration < 0:
            self.fail("invalid_control_boundary_floor_observation")
            return
        self.penetration = max(self.penetration, penetration)

    def consume(self, sample):
        if not isinstance(sample, SubstepSample):
            raise ValueError("Placement requires immutable engine substep samples")
        step, substep = divmod(self.samples, self.substeps)
        self.samples += 1
        duration = (self.last_dt if step == self.horizon - 1 else self.dt) / self.substeps
        if (step >= self.horizon or sample.step_index != step + 1 or sample.substep_index != substep + 1
                or sample.interval_end_s is None or not math.isfinite(sample.interval_end_s)
                or not math.isclose(sample.interval_start_s, self.elapsed, rel_tol=0, abs_tol=1e-9)
                or not math.isclose(sample.contact_time_s, sample.interval_start_s, rel_tol=0, abs_tol=1e-9)
                or not math.isclose(sample.interval_end_s - sample.interval_start_s, duration, rel_tol=0, abs_tol=1e-9)):
            self.fail("incomplete_or_misaligned_substep_clock")
            return
        self.elapsed = sample.interval_end_s
        if not sample.valid or sample.failure:
            self.fail(sample.failure or "invalid_substep")
            return
        q, velocity, target, action = [np.asarray(getattr(sample, name), dtype=float) for name in
                                     ("human_qpos", "human_qvel", "target_qpos", "human_action")]
        if any(row.shape != (self.count,) or not np.isfinite(row).all() for row in (q, velocity, target, action)) or np.any(abs(action) > 1):
            self.fail("invalid_human_observation")
            return
        equipment = {item.object_id: item for item in sample.equipment}
        if set(equipment) != self.ids or len(equipment) != len(sample.equipment):
            self.fail("incomplete_equipment_observation")
            return
        for item in equipment.values():
            if any(len(getattr(item, field)) != size or not all(math.isfinite(v) for v in getattr(item, field))
                   for field, size in (("position", 3), ("velocity", 3), ("quaternion", 4), ("angular_velocity", 3), ("virtual_force", 3))):
                self.fail("invalid_equipment_observation")
                return
        penetration = sample.max_human_floor_penetration_model_units
        if type(penetration) not in (int, float) or not math.isfinite(penetration) or penetration < 0:
            self.fail("invalid_floor_observation")
            return
        self.valid_samples += 1
        angular = float(np.mean((q[self.angular] - target[self.angular]) ** 2))
        root = float(np.mean(((q[self.roots] - target[self.roots]) / self.scale) ** 2))
        self.angular_integral += angular * duration
        self.root_integral += root * duration
        self.angular_peak = max(self.angular_peak, math.sqrt(angular))
        self.root_peak = max(self.root_peak, math.sqrt(root))
        self.penetration = max(self.penetration, penetration)
        item = equipment[self.goal["object_id"]]
        self.distance = math.dist(item.position, self.goal["position_world"])
        self.speed = math.sqrt(sum(v*v for v in item.velocity))
        positioned = self.distance <= self.goal["position_tolerance_world"] and self.speed <= self.goal["speed_tolerance_world_s"]
        self.hold = self.hold + duration if positioned else 0.
        tracking = math.exp(-angular / BODY_GATES["angular_rmse_rad"]**2 - root / BODY_GATES["root_rmse_model_units"]**2)
        placement = math.exp(-(self.distance / self.goal["position_tolerance_world"])**2
                             -(self.speed / self.goal["speed_tolerance_world_s"])**2)
        self.reward_integral += (.5 * tracking + .5 * placement) * duration

    def metrics(self):
        values = {"angular_rmse_rad": math.sqrt(self.angular_integral / self.elapsed) if self.elapsed else None,
                  "root_rmse_model_units": math.sqrt(self.root_integral / self.elapsed) if self.elapsed else None,
                  "peak_angular_rmse_rad": self.angular_peak, "peak_root_rmse_model_units": self.root_peak,
                  "max_floor_penetration_model_units": self.penetration}
        failures = [name for name, value in values.items() if value is None or value > BODY_GATES[name]]
        complete = (self.samples == self.valid_samples == self.horizon * self.substeps
                    and math.isclose(self.elapsed, self.duration, rel_tol=0, abs_tol=1e-8) and self.failure is None)
        if self.hold + 1e-9 < self.goal["terminal_consecutive_hold_s"]:
            failures.append("terminal_consecutive_placement_hold")
        return {"profile": PROFILE, **values, "observed_substeps": self.samples, "valid_substeps": self.valid_samples,
                "elapsed_s": self.elapsed, "duration_s": self.duration, "object_distance_world": self.distance,
                "object_speed_world_s": self.speed, "terminal_consecutive_hold_s": self.hold,
                "completed_interval": complete, "success": bool(complete and not failures),
                "failure": self.failure, "failed_criteria": failures, "mean_reward": self.reward_integral / self.duration,
                "task_sha256": self.task_sha256, "scope": SCOPE}


class MovingEquipmentTaskEnv(gym.Env):
    """Reward-only wrapper: the original simulator owns every state transition and action channel."""

    metadata = SharedWorldEnv.metadata

    def __init__(self, scene, task, selected_bundle, render_mode=None):
        self._scene = validate_shared_world(scene)
        self._task = validate_moving_equipment_task(task, self._scene, selected_bundle)
        self._metrics = None
        self._done = True
        self._engine = SharedWorldEnv(self._scene, render_mode, substep_observer=self._observe)
        self.action_names = tuple(self._scene["action_contract"]["names"])
        self.action_space = self._engine.action_space
        # The engine's last three goal coordinates describe its legacy parking diagnostic, not this task.
        size = self._engine.observation_space.shape[0] + 8
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)

    @property
    def scene(self):
        return deepcopy(self._scene)

    @property
    def task(self):
        return deepcopy(self._task)

    def _observe(self, sample):
        self._metrics.consume(sample)

    def _observation(self, base):
        goal = self._task["goal"]
        return np.concatenate((base, [*goal["position_world"], goal["position_tolerance_world"],
            goal["speed_tolerance_world_s"], goal["terminal_consecutive_hold_s"], self._metrics.hold,
            self._metrics.elapsed / self._metrics.duration])).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if options is not None and (options != {"jitter": False} or type(options.get("jitter")) is not bool):
            raise ValueError("Placement v1 permits only the exact original start; changed-start evaluation is unqualified")
        self._done, self._metrics = True, None
        super().reset(seed=seed)
        observation, info = self._engine.reset(seed=seed, options={"jitter": False})
        state = self._engine.snapshot()
        self._metrics = PlacementMetrics(self._scene, self._task, state["human"]["mobility"]["max_penetration"])
        self._done = False
        return self._observation(observation), {"task": self.metrics(), "engine_reset": info, "scope": SCOPE}

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset placement before stepping and after completion")
        before = self._metrics.reward_integral
        try:
            observation, _, terminated, truncated, diagnostics = self._engine.step(action)
        except RuntimeError as error:
            self._metrics.fail("engine_failure:" + str(error))
            self._done = True
            raise
        self._metrics.consume_boundary(diagnostics)
        self._done = bool(terminated or truncated)
        if terminated:
            self._metrics.fail(diagnostics.get("failure") or "engine_terminated")
        reward = (self._metrics.reward_integral - before) / self._metrics.duration
        return self._observation(observation), reward, terminated, truncated, {"task": self.metrics(), "engine_diagnostics": diagnostics}

    def metrics(self):
        if self._metrics is None:
            raise RuntimeError("Reset before requesting placement metrics")
        return self._metrics.metrics()

    def snapshot(self):
        return {"task": self.metrics(), "engine_diagnostics": self._engine.snapshot()}

    def close(self):
        self._engine.close()
