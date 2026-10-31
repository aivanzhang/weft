"""Pure authored static-support tasks; callers independently validate the native scene.

Only saved scene metadata and immutable engine substep samples enter this module.
Static collision shapes, load directions and thresholds are declared assumptions,
not recovered contact, grasp, anatomy or physical-validation evidence.
"""

from copy import deepcopy
import json
import math
import re
from types import MappingProxyType

from weft.io import canonical_hash
from weft.shared_world_observation import EquipmentState, HumanObjectContact, SubstepSample


SCENE_PROFILE = "shared-world-v4"
PROFILE = "shared-support-task-v1"
GATE_PROFILE = "shared-support-gates-v1"
REWARD_PROFILE = "tracking-times-declared-support-v1"
MAX_TASK_BYTES = 32 << 10
MAX_SCENE_BYTES = 8 << 20
MAX_REQUIREMENTS = 24
MAX_CONTACTS = 256
MAX_STEPS = 1800
SEGMENTS = frozenset(("torso", "head", "pelvis", "shoulders", "left_upper_arm", "left_forearm",
                      "left_thigh", "left_shin", "right_upper_arm", "right_forearm", "right_thigh", "right_shin"))
# These existing exercise limits are unchanged; the task identity is separate.
GATES = MappingProxyType({
    "angular_rmse_rad": .35, "root_rmse_model_units": .15,
    "max_floor_penetration_model_units": .025,
    "peak_angular_rmse_rad": .70, "peak_root_rmse_model_units": .30,
    "contact_distance_model_units": .001, "contact_normal_body_weight_fraction": .001,
    "contact_duration_fraction": .95, "max_contact_gap_s": .10,
})
GATES_SHA256 = canonical_hash(dict(GATES))
GATES_PROVENANCE = "Frozen authored simulator criteria, reusing existing exercise limits; no physical validation."


def evaluator_identity(implementation_sha256):
    """Combine caller-verified evaluator files; supplied hashes do not authenticate bytes."""
    if (not isinstance(implementation_sha256, dict)
            or set(implementation_sha256) != {"shared_exercise_task.py", "shared_support_task.py"}
            or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in implementation_sha256.values())):
        raise ValueError("Support evaluator identity requires exactly its two implementation hashes")
    return canonical_hash(implementation_sha256)


def _json(value, limit, label):
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(f"{label} must be finite bounded JSON") from error
    if len(encoded) > limit:
        raise ValueError(f"{label} exceeds its byte bound")


def _number(value, label, lower=-1e6, upper=1e6):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f"{label} must be a finite number in [{lower}, {upper}]")
    return float(value)


def _vector(value, length, label):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} requires {length} finite values")
    return tuple(_number(item, label) for item in value)


def _context(scene):
    _json(scene, MAX_SCENE_BYTES, "Support scene")
    if not isinstance(scene, dict) or scene.get("profile") != SCENE_PROFILE:
        raise ValueError("Support tasks require the explicit static-support scene profile")
    if scene.get("physical_validation") != "unverified" or scene.get("controlled_id") != "human":
        raise ValueError("Support scenes must retain human control and unverified physical scope")
    try:
        dt = _number(scene["dt"], "Control dt", 1 / 30 - 1e-12, 1 / 30 + 1e-12)
        substeps = scene["substeps"]
        interval = {key: scene["task"][key] for key in ("start_offset_s", "horizon_steps")}
        scale = _number(scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"], "Scale", .5, 2.)
        gravity = _vector(scene["gravity"], 3, "Gravity")
        clock = scene["coordinate_frame"]["clock_binding"]
        objects = scene["objects"]
        if (type(substeps) is not int or not 8 <= substeps <= 64
                or type(interval["horizon_steps"]) is not int or not 1 <= interval["horizon_steps"] <= MAX_STEPS
                or type(interval["start_offset_s"]) not in (int, float) or interval["start_offset_s"] != 0
                or clock["physics_substeps"] != substeps or clock["control_dt_s"] != dt):
            raise ValueError("Support clocks must preserve a bounded complete 30-Hz interval")
        if (scene["task"]["type"] != "human_support_v1" or scene["task"]["authored_addition"] is not True
                or scene["task"]["object_goal"] is not None or scene.get("target") is not None):
            raise ValueError("Static support has no movable-object goal or target")
        if not isinstance(objects, list) or not 1 <= len(objects) <= 12:
            raise ValueError("Support scenes require 1–12 declared static objects")
        by_id = {}
        for obj in objects:
            name = obj["id"]
            if (not isinstance(name, str) or not 1 <= len(name) <= 160 or name in by_id
                    or obj["role"] != "static" or obj["shape"] not in {"box", "sphere", "compound"}
                    or not isinstance(obj["provenance"], dict) or not obj["provenance"]):
                raise ValueError("Each support needs a unique static identity, shape and explicit provenance")
            _vector(obj["position"], 3, "Support position")
            _vector(obj["quaternion"], 4, "Support orientation")
            if any(_vector(obj[key], 3, "Static support velocity") != (0., 0., 0.)
                   for key in ("velocity", "angular_velocity")):
                raise ValueError("Static support cannot have velocity")
            by_id[name] = obj
        norm = math.sqrt(sum(value * value for value in gravity))
        if not 1e-9 < norm <= 20:
            raise ValueError("Support load fractions require nonzero bounded gravity")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Support scene metadata is incomplete or malformed") from error
    return {"dt": dt, "substeps": substeps, "interval": interval, "scale": scale,
            "gravity_norm": norm, "objects": by_id}


def _scope(objects):
    return {"success_scope": "authored_simulated_support_task", "physical_validation": "unverified",
            "source_contact_verified": False, "grasp_supported": False,
            "geometry_scope": "declared_static_collision_geometry_not_recovered_contact",
            "clock_alignment": "preserved_scene_clocks_not_retimed",
            "supports": [{"object_id": name, "object_sha256": canonical_hash(obj)} for name, obj in objects.items()]}


def build_shared_support_task(scene, *, support_requirements, root_assistance=False):
    """Bind authored segment/load windows to an independently validated static scene."""
    context = _context(scene)
    if type(root_assistance) is not bool:
        raise ValueError("root_assistance must be an explicit boolean")
    if not isinstance(support_requirements, (list, tuple)) or not 1 <= len(support_requirements) <= MAX_REQUIREMENTS:
        raise ValueError("Support tasks require 1–24 explicit load requirements")
    task = {"schema_version": 1, "profile": PROFILE, "scene_sha256": canonical_hash(scene),
            "gate_profile": GATE_PROFILE, "gates_sha256": GATES_SHA256, "reward_profile": REWARD_PROFILE,
            "interval": context["interval"], "authored_addition": True,
            "control": {"root_assistance": "explicit_3_axis" if root_assistance else "disabled",
                        "object_forces": "forbidden"},
            "support_requirements": deepcopy(list(support_requirements)), "source_scope": _scope(context["objects"])}
    return validate_shared_support_task(task, scene)


def validate_shared_support_task(task, scene):
    """Check metadata only; native scene/source admission remains the caller's prerequisite."""
    context = _context(scene)
    _json(task, MAX_TASK_BYTES, "Support task")
    if not isinstance(task, dict) or set(task) != {
        "schema_version", "profile", "scene_sha256", "gate_profile", "gates_sha256", "reward_profile",
        "interval", "authored_addition", "control", "support_requirements", "source_scope",
    }:
        raise ValueError("Support task fields differ from the versioned contract")
    if (type(task["schema_version"]) is not int or task["schema_version"] != 1 or task["profile"] != PROFILE
            or task["scene_sha256"] != canonical_hash(scene) or task["gate_profile"] != GATE_PROFILE
            or task["gates_sha256"] != GATES_SHA256 or task["reward_profile"] != REWARD_PROFILE
            or task["authored_addition"] is not True):
        raise ValueError("Support task identity, authored scope or frozen gates differ")
    interval = task["interval"]
    if (not isinstance(interval, dict) or set(interval) != {"start_offset_s", "horizon_steps"}
            or type(interval["horizon_steps"]) is not int or type(interval["start_offset_s"]) not in (int, float)
            or interval != context["interval"]):
        raise ValueError("Support task must preserve the complete declared scene interval")
    control = task["control"]
    if (not isinstance(control, dict) or set(control) != {"root_assistance", "object_forces"}
            or not isinstance(control["root_assistance"], str)
            or control["root_assistance"] not in {"disabled", "explicit_3_axis"} or control["object_forces"] != "forbidden"):
        raise ValueError("Support controls must forbid object forces and explicitly declare root assistance")
    rows = task["support_requirements"]
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_REQUIREMENTS:
        raise ValueError("Support tasks require 1–24 explicit load requirements")
    windows = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "object_id", "human_segment", "direction_world", "minimum_load_body_weight_fraction", "start_step", "end_step",
        }:
            raise ValueError("Support requirement fields differ; grasp and inferred palm/sole contacts are unsupported")
        if (not isinstance(row["object_id"], str) or row["object_id"] not in context["objects"]
                or not isinstance(row["human_segment"], str) or row["human_segment"] not in SEGMENTS):
            raise ValueError("Support requires a declared static object and existing named body segment; ground is unsupported")
        direction = _vector(row["direction_world"], 3, "Support direction")
        if not math.isclose(sum(value * value for value in direction), 1., rel_tol=0, abs_tol=1e-9):
            raise ValueError("Support direction must be an explicit unit vector, never silently normalized")
        threshold = _number(row["minimum_load_body_weight_fraction"], "Minimum support load", 0., 2.)
        if threshold <= 0:
            raise ValueError("Minimum support load must be positive")
        start, end = row["start_step"], row["end_step"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= interval["horizon_steps"]:
            raise ValueError("Support windows must be bounded half-open control-step intervals")
        key = (row["object_id"], row["human_segment"])
        if any(start < old_end and old_start < end for old_start, old_end in windows.get(key, [])):
            raise ValueError("The same support/segment load cannot count in overlapping requirements")
        windows.setdefault(key, []).append((start, end))
    if task["source_scope"] != _scope(context["objects"]):
        raise ValueError("Support task must preserve exact geometry/material/provenance identities and uncertainty")
    return deepcopy(task)


class SupportMetrics:
    """Bounded substep force accounting; no trajectory storage, physics or inferred forces."""

    def __init__(self, scene, task, initial_supports, initial_penetration, human_mass):
        self.task = validate_shared_support_task(task, scene)
        context = _context(scene)
        self.scene_sha256, self.task_sha256 = canonical_hash(scene), canonical_hash(task)
        self.dt, self.substeps, self.scale = context["dt"], context["substeps"], context["scale"]
        self.horizon = context["interval"]["horizon_steps"]
        self.duration, self.maximum_samples = self.dt * self.horizon, self.horizon * self.substeps
        self.weight = _number(human_mass, "Human mass", 1e-9, 1e6) * context["gravity_norm"]
        self._supports = {name: (tuple(obj["position"]), tuple(obj["quaternion"])) for name, obj in context["objects"].items()}
        self._check_supports(initial_supports)
        self.max_penetration = _number(initial_penetration, "Initial floor penetration", 0.)
        self.samples = self.valid_samples = 0
        self.elapsed = self.valid_duration = self.angular_squared = self.root_squared = 0.
        self.angular_peak = self.root_peak = self.joint_effort = self.root_effort = self.reward_integral = 0.
        self.failure_penalty = 0.
        self.failure = None
        self.clock_status = "not_observed"
        self.requirements = [{"required_duration_s": (row["end_step"] - row["start_step"]) * self.dt,
                              "observed_duration_s": 0., "qualifying_duration_s": 0., "current_gap_s": 0.,
                              "max_gap_s": 0., "active_now": False, "qualifying_now": False,
                              "load_body_weight_fraction": None, "peak_load_body_weight_fraction": None}
                             for row in self.task["support_requirements"]]
        self._pairs = {(row["object_id"], row["human_segment"]) for row in self.task["support_requirements"]}

    def _check_supports(self, values):
        if not isinstance(values, (list, tuple)) or len(values) != len(self._supports):
            raise ValueError("Every static support must be present in each observation")
        seen = set()
        for item in values:
            if (not isinstance(item, EquipmentState) or not isinstance(item.object_id, str)
                    or item.object_id not in self._supports or item.object_id in seen):
                raise ValueError("Static support observation identities differ")
            seen.add(item.object_id)
            position, quaternion = self._supports[item.object_id]
            if (_vector(item.position, 3, "Support position") != position
                    or _vector(item.quaternion, 4, "Support orientation") != quaternion
                    or any(_vector(getattr(item, field), 3, "Support motion/force") != (0.,) * 3
                           for field in ("velocity", "angular_velocity", "virtual_force"))):
                raise ValueError("Static support moved or received forbidden virtual actuation")

    def fail(self, reason):
        self.failure = self.failure or str(reason)[:240]

    def penalize_failure(self):
        if not self.failure_penalty:
            self.failure_penalty = -1.
            self.reward_integral -= self.dt

    def _loads(self, contacts):
        if not isinstance(contacts, (list, tuple)) or len(contacts) > MAX_CONTACTS:
            raise ValueError("Support contacts exceed the bounded engine contract")
        loads = {pair: [0., 0., 0., 0.] for pair in self._pairs}
        for contact in contacts:
            if (not isinstance(contact, HumanObjectContact) or not isinstance(contact.object_id, str)
                    or contact.object_id not in self._supports or not isinstance(contact.human_segment, str)
                    or contact.human_segment not in SEGMENTS):
                raise ValueError("Contact must name an existing body segment and declared static support")
            _vector(contact.position, 3, "Contact position")
            _vector(contact.normal_on_human_world, 3, "Contact normal")
            force = _vector(contact.force_on_human_world, 3, "Signed human contact force")
            distance = _number(contact.distance, "Contact distance")
            normal = _number(contact.normal_force, "Contact normal force", 0.)
            pair = (contact.object_id, contact.human_segment)
            if pair not in loads or distance > GATES["contact_distance_model_units"] * self.scale:
                continue
            loads[pair][0] += normal
            for axis in range(3):
                loads[pair][axis + 1] += force[axis]
        return loads

    def consume(self, sample):
        if not isinstance(sample, SubstepSample):
            raise ValueError("Support accounting requires an immutable engine SubstepSample")
        if self.samples >= self.maximum_samples:
            self.fail("substep_budget_exceeded")
            raise ValueError("Support substep budget exceeded")
        step, substep = divmod(self.samples, self.substeps)
        self.samples += 1
        if (type(sample.step_index) is not int or type(sample.substep_index) is not int
                or sample.step_index != step + 1 or sample.substep_index != substep + 1):
            self.fail("incomplete_or_misaligned_substep_observation")
            return
        start = _number(sample.interval_start_s, "Substep start", 0., 60.)
        contact_time = _number(sample.contact_time_s, "Contact clock", 0., 60.)
        if sample.interval_end_s is None:
            self.clock_status = "unavailable_after_last_observed_time"
            self.fail("missing_substep_clock")
            return
        end = _number(sample.interval_end_s, "Substep end", 0., 60. + 1e-8)
        if (not math.isclose(start, self.elapsed, rel_tol=0, abs_tol=1e-9)
                or not math.isclose(contact_time, start, rel_tol=0, abs_tol=1e-9)
                or not math.isclose(end - start, self.dt / self.substeps, rel_tol=0, abs_tol=1e-9)):
            self.clock_status = "invalid_substep_clock"
            self.fail("incomplete_or_misaligned_substep_observation")
            return
        self.elapsed, self.clock_status = end, "observed"
        if sample.valid is not True:
            self.fail(sample.failure or "invalid_substep_observation")
            return
        if sample.failure is not None:
            self.fail(sample.failure)
        q, _, target, action = (_vector(getattr(sample, field), 22, field) for field in
                                ("human_qpos", "human_qvel", "target_qpos", "human_action"))
        if any(abs(value) > 1 for value in action) or self.task["control"]["root_assistance"] == "disabled" and any(action[19:]):
            self.fail("forbidden_human_actuation")
        penetration = _number(sample.max_human_floor_penetration_model_units, "Observed floor penetration", 0.)
        self._check_supports(sample.equipment)
        loads = self._loads(sample.contacts)
        dt = end - start
        angular = sum((q[i] - target[i]) ** 2 for i in range(19)) / 19
        root = sum(((q[i] - target[i]) / self.scale) ** 2 for i in range(19, 22)) / 3
        self.valid_samples += 1
        self.valid_duration += dt
        self.angular_squared += angular * dt
        self.root_squared += root * dt
        self.angular_peak, self.root_peak = max(self.angular_peak, math.sqrt(angular)), max(self.root_peak, math.sqrt(root))
        self.max_penetration = max(self.max_penetration, penetration)
        scores = []
        for row, state in zip(self.task["support_requirements"], self.requirements):
            active = row["start_step"] <= sample.step_index - 1 < row["end_step"]
            state.update(active_now=active, qualifying_now=False, load_body_weight_fraction=None)
            if not active:
                continue
            normal, *force = loads[(row["object_id"], row["human_segment"])]
            load = max(0., sum(value * direction for value, direction in zip(force, row["direction_world"]))) / self.weight
            normal_ok = normal / self.weight >= GATES["contact_normal_body_weight_fraction"]
            qualifying = normal_ok and load >= row["minimum_load_body_weight_fraction"]
            state["load_body_weight_fraction"] = load
            state["peak_load_body_weight_fraction"] = max(state["peak_load_body_weight_fraction"] or 0., load)
            state["observed_duration_s"] += dt
            state["qualifying_now"] = bool(qualifying)
            state["qualifying_duration_s"] += dt if qualifying else 0.
            state["current_gap_s"] = 0. if qualifying else state["current_gap_s"] + dt
            state["max_gap_s"] = max(state["max_gap_s"], state["current_gap_s"])
            scores.append(min(1., load / row["minimum_load_body_weight_fraction"]) if normal_ok else 0.)
        effort = sum(value * value for value in action[:19]) / 19
        root_effort = sum(value * value for value in action[19:]) / 3
        self.joint_effort += effort * dt
        self.root_effort += root_effort * dt
        tracking = math.exp(-angular / GATES["angular_rmse_rad"] ** 2 - root / GATES["root_rmse_model_units"] ** 2)
        support = sum(scores) / len(scores) if scores else 1.
        self.reward_integral += (tracking * support - .01 * effort - .02 * root_effort) * dt

    def metrics(self):
        angular = math.sqrt(self.angular_squared / self.valid_duration) if self.valid_duration else None
        root = math.sqrt(self.root_squared / self.valid_duration) if self.valid_duration else None
        completed = (self.samples == self.valid_samples == self.maximum_samples
                     and math.isclose(self.elapsed, self.duration, rel_tol=0, abs_tol=1e-8) and self.failure is None)
        failures = []
        for name, value in (("angular_rmse_rad", angular), ("root_rmse_model_units", root),
                            ("peak_angular_rmse_rad", self.angular_peak), ("peak_root_rmse_model_units", self.root_peak),
                            ("max_floor_penetration_model_units", self.max_penetration)):
            if value is None or value > GATES[name] + 1e-12:
                failures.append(name)
        requirements = []
        for index, (row, state) in enumerate(zip(self.task["support_requirements"], self.requirements)):
            fraction = state["qualifying_duration_s"] / state["required_duration_s"] if state["observed_duration_s"] else None
            passed = (math.isclose(state["observed_duration_s"], state["required_duration_s"], rel_tol=0, abs_tol=1e-8)
                      and fraction is not None and fraction >= GATES["contact_duration_fraction"] - 1e-12
                      and state["max_gap_s"] <= GATES["max_contact_gap_s"] + 1e-12)
            requirements.append({**deepcopy(row), **state, "qualifying_fraction": fraction, "satisfied": passed})
            if not passed:
                failures.append("support_requirement:" + str(index))
        if self.failure:
            failures.insert(0, self.failure)
        return {"profile": PROFILE, "scene_sha256": self.scene_sha256, "task_sha256": self.task_sha256,
                "completed_interval": completed, "success": completed and not failures, "failure": self.failure,
                "failed_gates": failures, "observed_substeps": self.samples, "valid_substeps": self.valid_samples,
                "simulated_duration_s": self.elapsed, "valid_observed_duration_s": self.valid_duration,
                "declared_duration_s": self.duration, "clock_status": self.clock_status,
                "angular_rmse_rad": angular, "root_rmse_model_units": root,
                "peak_angular_rmse_rad": self.angular_peak if self.valid_samples else None,
                "peak_root_rmse_model_units": self.root_peak if self.valid_samples else None,
                "max_floor_penetration_model_units": self.max_penetration,
                "support_requirements": requirements, "equipment": [],
                "gate_applicability": {"equipment_stability": "not_applicable", "declared_support": "required"},
                "joint_effort_mean_squared": self.joint_effort / self.valid_duration if self.valid_duration else None,
                "root_assistance_mean_squared": self.root_effort / self.valid_duration if self.valid_duration else None,
                "reward_integral": self.reward_integral, "mean_reward": self.reward_integral / self.duration,
                "failure_reward_penalty": self.failure_penalty, "reward_profile": REWARD_PROFILE,
                "gates_sha256": GATES_SHA256, "gate_profile": GATE_PROFILE, "gates_provenance": GATES_PROVENANCE,
                "source_scope": deepcopy(self.task["source_scope"]),
                "root_assistance": self.task["control"]["root_assistance"], "virtual_object_forces": "forbidden"}
