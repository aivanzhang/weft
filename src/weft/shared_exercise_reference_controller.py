"""Source-fit control alternatives; the original physical scene and scored targets stay fixed.

The caller authenticates seals/runtime bytes, validates the original scene/task, and
supplies the trusted human.reference_velocity_index from that pinned runtime. Hash
claims and a callback supplied by an arbitrary caller are not authentication.
"""

from bisect import bisect_right
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import re
from types import MappingProxyType
from typing import Callable

import numpy as np

from weft.io import canonical_hash
from weft.spatial_pose import SPATIAL_JOINT_LIMITS, SPATIAL_JOINT_NAMES


PROFILE = "shared-exercise-source-fit-controller-v1"
RECIPE_PROFILE = "shared-exercise-source-fit-controller-recipe-v1"
MAX_RECIPE_BYTES = 1 << 20
RECIPE_INTERPRETATION = "archived_execution_only_not_source_fit_admission"
RUNTIME_SHA256 = MappingProxyType({
    "human.py": "dab7a88a688e3a1c243d0ed64b3abf49dfe5704dac8d9d3101926fd5937cb340",
    "shared_world.py": "5400e2312523061dee4015d177f34a49a74b6f565bd79bfd03c6043b67970c10",
})
PROPOSAL_GATES = frozenset({
    "complete_bounded_solve", "first_reference_frozen", "fixed_nonreference_fields",
    "source_times_and_count", "angle_correction", "correction_rate", "joint_limits",
    "pointwise_floor_nonregression", "floor_feasibility", "endpoint_depth_change",
    "source_endpoint_image_nonregression",
})
METHODS = frozenset({"blocked-whole-body-floor-reference-v1", "blocked-whole-body-floor-first-reference-v1"})
MAX_REFERENCES = 900
TOLERANCE = 1e-8
_SHA = re.compile(r"[0-9a-f]{64}")
_ROOT_NAMES = ("root_x", "root_y", "root_z")
_UNITS = {"angular_velocity": "world_radians_per_second", "length": "body_scaled_world_units",
          "mass": "kilogram_prior", "orientation": "quaternion_wxyz", "time": "second"}
_SCOPE = ("Source-fit authored controller alternative on an unchanged original simulator task. "
          "Not a learned policy, repaired geometry, independent anatomy validation or physical transfer.")
_RECIPE_KEYS = frozenset({"schema_version", "profile", "interpretation", "scene_sha256", "task_sha256",
    "gates_sha256", "action_contract_sha256", "original_human_canonical_sha256", "input_human_sha256",
    "candidate_human_sha256", "candidate_canonical_sha256", "proposal_report_sha256", "input_evidence_sha256",
    "source_sha256", "runtime_sha256", "feedback_gains", "clock_binding", "refinement_marker", "targets"})
_PROVENANCE_KEYS = ("input_human_sha256", "candidate_human_sha256", "proposal_report_sha256", "input_evidence_sha256")


def _require(condition, message):
    if not condition:
        raise ValueError("Source-fit reference controller: " + message)


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _json(raw, limit):
    _require(type(raw) is bytes and 0 < len(raw) <= limit, "bounded exact JSON bytes required")
    def pairs(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=pairs)
        json.dumps(result, allow_nan=False)
    except (UnicodeError, ValueError, TypeError, RecursionError) as error:
        raise ValueError("Source-fit reference controller: invalid finite JSON") from error
    _require(type(result) is dict, "JSON object required")
    return result


def _copy(value, limit):
    _require(type(value) is dict, "validated scene/task/action objects required")
    return _json(json.dumps(value, allow_nan=False).encode(), limit)


def _tuple_rows(values):
    return tuple(tuple(map(float, row)) for row in values)


def _same_json(left, right):
    # Python equality conflates JSON booleans and numbers; fixed source fields must not.
    return canonical_hash(left) == canonical_hash(right)


def _same_action_contract(actual, items):
    return (type(actual) is dict and set(actual) == {k for k, _ in items}
            and all((isinstance(actual[k], list) and tuple(actual[k]) == v) if isinstance(v, tuple)
                    else actual[k] == v for k, v in items))


@dataclass(frozen=True, slots=True)
class ReferenceController:
    """Prepared immutable provider; bind it to its authenticated environment once."""

    _times: tuple
    _original: tuple
    _candidate: tuple
    _original_slopes: tuple
    _candidate_slopes: tuple
    _limits: tuple
    _action_contract_items: tuple
    _identity_json: str
    _recipe_json: str
    _scene_sha256: str
    _task_sha256: str
    _gates_sha256: str
    _duration: float
    _timestep: float
    _velocity_index: Callable = field(repr=False, compare=False)
    _unchanged: bool

    def describe(self):
        return json.loads(self._identity_json)

    def recipe(self):
        """Return detached execution data; the recording seal must authenticate it."""
        return json.loads(self._recipe_json)

    def target_delta(self, elapsed_s):
        """Return joint-angle/rate differences at the existing native simulation clock."""
        _require(_number(elapsed_s, 0., self._duration + 1e-9), "elapsed clock outside the fixed task")
        if self._unchanged:
            return np.zeros(19), np.zeros(19)
        timestamp = float(np.clip(self._times[0] + elapsed_s, self._times[0], self._times[-1]))
        index = min(len(self._times) - 2, max(0, bisect_right(self._times, timestamp) - 1))
        offset = timestamp - self._times[index]
        old = np.asarray(self._original[index]) + np.asarray(self._original_slopes[index]) * offset
        new = np.asarray(self._candidate[index]) + np.asarray(self._candidate_slopes[index]) * offset
        velocity_index = self._velocity_index(self._times, timestamp, elapsed=elapsed_s,
            timestep=self._timestep, source_start=self._times[0])
        _require(velocity_index is None or type(velocity_index) is int and 0 <= velocity_index < len(self._times) - 1,
                 "trusted velocity selector returned an invalid index")
        velocity = (np.zeros(19) if velocity_index is None else
                    np.asarray(self._candidate_slopes[velocity_index]) - np.asarray(self._original_slopes[velocity_index]))
        position = new - old
        _require(np.isfinite(position).all() and np.isfinite(velocity).all()
                 and np.max(abs(position)) <= .15 + TOLERANCE
                 and np.max(abs(velocity)) <= 1.5 + TOLERANCE, "sampled correction exceeds admitted bounds")
        return position, velocity

    def action(self, env):
        """Read bounded public state; never change the environment, task or source arrays."""
        contract = env.action_contract()
        _require(_same_action_contract(contract, self._action_contract_items), "environment action contract changed")
        metrics = env.metrics()
        _require(isinstance(metrics, dict) and metrics.get("scene_sha256") == self._scene_sha256
                 and metrics.get("task_sha256") == self._task_sha256
                 and metrics.get("gates_sha256") == self._gates_sha256
                 and metrics.get("clock_status") == "observed" and metrics.get("failure") is None,
                 "environment identity or observed clock differs")
        position, velocity = self.target_delta(metrics.get("simulated_duration_s"))
        raw = np.asarray(env.feature_matrix())
        _require(raw.shape == (len(self._limits), 3) and raw.dtype == np.dtype(float),
                 "public feedback must retain the pinned float64 matrix contract")
        features = raw.astype(float, copy=True)
        _require(np.isfinite(features).all() and np.max(abs(features)) <= 1e6, "invalid public feedback features")
        unclipped = features.sum(axis=1)
        # These exact 40/5 gains belong to the two reviewed runtime hashes above.
        delta = (40. * position + 5. * velocity) / np.asarray(self._limits[:19])
        action = np.clip(unclipped, -1., 1.)
        changed = np.flatnonzero(delta)
        action[changed] = np.clip(unclipped[changed] + delta[changed], -1., 1.)
        _require(np.isfinite(action).all() and np.max(abs(action)) <= 1., "invalid normalized action")
        return action


def prepare_reference_controller(original_human_bytes, candidate_human_bytes, report_bytes, *,
                                 scene, task, action_contract, runtime_sha256, reference_velocity_index):
    """Bind authenticated inputs to public-action feedback without admitting new physics.

    Only complete shared-world-v2/v3 tasks with the original nominal frame/FPS
    mapping are supported. The caller verifies seals, current runtime bytes and
    scene/task admission; this provider checks their numerical/identity boundary.
    """
    try:
        return _prepare(original_human_bytes, candidate_human_bytes, report_bytes, scene=scene, task=task,
                        action_contract=action_contract, runtime_sha256=runtime_sha256,
                        reference_velocity_index=reference_velocity_index)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError("Source-fit reference controller: malformed input contract") from error


def prepare_reference_recipe(recipe, *, scene, task, action_contract, runtime_sha256, reference_velocity_index):
    """Replay an externally sealed recipe, without requalifying its source-fit claim.

    Raw artifact hashes remain archived identifiers. Only the reconstructed target's
    canonical identity, fixed original scene/task and executable bounds are checked.
    """
    try:
        return _prepare_recipe(recipe, scene, task, action_contract, runtime_sha256, reference_velocity_index)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError("Source-fit reference controller: malformed archived recipe") from error


def _prepare_recipe(recipe, scene, task, action_contract, runtime_sha256, reference_velocity_index):
    recipe = _copy(recipe, MAX_RECIPE_BYTES)
    scene, task, action = _copy(scene, 8 << 20), _copy(task, 64 << 10), _copy(action_contract, 16 << 10)
    _require(set(recipe) == _RECIPE_KEYS and type(recipe["schema_version"]) is int and recipe["schema_version"] == 1
             and recipe["profile"] == RECIPE_PROFILE and recipe["interpretation"] == RECIPE_INTERPRETATION,
             "unsupported archived recipe fields/profile")
    _require(all(isinstance(recipe[k], str) and _SHA.fullmatch(recipe[k])
                 for k in _RECIPE_KEYS if k.endswith("_sha256") and k != "runtime_sha256"),
             "invalid archived artifact identity")
    original = scene["human_spec"]
    _require(recipe["scene_sha256"] == canonical_hash(scene) and recipe["task_sha256"] == canonical_hash(task)
             and recipe["action_contract_sha256"] == canonical_hash(action)
             and recipe["original_human_canonical_sha256"] == canonical_hash(original)
             and recipe["source_sha256"] == original["source_sha256"]
             and recipe["gates_sha256"] == task["gates_sha256"], "archived original scene/task identity differs")
    _require(_same_json(recipe["runtime_sha256"], dict(RUNTIME_SHA256))
             and _same_json(recipe["feedback_gains"], {"joint_position": 40., "joint_velocity": 5.})
             and _same_json(recipe["clock_binding"], scene["coordinate_frame"]["clock_binding"]),
             "archived runtime, gains or clock mapping changed")
    marker = recipe["refinement_marker"]
    _require(isinstance(marker, dict) and marker.get("method") in METHODS, "unsupported archived refinement marker")
    expected_marker = {"schema_version": 1, "method": marker["method"], "status": "unadmitted_construction_proposal",
        "promoted": False, "input_human_sha256": recipe["input_human_sha256"],
        "input_evidence_sha256": recipe["input_evidence_sha256"]}
    _require(_same_json(marker, expected_marker), "archived refinement marker changed")
    targets, references = recipe["targets"], original["reference"]
    _require(isinstance(targets, list) and isinstance(references, list)
             and 2 <= len(targets) == len(references) <= MAX_REFERENCES, "archived reference count differs")
    candidate = deepcopy(original)
    identity_rows = []
    for target, row in zip(targets, candidate["reference"], strict=True):
        _require(isinstance(target, dict) and set(target) == {"evidence_id", "time_s", "joints"}
                 and _same_json(target["time_s"], row["time_s"]), "archived target fields or clocks changed")
        match = re.fullmatch(r"[0-9a-f]{64}:frame:(0|[1-9][0-9]*)", target["evidence_id"])
        _require(match is not None and len(match[1]) <= 7, "invalid archived reference identity")
        identity_rows.append({"evidence_id": target["evidence_id"], "time_s": target["time_s"], "frame": int(match[1])})
        row["joints"] = deepcopy(target["joints"])
    candidate["whole_body_reference_refinement"] = deepcopy(marker)
    _require(canonical_hash(candidate) == recipe["candidate_canonical_sha256"], "archived candidate identity differs")
    _context(original, candidate, scene, runtime_sha256, reference_velocity_index)
    provider = _provider(original, candidate, identity_rows, scene, task, action,
        {k: recipe[k] for k in _PROVENANCE_KEYS}, reference_velocity_index, RECIPE_INTERPRETATION)
    _require(_same_json(provider.recipe(), recipe), "archived execution document changed on reconstruction")
    return provider


def _prepare(original_raw, candidate_raw, report_raw, *, scene, task, action_contract, runtime_sha256,
             reference_velocity_index):
    original, candidate = _json(original_raw, 8 << 20), _json(candidate_raw, 8 << 20)
    report = _json(report_raw, 16 << 20)
    scene, task, action = _copy(scene, 8 << 20), _copy(task, 64 << 10), _copy(action_contract, 16 << 10)
    _context(original, candidate, scene, runtime_sha256, reference_velocity_index)
    source_sha = original["source_sha256"]
    _require(report.get("source_sha256") == source_sha, "source identity mismatch")
    method = report.get("profile")
    status = "unadmitted_construction_proposal"
    _require(type(report.get("schema_version")) is int and report["schema_version"] == 1
             and method in METHODS and report.get("status") == status
             and report.get("promoted") is False and report.get("physical_validation") == "unverified"
             and report.get("geometry_admission") == "unadmitted_pending_independent_current_scene_checks"
             and report.get("failures") == [] and report.get("budget_reason") is None,
             "proposal is not complete, passing and unpromoted")
    _require(report.get("input_human_sha256") == sha256(original_raw).hexdigest()
             and report.get("candidate_sha256") == canonical_hash(candidate)
             and isinstance(report.get("input_evidence_sha256"), str) and _SHA.fullmatch(report["input_evidence_sha256"]),
             "proposal input/candidate hashes differ")
    marker = {"schema_version": 1, "method": method, "status": status, "promoted": False,
              "input_human_sha256": report["input_human_sha256"], "input_evidence_sha256": report["input_evidence_sha256"]}
    _require(_same_json(candidate.get("whole_body_reference_refinement"), marker), "candidate provenance differs")
    gates = report.get("gates")
    _require(isinstance(gates, dict) and set(gates) == PROPOSAL_GATES and all(v is True for v in gates.values()),
             "all eleven unchanged proposal gates must pass")
    contract = report["contract"]
    _require(all(contract.get(k) == v for k, v in {
        "method": method, "maximum_references": MAX_REFERENCES, "maximum_angle_correction_rad": .15,
        "maximum_correction_rate_rad_s": 1.5, "maximum_temporal_gap_s": .25,
        "floor_penetration_limit_model_units": .025, "numerical_tolerance": TOLERANCE,
    }.items()), "proposal correction or floor limits changed")
    provenance = {"input_human_sha256": sha256(original_raw).hexdigest(),
                  "candidate_human_sha256": sha256(candidate_raw).hexdigest(),
                  "proposal_report_sha256": sha256(report_raw).hexdigest(),
                  "input_evidence_sha256": report["input_evidence_sha256"]}
    return _provider(original, candidate, report["references"], scene, task, action, provenance,
                     reference_velocity_index, "saved_passed_proposal_boundary_checked")


def _context(original, candidate, scene, runtime_sha256, reference_velocity_index):
    _require(isinstance(runtime_sha256, dict) and all(runtime_sha256.get(k) == v for k, v in RUNTIME_SHA256.items())
             and callable(reference_velocity_index), "unsupported feedback/velocity runtime")
    _require(scene.get("profile") in {"shared-world-v2", "shared-world-v3"}
             and scene.get("physical_validation") == "unverified" and scene.get("controlled_id") == "human"
             and _same_json(scene.get("human_spec"), original), "original scene/body mismatch")
    _require(original.get("schema_version") == 3 and original.get("profile") == "exercise-v3"
             and "camera_reference" not in original and "whole_body_reference_refinement" not in original,
             "unsupported or already-refined input body")
    _require(_same_json({k: v for k, v in candidate.items() if k not in {"reference", "whole_body_reference_refinement"}},
                       {k: v for k, v in original.items() if k != "reference"}), "nonreference body fields changed")
    source_sha = original.get("source_sha256")
    _require(isinstance(source_sha, str) and _SHA.fullmatch(source_sha)
             and scene["source_binding"]["source_sha256"] == source_sha, "source identity mismatch")


def reference_action_contract(scene, task):
    """Derive the existing public human-only action shape from validated scene/task data."""
    try:
        control = task["control"]
        _require(control in ({"root_assistance": "disabled", "object_forces": "forbidden"},
                             {"root_assistance": "explicit_3_axis", "object_forces": "forbidden"}), "unsupported control task")
        count = 22 if control["root_assistance"] == "explicit_3_axis" else 19
        source = scene["action_contract"]
        return {"names": list(source["names"][:count]), "normalized_bounds": [-1., 1.],
            "joint_channels": list(range(19)), "root_assistance_channels": list(range(19, count)),
            "units": list(source["units"][:count]), "physical_limits": list(source["physical_limits"][:count]),
            "root_assistance": control["root_assistance"], "object_force_channels": []}
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Source-fit reference controller: malformed action contract") from error


def _provider(original, candidate, identity_rows, scene, task, action, provenance, reference_velocity_index,
              qualification):
    source_sha = original["source_sha256"]
    old_refs, new_refs, records, rows = (original["reference"], candidate["reference"],
                                       original["reference_evidence"], identity_rows)
    _require(all(isinstance(v, list) for v in (old_refs, new_refs, records, rows))
             and 2 <= len(old_refs) == len(new_refs) == len(records) == len(rows) <= MAX_REFERENCES,
             "reference counts differ or exceed bounds")
    names = tuple(action["names"][:19])
    _require(len(names) == 19 and set(names) == set(SPATIAL_JOINT_NAMES), "nineteen unique joint channels required")
    times, old_values, new_values = [], [], []
    frame_ids = set()
    clock = scene["coordinate_frame"]["clock_binding"]
    fps = clock["source_fps"]
    _require(_number(fps, 5., 240.), "unsupported source FPS")
    for old, new, record, row in zip(old_refs, new_refs, records, rows, strict=True):
        stamp = old["time_s"]
        _require(_number(stamp, 0., 60.) and (not times or times[-1] < stamp <= times[-1] + .25 + 1e-9),
                 "invalid clock or unsupported source gap")
        _require(_same_json({k: v for k, v in old.items() if k != "joints"},
                            {k: v for k, v in new.items() if k != "joints"})
                 and _number(record["time_s"], 0., 60.) and _number(row["time_s"], 0., 60.)
                 and record["time_s"] == row["time_s"] == stamp, "source/reference clocks changed")
        frame = row["frame"]
        identity = f"{source_sha}:frame:{frame}"
        _require(type(frame) is int and 0 <= frame <= 1_000_000 and frame not in frame_ids
                 and record["evidence_id"] == row["evidence_id"] == identity and stamp == frame / fps,
                 "evidence identity or nominal source clock mismatch")
        frame_ids.add(frame)
        for item in (old, new):
            _require(isinstance(item["joints"], dict) and set(item["joints"]) == set(names)
                     and all(_number(item["joints"][name], *SPATIAL_JOINT_LIMITS[name]) for name in names),
                     "invalid joint values/names")
        times.append(float(stamp))
        old_values.append([old["joints"][name] for name in names])
        new_values.append([new["joints"][name] for name in names])
    _require(_same_json(old_refs[0], new_refs[0]), "first pose changed")
    roots = original["root_reference"]
    _require(isinstance(roots, list) and len(roots) == len(times)
             and all(r["time_s"] == t and isinstance(r["translation"], list) and len(r["translation"]) == 3
                     and all(_number(v, -1e4, 1e4) for v in r["translation"]) for r, t in zip(roots, times)),
             "root/reference clocks differ")
    old_array, new_array = np.asarray(old_values, dtype=float), np.asarray(new_values, dtype=float)
    spans = np.diff(times)[:, None]
    correction = new_array - old_array
    correction_rate = np.diff(correction, axis=0) / spans
    _require(np.max(abs(correction)) <= .15 + TOLERANCE and np.max(abs(correction_rate)) <= 1.5 + TOLERANCE,
             "actual correction/rate exceeds proposal gates")
    old_slopes, new_slopes = np.diff(old_array, axis=0) / spans, np.diff(new_array, axis=0) / spans
    _require(np.isfinite(old_slopes).all() and np.isfinite(new_slopes).all()
             and max(np.max(abs(old_slopes)), np.max(abs(new_slopes))) <= 1e4, "reference velocity exceeds engine envelope")
    interval, scene_interval = task["interval"], scene["task"]
    horizon, dt, substeps = interval["horizon_steps"], scene["dt"], scene["substeps"]
    _require(dt == original.get("dt") == 1 / 30 and type(substeps) is int and 8 <= substeps <= 64
             and type(horizon) is int and 1 <= horizon <= 1800 and interval["start_offset_s"] == 0
             and interval == {k: scene_interval[k] for k in ("start_offset_s", "horizon_steps")}
             and math.isclose(horizon * dt, times[-1] - times[0], rel_tol=0., abs_tol=1e-9),
             "only unchanged complete reference intervals are supported")
    expected_clock = {"alignment_method": "pose_frame_index_over_fps_prior_v1", "control_dt_s": dt,
                      "physics_substeps": substeps, "source_fps": fps, "source_time_offset_s": times[0]}
    frame = scene["coordinate_frame"]
    _require(clock == expected_clock and frame.get("axes") == "x_right_y_depth_z_up" and frame.get("units") == _UNITS
             and frame["scale_bridge"]["method"] == "monocular_body_height_world_prior_v1"
             and _number(frame["scale_bridge"]["body_height_to_world_scale"], .5, 2.),
             "unsupported source/world clock, axes or unit mapping")
    profile = "shared-exercise-task-v2" if scene["profile"] == "shared-world-v3" else "shared-exercise-task-v1"
    _require(task.get("profile") == profile and type(task.get("schema_version")) is int and task["schema_version"] == 1
             and task.get("scene_sha256") == canonical_hash(scene) and task.get("gate_profile") == "shared-exercise-gates-v1"
             and isinstance(task.get("gates_sha256"), str) and _SHA.fullmatch(task["gates_sha256"]), "task/scene identity differs")
    control = task["control"]
    _require(control in ({"root_assistance": "disabled", "object_forces": "forbidden"},
                         {"root_assistance": "explicit_3_axis", "object_forces": "forbidden"}), "unsupported control task")
    count = 22 if control["root_assistance"] == "explicit_3_axis" else 19
    expected_action = reference_action_contract(scene, task)
    _require(action == expected_action and len(action["names"]) == count
             and (count == 19 or tuple(action["names"][19:]) == _ROOT_NAMES)
             and len(action["units"]) == len(action["physical_limits"]) == count
             and all(_number(v, 1e-9, 1e9) for v in action["physical_limits"]), "action channels/limits differ")
    _require(all(action["physical_limits"][i] == original.get("motor_limits", {}).get(name, 30.)
                 for i, name in enumerate(names)), "joint limits differ from pinned feedback normalization")
    identity = {"profile": PROFILE, "controller_kind": "source_fit_authored_alternative", "learned": False,
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "gates_sha256": task["gates_sha256"],
        **provenance, "candidate_canonical_sha256": canonical_hash(candidate), "source_sha256": source_sha,
        "reference_count": len(times), "clock_binding": expected_clock, "runtime_sha256": dict(RUNTIME_SHA256),
        "feedback_gains": {"joint_position": 40., "joint_velocity": 5.}, "scope": _SCOPE,
        "qualification": qualification}
    recipe = {"schema_version": 1, "profile": RECIPE_PROFILE, "interpretation": RECIPE_INTERPRETATION,
        **{k: identity[k] for k in (*_PROVENANCE_KEYS, "candidate_canonical_sha256", "source_sha256",
            "scene_sha256", "task_sha256", "gates_sha256", "runtime_sha256", "feedback_gains", "clock_binding")},
        "action_contract_sha256": canonical_hash(action), "original_human_canonical_sha256": canonical_hash(original),
        "refinement_marker": candidate["whole_body_reference_refinement"],
        "targets": [{"evidence_id": record["evidence_id"], "time_s": row["time_s"], "joints": row["joints"]}
                    for record, row in zip(records, new_refs, strict=True)]}
    recipe_json = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    _require(len(recipe_json.encode()) <= MAX_RECIPE_BYTES, "controller recipe exceeds 1 MiB")
    identity["recipe_sha256"] = canonical_hash(recipe)
    action_items = tuple((k, tuple(v) if isinstance(v, list) else v) for k, v in action.items())
    return ReferenceController(tuple(times), _tuple_rows(old_values), _tuple_rows(new_values), _tuple_rows(old_slopes),
        _tuple_rows(new_slopes), tuple(action["physical_limits"]), action_items, json.dumps(identity), recipe_json,
        identity["scene_sha256"], identity["task_sha256"], identity["gates_sha256"],
        horizon * dt, dt / substeps, reference_velocity_index, bool(np.array_equal(old_array, new_array)))
