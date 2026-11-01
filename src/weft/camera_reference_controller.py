"""Camera angle overlays as bounded control alternatives, never replacement bodies.

The caller authenticates source seals and current runtime bytes, admits the actual
scene/task, and supplies that runtime's trusted reference_velocity_index. This
pure boundary cannot authenticate hash declarations or an arbitrary callback.
"""

from copy import deepcopy
from hashlib import sha256
import json
import math
import re
import time
from types import MappingProxyType

import numpy as np

from weft.camera_shared_contract import _bounded_json, validate_camera_coupling
from weft.camera_shared_receipt import IDENTITIES
from weft.io import canonical_hash
from weft import shared_exercise_reference_controller as shared
from weft.spatial_pose import SPATIAL_JOINT_LIMITS, SPATIAL_JOINT_NAMES


PROFILE = "camera-source-fit-controller-v1"
RECIPE_PROFILE = "camera-source-fit-controller-recipe-v1"
RECIPE_INTERPRETATION = "archived_execution_only_not_source_fit_admission"
METHOD = "blocked-camera-body-surface-interval-reference-v1"
OVERLAY_PROFILE = "camera-body-angle-proposal-v1"
MAX_RECIPE_BYTES, MAX_SECONDS = 1 << 20, 5.
PROPOSAL_GATES = shared.PROPOSAL_GATES | frozenset({
    "surface_interval_nonregression", "camera_projection_domain", "camera_endpoint_depth_change"})
GATES_SHA256 = "063a93773406db92b583daf947855098727e34009bf645e82e0a5a2e044e0002"
# These reviewed bytes implement the reused 40/5 kernel and unchanged camera scorer.
RUNTIME_SHA256 = MappingProxyType({
    "human.py": "dab7a88a688e3a1c243d0ed64b3abf49dfe5704dac8d9d3101926fd5937cb340",
    "shared_world.py": "19c0f07fc9174bd23d086250bb9d5cc3e2c6ded7c1e84612e120d82772d309df",
    "shared_exercise_reference_controller.py": "6f656699c08910aecad9bc0f6a03a5106da7c49c250e222032cc57acc059dd90",
    "shared_exercise_task.py": "c71a53dec0f10a13fce7b868d810eb66dc56e2e51783cd892c130fcd864c24eb",
    "camera_shared_task.py": "fe877d6e66d4c48bd4aee7042ff95d2ad9f1b27c860cd6ef1179b804bd4bfd99",
})
REQUIRED_RUNTIME_FILES = frozenset(RUNTIME_SHA256) | frozenset({
    "camera_reference_controller.py", "camera_shared_contract.py", "camera_shared_receipt.py",
    "camera_body.py", "camera_pose.py", "camera_reference.py", "camera_ground.py", "multiview_root.py",
    "multiview_body.py", "multiview_pose.py", "multiview_segments.py", "body_frame.py", "root_motion.py",
    "spatial_pose.py", "io.py", "errors.py"})
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROOTS = ("root_x", "root_y", "root_z")
_STATUS = "unadmitted_construction_proposal"
_INSTALLATION = "unsupported_angle_overlay_not_a_human_spec"
_SCOPE = ("Source-fit authored joint corrections on the unchanged camera simulator task; not a learned policy, "
          "installed body, recovered balance, verified contact or physical transfer.")
_TASK_SCOPE = {
    "success_scope": "authored_simulated_camera_tracking_criteria", "physical_validation": "unverified",
    "source_contact_verified": False, "ground_support_verified": False, "equipment_scoring": "unsupported",
    "foot_controls": "unsupported", "learning": "unqualified",
    "body_scale": "supplied_camera_root_scale_unchanged_monocular_shape", "body_mass": "unchanged_authored_model",
    "clock_alignment": "original_source_grid_with_unchanged_body_period"}
_LIMITS = {"maximum_references": 900, "maximum_angle_correction_rad": .15,
    "maximum_correction_rate_rad_s": 1.5, "maximum_temporal_gap_s": .25,
    "maximum_endpoint_depth_change_model_units": .08, "floor_penetration_limit_model_units": .025,
    "numerical_tolerance": 1e-8}
_PINS = frozenset({"body_raw_sha256", "evidence_raw_sha256", "camera_raw_sha256", "body_canonical_sha256",
    "camera_canonical_sha256", "pose_evidence", "camera_binding_canonical_sha256", "coupling_canonical_sha256",
    "root_reference_canonical_sha256", "ground_canonical_sha256", "original_angular_reference_canonical_sha256"})
_OVERLAY_KEYS = frozenset({"schema_version", "profile", "input_bindings", "reference_candidate",
    "whole_body_reference_refinement", "installation", "body_installed", "raw_camera_changed",
    "root_or_floor_changed", "promoted", "contact", "physical_validation"})
_RECIPE_KEYS = frozenset({"schema_version", "profile", "interpretation", "scene_sha256", "task_sha256",
    "gates_sha256", "action_contract_sha256", *IDENTITIES, "input_human_sha256", "input_evidence_sha256",
    "angle_proposal_sha256", "angle_proposal_canonical_sha256", "proposal_report_sha256",
    "runtime_sha256", "feedback_gains", "clock_binding", "proposal_gates", "angle_proposal"})


def _require(value, message):
    if not value:
        raise ValueError("Camera reference controller: " + message)


def _check(deadline):
    _require(time.monotonic() < deadline, "cooperative preparation deadline exceeded")


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _hash(value):
    return type(value) is str and _SHA.fullmatch(value) is not None


def _same(a, b):
    return canonical_hash(a) == canonical_hash(b)


def _copy(value, limit, deadline):
    _require(type(value) is dict, "plain object required")
    _bounded_json(value, limit, deadline, "Camera controller input")
    return deepcopy(value)


def camera_reference_action_contract(scene, task):
    """Derive public controls from admitted camera metadata; this is not scene admission."""
    _require(type(scene) is dict and scene.get("profile") == "shared-world-v6"
             and type(task) is dict and task.get("profile") == "camera-shared-task-v1",
             "camera scene and task profiles required")
    return shared.reference_action_contract(scene, task)


def _context(original, scene, task, action, runtime, selector, deadline):
    _require(callable(selector) and REQUIRED_RUNTIME_FILES <= runtime.keys() and len(runtime) <= 128
             and all(type(k) is str and re.fullmatch(r"[A-Za-z0-9_]+\.py", k) and _hash(v) for k, v in runtime.items())
             and all(runtime.get(k) == v for k, v in RUNTIME_SHA256.items()), "unreviewed feedback/scoring runtime")
    _require(scene.get("profile") == "shared-world-v6" and scene.get("controlled_id") == "human"
             and scene.get("physical_validation") == "unverified" and scene.get("objects") == []
             and scene.get("target") is None and scene.get("learned_policy_available") is False
             and "foot_extension" not in scene and _same(scene.get("human_spec"), original)
             and "whole_body_reference_refinement" not in original, "unchanged original camera scene/body required")
    coupling = validate_camera_coupling(original, scene["camera_coupling"], dt=scene["dt"],
        coordinate_frame=scene["coordinate_frame"], source_binding=scene["source_binding"])
    clock = coupling["clock"]
    interval = {"start_offset_s": 0., "horizon_steps": clock["horizon_steps"]}
    _require(type(scene["substeps"]) is int and 8 <= scene["substeps"] <= 64
             and scene["substeps"] == scene["coordinate_frame"]["clock_binding"]["physics_substeps"]
             and scene["dt"] / scene["substeps"] <= 1 / 240 + 1e-12
             and scene["task"].get("object_goal") is None
             and _same({k: scene["task"][k] for k in interval}, interval), "unchanged full source interval required")
    _require(set(task) == {"schema_version", "profile", "scene_sha256", *IDENTITIES, "metric_context",
                          "gate_profile", "gates_sha256", "interval", "control", "scope"}
             and type(task["schema_version"]) is int and task["schema_version"] == 1
             and task["profile"] == "camera-shared-task-v1"
             and task["scene_sha256"] == canonical_hash(scene)
             and task["gate_profile"] == "shared-exercise-gates-v1" and task["gates_sha256"] == GATES_SHA256
             and _same(task["interval"], interval) and _same(task["scope"], _TASK_SCOPE), "camera task identity/gates differ")
    control = task["control"]
    _require(control in ({"root_assistance": "disabled", "object_forces": "forbidden"},
                         {"root_assistance": "explicit_3_axis", "object_forces": "forbidden"}), "explicit human-only controls required")
    expected_pins = {"camera_coupling_sha256": canonical_hash(coupling), "source_sha256": original["source_sha256"],
        "body_sha256": canonical_hash(original), "source_clock_sha256": coupling["source_clock_sha256"],
        "kernel_evaluator_sha256": runtime["shared_exercise_task.py"],
        "adapter_implementation_sha256": runtime["camera_shared_task.py"],
        "evaluator_sha256": canonical_hash({k: runtime[k] for k in ("camera_shared_task.py", "shared_exercise_task.py")})}
    _require(all(task[k] == v for k, v in expected_pins.items()), "camera task source/runtime pins differ")
    arguments = {"profile": "shared-exercise-task-v2", "interval": interval, "control": control,
                 "equipment_ids": [], "contact_requirements": [], "source_scope": task["scope"]}
    context = {"schema_version": 1, "profile": "camera-shared-metric-context-v1", "scene_sha256": task["scene_sha256"],
        "kernel": "weft.shared_exercise_task._ExerciseMetrics", "kernel_branch": "shared-exercise-task-v2",
        "valid_legacy_task": False, "projection": "identity_22_motor_samples_without_equipment",
        "control_dt_s": scene["dt"], "source_body_dt_s": original["dt"], "substeps": scene["substeps"],
        "kernel_arguments": arguments, "kernel_arguments_sha256": canonical_hash(arguments)}
    _require(_same(task["metric_context"], context) and task["metric_context_sha256"] == canonical_hash(context),
             "actual camera scoring context differs")
    source_action = scene["action_contract"]
    _require(source_action["names"] == [*SPATIAL_JOINT_NAMES, *_ROOTS]
             and type(source_action["human_motor_count"]) is int and source_action["human_motor_count"] == 22
             and type(source_action["object_force_channels"]) is int and source_action["object_force_channels"] == 0
             and source_action["dynamic_object_ids"] == []
             and _same(source_action["normalized_bounds"], [-1., 1.])
             and source_action["substeps_per_action"] == scene["substeps"] and source_action["hidden_assistance"] is False,
             "engine must retain exactly nineteen joints and three roots")
    count = 22 if control["root_assistance"] == "explicit_3_axis" else 19
    _require(_same(action, camera_reference_action_contract(scene, task)) and len(action["names"]) == count
             and len(source_action["units"]) == len(source_action["physical_limits"]) == 22
             and all(type(v) is str and 0 < len(v) <= 160 for v in source_action["units"])
             and all(_number(v, 1e-9, 1e9) for v in source_action["physical_limits"])
             and all(action["physical_limits"][i] == original.get("motor_limits", {}).get(name, 30.)
                     for i, name in enumerate(SPATIAL_JOINT_NAMES)), "public action order/normalization changed")
    _check(deadline)
    return {"control_dt_s": scene["dt"], "physics_substep_dt_s": scene["dt"] / scene["substeps"],
        "source_body_dt_s": original["dt"], "source_time_offset_s": clock["source_time_offset_s"],
        "source_fps": clock["source_fps"], "source_clock_sha256": coupling["source_clock_sha256"],
        "horizon_steps": clock["horizon_steps"]}


def _overlay(original, overlay, scene, original_raw_pin, deadline):
    _require(set(overlay) == _OVERLAY_KEYS and type(overlay["schema_version"]) is int and overlay["schema_version"] == 1
             and overlay["profile"] == OVERLAY_PROFILE and overlay["installation"] == _INSTALLATION
             and all(overlay[k] is False for k in ("body_installed", "raw_camera_changed", "root_or_floor_changed", "promoted"))
             and overlay["contact"] == "unknown" and overlay["physical_validation"] == "unverified", "uninstalled angle overlay required")
    pins, binding = overlay["input_bindings"], original["camera_reference"]
    _require(type(pins) is dict and set(pins) == _PINS
             and all(_hash(v) for k, v in pins.items() if k != "pose_evidence"), "exact original input pins required")
    fixed = {"body_raw_sha256": original_raw_pin, "body_canonical_sha256": canonical_hash(original),
        "camera_canonical_sha256": binding["calibration_sha256"], "camera_binding_canonical_sha256": canonical_hash(binding),
        "coupling_canonical_sha256": canonical_hash(scene["camera_coupling"]),
        "root_reference_canonical_sha256": canonical_hash(original["root_reference"]),
        "ground_canonical_sha256": canonical_hash(binding["ground"]),
        "original_angular_reference_canonical_sha256": canonical_hash(original["reference"])}
    _require(all(pins[k] == v for k, v in fixed.items()), "overlay changed original body/camera/root/ground/angles")
    pose = pins["pose_evidence"]
    _require(type(pose) is dict and _hash(pose.get("canonical_sha256")) and (
        set(pose) == {"canonical_sha256", "scope"} and pose["scope"] == "complete_supplied_pose_document"
        and pose["canonical_sha256"] == binding["pose_evidence_sha256"] or
        set(pose) == {"canonical_sha256", "original_canonical_sha256", "scope"}
        and pose["scope"] == "original_pose_plus_separately_checked_dimension_provenance"
        and pose["original_canonical_sha256"] == binding["pose_evidence_sha256"]), "original pose evidence binding changed")
    marker = overlay["whole_body_reference_refinement"]
    _require(type(marker) is dict and set(marker) == {"schema_version", "method", "status", "promoted",
        "input_human_sha256", "input_evidence_sha256", "surface_intervals"}
        and _same({k: v for k, v in marker.items() if k != "surface_intervals"}, {
            "schema_version": 1, "method": METHOD, "status": _STATUS, "promoted": False,
            "input_human_sha256": original_raw_pin, "input_evidence_sha256": pins["evidence_raw_sha256"]}), "overlay refinement provenance changed")
    refs, evidence, roots = (original[k] for k in ("reference", "reference_evidence", "root_reference"))
    targets = overlay["reference_candidate"]
    _require(all(type(rows) is list for rows in (refs, targets, evidence, roots))
             and 3 <= len(refs) == len(targets) == len(evidence) == len(roots) <= 900, "complete original reference count required")
    times, old, new = [], [], []
    for index, (row, target, record, root) in enumerate(zip(refs, targets, evidence, roots, strict=True)):
        if index % 64 == 0:
            _check(deadline)
        stamp = row["time_s"]
        _require(type(target) is dict and set(target) == set(row) == {"time_s", "joints"}
                 and _number(stamp, 0., 60.) and _same(target["time_s"], stamp)
                 and record["time_s"] == root["time_s"] == stamp
                 and (not times or times[-1] < stamp <= times[-1] + .25 + 1e-9), "original reference clock/count changed")
        for value, output in ((row, old), (target, new)):
            joints = value["joints"]
            _require(type(joints) is dict and set(joints) == set(SPATIAL_JOINT_NAMES)
                     and all(_number(joints[name], *SPATIAL_JOINT_LIMITS[name]) for name in SPATIAL_JOINT_NAMES), "finite original joint bounds required")
            output.append([joints[name] for name in SPATIAL_JOINT_NAMES])
        times.append(float(stamp))
    _require(_same(refs[0], targets[0]), "first reference must remain exactly frozen")
    old, new = np.asarray(old), np.asarray(new)
    spans = np.diff(times)[:, None]
    correction, rate = new - old, np.diff(new - old, axis=0) / spans
    slopes = (np.diff(old, axis=0) / spans, np.diff(new, axis=0) / spans)
    _require(np.max(abs(correction)) <= .15 + shared.TOLERANCE and np.max(abs(rate)) <= 1.5 + shared.TOLERANCE
             and all(np.isfinite(s).all() and np.max(abs(s)) <= 1e4 for s in slopes), "actual angle/rate bounds exceeded")
    duration = scene["task"]["horizon_steps"] * scene["dt"]
    _require(0 < duration <= 60. and math.isclose(duration, times[-1] - times[0], rel_tol=0., abs_tol=1e-9),
             "full source horizon changed")
    return tuple(times), old, new, slopes


def _report(report, overlay, original, scene, deadline):
    pins = overlay["input_bindings"]
    _require(type(report.get("schema_version")) is int and report["schema_version"] == 1
             and report.get("profile") == METHOD and report.get("status") == _STATUS
             and report.get("candidate_kind") == OVERLAY_PROFILE and report.get("geometry_admission") == _INSTALLATION
             and report.get("promoted") is False and report.get("physical_validation") == "unverified"
             and report.get("failures") == [] and report.get("budget_reason") is None,
             "complete passing unadmitted camera proposal required")
    _require(report.get("input_human_sha256") == pins["body_raw_sha256"]
             and report.get("input_evidence_sha256") == pins["evidence_raw_sha256"]
             and report.get("source_sha256") == original["source_sha256"]
             and report.get("candidate_sha256") == canonical_hash(overlay), "proposal report identity differs")
    _require(type(report.get("gates")) is dict and set(report["gates"]) == PROPOSAL_GATES
             and all(value is True for value in report["gates"].values()), "all fourteen unchanged proposal gates must pass")
    _require(report["contract"].get("method") == METHOD and all(
        _same(report["contract"].get(k), v) for k, v in _LIMITS.items()), "proposal numerical limits changed")
    surface, marker = report["surface_intervals"], overlay["whole_body_reference_refinement"]["surface_intervals"]
    inputs = surface["inputs"]
    expected_marker = {"profile": "whole-body-surface-interval-fit-v1", "input_bindings": {
        "objective_profile": inputs["profile"], **{k: inputs[k] for k in (
            "source_sha256", "body_sha256", "binding_sha256", "envelope_cache_sha256", "envelope_bindings")}},
        **{k: surface[k] for k in ("weight", "weight_status", "work_limits", "summary")}}
    _require(_same(marker, expected_marker) and inputs["source_sha256"] == original["source_sha256"]
             and inputs["body_sha256"] == pins["body_raw_sha256"] and _number(surface["weight"], 0., 1000.) and surface["weight"] > 0
             and surface["weight_status"] == "authored" and surface.get("contact") == "unknown"
             and surface.get("physical_validation") == "unverified"
             and surface.get("weight_units") == "authored_camera_loss_units_squared_per_supplied_world_metre_squared",
             "surface objective provenance differs")
    fit = report["camera_fit"]
    _require(fit.get("profile") == "fixed-camera-body-fit-inputs-v1" and _same(fit.get("input_bindings"), pins)
             and _same(fit.get("coupling"), scene["camera_coupling"])
             and report.get("root_audit_sha256") == canonical_hash({k: v for k, v in fit.items() if k not in {
                 "projection_calls", "maximum_projection_calls", "grid_camera_projection_calls"}})
             and _number(fit.get("image_loss_scale_per_pixel"), 1e-6, 1.)
             and fit.get("image_loss_scale_status") == "authored" and fit.get("image_loss_units") == "authored_loss_units_not_metres"
             and report["objective"].get("profile") == "camera_surface_interval_v1", "camera objective/coupling differs")
    rows, refs = report["references"], original["reference"]
    _require(type(rows) is list and len(rows) == len(refs), "report dropped original reference rows")
    source_frames = []
    for index, (row, old, target, meta) in enumerate(zip(rows, refs, overlay["reference_candidate"], original["reference_evidence"], strict=True)):
        frame = row["frame"]
        _require(type(frame) is int and 0 <= frame < scene["camera_coupling"]["clock"]["total_source_frames"]
                 and (not source_frames or frame > source_frames[-1])
                 and row["evidence_id"] == meta["evidence_id"] == f"{original['source_sha256']}:frame:{frame}"
                 and row["time_s"] == meta["time_s"] == old["time_s"]
                 and abs(old["time_s"] - frame / fit["coupling"]["clock"]["source_fps"]) <= 1e-10
                 and type(row["correction_rad"]) is dict and set(row["correction_rad"]) == set(SPATIAL_JOINT_NAMES)
                 and all(_number(row["correction_rad"][name], -.15 - shared.TOLERANCE, .15 + shared.TOLERANCE)
                         and math.isclose(row["correction_rad"][name], target["joints"][name] - old["joints"][name],
                                          rel_tol=0., abs_tol=1e-15) for name in SPATIAL_JOINT_NAMES), "report reference/correction join differs")
        source_frames.append(frame)
    counts, slots = report["source_counts"], fit["source_slots"]
    declared, decoded = counts["declared_frame_count"], counts["declared_decoded_frames"]
    _require(type(declared) is int and declared == scene["camera_coupling"]["clock"]["total_source_frames"]
             and (decoded is None or type(decoded) is int and 0 <= decoded <= 1800)
             and type(slots) is list and len(slots) == max(declared, decoded or 0) <= 1800
             and fit["complete_source_frame_slots"] == len(slots) and counts["body_reference_knots"] == len(refs),
             "complete original source slot denominator changed")
    retained = report["retained_pose_slots"]
    _require(type(retained) is list and len(retained) == counts["retained_pose_slots"] <= 900,
             "complete retained pose inventory required")
    retained_by_frame = {}
    for row in retained:
        frame = row["frame"]
        _require(type(frame) is int and 0 <= frame < len(slots) and frame not in retained_by_frame
                 and row["id"] == f"{original['source_sha256']}:frame:{frame}"
                 and _number(row["time_s"], 0., 60.)
                 and (row.get("decoder_timestamp_s") is None or _number(row["decoder_timestamp_s"], 0., 60.)),
                 "retained original pose identity/clock differs")
        retained_by_frame[frame] = row
    by_frame = dict(zip(source_frames, range(len(refs)), strict=True))
    for frame, slot in enumerate(slots):
        if frame % 64 == 0:
            _check(deadline)
        index = by_frame.get(frame)
        retained_row = retained_by_frame.get(frame)
        _require(type(slot) is dict and type(slot.get("source_frame")) is int and slot["source_frame"] == frame
                 and (slot.get("reference_index") is None if index is None else
                      type(slot.get("reference_index")) is int and slot["reference_index"] == index)
                 and (index is None or slot.get("evidence_id") == rows[index]["evidence_id"]
                      and slot.get("source_time_s") == slot.get("camera_time_s") == refs[index]["time_s"]),
                 "report source slot/reference clock differs")
        _require((frame < declared and slot.get("camera_status") == "supplied"
                  and _number(slot.get("camera_time_s"), 0., 60.)
                  and abs(slot["camera_time_s"] - frame / fit["coupling"]["clock"]["source_fps"]) <= 1e-10
                  or frame >= declared and slot.get("camera_status") == "unavailable_original_decoded_slot"
                  and slot.get("camera_time_s") is None)
                 and slot.get("evidence_id") == (retained_row["id"] if retained_row else None)
                 and slot.get("source_time_s") == (retained_row["time_s"] if retained_row else None)
                 and (retained_row is None or _number(slot.get("source_time_s"), 0., 60.))
                 and slot.get("decoder_timestamp_s") == (retained_row.get("decoder_timestamp_s") if retained_row else None)
                 and (slot.get("decoder_timestamp_s") is None or _number(slot["decoder_timestamp_s"], 0., 60.))
                 and slot.get("status") == (retained_row.get("status") if retained_row else "unsampled"),
                 "complete source clocks/statuses were changed")


def _provider(original, overlay, scene, task, action, runtime, selector, clock, arrays, provenance, qualification):
    times, old, new, slopes = arrays
    recipe = {"schema_version": 1, "profile": RECIPE_PROFILE, "interpretation": RECIPE_INTERPRETATION,
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "gates_sha256": task["gates_sha256"],
        "action_contract_sha256": canonical_hash(action), **{k: task[k] for k in IDENTITIES}, **provenance,
        "runtime_sha256": runtime, "feedback_gains": {"joint_position": 40., "joint_velocity": 5.},
        "clock_binding": clock, "proposal_gates": {k: True for k in sorted(PROPOSAL_GATES)}, "angle_proposal": overlay}
    recipe_json = json.dumps(recipe, sort_keys=True, separators=(",", ":"), allow_nan=False)
    _require(len(recipe_json.encode()) <= MAX_RECIPE_BYTES, "archived recipe exceeds one MiB")
    identity = {k: v for k, v in recipe.items() if k not in {"schema_version", "profile", "interpretation", "angle_proposal"}}
    identity.update(profile=PROFILE, controller_kind="camera_source_fit_authored_alternative", learned=False,
        qualification=qualification, reference_count=len(times), recipe_sha256=canonical_hash(recipe),
        controller_implementation_sha256=runtime["camera_reference_controller.py"], scope=_SCOPE,
        body_installed=False, physical_validation="unverified", contact="unknown")
    items = tuple((key, tuple(value) if isinstance(value, list) else value) for key, value in action.items())
    # Reuse the numerical methods, not the legacy factories or their admission allowlists.
    return shared.ReferenceController(times, shared._tuple_rows(old), shared._tuple_rows(new),
        shared._tuple_rows(slopes[0]), shared._tuple_rows(slopes[1]), tuple(action["physical_limits"]), items,
        json.dumps(identity, sort_keys=True, allow_nan=False), recipe_json, recipe["scene_sha256"], recipe["task_sha256"],
        task["gates_sha256"], clock["horizon_steps"] * clock["control_dt_s"], clock["physics_substep_dt_s"],
        selector, bool(np.array_equal(old, new)))


def prepare_camera_reference_controller(original_human_bytes, angle_proposal_bytes, report_bytes, *,
        scene, task, action_contract, runtime_sha256, reference_velocity_index):
    """Prepare exact saved passing overlay controls; no physics or source admission occurs."""
    try:
        deadline = time.monotonic() + MAX_SECONDS
        original, overlay, report = (shared._json(raw, limit) for raw, limit in (
            (original_human_bytes, 8 << 20), (angle_proposal_bytes, 8 << 20), (report_bytes, 16 << 20)))
        scene, task, action, runtime = (_copy(value, limit, deadline) for value, limit in (
            (scene, 8 << 20), (task, 64 << 10), (action_contract, 16 << 10), (runtime_sha256, 64 << 10)))
        clock = _context(original, scene, task, action, runtime, reference_velocity_index, deadline)
        raw_pin = sha256(original_human_bytes).hexdigest()
        arrays = _overlay(original, overlay, scene, raw_pin, deadline)
        _report(report, overlay, original, scene, deadline)
        provenance = {"input_human_sha256": raw_pin, "input_evidence_sha256": report["input_evidence_sha256"],
            "angle_proposal_sha256": sha256(angle_proposal_bytes).hexdigest(),
            "angle_proposal_canonical_sha256": canonical_hash(overlay), "proposal_report_sha256": sha256(report_bytes).hexdigest()}
        result = _provider(original, overlay, scene, task, action, runtime, reference_velocity_index, clock, arrays,
                           provenance, "saved_passed_camera_overlay_boundary_checked")
        _check(deadline)
        return result
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError("Camera reference controller: malformed input contract") from error


def prepare_camera_reference_recipe(recipe, *, scene, task, action_contract, runtime_sha256, reference_velocity_index):
    """Reconstruct externally authenticated execution data, not source-fit qualification."""
    try:
        deadline = time.monotonic() + MAX_SECONDS
        recipe = _copy(recipe, MAX_RECIPE_BYTES, deadline)
        _require(set(recipe) == _RECIPE_KEYS and type(recipe["schema_version"]) is int and recipe["schema_version"] == 1
                 and recipe["profile"] == RECIPE_PROFILE and recipe["interpretation"] == RECIPE_INTERPRETATION
                 and all(_hash(recipe[k]) for k in recipe if k.endswith("_sha256") and k != "runtime_sha256"),
                 "unsupported archived recipe fields or identity")
        scene, task, action, runtime = (_copy(value, limit, deadline) for value, limit in (
            (scene, 8 << 20), (task, 64 << 10), (action_contract, 16 << 10), (runtime_sha256, 64 << 10)))
        original, overlay = scene["human_spec"], recipe["angle_proposal"]
        clock = _context(original, scene, task, action, runtime, reference_velocity_index, deadline)
        arrays = _overlay(original, overlay, scene, recipe["input_human_sha256"], deadline)
        provenance = {k: recipe[k] for k in ("input_human_sha256", "input_evidence_sha256", "angle_proposal_sha256",
                                            "angle_proposal_canonical_sha256", "proposal_report_sha256")}
        _require(recipe["angle_proposal_canonical_sha256"] == canonical_hash(overlay)
                 and recipe["input_evidence_sha256"] == overlay["input_bindings"]["evidence_raw_sha256"], "archived overlay identity differs")
        result = _provider(original, overlay, scene, task, action, runtime, reference_velocity_index, clock, arrays,
                           provenance, RECIPE_INTERPRETATION)
        _require(_same(result.recipe(), recipe), "archived execution changed on reconstruction")
        _check(deadline)
        return result
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError("Camera reference controller: malformed archived recipe") from error
