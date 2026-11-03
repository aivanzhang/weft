"""Compare a fixed authored goal with source-reference geometry, not an executed trajectory."""

import json
import math
import time
from types import MappingProxyType

import numpy as np

from weft import moving_equipment_contact_search as contact
from weft import moving_equipment_support as support
from weft.io import canonical_hash
from weft.native_interval import elapsed_substeps

PROFILE = "moving-equipment-body-sweep-v1"
LIMITS = MappingProxyType({"seconds": 30, "boundaries": 28801, "geometries": 64,
    "distance_pairs": 250000, "intervals": 4096, "output_bytes": 32 << 20,
    "overlap_tolerance_world": 1e-8})
SCOPE = ("Fixed authored goal sphere against the unchanged interpolated source-body reference at every native "
    "boundary. Not an executed trajectory, continuous-time clearance proof, recovered contact, task impossibility "
    "or permission to change source, goal, forces or task gates. Manual controls remain available.")


def _require(value, message):
    if not value:
        raise ValueError("Body sweep: " + message)


def _bytes(value, maximum=LIMITS["output_bytes"]):
    raw = bytearray()
    for chunk in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        block = chunk.encode()
        _require(len(raw) + len(block) <= maximum, "output byte limit exceeded; no rows omitted")
        raw.extend(block)
    return bytes(raw)


def _schedule(scene, task):
    count = task["interval"]["horizon_steps"] * scene["substeps"]
    _require(1 <= count < LIMITS["boundaries"], "full native interval exceeds supported boundary count")
    times = [float(elapsed_substeps(scene, i)) for i in range(count + 1)]
    _require(all(math.isfinite(t) for t in times) and times[0] == 0 and
             all(a < b for a, b in zip(times, times[1:])) and times[-1] <= 60,
             "native interval must be finite, complete and strictly ordered")
    return times


def _separation(model, data, human_gid, object_gid):
    import mujoco

    points = np.zeros(6)
    distance = float(mujoco.mj_geomDistance(model, data, human_gid, object_gid, 100., points))
    _require(math.isfinite(distance) and distance < 100 and np.isfinite(points).all(),
             "native distance unavailable or outside supported world envelope")
    normal = None
    if abs(distance) > 1e-12:
        direction = (points[3:] - points[:3]) / distance
        length = float(np.linalg.norm(direction))
        if abs(length - 1.) <= 1e-6:
            normal = (direction / length).tolist()
    return distance, {"signed_distance_world": distance, "human_surface_world": points[:3].tolist(),
        "goal_surface_world": points[3:].tolist(), "normal_human_to_goal_world": normal,
        "normal_status": "native_signed_surface_separation" if normal is not None else "degenerate_or_unavailable"}


def _intervals(rows, names, complete):
    result = []
    for index, name in enumerate(names):
        active = None
        for i, row in enumerate(rows):
            distance = row["distances_world"][index]
            if distance < -LIMITS["overlap_tolerance_world"]:
                if active is None:
                    active = {"geometry": name, "first_sample_s": row["time_s"], "last_sample_s": row["time_s"],
                        "previous_nonoverlap_sample_s": rows[i-1]["time_s"] if i else None,
                        "next_nonoverlap_sample_s": None, "samples": 0, "minimum_distance_world": distance}
                active["samples"] += 1
                active["last_sample_s"] = row["time_s"]
                active["minimum_distance_world"] = min(active["minimum_distance_world"], distance)
            elif active is not None:
                active["next_nonoverlap_sample_s"] = row["time_s"]
                result.append(active); active = None
                _require(len(result) <= LIMITS["intervals"], "suspect interval bound exceeded")
        if active is not None:
            active["ends_at_complete_reference_endpoint"] = complete
            result.append(active)
        _require(len(result) <= LIMITS["intervals"], "suspect interval bound exceeded")
    return result


def screen_moving_body_sweep(admission, *, max_seconds=30, max_evaluations=30000):
    """One source admission capability, one scratch model, zero physics steps or controls."""
    from weft.moving_equipment_task import validate_moving_equipment_task
    import mujoco

    started = time.monotonic()
    _require(type(max_seconds) in (int, float) and 0 < max_seconds <= 30 and math.isfinite(max_seconds),
             "deadline must be within(0,30]")
    _require(type(max_evaluations) is int and 1 <= max_evaluations <= 30000, "forward bound must be1..30000")
    admission = support._admitted(admission)
    scene, task = admission.scene, admission.task
    validate_moving_equipment_task(task, scene, admission.bundle)
    times = _schedule(scene, task)
    budget = support._Budget(started + max_seconds, max_evaluations)
    runtime = contact._runtime(admission, scene, budget)
    rows, names, geometry, failure = [], [], None, None
    model_pin = None
    output_size = 0
    try:
        geometry = support._Geometry(scene, task, budget)
        model, data = geometry.model, geometry.data
        model_pin = canonical_hash({"xml": geometry.env._model_xml} if hasattr(geometry.env, "_model_xml") else
            {key: getattr(model, key).tolist() for key in ("geom_size", "geom_type", "geom_contype", "geom_conaffinity",
                "body_mass", "body_pos", "jnt_range", "actuator_ctrlrange", "geom_friction")})
        ids = sorted(geometry.human_ids, key=lambda gid: geometry.env._human_geom_to_name[gid])
        names = [geometry.env._human_geom_to_name[gid] for gid in ids]
        _require(1 <= len(ids) <= LIMITS["geometries"] and len(ids) * len(times) <= LIMITS["distance_pairs"],
                 "complete geometry/time inventory exceeds distance-pair bound; no subsampling")
        _require(model.npair == 0 and model.nexclude == 0 and
                 not int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CONTACT),
                 "explicit exclusions/pairs or disabled contacts require a separate adapter")
        for gid in ids:
            _require(int(model.geom_type[gid]) in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_SPHERE)),
                     "exact native sphere/capsule body geometry required")
            _require((int(model.geom_contype[gid]) & int(model.geom_conaffinity[geometry.ogid])) or
                     (int(model.geom_contype[geometry.ogid]) & int(model.geom_conaffinity[gid])),
                     "body/object geometry is collision-filtered; cannot imply an active collision")
        for stamp in times:
            geometry.pose(stamp, np.array([], dtype=int), np.array([]))
            distances, details = [], []
            for index, gid in enumerate(ids):
                budget.check()
                distance, detail = _separation(model, data, gid, geometry.ogid)
                distances.append(distance)
                if distance < -LIMITS["overlap_tolerance_world"]:
                    details.append({"geometry_index": index, **detail})
            row = {"time_s": stamp, "distances_world": distances, "overlaps": details}
            output_size += len(_bytes(row, 128 << 10))
            _require(output_size <= LIMITS["output_bytes"] - (4 << 20), "complete rows exceed output reservation")
            rows.append(row)
    except (TimeoutError, ValueError) as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    finally:
        if geometry is not None:
            try:
                if model_pin is not None:
                    after = canonical_hash({"xml": geometry.env._model_xml} if hasattr(geometry.env, "_model_xml") else
                        {key: getattr(geometry.model, key).tolist() for key in ("geom_size", "geom_type", "geom_contype", "geom_conaffinity",
                            "body_mass", "body_pos", "jnt_range", "actuator_ctrlrange", "geom_friction")})
                    _require(after == model_pin, "compiled geometry/model changed during sweep")
            finally:
                geometry.close()
    verified = False
    try:
        _require(contact._runtime(admission, scene, budget) == runtime, "runtime changed during body sweep")
        verified = True
    except TimeoutError as error:
        failure = {"type": type(error).__name__, "message": str(error)}
    complete = failure is None and len(rows) == len(times)
    intervals = _intervals(rows, names, complete)
    result = {"schema_version": 1, "profile": PROFILE,
        "status": "complete_advisory" if complete else "incomplete_or_unsupported", "failure": failure,
        "input_pins": {"task_manifest_sha256": admission.manifest_sha256, "parent_manifest_sha256": admission.parent_manifest_sha256,
            "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "bundle_sha256": canonical_hash(admission.bundle)},
        "runtime": runtime, "runtime_readback_complete": verified, "model_sha256": model_pin,
        "target_object_id": task["goal"]["object_id"], "fixed_goal_position_world": task["goal"]["position_world"],
        "geometry_names": names, "schedule": {"native_boundary_times_s": times, "expected_samples": len(times),
            "completed_samples": len(rows), "subsampling": False, "source_time_offset_s": scene["task"]["start_offset_s"]},
        "rows": rows, "suspect_intervals": intervals,
        "summary": {"overlapping_samples": sum(bool(row["overlaps"]) for row in rows),
            "minimum_distance_world": min((min(row["distances_world"]) for row in rows), default=None),
            "first_overlap_sample_s": next((row["time_s"] for row in rows if row["overlaps"]), None),
            "interval_count": len(intervals)},
        "sampling_scope": "Sampled native boundaries only; entering/leaving brackets are not exact continuous collision times.",
        "source_uncertainty": "Original monocular pose, missing/low-confidence joints, camera/depth/shape assumptions remain unchanged. No physical contact or occupied-space truth inferred.",
        "limits": dict(LIMITS), "requested_bounds": {"seconds": max_seconds, "evaluations": max_evaluations},
        "physics_steps": 0, "model_compilations": int(geometry is not None), "native_forward_evaluations": budget.used,
        "manual_controls_blocked": False, "task_impossible": False, "scope": SCOPE, "seconds": time.monotonic() - started}
    _bytes(result)
    try:
        budget.check()
    except TimeoutError as error:
        result.update(status="incomplete_or_unsupported", failure={"type": type(error).__name__, "message": str(error)})
    result["seconds"] = time.monotonic() - started
    return result
