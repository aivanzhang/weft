"""Bounded camera/root/leg hypotheses; source evidence and protected tasks remain unchanged."""

from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.body_camera_root_refinement import _rotation, project_with_jacobians
from weft.body_frame import CAMERA_TO_SIM
from weft.constrained_qp import solve_feasible_qp
from weft.native_reference_inputs import validate_native_reference_view
from weft.root_motion import build_root_reference
from weft.spatial_pose import SPATIAL_JOINT_LIMITS, SPATIAL_JOINT_NAMES
from weft.whole_body_reference_refinement import _NativeGeometry, _schedule, _targets

PROFILE = "native-root-leg-source-fit-v1"
NAMES = SPATIAL_JOINT_NAMES
LEGS = tuple(tuple(NAMES.index(name) for name in NAMES if name.startswith(side)
                   and ("hip" in name or "knee" in name)) for side in ("left_", "right_"))
CONTRACT = {"maximum_references": 128, "maximum_seconds": 30., "finalization_seconds": 8.,
    "maximum_sweeps": 2, "maximum_backtracks": 6, "maximum_geometry_calls": 30000,
    "maximum_output_bytes": 16 << 20, "maximum_gap_s": .25, "sample_hz": 30,
    "maximum_angle_correction_rad": .15, "maximum_update_rad": .10,
    "maximum_angle_correction_speed_rad_s": 1.5, "maximum_root_correction": .08,
    "maximum_root_correction_speed": .3, "maximum_camera_rotation_rad": .15,
    "maximum_endpoint_depth_change": .08, "minimum_camera_depth": .05,
    "floor_penetration_limit": .025, "numerical_tolerance": 1e-8,
    "huber_delta": .05, "original_regularization": .001, "temporal_regularization": .0005,
    "gauge": "focal=source_width; fixed original scale, camera center and first root; zero distortion",
    "scope": "Exposed-development kinematic proposal, not unique reconstruction or physical validation",
    "floor_scope": "All authored human collision supports at original source and 30 Hz samples; not continuous collision proof",
    "deadline_scope": "Cooperative native-call checks; caller must enforce process resource limits"}


def _require(ok, message):
    if not ok:
        raise ValueError("Root/leg source fit: " + message)


def _prepare(human_raw, original_raw, derived_raw, dimensions_raw):
    human, evidence, dimensions = validate_native_reference_view(human_raw, original_raw, derived_raw, dimensions_raw)
    _require(human.get("dt") == 1 / 30, "original 30 Hz body required")
    refs = human["reference"]
    _require(3 <= len(refs) <= CONTRACT["maximum_references"], "3–128 complete references required; no truncation")
    times = np.array([row["time_s"] for row in refs], dtype=float)
    _require(np.isfinite(times).all() and times[0] >= 0 and times[-1] <= 60
             and np.all(np.diff(times) > 0) and np.all(np.diff(times) <= .25 + 1e-9), "unsupported source clock")
    _require(all(set(row) == {"time_s", "joints"} and set(row["joints"]) == set(NAMES) for row in refs),
             "complete original joint chart required")
    angles = np.array([[row["joints"][name] for name in NAMES] for row in refs], dtype=float)
    limits = np.array([SPATIAL_JOINT_LIMITS[name] for name in NAMES])
    _require(np.isfinite(angles).all() and np.all(angles >= limits[:, 0]) and np.all(angles <= limits[:, 1]),
             "original joint bounds differ")
    root = build_root_reference(evidence, torso_length=human["lengths"]["torso"])
    _require(root["status"] == "ready" and human["root_reference"] ==
             [{"time_s": row["time_s"], "translation": row["translation"]} for row in root["reference"]],
             "original root/scale reconstruction differs")
    roots = np.array([row["translation"] for row in root["reference"]])
    scale = root["scale"]["model_units_per_pixel"]
    frames = {row["id"]: row for row in evidence["frames"]}
    targets, weights = [], []
    for reference, origin in zip(refs, root["reference"], strict=True):
        _require(reference["time_s"] == origin["time_s"] and origin["status"] == "observed_image_estimate",
                 "all roots need original observed pelvis evidence")
        target, weight, _ = _targets(frames[origin["evidence_id"]], origin,
                                     dimensions["width"], dimensions["height"], scale)
        targets.append(target / scale + origin["pelvis_image_px"]); weights.append(weight)
    _require(np.all(np.sum(np.array(weights) > 0, axis=1) >= 6), "six confident original endpoints per knot required")
    focal = dimensions["width"]
    K = np.array([[focal, 0., focal / 2], [0., focal, dimensions["height"] / 2], [0., 0., 1.]])
    rotation = CAMERA_TO_SIM.T.copy()
    pelvis = roots[0] + [0., 0., human["mobility"]["base_height"]]
    ray = np.linalg.solve(K, [*root["reference"][0]["pelvis_image_px"], 1.])
    center = pelvis - rotation.T @ (focal * scale * ray)
    return human, times, angles, roots, np.array(targets), np.array(weights), scale, rotation, center, K


def _state(engine, angles, root, derivatives=False):
    value = engine.evaluate(angles, root, derivatives, surface_intervals=True)
    value["absolute_points"] = value["points"] + value["pelvis_position"]
    if derivatives:
        value["absolute_jacobian"] = value["point_jac"] + value["pelvis_jac"][None, :, :]
        # Read native root axes rather than assuming their ordering or orientation.
        model, data = engine.env.model, engine.data
        scratch = np.zeros((3, model.nv))
        engine.mujoco.mj_jacSite(model, data, scratch, None, engine.env._site_ids["pelvis"])
        value["root_jacobian"] = scratch[:, engine.env.mobility.v].copy()
        normal = data.geom_xmat[engine.floor].reshape(3, 3)[:, 2]
        value["root_gap_jacobian"] = normal @ value["root_jacobian"]
    return value


def _bounded(qdelta, rdelta, times, rotation, original_rotation):
    tol = CONTRACT["numerical_tolerance"]
    angle = math.acos(float(np.clip((np.trace(rotation @ original_rotation.T) - 1) / 2, -1., 1.)))
    return (np.max(abs(qdelta)) <= .15 + tol and np.max(np.linalg.norm(rdelta, axis=1)) <= .08 + tol
        and np.array_equal(rdelta[0], np.zeros(3)) and angle <= .15 + tol
        and np.all(abs(np.diff(qdelta, axis=0)) <= 1.5 * np.diff(times)[:, None] + tol)
        and np.all(np.linalg.norm(np.diff(rdelta, axis=0), axis=1) <= .3 * np.diff(times) + tol))


def _floor_summary(states, rows, lookup):
    result = {}
    for kind in ("source_pose", "control_grid"):
        selected = [(stamp, states[lookup[stamp]]["gaps"]) for tag, stamp, _ in rows if tag == kind]
        penetrations = [max(0., -float(gaps.min())) for _, gaps in selected]
        result[kind] = {"checks": len(selected), "failed": sum(value > .025 for value in penetrations),
                       "maximum_penetration": max(penetrations), "excess_sum": sum(max(0., v - .025) for v in penetrations)}
    return result


def propose_root_leg_source_fit(human_bytes, original_evidence_bytes, derived_evidence_bytes, dimensions_bytes,
                                *, maximum_seconds=30., maximum_sweeps=2):
    """Return a separate uninstalled hypothesis with every accepted and rejected local attempt."""
    _require(type(maximum_seconds) in (int, float) and .1 <= maximum_seconds <= 30., "invalid tighter time bound")
    _require(type(maximum_sweeps) is int and 1 <= maximum_sweeps <= 2, "invalid tighter sweep bound")
    started = time.monotonic(); deadline = started + maximum_seconds
    values = _prepare(human_bytes, original_evidence_bytes, derived_evidence_bytes, dimensions_bytes)
    human, times, angles, roots, target, confidence, scale, rotation0, center, K = values
    rows, samples, lookup = _schedule(times)
    source_samples = [lookup[float(t)] for t in times]
    affected = [[j for j, sample in enumerate(samples) if any(k == i for k, _ in sample["weights"])]
                for i in range(len(times))]
    qdelta, rdelta, rotation = np.zeros_like(angles), np.zeros_like(roots), rotation0.copy()
    calls, journal, stop_reason = 0, [], "sweep_limit"
    finalizing = False
    tolerance = CONTRACT["numerical_tolerance"]
    limits = np.array([SPATIAL_JOINT_LIMITS[name] for name in NAMES])

    def check():
        ceiling = CONTRACT["maximum_geometry_calls"] - (0 if finalizing else len(samples))
        if time.monotonic() >= deadline or calls >= ceiling:
            raise TimeoutError("source fit work bound")

    def evaluate(engine, index, qd, rd, derivatives=False):
        nonlocal calls
        check(); calls += 1
        weights = samples[index]["weights"]
        q = sum(w * (angles[k] + qd[k]) for k, w in weights)
        r = sum(w * (roots[k] + rd[k]) for k, w in weights)
        return _state(engine, q, r, derivatives)

    def project(value, rot):
        return project_with_jacobians(value["absolute_points"], rot, center, K)

    def errors(projections):
        return np.linalg.norm(np.array([p["pixels"] for p in projections]) - target, axis=2) * scale

    def valid_projection(projections):
        return all(p["valid"].all() and np.isfinite(p["pixels"]).all()
                   and p["camera_points"][:, 2].min() >= .05 for p in projections)

    def objective(projections, qd, rd):
        norm = errors(projections); huber = .05
        cost = float(np.sum(confidence * np.where(norm <= huber, .5 * norm ** 2, huber * (norm - .5 * huber))))
        cost += .0005 * (float(np.sum(qd ** 2)) + float(np.sum(rd ** 2)))
        cost += .00025 * (float(np.sum(np.diff(qd, axis=0) ** 2)) + float(np.sum(np.diff(rd, axis=0) ** 2)))
        return cost

    engine = _NativeGeometry(human)
    try:
        states = [evaluate(engine, j, qdelta, rdelta) for j in range(len(samples))]
        original_states = states.copy()
        projections = [project(states[j], rotation) for j in source_samples]
        _require(valid_projection(projections), "authored gauge puts an original endpoint behind the camera")
        original_errors = errors(projections)
        original_depth = np.array([p["camera_points"][:, 2] for p in projections])
        original_cost = objective(projections, qdelta, rdelta)
        try:
            for sweep in range(maximum_sweeps):
                blocks = [(None, ())] + [(i, leg) for i in range(len(times)) for leg in LEGS]
                for knot, leg in blocks:
                    check(); camera = knot is None; n = 3 if camera else 7
                    H, gradient = np.eye(n) * .001, np.zeros(n)
                    A, lower = [], []
                    active_indices = range(len(times)) if camera else (knot,)
                    current_errors = errors(projections)
                    for index in active_indices:
                        value = states[source_samples[index]] if camera else evaluate(engine, source_samples[index], qdelta, rdelta, True)
                        p = project(value, rotation)
                        if camera:
                            jac = p["rotation_jacobian"]
                        else:
                            root_jac = p["root_jacobian"] @ value["root_jacobian"]
                            joint_jac = p["root_jacobian"] @ value["absolute_jacobian"][:, :, list(leg)]
                            jac = np.concatenate((root_jac, joint_jac), axis=2)
                        residual = (p["pixels"] - target[index]) * scale; jac = jac * scale
                        for point in np.flatnonzero(confidence[index] > 0):
                            r, J = residual[point], jac[point]; norm = float(np.linalg.norm(r))
                            weight = confidence[index, point] * min(1., .05 / max(norm, 1e-15))
                            H += weight * J.T @ J; gradient += weight * J.T @ r
                            if norm > 1e-12:
                                A.append(-r @ J / norm); lower.append(norm - current_errors[index, point] - tolerance)
                            else:
                                for axis in range(2):
                                    A.extend((J[axis], -J[axis])); lower.extend((-tolerance / 2,) * 2)
                    if not camera:
                        correction = np.r_[rdelta[knot], qdelta[knot, list(leg)]]
                        gradient += .001 * correction
                        for neighbor in (knot - 1, knot + 1):
                            if 0 <= neighbor < len(times):
                                H += np.eye(n) * .0005
                                gradient += .0005 * (correction - np.r_[rdelta[neighbor], qdelta[neighbor, list(leg)]])
                        for j in affected[knot]:
                            state = evaluate(engine, j, qdelta, rdelta, True)
                            factor = dict(samples[j]["weights"])[knot]
                            jac = np.c_[np.tile(state["root_gap_jacobian"], (len(state["gaps"]), 1)), state["gap_jac"][:, list(leg)]] * factor
                            threshold = np.maximum(np.minimum(0., original_states[j]["gaps"]), np.minimum(0., states[j]["gaps"]))
                            A.extend(jac); lower.extend(threshold - state["gaps"] - tolerance)
                        low = np.r_[[-.08] * 3, np.maximum(-.15, limits[list(leg), 0] - angles[knot, list(leg)])] - correction
                        high = np.r_[[.08] * 3, np.minimum(.15, limits[list(leg), 1] - angles[knot, list(leg)])] - correction
                        if knot == 0:
                            low[:3] = high[:3] = 0.
                        for axis in range(n):
                            unit = np.eye(n)[axis]; A.extend((unit, -unit)); lower.extend((max(-.10, low[axis]), -min(.10, high[axis])))
                    else:
                        A.extend(np.r_[np.eye(3), -np.eye(3)]); lower.extend([-.10] * 6)
                    solved = solve_feasible_qp(H, gradient, A, lower)
                    record = {"sweep": sweep, "knot": knot, "joint_names": [NAMES[i] for i in leg],
                              "qp": solved, "trials": [], "accepted": False}
                    journal.append(record)
                    if solved["step"] is None:
                        continue
                    step = np.array(solved["step"])
                    if np.linalg.norm(step) < 1e-12:
                        record["outcome"] = "zero_step"; continue
                    for backtrack in range(6):
                        check(); update = step * 2. ** -backtrack
                        qd, rd = qdelta.copy(), rdelta.copy(); rot = rotation.copy()
                        if camera:
                            rot = _rotation(update) @ rotation
                        else:
                            rd[knot] += update[:3]; qd[knot, list(leg)] += update[3:]
                            if knot == 0: rd[0] = 0.
                        rejection = None
                        if (not _bounded(qd, rd, times, rot, rotation0) or np.any(angles + qd < limits[:, 0] - tolerance)
                                or np.any(angles + qd > limits[:, 1] + tolerance)):
                            rejection = "correction_bounds"
                        candidate_states = states.copy()
                        if rejection is None and not camera:
                            for j in affected[knot]:
                                candidate_states[j] = evaluate(engine, j, qd, rd)
                                threshold = np.maximum(np.minimum(0., original_states[j]["gaps"]), np.minimum(0., states[j]["gaps"]))
                                if np.any(candidate_states[j]["gaps"] < threshold - tolerance): rejection = "floor_nonregression"; break
                        candidate_projection = ([project(candidate_states[j], rot) for j in source_samples]
                                                if rejection is None else None)
                        if rejection is None and not valid_projection(candidate_projection): rejection = "invalid_projection"
                        if rejection is None and np.any(abs(np.array([p["camera_points"][:, 2] for p in candidate_projection]) - original_depth) > .08 + tolerance):
                            rejection = "endpoint_depth"
                        if rejection is None and np.any((errors(candidate_projection) > np.minimum(original_errors, current_errors) + tolerance) & (confidence > 0)):
                            rejection = "endpoint_image_nonregression"
                        cost = objective(candidate_projection, qd, rd) if rejection is None else None
                        if rejection is None and cost >= objective(projections, qdelta, rdelta) - 1e-12: rejection = "no_objective_improvement"
                        record["trials"].append({"backtrack": backtrack, "update": update.tolist(), "rejection": rejection, "objective": cost})
                        if rejection is None:
                            qdelta, rdelta, rotation, states, projections = qd, rd, rot, candidate_states, candidate_projection
                            record["accepted"] = True; break
        except TimeoutError:
            stop_reason = "work_bound"
    finally:
        engine.close()

    # A new model audits all original source/grid samples, including unaffected poses.
    finalizing = True
    deadline = time.monotonic() + CONTRACT["finalization_seconds"]
    audit_engine = _NativeGeometry(human)
    try:
        final = [evaluate(audit_engine, j, qdelta, rdelta) for j in range(len(samples))]
    finally:
        audit_engine.close()
    final_projection = [project(final[j], rotation) for j in source_samples]
    final_errors = errors(final_projection)
    floor_ok = all(np.all(after["gaps"] >= np.minimum(0., before["gaps"]) - tolerance)
                   for before, after in zip(original_states, final, strict=True))
    image_ok = valid_projection(final_projection) and not np.any((final_errors > original_errors + tolerance) & (confidence > 0))
    depth_ok = bool(np.all(abs(np.array([p["camera_points"][:, 2] for p in final_projection]) - original_depth) <= .08 + tolerance))
    before_floor, after_floor = (_floor_summary(s, rows, lookup) for s in (original_states, final))
    image_improved = float(np.sum(confidence * (original_errors - final_errors))) > 1e-8
    floor_improved = all(after_floor[k]["excess_sum"] < before_floor[k]["excess_sum"] - 1e-8 for k in before_floor)
    checked = bool(floor_ok and image_ok and depth_ok and _bounded(qdelta, rdelta, times, rotation, rotation0))
    fully_admissible = bool(checked and image_improved and floor_improved and all(v["failed"] == 0 for v in after_floor.values()))
    result = {"schema_version": 1, "profile": PROFILE, "contract": CONTRACT.copy(),
        "status": "uninstalled_candidate" if fully_admissible else "rejected_unpromoted",
        "physical_validation": "unverified", "stop_reason": stop_reason,
        "bindings": {name: sha256(raw).hexdigest() for name, raw in
                     (("human", human_bytes), ("original_evidence", original_evidence_bytes),
                      ("derived_evidence", derived_evidence_bytes), ("dimensions", dimensions_bytes))},
        "camera": {"rotation_original": rotation0.tolist(), "rotation": rotation.tolist(), "center": center.tolist(), "K": K.tolist()},
        "support_names": list(audit_engine.support_names),
        "sampling_rows": [{"kind": kind, "time_s": stamp, "source_index": index} for kind, stamp, index in rows],
        "candidate": {"reference": [{"time_s": float(t), "joints": dict(zip(NAMES, map(float, q)))} for t, q in zip(times, angles + qdelta)],
                      "root_reference": [{"time_s": float(t), "translation": r.tolist()} for t, r in zip(times, roots + rdelta)]},
        "audit": {"nonregression": checked, "source_image_improved": bool(image_improved), "floor_improved": bool(floor_improved),
                  "complete_original_floor_gate": all(v["failed"] == 0 for v in after_floor.values()),
                  "floor_before": before_floor, "floor_after": after_floor,
                  "weighted_mean_pixel_error_before": float(np.sum(confidence * original_errors) / np.sum(confidence) / scale),
                  "weighted_mean_pixel_error_after": float(np.sum(confidence * final_errors) / np.sum(confidence) / scale),
                  "objective_before": original_cost, "objective_after": objective(final_projection, qdelta, rdelta)},
        "samples": [{"time_s": s["time_s"], "before_gaps": b["gaps"].tolist(), "after_gaps": a["gaps"].tolist()}
                    for s, b, a in zip(samples, original_states, final)],
        "source_endpoints": [{"time_s": float(t), "weights": w.tolist(), "before_error_px": (b / scale).tolist(), "after_error_px": (a / scale).tolist()}
                             for t, w, b, a in zip(times, confidence, original_errors, final_errors)],
        "journal": journal, "geometry_calls": calls, "seconds": time.monotonic() - started}
    _require(len(json.dumps(result, allow_nan=False).encode()) <= CONTRACT["maximum_output_bytes"], "output bound exceeded")
    return result
