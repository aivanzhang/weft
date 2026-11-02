"""Source-only local observability and simultaneous tangent directions; never a body repair."""

from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.body_camera_root_refinement import _array, project_with_jacobians
from weft.constrained_qp import solve_feasible_qp
from weft import root_leg_source_fit as source_fit
from weft.spatial_pose import SPATIAL_JOINT_LIMITS
from weft.whole_body_reference_refinement import _schedule

PROFILE = "native-root-leg-direction-diagnostic-v1"
CONTRACT = {"maximum_references": 64, "maximum_samples": 256, "maximum_constraints": 4096,
    "maximum_geometry_calls": 1200, "maximum_seconds": 20., "maximum_output_bytes": 16 << 20,
    "maximum_modes": 8, "rank_relative_tolerance": 1e-10, "active_tolerance": 1e-8,
    "direction_tolerance": 1e-8, "minimum_floor_direction_gain": 1e-8,
    "column_scales": {"camera_rad": .15, "root_model_units": .08, "joint_rad": .15},
    "basis": "3 camera-compensation modes, 3 leading failed-floor modes, 2 weakest data-image modes",
    "scope": "First-order witness in a deterministic subspace under a fixed authored gauge; no finite correction or global infeasibility claim",
    "rank_scope": "Confidence-weighted source-image Jacobian only; no priors, floor constraints or regularization supply rank",
    "deadline_scope": "Cooperative bounded native calls; parent must enforce process limits"}


def _require(ok, message):
    if not ok:
        raise ValueError("Root/leg direction diagnostic: " + message)


def _rank(matrix):
    _, singular, vt = np.linalg.svd(matrix, full_matrices=matrix.shape[0] < matrix.shape[1])
    threshold = float(singular[0]) * CONTRACT["rank_relative_tolerance"] if len(singular) else 0.
    rank = int(np.count_nonzero(singular > threshold))
    return {"rank": rank, "variables": matrix.shape[1], "rows": matrix.shape[0],
            "nullity": matrix.shape[1] - rank, "threshold": threshold,
            "condition": float(singular[0] / singular[-1]) if rank == matrix.shape[1] else None,
            "singular_values": singular.tolist(), "regularization_used": False}, vt


def _basis(image, floor):
    rank, right = _rank(image)
    # Camera columns move jointly with the minimum-norm data-only body compensation.
    compensation, _, _, _ = np.linalg.lstsq(image[:, 3:], -image[:, :3], rcond=1e-10)
    candidates = [np.r_[np.eye(3)[:, i], compensation[:, i]] for i in range(3)]
    labels = ["camera_compensation_" + str(i) for i in range(3)]
    if len(floor):
        _, _, fv = np.linalg.svd(floor, full_matrices=False)
        candidates += list(fv[:3]); labels += ["failed_floor_" + str(i) for i in range(min(3, len(fv)))]
    candidates += list(right[-2:]); labels += ["weak_image_" + str(i) for i in range(min(2, len(right)))]
    kept, selected = [], []
    for vector, label in zip(candidates, labels, strict=True):
        value = vector.copy()
        for _ in range(2):
            for prior in kept:
                value -= prior * (prior @ value)
        norm = float(np.linalg.norm(value))
        if norm > 1e-10 * max(1., float(np.linalg.norm(vector))):
            kept.append(value / norm); selected.append(label)
    return np.array(kept).T, selected, rank


def analyze_linear_directions(image, floor_targets, constraints, *, deadline=None):
    """Find a checked tangent witness; a missing witness says nothing about other subspaces."""
    image, floor_targets, constraints = (np.asarray(v, dtype=float) for v in (image, floor_targets, constraints))
    n = image.shape[1] if image.ndim == 2 else 0
    _require(3 < n <= 704 and 1 <= len(image) <= 1536 and image.ndim == 2,
             "bounded source-image matrix required")
    _require(floor_targets.ndim == constraints.ndim == 2 and floor_targets.shape[1] == constraints.shape[1] == n
             and len(floor_targets) <= 4096 and len(constraints) <= CONTRACT["maximum_constraints"] - 16,
             "bounded compatible direction matrices required")
    _require(all(np.isfinite(v).all() and np.max(abs(v), initial=0.) <= 1e10 for v in (image, floor_targets, constraints)),
             "finite bounded derivative coefficients required")
    basis, labels, rank = _basis(image, floor_targets)
    _require(basis.shape[1] <= 8, "mode bound")
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("direction diagnostic matrix deadline")
    objective = floor_targets.sum(axis=0)
    reduced_objective = objective @ basis
    norm = float(np.linalg.norm(reduced_objective))
    reduced = constraints @ basis
    rows = np.r_[reduced, np.eye(basis.shape[1]), -np.eye(basis.shape[1])]
    lower = np.r_[np.zeros(len(reduced)), -np.ones(2 * basis.shape[1])]
    solved = solve_feasible_qp(np.eye(basis.shape[1]), -reduced_objective / max(norm, 1e-15), rows, lower)
    direction = basis @ np.array(solved["step"]) if solved["step"] is not None else np.zeros(n)
    length = float(np.linalg.norm(direction))
    if length > 1e-12:
        direction /= length
    derivatives = constraints @ direction
    gains = floor_targets @ direction
    violation = max(0., -float(np.min(derivatives, initial=0.)))
    gain = float(np.sum(gains))
    witness = solved["status"] == "optimal" and violation <= 1e-8 and gain > 1e-8
    return {"status": "first_order_subspace_witness" if witness else "no_verified_subspace_witness",
            "observability": rank, "mode_labels": labels, "basis_normalized_coordinates": basis.tolist(),
            "direction_normalized_coordinates": direction.tolist(), "constraint_directional_values": derivatives.tolist(),
            "floor_target_directional_values": gains.tolist(), "maximum_linear_violation": violation,
            "summed_floor_direction_gain": gain, "qp": solved,
            "finite_step_validated": False, "global_infeasibility_claim": False}


def diagnose_dependent_cone(constraints, objective, *, maximum_seconds=5.):
    """Restrict a saved ≤8D cone along positive dependency cycles; verify every original row."""
    C, g = np.asarray(constraints, dtype=float), np.asarray(objective, dtype=float)
    _require(g.ndim == 1 and 1 <= len(g) <= 8 and C.ndim == 2 and C.shape[1] == len(g)
             and 0 <= len(C) <= 4080, "bounded saved cone matrices required")
    _require(np.isfinite(C).all() and np.isfinite(g).all() and np.max(abs(C), initial=0.) <= 1e10
             and np.max(abs(g), initial=0.) <= 1e10, "finite saved cone coefficients required")
    _require(type(maximum_seconds) in (int, float) and .01 <= maximum_seconds <= 5., "invalid cone time bound")
    started = time.monotonic(); deadline = started + maximum_seconds
    embedding, history, direction = np.eye(len(g)), [], np.zeros(len(g))
    redundant = set()
    status = "no_verified_subspace_witness"
    for _ in range(32):
        if time.monotonic() >= deadline:
            status = "work_bound"; break
        n = embedding.shape[1]
        if not n: break
        matrix = C @ embedding
        # Near-null rows cannot exceed the original witness tolerance on a unit direction.
        norms = np.linalg.norm(matrix, axis=1)
        retained = np.array([i for i in np.flatnonzero(norms > 1e-12) if i not in redundant], dtype=int)
        rows = np.r_[matrix[retained], embedding, -embedding]
        lower = np.r_[np.zeros(len(retained)), -np.ones(2 * len(g))]
        reduced = g @ embedding
        solved = solve_feasible_qp(np.eye(n), -reduced / max(float(np.linalg.norm(reduced)), 1e-15), rows, lower)
        record = {"variables": n, "qp": solved, "search_rows": retained.tolist(),
                  "near_null_search_rows": np.flatnonzero(norms <= 1e-12).tolist(),
                  "redundant_search_rows": sorted(redundant),
                  "near_null_unit_direction_maximum_effect": float(np.max(norms[norms <= 1e-12], initial=0.)),
                  "all_original_rows_retained_for_verification": True}; history.append(record)
        if solved["step"] is not None:
            value = embedding @ np.array(solved["step"])
            norm = float(np.linalg.norm(value))
            if norm > 1e-12:
                value /= norm
                if np.min(C @ value, initial=0.) >= -1e-8 and g @ value > 1e-8:
                    direction = value; status = "first_order_subspace_witness"
            break
        report = solved["report"]
        if solved["status"] != "dependent_blocker": break
        active = report["active_constraints"]
        blocked = report["dependent_blockers"][-1]
        if not active or any(i >= len(retained) for i in [*active, blocked]): break
        active = [int(retained[i]) for i in active]
        blocked = int(retained[blocked])
        coefficients = np.linalg.lstsq(matrix[active].T, -matrix[blocked], rcond=1e-12)[0]
        residue = matrix[active].T @ coefficients + matrix[blocked]
        relative_error = float(np.linalg.norm(residue) / max(np.linalg.norm(matrix[blocked]), 1e-15))
        positive_cycle = bool(np.min(coefficients) >= -1e-12 and relative_error <= 1e-10)
        record["dependency"] = {"active_rows": active, "blocked_row": blocked,
            "positive_weights": coefficients.tolist(), "relative_residual": relative_error,
            "positive_cycle": positive_cycle, "restriction_not_global_infeasibility": True}
        if not positive_cycle:
            if np.max(coefficients) <= 1e-12 and float(np.linalg.norm(residue)) <= 1e-12:
                # Nonnegative combinations add no search restriction; original rows still verify the witness.
                coefficients_all = np.linalg.lstsq(matrix[active].T, matrix.T, rcond=1e-12)[0]
                errors = np.linalg.norm(matrix[active].T @ coefficients_all - matrix.T, axis=0)
                excluded = [int(i) for i in retained if i not in active and np.min(coefficients_all[:, i]) >= -1e-12
                            and errors[i] <= 1e-12]
                record["redundancy"] = {"retained_generators": active, "excluded_search_rows": excluded,
                    "maximum_unit_residual": float(np.max(errors[excluded], initial=0.)),
                    "all_original_rows_retained_for_verification": True}
                if blocked not in excluded: break
                redundant.update(excluded)
                continue
            break
        # Restriction is conservative near floating-point rank boundaries; no original guard is dropped.
        _, singular, vh = np.linalg.svd(matrix[[*active, blocked]], full_matrices=True)
        rank = int(np.count_nonzero(singular > singular[0] * 1e-12))
        if not rank: break
        embedding = embedding @ vh[rank:].T
    values = C @ direction
    result = {"schema_version": 1, "profile": "saved-dependent-tangent-cone-v1", "status": status,
        "direction": direction.tolist(), "directional_constraint_values": values.tolist(),
        "maximum_linear_violation": max(0., -float(np.min(values, initial=0.))),
        "objective_direction_gain": float(g @ direction), "unit_direction": bool(np.linalg.norm(direction) > .999999),
        "remaining_dimensions": embedding.shape[1], "rounds": history, "seconds": time.monotonic()-started,
        "all_original_constraints_verified": True, "finite_step_validated": False,
        "global_infeasibility_claim": False, "conservative_restriction_may_omit_other_directions": True}
    return result


def diagnose_root_leg_directions(human_bytes, original_evidence_bytes, derived_evidence_bytes, dimensions_bytes,
                                candidate_bytes, *, maximum_seconds=20.):
    _require(type(maximum_seconds) in (int, float) and .1 <= maximum_seconds <= 20., "invalid tighter deadline")
    started = time.monotonic(); deadline = started + maximum_seconds
    inputs = (human_bytes, original_evidence_bytes, derived_evidence_bytes, dimensions_bytes)
    human, times, angles, roots, target, confidence, scale, rotation0, center, K = source_fit._prepare(*inputs)
    _require(len(times) <= 64, "at most64 complete references; no source truncation")
    _require(type(candidate_bytes) is bytes and 0 < len(candidate_bytes) <= 16 << 20, "bounded candidate bytes required")
    candidate = json.loads(candidate_bytes)
    _require(candidate.get("profile") == source_fit.PROFILE and candidate.get("contract") == source_fit.CONTRACT,
             "unknown source-fit candidate contract")
    _require(candidate.get("bindings") == dict(zip(("human", "original_evidence", "derived_evidence", "dimensions"),
                                                  (sha256(raw).hexdigest() for raw in inputs))), "candidate source bindings differ")
    camera = candidate["camera"]
    _require(camera["K"] == K.tolist() and camera["center"] == center.tolist()
             and camera["rotation_original"] == rotation0.tolist(), "frozen original gauge differs")
    rotation = _array(camera["rotation"], (3, 3), 1. + 1e-10)
    _require(np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-10, rtol=0)
             and abs(np.linalg.det(rotation) - 1) < 1e-10, "proper camera rotation required")
    refs, root_refs = candidate["candidate"]["reference"], candidate["candidate"]["root_reference"]
    _require(len(refs) == len(root_refs) == len(times) and [row["time_s"] for row in refs] == times.tolist()
             and [row["time_s"] for row in root_refs] == times.tolist()
             and all(set(row["joints"]) == set(source_fit.NAMES) for row in refs), "complete exact candidate clock required")
    q = _array([[row["joints"][name] for name in source_fit.NAMES] for row in refs], angles.shape)
    root = _array([row["translation"] for row in root_refs], roots.shape)
    qd, rd = q - angles, root - roots
    legs = sorted(set(i for leg in source_fit.LEGS for i in leg))
    protected = sorted(set(range(len(source_fit.NAMES))) - set(legs))
    _require(np.array_equal(q[:, protected], angles[:, protected]) and source_fit._bounded(qd, rd, times, rotation, rotation0),
             "candidate changed protected angles or correction bounds")
    limits = np.array([SPATIAL_JOINT_LIMITS[name] for name in source_fit.NAMES])
    _require(np.all(q >= limits[:, 0] - 1e-8) and np.all(q <= limits[:, 1] + 1e-8), "candidate joint limits differ")
    _, samples, lookup = _schedule(times)
    _require(len(samples) <= 256, "complete source/grid sampling exceeds diagnostic bound")
    columns = [{"kind": "camera", "axis": axis, "scale": .15} for axis in range(3)]
    root_columns, joint_columns = {}, {}
    for knot in range(len(times)):
        if knot:
            root_columns[knot] = list(range(len(columns), len(columns) + 3))
            columns += [{"kind": "root", "source_index": knot, "axis": axis, "scale": .08} for axis in range(3)]
        joint_columns[knot] = list(range(len(columns), len(columns) + len(legs)))
        columns += [{"kind": "joint", "source_index": knot, "name": source_fit.NAMES[axis], "scale": .15} for axis in legs]
    n = len(columns); scales = np.array([c["scale"] for c in columns])
    constraints, labels, floors, floor_labels, images = [], [], [], [], []
    calls = 0

    def check():
        if time.monotonic() >= deadline or calls >= CONTRACT["maximum_geometry_calls"]:
            raise TimeoutError("source-only diagnostic work bound")

    def add(row, label):
        _require(len(constraints) < CONTRACT["maximum_constraints"] - 16, "complete constraints exceed diagnostic bound")
        constraints.append(np.asarray(row) * scales); labels.append(label)

    def axis_row(knot, axis):
        row = np.zeros(n)
        if axis < 3:
            if knot: row[root_columns[knot][axis]] = 1.
        else:
            row[joint_columns[knot][axis - 3]] = 1.
        return row

    engine = source_fit._NativeGeometry(human)
    try:
        current, original = [], []
        for sample in samples:
            check()
            for states, aq, ar, derivative in ((original, angles, roots, False), (current, q, root, True)):
                check(); calls += 1
                weights = sample["weights"]
                states.append(source_fit._state(engine, sum(w * aq[i] for i, w in weights),
                                                sum(w * ar[i] for i, w in weights), derivative))
        for sample, before, after in zip(samples, original, current, strict=True):
            _require(np.all(after["gaps"] >= np.minimum(0., before["gaps"]) - 1e-8), "candidate floor regression")
            for support, gap in enumerate(after["gaps"]):
                if gap > 1e-8: continue
                row = np.zeros(n)
                for knot, weight in sample["weights"]:
                    if knot: row[root_columns[knot]] += weight * after["root_gap_jacobian"]
                    row[joint_columns[knot]] += weight * after["gap_jac"][support, legs]
                label = {"kind": "floor", "time_s": sample["time_s"], "support": engine.support_names[support], "gap": float(gap)}
                add(row, label)
                if gap < -.025:
                    floors.append(row * scales); floor_labels.append(label)
        for knot, stamp in enumerate(times):
            check(); before, after = original[lookup[float(stamp)]], current[lookup[float(stamp)]]
            baseline = project_with_jacobians(before["absolute_points"], rotation0, center, K)
            p = project_with_jacobians(after["absolute_points"], rotation, center, K)
            _require(p["valid"].all() and baseline["valid"].all() and p["camera_points"][:, 2].min() >= .05,
                     "candidate projection invalid")
            residual = (p["pixels"] - target[knot]) * scale
            baseline_error = np.linalg.norm((baseline["pixels"] - target[knot]) * scale, axis=1)
            error = np.linalg.norm(residual, axis=1)
            _require(np.all((error <= baseline_error + 1e-8) | (confidence[knot] == 0)), "candidate endpoint image regression")
            depth_change = p["camera_points"][:, 2] - baseline["camera_points"][:, 2]
            _require(np.max(abs(depth_change)) <= .08 + 1e-8, "candidate endpoint depth regression")
            J, depth = np.zeros((12, 2, n)), np.zeros((12, n))
            J[:, :, :3] = p["rotation_jacobian"]
            depth[:, :3] = np.c_[p["camera_points"][:, 1], -p["camera_points"][:, 0], np.zeros(12)]
            if knot:
                J[:, :, root_columns[knot]] = p["root_jacobian"] @ after["root_jacobian"]
                depth[:, root_columns[knot]] = rotation[2] @ after["root_jacobian"]
            J[:, :, joint_columns[knot]] = p["root_jacobian"] @ after["absolute_jacobian"][:, :, legs]
            depth[:, joint_columns[knot]] = rotation[2] @ after["absolute_jacobian"][:, :, legs]
            for point in range(12):
                if confidence[knot, point] > 0:
                    jac = J[point] * scale
                    images.extend(jac * scales * math.sqrt(confidence[knot, point]))
                    if error[point] > 1e-12:
                        add(-residual[point] @ jac / error[point], {"kind": "image", "source_index": knot, "point": point})
                    else:
                        for axis in range(2):
                            for sign in (-1, 1):
                                add(sign * jac[axis], {"kind": "zero_image", "source_index": knot, "point": point, "axis": axis, "sign": sign})
                for sign in (-1, 1):
                    if sign * depth_change[point] >= .08 - 1e-8:
                        add(-sign * depth[point], {"kind": "depth", "source_index": knot, "point": point, "sign": sign})
                if p["camera_points"][point, 2] <= .05 + 1e-8:
                    add(depth[point], {"kind": "minimum_depth", "source_index": knot, "point": point})
    finally:
        engine.close()

    for knot in range(len(times)):
        for axis, joint in enumerate(legs):
            row = axis_row(knot, axis + 3)
            for sign in (-1, 1):
                limit = limits[joint, 1] if sign > 0 else -limits[joint, 0]
                if sign * qd[knot, joint] >= .15 - 1e-8 or sign * q[knot, joint] >= limit - 1e-8:
                    add(-sign * row, {"kind": "joint_bound", "source_index": knot, "joint": source_fit.NAMES[joint], "sign": sign})
                if knot and sign * (qd[knot, joint] - qd[knot-1, joint]) >= 1.5 * (times[knot]-times[knot-1]) - 1e-8:
                    add(-sign * (row-axis_row(knot-1, axis+3)), {"kind": "joint_speed", "source_index": knot, "joint": source_fit.NAMES[joint], "sign": sign})
        for value, bound, previous, kind in ((rd[knot], .08, None, "root_bound"),
                (rd[knot]-rd[knot-1], .3*(times[knot]-times[knot-1]), knot-1, "root_speed")) if knot else ():
            norm = float(np.linalg.norm(value))
            if norm >= bound - 1e-8:
                row = sum(value[axis]/norm * (axis_row(knot,axis) - (axis_row(previous,axis) if previous is not None else 0)) for axis in range(3))
                add(-row, {"kind": kind, "source_index": knot})
    relative = rotation @ rotation0.T
    vector = np.array([relative[2,1]-relative[1,2], relative[0,2]-relative[2,0], relative[1,0]-relative[0,1]]) / 2
    angle = math.atan2(float(np.linalg.norm(vector)), float((np.trace(relative)-1)/2))
    if angle >= .15 - 1e-8:
        row = np.zeros(n); row[:3] = -vector / np.linalg.norm(vector)
        add(row, {"kind": "camera_bound"})
    check()
    result = analyze_linear_directions(np.array(images), np.array(floors).reshape(-1, n), np.array(constraints), deadline=deadline)
    physical = np.array(result["direction_normalized_coordinates"]) * scales
    result.update(schema_version=1, profile=PROFILE, contract=CONTRACT.copy(),
        bindings={**candidate["bindings"], "candidate_sha256": sha256(candidate_bytes).hexdigest()},
        columns=columns, constraint_labels=labels, floor_target_labels=floor_labels,
        direction_physical_coordinates=physical.tolist(),
        constraints_reduced=(np.array(constraints) @ np.array(result["basis_normalized_coordinates"])).tolist(),
        source_times=times.tolist(), sample_times=[s["time_s"] for s in samples],
        geometry_calls=calls, seconds=time.monotonic()-started, inference_calls=0, rollouts=0, physical_validation="unverified")
    _require(len(json.dumps(result, allow_nan=False).encode()) <= 16 << 20, "diagnostic output exceeds bound")
    return result
