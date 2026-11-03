"""Coupled arm contact hypotheses with full native-boundary constraints, never dynamics."""

from hashlib import sha256
import json
import math
from pathlib import Path
import sys
import time
from types import MappingProxyType

import numpy as np

from weft.io import canonical_hash
from weft import moving_equipment_support as support

PROFILE = "moving-equipment-coupled-contact-search-v1"
LIMITS = MappingProxyType({"maximum_seconds": 60, "maximum_evaluations": 24000, "maximum_iterations": 6,
    "maximum_variables": 518, "maximum_rows": 8192, "maximum_samples": 256, "maximum_output_bytes": 8 << 20,
    "angle_correction_rad": .5, "angle_update_rad": .15, "backtracks": 4, "qp_seconds": .4, "qp_sweeps": 128,
    "matrix_workspace_bound_bytes": 128 << 20, "restoration_normal_scale_world": .1})
SCOPE = ("Two restricted coupled-arm geometry searches, with every retained terminal native boundary constrained. "
    "Restoration slack is search-only; no relaxed acceptance, dynamic support, source edit, attachment or task success.")


def _require(value, message):
    if not value: raise ValueError("Moving contact search: " + message)


def _bytes(value):
    result = bytearray()
    for text in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        raw = text.encode()
        # Reserve room for a final deadline/refusal annotation after encoding has completed.
        _require(len(result)+len(raw) <= LIMITS["maximum_output_bytes"]-4096, "report exceeds byte bound")
        result.extend(raw)
    return bytes(result)


def _weights(knots, timestamp):
    index = min(len(knots)-2, max(0, int(np.searchsorted(knots, timestamp, side="right"))-1))
    weight = (timestamp-knots[index])/(knots[index+1]-knots[index])
    return ((index, 1-weight), (index+1, weight))


def _expand(jacobian, weights, count):
    row = np.zeros(count*8)
    for index, weight in weights: row[index*8:(index+1)*8] += weight*jacobian
    return row


def _capsule_state(model, data, gid, columns, center):
    """Differentiate the continuous closest point, including its interior segment coordinate."""
    import mujoco
    p = data.geom_xpos[gid].copy()
    axis = data.geom_xmat[gid].reshape(3, 3)[:, 2].copy()
    radius, half = map(float, model.geom_size[gid, :2])
    _require(int(model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE), "movable human capsule required")
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jp, jr, p, int(model.geom_bodyid[gid]))
    jp = jp[:, columns]
    ja = np.cross(jr[:, columns].T, axis).T
    raw_coordinate = float((center-p) @ axis)
    coordinate = float(np.clip(raw_coordinate, -half, half))
    jpoint = jp + coordinate*ja
    if -half < raw_coordinate < half:
        jpoint += axis[:, None] * ((center-p) @ ja - axis @ jp)[None, :]
    point = p + coordinate*axis
    difference = center-point
    length = float(np.linalg.norm(difference))
    _require(length > 1e-10, "coincident capsule centerline has no qualified distance derivative")
    normal = difference/length
    jnormal = -(np.eye(3)-np.outer(normal, normal)) @ jpoint/length
    return {"center": p, "axis": axis, "radius": radius, "half": half, "point": point,
        "normal": normal, "distance": length, "gap_jacobian": -normal@jpoint,
        "normal_jacobian": jnormal, "center_jacobian": jp, "axis_jacobian": ja,
        "closest_coordinate": coordinate}


class _Coupled:
    def __init__(self, scene, task, budget, knots, times):
        self.geometry = support._Geometry(scene, task, budget)
        try:
            self.env, self.model, self.data = self.geometry.env, self.geometry.model, self.geometry.data
            self.knots, self.times, self.budget = knots, times, budget
            left, li, lg = self.geometry.arm("left")
            right, ri, rg = self.geometry.arm("right")
            self.names, self.indices = left+right, np.r_[li, ri]
            self.columns = self.env._v[self.indices]
            self.supports = {"left": lg, "right": rg}
            self.moving = {self.model.geom(side+joint+"_geom").id for side in ("left", "right") for joint in ("_shoulder", "_elbow")}
            self.fixed = set(self.geometry.human_ids)-self.moving
            self.rows = [(t, _weights(knots, t)) for t in times]
            self.ranges = self.model.jnt_range[np.asarray(self.env._human_joint_ids)[self.indices]].copy()
            self.model_pin = self._model_identity()
        except BaseException:
            self.geometry.close(); raise

    def _model_identity(self):
        names = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "geom_size", "geom_friction", "geom_contype",
            "geom_conaffinity", "geom_solref", "geom_solimp", "jnt_range", "dof_damping", "actuator_gear", "actuator_ctrlrange")
        return canonical_hash({**{name: getattr(self.model, name).tolist() for name in names},
            "gravity": self.model.opt.gravity.tolist(), "timestep": self.model.opt.timestep})

    def close(self):
        try: _require(self._model_identity() == self.model_pin, "protected model parameters changed")
        finally: self.geometry.close()

    def fixed_obstructions(self):
        import mujoco
        rows = []
        radius = self.geometry.object["radius"]
        for stamp, _ in self.rows:
            self.geometry.pose(stamp, self.indices, np.zeros(8))
            floor = float(mujoco.mj_geomDistance(self.model, self.data, self.geometry.ogid, self.env._floor_geom, 100., None))
            failed = []
            if floor < -1e-8: failed.append({"kind": "object_floor", "value": floor})
            if np.any(self.geometry.target-radius < -1e-8) or np.any(self.geometry.target+radius > np.asarray(self.geometry.scene["bounds"])+1e-8):
                failed.append({"kind": "object_world_containment", "value": False})
            for gid in sorted(self.fixed):
                gap = float(mujoco.mj_geomDistance(self.model, self.data, gid, self.geometry.ogid, 100., None))
                floor = float(mujoco.mj_geomDistance(self.model, self.data, gid, self.env._floor_geom, 100., None))/self.env.scale
                if gap < -1e-8: failed.append({"kind": "fixed_body_object_overlap", "geometry": self.env._human_geom_to_name[gid], "value": gap})
                if floor < -self.geometry.task["body_gates"]["max_floor_penetration_model_units"]:
                    failed.append({"kind": "fixed_body_floor", "geometry": self.env._human_geom_to_name[gid], "value": floor})
            if failed: rows.append({"time_s": stamp, "failed": failed})
        return rows

    def constraints(self, correction, side, *, derivatives):
        values, matrix, labels, hard = [], [], [], []
        count = len(self.knots)
        def add(value, jac, weight, name, stamp, fixed=False):
            values.append(float(value)); labels.append({"kind": name, "time_s": stamp}); hard.append(fixed)
            if derivatives: matrix.append(_expand(jac, weight, count))
        for stamp, weights in self.rows:
            delta = sum(weight*correction[index] for index, weight in weights)
            reference = self.geometry.pose(stamp, self.indices, delta)
            floor_normal = self.data.geom_xmat[self.env._floor_geom].reshape(3, 3)[:, 2]
            floor_origin = self.data.geom_xpos[self.env._floor_geom]
            for gid in sorted(self.moving):
                state = _capsule_state(self.model, self.data, gid, self.columns, self.geometry.target)
                name = self.env._human_geom_to_name[gid]
                gap = state["distance"]-state["radius"]-self.geometry.object["radius"]
                add(gap+1e-8, state["gap_jacobian"], weights, "nonpenetration:"+name, stamp)
                sign = -1. if floor_normal @ state["axis"] >= 0 else 1.
                bottom = state["center"]+sign*state["half"]*state["axis"]
                floor_gap = (floor_normal@(bottom-floor_origin)-state["radius"])/self.env.scale
                floor_jac = floor_normal@(state["center_jacobian"]+sign*state["half"]*state["axis_jacobian"])/self.env.scale
                add(floor_gap+self.geometry.task["body_gates"]["max_floor_penetration_model_units"], floor_jac, weights, "floor:"+name, stamp)
                if gid == self.supports[side]:
                    add(1e-4-gap, -state["gap_jacobian"], weights, "support_gap", stamp)
                    scale = LIMITS["restoration_normal_scale_world"]
                    add((state["normal"][2]-.9)*scale, state["normal_jacobian"][2]*scale, weights, "upward_normal_scaled", stamp)
            for i, name in enumerate(self.names):
                q = reference[self.indices[i]]+delta[i]
                add(q-self.ranges[i, 0], np.eye(8)[i], weights, "joint_lower:"+name, stamp, True)
                add(self.ranges[i, 1]-q, -np.eye(8)[i], weights, "joint_upper:"+name, stamp, True)
        _require(len(values)+2*count*8+2 <= LIMITS["maximum_rows"], "complete constraint matrix exceeds row bound")
        return np.asarray(values), np.asarray(matrix) if derivatives else None, labels, np.asarray(hard)


def _proposal(correction, values, matrix, hard, solve, budget):
    n = correction.size
    _require(n+1 <= LIMITS["maximum_variables"], "trajectory variable bound exceeded")
    current = correction.ravel()
    low = np.maximum(-LIMITS["angle_update_rad"], -.5-current)
    high = np.minimum(LIMITS["angle_update_rad"], .5-current)
    constraints = np.vstack((matrix, np.eye(n), -np.eye(n)))
    lower = np.r_[-values, low, -high]
    hessian, gradient = np.eye(n)*.02, .0001*current
    workspace = constraints.nbytes*4 + hessian.nbytes*4
    _require(workspace <= LIMITS["matrix_workspace_bound_bytes"], "matrix workspace bound exceeded")
    budget.check()
    full = solve(hessian, gradient, constraints, lower,
        maximum_seconds=min(LIMITS["qp_seconds"], max(.001, budget.deadline-time.monotonic())), maximum_sweeps=LIMITS["qp_sweeps"])
    budget.check()
    reports = [{"phase": "full_original_constraints", "status": full["status"], "report": full["report"]}]
    if full["step"] is not None:
        return np.asarray(full["step"]), reports, 0.
    # Only restoration intermediates carry slack; every final witness rechecks the original nonlinear constraints.
    slack = max(0., float(np.max(-values[~hard], initial=0.))) + 1e-7
    soft = np.r_[(~hard).astype(float), np.zeros(2*n)]
    enlarged = np.column_stack((constraints, soft))
    enlarged = np.vstack((enlarged, np.r_[np.zeros(n), 1.], np.r_[np.zeros(n), -1.]))
    bound = np.r_[lower, 0., -slack]
    H = np.zeros((n+1, n+1)); H[:n, :n] = hessian; H[-1, -1] = .01
    g = np.r_[gradient, 1.]
    budget.check()
    restored = solve(H, g, enlarged, bound, maximum_seconds=min(LIMITS["qp_seconds"], max(.001,budget.deadline-time.monotonic())),
        maximum_sweeps=LIMITS["qp_sweeps"], feasible_initial=np.r_[np.zeros(n), slack])
    budget.check()
    reports.append({"phase": "search_only_restoration", "status": restored["status"], "report": restored["report"]})
    if restored["step"] is None: return None, reports, slack
    return np.asarray(restored["step"][:-1]), reports, float(restored["step"][-1])


def _restoration_groups(labels, hard):
    collision = np.array([r["kind"].startswith(("nonpenetration:", "floor:")) for r in labels])
    support_rows = np.array([r["kind"] in {"support_gap", "upward_normal_scaled"} for r in labels])
    _require(np.all(collision | support_rows | hard) and not np.any((collision | support_rows) & hard),
             "restoration row classification differs")
    return collision, support_rows


def _separate_proposal(correction, values, matrix, labels, hard, budget):
    from weft.moving_equipment_contact_restore import restore_contact_direction
    collision, support_rows = _restoration_groups(labels, hard)
    current = correction.ravel()
    n = len(current)
    low = np.maximum(-LIMITS["angle_update_rad"], -.5-current)
    high = np.minimum(LIMITS["angle_update_rad"], .5-current)
    C = np.vstack((matrix[hard], np.eye(n), -np.eye(n)))
    lower = np.r_[-values[hard], low, -high]
    budget.check()
    result = restore_contact_direction(matrix[collision], values[collision], matrix[support_rows], values[support_rows],
        C, lower, max_seconds=min(2., max(.001, budget.deadline-time.monotonic())))
    budget.check()
    hard_labels = [labels[i] for i in np.flatnonzero(hard)]
    hard_labels += [{"kind": "update_or_correction_lower", "variable": i} for i in range(n)]
    hard_labels += [{"kind": "update_or_correction_upper", "variable": i} for i in range(n)]
    initial_rows = [{"row": int(i), **hard_labels[i], "residual": float(-lower[i])} for i in np.flatnonzero(lower > 0)]
    report = {**result["report"], "initial_hard_anchor": {"violations": initial_rows, "original_hard_rows": len(lower),
        "scope": "Exact caller residuals before bounded Phase-I; no timestamp, pose or bound clamp."}}
    reports = [{"phase": "separate_collision_support_restoration", "status": result["status"],
                "report": report, "failure": result["failure"]}]
    step = None if result["status"] in {"incomplete_resource_bound", "unsupported_hard_anchor"} or result["step"] is None else np.asarray(result["step"])
    return step, reports, None


def _restoration_progress(before, after, labels, hard):
    collision, support_rows = _restoration_groups(labels, hard)
    def cost(values, mask):
        residual = np.minimum(values[mask], 0.)
        return float(residual @ residual)/max(1, len(residual))
    old_collision, new_collision = cost(before, collision), cost(after, collision)
    old_support, new_support = cost(before, support_rows), cost(after, support_rows)
    preserved = np.min(after[collision]-np.minimum(before[collision], 0.), initial=0.) >= -1e-8
    accepted = new_collision < old_collision-1e-14 or (preserved and new_support < old_support-1e-14)
    return accepted, {"before_collision_squared_deficit": old_collision, "after_collision_squared_deficit": new_collision,
        "before_support_squared_deficit": old_support, "after_support_squared_deficit": new_support,
        "previous_collision_bounds_preserved": bool(preserved)}


def _runtime(admission, scene, budget=None):
    from weft.moving_equipment_task_recording import runtime_identity
    from weft.shared_world_packaging import read_package_file
    if budget is not None: budget.check()
    identity = runtime_identity(scene)
    if budget is not None: budget.check()
    _require(all(admission.runtime_sha256.get(name) == pin for name, pin in identity["runtime_sha256"].items()), "admitted task runtime changed")
    # Imported numerical helpers have transitive local imports; pin the actual loaded package closure, never execute saved code.
    package = Path(__file__).parent.resolve()
    names = set(identity["runtime_sha256"])
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("weft.") and path and Path(path).parent.resolve() == package and Path(path).suffix == ".py":
            names.add(Path(path).name)
    _require(len(names) <= 256, "loaded runtime closure exceeds 256 modules")
    total, pins = 0, {}
    for name in sorted(names):
        if budget is not None: budget.check()
        raw = read_package_file(package, name, max_bytes=2 << 20); total += len(raw)
        _require(total <= 16 << 20, "runtime closure exceeds 16 MiB")
        pins[name] = sha256(raw).hexdigest()
    if budget is not None: budget.check()
    return {"runtime_sha256": pins, "runtime_versions": identity["runtime_versions"]}


def search_moving_contacts(admission, *, max_seconds=60, max_evaluations=24000, max_iterations=6,
                           strategy="shared-slack-v1"):
    """One immutable admitted world; two coupled alternatives, no source, physics or scorer changes."""
    from weft.constrained_limb_camera_fit import solve_source_qp
    from weft.moving_equipment_task import validate_moving_equipment_task
    started = time.monotonic()
    _require(strategy in ("shared-slack-v1", "separate-collision-support-v1"), "unsupported restoration strategy")
    separate = strategy == "separate-collision-support-v1"
    if separate:
        from weft import moving_equipment_contact_restore
    _require(type(max_seconds) in (int,float) and math.isfinite(max_seconds) and 0 < max_seconds <= 60, "deadline must be in (0,60]")
    _require(type(max_evaluations) is int and 1 <= max_evaluations <= 24000, "evaluation bound must be 1..24000")
    _require(type(max_iterations) is int and 1 <= max_iterations <= 6, "iteration bound must be 1..6")
    admission = support._admitted(admission)
    scene, task = admission.scene, admission.task
    validate_moving_equipment_task(task, scene, admission.bundle)
    knots, times = support._schedule(scene, task)
    _require(8*len(knots)+1 <= 518, "complete trajectory exceeds variable bound")
    if separate:
        _require(8*len(knots) <= 192, "complete separate-restoration trajectory exceeds 192 variables")
    budget = support._Budget(started+max_seconds, max_evaluations)
    runtime = _runtime(admission, scene, budget); budget.check()
    rows, fixed, failure, model = [], [], None, None
    try:
        model = _Coupled(scene, task, budget, knots, times)
        fixed = model.fixed_obstructions()
        for side in ("left", "right"):
            row = {"support_side": side, "variables": model.names, "status": "running", "iterations": [], "audit": []}
            rows.append(row)
            if fixed:
                row["status"] = "immutable_geometry_obstruction"
                continue
            correction = np.zeros((len(knots), 8))
            for iteration in range(max_iterations):
                values, jac, labels, hard = model.constraints(correction, side, derivatives=True)
                violation = max(0., float(np.max(-values, initial=0.)))
                if violation <= 1e-10: break
                if separate:
                    step, reports, slack = _separate_proposal(correction, values, jac, labels, hard, budget)
                else:
                    step, reports, slack = _proposal(correction, values, jac, hard, solve_source_qp, budget)
                receipt = {"iteration": iteration+1, "before_worst_scaled_violation": violation,
                    "search_restoration_slack": slack, "solvers": reports, "accepted": False}
                row["iterations"].append(receipt)
                if step is None: break
                # No post-hoc coordinate clamp: original joint constraints enter every native-boundary trial.
                for power in range(LIMITS["backtracks"]):
                    trial = correction + step.reshape(correction.shape)*2.**(-power)
                    candidate, _, _, hard_flags = model.constraints(trial, side, derivatives=False)
                    new_violation = max(0., float(np.max(-candidate, initial=0.)))
                    progress, detail = (_restoration_progress(values, candidate, labels, hard_flags) if separate
                                        else (new_violation < violation-1e-10, None))
                    if (np.max(abs(trial)) <= .5+1e-10 and np.min(candidate[hard_flags], initial=0.) >= -1e-10 and progress):
                        correction = trial
                        receipt.update(accepted=True, fraction=2.**(-power), after_worst_scaled_violation=new_violation)
                        if separate: receipt["restoration_progress"] = detail
                        break
                if not receipt["accepted"]: break
            # Final native distances/static diagnostics are independently recomputed, not copied from the QP model.
            row["knots"] = [{"time_s": stamp, "delta_rad": values.tolist()} for stamp, values in zip(knots, correction)]
            for stamp, weights in model.rows:
                delta = sum(weight*correction[index] for index, weight in weights)
                row["audit"].append(model.geometry.audit(stamp, model.indices, model.supports[side], delta))
            values, _, labels, hard = model.constraints(correction, side, derivatives=False)
            row["constraint_failures"] = [{**label, "residual": float(value)} for value, label in zip(values,labels) if value < -1e-10]
            geometric = all(all(passed for key,passed in x["checks"].items() if not key.startswith("static_")) for x in row["audit"])
            geometric = geometric and not row["constraint_failures"]
            unsupported = separate and any(s["status"] == "unsupported_hard_anchor" for i in row["iterations"] for s in i["solvers"])
            row["status"] = "unsupported_hard_anchor" if unsupported else "geometric_witness_only" if geometric else "no_witness_in_bounded_coupled_search"
            row["static_load_hypothesis_passed"] = geometric and all(x["geometric_static_witness"] for x in row["audit"])
            row["dynamic_contact_verified"] = False
            row["task_success"] = False
    except TimeoutError as error:
        failure = str(error)
        if rows and rows[-1]["status"] == "running": rows[-1]["status"] = "incomplete_resource_bound"
    finally:
        if model is not None: model.close()
    runtime_verified = False
    try:
        _require(_runtime(admission, scene, budget) == runtime, "runtime changed during contact search")
        runtime_verified = True
    except TimeoutError as error:
        failure = str(error)
    result = {"schema_version": 1, "profile": PROFILE, "status": "incomplete_resource_bound" if failure else "complete",
        "failure": failure, "input_pins": {"task_manifest_sha256": admission.manifest_sha256,
            "parent_manifest_sha256": admission.parent_manifest_sha256, "scene_sha256": canonical_hash(scene),
            "task_sha256": canonical_hash(task), "bundle_sha256": canonical_hash(admission.bundle)},
        "runtime": runtime, "runtime_readback_complete":runtime_verified,
        "limits": dict(LIMITS), "requested_bounds": {"seconds":max_seconds,"evaluations":max_evaluations,"iterations":max_iterations},
        "schedule": {"control_knots_s":knots,"native_boundary_times_s":times,"subsampling":False},
        "fixed_obstructions":fixed,"alternatives":rows,"unattempted_alternatives":2-len(rows),
        "physics_steps":0,"model_compilations":int(model is not None),"native_forward_evaluations":budget.used,
        "seconds":time.monotonic()-started,"scope":SCOPE,
        "qualification":{"task_success":False,"dynamic_contact_verified":False,"contact_forces_measured":False,
            "approach_transition_checked":False,"full_interval_score_checked":False,
            "static_load_scope":"Restrictive motor-only load estimate; floor load sharing is not solved. Failure is not actuator infeasibility; passing is not an equilibrium certificate.",
            "identity_scope":"Original conditional scene/shape/depth/unknown prefix retained; no recovered grasp claim."}}
    if separate:
        result["restoration_strategy"] = {"profile": strategy,
            "helper_profile": moving_equipment_contact_restore.PROFILE,
            "helper_sha256": runtime["runtime_sha256"]["moving_equipment_contact_restore.py"],
            "scope": "Collision-first linear proposal and nonlinear progress only; original final native gates unchanged."}
    _bytes(result)
    try: budget.check()
    except TimeoutError as error:
        result.update(status="incomplete_resource_bound", failure=str(error))
    result["seconds"] = time.monotonic()-started
    return result
