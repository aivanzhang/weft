"""Two-forearm near-contact hypotheses; static witnesses are not dynamic support."""

import math
import time

import numpy as np

from weft import moving_equipment_contact_search as coupled
from weft import moving_equipment_contact_restore
from weft import moving_equipment_support as support
from weft.contact_force_balance import solve_contact_force_balance
from weft.io import canonical_hash

PROFILE = "moving-equipment-bilateral-contact-v1"
SCOPE = ("One explicit two-forearm sliding-friction hypothesis over the complete terminal native window. "
    "Near-contact gaps are not measured or necessarily active contacts. Static object balance and supplied "
    "motor rows do not establish full-body equilibrium, approach feasibility, dynamic grasp or task success.")
_require = coupled._require


def _bilateral_rows(values, matrix, labels, hard):
    """Reuse every original row except the replaced single upward-normal assumption."""
    kept = [i for i, row in enumerate(labels) if row["kind"] != "upward_normal_scaled"]
    right = [i for i, row in enumerate(labels) if row["kind"] == "nonpenetration:right_elbow_geom"]
    left = [i for i, row in enumerate(labels) if row["kind"] == "support_gap"]
    _require(len(right) == len(left) and [labels[i]["time_s"] for i in right] ==
             [labels[i]["time_s"] for i in left], "complete matching two-contact rows required")
    left_set = set(left)
    result_labels = [{**labels[i], **({"side": "left"} if i in left_set else {})} for i in kept]
    result_labels += [{"kind": "support_gap", "time_s": labels[i]["time_s"], "side": "right"} for i in right]
    result = np.r_[values[kept], 1e-4 + 1e-8 - values[right]]
    jac = None if matrix is None else np.vstack((matrix[kept], -matrix[right]))
    return result, jac, result_labels, np.r_[hard[kept], np.zeros(len(right), dtype=bool)]


def _sliding_friction(model, first, second):
    # Explicit pairs can carry anisotropy; this first adapter refuses rather than guessing their frame.
    _require(model.npair == 0, "explicit contact pairs are unsupported by this geometry adapter")
    _require(min(int(model.geom_condim[first]), int(model.geom_condim[second])) >= 3,
             "both geometries must support sliding friction")
    a, b = int(model.geom_priority[first]), int(model.geom_priority[second])
    friction = (model.geom_friction[first] if a > b else model.geom_friction[second] if b > a
                else np.maximum(model.geom_friction[first], model.geom_friction[second]))
    return [float(friction[0]), float(friction[0])]


class _Bilateral(coupled._Coupled):
    def __init__(self, scene, task, budget, knots, times):
        super().__init__(scene, task, budget, knots, times)
        self.force_pins = {"source_sha256": scene["source_binding"]["source_sha256"],
                           "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task)}

    def _model_identity(self):
        names = ("geom_priority", "geom_condim", "geom_solmix", "pair_geom1", "pair_geom2", "pair_friction")
        return canonical_hash({"original": super()._model_identity(),
            **{name: getattr(self.model, name).tolist() for name in names}})

    def constraints(self, correction, *, derivatives):
        return _bilateral_rows(*super().constraints(correction, "left", derivatives=derivatives))

    def audit(self, stamp, delta):
        original = {side: self.geometry.audit(stamp, self.indices, self.supports[side], delta)
                    for side in ("left", "right")}
        checks = {side: {key: value for key, value in audit["checks"].items()
                         if key != "upward_normal" and not key.startswith("static_")}
                  for side, audit in original.items()}
        return {"time_s": stamp, "delta_rad": delta.tolist(), "original_single_contact_diagnostics": original,
                "checks": checks, "geometric_passed": all(all(row.values()) for row in checks.values()),
                "force_balance": None}

    def force_hypothesis(self, stamp, delta, normal_force_cap):
        import mujoco

        self.geometry.pose(stamp, self.indices, delta)
        g, env = self.geometry, self.env
        contacts, matrices, surfaces = [], [], []
        for side in ("left", "right"):
            gid = self.supports[side]
            state = coupled._capsule_state(self.model, self.data, gid, self.columns, g.target)
            normal = state["normal"]
            gap = state["distance"] - state["radius"] - g.object["radius"]
            _require(-1e-8 <= gap <= 1e-4, "force hypothesis requires both original contact-gap gates")
            human_point = state["point"] + state["radius"] * normal
            object_point = g.target - g.object["radius"] * normal
            tangent = np.eye(3)[int(np.argmin(abs(normal)))]
            tangent = tangent - normal * np.dot(normal, tangent)
            tangent /= np.linalg.norm(tangent)
            jp = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, self.data, jp, None, human_point, int(self.model.geom_bodyid[gid]))
            # Object receives +f, human receives -f; static motor compensation is bias + Jᵀf.
            matrices.append(jp[:, env._v].T)
            contacts.append({"id": side + "_forearm", "position_world": object_point.tolist(),
                "normal_on_object_world": normal.tolist(), "tangent_world": tangent.tolist(),
                "sliding_friction": _sliding_friction(self.model, gid, g.ogid),
                "normal_force_max": normal_force_cap})
            surfaces.append({"side": side, "gap_world": float(gap), "human_surface_world": human_point.tolist(),
                "object_surface_world": object_point.tolist(),
                "scope": "Native capsule/sphere closest surfaces bridged within the original near-contact tolerance; not observed contact."})
        com = self.data.xipos[int(self.model.geom_bodyid[g.ogid])].copy()
        _require(np.max(abs(com - g.target)) <= 1e-10, "sphere COM differs from the fixed object target")
        cap = g.object["mass"] * g.scene["task"]["object_goal"]["virtual_acceleration"]
        return {"schema_version": 1, "profile": "contact-force-balance-v1",
            "provenance": dict(self.force_pins),
            "contact_model": "point_sliding", "center_of_mass_world": com.tolist(),
            "mass": g.object["mass"], "gravity_world": list(g.scene["gravity"]),
            "external_force_world": [0., 0., 0.], "external_moment_world": [0., 0., 0.],
            "direct_force_max_world": [float(cap)] * 3, "contacts": contacts,
            "motor_limits": {"matrix": np.hstack(matrices).tolist(),
                "bias": self.data.qfrc_bias[env._v].tolist(), "lower": (-env.human_limits).tolist(),
                "upper": env.human_limits.tolist()}}, surfaces


def search_bilateral_contacts(admission, *, normal_force_cap, max_seconds=60, max_evaluations=24000, max_iterations=6):
    """Search one new contact family without modifying the source, task, model or old family."""
    from weft.moving_equipment_task import validate_moving_equipment_task

    started = time.monotonic()
    _require(type(normal_force_cap) in (int, float) and 0 <= normal_force_cap <= 1e4
             and math.isfinite(normal_force_cap), "explicit normal-force search cap must be within0..10000")
    _require(type(max_seconds) in (int, float) and 0 < max_seconds <= 60 and math.isfinite(max_seconds), "deadline must be in(0,60]")
    _require(type(max_evaluations) is int and 1 <= max_evaluations <= 24000, "evaluation bound must be1..24000")
    _require(type(max_iterations) is int and 1 <= max_iterations <= 6, "iteration bound must be1..6")
    admission = support._admitted(admission)
    scene, task = admission.scene, admission.task
    validate_moving_equipment_task(task, scene, admission.bundle)
    knots, times = support._schedule(scene, task)
    _require(8 * len(knots) <= 192, "complete restoration trajectory exceeds192variables")
    budget = support._Budget(started + max_seconds, max_evaluations)
    runtime = coupled._runtime(admission, scene, budget)
    model, failure, fixed = None, None, []
    row = {"status": "not_attempted", "iterations": [], "audit": [], "force_witnesses": [],
        "task_success": False, "dynamic_contact_verified": False}
    try:
        budget.check()
        model = _Bilateral(scene, task, budget, knots, times)
        fixed = model.fixed_obstructions()
        row.update(status="running", variables=model.names)
        if fixed:
            row["status"] = "immutable_geometry_obstruction"
        else:
            correction = np.zeros((len(knots), 8))
            for iteration in range(max_iterations):
                values, jac, labels, hard = model.constraints(correction, derivatives=True)
                violation = max(0., float(np.max(-values, initial=0.)))
                if violation <= 1e-10:
                    break
                step, reports, _ = coupled._separate_proposal(correction, values, jac, labels, hard, budget)
                receipt = {"iteration": iteration + 1, "before_worst_violation": violation,
                           "solvers": reports, "accepted": False}
                row["iterations"].append(receipt)
                if step is None:
                    break
                for power in range(coupled.LIMITS["backtracks"]):
                    trial = correction + step.reshape(correction.shape) * 2. ** (-power)
                    candidate, _, _, flags = model.constraints(trial, derivatives=False)
                    progress, detail = coupled._restoration_progress(values, candidate, labels, flags)
                    if np.max(abs(trial)) <= .5 + 1e-10 and np.min(candidate[flags], initial=0.) >= -1e-10 and progress:
                        correction = trial
                        receipt.update(accepted=True, fraction=2. ** (-power), restoration_progress=detail)
                        break
                if not receipt["accepted"]:
                    break
            row["knots"] = [{"time_s": stamp, "delta_rad": value.tolist()} for stamp, value in zip(knots, correction)]
            for stamp, weights in model.rows:
                delta = sum(weight * correction[index] for index, weight in weights)
                row["audit"].append(model.audit(stamp, delta))
            values, _, labels, _ = model.constraints(correction, derivatives=False)
            row["constraint_failures"] = [{**label, "residual": float(value)} for value, label in zip(values, labels) if value < -1e-10]
            geometry = not row["constraint_failures"] and all(x["geometric_passed"] for x in row["audit"])
            unsupported = any(s["status"] == "unsupported_hard_anchor" for i in row["iterations"] for s in i["solvers"])
            row["status"] = "unsupported_hard_anchor" if unsupported else "geometric_witness_only" if geometry else "no_witness_in_bounded_search"
            if geometry:
                for audit in row["audit"]:
                    budget.check()
                    hypothesis, surfaces = model.force_hypothesis(audit["time_s"], np.asarray(audit["delta_rad"]), float(normal_force_cap))
                    budget.check()
                    remaining = min(2., budget.deadline - time.monotonic())
                    if remaining < .001:
                        raise TimeoutError("No remaining force-witness budget")
                    witness = solve_contact_force_balance(hypothesis, max_seconds=remaining)
                    audit["force_balance"] = {"hypothesis": hypothesis, "surfaces": surfaces, "result": witness}
                    row["force_witnesses"].append({"time_s": audit["time_s"], "status": witness["status"]})
                    budget.check()
                row["status"] = ("static_two_contact_witness_only" if all(x["status"] == "feasible" for x in row["force_witnesses"])
                                 else "geometry_only_static_force_refused")
    except TimeoutError as error:
        failure = str(error)
        row["status"] = "incomplete_resource_bound"
    except ValueError as error:
        failure = str(error)
        row["status"] = "unsupported_input"
    finally:
        if model is not None:
            model.close()
    verified = False
    try:
        _require(coupled._runtime(admission, scene, budget) == runtime, "runtime changed during bilateral search")
        verified = True
    except TimeoutError as error:
        failure = str(error)
    result = {"schema_version": 1, "profile": PROFILE,
        "status": "incomplete_resource_bound" if failure and row["status"] != "unsupported_input" else "unsupported_input" if failure else "complete",
        "failure": failure, "input_pins": {"task_manifest_sha256": admission.manifest_sha256,
            "parent_manifest_sha256": admission.parent_manifest_sha256, "scene_sha256": canonical_hash(scene),
            "task_sha256": canonical_hash(task), "bundle_sha256": canonical_hash(admission.bundle)},
        "runtime": runtime, "runtime_readback_complete": verified,
        "limits": dict(coupled.LIMITS), "requested_bounds": {"seconds": max_seconds, "evaluations": max_evaluations, "iterations": max_iterations},
        "normal_force_cap": {"per_contact": normal_force_cap, "scope": "Explicit search prior, not measured actuator/contact capacity."},
        "schedule": {"control_knots_s": knots, "native_boundary_times_s": times, "subsampling": False},
        "fixed_obstructions": fixed, "hypothesis": row, "physics_steps": 0,
        "model_compilations": int(model is not None), "native_forward_evaluations": budget.used,
        "qualification": {"source_authenticated_by_admission": True, "dynamic_contact_verified": False,
            "task_success": False, "approach_transition_checked": False, "full_interval_score_checked": False,
            "global_infeasibility_claim": False, "contact_forces_measured": False},
        "scope": SCOPE, "seconds": time.monotonic() - started}
    coupled._bytes(result)
    try:
        budget.check()
    except TimeoutError as error:
        result.update(status="incomplete_resource_bound", failure=str(error))
    result["seconds"] = time.monotonic() - started
    return result
