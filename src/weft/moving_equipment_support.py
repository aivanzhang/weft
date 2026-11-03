"""Bounded zero-step support hypotheses; geometry and static loads are not task success."""

from hashlib import sha256
import json
import math
from pathlib import Path
import time
from types import MappingProxyType

from weft.io import canonical_hash

PROFILE = "moving-equipment-support-screen-v1"
LIMITS = MappingProxyType({"maximum_seconds": 60, "maximum_evaluations": 6000, "maximum_samples": 256,
    "maximum_output_bytes": 4 << 20, "iterations": 8, "backtracks": 4,
    "angle_correction_rad": .5, "angle_update_rad": .15, "support_gap_world": 1e-4,
    "minimum_upward_normal": .9, "overlap_tolerance_world": 1e-8})
SCOPE = ("Restricted arm-only zero-step geometric and static-load hypotheses. Not dynamic contact, "
    "trajectory reachability, measured grasp, full-interval feasibility or task success. No-solution means only this bounded search found no witness.")


def _require(value, message):
    if not value:
        raise ValueError("Moving support: " + message)


def _encoded(value):
    output = bytearray()
    for text in json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False).iterencode(value):
        block = text.encode()
        _require(len(output) + len(block) <= LIMITS["maximum_output_bytes"], "output byte bound exceeded")
        output.extend(block)
    return bytes(output)


def _admitted(value):
    from weft.moving_equipment_task_lab import validate_task_lab_admission
    return validate_task_lab_admission(value)


def _schedule(scene, task):
    from weft.native_interval import elapsed_substeps
    horizon, substeps = task["interval"]["horizon_steps"], scene["substeps"]
    duration = float(elapsed_substeps(scene, horizon * substeps))
    start = duration - task["goal"]["terminal_consecutive_hold_s"]
    _require(0 <= start < duration <= 60 and horizon * substeps <= 28800, "unsupported complete task interval")
    # All terminal boundaries are retained; a long hold refuses instead of silently thinning the clock.
    audit = sorted({start, duration, *(float(elapsed_substeps(scene, i)) for i in range(horizon * substeps + 1)
                                     if start <= elapsed_substeps(scene, i) <= duration)})
    _require(len(audit) <= LIMITS["maximum_samples"], "terminal native boundary inventory exceeds 256; no subsampling performed")
    knots = sorted({start, duration, *(float(elapsed_substeps(scene, i * substeps)) for i in range(horizon + 1)
                                     if start <= elapsed_substeps(scene, i * substeps) <= duration)})
    return knots, audit


def force_deficit(scene, task):
    """Continuous free-flight necessity only; the discrete task scorer remains authoritative."""
    goal = task["goal"]
    objects = [obj for obj in scene["objects"] if obj["id"] == goal["object_id"]]
    _require(len(objects) == 1 and objects[0]["shape"] == "sphere" and objects[0]["role"] != "static",
             "one selected dynamic sphere required")
    _require(scene["gravity"][:2] == [0., 0.] and scene["gravity"][2] < 0, "vertical authored gravity required")
    mass = objects[0]["mass"]
    deficit = max(0., -scene["gravity"][2] - scene["task"]["object_goal"]["virtual_acceleration"])
    return {"steady_upward_force_model_units": mass * deficit,
        "continuous_window_mean_upward_force_lower_bound_model_units": mass * max(0., deficit - 2 * goal["speed_tolerance_world_s"] / goal["terminal_consecutive_hold_s"]),
        "unsupported_minimum_vertical_speed_change_world_s": deficit * goal["terminal_consecutive_hold_s"],
        "maximum_endpoint_speed_difference_world_s": 2 * goal["speed_tolerance_world_s"],
        "scope": "Necessary continuous free-flight bound, not an exact discrete scorer certificate or calibrated physical force."}


class _Budget:
    def __init__(self, deadline, maximum):
        self.deadline, self.maximum, self.used = deadline, maximum, 0

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Moving support aggregate deadline reached")

    def forward(self, model, data):
        import mujoco
        self.check()
        if self.used >= self.maximum:
            raise TimeoutError("Moving support native evaluation allowance reached")
        self.used += 1
        mujoco.mj_forward(model, data)


class _Geometry:
    def __init__(self, scene, task, budget):
        from weft.shared_world import SharedWorldEnv
        _require(len(scene["objects"]) == 1 and scene["objects"][0]["shape"] == "sphere",
                 "initial screen requires one sphere; no other collision bodies may be omitted")
        self.env = SharedWorldEnv(scene)
        try:
            self._initialize(scene, task, budget)
        except BaseException:
            self.env.close()
            raise

    def _initialize(self, scene, task, budget):
        import mujoco
        import numpy as np
        self.model, self.data = self.env.model, mujoco.MjData(self.env.model)
        self.scene, self.task, self.budget = scene, task, budget
        self.target = np.asarray(task["goal"]["position_world"], dtype=float)
        self.object = next(obj for obj in scene["objects"] if obj["id"] == task["goal"]["object_id"])
        _require(len(scene["objects"]) == 1 and self.object["shape"] == "sphere", "initial screen requires one sphere; no other collision bodies may be omitted")
        self.ogid = self.env._object_geometry_ids[self.object["id"]][0]
        joint = self.model.joint(self.object["id"] + "_joint").id
        self.oq = int(self.model.jnt_qposadr[joint])
        self.human_ids = sorted(self.env._human_geom_to_name)
        _require(len(self.human_ids) <= 64 and self.env.human_motor_count == 22,
                 "bounded original spatial body geometry required")
        self.baseline = np.array(self.model.qpos0, copy=True)
        self.baseline[self.oq:self.oq+3] = self.target
        self.baseline[self.oq+3:self.oq+7] = self.object["quaternion"]
        self.force = force_deficit(scene, task)["steady_upward_force_model_units"]

    def close(self): self.env.close()

    def arm(self, side):
        import numpy as np
        names = [side + suffix for suffix in ("_shoulder_twist", "_shoulder_lateral", "_shoulder", "_elbow")]
        indices = np.array([self.env.human_action_names.index(name) for name in names])
        gid = self.model.geom(side + "_elbow_geom").id
        return names, indices, gid

    def pose(self, stamp, indices, delta):
        import numpy as np
        target, _ = self.env._target_at(stamp)
        self.data.qpos[:] = self.baseline
        self.data.qpos[self.env._q] = target
        self.data.qpos[self.env._q[indices]] += delta
        self.data.qvel[:] = 0
        self.data.ctrl[:] = 0
        self.data.qfrc_applied[:] = 0
        self.data.xfrc_applied[:] = 0
        self.data.time = stamp
        self.budget.forward(self.model, self.data)
        _require(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qfrc_bias).all(), "nonfinite scratch geometry")
        return target

    def residual(self, gid, indices, fraction, derivatives=False):
        import mujoco
        import numpy as np
        _require(int(self.model.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE), "exact forearm capsule required")
        half, radius = float(self.model.geom_size[gid, 1]), float(self.model.geom_size[gid, 0])
        axis = self.data.geom_xmat[gid].reshape(3, 3)[:, 2].copy()
        point = self.data.geom_xpos[gid] + (2*fraction-1)*half*axis
        desired = self.target - np.array([0., 0., self.object["radius"] + radius])
        error = point - desired
        if fraction == .5:
            error = np.r_[error, half * axis[2]]
        if not derivatives:
            return error
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jp, jr, point, int(self.model.geom_bodyid[gid]))
        columns = self.env._v[indices]
        jac = jp[:, columns]
        if fraction == .5:
            jac = np.vstack((jac, half * np.cross(jr[:, columns].T, axis)[:, 2]))
        return error, jac

    def solve(self, stamp, indices, gid, fraction, initial):
        import numpy as np
        reference, _ = self.env._target_at(stamp)
        ranges = self.model.jnt_range[np.asarray(self.env._human_joint_ids)[indices]]
        low = np.maximum(-LIMITS["angle_correction_rad"], ranges[:, 0] - reference[indices])
        high = np.minimum(LIMITS["angle_correction_rad"], ranges[:, 1] - reference[indices])
        _require(np.all(low <= high), "original reference is outside diagnostic joint/correction bounds")
        delta = np.clip(initial, low, high)
        for _ in range(LIMITS["iterations"]):
            self.pose(stamp, indices, delta)
            error, jac = self.residual(gid, indices, fraction, True)
            if np.linalg.norm(error) <= 1e-5: break
            update = np.linalg.solve(jac.T @ jac + .0025*np.eye(4), -jac.T @ error - .0001*delta)
            update = np.clip(update, -LIMITS["angle_update_rad"], LIMITS["angle_update_rad"])
            accepted = False
            for power in range(LIMITS["backtracks"]):
                trial = np.clip(delta + update * 2.**(-power), low, high)
                self.pose(stamp, indices, trial)
                if np.linalg.norm(self.residual(gid, indices, fraction)) < np.linalg.norm(error):
                    delta, accepted = trial, True
                    break
            if not accepted: break
        return delta

    def audit(self, stamp, indices, gid, delta):
        import mujoco
        import numpy as np
        reference = self.pose(stamp, indices, delta)
        distances = {self.env._human_geom_to_name[x]: float(mujoco.mj_geomDistance(self.model, self.data, x, self.ogid, 100., None)) for x in self.human_ids}
        gap = distances[self.env._human_geom_to_name[gid]]
        center, half = self.data.geom_xpos[gid], float(self.model.geom_size[gid, 1])
        axis = self.data.geom_xmat[gid].reshape(3, 3)[:, 2]
        point = center + np.clip(np.dot(self.target-center, axis), -half, half) * axis
        direction = self.target - point
        normal = direction / max(float(np.linalg.norm(direction)), 1e-12)
        normal_z = float(normal[2])
        contact = point + float(self.model.geom_size[gid, 0]) * normal
        jp = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jp, None, contact, int(self.model.geom_bodyid[gid]))
        reaction = normal * self.force / normal_z if normal_z > 1e-8 else np.zeros(3)
        torque = self.data.qfrc_bias[self.env._v] + jp[:, self.env._v].T @ reaction
        object_force = -np.asarray(self.scene["gravity"]) * self.object["mass"] - reaction
        object_limit = self.object["mass"] * self.scene["task"]["object_goal"]["virtual_acceleration"]
        object_floor = float(mujoco.mj_geomDistance(self.model, self.data, self.ogid, self.env._floor_geom, 100., None))
        radius = self.object["radius"]
        containment = bool(np.all(self.target-radius >= -LIMITS["overlap_tolerance_world"])
            and np.all(self.target+radius <= np.asarray(self.scene["bounds"])+LIMITS["overlap_tolerance_world"]))
        floor = max(0., -min(float(mujoco.mj_geomDistance(self.model, self.data, x, self.env._floor_geom, 100., None)) for x in self.human_ids)) / self.env.scale
        angular = float(np.sqrt(np.mean((self.data.qpos[self.env._q[:19]]-reference[:19])**2)))
        ranges = self.model.jnt_range[np.asarray(self.env._human_joint_ids)]
        values = self.data.qpos[self.env._q]
        joint_bounds = bool(np.all(values >= ranges[:, 0]-1e-10) and np.all(values <= ranges[:, 1]+1e-10))
        flags = {"support_gap": -LIMITS["overlap_tolerance_world"] <= gap <= LIMITS["support_gap_world"],
            "upward_normal": normal_z >= LIMITS["minimum_upward_normal"],
            "no_other_object_overlap": all(value >= -LIMITS["overlap_tolerance_world"] for name, value in distances.items() if name != self.env._human_geom_to_name[gid]),
            "floor": floor <= self.task["body_gates"]["max_floor_penetration_model_units"],
            "reference_pointwise_conservative": angular <= self.task["body_gates"]["angular_rmse_rad"],
            "reference_peak": angular <= self.task["body_gates"]["peak_angular_rmse_rad"],
            "joint_bounds": joint_bounds, "correction_bound": bool(np.max(abs(delta)) <= LIMITS["angle_correction_rad"]+1e-10),
            "object_floor_clearance": object_floor >= -LIMITS["overlap_tolerance_world"],
            "object_world_containment": containment,
            "static_human_motor_capacity": bool(np.all(abs(torque) <= self.env.human_limits+1e-8)),
            "static_object_force_capacity": bool(np.all(abs(object_force) <= object_limit+1e-8))}
        return {"time_s": stamp, "delta_rad": delta.tolist(), "support_gap_world": gap,
            "upward_normal": normal_z, "floor_penetration_model_units": floor, "angular_rmse_rad": angular,
            "object_floor_clearance_world": object_floor,
            "human_static_capacity_ratio": float(np.max(abs(torque)/self.env.human_limits)),
            "object_static_capacity_ratio": float(np.max(abs(object_force)/object_limit)),
            "static_reaction_world": reaction.tolist(), "checks": flags,
            "geometric_static_witness": all(flags.values())}


def diagnose_moving_support(admission, *, max_seconds=60, max_evaluations=6000):
    """One unchanged model, six explicit restricted hypotheses, no physics step or installed controller."""
    import numpy as np
    from weft.moving_equipment_task import validate_moving_equipment_task
    from weft.moving_equipment_task_recording import runtime_identity
    from weft.shared_world_packaging import read_package_file
    started = time.monotonic()
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 60,
             "finite positive deadline up to 60 seconds required")
    _require(type(max_evaluations) is int and 1 <= max_evaluations <= 6000, "evaluation bound must be 1..6000")
    admission = _admitted(admission)
    scene, task = admission.scene, admission.task
    validate_moving_equipment_task(task, scene, admission.bundle)
    knots, times = _schedule(scene, task)
    force = force_deficit(scene, task)
    budget = _Budget(started + max_seconds, max_evaluations)
    runtime = runtime_identity(scene)
    _require(all(admission.runtime_sha256.get(name) == pin for name, pin in runtime["runtime_sha256"].items()),
             "source-admitted task runtime changed")
    own_raw = read_package_file(Path(__file__).parent, Path(__file__).name, max_bytes=2 << 20)
    pins = {"task_manifest_sha256": admission.manifest_sha256, "parent_manifest_sha256": admission.parent_manifest_sha256,
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "bundle_sha256": canonical_hash(admission.bundle),
        "helper_sha256": sha256(own_raw).hexdigest()}
    budget.check()
    geometry, rows, failure = None, [], None
    try:
        geometry = _Geometry(scene, task, budget)
        for side in ("left", "right"):
            names, indices, gid = geometry.arm(side)
            for fraction in (0., .5, 1.):
                row = {"id": f"{side}:{fraction:g}", "side": side, "forearm_point_fraction": fraction,
                    "variables": names, "status": "running", "knots": [], "audit": []}
                rows.append(row)
                delta = np.zeros(4)
                for stamp in knots:
                    delta = geometry.solve(stamp, indices, gid, fraction, delta)
                    row["knots"].append({"time_s": stamp, "delta_rad": delta.tolist()})
                corrections = np.asarray([x["delta_rad"] for x in row["knots"]])
                for stamp in times:
                    delta = np.array([np.interp(stamp, knots, corrections[:, k]) for k in range(4)])
                    row["audit"].append(geometry.audit(stamp, indices, gid, delta))
                row["status"] = "geometric_static_witness_only" if all(x["geometric_static_witness"] for x in row["audit"]) else "no_witness_in_restricted_search"
                row["failed_checks"] = sorted({key for x in row["audit"] for key, passed in x["checks"].items() if not passed})
    except TimeoutError as error:
        failure = str(error)
        if rows and rows[-1]["status"] == "running": rows[-1]["status"] = "incomplete_resource_bound"
    finally:
        if geometry is not None: geometry.close()
    _require(runtime_identity(scene) == runtime and read_package_file(Path(__file__).parent, Path(__file__).name, max_bytes=2 << 20) == own_raw,
             "runtime changed during support diagnosis")
    result = {"schema_version": 1, "profile": PROFILE, "status": "incomplete_resource_bound" if failure else "complete",
        "failure": failure, "input_pins": pins, "runtime": runtime, "limits": dict(LIMITS), "force_deficit": force,
        "schedule": {"window_start_s": times[0], "window_end_s": times[-1], "native_boundary_times_s": times,
            "control_knots_s": knots, "subsampling": False}, "hypotheses": rows,
        "unattempted_hypotheses": 6-len(rows), "physics_steps": 0, "model_compilations": int(geometry is not None),
        "native_forward_evaluations": budget.used, "seconds": time.monotonic()-started, "scope": SCOPE,
        "qualification": {"contact_force_measured": False, "dynamic_contact_verified": False, "task_success": False,
            "transition_into_window_checked": False, "full_interval_body_score_checked": False,
            "static_load_assumptions": "Restrictive motor-only static hypothesis: zero velocity/acceleration, one frictionless upward forearm normal and original clipped object forces. Floor load sharing is not solved; a motor-capacity failure does not prove actuator infeasibility, and passing is not an equilibrium certificate.",
            "reference_gate_scope": "Pointwise angular mean gate is a stricter geometric search prior; it is not a simulated time-averaged task score.",
            "scope_of_failure": "Six restricted local searches, not global impossibility."}}
    _encoded(result)
    return result
