"""Authored mobile support and floor dynamics, not recovered balance or contact forces."""

from copy import deepcopy
import math
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from weft.errors import UnsupportedVideo
from weft.root_motion import ROOT_MOTION_CONTRACT, build_root_reference


ROOT_AXES = ("root_x", "root_y", "root_z")
MOBILITY_DEFAULTS = {"schema_version": 1, "base_height": 1., "root_mass": .1,
                     "slide_limits": [-5., 5.], "floor_height": 0., "clearance": .005,
                     "kp_per_mass": 60., "kd_per_mass": 12., "force_limit_weight_multiple": 3.,
                     "maximum_force": 2000., "push_acceleration": 6.}
GROUND_SCOPE = "authored_camera_relative_plane_and_normal_aligned_gravity_not_measured"


def _finite(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"Invalid mobile {name}")


def _vector(value, name, bound):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Mobile {name} requires three coordinates")
    for item in value:
        _finite(item, name, -bound, bound)


def _validate_ground_plane(plane, source_sha256):
    fields = {"schema_version", "normal", "method", "source_sha256", "evidence_ids", "provenance", "gravity_assumption"}
    if not isinstance(plane, dict) or set(plane) != fields:
        raise ValueError("Ground plane fields differ from schema")
    if type(plane["schema_version"]) is not int or plane["schema_version"] != 1:
        raise ValueError("Unsupported ground plane schema")
    _vector(plane["normal"], "ground plane normal", 1)
    if not math.isclose(math.sqrt(sum(x * x for x in plane["normal"])), 1., rel_tol=0, abs_tol=1e-6) or plane["normal"][2] < .25:
        raise ValueError("Ground plane requires an upward unit normal with nz >= .25")
    if (plane["method"] != "camera_relative_support_plane_v1"
            or plane["gravity_assumption"] != "authored_normal_alignment_not_measured"):
        raise ValueError("Ground plane must disclose its authored gravity hypothesis")
    if (not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256)
            or plane["source_sha256"] != source_sha256):
        raise ValueError("Ground plane source must match the model source")
    ids = plane["evidence_ids"]
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 1800
            or any(not isinstance(item, str) or not re.fullmatch(source_sha256 + r":frame:[0-9]{1,8}", item) for item in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Ground plane requires bounded unique source frame evidence IDs")
    if not isinstance(plane["provenance"], str) or not 1 <= len(plane["provenance"].strip()) <= 2000:
        raise ValueError("Ground plane requires explicit bounded provenance")


def ground_normal(spec):
    """Canonical unit normal; coordinates and root slide axes remain camera-relative."""
    normal = np.array(spec["mobility"].get("ground_plane", {}).get("normal", [0., 0., 1.]), dtype=float)
    return normal / np.linalg.norm(normal)


def validate_mobility(spec):
    if not isinstance(spec, dict) or spec.get("profile") != "exercise-v3" or type(spec.get("schema_version")) is not int or spec["schema_version"] != 3:
        raise ValueError("Unsupported mobile human schema/profile")
    camera_world = 'camera_reference' in spec
    root_limit = 5.
    if camera_world:
        from weft.camera_body import ROOT_LIMIT, validate_camera_binding
        validate_camera_binding(spec)
        root_limit = ROOT_LIMIT
    mobility = spec.get("mobility")
    if not isinstance(mobility, dict) or not set(MOBILITY_DEFAULTS) <= set(mobility) or set(mobility) - set(MOBILITY_DEFAULTS) - {"floor_fit", "ground_plane"}:
        raise ValueError("Mobile support fields differ from schema")
    if type(mobility["schema_version"]) is not int or mobility["schema_version"] != 1:
        raise ValueError("Unsupported mobile support schema")
    if "ground_plane" in mobility:
        _validate_ground_plane(mobility["ground_plane"], spec.get("source_sha256"))
    for name, low, high in (("base_height", .01, 5), ("root_mass", .01, 5), ("floor_height", 0, 0),
                           ("clearance", .001, .02), ("kp_per_mass", 1, 100), ("kd_per_mass", 1, 30),
                           ("force_limit_weight_multiple", .1, 3), ("maximum_force", 1, 2000),
                           ("push_acceleration", 0, 6)):
        _finite(mobility[name], name, low, high)
    if (not isinstance(mobility["slide_limits"], list) or len(mobility["slide_limits"]) != 2
            or any(type(value) not in (int, float) for value in mobility["slide_limits"])
            or mobility["slide_limits"] != [-root_limit, root_limit]):
        raise ValueError("Mobile slide limits differ from the explicit coordinate contract")
    roots, angles = spec.get("root_reference"), spec.get("reference")
    if not isinstance(roots, list) or not isinstance(angles, list) or not 2 <= len(roots) <= 1800 or len(roots) != len(angles):
        raise ValueError("Mobile root references must align with every angular reference")
    previous = -1.
    for root, angular in zip(roots, angles):
        if not isinstance(root, dict) or set(root) != {"time_s", "translation"} or not isinstance(angular, dict):
            raise ValueError("Mobile reference requires time and translation")
        _finite(root["time_s"], "reference time", 0, 60)
        if root["time_s"] <= previous or root["time_s"] != angular.get("time_s"):
            raise ValueError("Mobile root and angular times must match exactly and increase")
        if previous >= 0 and root["time_s"] - previous > ROOT_MOTION_CONTRACT["maximum_reference_gap_seconds"] + 1e-9:
            raise ValueError("Mobile root reference gap exceeds the bounded interpolation horizon")
        _vector(root["translation"], "reference translation", root_limit)
        if not camera_world and root["translation"][1] != 0:
            raise ValueError("Mobile reference depth must remain authored zero, not inferred from video")
        previous = root["time_s"]
    if roots[0]["translation"] != [0., 0., 0.]:
        raise ValueError("Mobile first root reference must preserve the zero displacement origin")
    if "floor_fit" in mobility:
        fit = mobility["floor_fit"]
        method = "first_pose_inclined_plane_clearance_v1" if "ground_plane" in mobility else "first_pose_collision_surface_clearance_v1"
        if (not isinstance(fit, dict) or fit.get("method") != method
                or fit.get("reference_time_s") != roots[0]["time_s"] or fit.get("future_poses_used") is not False):
            raise ValueError("Mobile floor fit must use only the first pose")
        for name in ("original_base_height", "lowest_surface_before", "base_height_shift", "lowest_surface_after"):
            _finite(fit.get(name), "floor fit " + name, -10, 10)
        if not math.isclose(fit["lowest_surface_after"], mobility["clearance"], abs_tol=1e-8):
            raise ValueError("Mobile floor fit must retain its authored clearance")
        if "ground_plane" in mobility:
            if (fit.get("normal") != ground_normal(spec).tolist() or fit.get("scope") != GROUND_SCOPE
                    or fit.get("reference_scope") != "offset_only_orientation_is_supplied_separately"):
                raise ValueError("Inclined floor fit must retain its exact normal and assumption scope")
            if (not math.isclose(fit["original_base_height"] + fit["base_height_shift"], mobility["base_height"], abs_tol=1e-8)
                    or not math.isclose(fit["lowest_surface_before"] + fit["base_height_shift"] * ground_normal(spec)[2],
                                        fit["lowest_surface_after"], abs_tol=1e-8)):
                raise ValueError("Inclined floor clearance must use the recorded global-z shift")
    return deepcopy(spec)


def prepare_mobile(spec, evidence, *, root_audit=None):
    if spec.get("profile") != "exercise-v2" or spec.get("schema_version") != 2:
        raise ValueError("Mobile preparation requires the spatial exercise-v2 model")
    audit = build_root_reference(evidence, torso_length=spec["lengths"]["torso"]) if root_audit is None else deepcopy(root_audit)
    if audit.get("status") != "ready":
        error = UnsupportedVideo("Mobile root reference unavailable: " + "; ".join(audit.get("quality", {}).get("errors", [])))
        error.root_motion_audit = audit
        raise error
    if audit.get("source_sha256") != evidence.get("source_sha256") or spec.get("source_sha256") != evidence.get("source_sha256"):
        raise ValueError("Mobile root audit source identity does not match pose evidence")
    result = deepcopy(spec)
    result.update(schema_version=3, profile="exercise-v3", mobility=deepcopy(MOBILITY_DEFAULTS),
                  root_reference=[{"time_s": row["time_s"], "translation": deepcopy(row["translation"])} for row in audit["reference"]])
    result["mobility"]["base_height"] = min(5., max(.01, spec["lengths"]["thigh"] + spec["lengths"]["shin"] + .08))
    result["assumptions"] = [item for item in result.get("assumptions", [])
                             if not ("anchor" in item.lower() and ("pelvis" in item.lower() or "root" in item.lower()))]
    if "body_frame" in result:
        result["body_frame"]["scope"] = "Camera-relative torso orientation fit only; root translation is controlled separately, not a fixed-pelvis constraint or calibrated world frame"
    result["assumptions"].extend([
        "Root x/z displacement follows image pelvis estimates under a fixed uncalibrated model scale; depth is not observed",
        "Three slide joints and virtual mass-scaled support forces are authored, not learned balance or measured human strength",
        "Floor, collision capsules, friction and support forces are simulated assumptions; self-collision remains disabled",
        "Release disables all virtual motors; pushes remain authored forces, and assistance is restored only explicitly",
        "The floor is aligned only to the first reference pose; later penetration is reported, not removed by moving the floor",
    ])
    return validate_mobility(result), audit


def append_mobile_world(world, spec):
    mobility = spec["mobility"]
    attributes = {}
    if "ground_plane" in mobility:
        normal = ground_normal(spec)
        quaternion = np.array([1 + normal[2], -normal[1], normal[0], 0.])
        quaternion /= np.linalg.norm(quaternion)
        attributes["quat"] = " ".join(map(str, quaternion))
    ET.SubElement(world, "geom", name="mobile_floor", type="plane", pos="0 0 0", size="5 5 .1", **attributes,
                  contype="1", conaffinity="2", friction=".8 .01 .001", rgba=".85 .80 .72 1")
    parent = ET.SubElement(world, "body", name="mobile_root", pos=f"0 0 {mobility['base_height']}")
    ET.SubElement(parent, "inertial", pos="0 0 0", mass=str(mobility["root_mass"]), diaginertia=".001 .001 .001")
    for name, axis in zip(ROOT_AXES, ("1 0 0", "0 1 0", "0 0 1")):
        ET.SubElement(parent, "joint", name=name, type="slide", axis=axis, limited="true",
                      range=" ".join(map(str, mobility['slide_limits'])), damping="0", armature="0")
    return parent


def add_mobile_actuators(actuator, spec):
    bound = spec["mobility"]["maximum_force"]
    for name in ROOT_AXES:
        ET.SubElement(actuator, "motor", name=name + "_motor", joint=name, gear="1", ctrllimited="true",
                      ctrlrange=f"{-bound} {bound}", forcelimited="true", forcerange=f"{-bound} {bound}")


def calibrate_floor(spec):
    from weft.human import human_xml

    if 'ground' in spec.get('camera_reference', {}):
        raise ValueError('Supplied camera-world ground must not be replaced by a pose-fitted floor')
    result = validate_mobility(spec)
    model = mujoco.MjModel.from_xml_string(human_xml(result))
    data = mujoco.MjData(model)
    for name, angle in result["reference"][0]["joints"].items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError("Initial floor fit references a missing rotation joint")
        data.qpos[model.jnt_qposadr[joint]] = angle
    mujoco.mj_forward(model, data)
    normal = ground_normal(result)
    surfaces = []
    for index in range(model.ngeom):
        if not model.geom_contype[index] & 2:
            continue
        kind = model.geom_type[index]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            surface = np.dot(normal, data.geom_xpos[index]) - model.geom_size[index, 0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            surface = (np.dot(normal, data.geom_xpos[index])
                       - abs(np.dot(normal, data.geom_xmat[index].reshape(3, 3)[:, 2])) * model.geom_size[index, 1]
                       - model.geom_size[index, 0])
        else:
            raise ValueError("First-pose floor fit supports only the authored sphere/capsule collision surfaces")
        surfaces.append(float(surface))
    if not surfaces:
        raise ValueError("First-pose floor fit needs collision-enabled body surfaces")
    before = min(surfaces)
    mobility = result["mobility"]
    original = mobility["base_height"]
    shift = (mobility["clearance"] - before) / normal[2]
    mobility["base_height"] = original + shift
    mobility["floor_fit"] = {"method": "first_pose_collision_surface_clearance_v1",
                             "reference_time_s": result["reference"][0]["time_s"], "future_poses_used": False,
                             "original_base_height": original, "lowest_surface_before": before,
                             "base_height_shift": float(shift), "lowest_surface_after": float(before + shift * normal[2])}
    mobility["base_height"] = float(mobility["base_height"])
    if "ground_plane" in mobility:
        mobility["floor_fit"].update(method="first_pose_inclined_plane_clearance_v1", normal=normal.tolist(), scope=GROUND_SCOPE,
                                     reference_scope="offset_only_orientation_is_supplied_separately")
    return validate_mobility(result)


class MobileDynamics:
    def __init__(self, env):
        self.env = env
        mobility = env.human_spec["mobility"]
        self.state_translation_limit = float(mobility['slide_limits'][1]) + 5.
        joint_ids = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ROOT_AXES]
        self.actuators = np.array([mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor") for name in ROOT_AXES])
        self.floor = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "mobile_floor")
        if min(joint_ids) < 0 or np.min(self.actuators) < 0 or self.floor < 0:
            raise ValueError("Mobile model is missing root joints, actuators or floor")
        self.q = np.array([int(env.model.jnt_qposadr[i]) for i in joint_ids])
        self.v = np.array([int(env.model.jnt_dofadr[i]) for i in joint_ids])
        self.mass = float(np.sum(env.model.body_mass))
        self.limit = min(mobility["maximum_force"], mobility["force_limit_weight_multiple"] * self.mass * 9.81)
        self.kp, self.kd = mobility["kp_per_mass"] * self.mass, mobility["kd_per_mass"] * self.mass
        self.push = mobility["push_acceleration"] * self.mass
        self.times = np.array([row["time_s"] for row in env.human_spec["root_reference"]])
        self.positions = np.array([row["translation"] for row in env.human_spec["root_reference"]])
        self.slopes = np.diff(self.positions, axis=0) / np.diff(self.times)[:, None]
        self.support_enabled, self.mode, self.hold = True, "hold", self.positions[0].copy()

    def reset(self, data):
        data.qpos[self.q] = self.positions[0]
        data.qvel[self.v] = 0.
        data.ctrl[self.actuators] = 0.
        self.support_enabled, self.mode, self.hold = True, "hold", self.positions[0].copy()

    def prepare(self, action):
        if type(action) is not int or not 0 <= action <= 9:
            raise ValueError("Mobile action must be an integer from 0 to 9")
        if action == 6:
            self.support_enabled, self.mode = False, "released"
        elif action == 5:
            self.support_enabled, self.mode = True, "follow"
        elif action == 9:
            self.support_enabled, self.mode = True, "hold"
            self._capture_hold()
        elif self.mode == "follow":
            self.mode = "hold"
            self._capture_hold()

    def _capture_hold(self):
        self.hold = self.env.data.qpos[self.q].copy()
        self.env._hold = self.env.data.qpos[self.env._q_array].copy()
        self.env.hold_angles = dict(zip(self.env.joint_names, map(float, self.env._hold)))

    def apply(self, action):
        data = self.env.data
        force = np.zeros(3)
        if not self.support_enabled:
            # Release persists across steps; no hidden angular servo survives the root release.
            data.ctrl[:] = 0.
        if self.support_enabled:
            target, velocity = self.hold, np.zeros(3)
            if self.mode == "follow":
                timestamp = float(np.clip(self.times[0] + data.time, self.times[0], self.times[-1]))
                index = min(len(self.times) - 2, max(0, int(np.searchsorted(self.times, timestamp, side="right")) - 1))
                target = self.positions[index] + self.slopes[index] * (timestamp - self.times[index])
                velocity = self.slopes[index] if timestamp < self.times[-1] else np.zeros(3)
            force = self.kp * (target - data.qpos[self.q]) + self.kd * (velocity - data.qvel[self.v]) + data.qfrc_bias[self.v]
        if action in (7, 8):
            force[0] += -self.push if action == 7 else self.push
        if not np.isfinite(force).all():
            raise RuntimeError("Nonfinite virtual root support force")
        data.ctrl[self.actuators] = np.clip(force, -self.limit, self.limit)

    def reference(self, data, index):
        if type(index) is not int or not 0 <= index < len(self.positions):
            raise ValueError("Mobile reference index is out of bounds")
        data.qpos[self.q] = self.positions[index]
        data.qvel[self.v] = 0.
        data.ctrl[self.actuators] = 0.

    def snapshot(self, data):
        kinematic = data is not self.env.data
        translation = data.qpos[self.q].tolist()
        velocity = data.qvel[self.v].tolist()
        force = [0., 0., 0.] if kinematic else data.ctrl[self.actuators].tolist()
        _vector(translation, "state translation", self.state_translation_limit)
        _vector(velocity, "linear velocity", 100)
        _vector(force, "virtual force", 2000)
        contacts, count, total, penetration = [], 0, 0., 0.
        for index in range(data.ncon):
            contact = data.contact[index]
            pair = (contact.geom1, contact.geom2)
            if self.floor not in pair:
                continue
            other = pair[1] if pair[0] == self.floor else pair[0]
            normal = 0.
            if not kinematic:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.env.model, data, index, wrench)
                normal = max(0., float(wrench[0]))
            _finite(normal, "contact normal force", 0, 1e9)
            distance = float(contact.dist)
            _finite(distance, "contact distance", -100, 100)
            position = contact.pos.tolist()
            _vector(position, "contact position", 100)
            count += 1
            total += normal
            penetration = max(penetration, -distance)
            if len(contacts) < 32:
                contacts.append({"geom": mujoco.mj_id2name(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, other) or f"geom-{other}",
                                 "position": position, "distance": distance, "normal_force": normal})
        _finite(total, "total normal force", 0, 1e9)
        if count > 128:
            raise RuntimeError("Mobile floor contact count exceeds the recording envelope")
        result = {"support_enabled": False if kinematic else self.support_enabled, "translation": translation,
                "linear_velocity": velocity, "support_force": force, "floor_height": 0.,
                "contacts": contacts, "contact_count": count, "total_normal_force": total,
                "max_penetration": penetration,
                "scope": "kinematic_overlap_not_dynamic_force" if kinematic else
                         "simulated_floor_contact_and_virtual_support_not_measured_forces"}
        if "ground_plane" in self.env.human_spec["mobility"]:
            result.update(floor_normal=ground_normal(self.env.human_spec).tolist(), floor_scope=GROUND_SCOPE)
        return result
