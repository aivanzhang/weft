"""Opt-in ankle capsules; keep the source body, root and floor unchanged."""

from copy import deepcopy
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from weft.foot_reference import FOOT_CONTRACT, FOOT_IMAGE_CONTRACT
from weft.human import HumanEnv, human_xml, validate_human
from weft.io import canonical_hash
from weft.spatial_pose import rotation_to_zyx
from weft.whole_body import SUCCESS_LIMITS
from weft.whole_body_feasibility import audit_reference_feasibility


SIDES = ("left", "right")
ANKLE_NAMES = tuple(f"{side}_ankle_{axis}" for side in SIDES for axis in ("z", "y", "x"))
FOOT_BODY_CONTRACT = {
    "method": "ankle-capsule-candidate-v1", "radius_model_units": .018,
    "mass_per_foot_kg": .5, "motor_limit_nm": 12., "rotation_chart": "local intrinsic ZYX",
    "joint_range_rad": [-math.tau, math.tau], "body_or_floor_refit": False,
    "source_interpolation": "none", "sole_normal": None, "measured_load": None,
    "physical_validation": "unverified", "self_collision": False,
    "assumptions": "Capsule radius, mass, motors and joint limits are authored, not measured anatomy",
}


def _number(value, low, high):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def _vector(value):
    if not isinstance(value, list) or len(value) != 3 or not all(_number(v, -1, 1) for v in value):
        raise ValueError("Foot directions require three finite unit-vector components")
    return np.asarray(value, dtype=float)


def foot_rotation(foot):
    reference = foot.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("Available foot requires its inferred rotation reference")
    x = _vector(reference.get("heel_to_index_axis"))
    z = _vector(reference.get("axis_to_ankle_direction"))
    rotation = np.column_stack((x, np.cross(z, x), z))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0):
        raise ValueError("Foot directions must be orthonormal; no implicit repair")
    return rotation


def validate_reference(spec, reference):
    validate_human(spec)
    if spec.get("profile") != "exercise-v3" or spec.get("dt") != 1 / 30:
        raise ValueError("Ankle candidate requires a 30-Hz mobile exercise-v3 body")
    contracts = {"kinematic-foot-reference-v1": FOOT_CONTRACT,
                 "image-directed-foot-reference-v1": FOOT_IMAGE_CONTRACT}
    if (not isinstance(reference, dict) or reference.get("profile") not in contracts
            or reference.get("schema_version") != 1 or reference.get("contract") != contracts[reference["profile"]]
            or reference.get("body_model_sha256") != canonical_hash(spec)
            or reference.get("source_sha256") != spec.get("source_sha256")):
        raise ValueError("Foot reference must match the unchanged source body and versioned contract")
    shapes, frames = reference.get("fixed_shapes"), reference.get("frames")
    if not isinstance(shapes, dict) or set(shapes) != set(SIDES):
        raise ValueError("Foot reference requires both side shapes, including unavailable shapes")
    for shape in shapes.values():
        if not isinstance(shape, dict) or shape.get("status") not in {"unavailable", "inferred_fixed_triangle"}:
            raise ValueError("Unsupported fixed foot shape status")
        parameters = shape.get("fixed_parameters")
        if shape["status"] == "unavailable":
            if parameters is not None:
                raise ValueError("Unavailable foot shapes cannot contain inferred parameters")
            continue
        if (not isinstance(parameters, dict) or set(parameters) != {"length", "along", "height"}
                or not _number(parameters["length"], .03, .5)
                or not _number(parameters["height"], .003, .25)
                or not _number(parameters["along"], -.5 * parameters["length"], 1.5 * parameters["length"])):
            raise ValueError("Fixed foot geometry exceeds the inferred-triangle envelope")
    if not isinstance(frames, list) or not 2 <= len(frames) <= 900:
        raise ValueError("Foot candidate requires 2–900 retained foot frames")
    previous_time, previous_frame = -1., -1
    for row in frames:
        if (not isinstance(row, dict) or not _number(row.get("time_s"), 0, 60)
                or row["time_s"] <= previous_time or type(row.get("frame")) is not int
                or not previous_frame < row["frame"] < 1800
                or row.get("evidence_id") != f'{reference["source_sha256"]}:frame:{row["frame"]}'
                or not isinstance(row.get("feet"), dict) or set(row["feet"]) != set(SIDES)):
            raise ValueError("Foot frames must preserve bounded, unique source times and identities")
        previous_time, previous_frame = row["time_s"], row["frame"]
        for side, foot in row["feet"].items():
            if not isinstance(foot, dict) or foot.get("status") not in {"unavailable", "kinematic_reference"}:
                raise ValueError("Foot frame must explicitly identify availability")
            if foot["status"] == "unavailable":
                if foot.get("reference") is not None or foot.get("endpoint_geometry") is not None:
                    raise ValueError("Unavailable feet cannot supply a fabricated pose")
                continue
            if shapes[side]["status"] != "inferred_fixed_triangle":
                raise ValueError("Available foot requires its fixed triangle")
            rotation = foot_rotation(foot)
            if reference["profile"] == "image-directed-foot-reference-v1":
                refinement = foot.get("image_refinement")
                if not isinstance(refinement, dict) or refinement.get("status") not in {"refined", "unavailable_image_axis", "edge_on_model"}:
                    raise ValueError("Image-directed feet require explicit per-pose refinement provenance")
                delta = refinement.get("angle_change_rad")
                bound = FOOT_IMAGE_CONTRACT["maximum_orientation_change_rad"]
                if not _number(delta, -bound, bound):
                    raise ValueError("Foot orientation change exceeds its declared bound")
                if refinement["status"] != "refined" and delta != 0:
                    raise ValueError("Unavailable image directions cannot change a foot pose")
                original = foot_rotation({"reference": refinement.get("source_world_reference")})
                c, s = math.cos(delta), math.sin(delta)
                correction = np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
                if not np.allclose(rotation, correction @ original, atol=1e-9, rtol=0):
                    raise ValueError("Refined foot orientation does not match its recorded bounded rotation")
    if reference["profile"] == "image-directed-foot-reference-v1":
        provenance = reference.get("orientation_refinement")
        if (not isinstance(provenance, dict) or provenance.get("method") != FOOT_IMAGE_CONTRACT["method"]
                or provenance.get("source_profile") != "kinematic-foot-reference-v1"
                or provenance.get("floor_used") is not False
                or not isinstance(provenance.get("source_reference_sha256"), str)
                or len(provenance["source_reference_sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in provenance["source_reference_sha256"])):
            raise ValueError("Refined foot references require their original reference hash and method")
    return reference


def foot_xml(spec, reference):
    validate_reference(spec, reference)
    if any(shape["status"] != "inferred_fixed_triangle" for shape in reference["fixed_shapes"].values()):
        raise ValueError("Candidate unavailable: both fixed foot shapes are required")
    root = ET.fromstring(human_xml(spec))
    actuator = root.find("actuator")
    for side in SIDES:
        parameters = reference["fixed_shapes"][side]["fixed_parameters"]
        heel = [-parameters["along"], 0., -parameters["height"]]
        toe = [parameters["length"] - parameters["along"], 0., -parameters["height"]]
        parent = root.find(f'.//body[@name="{side}_knee_body"]')
        foot = ET.SubElement(parent, "body", name=f"{side}_foot_body", pos=f'0 0 {-spec["lengths"]["shin"]}')
        for axis, vector in (("z", "0 0 1"), ("y", "0 1 0"), ("x", "1 0 0")):
            name = f"{side}_ankle_{axis}"
            ET.SubElement(foot, "joint", name=name, axis=vector, range=f"{-math.tau} {math.tau}")
            ET.SubElement(actuator, "motor", name=name + "_motor", joint=name,
                          ctrllimited="true", ctrlrange="-12 12", gear="1")
        ET.SubElement(foot, "geom", name=f"{side}_foot_geom", type="capsule",
                      fromto=" ".join(map(str, heel + toe)), size=str(FOOT_BODY_CONTRACT["radius_model_units"]),
                      mass=str(FOOT_BODY_CONTRACT["mass_per_foot_kg"]))
        for name, point in (("heel", heel), ("foot_index", toe)):
            ET.SubElement(foot, "site", name=f"{side}_{name}_site", pos=" ".join(map(str, point)), size=".01")
    return ET.tostring(root, encoding="unicode")


class FootBody(HumanEnv):
    """Native geometry and source-pose projection; direct control lives in FootEnv."""

    def __init__(self, spec, reference):
        self.foot_reference = deepcopy(validate_reference(spec, reference))
        super().__init__(deepcopy(spec))
        self.ankle_q, self.ankle_v, self.ankle_actuators = [], [], []
        for name in ANKLE_NAMES:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.ankle_q.append(int(self.model.jnt_qposadr[joint]))
            self.ankle_v.append(int(self.model.jnt_dofadr[joint]))
            self.ankle_actuators.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor"))
        self.ankle_q, self.ankle_v, self.ankle_actuators = map(np.array, (self.ankle_q, self.ankle_v, self.ankle_actuators))
        self._foot_frames = {row["time_s"]: row for row in self.foot_reference["frames"]}
        self._knee_bodies = {side: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, side + "_knee_body") for side in SIDES}
        for side in SIDES:
            self.segment_defs.append((f"{side}_foot", f"{side}_heel", f"{side}_foot_index", FOOT_BODY_CONTRACT["radius_model_units"]))
            for name in (f"{side}_heel", f"{side}_foot_index"):
                self._site_ids[name] = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name + "_site")
            self.scene["objects"].append({"id": f"{side}_foot", "label": f"{side} foot candidate", "shape": "circle",
                                          "radius": FOOT_BODY_CONTRACT["radius_model_units"] / self.height,
                                          "color": "#628A8C", "role": "passive"})

    def _compile_model(self, spec):
        return mujoco.MjModel.from_xml_string(foot_xml(spec, self.foot_reference))

    def reset(self, **kwargs):
        raise RuntimeError("FootBody is a source-pose projector; use FootEnv for explicit direct motor control")

    def step(self, action):
        raise RuntimeError("Use FootEnv; the original assisted body controller does not control the added ankles")

    def advance_physics(self, action=0):
        return self.step(action)

    def source_pose(self, index):
        if type(index) is not int or not 0 <= index < len(self.human_spec["reference"]):
            raise ValueError("Source pose index is outside the retained body reference")
        pose = self.human_spec["reference"][index]
        row = self._foot_frames.get(pose["time_s"])
        if row is None or any(row["feet"][side]["status"] != "kinematic_reference" for side in SIDES):
            return None
        data = mujoco.MjData(self.model)
        data.qpos[self._q_array] = [pose["joints"][name] for name in self.joint_names]
        self.mobility.reference(data, index)
        mujoco.mj_forward(self.model, data)
        for i, side in enumerate(SIDES):
            desired = foot_rotation(row["feet"][side])
            local = data.xmat[self._knee_bodies[side]].reshape(3, 3).T @ desired
            angles, _ = rotation_to_zyx(local)
            data.qpos[self.ankle_q[i * 3:i * 3 + 3]] = angles
        data.time = pose["time_s"] - self.human_spec["reference"][0]["time_s"]
        mujoco.mj_forward(self.model, data)
        return data

    def reference_snapshot(self, index):
        data = self.source_pose(index)
        if data is None:
            return None
        state = self._snapshot(data)
        state.update(source_time_s=self.human_spec["reference"][index]["time_s"], steps=index,
                     scope="Kinematic candidate pose; no physics step or measured contact force")
        state["human"]["reference"] = True
        return state

    def _snapshot(self, data):
        state = super()._snapshot(data)
        human = state["human"]
        for i, name in enumerate(ANKLE_NAMES):
            side = name.split("_")[0]
            human["angles"][name] = float(data.qpos[self.ankle_q[i]])
            human["angular_velocities"][name] = float(data.qvel[self.ankle_v[i]])
            for label, sign in (("positive", 1), ("negative", -1)):
                human["muscles"].append({"name": name + "_" + label,
                    "a": human["joints"][side + "_ankle"], "b": human["joints"][side + "_foot_index"],
                    "activation": max(0., sign * float(data.ctrl[self.ankle_actuators[i]]) / FOOT_BODY_CONTRACT["motor_limit_nm"]),
                    "status": "illustrative"})
        human["foot_candidate"] = deepcopy(FOOT_BODY_CONTRACT)
        return state


def audit_foot_body(spec, reference):
    validate_reference(spec, reference)
    baseline = audit_reference_feasibility(spec)
    result = {"schema_version": 1, "contract": deepcopy(FOOT_BODY_CONTRACT),
              "body_sha256": canonical_hash(spec), "foot_reference_sha256": canonical_hash(reference),
              "baseline": baseline, "checks": [], "interactive_start_available": False,
              "physical_validation": "unverified", "promoted": False}
    shapes_available = all(s["status"] == "inferred_fixed_triangle" for s in reference["fixed_shapes"].values())
    env = FootBody(spec, reference) if shapes_available else None
    try:
        for i, pose in enumerate(spec["reference"]):
            data = env.source_pose(i) if env else None
            row = {"body_reference_index": i, "time_s": pose["time_s"], "status": "unavailable",
                   "max_penetration_model_units": None, "passes_floor_limit": None,
                   "reason": "missing_fixed_shapes" if not env else "missing_exact_time_foot_pose"}
            if data is not None:
                contacts = []
                for contact in data.contact:
                    if env.mobility.floor not in (contact.geom1, contact.geom2):
                        continue
                    geom = contact.geom2 if contact.geom1 == env.mobility.floor else contact.geom1
                    contacts.append({"geom": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                                     "distance": float(contact.dist), "position": contact.pos.tolist()})
                if len(contacts) > 32:
                    raise ValueError("Candidate floor contacts exceed complete-report bound")
                penetration = max((max(0., -c["distance"]) for c in contacts), default=0.)
                row.update(status="checked", reason=None, max_penetration_model_units=penetration,
                           passes_floor_limit=penetration <= SUCCESS_LIMITS["max_floor_penetration_model_units"],
                           floor_contacts=contacts, ankle_angles_rad=data.qpos[env.ankle_q].tolist())
            result["checks"].append(row)
        result["interactive_start_available"] = result["checks"][0]["status"] == "checked"
        checked = [r for r in result["checks"] if r["status"] == "checked"]
        result["summary"] = {"body_source_poses": len(result["checks"]), "checked": len(checked),
                             "unavailable": len(result["checks"]) - len(checked),
                             "failed_floor_checks": sum(not r["passes_floor_limit"] for r in checked),
                             "max_penetration_model_units": max((r["max_penetration_model_units"] for r in checked), default=None),
                             "physics_steps": 0, "source_interpolation": False,
                             "foot_control_grid_checked": False}
        result["status"] = "diagnostic_candidate" if shapes_available else "unavailable"
        return result
    finally:
        if env:
            env.close()
