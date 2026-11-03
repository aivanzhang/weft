"""Articulated exercise models; motor commands are not measured muscle forces."""

from bisect import bisect_right
from copy import deepcopy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from weft.body_frame import fit_body_frame, validate_body_frame
from weft.io import canonical_hash, write_json
from weft.spatial_pose import SPATIAL_JOINT_NAMES, SPATIAL_JOINT_LIMITS, build_spatial_reference


JOINT_NAMES = tuple(f"{side}_{joint}" for side in ("left", "right") for joint in ("shoulder", "elbow", "hip", "knee"))
JOINT_LIMITS = {name: ([-1.0, 2.9] if "shoulder" in name else [-0.5, 2.3] if "hip" in name else [0.0, 2.6])
                for name in JOINT_NAMES}
LENGTH_BOUNDS = {"torso": (0.18, 0.9), "upper_arm": (0.1, 0.65), "forearm": (0.1, 0.65),
                 "thigh": (0.15, 0.85), "shin": (0.15, 0.85), "shoulder_width": (0.15, 0.8), "hip_width": (0.1, 0.6)}
ACTIONS = ("coast", "flex", "extend", "brake", "hold", "reference servo")


def reference_velocity_index(times, timestamp, *, elapsed, timestep, source_start):
    """Select a derivative only; ``times`` must already be finite and strictly sorted.

    Native clocks add a positive timestep repeatedly. Near an aligned knot, their
    summation error must not defer the right derivative; positions keep their old lookup.
    """
    timestamp, elapsed, timestep, source_start = map(float, (timestamp, elapsed, timestep, source_start))
    if len(times) < 2 or not all(map(math.isfinite, (timestamp, elapsed, timestep, source_start))) or timestep <= 0:
        raise ValueError("Reference velocity requires finite clocks, a positive timestep and at least two knots")
    index = min(len(times) - 2, max(0, bisect_right(times, timestamp) - 1))
    if timestamp >= times[-1]:
        return None
    if elapsed <= 0 or timestamp < times[0] or timestamp != source_start + elapsed:
        return index

    quotient = elapsed / timestep
    if not math.isfinite(quotient):
        return index
    n = math.ceil(math.nextafter(quotient, math.inf))
    unit_roundoff = 2.0 ** -53
    if n * unit_roundoff >= 1:
        return index
    gamma = math.nextafter(n * unit_roundoff / (1 - n * unit_roundoff), math.inf)
    # An uncertain step count cannot identify a unique native-clock boundary.
    if gamma * n >= .5:
        return index
    nearest = round(quotient)
    if nearest < 1:
        return index
    lattice_elapsed = nearest * timestep
    lattice_timestamp = source_start + lattice_elapsed
    knot = float(times[index + 1])
    half_gap = math.nextafter(min(knot - float(times[index]),
                                 float(times[index + 2]) - knot if index + 2 < len(times) else math.inf) / 2, 0.)
    # Two timestep roundings cover dt representation and dt/substeps; all terms round outward.
    step_error = math.nextafter(math.fsum((2 * n * math.ulp(timestep),
                                          math.nextafter(timestep * math.ulp(quotient), math.inf),
                                          math.ulp(lattice_elapsed))), math.inf)
    source_error = math.nextafter(math.fsum((math.ulp(source_start), math.ulp(lattice_timestamp),
                                            math.ulp(timestamp), math.ulp(knot))), math.inf)
    lattice_error = math.nextafter(step_error + source_error, math.inf)
    accumulation_error = math.nextafter(gamma * math.nextafter(n * timestep, math.inf), math.inf)
    clock_error = math.nextafter(accumulation_error + step_error, math.inf)
    tolerance = min(1e-9, half_gap, math.nextafter(clock_error + source_error, math.inf))
    # Reject ambiguous nearby knots and non-native queries, even inside the broad summation bound.
    if lattice_error >= half_gap or math.nextafter(abs(elapsed - lattice_elapsed), math.inf) > clock_error:
        return index
    if math.nextafter(abs(knot - lattice_timestamp), math.inf) > lattice_error:
        return index
    if math.nextafter(knot - timestamp, math.inf) > tolerance:
        return index
    return index + 1 if index + 1 < len(times) - 1 else None


def _finite(value, name, low, high):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and between {low} and {high}")
    return float(value)


def joint_contract(spec):
    return ((SPATIAL_JOINT_NAMES, SPATIAL_JOINT_LIMITS) if spec.get("profile") in {"exercise-v2", "exercise-v3"}
            else (JOINT_NAMES, JOINT_LIMITS))


def joint_segment(name):
    if name.startswith("torso_"):
        return "torso", "pelvis", "neck"
    side, joint = name.split("_")[:2]
    segment, tip = {"shoulder": ("upper_arm", "elbow"), "elbow": ("forearm", "wrist"),
                    "hip": ("thigh", "knee"), "knee": ("shin", "ankle")}[joint]
    return f"{side}_{segment}", f"{side}_{joint}", f"{side}_{tip}"


def validate_human(spec):
    if isinstance(spec, dict) and "provenance_projection" in spec:
        raise ValueError("Runtime human projection requires its complete sidecar for full-body/source qualification")
    if not isinstance(spec, dict) or type(spec.get("schema_version")) is not int or (spec["schema_version"], spec.get("profile")) not in ((1, "exercise-v1"), (2, "exercise-v2"), (3, "exercise-v3")):
        raise ValueError("Unsupported human model schema")
    result = deepcopy(spec)
    if "native_clock" in result:
        from weft.source_clock import validate_summary
        if (result["profile"] not in {"exercise-v2", "exercise-v3"} or "camera_reference" in result
                or not isinstance(result.get("source_sha256"), str)):
            raise ValueError("Native source-clock bodies require the spatial pose reconstruction path")
        result["native_clock"] = validate_summary(result["native_clock"], source_sha256=result.get("source_sha256"))
    if 'camera_reference' in result and result['profile'] != 'exercise-v3':
        raise ValueError('Camera-world references require an explicit mobile body')
    names, limits = joint_contract(result)
    if result.get("controller", "pd") not in ("pd", "bias_compensated_pd"):
        raise ValueError("Unknown authored human controller")
    if "motor_limits" in result:
        if not isinstance(result["motor_limits"], dict) or set(result["motor_limits"]) != set(names):
            raise ValueError("Motor limits must cover every model axis")
        for name in names:
            _finite(result["motor_limits"][name], "motor torque limit", 1, 120)
    if "body_frame" in result:
        validate_body_frame(result["body_frame"])
    if not isinstance(result.get("lengths"), dict) or set(result["lengths"]) != set(LENGTH_BOUNDS):
        raise ValueError("Human model requires the seven documented body lengths")
    for key, bounds in LENGTH_BOUNDS.items():
        result["lengths"][key] = _finite(result["lengths"][key], key, *bounds)
    result["dt"] = _finite(result.get("dt"), "dt", 1 / 120, 1 / 20)
    reference = result.get("reference")
    if not isinstance(reference, list) or not 2 <= len(reference) <= 1800:
        raise ValueError("Human model requires 2–1800 observed pose samples")
    previous = -1.0
    for frame in reference:
        if not isinstance(frame, dict) or set(frame) != {"time_s", "joints"}:
            raise ValueError("Reference samples require time_s and joints")
        timestamp = _finite(frame["time_s"], "reference time", 0, 3600)
        if timestamp <= previous:
            raise ValueError("Reference times must increase strictly")
        previous = timestamp
        if not isinstance(frame["joints"], dict) or set(frame["joints"]) != set(names):
            raise ValueError("Reference sample must contain all joints declared by the body model")
        for name in names:
            frame["joints"][name] = _finite(frame["joints"][name], name, *limits[name])
    if not isinstance(result.get("assumptions"), list) or any(not isinstance(item, str) for item in result["assumptions"]):
        raise ValueError("Human assumptions must be explicit text")
    if result["profile"] == "exercise-v3":
        from weft.mobile_human import validate_mobility
        validate_mobility(result)
        if "ground_plane" in result["mobility"]:
            result["assumptions"] = [item for item in result["assumptions"]
                                     if "camera up is assumed gravity-up" not in item.lower()
                                     and "floor is aligned only to the first reference pose" not in item.lower()]
            assumption = "Camera-relative inclined floor and normal-aligned gravity are authored hypotheses, not measured ground or gravity; scale, perspective and depth remain uncalibrated"
            if assumption not in result["assumptions"]:
                result["assumptions"].append(assumption)
            offset_scope = "Only floor clearance offset uses the first pose; plane orientation has separate source provenance, and later penetration is not removed"
            if offset_scope not in result["assumptions"]:
                result["assumptions"].append(offset_scope)
    return result


def _body_artifact(out, name, value, artifact_sink):
    if artifact_sink is None:
        if isinstance(value, str):
            (out / name).write_text(value)
        else:
            write_json(out / name, value)
    elif isinstance(value, str):
        artifact_sink.write_text(name, value)
    else:
        artifact_sink.write_json(name, value)


def build_human(evidence, out, *, body_model="hinge", artifact_sink=None):
    if body_model not in ("hinge", "spatial", "mobile"):
        raise ValueError("Body model must be hinge, spatial or mobile")
    if body_model in ("spatial", "mobile"):
        return build_spatial_human(evidence, out, mobile=body_model == "mobile", artifact_sink=artifact_sink)
    if not isinstance(evidence, dict) or not isinstance(evidence.get("reference"), list):
        raise ValueError("Pose evidence must contain reference joint angles")
    references = [{"time_s": row["time_s"], "joints": deepcopy(row["joints"])} for row in evidence["reference"]]
    clipped = []
    for index, frame in enumerate(references):
        if not isinstance(frame.get("joints"), dict) or set(frame["joints"]) != set(JOINT_NAMES):
            raise ValueError("Pose evidence must contain all eight joint angles")
        for name, angle in frame["joints"].items():
            _finite(angle, name, -math.pi, math.pi)
            limited = float(np.clip(angle, *JOINT_LIMITS[name]))
            if limited != angle:
                clipped.append({"sample": index, "joint": name, "observed_radians": angle, "model_radians": limited})
            frame["joints"][name] = limited
    body_frame = fit_body_frame(evidence)
    spec = validate_human({"schema_version": 1, "profile": "exercise-v1", "dt": 1 / 30,
                          "body_frame": body_frame,
                          "lengths": deepcopy(evidence.get("lengths")), "reference": references,
                          "assumptions": [
                              "Monocular body dimensions and depth are inferred, not metrically calibrated measurements",
                              "Pelvis and torso are anchored; this model cannot establish balance, ground reaction, or full-body transfer",
                              "Anchor orientation is fitted to camera-relative torso estimates; camera up is assumed gravity-up, not calibrated",
                              "Eight authored sagittal-plane hinge joints; no spinal, ankle, wrist, or out-of-plane shoulder mechanics",
                              "Limb masses, gravity, damping, motor limits, and joint limits are authored model assumptions",
                              "Body self-collision is disabled; reference playback is a servo baseline, not recovered muscle control",
                              "Muscle bands show illustrative antagonist motor-command proxies, not anatomy, EMG, muscle force, or medical analysis",
                              "Unselected joints use authored stabilizing controllers; selected joint is the learning/control task",
                          ]})
    spec["source_sha256"] = evidence.get("source_sha256")
    spec["reconstruction_coverage"] = {
        **deepcopy(evidence.get("quality", {}).get("model_coverage", {})),
        "clamped_joint_values": len(clipped), "total_joint_values": len(references) * len(JOINT_NAMES),
        "clamped_fraction": len(clipped) / max(1, len(references) * len(JOINT_NAMES)),
        "anchor_frame": body_frame.get("fit", {"status": "legacy upright assumption"}),
        "physical_validation": "unavailable",
    }
    spec["clamped_reference_samples"] = clipped
    spec["reference_evidence"] = [{"time_s": row["time_s"], "joint_confidence": row.get("joint_confidence", {}),
                                  "pose_status": row.get("pose_status", "confidence not supplied"),
                                  **({"image_refinement": row["image_refinement"]} if "image_refinement" in row else {})}
                                 for row in evidence["reference"]]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "human.json").exists() or (out / "human.xml").exists():
        raise FileExistsError("Human model already exists; use a new output directory")
    _body_artifact(out, "human.json", spec, artifact_sink)
    _body_artifact(out, "body-frame.json", body_frame, artifact_sink)
    _body_artifact(out, "human.xml", human_xml(spec), artifact_sink)
    return spec


def build_spatial_human(evidence, out, *, mobile=False, artifact_sink=None):
    native_clock = None
    if "source_clock" in evidence:
        from weft.source_clock import clock_summary, validate_derived_evidence
        native_clock = clock_summary(validate_derived_evidence(evidence))
    body_frame = fit_body_frame(evidence)
    reference, audit = build_spatial_reference(evidence, body_frame)
    spec = validate_human({"schema_version": 2, "profile": "exercise-v2", "dt": 1 / 30,
                          "controller": "bias_compensated_pd",
                          "motor_limits": {name: 120 if name.startswith("torso_") else 30 for name in SPATIAL_JOINT_NAMES},
                          "body_frame": body_frame, "lengths": deepcopy(evidence.get("lengths")),
                          "reference": reference, "source_sha256": evidence.get("source_sha256"),
                          "assumptions": [
                              "Nineteen authored rotation axes fit monocular pose estimates, not measured anatomy",
                              "Pelvis position is anchored; torso and limbs rotate dynamically, without ground contact or balance",
                              "Camera up is assumed gravity-up; scale, perspective, gravity alignment and support are uncalibrated",
                              "Rotation chart bounds are numerical limits, not anatomical range-of-motion measurements",
                              "Straight or occluded limbs do not identify twist; assumed orientations retain zero twist confidence",
                              "Masses, damping, motor limits and bias-compensated PD stabilization are authored; self-collision is disabled",
                              "Reference motors use interpolated target velocity and simulated gravity/Coriolis compensation, not measured forces",
                              "Virtual torso support motors allow 120 Nm per axis; limb motors allow 30 Nm, not measured human strengths",
                              "Muscle bands show motor-command proxies, not anatomical muscle activation or force",
                              "The learned task is one selected rotation axis, not whole-exercise imitation",
                          ]})
    spec["reference_evidence"] = deepcopy(audit["frames"])
    if native_clock is not None:
        spec["native_clock"] = native_clock
    spec["reconstruction_coverage"] = {"rotation_axes": len(SPATIAL_JOINT_NAMES),
                                       "spatial_pose": audit, "physical_validation": "unavailable"}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "human.json").exists() or (out / "human.xml").exists():
        raise FileExistsError("Human model already exists; use a new output directory")
    if mobile:
        from weft.mobile_human import prepare_mobile, calibrate_floor
        from weft.root_motion import build_root_reference
        root_audit = build_root_reference(evidence)
        _body_artifact(out, "root-motion.json", root_audit, artifact_sink)
        spec, _ = prepare_mobile(spec, evidence, root_audit=root_audit)
        spec = validate_human(calibrate_floor(spec))
        _body_artifact(out, "floor-assumption.json", spec["mobility"]["floor_fit"], artifact_sink)
    _body_artifact(out, "human.json", spec, artifact_sink)
    _body_artifact(out, "body-frame.json", body_frame, artifact_sink)
    _body_artifact(out, "spatial-pose.json", audit, artifact_sink)
    _body_artifact(out, "human.xml", human_xml(spec), artifact_sink)
    return spec


def human_xml(spec):
    spec = validate_human(spec)
    lengths = spec["lengths"]
    names, limits = joint_contract(spec)
    spatial = spec["profile"] in {"exercise-v2", "exercise-v3"}
    mobile = spec["profile"] == "exercise-v3"
    root = ET.Element("mujoco", model="weft_mobile_exercise" if mobile else "weft_anchored_exercise")
    ET.SubElement(root, "compiler", angle="radian")
    gravity = "0 0 -9.81"
    if mobile and "ground_plane" in spec["mobility"]:
        from weft.mobile_human import ground_normal
        gravity = " ".join(map(str, -9.81 * ground_normal(spec)))
    ET.SubElement(root, "option", timestep=str(spec["dt"] / 8), gravity=gravity, integrator="implicitfast")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", type="hinge", damping="1.5", armature="0.08", limited="true")
    ET.SubElement(default, "geom", type="capsule", size="0.04", density="700", contype="2" if mobile else "0", conaffinity="1" if mobile else "0", rgba="0.34 0.52 0.54 1")
    world = ET.SubElement(root, "worldbody")
    pelvis_height = lengths["thigh"] + lengths["shin"] + 0.08
    orientation = np.asarray(spec.get("body_frame", {}).get("rotation", np.eye(3)), dtype=float)
    axes = " ".join(str(value) for value in np.concatenate((orientation[:, 0], orientation[:, 1])))
    parent = world
    if mobile:
        from weft.mobile_human import append_mobile_world, add_mobile_actuators
        parent = append_mobile_world(world, spec)
    torso = ET.SubElement(parent, "body", name="torso", pos="0 0 0" if mobile else f"0 0 {pelvis_height}", xyaxes=axes)
    if spatial:
        for name, axis in (("torso_twist", "0 0 1"), ("torso_lateral", "0 1 0"), ("torso_flex", "1 0 0")):
            ET.SubElement(torso, "joint", name=name, axis=axis, range=" ".join(map(str, limits[name])))
    ET.SubElement(torso, "geom", name="torso_geom", fromto=f"0 0 0 0 0 {lengths['torso']}", size="0.09")
    ET.SubElement(torso, "geom", name="shoulder_frame", fromto=f"{-lengths['shoulder_width']/2} 0 {lengths['torso']} {lengths['shoulder_width']/2} 0 {lengths['torso']}", size="0.05")
    ET.SubElement(torso, "geom", name="pelvis_frame", fromto=f"{-lengths['hip_width']/2} 0 0 {lengths['hip_width']/2} 0 0", size="0.06")
    ET.SubElement(torso, "site", name="pelvis_site", pos="0 0 0", size="0.01")
    ET.SubElement(torso, "site", name="neck_site", pos=f"0 0 {lengths['torso']}", size="0.01")
    ET.SubElement(torso, "site", name="head_site", pos=f"0 0 {lengths['torso'] + 0.18}", size="0.01")
    ET.SubElement(torso, "geom", name="head_geom", type="sphere", pos=f"0 0 {lengths['torso'] + 0.13}", size="0.10")
    actuator = ET.SubElement(root, "actuator")
    for side, sign in (("left", -1), ("right", 1)):
        for limb, proximal, distal, length1, length2, width, elevation in (
            ("arm", "shoulder", "elbow", "upper_arm", "forearm", "shoulder_width", lengths["torso"]),
            ("leg", "hip", "knee", "thigh", "shin", "hip_width", 0),
        ):
            base = ET.SubElement(torso, "body", name=f"{side}_{proximal}_body", pos=f"{sign * lengths[width] / 2} 0 {elevation}")
            joint = f"{side}_{proximal}"
            if spatial:
                for suffix, axis in (("twist", "0 0 1"), ("lateral", "0 1 0")):
                    extra = f"{joint}_{suffix}"
                    ET.SubElement(base, "joint", name=extra, axis=axis, range=" ".join(map(str, limits[extra])))
            ET.SubElement(base, "joint", name=joint, axis="1 0 0", range=" ".join(map(str, limits[joint])))
            ET.SubElement(base, "site", name=f"{joint}_site", pos="0 0 0", size="0.01")
            ET.SubElement(base, "geom", name=f"{joint}_geom", fromto=f"0 0 0 0 0 {-lengths[length1]}", size="0.045" if limb == "leg" else "0.033")
            child = ET.SubElement(base, "body", name=f"{side}_{distal}_body", pos=f"0 0 {-lengths[length1]}")
            child_joint = f"{side}_{distal}"
            ET.SubElement(child, "joint", name=child_joint, axis="-1 0 0" if distal == "knee" else "1 0 0",
                          range=" ".join(map(str, limits[child_joint])))
            ET.SubElement(child, "site", name=f"{child_joint}_site", pos="0 0 0", size="0.01")
            ET.SubElement(child, "geom", name=f"{child_joint}_geom", fromto=f"0 0 0 0 0 {-lengths[length2]}", size="0.038" if limb == "leg" else "0.027")
            tip = "ankle" if limb == "leg" else "wrist"
            ET.SubElement(child, "site", name=f"{side}_{tip}_site", pos=f"0 0 {-lengths[length2]}", size="0.01")
    for name in names:
        limit = spec.get("motor_limits", {}).get(name, 30)
        ET.SubElement(actuator, "motor", name=f"{name}_motor", joint=name, ctrllimited="true", ctrlrange=f"{-limit} {limit}", gear="1")
    if mobile:
        add_mobile_actuators(actuator, spec)
    return ET.tostring(root, encoding="unicode")


class HumanEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    joint_names = JOINT_NAMES

    def __init__(self, spec, render_mode=None):
        self.human_spec = validate_human(spec)
        self.joint_names, self.joint_limits = joint_contract(self.human_spec)
        self.render_mode = render_mode
        self.model = self._compile_model(spec)
        self.data = mujoco.MjData(self.model)
        self._joint_ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.joint_names}
        self._actuator_ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + "_motor") for name in self.joint_names}
        self._q = {name: int(self.model.jnt_qposadr[index]) for name, index in self._joint_ids.items()}
        self._v = {name: int(self.model.jnt_dofadr[index]) for name, index in self._joint_ids.items()}
        self._q_array = np.array([self._q[name] for name in self.joint_names])
        self._v_array = np.array([self._v[name] for name in self.joint_names])
        self._a_array = np.array([self._actuator_ids[name] for name in self.joint_names])
        self._motor_limits = np.array([spec.get("motor_limits", {}).get(name, 30) for name in self.joint_names])
        self.selected_joint = "left_knee"
        self.targets = {name: float(np.clip(np.percentile([row["joints"][name] for row in spec["reference"]], 75),
                                            self.joint_limits[name][0] + 0.1, self.joint_limits[name][1] - 0.1)) for name in self.joint_names}
        self.action_space = spaces.Discrete(10 if spec["profile"] == "exercise-v3" else 6)
        error_bound = 13 if spec["profile"] in {"exercise-v2", "exercise-v3"} else 4
        self.observation_space = spaces.Box(np.array([-error_bound, -30], dtype=np.float32),
                                            np.array([error_bound, 30], dtype=np.float32))
        self._done = True
        self._initialized = False
        self._reference_times = np.array([row["time_s"] for row in spec["reference"]])
        self._reference_angles = {name: np.array([row["joints"][name] for row in spec["reference"]]) for name in self.joint_names}
        self._reference_matrix = np.array(list(self._reference_angles.values()))
        self._reference_slopes = np.diff(self._reference_matrix, axis=1) / np.diff(self._reference_times)
        lengths = spec["lengths"]
        self.height = lengths["thigh"] + lengths["shin"] + lengths["torso"] + 0.34
        reach = max(math.hypot(lengths["torso"], lengths["shoulder_width"] / 2)
                    + lengths["upper_arm"] + lengths["forearm"],
                    lengths["hip_width"] / 2 + lengths["thigh"] + lengths["shin"], lengths["torso"] + .23)
        self._body_view_height = 2.2 * (reach + .3)
        self.view = ({"view_height": 2.2 * reach,
                      "view_origin": [0, 0, lengths["thigh"] + lengths["shin"] + .08]}
                     if spec["profile"] in {"exercise-v2", "exercise-v3"} else {})
        self.mobility = None
        if spec["profile"] == "exercise-v3":
            from weft.mobile_human import MobileDynamics
            self.mobility = MobileDynamics(self)
            roots = np.array([row["translation"] for row in spec["root_reference"]])
            center = (roots.min(axis=0) + roots.max(axis=0)) / 2
            center[2] += spec["mobility"]["base_height"]
            self.view = {"view_height": float(2.2 * (reach + np.ptp(roots, axis=0).max() / 2 + .3)),
                         "view_origin": center.tolist()}
        self.segment_defs = [("torso", "pelvis", "neck", 0.09), ("head", "neck", "head", 0.10),
                             ("pelvis", "left_hip", "right_hip", 0.06),
                             ("shoulders", "left_shoulder", "right_shoulder", 0.05)]
        for side in ("left", "right"):
            self.segment_defs += [(f"{side}_upper_arm", f"{side}_shoulder", f"{side}_elbow", 0.033),
                                  (f"{side}_forearm", f"{side}_elbow", f"{side}_wrist", 0.027),
                                  (f"{side}_thigh", f"{side}_hip", f"{side}_knee", 0.045),
                                  (f"{side}_shin", f"{side}_knee", f"{side}_ankle", 0.038)]
        self._site_ids = {name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name + "_site")
                          for name in {endpoint for _, a, b, _ in self.segment_defs for endpoint in (a, b)}}
        self.scene = {"schema_version": 1, "profile": spec["profile"], "width": 1, "height": 1.2, "dt": self.human_spec["dt"],
                      "controlled_id": "left_shin", "target": [0.5, 0.5],
                      "task": {"type": "joint_angle", "angle_tolerance_radians": 0.08,
                               "angular_speed_tolerance_radians_per_second": 0.15, "max_steps": 240,
                               "selected_joint": self.selected_joint, "target_angle": self.target_angle,
                               "target_provenance": "Authored task using reference 75th-percentile angle, clipped inside authored joint limits"},
                      "objects": [{"id": segment[0], "label": segment[0].replace("_", " "), "shape": "circle",
                                   "radius": segment[3] / self.height, "color": "#628A8C", "role": "passive"}
                                  for segment in self.segment_defs],
                      "assumptions": self.human_spec["assumptions"],
                      "human": {"profile": spec["profile"], "lengths": deepcopy(spec["lengths"]),
                                "joint_names": list(self.joint_names), "controller": spec.get("controller", "pd"),
                                "source_sha256": spec.get("source_sha256"), "model_artifact": "human.json"}}
        if self.mobility:
            self.scene["human"]["mobility"] = {"translation_axes": 3, "support": "authored virtual motors",
                "floor": "assumed first-pose support plane", "root_depth": "unobserved; zero reference"}
            if 'camera_reference' in spec:
                self.scene['human']['root_method'] = spec['camera_reference']['method']
                self.scene['human']['mobility'].update(
                    root_depth='Estimated using supplied cameras and assumed pose-model scale; not measured',
                    coordinates='Supplied right-handed z-up world; first root removed',
                    travel_limits=list(spec['mobility']['slide_limits']))
                self.scene['learned_policy_available'] = False
                if 'root_refinement' in spec['camera_reference']:
                    view_count = len(spec['camera_reference']['root_refinement']['views'])
                    view_label = 'Multi-view' if view_count > 2 else 'Two-view'
                    if view_count > 2:
                        self.scene['human']['camera_view_count'] = view_count
                    self.scene['human']['mobility']['root_depth'] = (
                        f'{view_label} hip midpoint using supplied cameras and declared alignment; body shape remains monocular, not measured')
                if 'body_refinement' in spec['camera_reference']:
                    self.scene['human']['mobility']['root_depth'] = (
                        f'{view_label} hip midpoint; partial {view_label.lower()} body geometry with explicit monocular priors, not measured')
                    self.scene['human']['geometry_mode'] = spec['camera_reference']['body_refinement']['mode']
                    self.scene['human']['geometry_coverage'] = deepcopy(spec['reconstruction_coverage']['two_view_body']['group_coverage'])
                    if 'segment_contract' in spec['camera_reference']['body_refinement']:
                        self.scene['human']['partial_segments'] = True
                    if 'limb_position_fit' in spec['reconstruction_coverage']['two_view_body']:
                        self.scene['human']['limb_position_fit'] = deepcopy(spec['reconstruction_coverage']['two_view_body']['limb_position_fit'])
                        self.scene['human']['mobility']['root_depth'] = f'{view_label} hip midpoint; fixed-length source-position angle proposals, not measured anatomy or contact'
                    if 'partial_pixels' in spec['reconstruction_coverage']['two_view_body']:
                        self.scene['human']['partial_pixels'] = deepcopy(spec['reconstruction_coverage']['two_view_body']['partial_pixels'])
                    if 'depth_branches' in spec['reconstruction_coverage']['two_view_body']:
                        self.scene['human']['depth_branches'] = deepcopy(spec['reconstruction_coverage']['two_view_body']['depth_branches'])
                    if 'joint_pixels' in spec['reconstruction_coverage']['two_view_body']:
                        self.scene['human']['joint_pixels'] = deepcopy(spec['reconstruction_coverage']['two_view_body']['joint_pixels'])
                if 'ground' in spec['camera_reference']:
                    ground = spec['camera_reference']['ground']
                    self.scene['human']['mobility'].update(
                        floor='Supplied horizontal ground in the camera-world frame; not fitted to body poses',
                        coordinates='Supplied z-up world; first-root x/y removed, z relative to supplied ground',
                        ground_quality=ground['quality'], ground_height_world_m=ground['height_m'])
            if "ground_plane" in spec["mobility"]:
                from weft.mobile_human import ground_normal
                self.scene["human"]["mobility"].update(
                    floor="camera-relative inclined plane hypothesis; first-pose clearance only",
                    gravity="authored normal alignment, not measured gravity",
                    floor_normal=ground_normal(spec).tolist())

    def _compile_model(self, spec):
        return mujoco.MjModel.from_xml_string(human_xml(spec))

    def observation(self):
        return np.array([self.targets[self.selected_joint] - self.data.qpos[self._q[self.selected_joint]],
                         self.data.qvel[self._v[self.selected_joint]]], dtype=np.float32)

    @property
    def target_angle(self):
        return self.targets[self.selected_joint]

    def select_joint(self, name):
        if name not in self.joint_names:
            raise ValueError("Unknown human joint")
        self.selected_joint = name
        self.scene["controlled_id"] = joint_segment(name)[0]
        self.scene["task"].update(selected_joint=name, target_angle=self.target_angle)
        if self._initialized:
            self._done = self.terminated = self.truncated = self.success = False
        return self.observation()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {"jitter", "selected_joint"} or type(options.get("jitter", True)) is not bool:
            raise ValueError("Reset accepts boolean jitter and selected_joint")
        if "selected_joint" in options:
            self.select_joint(options["selected_joint"])
        mujoco.mj_resetData(self.model, self.data)
        for name in self.joint_names:
            value = self.human_spec["reference"][0]["joints"][name]
            if options.get("jitter", True):
                value += float(self.np_random.uniform(-0.25, 0.25))
            self.data.qpos[self._q[name]] = np.clip(value, self.joint_limits[name][0] + 0.01, self.joint_limits[name][1] - 0.01)
        if self.mobility:
            self.mobility.reset(self.data)
        mujoco.mj_forward(self.model, self.data)
        self.hold_angles = {name: float(self.data.qpos[self._q[name]]) for name in self.joint_names}
        self._hold = np.array([self.hold_angles[name] for name in self.joint_names])
        self.steps = 0
        self._initialized = True
        self._done = self.terminated = self.truncated = self.success = False
        return self.observation(), {"is_success": False, "selected_joint": self.selected_joint}

    def reference_angles(self, time_s):
        origin, end = self._reference_times[0], self._reference_times[-1]
        timestamp = float(np.clip(origin + time_s, origin, end))
        return {name: float(np.interp(timestamp, self._reference_times, values)) for name, values in self._reference_angles.items()}

    def step(self, action):
        if self._done:
            raise RuntimeError("Reset before stepping and after a human episode ends")
        return self._advance(action, task=True)

    def advance_physics(self, action=0):
        if not self._initialized or self.steps >= 1800:
            raise RuntimeError("Human physics replay is bounded to 1800 steps")
        self._advance(action, task=False)
        return self.snapshot()

    def _advance(self, action, task):
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"Human action must be an integer from 0 to {self.action_space.n - 1}")
        if self.mobility:
            self.mobility.prepare(int(action))
        joint_action = 4 if action >= 6 else action
        name = self.selected_joint
        before = abs(self.target_angle - self.data.qpos[self._q[name]])
        reference = (np.array(list(self.reference_angles(self.data.time).values())) if action == 5 else self._hold)
        selected = self.joint_names.index(name)
        compensated = self.human_spec.get("controller") == "bias_compensated_pd"
        for _ in range(8):
            angles, speeds = self.data.qpos[self._q_array], self.data.qvel[self._v_array]
            torques = 40 * (reference - angles) - 5 * speeds
            if compensated:
                if action == 5:
                    timestamp = min(self._reference_times[0] + self.data.time, self._reference_times[-1])
                    index = min(len(self._reference_times) - 2,
                                max(0, int(np.searchsorted(self._reference_times, timestamp, side="right")) - 1))
                    velocity_index = reference_velocity_index(
                        self._reference_times, timestamp, elapsed=self.data.time,
                        timestep=self.model.opt.timestep, source_start=self._reference_times[0])
                    velocity = self._reference_slopes[:, velocity_index] if velocity_index is not None else 0
                    target = self._reference_matrix[:, index] + self._reference_slopes[:, index] * (timestamp - self._reference_times[index])
                    torques = 40 * (target - angles) + 5 * (velocity - speeds)
                torques += self.data.qfrc_bias[self._v_array]
            if joint_action != 5:
                effort = self._motor_limits[selected] * .4
                torques[selected] = (0 if joint_action == 0 else effort if joint_action == 1 else -effort if joint_action == 2 else
                                     -8 * speeds[selected] if joint_action == 3 else
                                     40 * (self._hold[selected] - angles[selected]) - 5 * speeds[selected]
                                     + (self.data.qfrc_bias[self._v_array[selected]] if compensated else 0))
            if self.mobility and not self.mobility.support_enabled:
                torques[:] = 0
            self.data.ctrl[self._a_array] = np.clip(torques, -self._motor_limits, self._motor_limits)
            if self.mobility:
                self.mobility.apply(int(action))
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)) or np.max(np.abs(self.data.qvel)) > 30:
            self._done = True
            raise RuntimeError("Human dynamics exceeded finite state/speed bounds")
        self.steps += 1
        error, speed = map(float, self.observation())
        self.success = bool(task and abs(error) <= 0.08 and abs(speed) <= 0.15)
        self.terminated = self.success
        self.truncated = bool(task and self.steps >= 240 and not self.success)
        self._done = self.terminated or self.truncated
        reward = 5 * (before - abs(error)) - 0.02 - 0.01 * abs(speed) + (5 if self.success else 0)
        return self.observation(), float(reward), self.terminated, self.truncated, {
            "is_success": self.success, "angle_error": error, "angular_speed": speed, "selected_joint": name,
            "motor_torque": float(self.data.ctrl[self._actuator_ids[name]])}

    def snapshot(self):
        if not self._initialized:
            raise RuntimeError("Reset before requesting a human snapshot")
        return self._snapshot(self.data)

    def reference_snapshot(self, index):
        if type(index) is not int or not 0 <= index < len(self.human_spec["reference"]):
            raise ValueError("Reference frame index is outside the observed samples")
        frame = self.human_spec["reference"][index]
        data = mujoco.MjData(self.model)
        for name in self.joint_names:
            data.qpos[self._q[name]] = frame["joints"][name]
        if self.mobility:
            self.mobility.reference(data, index)
        data.time = frame["time_s"] - self.human_spec["reference"][0]["time_s"]
        mujoco.mj_forward(self.model, data)
        snapshot = self._snapshot(data)
        snapshot.update(steps=index, time_s=data.time, success=False, terminated=False, truncated=False,
                        source_time_s=frame["time_s"], scope="Kinematic reference projection onto authored body model; not dynamic simulation")
        snapshot["human"]["reference"] = True
        return snapshot

    def _snapshot(self, data):
        points = {name: data.site_xpos[index].copy() for name, index in self._site_ids.items()}
        velocities = {}
        for name, index in self._site_ids.items():
            spatial = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_SITE, index, spatial, 0)
            velocities[name] = spatial[3:]
        segments, objects, muscles = [], [], []
        for identifier, a, b, radius in self.segment_defs:
            segments.append({"id": identifier, "a": points[a].tolist(), "b": points[b].tolist(), "radius": radius})
            center = (points[a] + points[b]) / 2
            velocity = (velocities[a] + velocities[b]) / 2
            objects.append({"id": identifier, "position": [float(0.5 + (center[0] + center[1] * 0.5) / self.height * 0.75),
                                                               float(1.15 - center[2] / self.height)],
                            "velocity": [float((velocity[0] + velocity[1] * 0.5) / self.height * 0.75),
                                         float(-velocity[2] / self.height)],
                            "angle": 0, "shape": "circle", "radius": radius / self.height,
                            "color": "#628A8C", "label": identifier.replace("_", " ")})
        for name in self.joint_names:
            _, start, end = joint_segment(name)
            a, b = points[start], points[end]
            command = float(data.ctrl[self._actuator_ids[name]]) / self.human_spec.get("motor_limits", {}).get(name, 30)
            for antagonist, sign in (("flexor", 1), ("extensor", -1)):
                offset = np.array([sign * 0.025, sign * 0.025, 0])
                muscles.append({"name": f"{name}_{antagonist}", "a": (a + offset).tolist(), "b": (b + offset).tolist(),
                                "activation": max(0.0, sign * command), "status": "illustrative"})
        result = {"objects": objects, "target": self.scene["target"], "controlled_id": self.scene["controlled_id"],
                "steps": getattr(self, "steps", 0), "time_s": float(data.time), "success": getattr(self, "success", False),
                "terminated": getattr(self, "terminated", False), "truncated": getattr(self, "truncated", False),
                "human": {"segments": segments, "muscles": muscles, "height": self.height, **self.view,
                          "joints": {name: value.tolist() for name, value in points.items()},
                          "angles": {name: float(data.qpos[self._q[name]]) for name in self.joint_names},
                          "angular_velocities": {name: float(data.qvel[self._v[name]]) for name in self.joint_names},
                          "selected_joint": self.selected_joint, "target_angle": self.target_angle,
                          "target_source": "Authored task: reach the inferred reference's 75th-percentile angle, clipped inside model joint limits",
                          "muscle_status": "Illustrative motor-command proxies; not physiological muscle activation"}}
        if self.mobility:
            result["human"]["mobility"] = self.mobility.snapshot(data)
            if 'camera_reference' in self.human_spec:
                center = data.qpos[self.mobility.q].copy()
                center[2] += self.human_spec['mobility']['base_height'] + self.human_spec['lengths']['torso']/2
                result['human'].update(view_mode='body_follow', view_height=self._body_view_height,
                                       view_origin=center.tolist())
        return result

    def render(self):
        from weft.recording import render_snapshot
        return render_snapshot(self.snapshot(), self.scene, width=640)

    def close(self):
        self._done = True
        self._initialized = False


def evaluate_reference(spec, out):
    env = HumanEnv(spec)
    env.reset(seed=0, options={"jitter": False})
    rows = [env.snapshot()]
    errors, root_errors, penetrations = [], [], []
    joint_errors = {name: [] for name in env.joint_names}
    reference_duration = spec["reference"][-1]["time_s"] - spec["reference"][0]["time_s"]
    root_reference = np.array([row["translation"] for row in spec["root_reference"]]).T if env.mobility else None
    duration = min(60.0, 1800 * spec["dt"], reference_duration)
    for _ in range(max(1, round(duration / spec["dt"]))):
        state = env.advance_physics(5)
        expected = env.reference_angles(state["time_s"])
        if env.mobility:
            root = state["human"]["mobility"]
            timestamp = spec["reference"][0]["time_s"] + state["time_s"]
            target = [np.interp(timestamp, env._reference_times, values) for values in root_reference]
            root_errors.append(float(np.sum((np.array(root["translation"]) - target) ** 2)))
            penetrations.append(root["max_penetration"])
        for name in env.joint_names:
            squared_error = (state["human"]["angles"][name] - expected[name]) ** 2
            errors.append(squared_error)
            joint_errors[name].append(squared_error)
        rows.append(state)
    result = {"reference_servo_rmse_radians": float(np.sqrt(np.mean(errors))), "frames": len(rows),
              "model_profile": spec["profile"], "joint_axes": list(env.joint_names),
              "comparison_boundary": "Different joint models or rotation coordinates are not matched servo-error benchmarks",
              "per_joint_rmse_radians": {name: float(np.sqrt(np.mean(values))) for name, values in joint_errors.items()},
              "reference_duration_s": reference_duration, "evaluated_duration_s": rows[-1]["time_s"],
              "duration_coverage_fraction": min(1.0, rows[-1]["time_s"] / reference_duration),
              "scope": "Authored servo tracking of inferred pose angles; not recovered muscle forces or independent physical validation"}
    if env.mobility:
        result.update(root_reference_rmse_model_units=float(np.sqrt(np.mean(root_errors))),
                      maximum_floor_penetration_model_units=max(penetrations, default=0),
                      root_scope="Uncalibrated image-derived translation with authored virtual support and floor; not physical balance or contact-force validation")
    write_json(Path(out) / "human-reference-rollout.json", {"result": result, "states": rows})
    write_json(Path(out) / "human-evaluation.json", result)
    env.close()
    return result
