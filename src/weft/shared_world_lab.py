"""Construct, adapt, and package coupled shared-world-v1 runs into inspectable local labs."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time

import numpy as np

from weft.human import validate_human
from weft.io import canonical_hash, digest, local_time, write_json
from weft.mobile_human import MOBILITY_DEFAULTS, calibrate_floor, ground_normal
from weft.pipeline import verify_artifacts
from weft.observations import adapt_run
from weft.shared_world_evidence import build_shared_world_binding, verify_shared_world_sources
from weft.shared_world import (
    OBJECT_ACTIONS,
    PROFILE,
    SharedWorldEnv,
    audit_shared_world_feasibility,
    validate_shared_world,
)
from weft.shared_world_learning import validate_shared_policy
from weft.shared_world_packaging import (
    collect_shared_source_files, copy_package_file, read_package_file, validate_package_layout, write_package_files,
)
from weft.spatial_pose import SPATIAL_JOINT_NAMES
from weft.whole_body import SUCCESS_LIMITS


REJECTED_SOURCE_STATUSES = frozenset({"failed", "error", "unsupported", "rejected", "cancelled"})
REQUIRED_POLICY_FILES = frozenset(
    {"scene.json", "policy.json", "training.json", "rl-evaluation.json", "policy-rollout.json"}
)


def _analytic_mobile_human_spec():
    """Build a deterministic 30-Hz mobile exercise-v3 human spec performing a chest-height push."""
    lengths = {
        "torso": 0.50,
        "upper_arm": 0.28,
        "forearm": 0.25,
        "thigh": 0.42,
        "shin": 0.40,
        "shoulder_width": 0.36,
        "hip_width": 0.24,
    }
    reference = []
    root_reference = []
    n_frames = 61
    for idx in range(n_frames):
        t = idx / 30.0
        # Smooth chest-height forward push over 0..0.55s, then elevated hold/retract keeping forearm above support.
        push_phase = min(1.0, t / 0.55)
        smooth = 0.5 - 0.5 * math.cos(math.pi * push_phase)
        retract = 0.12 * max(0.0, min(1.0, (t - 0.80) / 0.50))

        joints = {name: 0.0 for name in SPATIAL_JOINT_NAMES}
        joints["torso_flex"] = -0.14 * smooth
        joints["left_hip"] = 0.08 + 0.04 * smooth
        joints["right_hip"] = 0.08 + 0.04 * smooth
        joints["left_knee"] = 0.16 + 0.08 * smooth
        joints["right_knee"] = 0.16 + 0.08 * smooth
        joints["left_shoulder"] = 0.18
        joints["left_elbow"] = 0.35
        joints["right_shoulder"] = 0.15 + 0.70 * smooth - retract
        joints["right_elbow"] = 1.25 - 0.75 * smooth + 1.5 * retract

        reference.append({"time_s": t, "joints": joints})
        root_z = -0.008 * smooth if idx > 0 else 0.0
        root_reference.append({"time_s": t, "translation": [0.0, 0.0, float(root_z)]})

    spec = {
        "schema_version": 3,
        "profile": "exercise-v3",
        "dt": 1.0 / 30.0,
        "controller": "bias_compensated_pd",
        "motor_limits": {name: 120.0 if name.startswith("torso_") else 30.0 for name in SPATIAL_JOINT_NAMES},
        "lengths": lengths,
        "reference": reference,
        "root_reference": root_reference,
        "mobility": deepcopy(MOBILITY_DEFAULTS),
        "assumptions": [
            "Analytic 19-joint + 3-root mobile body fixture for coupled human-object simulation.",
            "Virtual root motors and first-pose floor clearance are explicit assumptions, not unassisted balance.",
            "Muscle bands represent motor-command proxies, not physiological muscle activation.",
        ],
    }
    return validate_human(calibrate_floor(validate_human(spec)))


def authored_shared_world_scene():
    """Return a validated shared-world-v1 coupled human + multi-object 3D scene fixture."""
    human_spec = _analytic_mobile_human_spec()
    origin_world = [0.90, 0.60, 0.0]
    base_quat = [1.0, 0.0, 0.0, 0.0]
    zero3 = [0.0, 0.0, 0.0]

    # Measure exact wrist position at t=0 so the ball sits 2.2 cm in front of the right forearm above the table.
    sh0 = human_spec["reference"][0]["joints"]["right_shoulder"]
    el0 = human_spec["reference"][0]["joints"]["right_elbow"]
    sh_x = origin_world[0] + human_spec["lengths"]["shoulder_width"] / 2.0
    sh_y = origin_world[1]
    sh_z = origin_world[2] + human_spec["mobility"]["base_height"] + human_spec["lengths"]["torso"]
    wrist_y0 = sh_y + human_spec["lengths"]["upper_arm"] * math.sin(sh0) + human_spec["lengths"]["forearm"] * math.sin(
        sh0 + el0
    )
    wrist_z0 = sh_z - human_spec["lengths"]["upper_arm"] * math.cos(sh0) - human_spec["lengths"]["forearm"] * math.cos(
        sh0 + el0
    )

    ball_radius = 0.055
    table_top_z = round(wrist_z0 - 0.035, 4)
    table_thickness = 0.04
    ball_y0 = round(wrist_y0 + 0.027 + ball_radius + 0.018, 4)
    ball_z0 = round(table_top_z + ball_radius, 4)
    target_y = round(ball_y0 + 0.27, 4)

    scene = {
        "schema_version": 1,
        "profile": PROFILE,
        "physical_validation": "unverified",
        "dt": 1.0 / 30.0,
        "substeps": 16,
        "bounds": [1.80, 2.00, 2.40],
        "gravity": [0.0, 0.0, -9.81],
        "contact": {"time_constant": 0.012, "damping_ratio": 1.0},
        "coordinate_frame": {
            "axes": "x_right_y_depth_z_up",
            "units": {
                "length": "world_meter_prior",
                "mass": "kilogram_prior",
                "time": "second",
                "orientation": "quaternion_wxyz",
                "angular_velocity": "world_radians_per_second",
            },
            "calibration_status": "unverified",
            "scale_bridge": {
                "body_height_to_world_scale": 1.0,
                "origin_world": origin_world,
                "method": "analytic_unit_scale_v1",
                "uncertainty_note": "Analytic coupled fixture; body and rigid objects share authored 1:1 world units.",
            },
            "clock_binding": {
                "control_dt_s": 1.0 / 30.0,
                "physics_substeps": 16,
                "source_fps": 30.0,
                "source_time_offset_s": 0.0,
                "alignment_method": "analytic_synchronous_30hz_v1",
            },
        },
        "source_binding": {
            "mode": "analytic_fixture",
            "evidence_documents": {},
            "source_sha256": None,
            "total_source_frames": None,
            "entity_bindings": [],
        },
        "human_spec": human_spec,
        "objects": [
            {
                "id": "ball",
                "label": "Target sphere",
                "role": "passive",
                "shape": "sphere",
                "position": [round(sh_x, 4), ball_y0, ball_z0],
                "quaternion": base_quat,
                "velocity": zero3,
                "angular_velocity": zero3,
                "radius": ball_radius,
                "mass": 0.45,
                "friction": [0.60, 0.005, 0.004],
                "color": "#965644",
                "provenance": {"scope": "authored_analytic_fixture", "physical_validation": "unverified"},
            },
            {
                "id": "marker_box",
                "label": "Companion block",
                "role": "passive",
                "shape": "box",
                "position": [round(origin_world[0] - 0.16, 4), round(target_y + 0.08, 4), round(table_top_z + 0.035, 4)],
                "quaternion": base_quat,
                "velocity": zero3,
                "angular_velocity": zero3,
                "size": [0.07, 0.07, 0.07],
                "mass": 0.60,
                "friction": [0.55, 0.01, 0.005],
                "color": "#C7795D",
                "provenance": {"scope": "authored_analytic_fixture", "physical_validation": "unverified"},
            },
            {
                "id": "table",
                "label": "Support pedestal",
                "role": "static",
                "shape": "box",
                "position": [
                    round(origin_world[0] + 0.02, 4),
                    round(ball_y0 + 0.22, 4),
                    round(table_top_z - table_thickness / 2.0, 4),
                ],
                "quaternion": base_quat,
                "velocity": zero3,
                "angular_velocity": zero3,
                "size": [0.64, 0.62, table_thickness],
                "mass": 10.0,
                "friction": [0.65, 0.005, 0.004],
                "color": "#D8C9BA",
                "provenance": {"scope": "authored_support_surface", "physical_validation": "unverified"},
            },
        ],
        "controlled_id": "human",
        "target": [round(sh_x, 4), target_y, ball_z0],
        "task": {
            "type": "coupled_human_object_v1",
            "authored_addition": True,
            "start_offset_s": 0.0,
            "horizon_steps": 45,
            "human_limits": deepcopy(SUCCESS_LIMITS),
            "object_goal": {
                "target_object_id": "ball",
                "position_tolerance": 0.06,
                "speed_tolerance": 0.10,
                "virtual_acceleration": 6.0,
                "require_human_contact": True,
            },
        },
        "assumptions": [
            "Coupled articulated-human and 3D rigid-object simulation in one MuJoCo model and clock.",
            "Human limb capsules collide bidirectionally with dynamic rigid objects; adjacent human self-collision remains disabled.",
            "Virtual root motors and optional virtual object forces are explicit; physical validation remains unverified.",
        ],
    }
    return validate_shared_world(scene)


def _probe_shared_interventions(scene):
    """Verify coupled human feedback and 8-direction rigid-object force interventions."""
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        initial = env.snapshot()
        # Probe 1: 15 steps of authored human reference feedback.
        for _ in range(15):
            if env._done:
                break
            env.step(env.authored_action())
        after_human = env.snapshot()

        # Probe 2: 8 discrete 3D object actions applied to target_object_id.
        target_obj_id = scene["task"]["object_goal"]["target_object_id"]
        obj_probes = {}
        for code, name in enumerate(OBJECT_ACTIONS):
            env.reset(seed=0, options={"jitter": False})
            vec = env.manual_action_vector(object_id=target_obj_id, object_action_code=code)
            _, reward, term, trunc, info = env.step(vec)
            snap = env.snapshot()
            obj_state = next(o for o in snap["objects"] if o["id"] == target_obj_id)
            obj_probes[name] = {
                "action_code": code,
                "position": obj_state["position"],
                "velocity": obj_state["velocity"],
                "virtual_force": obj_state["virtual_force"],
                "reward": float(reward),
                "terminated": bool(term),
                "truncated": bool(trunc),
            }
        return {
            "schema_version": 1,
            "profile": "shared-world-interventions-v1",
            "created_at": local_time(),
            "initial_target_object_position": next(
                o["position"] for o in initial["objects"] if o["id"] == target_obj_id
            ),
            "after_15_authored_human_steps": {
                "steps": after_human["steps"],
                "human_object_contacts_total": after_human["coupled_metrics"]["human_object_contacts_total"],
                "max_human_object_normal_force": after_human["coupled_metrics"]["max_human_object_normal_force"],
                "contact_force_units": after_human["coupled_metrics"]["contact_force_units"],
                "target_object_position": next(
                    o["position"] for o in after_human["objects"] if o["id"] == target_obj_id
                ),
            },
            "object_force_probes": obj_probes,
            "physical_validation": "unverified",
            "scope": "Simulator actuation and coupled contact verification only; not real-world accuracy.",
        }
    finally:
        env.close()


def _reference_frames_artifact(scene):
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        count = min(180, len(scene["human_spec"]["reference"]))
        frames = [env.reference_snapshot(i) for i in range(count)]
        return {
            "schema_version": 1,
            "profile": "shared-world-kinematic-reference-v1",
            "frames": frames,
            "scope": "Kinematic human pose projection alongside initial 3D rigid geometry; no physics solver advanced.",
        }
    finally:
        env.close()


def _seal_directory(directory):
    manifest = {
        "schema_version": 1,
        "created_at": local_time(),
        "files": {
            str(path.relative_to(directory)): digest(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path != directory / "manifest.json"
        },
    }
    write_json(directory / "manifest.json", manifest)
    return verify_artifacts(directory)


def write_sealed_shared_world_fixture(out):
    """Write a standalone sealed shared-world-v1 fixture directory."""
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError("Choose a new shared-world fixture directory")
    scene = authored_shared_world_scene()
    scene["learned_policy_available"] = False
    scene = validate_shared_world(scene)
    feasibility = audit_shared_world_feasibility(scene)
    interventions = _probe_shared_interventions(scene)
    ref_frames = _reference_frames_artifact(scene)

    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "scene.json", scene)
    write_json(out / "human.json", scene["human_spec"])
    write_json(out / "task.json", scene["task"])
    write_json(out / "reference-feasibility.json", feasibility)
    write_json(out / "interventions.json", interventions)
    write_json(out / "human-reference.json", ref_frames)
    write_json(
        out / "reconstruction.json",
        {
            "schema_version": 1,
            "profile": PROFILE,
            "mode": "authored_coupled_fixture",
            "physical_validation": "unverified",
            "reference_feasibility": {k: feasibility[k] for k in ("status", "summary")},
            "scope": "Analytic coupled articulated-human and 3D rigid-object fixture; not reconstructed from a video.",
        },
    )
    write_json(out / "repair-ledger.json", {"proposals": [], "scope": "No repair loop executed on analytic fixture."})
    return _seal_directory(out)


def adapt_to_shared_world(human_source, out, *, scale=1.0):
    """Adapt an immutable mobile exercise-v3 run/lab into a source-qualified shared-world-v1 run."""
    human_source, out = Path(human_source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Choose a new output directory for shared-world adaptation")
    if out == human_source or out.is_relative_to(human_source):
        raise ValueError("Keep adapted shared-world output outside the immutable source directory")

    run_dir = human_source / "run" if (human_source / "run" / "human.json").is_file() else human_source
    if not (run_dir / "manifest.json").is_file() or not (run_dir / "human.json").is_file():
        raise ValueError("Source must be a sealed run or lab containing human.json and manifest.json")
    source_verified = verify_artifacts(run_dir)
    source_manifest = json.loads((run_dir / "manifest.json").read_text())
    if not {"human.json", "scene.json", "evidence.json"} <= set(source_manifest["files"]):
        raise ValueError("Source-qualified adaptation requires sealed human, scene and observation evidence, not counts alone")
    human_spec = validate_human(json.loads((run_dir / "human.json").read_text()))
    if human_spec.get("profile") != "exercise-v3" or "camera_reference" in human_spec:
        raise ValueError("shared-world adaptation requires a validated 30-Hz mobile exercise-v3 human.json")

    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0.5 <= scale <= 2.0:
        raise ValueError("Shared-world scale prior must be finite and within [0.5, 2.0]")
    scale = float(scale)
    roots = np.asarray([row["translation"] for row in human_spec["root_reference"]], dtype=float) * scale
    span_xy = np.maximum(0.4, np.ptp(roots[:, :2], axis=0))
    bx = float(round(max(2.0, 1.8 * scale + span_xy[0]), 3))
    by = float(round(max(2.2, 2.0 * scale + span_xy[1]), 3))
    bz = float(round(max(2.4, 2.3 * scale), 3))
    origin_world = [round(bx * 0.5 - float(roots[:, 0].mean()), 4), round(by * 0.34 - float(roots[:, 1].mean()), 4), 0.0]
    origin_world[0] = float(np.clip(origin_world[0], 0.45, bx - 0.45))
    origin_world[1] = float(np.clip(origin_world[1], 0.40, by - 0.85))

    gravity = [0.0, 0.0, -9.81]
    if "ground_plane" in human_spec["mobility"]:
        gravity = (-9.81 * ground_normal(human_spec)).tolist()

    # Place a coupled target ball on a pedestal safely in front of the human's initial reach envelope.
    ball_radius = round(0.055 * scale, 4)
    table_top_z = round(max(0.45, 0.58 * scale * human_spec["mobility"]["base_height"]), 4)
    table_y = round(min(by - 0.35, origin_world[1] + 0.68 * scale), 4)
    ball_y = round(table_y - 0.10 * scale, 4)
    target_y = round(min(by - 0.18, ball_y + 0.18 * scale), 4)
    ball_z = round(table_top_z + ball_radius, 4)

    ref_duration = float(human_spec["reference"][-1]["time_s"] - human_spec["reference"][0]["time_s"])
    horizon_steps = max(10, min(60, int(math.floor(ref_duration * 30.0 - 1e-6))))
    observation_document = adapt_run(run_dir)
    source_sha = observation_document["source"]["sha256"]
    if human_spec.get("source_sha256") != source_sha:
        raise ValueError("Human model and observation evidence must bind the same original source")
    raw_observations = (json.dumps(observation_document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    observation_sha = hashlib.sha256(raw_observations).hexdigest()
    source_binding, construction_bytes = build_shared_world_binding(
        {observation_sha: raw_observations}, source_sha256=source_sha,
        world_ids=["human", "ball", "table"],
        entity_refs={"human": (observation_sha, "person-0"), "ball": None, "table": None})
    source_fps = float(observation_document["source"]["fps"])

    base_quat = [1.0, 0.0, 0.0, 0.0]
    zero3 = [0.0, 0.0, 0.0]
    scene = {
        "schema_version": 1,
        "profile": PROFILE,
        "physical_validation": "unverified",
        "learned_policy_available": False,
        "dt": 1.0 / 30.0,
        "substeps": 16,
        "bounds": [bx, by, bz],
        "gravity": gravity,
        "contact": {"time_constant": 0.012, "damping_ratio": 1.0},
        "coordinate_frame": {
            "axes": "x_right_y_depth_z_up",
            "units": {
                "length": "body_scaled_world_units",
                "mass": "kilogram_prior",
                "time": "second",
                "orientation": "quaternion_wxyz",
                "angular_velocity": "world_radians_per_second",
            },
            "calibration_status": "estimated_prior",
            "scale_bridge": {
                "body_height_to_world_scale": scale,
                "origin_world": origin_world,
                "method": "monocular_body_height_world_prior_v1",
                "uncertainty_note": (
                    "Monocular pose proportions and root displacements bridged into 3D world units via explicit scale "
                    "prior; camera depth and object metric scale are not calibrated."
                ),
            },
            "clock_binding": {
                "control_dt_s": 1.0 / 30.0,
                "physics_substeps": 16,
                "source_fps": max(5.0, min(240.0, source_fps)),
                "source_time_offset_s": float(human_spec["reference"][0]["time_s"]),
                "alignment_method": "pose_frame_index_over_fps_prior_v1",
            },
        },
        "source_binding": source_binding,
        "human_spec": human_spec,
        "objects": [
            {
                "id": "ball",
                "label": "Coupled target ball",
                "role": "passive",
                "shape": "sphere",
                "position": [origin_world[0], ball_y, ball_z],
                "quaternion": base_quat,
                "velocity": zero3,
                "angular_velocity": zero3,
                "radius": ball_radius,
                "mass": 0.45,
                "friction": [0.65, 0.02, 0.018],
                "color": "#965644",
                "provenance": {
                    "scope": "authored_coupled_rigid_addition",
                    "source_human_run": str(run_dir),
                    "physical_validation": "unverified",
                },
            },
            {
                "id": "table",
                "label": "Support pedestal",
                "role": "static",
                "shape": "box",
                "position": [origin_world[0], table_y, round(table_top_z - 0.02, 4)],
                "quaternion": base_quat,
                "velocity": zero3,
                "angular_velocity": zero3,
                "size": [round(0.56 * scale, 4), round(0.52 * scale, 4), 0.04],
                "mass": 10.0,
                "friction": [0.70, 0.02, 0.015],
                "color": "#D8C9BA",
                "provenance": {"scope": "authored_support_surface", "physical_validation": "unverified"},
            },
        ],
        "controlled_id": "human",
        "target": [origin_world[0], target_y, ball_z],
        "task": {
            "type": "coupled_human_object_v1",
            "authored_addition": True,
            "start_offset_s": 0.0,
            "horizon_steps": horizon_steps,
            "human_limits": deepcopy(SUCCESS_LIMITS),
            "object_goal": {
                "target_object_id": "ball",
                "position_tolerance": 0.06,
                "speed_tolerance": 0.10,
                "virtual_acceleration": 6.0,
                "require_human_contact": False,
            },
        },
        "assumptions": list(human_spec["assumptions"])
        + [
            "Adapted into shared-world-v1 by coupling the source-linked mobile body with authored 3D rigid objects in one MuJoCo model.",
            "Rigid objects in this adapted world are authored additions with explicit zero observed source slots.",
            "Physical validation remains unverified.",
        ],
    }
    scene = validate_shared_world(scene)
    feasibility = audit_shared_world_feasibility(scene)
    interventions = _probe_shared_interventions(scene)
    ref_frames = _reference_frames_artifact(scene)

    out.mkdir(parents=True, exist_ok=False)
    observation_path = out / source_binding["evidence_documents"][observation_sha]
    observation_path.parent.mkdir()
    observation_path.write_bytes(raw_observations)
    (out / "construction-evidence.json").write_bytes(construction_bytes)
    source_check = verify_shared_world_sources(scene, out)
    write_json(out / "source-binding-verification.json", source_check)
    input_copy = out / "inputs" / "human-source"
    input_copy.mkdir(parents=True)
    for name in ("human.json", "scene.json", "evidence.json", "manifest.json"):
        shutil.copyfile(run_dir / name, input_copy / ("source-manifest.json" if name == "manifest.json" else name))
    for optional_name in ("source.json", "pose-quality.json", "root-motion.json", "floor-assumption.json"):
        if (run_dir / optional_name).is_file():
            shutil.copyfile(run_dir / optional_name, out / optional_name)
    for vid in sorted(run_dir.glob("source.*")):
        if vid.suffix.lower() in (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv") and vid.is_file():
            shutil.copyfile(vid, out / vid.name)

    write_json(out / "scene.json", scene)
    write_json(out / "human.json", scene["human_spec"])
    write_json(out / "task.json", scene["task"])
    write_json(out / "reference-feasibility.json", feasibility)
    write_json(out / "interventions.json", interventions)
    write_json(out / "human-reference.json", ref_frames)
    write_json(
        out / "adaptation-provenance.json",
        {
            "schema_version": 1,
            "created_at": local_time(),
            "source_human_directory": str(run_dir),
            "source_verification": source_verified,
            "human_spec_sha256": canonical_hash(human_spec),
            "scene_sha256": canonical_hash(scene),
            "source_binding": source_binding,
            "scale_bridge": scene["coordinate_frame"]["scale_bridge"],
            "reference_feasibility": {k: feasibility[k] for k in ("status", "summary")},
            "physical_validation": "unverified",
        },
    )
    write_json(
        out / "reconstruction.json",
        {
            "schema_version": 1,
            "profile": PROFILE,
            "mode": "source_adapted_shared_world",
            "source_sha256": source_binding["source_sha256"],
            "reference_feasibility": {k: feasibility[k] for k in ("status", "summary")},
            "physical_validation": "unverified",
            "scope": "Source-linked mobile exercise body coupled with authored 3D rigid geometry in one MuJoCo world.",
        },
    )
    write_json(out / "repair-ledger.json", {"proposals": [], "scope": "No repair loop executed on adapted shared world."})
    return _seal_directory(out)


def prepare_shared_world_lab(source_run, out, *, policy_run=None):
    """Package an immutable sealed shared-world-v1 run (and optional policy run) into a new local lab."""
    started = time.perf_counter()
    source_run, out = Path(source_run).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Choose a new shared-world lab directory")
    if out == source_run or out.is_relative_to(source_run) or source_run.is_relative_to(out):
        raise ValueError("Keep the new shared-world lab outside the immutable source run")
    if policy_run is not None:
        policy_run = Path(policy_run).resolve()
        if out == policy_run or out.is_relative_to(policy_run) or policy_run.is_relative_to(out):
            raise ValueError("Keep the new shared-world lab outside the immutable policy run")

    if (source_run / "status.json").is_file():
        st = json.loads((source_run / "status.json").read_text())
        if st.get("status") in REJECTED_SOURCE_STATUSES:
            raise ValueError(f"Cannot package a {st.get('status')} source lab into shared-world-lab")

    source_dir = source_run / "run" if (source_run / "run" / "manifest.json").is_file() else source_run
    if not (source_dir / "manifest.json").is_file() or not (source_dir / "scene.json").is_file():
        raise ValueError("Shared-world lab import requires a sealed directory containing scene.json and manifest.json")
    source_verified = verify_artifacts(source_dir)
    scene_raw = read_package_file(source_dir, "scene.json", max_bytes=64 * 1024 * 1024)
    scene = validate_shared_world(json.loads(scene_raw))
    source_files = collect_shared_source_files(
        scene, source_dir, expected_manifest_sha256=source_verified["manifest_sha256"])
    source_manifest = json.loads(source_files.manifest_bytes)
    if source_files.manifest_files.get("scene.json") != hashlib.sha256(scene_raw).hexdigest():
        raise ValueError("Lab scene must match its sealed source identity")
    generated_paths = {"manifest.json", "source-input-manifest.json", "policy-input-manifest.json",
                       "implementation", "lab-implementation", "policy-implementation", "scene.json", "human.json",
                       "task.json", "reference-feasibility.json", "interventions.json", "human-reference.json",
                       "repair-ledger.json", "reconstruction.json", "lab-provenance.json", "policy.json",
                       "training.json", "rl-evaluation.json", "policy-rollout.json", "settings.json"}
    source_files.validate_layout(reserved_paths=generated_paths)

    for status_file in ("status.json", "report.json", "reconstruction.json"):
        if (source_dir / status_file).is_file():
            doc = json.loads((source_dir / status_file).read_text())
            if isinstance(doc, dict) and doc.get("status") in REJECTED_SOURCE_STATUSES:
                raise ValueError(f"Source artifact {status_file} has rejected status {doc.get('status')!r}")

    source_binding_check = source_files.report

    policy_verified = None
    policy_manifest = None
    policy_source_check = None
    policy_manifest_bytes = None
    if policy_run is not None:
        policy_verified = verify_artifacts(policy_run)
        policy_scene_raw = read_package_file(policy_run, "scene.json", max_bytes=64 * 1024 * 1024)
        pol_scene = validate_shared_world(json.loads(policy_scene_raw))
        policy_files = collect_shared_source_files(
            pol_scene, policy_run, expected_manifest_sha256=policy_verified["manifest_sha256"])
        policy_files.validate_layout(reserved_paths=generated_paths)
        policy_manifest = json.loads(policy_files.manifest_bytes)
        policy_source_check = policy_files.report
        policy_manifest_bytes = policy_files.manifest_bytes
        if policy_files.manifest_files.get("scene.json") != hashlib.sha256(policy_scene_raw).hexdigest():
            raise ValueError("Policy scene must match its sealed source identity")
        if not REQUIRED_POLICY_FILES <= set(policy_manifest["files"]):
            raise ValueError("Policy run must seal scene.json, policy.json, training.json, rl-evaluation.json, policy-rollout.json")
        pol_doc = json.loads(read_package_file(policy_run, "policy.json", max_bytes=1024 * 1024,
                                              expected_sha256=policy_manifest["files"]["policy.json"]))
        # Compare core scene identity ignoring learned_policy_available flag toggle.
        s_cmp = deepcopy(scene)
        s_cmp["learned_policy_available"] = True
        s_cmp = validate_shared_world(s_cmp)
        if canonical_hash(pol_scene) != canonical_hash(s_cmp):
            raise ValueError("Shared-world policy run was trained on a different scene")
        validate_shared_policy(pol_doc, pol_scene)
        scene = pol_scene
        del policy_files
    elif "policy.json" in source_manifest["files"]:
        pol_doc = json.loads(read_package_file(source_dir, "policy.json", max_bytes=1024 * 1024,
                                              expected_sha256=source_manifest["files"]["policy.json"]))
        scene["learned_policy_available"] = True
        scene = validate_shared_world(scene)
        validate_shared_policy(pol_doc, scene)
    else:
        scene["learned_policy_available"] = False
        scene = validate_shared_world(scene)

    rewritten = {"scene.json", "human.json", "task.json", "reference-feasibility.json", "interventions.json",
                 "human-reference.json", "lab-provenance.json", "source-input-manifest.json", "policy-input-manifest.json"}
    policy_outputs = {"policy.json", "training.json", "rl-evaluation.json", "policy-rollout.json", "settings.json"}
    modules = ("shared_world_lab.py", "shared_world.py", "shared_world_observation.py",
               "shared_world_packaging.py", "shared_world_evidence.py", "source_equivalence.py",
               "world_evidence.py", "observations.py", "human.py", "mobile_human.py", "whole_body.py",
               "body_frame.py", "spatial_pose.py", "root_motion.py", "errors.py", "spatial_rigid.py")
    copied_names = {name for name in source_manifest["files"] if name not in rewritten
                    and not name.startswith(("lab-implementation/", "policy-implementation/"))
                    and not (policy_run is not None and name in policy_outputs)}
    policy_implementation = {"policy-" + name for name in (policy_manifest or {}).get("files", {})
                             if name.startswith("implementation/")}
    validate_package_layout(copied_names | rewritten | {"repair-ledger.json", "reconstruction.json"}
                            | {f"lab-implementation/{name}" for name in modules} | policy_implementation
                            | (policy_outputs if policy_run is not None else set()))

    def sealed_json(name):
        return json.loads(read_package_file(source_dir, name, max_bytes=64 * 1024 * 1024,
                                            expected_sha256=source_manifest["files"][name]))

    feasibility = (
        sealed_json("reference-feasibility.json")
        if "reference-feasibility.json" in source_manifest["files"]
        else audit_shared_world_feasibility(scene)
    )
    interventions = (
        sealed_json("interventions.json")
        if "interventions.json" in source_manifest["files"]
        else _probe_shared_interventions(scene)
    )
    ref_frames = (
        sealed_json("human-reference.json")
        if "human-reference.json" in source_manifest["files"]
        else _reference_frames_artifact(scene)
    )

    out.mkdir(parents=True, exist_ok=False)
    run = out / "run"
    run.mkdir()
    source_files.copy_to(run, reserved_paths=generated_paths)
    for rel in source_manifest["files"]:
        if (rel in source_files.files or rel in rewritten or rel.startswith(("lab-implementation/", "policy-implementation/"))
                or (policy_run is not None and rel in policy_outputs)):
            continue
        copy_package_file(source_dir, run, rel, source_manifest["files"][rel])
    write_package_files(run, {"source-input-manifest.json": source_files.manifest_bytes})
    del source_files

    if policy_run is not None:
        for name in policy_outputs:
            if name in policy_manifest["files"]:
                copy_package_file(policy_run, run, name, policy_manifest["files"][name])
        for name, sha in policy_manifest["files"].items():
            if name.startswith("implementation/"):
                copy_package_file(policy_run, run, name, sha, target_name="policy-" + name)
        write_package_files(run, {"policy-input-manifest.json": policy_manifest_bytes})

    implementation = {name: (Path(__file__).parent / name).read_bytes() for name in modules}
    write_package_files(run, {f"lab-implementation/{name}": raw for name, raw in implementation.items()})

    write_json(run / "scene.json", scene)
    write_json(run / "human.json", scene["human_spec"])
    write_json(run / "task.json", scene["task"])
    write_json(run / "reference-feasibility.json", feasibility)
    write_json(run / "interventions.json", interventions)
    write_json(run / "human-reference.json", ref_frames)
    if not (run / "repair-ledger.json").is_file():
        write_json(run / "repair-ledger.json", {"proposals": [], "scope": "No repair loop executed during lab packaging."})
    if not (run / "reconstruction.json").is_file():
        write_json(
            run / "reconstruction.json",
            {
                "schema_version": 1,
                "profile": PROFILE,
                "physical_validation": "unverified",
                "reference_feasibility": {k: feasibility[k] for k in ("status", "summary")},
            },
        )
    packaged_source_check = verify_shared_world_sources(scene, run)
    provenance = {
        "schema_version": 1,
        "created_at": local_time(),
        "source_run": str(source_dir),
        "source_verification": source_verified,
        "source_binding_verification": source_binding_check,
        "packaged_source_verification": packaged_source_check,
        "policy_run": str(policy_run) if policy_run is not None else None,
        "policy_verification": policy_verified,
        "policy_source_verification": policy_source_check,
        "packaging_module_hashes": {name: hashlib.sha256(raw).hexdigest() for name, raw in implementation.items()},
        "scene_sha256": canonical_hash(scene),
        "learned_policy_available": bool(scene.get("learned_policy_available", False)),
        "preparation_seconds": time.perf_counter() - started,
        "physical_validation": "unverified",
        "scope": "Packaged immutable shared-world-v1 artifacts into a new local lab directory; original inputs unchanged.",
    }
    write_json(run / "lab-provenance.json", provenance)
    sealed = _seal_directory(run)

    lab_config = {
        "schema_version": 1,
        "created_at": local_time(),
        "profile": "shared-world",
        "source_run": str(source_dir),
        "policy_run": str(policy_run) if policy_run is not None else None,
        "learned_policy_available": bool(scene.get("learned_policy_available", False)),
        "reference_feasibility": {k: feasibility[k] for k in ("status", "summary")},
    }
    write_json(out / "lab-config.json", lab_config)
    event = {
        "stage": "shared_world_import",
        "message": "Verified coupled human + 3D rigid-object world and sealed lab artifacts",
        "time": local_time(),
    }
    write_json(
        out / "status.json",
        {
            "status": "ready",
            "events": [event],
            "result": {
                "profile": PROFILE,
                "manifest_sha256": sealed["manifest_sha256"],
                "learned_policy_available": bool(scene.get("learned_policy_available", False)),
                "reference_feasibility": {k: feasibility[k] for k in ("status", "summary")},
            },
        },
    )
    (out / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    return {
        "lab_directory": str(out),
        "run_directory": str(run),
        "manifest_sha256": sealed["manifest_sha256"],
        "learned_policy_available": bool(scene.get("learned_policy_available", False)),
        "reference_feasibility": feasibility["status"],
        "seconds": time.perf_counter() - started,
    }
