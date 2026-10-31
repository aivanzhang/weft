"""A fixed analytical load-bearing example, not a reconstructed exercise or training result."""

from copy import deepcopy
from pathlib import Path

from weft.human import validate_human
from weft.io import canonical_hash, digest, local_time, write_json
from weft.mobile_human import MOBILITY_DEFAULTS
from weft.shared_support_task import build_shared_support_task
from weft.shared_world import STATIC_SUPPORT_PROFILE, validate_shared_world
from weft.spatial_pose import SPATIAL_JOINT_NAMES
from weft.whole_body import SUCCESS_LIMITS


def static_support_fixture(*, lowered=False):
    """Only support height changes between the two predeclared two-second conditions."""
    if type(lowered) is not bool:
        raise ValueError("lowered must be a boolean")
    joints = {name: 0. for name in SPATIAL_JOINT_NAMES}
    human = validate_human({
        "schema_version": 3, "profile": "exercise-v3", "dt": 1 / 30,
        "controller": "bias_compensated_pd",
        "motor_limits": {name: 120. if name.startswith("torso_") else 30. for name in SPATIAL_JOINT_NAMES},
        "lengths": {"torso": .5, "upper_arm": .28, "forearm": .25, "thigh": .42,
                    "shin": .4, "shoulder_width": .36, "hip_width": .24},
        "reference": [{"time_s": index / 30, "joints": dict(joints)} for index in range(61)],
        "root_reference": [{"time_s": index / 30, "translation": [0., 0., 0.]} for index in range(61)],
        "mobility": deepcopy(MOBILITY_DEFAULTS),
        "body_frame": {"schema_version": 1, "status": "assumed", "evidence_ids": [], "reference": [],
            "rotation": [[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]],
            "scope": "Authored horizontal body frame; no camera or anatomy recovery"},
        "assumptions": ["Analytical horizontal rest posture with authored joints, shape and mass.",
                        "Motor commands are illustrative, not measured muscle activity."],
    })
    scene = {
        "schema_version": 1, "profile": STATIC_SUPPORT_PROFILE, "physical_validation": "unverified",
        "learned_policy_available": False, "dt": 1 / 30, "substeps": 16,
        "bounds": [2., 2.4, 2.4], "gravity": [0., 0., -9.81],
        "contact": {"time_constant": .012, "damping_ratio": 1.},
        "coordinate_frame": {"axes": "x_right_y_depth_z_up",
            "units": {"length": "world_meter_prior", "mass": "kilogram_prior", "time": "second",
                      "orientation": "quaternion_wxyz", "angular_velocity": "world_radians_per_second"},
            "calibration_status": "unverified",
            "scale_bridge": {"body_height_to_world_scale": 1., "origin_world": [1., 1., 0.],
                "method": "analytic_unit_scale_v1", "uncertainty_note": "Authored unit scale, not measured geometry."},
            "clock_binding": {"control_dt_s": 1 / 30, "physics_substeps": 16, "source_fps": 30.,
                "source_time_offset_s": 0., "alignment_method": "analytic_synchronous_30hz_v1"}},
        "source_binding": {"mode": "analytic_fixture", "source_sha256": None, "total_source_frames": None,
                           "entity_bindings": [], "evidence_documents": {}},
        "human_spec": human,
        "objects": [{"id": "support", "label": "Authored rest support", "role": "static", "shape": "box",
            "position": [1., .75, .8895 - (.6 if lowered else 0.)], "quaternion": [1., 0., 0., 0.],
            "velocity": [0., 0., 0.], "angular_velocity": [0., 0., 0.], "size": [.6, .5, .04],
            "mass": 10., "friction": [.7, .02, .015], "color": "#D8C9BA",
            "provenance": {"scope": "authored_static_support", "physical_validation": "unverified"}}],
        "controlled_id": "human", "target": None,
        "task": {"type": "human_support_v1", "authored_addition": True, "start_offset_s": 0.,
                 "horizon_steps": 60, "human_limits": dict(SUCCESS_LIMITS), "object_goal": None},
        "assumptions": [
            "Authored two-second horizontal-rest fixture, not video reconstruction or a learned policy.",
            "Nineteen explicit joint torques, including authored torso actuators; no translational root forces, object actuation, grasp or tether.",
            "The existing body template and generalized torso actuation do not establish free-floating balance or human biomechanics.",
            "Support load is required after the first 0.2 seconds; all tracking/floor limits apply throughout.",
            "The matched counterfactual changes only support position Z by -0.6 world units.",
            "Source-contact and physical accuracy remain unverified; shape, material and scale are priors.",
        ],
    }
    scene = validate_shared_world(scene)
    task = build_shared_support_task(scene, root_assistance=False, support_requirements=[{
        "object_id": "support", "human_segment": "torso", "direction_world": [0., 0., 1.],
        "minimum_load_body_weight_fraction": .2, "start_step": 6, "end_step": 60,
    }])
    return scene, task


def write_static_support_fixture(out, *, lowered=False):
    """Seal an unexecuted fixture; lab admission still checks initial collision geometry."""
    out = Path(out).absolute()
    if ".." in out.parts or any(part.is_symlink() for part in (out, *out.parents)) or out.exists():
        raise ValueError("Choose a new fixture directory without symlinks or traversal")
    scene, task = static_support_fixture(lowered=lowered)
    out.mkdir(parents=True, exist_ok=False)
    try:
        for name, value in {"scene.json": scene, "human.json": scene["human_spec"],
                            "exercise-task.json": task, "task.json": scene["task"]}.items():
            write_json(out / name, value)
        write_json(out / "manifest.json", {"schema_version": 1, "created_at": local_time(), "status": "ready",
            "profile": STATIC_SUPPORT_PROFILE, "simulation_ready": False,
            "files": {path.name: digest(path) for path in sorted(out.iterdir())}})
    except Exception as error:
        write_json(out / "failure.json", {"status": "failed", "message": str(error)[:2000]})
        raise
    return {"out": str(out), "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
            "manifest_sha256": digest(out / "manifest.json"), "simulation_ready": False,
            "scope": "Unexecuted analytical static-support fixture; no video or physical validation."}
