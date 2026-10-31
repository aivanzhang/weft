"""Shared environment, control and RL task capability descriptor across Weft profiles."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts


CAPABILITY_PROFILE = "environment-capabilities-v1"
REPRESENTATION_FAMILIES = frozenset(
    {
        "rigid_1d_support",
        "rigid_planar_2d",
        "rigid_spatial_3d",
        "articulated_human_3d",
        "mixed_rigid_articulated_3d",
    }
)
PHYSICS_BACKENDS = frozenset(
    {
        "analytic_rolling_1d",
        "pymunk_2d",
        "mujoco_3d_rigid",
        "mujoco_3d_articulated",
        "mujoco_3d_shared_world",
    }
)


def planar_training_enabled(directory, *, lab_config=None):
    """Require a sealed explicit opt-out before a video lab can omit its policy."""
    root = Path(directory)
    path = root / "run-config.json"
    settings = json.loads(path.read_text()) if path.is_file() else {}
    if not isinstance(settings, dict):
        raise ValueError("Planar run settings must be a JSON object")
    enabled = settings.get("training_enabled", True)
    if type(enabled) is not bool:
        raise ValueError("Planar training_enabled must be a boolean")
    if lab_config is not None:
        configured = lab_config.get("training_enabled", True)
        if type(configured) is not bool or configured is not enabled:
            raise ValueError("Lab training mode differs from its sealed run settings")
    if not enabled:
        manifest = json.loads((root / "manifest.json").read_text())
        if (manifest.get("files", {}).get("run-config.json") != digest(path)
                or settings.get("profile") not in ("planar", "appearance")
                or (lab_config is not None and settings["profile"] != lab_config.get("profile"))
                or settings.get("episodes") is not None or settings.get("eval_episodes") is not None):
            raise ValueError("Manual planar mode requires matching sealed no-training settings")
        policy = root / "policy.json"
        if policy.exists() or policy.is_symlink():
            raise ValueError("A manual-only planar run must not contain a policy")
    return enabled


def validate_capabilities(doc):
    """Validate an environment-capabilities-v1 document and return a defensive copy."""
    if not isinstance(doc, dict):
        raise ValueError("Capabilities document must be a JSON object")
    required = {
        "schema_version",
        "profile",
        "scene_profile",
        "lab_profile",
        "representation_family",
        "physics_backend",
        "coordinate_frame",
        "entities",
        "action_contract",
        "observation_contract",
        "task_contract",
        "reset_contract",
        "controllers",
        "evidence_separation",
        "unsupported_capabilities",
    }
    if set(doc) != required:
        raise ValueError(f"Capabilities fields differ from contract: {sorted(set(doc) ^ required)}")
    if type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
        raise ValueError("Capabilities schema_version must be integer 1")
    if doc["profile"] != CAPABILITY_PROFILE:
        raise ValueError(f"Capabilities profile must be {CAPABILITY_PROFILE!r}")
    if doc["representation_family"] not in REPRESENTATION_FAMILIES:
        raise ValueError(f"Unsupported representation_family: {doc['representation_family']!r}")
    if doc["physics_backend"] not in PHYSICS_BACKENDS:
        raise ValueError(f"Unsupported physics_backend: {doc['physics_backend']!r}")
    frame = doc["coordinate_frame"]
    if not isinstance(frame, dict) or frame.get("dimensions") not in (2, 3) or not isinstance(frame.get("units"), dict):
        raise ValueError("Capabilities coordinate_frame must specify 2 or 3 dimensions and explicit units")
    entities = doc["entities"]
    base_profile = (doc["task_contract"].get("base_scene_profile")
                    if doc["scene_profile"] == "shared-world-v5" and isinstance(doc["task_contract"], dict)
                    else doc["scene_profile"])
    if doc["scene_profile"] == "shared-world-v5" and base_profile not in ("shared-world-v2", "shared-world-v3", "shared-world-v4"):
        raise ValueError("Feet capabilities require their explicit base scene profile")
    expected_mixed = base_profile in ("shared-world-v1", "shared-world-v2", "shared-world-v4")
    if (
        not isinstance(entities, dict)
        or type(entities.get("total_count")) is not int
        or entities["total_count"] < 1
        or entities.get("mixed_rigid_articulated") is not expected_mixed
        or not isinstance(entities.get("controllable_actors"), list)
        or not entities["controllable_actors"]
    ):
        raise ValueError("Capabilities entities must list controllable actors and keep mixed_rigid_articulated explicit")
    action = doc["action_contract"]
    if not isinstance(action, dict) or action.get("space_type") not in ("discrete", "continuous_box") or type(action.get("dimension")) is not int:
        raise ValueError("Capabilities action_contract must declare space_type and dimension")
    obs = doc["observation_contract"]
    if not isinstance(obs, dict) or obs.get("space_type") != "continuous_box" or type(obs.get("dimension")) is not int or obs["dimension"] < 1:
        raise ValueError("Capabilities observation_contract must declare positive continuous_box dimension")
    task = doc["task_contract"]
    if not isinstance(task, dict) or task.get("authored_addition") is not True or type(task.get("truncation_steps")) is not int:
        raise ValueError("Capabilities task_contract must explicitly mark authored_addition and truncation_steps")
    evidence = doc["evidence_separation"]
    if not isinstance(evidence, dict) or evidence.get("physical_validation") != "unverified":
        raise ValueError("Capabilities evidence_separation must preserve physical_validation='unverified'")
    unsupported = doc["unsupported_capabilities"]
    if not isinstance(unsupported, list) or not unsupported or any(not isinstance(item, str) or not item.strip() for item in unsupported):
        raise ValueError("Capabilities must disclose non-empty unsupported_capabilities")
    json.dumps(doc, allow_nan=False)
    return deepcopy(doc)


def describe_capabilities(
    scene,
    *,
    lab_profile="standalone",
    learned_policy_available=False,
    trained_target_id=None,
    source_video_present=False,
    reference_available=False,
    exercise_task=None,
):
    """Build a validated environment-capabilities-v1 summary for any supported Weft scene."""
    if not isinstance(scene, dict) or not isinstance(scene.get("profile"), str):
        raise ValueError("Scene must be an object with a profile string")
    scene_profile = scene["profile"]
    if scene_profile == "shared-world-v6":
        from weft.camera_shared_capabilities import describe_camera_capabilities
        return describe_camera_capabilities(scene, exercise_task, lab_profile=lab_profile,
            learned_policy_available=learned_policy_available, trained_target_id=trained_target_id,
            source_video_present=source_video_present, reference_available=reference_available)
    if exercise_task is not None and scene_profile not in ("shared-world-v1", "shared-world-v2", "shared-world-v3", "shared-world-v4", "shared-world-v5"):
        raise ValueError("Human-only exercise tasks require a shared-world scene")
    objects = scene.get("objects", [])

    if scene_profile == "rolling":
        doc = {
            "schema_version": 1,
            "profile": CAPABILITY_PROFILE,
            "scene_profile": scene_profile,
            "lab_profile": lab_profile,
            "representation_family": "rigid_1d_support",
            "physics_backend": "analytic_rolling_1d",
            "coordinate_frame": {
                "dimensions": 2,
                "axes": "x_horizontal_along_support_y_image_down",
                "units": {"length": "scene_width", "time": "second"},
            },
            "entities": {
                "total_count": 1,
                "dynamic_count": 1,
                "static_count": 1,
                "articulated_body_present": False,
                "rigid_objects_present": True,
                "mixed_rigid_articulated": False,
                "controllable_actors": [{"id": "ball-0", "label": "Ball", "kind": "rigid_body"}],
                "active_control_target": "ball-0",
                "multi_target_selection_endpoint": None,
            },
            "action_contract": {
                "space_type": "discrete",
                "dimension": 1,
                "discrete_actions": [
                    {"index": 0, "name": "coast"},
                    {"index": 1, "name": "left"},
                    {"index": 2, "name": "right"},
                ],
                "continuous_bounds": None,
                "actuator_names": ["ball-0"],
                "physical_units": "normalized_horizontal_acceleration",
                "hidden_assistance": False,
            },
            "observation_contract": {
                "space_type": "continuous_box",
                "dimension": 2,
                "components": ["position_x", "velocity_x"],
                "privileged_simulator_state": True,
            },
            "task_contract": {
                "task_type": "target_parking_1d",
                "authored_addition": True,
                "reward_summary": "Distance progress toward authored horizontal target minus step/speed penalties plus terminal parking bonus.",
                "success_criteria": {
                    "position_tolerance": scene.get("task", {}).get("position_tolerance", 0.03),
                    "speed_tolerance": scene.get("task", {}).get("speed_tolerance", 0.04),
                },
                "termination_conditions": ["target_parked", "left_horizontal_domain"],
                "truncation_steps": int(scene.get("task", {}).get("max_steps", 220)),
            },
            "reset_contract": {
                "deterministic_seeded_reset": True,
                "supports_jitter": True,
                "supports_explicit_perturbation": False,
                "default_lab_jitter": True,
            },
            "controllers": {
                "manual": True,
                "kinematic_pose_replay": False,
                "authored_reference_controller": False,
                "learned_policy_available": bool(learned_policy_available),
                "trained_target_id": trained_target_id or ("ball-0" if learned_policy_available else None),
            },
            "evidence_separation": {
                "physical_validation": "unverified",
                "source_video_present": bool(source_video_present),
                "muscle_display": "not_applicable",
                "kinematic_vs_dynamic_distinct": True,
            },
            "unsupported_capabilities": [
                "Vertical or 3D depth control",
                "Multi-object collisions or spin",
                "Articulated human bodies or coupled person-object interaction",
                "Calibrated physical meters or Newtons",
            ],
        }
        return validate_capabilities(doc)

    if scene_profile in ("planar-rigid-v1", "planar-rigid-v2"):
        from weft.scene import ACTIONS as PLANAR_ACTIONS, SCENE_UNITS

        dynamic_objs = [obj for obj in objects if obj.get("role") != "static"]
        static_objs = [obj for obj in objects if obj.get("role") == "static"]
        task = scene["task"]
        doc = {
            "schema_version": 1,
            "profile": CAPABILITY_PROFILE,
            "scene_profile": scene_profile,
            "lab_profile": lab_profile,
            "representation_family": "rigid_planar_2d",
            "physics_backend": "pymunk_2d",
            "coordinate_frame": {
                "dimensions": 2,
                "axes": "x_right_y_down",
                "units": deepcopy(scene.get("units", SCENE_UNITS)),
            },
            "entities": {
                "total_count": len(objects),
                "dynamic_count": len(dynamic_objs),
                "static_count": len(static_objs),
                "articulated_body_present": False,
                "rigid_objects_present": True,
                "mixed_rigid_articulated": False,
                "controllable_actors": [
                    {"id": obj["id"], "label": obj.get("label", obj["id"]), "kind": "rigid_body"}
                    for obj in dynamic_objs
                ],
                "active_control_target": scene["controlled_id"],
                "multi_target_selection_endpoint": "/api/actor" if len(dynamic_objs) > 1 else None,
            },
            "action_contract": {
                "space_type": "discrete",
                "dimension": 1,
                "discrete_actions": [{"index": idx, "name": name} for idx, name in enumerate(PLANAR_ACTIONS)],
                "continuous_bounds": None,
                "actuator_names": [scene["controlled_id"]],
                "physical_units": "virtual_2d_com_force",
                "hidden_assistance": False,
            },
            "observation_contract": {
                "space_type": "continuous_box",
                "dimension": 4,
                "components": ["target_offset_x", "target_offset_y", "controlled_velocity_x", "controlled_velocity_y"],
                "privileged_simulator_state": True,
            },
            "task_contract": {
                "task_type": "target_parking_2d",
                "authored_addition": True,
                "reward_summary": "20 * distance_reduction - 0.025 - 0.015 * speed (+10 on parking success, -10 on boundary escape).",
                "success_criteria": {
                    "position_tolerance": task["position_tolerance"],
                    "speed_tolerance": task["speed_tolerance"],
                },
                "termination_conditions": ["target_parked", "escaped_boundary_walls"],
                "truncation_steps": int(task["max_steps"]),
            },
            "reset_contract": {
                "deterministic_seeded_reset": True,
                "supports_jitter": True,
                "supports_explicit_perturbation": False,
                "default_lab_jitter": False,
            },
            "controllers": {
                "manual": True,
                "kinematic_pose_replay": False,
                "authored_reference_controller": False,
                "learned_policy_available": bool(learned_policy_available),
                "trained_target_id": trained_target_id if learned_policy_available else None,
            },
            "evidence_separation": {
                "physical_validation": "unverified",
                "source_video_present": bool(source_video_present),
                "muscle_display": "not_applicable",
                "kinematic_vs_dynamic_distinct": True,
            },
            "unsupported_capabilities": [
                "3D depth, elevation, or out-of-plane rotation",
                "Articulated human bodies or coupled person-object contacts in one scene",
                "Calibrated metric lengths, masses, or real-world force transfer",
            ],
        }
        return validate_capabilities(doc)

    if scene_profile == "spatial-rigid-v1":
        from weft.spatial_rigid import ACTIONS as SPATIAL_ACTIONS, UNITS as SPATIAL_UNITS

        dynamic_objs = [obj for obj in objects if obj.get("role") != "static"]
        static_objs = [obj for obj in objects if obj.get("role") == "static"]
        task = scene["task"]
        doc = {
            "schema_version": 1,
            "profile": CAPABILITY_PROFILE,
            "scene_profile": scene_profile,
            "lab_profile": lab_profile,
            "representation_family": "rigid_spatial_3d",
            "physics_backend": "mujoco_3d_rigid",
            "coordinate_frame": {
                "dimensions": 3,
                "axes": "x_right_y_depth_z_up",
                "units": deepcopy(scene.get("units", SPATIAL_UNITS)),
            },
            "entities": {
                "total_count": len(objects),
                "dynamic_count": len(dynamic_objs),
                "static_count": len(static_objs),
                "articulated_body_present": False,
                "rigid_objects_present": True,
                "mixed_rigid_articulated": False,
                "controllable_actors": [
                    {"id": obj["id"], "label": obj.get("label", obj["id"]), "kind": "rigid_body"}
                    for obj in dynamic_objs
                ],
                "active_control_target": scene["controlled_id"],
                "multi_target_selection_endpoint": "/api/actor" if len(dynamic_objs) > 1 else None,
            },
            "action_contract": {
                "space_type": "discrete",
                "dimension": 1,
                "discrete_actions": [{"index": idx, "name": name} for idx, name in enumerate(SPATIAL_ACTIONS)],
                "continuous_bounds": None,
                "actuator_names": [scene["controlled_id"]],
                "physical_units": "virtual_3d_com_force",
                "hidden_assistance": False,
            },
            "observation_contract": {
                "space_type": "continuous_box",
                "dimension": 13 * len(objects) + 4,
                "components": [
                    "per_object_position_xyz_quaternion_wxyz_velocity_xyz_spin_xyz",
                    "target_offset_xyz",
                    "episode_step_fraction",
                ],
                "privileged_simulator_state": True,
            },
            "task_contract": {
                "task_type": "target_parking_3d",
                "authored_addition": True,
                "reward_summary": "3D Euclidean target-distance reduction - 0.001 + 1.0 on terminal position-and-speed parking success.",
                "success_criteria": {
                    "position_tolerance": task["position_tolerance"],
                    "speed_tolerance": task["speed_tolerance"],
                },
                "termination_conditions": ["target_parked_3d", "left_expanded_model_bounds", "exceeded_finite_dynamics_envelope"],
                "truncation_steps": int(task["max_steps"]),
            },
            "reset_contract": {
                "deterministic_seeded_reset": True,
                "supports_jitter": True,
                "supports_explicit_perturbation": True,
                "default_lab_jitter": False,
            },
            "controllers": {
                "manual": True,
                "kinematic_pose_replay": False,
                "authored_reference_controller": False,
                "learned_policy_available": bool(learned_policy_available),
                "trained_target_id": trained_target_id if learned_policy_available else None,
            },
            "evidence_separation": {
                "physical_validation": "unverified",
                "source_video_present": bool(source_video_present),
                "muscle_display": "not_applicable",
                "kinematic_vs_dynamic_distinct": True,
            },
            "unsupported_capabilities": [
                "Direct rotational torque actuators (spin arises only through contact geometry)",
                "Non-convex meshes or deformable bodies",
                "Articulated human bodies sharing the same MuJoCo world (requires proposed shared-world-v1)",
                "Independently calibrated metric scale, mass, or real-world physical transfer",
            ],
        }
        return validate_capabilities(doc)

    if scene_profile in ("exercise-v1", "exercise-v2", "exercise-v3"):
        joints = list(scene.get("joint_names") or scene.get("human", {}).get("joint_names", []))
        selected_joint = scene.get("task", {}).get("selected_joint") or (joints[0] if joints else "left_knee")
        is_mobile = scene_profile == "exercise-v3"
        action_names = ["coast", "positive", "negative", "brake", "hold", "reference_servo"]
        if is_mobile:
            action_names.extend(["release_support", "push_left", "push_right", "restore_support"])
        task = scene["task"]
        doc = {
            "schema_version": 1,
            "profile": CAPABILITY_PROFILE,
            "scene_profile": scene_profile,
            "lab_profile": lab_profile,
            "representation_family": "articulated_human_3d",
            "physics_backend": "mujoco_3d_articulated",
            "coordinate_frame": {
                "dimensions": 3,
                "axes": "x_lateral_y_depth_z_up",
                "units": {"length": "body_model_units", "angle": "radian", "time": "second", "torque": "Nm"},
            },
            "entities": {
                "total_count": len(objects),
                "dynamic_count": len(objects),
                "static_count": 1 if is_mobile else 0,
                "articulated_body_present": True,
                "rigid_objects_present": False,
                "mixed_rigid_articulated": False,
                "controllable_actors": [
                    {"id": name, "label": name.replace("_", " "), "kind": "joint_axis"}
                    for name in joints
                ],
                "active_control_target": selected_joint,
                "multi_target_selection_endpoint": "/api/joint" if len(joints) > 1 else None,
            },
            "action_contract": {
                "space_type": "discrete",
                "dimension": 1,
                "discrete_actions": [{"index": idx, "name": name} for idx, name in enumerate(action_names)],
                "continuous_bounds": None,
                "actuator_names": [selected_joint] + (["virtual_root_support_xyz"] if is_mobile else []),
                "physical_units": "joint_torque_command_with_virtual_support" if is_mobile else "joint_torque_command",
                "hidden_assistance": is_mobile,
            },
            "observation_contract": {
                "space_type": "continuous_box",
                "dimension": 2,
                "components": ["selected_joint_target_angle_error_rad", "selected_joint_angular_velocity_rad_s"],
                "privileged_simulator_state": True,
            },
            "task_contract": {
                "task_type": "single_joint_target_angle",
                "authored_addition": True,
                "reward_summary": "Selected-joint angular error reduction minus step/speed penalties plus terminal target-hold bonus.",
                "success_criteria": {
                    "angle_tolerance_radians": task["angle_tolerance_radians"],
                    "angular_speed_tolerance_radians_per_second": task["angular_speed_tolerance_radians_per_second"],
                },
                "termination_conditions": ["selected_joint_at_target_angle_and_speed"],
                "truncation_steps": int(task["max_steps"]),
            },
            "reset_contract": {
                "deterministic_seeded_reset": True,
                "supports_jitter": True,
                "supports_explicit_perturbation": False,
                "default_lab_jitter": False,
            },
            "controllers": {
                "manual": True,
                "kinematic_pose_replay": bool(reference_available),
                "authored_reference_controller": True,
                "learned_policy_available": bool(learned_policy_available),
                "trained_target_id": trained_target_id if learned_policy_available else None,
            },
            "evidence_separation": {
                "physical_validation": "unverified",
                "source_video_present": bool(source_video_present),
                "muscle_display": "illustrative_motor_command_not_measured_activation",
                "kinematic_vs_dynamic_distinct": True,
            },
            "unsupported_capabilities": [
                "Unassisted full-body human balance or biomechanical ground-reaction validation",
                "Measured physiological muscle activation or tendon force",
                "Anatomical mesh, hand/finger, or wrist/ankle mechanics",
                "Coupled external rigid objects in the same MuJoCo scene (requires proposed shared-world-v1)",
            ],
        }
        return validate_capabilities(doc)

    if scene_profile in ("exercise-imitation-v1", "exercise-feet-posture-v1"):
        contract = scene["action_contract"]
        motor_names = list(contract["names"])
        is_feet = scene_profile == "exercise-feet-posture-v1"
        task = scene["task"]
        doc = {
            "schema_version": 1,
            "profile": CAPABILITY_PROFILE,
            "scene_profile": scene_profile,
            "lab_profile": lab_profile,
            "representation_family": "articulated_human_3d",
            "physics_backend": "mujoco_3d_articulated",
            "coordinate_frame": {
                "dimensions": 3,
                "axes": "x_lateral_y_depth_z_up",
                "units": {"length": "body_model_units", "angle": "radian", "time": "second", "actuators": "Nm_and_virtual_root_N"},
            },
            "entities": {
                "total_count": len(objects),
                "dynamic_count": len(objects),
                "static_count": 1,
                "articulated_body_present": True,
                "rigid_objects_present": False,
                "mixed_rigid_articulated": False,
                "controllable_actors": [
                    {"id": name, "label": name.replace("_", " "), "kind": "direct_motor"}
                    for name in motor_names
                ],
                "active_control_target": "all_direct_motors",
                "multi_target_selection_endpoint": "client_motor_vector",
            },
            "action_contract": {
                "space_type": "continuous_box",
                "dimension": len(motor_names),
                "discrete_actions": None,
                "continuous_bounds": list(contract["normalized_bounds"]),
                "actuator_names": motor_names,
                "physical_units": list(contract["units"]),
                "hidden_assistance": bool(contract.get("hidden_assistance", False)),
            },
            "observation_contract": {
                "space_type": "continuous_box",
                "dimension": 85 if is_feet else 155,
                "components": (
                    ["qpos_28", "qvel_28", "starting_target_qpos_28", "episode_step_fraction"]
                    if is_feet
                    else [
                        "position_22",
                        "velocity_22",
                        "target_position_22",
                        "target_velocity_22",
                        "policy_features_66",
                        "source_phase_1",
                    ]
                ),
                "privileged_simulator_state": True,
            },
            "task_contract": {
                "task_type": "starting_pose_hold" if is_feet else "whole_body_reference_tracking",
                "authored_addition": True,
                "reward_summary": "Exponential joint/root tracking kernel minus motor effort and floor-penetration penalty; strict terminal RMSE and floor-penetration gates.",
                "success_criteria": deepcopy(task["success_limits"]),
                "termination_conditions": ["nonfinite_or_out_of_envelope_dynamics", "floor_penetration_above_0.08_model_units"],
                "truncation_steps": int(task["horizon_steps"]),
            },
            "reset_contract": {
                "deterministic_seeded_reset": True,
                "supports_jitter": not is_feet,
                "supports_explicit_perturbation": not is_feet,
                "default_lab_jitter": not is_feet,
            },
            "controllers": {
                "manual": True,
                "kinematic_pose_replay": bool(reference_available),
                "authored_reference_controller": True,
                "learned_policy_available": bool(learned_policy_available) and not is_feet,
                "trained_target_id": "all_direct_motors" if (learned_policy_available and not is_feet) else None,
            },
            "evidence_separation": {
                "physical_validation": "unverified",
                "source_video_present": bool(source_video_present),
                "muscle_display": "illustrative_motor_command_not_measured_activation",
                "kinematic_vs_dynamic_distinct": True,
            },
            "unsupported_capabilities": [
                "Unassisted balance without the 3 explicit virtual root-force actuators",
                "Measured muscle activation, anatomical foot-ground compliance, or recovered human intent",
                "Coupled external rigid objects in this single-body profile (use shared-world-v1 for coupled worlds)",
            ],
        }
        return validate_capabilities(doc)

    if scene_profile in ("shared-world-v1", "shared-world-v2", "shared-world-v3", "shared-world-v4", "shared-world-v5"):
        from weft.shared_world import HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE, OBJECT_ACTIONS, validate_shared_world

        feet_task = None
        if scene_profile == "shared-world-v5":
            from weft.shared_exercise_feet_task import prepare_shared_exercise_feet_task

            if exercise_task is None or learned_policy_available or trained_target_id is not None:
                raise ValueError("Feet capabilities require an explicit feet task without learned policies or trained targets")
            validated, _, feet_task = prepare_shared_exercise_feet_task(exercise_task, scene)
        else:
            validated = validate_shared_world(scene)
        base_profile = validated["base_profile"] if feet_task else scene_profile
        human_only = base_profile == HUMAN_ONLY_PROFILE
        static_support = base_profile == STATIC_SUPPORT_PROFILE
        no_object_goal = human_only or static_support
        dyn_objs = [obj for obj in validated["objects"] if obj["role"] != "static"]
        static_objs = [obj for obj in validated["objects"] if obj["role"] == "static"]
        contract = validated["action_contract"]
        task = validated["task"]
        goal = task["object_goal"]
        actors = [{"id": "human", "label": "Articulated human (22 motors)", "kind": "articulated_human"}] + [
            {"id": obj["id"], "label": obj.get("label", obj["id"]), "kind": "rigid_body"} for obj in dyn_objs
        ]
        doc = {
            "schema_version": 1,
            "profile": CAPABILITY_PROFILE,
            "scene_profile": scene_profile,
            "lab_profile": lab_profile,
            "representation_family": "mixed_rigid_articulated_3d",
            "physics_backend": "mujoco_3d_shared_world",
            "coordinate_frame": {
                "dimensions": 3,
                "axes": validated["coordinate_frame"]["axes"],
                "units": deepcopy(validated["coordinate_frame"]["units"]),
            },
            "entities": {
                "total_count": 1 + len(validated["objects"]),
                "dynamic_count": 1 + len(dyn_objs),
                "static_count": len(static_objs),
                "articulated_body_present": True,
                "rigid_objects_present": True,
                "mixed_rigid_articulated": True,
                "controllable_actors": actors,
                "active_control_target": validated["controlled_id"],
                "multi_target_selection_endpoint": "/api/actor",
            },
            "action_contract": {
                "space_type": "continuous_box",
                "dimension": len(contract["names"]),
                "discrete_actions": [{"index": idx, "name": name} for idx, name in enumerate(OBJECT_ACTIONS)],
                "continuous_bounds": [-1.0, 1.0],
                "actuator_names": list(contract["names"]),
                "physical_units": list(contract["units"]),
                "hidden_assistance": False,
            },
            "observation_contract": {
                "space_type": "continuous_box",
                "dimension": 155 + 13 * len(validated["objects"]) + 3,
                "components": [
                    "human_qpos_22_qvel_22_target_qpos_22_target_qvel_22_policy_features_66_phase_1",
                    "per_object_position_xyz_quaternion_wxyz_velocity_xyz_spin_xyz",
                    "target_object_goal_offset_xyz",
                ],
                "privileged_simulator_state": True,
            },
            "task_contract": {
                "task_type": "coupled_human_object_v1",
                "authored_addition": True,
                "reward_summary": (
                    "Nonnegative coupled human reference-tracking exponential + 3D target-object proximity/settling "
                    "score (gated by human-object contact when required) minus control effort, plus +0.5 coupled bonus."
                ),
                "success_criteria": {
                    "human_limits": deepcopy(task["human_limits"]),
                    "object_goal": deepcopy(goal),
                },
                "termination_conditions": [
                    "nonfinite_dynamics",
                    "human_position_or_velocity_envelope_exceeded",
                    "rigid_velocity_envelope_exceeded",
                    "object_left_bounds",
                ],
                "truncation_steps": int(task["horizon_steps"]),
            },
            "reset_contract": {
                "deterministic_seeded_reset": True,
                "supports_jitter": True,
                "supports_explicit_perturbation": False,
                "default_lab_jitter": False,
            },
            "controllers": {
                "manual": True,
                "kinematic_pose_replay": bool(reference_available or validated["human_spec"].get("reference")),
                "authored_reference_controller": True,
                "learned_policy_available": bool(learned_policy_available),
                "trained_target_id": (
                    trained_target_id or (("human" if no_object_goal else "coupled_human_and_" + goal["target_object_id"])
                                          if learned_policy_available else None)
                ),
            },
            "evidence_separation": {
                "physical_validation": "unverified",
                "source_video_present": bool(source_video_present),
                "muscle_display": "illustrative_motor_command_not_measured_activation",
                "kinematic_vs_dynamic_distinct": True,
            },
            "unsupported_capabilities": [
                "Unassisted human balance without the 3 explicit virtual root-force actuators",
                "Dexterous multi-finger grasping, soft-tissue deformation, or measured EMG/contact forces",
                "Automatic end-to-end single-video coupled person+object 3D perception without explicit scale bridge",
            ],
        }
        if human_only:
            doc["representation_family"] = "articulated_human_3d"
            doc["entities"].update(rigid_objects_present=False, mixed_rigid_articulated=False,
                                   multi_target_selection_endpoint=None, support="implicit_ground_plane")
            doc["action_contract"]["discrete_actions"] = []
            doc["observation_contract"].update(dimension=155, components=[
                "human_qpos_22_qvel_22_target_qpos_22_target_qvel_22_policy_features_66_phase_1"])
            doc["task_contract"].update(task_type="human_motion_v1",
                reward_summary="Human reference tracking minus explicit control effort; no object-target score or equipment bonus.",
                success_criteria={"human_limits": deepcopy(task["human_limits"]), "object_goal": None},
                termination_conditions=["nonfinite_dynamics", "human_position_or_velocity_envelope_exceeded"])
            doc["evidence_separation"].update(inventory_basis=validated["inventory"]["basis"], object_absence_verified=False)
            doc["unsupported_capabilities"] = [
                "Equipment, obstacles and object-target tasks; none are declared in this person-only scene",
                "Unassisted human balance without the 3 explicit virtual root-force actuators",
                "Measured muscle activation, verified object absence or physical transfer",
            ]
        if static_support:
            if learned_policy_available and exercise_task is None:
                raise ValueError("A static-support learned policy requires its attached support task")
            doc["entities"].update(multi_target_selection_endpoint=None, support="declared_static_collision_geometry")
            doc["action_contract"]["discrete_actions"] = []
            doc["observation_contract"].update(dimension=155 + 13 * len(static_objs), components=[
                "human_qpos_22_qvel_22_target_qpos_22_target_qvel_22_policy_features_66_phase_1",
                "per_static_object_position_xyz_quaternion_wxyz_velocity_xyz_spin_xyz"])
            doc["task_contract"].update(task_type="human_support_v1",
                reward_summary="Engine reference tracking; declared support-load criteria require an attached support task.",
                success_criteria={"human_limits": deepcopy(task["human_limits"]), "object_goal": None},
                termination_conditions=["nonfinite_dynamics", "human_position_or_velocity_envelope_exceeded"])
            doc["unsupported_capabilities"] = [
                "Movable equipment, object-force controls and learning without a pinned support task/policy",
                "Avatar recording for this static-support profile",
                "Ground-force load requirements, finger grasping or measured muscle activation",
                "Recovered support geometry, verified physical contact or physical transfer",
            ]
        if exercise_task is not None:
            from weft.shared_exercise_task import GATES, HUMAN_ONLY_GATE_APPLICABILITY, validate_shared_exercise_task

            exercise = feet_task["base_task"] if feet_task else validate_shared_exercise_task(exercise_task, validated)
            assisted = exercise["control"]["root_assistance"] == "explicit_3_axis"
            count = (25 if feet_task else 19) + (3 if assisted else 0)
            if feet_task:
                from weft.shared_exercise_feet_projection import task_motor_indices

                indices = task_motor_indices(root_assistance=assisted)
            else:
                indices = tuple(range(count))
            doc["entities"].update(controllable_actors=[{
                "id": "human", "label": f"Human ({count} explicit motors)", "kind": "articulated_human"}],
                active_control_target="human", multi_target_selection_endpoint=None)
            doc["action_contract"].update(dimension=count, discrete_actions=[],
                actuator_names=[contract["names"][index] for index in indices],
                physical_units=[contract["units"][index] for index in indices],
                joint_channels=list(range(25 if feet_task else 19)),
                root_assistance_channels=list(range(25 if feet_task else 19, count)), object_force_channels=[])
            equipment_ids = [] if static_support else exercise["equipment_ids"]
            requirements = exercise["support_requirements"] if static_support else exercise["contact_requirements"]
            doc["observation_contract"].update(dimension=(198 if feet_task else 156) + 17 * len(equipment_ids)
                                               + 6 * len(requirements),
                components=["human_reference_conditioned_state", "passive_equipment_reset_relative_state",
                            "declared_contact_requirement_coverage_and_gaps", "elapsed_fraction"])
            doc["task_contract"] = {
                "task_type": exercise["profile"], "authored_addition": True,
                "reward_summary": "Human reference tracking and passive equipment stability minus explicit human effort; no legacy object-target score.",
                "success_criteria": {"frozen_gates": dict(GATES), "equipment_ids": equipment_ids,
                    "contact_requirements": requirements, "complete_substep_interval_required": True},
                "termination_conditions": ["explicit_engine_or_observation_failure", "completed_successful_interval"],
                "truncation_steps": exercise["interval"]["horizon_steps"],
                "source_scope": exercise["source_scope"], "root_assistance": exercise["control"]["root_assistance"],
            }
            doc["reset_contract"].update(supports_jitter=False, default_lab_jitter=False)
            doc["controllers"].update(learned_policy_available=bool(learned_policy_available),
                                      trained_target_id="human" if learned_policy_available else None, coast=True)
            doc["unsupported_capabilities"] = [
                "Object-force actions, object-target selection and legacy coupled learned policies",
                "Changing task gates, contact requirements or root assistance inside this lab",
                "Finger grasping, measured muscle activation, verified contact or physical transfer",
            ]
            if human_only:
                doc["observation_contract"]["components"] = ["human_reference_conditioned_state", "elapsed_fraction"]
                doc["task_contract"]["reward_summary"] = "Human reference tracking minus explicit human effort; equipment and contact checks do not apply and add no reward."
                doc["task_contract"]["success_criteria"].update(
                    frozen_gates={name: GATES[name] for name in ("angular_rmse_rad", "root_rmse_model_units",
                        "peak_angular_rmse_rad", "peak_root_rmse_model_units", "max_floor_penetration_model_units")},
                    gate_applicability=dict(HUMAN_ONLY_GATE_APPLICABILITY))
            if static_support:
                from weft.shared_support_task import GATES as SUPPORT_GATES

                doc["observation_contract"]["components"] = ["human_reference_conditioned_state",
                    "declared_support_load_coverage_and_gaps", "elapsed_fraction"]
                doc["task_contract"].update(
                    reward_summary="Human tracking multiplied by declared simulated support load, minus explicit human effort; no equipment-stability bonus.",
                    success_criteria={"frozen_gates": dict(SUPPORT_GATES), "support_requirements": requirements,
                        "complete_substep_interval_required": True, "equipment_stability": "not_applicable"})
                doc["unsupported_capabilities"] = [
                    "Movable equipment, object-force controls and policies not pinned to this support task",
                    "Avatar recording for this static-support profile",
                    "Ground-force load requirements, finger grasping or measured muscle activation",
                    "Recovered support geometry, verified physical contact or physical transfer",
                ]
            if feet_task:
                doc["action_contract"].update(body_joint_channels=list(range(19)), ankle_channels=list(range(19, 25)),
                    engine_motor_indices=list(indices), root_assistance=exercise["control"]["root_assistance"],
                    ankle_target_mode=feet_task["control"]["ankle_target_mode"])
                doc["task_contract"].update(task_type=feet_task["profile"], base_task_type=exercise["profile"],
                    base_scene_profile=base_profile,
                    identity={**{name: feet_task[name] for name in ("scene_sha256", "base_scene_sha256", "base_task_sha256",
                        "projection_profile", "projection_contract_sha256", "adapter_implementation_sha256", "foot_reference_sha256")},
                        "task_sha256": canonical_hash(feet_task)},
                    diagnostics={"source": "simulated_engine_substeps", "ankle_target_mode": feet_task["control"]["ankle_target_mode"],
                        "scored": False, "foot_support_credit": False, "ground_load_requirements": "unsupported",
                        "retention": "latest_observed_substep_plus_episode_aggregates"})
                doc["task_contract"]["reward_summary"] += " Original 19-joint scoring is unchanged; ankle effort and foot contacts add no score or support credit."
                doc["controllers"].update(learned_policy_available=False, trained_target_id=None)
                doc["unsupported_capabilities"] += [
                    "Learned policies and avatar skins for this feet task",
                    "Video-derived ankle tracking; ankles use authored original-start angle holds",
                    "Foot support credit, ground-load requirements or a complete saved substep contact trace",
                ]
        return validate_capabilities(doc)

    raise ValueError(f"Unsupported scene profile for capability inspection: {scene_profile!r}")


def describe_run_capabilities(directory, *, out=None):
    """Inspect a completed lab directory or standalone sealed run directory without modifying it."""
    target = Path(directory).resolve()
    if out is not None:
        out = Path(out).resolve()
        if out.exists() or out == target or out.is_relative_to(target):
            raise ValueError("Choose a new output file outside the inspected directory")

    if (target / "lab-config.json").is_file():
        from weft.lab import Lab

        lab = Lab(target)
        try:
            summary = lab.summary()
            if summary.get("status") != "ready" or "capabilities" not in summary:
                raise ValueError("Lab directory is not ready for capability inspection")
            doc = summary["capabilities"]
        finally:
            lab.close()
    else:
        exercise_task, exercise_policy, task_manifest_sha = None, None, None
        scene_raw = (target / "scene.json").read_bytes() if (target / "scene.json").is_file() else None
        scene = json.loads(scene_raw) if scene_raw is not None else None
        if (target / "exercise-lab.json").is_file():
            from weft.shared_exercise_lab import load_shared_exercise_lab
            loaded = load_shared_exercise_lab(target)
            exercise_task, exercise_policy = loaded["task"], loaded.get("policy")
        else:
            if scene is not None and scene.get("profile") == "shared-world-v5":
                from weft.shared_world_packaging import read_package_file, read_package_manifest
                from weft.world_evidence import _nonfinite, _unique_object

                # Bound task bytes before the existing verifier hashes every package member.
                raw_manifest, members = read_package_manifest(target)
                task_manifest_sha = sha256(raw_manifest).hexdigest()
                if (members.get("scene.json") != sha256(scene_raw).hexdigest()
                        or "exercise-task.json" not in members):
                    raise ValueError("Feet capability inspection requires the same sealed scene and explicit task manifest")
                raw_task = read_package_file(target, "exercise-task.json", max_bytes=128 * 1024,
                                             expected_sha256=members["exercise-task.json"])
                try:
                    exercise_task = json.loads(raw_task, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
                except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
                    raise ValueError("Feet capability inspection requires bounded strict task JSON") from error
            verified_manifest = verify_artifacts(target)
            if task_manifest_sha is not None and task_manifest_sha != verified_manifest["manifest_sha256"]:
                raise ValueError("Feet task manifest changed during capability verification")
        has_video = any((target / f"source.{ext}").is_file() for ext in ("mp4", "webm", "mov", "m4v", "avi", "mkv"))
        has_reference = (target / "human-reference.json").is_file()
        has_policy = (target / "policy.json").is_file()
        trained_target = None
        if has_policy:
            policy = json.loads((target / "policy.json").read_text())
            if isinstance(policy, dict):
                if policy.get("algorithm") == "shared_world_cem_v1":
                    trained_target = "coupled_human_and_" + str(policy.get("target_object_id", "object"))
                else:
                    trained_target = policy.get("selected_joint") or policy.get("controlled_id")
        if scene_raw is not None:
            if scene.get("profile") in ("planar-rigid-v1", "planar-rigid-v2"):
                planar_training_enabled(target)
            if trained_target is None and has_policy:
                trained_target = scene.get("controlled_id")
            doc = describe_capabilities(
                scene,
                lab_profile="shared_exercise" if exercise_task is not None else "standalone",
                learned_policy_available=(exercise_policy is not None if exercise_task is not None else
                                          has_policy and scene.get("learned_policy_available") is not False),
                trained_target_id="human" if exercise_policy is not None else trained_target,
                source_video_present=has_video,
                reference_available=has_reference,
                exercise_task=exercise_task,
            )
        elif (target / "human.json").is_file():
            spec = json.loads((target / "human.json").read_text())
            if (target / "foot-reference.json").is_file():
                from weft.foot_sandbox import FootEnv

                foot_ref = json.loads((target / "foot-reference.json").read_text())
                env = FootEnv(spec, foot_ref)
                try:
                    scene = env.scene
                finally:
                    env.close()
            elif has_policy and json.loads((target / "policy.json").read_text()).get("algorithm") == "whole_body_cem_v1":
                from weft.whole_body import WholeBodyEnv

                settings = json.loads((target / "settings.json").read_text())
                part = settings["partition"]
                env = WholeBodyEnv(spec, start_time_s=part["evaluation_start_times_s"][0], horizon_steps=part["horizon_steps"])
                try:
                    scene = env.scene
                finally:
                    env.close()
                trained_target = "all_direct_motors"
            else:
                from weft.human import HumanEnv

                env = HumanEnv(spec)
                try:
                    if trained_target in env.joint_names:
                        env.select_joint(trained_target)
                    env.scene["joint_names"] = list(env.joint_names)
                    scene = env.scene
                finally:
                    env.close()
            doc = describe_capabilities(
                scene,
                lab_profile="standalone",
                learned_policy_available=has_policy and scene.get("learned_policy_available") is not False,
                trained_target_id=trained_target,
                source_video_present=has_video,
                reference_available=has_reference or bool(spec.get("reference")),
            )
        elif (target / "world.json").is_file():
            from weft.env import RollingEnv
            from weft.world import World

            world = World.from_dict(json.loads((target / "world.json").read_text()))
            env = RollingEnv(world)
            has_npz = (target / "policy.npz").is_file()
            scene = {
                "schema_version": 1,
                "profile": "rolling",
                "task": env.task.to_dict(),
                "objects": [{"id": "ball-0", "label": "Ball", "role": "active", "shape": "circle", "radius": world.radius}],
            }
            doc = describe_capabilities(
                scene,
                lab_profile="standalone",
                learned_policy_available=has_npz,
                trained_target_id="ball-0" if has_npz else None,
                source_video_present=has_video,
                reference_available=False,
            )
        else:
            raise ValueError("Directory must contain lab-config.json, scene.json, human.json, or world.json")

    validated = validate_capabilities(doc)
    if out is not None:
        write_json(out, validated)
    return validated
