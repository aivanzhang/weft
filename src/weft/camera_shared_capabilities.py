"""Expose camera controls while keeping external policies separate from source qualification."""

from copy import deepcopy

from weft.camera_shared_task import prepare_camera_shared_task
from weft.io import canonical_hash
from weft.shared_exercise_task import GATES, HUMAN_ONLY_GATE_APPLICABILITY


IDENTITIES = ("scene_sha256", "camera_coupling_sha256", "source_sha256", "body_sha256", "source_clock_sha256",
              "metric_context_sha256", "kernel_evaluator_sha256", "adapter_implementation_sha256", "evaluator_sha256")


def describe_camera_capabilities(scene, task, *, lab_profile, learned_policy_available, trained_target_id,
                                 source_video_present, reference_available, camera_policy_admission=None):
    from weft.capabilities import CAPABILITY_PROFILE, validate_capabilities

    if task is None or type(learned_policy_available) is not bool:
        raise ValueError("Camera capabilities require an explicit task and controller availability")
    if camera_policy_admission is None:
        if learned_policy_available or trained_target_id is not None:
            raise ValueError("Camera learned controls require a distinct exact policy admission")
    elif learned_policy_available is not True or trained_target_id != "human":
        raise ValueError("An admitted camera policy must target the explicit human controller")
    actual, task = prepare_camera_shared_task(task, scene)
    policy_view = None
    if camera_policy_admission is not None:
        from weft.camera_shared_policy_receipt import admission_view, validate_camera_policy_admission

        admitted = validate_camera_policy_admission(camera_policy_admission, actual, task)
        policy_view = admission_view(admitted)
    assisted = task["control"]["root_assistance"] == "explicit_3_axis"
    count = 22 if assisted else 19
    names, units = actual["action_contract"]["names"], actual["action_contract"]["units"]
    return validate_capabilities({
        "schema_version": 1, "profile": CAPABILITY_PROFILE, "scene_profile": actual["profile"],
        "lab_profile": lab_profile, "representation_family": "articulated_human_3d",
        "physics_backend": "mujoco_3d_shared_world", "coordinate_frame": {
            "dimensions": 3, "axes": actual["coordinate_frame"]["axes"],
            "units": deepcopy(actual["coordinate_frame"]["units"])},
        "entities": {"total_count": 1, "dynamic_count": 1, "static_count": 0,
            "articulated_body_present": True, "rigid_objects_present": False, "mixed_rigid_articulated": False,
            "controllable_actors": [{"id": "human", "label": f"Human ({count} explicit motors)", "kind": "articulated_human"}],
            "active_control_target": "human", "multi_target_selection_endpoint": None,
            "support": "supplied_camera_ground_prior_not_verified_contact"},
        "action_contract": {"space_type": "continuous_box", "dimension": count, "discrete_actions": [],
            "continuous_bounds": [-1., 1.], "actuator_names": list(names[:count]), "physical_units": list(units[:count]),
            "hidden_assistance": False, "joint_channels": list(range(19)),
            "root_assistance_channels": list(range(19, count)), "object_force_channels": []},
        "observation_contract": {"space_type": "continuous_box", "dimension": 156,
            "components": ["human_reference_conditioned_state", "elapsed_fraction"], "privileged_simulator_state": True},
        "task_contract": {"task_type": task["profile"], "authored_addition": True,
            "reward_summary": "Human reference tracking minus explicit effort; equipment and contact add no reward.",
            "success_criteria": {"frozen_gates": {name: GATES[name] for name in (
                "angular_rmse_rad", "root_rmse_model_units", "peak_angular_rmse_rad", "peak_root_rmse_model_units",
                "max_floor_penetration_model_units")}, "gate_applicability": dict(HUMAN_ONLY_GATE_APPLICABILITY),
                "equipment_ids": [], "contact_requirements": [], "complete_substep_interval_required": True},
            "termination_conditions": ["explicit_engine_or_observation_failure", "completed_successful_interval"],
            "truncation_steps": task["interval"]["horizon_steps"], "source_scope": deepcopy(task["scope"]),
            "root_assistance": task["control"]["root_assistance"],
            "identity": {**{key: task[key] for key in IDENTITIES}, "task_sha256": canonical_hash(task)}},
        "reset_contract": {"deterministic_seeded_reset": True, "supports_jitter": False,
            "supports_explicit_perturbation": False, "default_lab_jitter": False},
        "controllers": {"manual": True, "coast": True, "kinematic_pose_replay": bool(reference_available),
            "authored_reference_controller": True, "learned_policy_available": policy_view is not None,
            "trained_target_id": "human" if policy_view is not None else None,
            "camera_policy_admission": policy_view},
        "evidence_separation": {"physical_validation": "unverified", "source_video_present": bool(source_video_present),
            "muscle_display": "illustrative_motor_command_not_measured_activation", "kinematic_vs_dynamic_distinct": True,
            "object_absence_verified": False, "source_contact_verified": False, "ground_support_verified": False,
            "source_clock_sha256": task["source_clock_sha256"], "source_learning_qualified": False,
            "external_policy_attached": policy_view is not None},
        "unsupported_capabilities": [
            "Recovered feet, ground support, equipment contact, grasp or physical accuracy",
            ("Source learning qualification, custom avatars and equipment scoring for this camera task"
             if policy_view is not None else "Learned policies without exact admission, custom avatars and equipment scoring for this camera task"),
            "Unassisted balance, measured muscle activation or recovered anatomy",
        ],
    })
