"""Saved camera identity checks; no native imports, reconstruction or replay."""


PROFILE = "camera-shared-recording-v1"
PROTOCOL = "camera-shared-full-state-replay-v1"
IDENTITIES = ("camera_coupling_sha256", "source_sha256", "body_sha256", "source_clock_sha256",
              "metric_context_sha256", "kernel_evaluator_sha256", "adapter_implementation_sha256", "evaluator_sha256")
IMPLEMENTATIONS = frozenset({"camera_shared_task.py", "camera_shared_state.py", "camera_shared_recording.py",
                             "camera_shared_receipt.py"})
BODY_RUNTIME_FILES = (
    "limb_position_fit.py", "constrained_stereo.py", "distal_pixel_fit.py", "multiview_distal_pixels.py",
    "distal_depth_branches.py", "multiview_depth_branches.py", "joint_pixel_trajectory.py", "multiview_joint_pixels.py",
)
RUNTIME_REQUIRED = frozenset({
    "shared_exercise_task.py", "shared_exercise_reset.py", "shared_world.py", "shared_world_observation.py",
    "shared_world_evidence.py", "source_equivalence.py", "world_evidence.py", "observations.py", "human.py",
    "mobile_human.py", "whole_body.py", "spatial_rigid.py", "body_frame.py", "spatial_pose.py", "root_motion.py",
    "errors.py", "io.py", "camera_shared_contract.py", "camera_body.py", "camera_ground.py", "camera_pose.py",
    "multiview_root.py", "multiview_body.py", "multiview_pose.py", "multiview_segments.py",
    "shared_exercise_recording.py", "recording.py",
    "shared_exercise_feet_state.py", "shared_exercise_feet_projection.py", "shared_support_task.py",
}) | IMPLEMENTATIONS


def runtime_required(scene):
    # Saved receipts must pin reconstruction validators without importing native code.
    coupling = scene.get("camera_coupling")
    extra = BODY_RUNTIME_FILES if isinstance(coupling, dict) and coupling.get("profile") == "camera-shared-coupling-v2" else ()
    return RUNTIME_REQUIRED | frozenset(extra)


def saved_camera_bindings(scene, task, envelope, contract, native, members):
    from weft.recording_receipt import _canonical, _hash, _require

    _require(scene.get("profile") == "shared-world-v6" and scene.get("objects") == []
             and scene.get("learned_policy_available") is False, "Camera receipt requires its unchanged human-only v6 scene")
    _require(type(envelope.get("schema_version")) is int and envelope["schema_version"] == 1
             and envelope.get("profile") == PROFILE and set(envelope) == {
                 "schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects"},
             "Camera recording fields differ; policies and avatars are unsupported")
    _require(type(task.get("schema_version")) is int and task["schema_version"] == 1
             and task.get("profile") == "camera-shared-task-v1" and set(task) == {
                 "schema_version", "profile", "scene_sha256", *IDENTITIES, "metric_context", "gate_profile",
                 "gates_sha256", "interval", "control", "scope"}, "Camera task fields differ from its distinct contract")
    _require(contract.get("learned_policy_available") is False and contract.get("policy_sha256") is None
             and not any(name in {"policy.json", "policy-evaluation.json"}
                         or name.startswith(("learning-run/", "policy-evaluation-run/")) for name in members)
             and not any(key in document for document in (scene, task, contract)
                         for key in ("avatar", "avatar_asset", "avatar_sha256", "learned_policy", "exercise_policy")),
             "Camera receipt cannot claim unqualified learning or appearance")
    _require(runtime_required(scene) <= native.keys(), "Camera runtime lacks its complete versioned recording dependency closure")
    pins = {key: _hash(task.get(key)) for key in IDENTITIES}
    coupling, body, context = scene.get("camera_coupling"), scene.get("human_spec"), task.get("metric_context")
    _require(isinstance(coupling, dict) and isinstance(body, dict) and isinstance(context, dict),
             "Camera receipt needs exact coupling, body and metric context")
    _require(pins["camera_coupling_sha256"] == _canonical(coupling)
             and pins["body_sha256"] == _canonical(body) == coupling.get("body_sha256")
             and pins["source_sha256"] == body.get("source_sha256") == coupling.get("source_sha256")
             and pins["source_clock_sha256"] == coupling.get("source_clock_sha256")
             and pins["metric_context_sha256"] == _canonical(context), "Camera source/body/clock/context identities differ")
    _require(pins["kernel_evaluator_sha256"] == native["shared_exercise_task.py"]
             and pins["adapter_implementation_sha256"] == native["camera_shared_task.py"]
             and pins["evaluator_sha256"] == _canonical({name: native[name] for name in (
                 "camera_shared_task.py", "shared_exercise_task.py")}), "Camera scoring identities differ from the frozen runtime")
    interval, control = scene.get("task"), task.get("control")
    _require(isinstance(interval, dict) and task.get("interval") == {
                 "start_offset_s": interval.get("start_offset_s"), "horizon_steps": interval.get("horizon_steps")}
             and isinstance(control, dict) and set(control) == {"root_assistance", "object_forces"}
             and control.get("root_assistance") in ("disabled", "explicit_3_axis")
             and control.get("object_forces") == "forbidden", "Camera receipt changed source interval or controls")
    arguments = context.get("kernel_arguments")
    _require(context.get("profile") == "camera-shared-metric-context-v1" and context.get("valid_legacy_task") is False
             and context.get("scene_sha256") == task["scene_sha256"] == _canonical(scene)
             and context.get("kernel_branch") == "shared-exercise-task-v2"
             and context.get("projection") == "identity_22_motor_samples_without_equipment"
             and context.get("control_dt_s") == scene.get("dt")
             and context.get("source_body_dt_s") == body.get("dt")
             and context.get("substeps") == scene.get("substeps")
             and isinstance(arguments, dict) and context.get("kernel_arguments_sha256") == _canonical(arguments)
             and arguments == {"profile": "shared-exercise-task-v2", "interval": task["interval"],
                 "control": control, "equipment_ids": [], "contact_requirements": [], "source_scope": task["scope"]},
             "Camera metric context differs from its actual scene or unchanged human-only arithmetic selector")
    _require(all(contract[key] == value for key, value in pins.items() if key in contract),
             "Camera lab metadata contradicts its task identities")
    return pins
