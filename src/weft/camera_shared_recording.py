"""Exact camera task recording without legacy reclassification, policies or physical claims."""

from copy import deepcopy
import math
import time

from weft.camera_shared_receipt import IDENTITIES, PROFILE, PROTOCOL
from weft.camera_shared_state import _validate_action_values, prepare_camera_shared_snapshot_validator
from weft.camera_shared_task import CameraSharedTaskEnv, prepare_camera_shared_task
from weft.io import canonical_hash


def recording_scene(scene, task, *, policy_document=None):
    if policy_document is not None:
        raise ValueError("Camera recordings do not support an unqualified learned policy")
    actual, task = prepare_camera_shared_task(task, scene)
    return {"schema_version": 1, "profile": PROFILE, "engine_scene": actual,
            "exercise_task": task, "dt": actual["dt"], "objects": []}


def validate_recording_scene(scene):
    if (not isinstance(scene, dict) or set(scene) != {
            "schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects"}
            or type(scene["schema_version"]) is not int or scene["schema_version"] != 1 or scene["profile"] != PROFILE):
        raise ValueError("Camera recording fields differ; policies and avatars are unsupported")
    expected = recording_scene(scene["engine_scene"], scene["exercise_task"])
    if canonical_hash(scene) != canonical_hash(expected):
        raise ValueError("Camera recording aliases must preserve the exact actual world and task")
    return expected


def validate_action(action, scene):
    mode = scene["exercise_task"]["control"]["root_assistance"]
    if mode not in ("disabled", "explicit_3_axis"):
        raise ValueError("Camera recordings require explicit root authority")
    _validate_action_values(action, 22 if mode == "explicit_3_axis" else 19)


def prepare_snapshot_validator(scene):
    return prepare_camera_shared_snapshot_validator(scene["engine_scene"], scene["exercise_task"])


def validate_snapshot(snapshot, scene):
    prepare_snapshot_validator(scene)(snapshot)


def validate_event(event, scene):
    if not isinstance(event, dict) or event.get("type") not in ("start", "reset", "step"):
        raise ValueError("Camera recording supports start/reset/step only; cancellation is lifecycle metadata")
    step = event["type"] == "step"
    required = ({"type", "action", "reward", "terminated", "truncated", "snapshot"} if step
                else {"type", "seed", "snapshot"})
    optional = {"index", "controller"} if step else {"index"}
    if not required <= event.keys() or event.keys() - required - optional:
        raise ValueError("Camera event fields differ; policy/proposal/avatar claims are unsupported")
    if "index" in event and (type(event["index"]) is not int or event["index"] < 0):
        raise ValueError("Camera journal index must be a nonnegative integer")
    state = event["snapshot"]
    if not isinstance(state, dict) or not isinstance(state.get("task_metrics"), dict):
        raise ValueError("Camera event requires its task snapshot")
    metrics = state["task_metrics"]
    if metrics.get("failure") == "cancelled":
        raise ValueError("Camera cancellation cannot masquerade as a recorded control or reset")
    if step:
        validate_action(event["action"], scene)
        controller = event.get("controller", "manual")
        if controller not in ("manual", "coast", "authored"):
            raise ValueError("Camera controllers are manual, coast or authored only")
        if controller == "coast" and any(event["action"]):
            raise ValueError("Camera coast controls must be zero")
        if list(event["action"]) != state["last_action"]:
            raise ValueError("Camera event action differs from its task snapshot")
        if (type(event["reward"]) not in (int, float) or not math.isfinite(event["reward"])
                or any(type(event[key]) is not bool or event[key] != state[key] for key in ("terminated", "truncated"))):
            raise ValueError("Camera event outcome must be finite and agree with task termination")
        return
    if type(event["seed"]) is not int or not 0 <= event["seed"] < 2**32:
        raise ValueError("Camera reset seed must be a bounded integer")
    engine = state["engine_state"]
    zeros = ("observed_substeps", "valid_substeps", "simulated_duration_s", "valid_observed_duration_s",
             "reward_integral", "mean_reward", "failure_reward_penalty",
             "joint_effort_mean_squared", "root_assistance_mean_squared")
    if (state["engine_state_status"] != "current" or any(state["last_action"]) or any(engine["normalized_action"])
            or any(engine[key] != 0 for key in ("steps", "episode_steps", "time_s"))
            or any(metrics[key] != 0 for key in zeros)
            or any(state[key] or engine[key] for key in ("terminated", "truncated"))
            or metrics["failure"] is not None or metrics["success"] or metrics["completed_interval"]
            or metrics["clock_status"] != "observed"):
        raise ValueError("Camera start/reset requires an empty genuine initial control boundary")


def replay_view(scene, events):
    task = scene["exercise_task"]
    return {"scene": scene["engine_scene"], "exercise_task": task,
            "camera_identity": {"scene_sha256": task["scene_sha256"], "task_sha256": canonical_hash(task),
                                **{key: task[key] for key in IDENTITIES}},
            "camera_recording_scope": "Actual camera-prior simulation and tracking criteria; not verified support, balance or anatomy",
            "events": [{**event, "snapshot": event["snapshot"]["engine_state"],
                        "exercise_metrics": event["snapshot"]["task_metrics"],
                        "engine_state_status": event["snapshot"]["engine_state_status"]} for event in events]}


def _annotate_frame(frame, snapshot):
    import numpy as np
    from PIL import Image, ImageDraw

    metrics = snapshot["task_metrics"]
    result = "PASSED" if metrics["success"] else "NOT PASSED" if metrics["completed_interval"] or metrics["failure"] else "IN PROGRESS"
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + 64), "#F5EADC")
    image.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(image)
    draw.text((10, frame.shape[0] + 4), f"CAMERA TRACKING TASK | {result} | equipment/contact: N/A", fill="#214F50")
    note = (f"Last valid geometry; last observed time {metrics['simulated_duration_s']:.3f}s."
            if snapshot["engine_state_status"] != "current" else
            "Supplied camera priors; no verified support, balance or measured anatomy.")
    draw.text((10, frame.shape[0] + 27), note, fill="#965644")
    return np.asarray(image)


def prepare_renderer(scene, width=640):
    from weft.shared_world import prepare_shared_renderer

    scene = validate_recording_scene(scene)
    validate = prepare_snapshot_validator(scene)
    render = prepare_shared_renderer(scene["engine_scene"], width=width, show_target=False)

    def frame(snapshot):
        validate(snapshot)
        return _annotate_frame(render(snapshot["engine_state"]), snapshot)

    return frame


def render_snapshot(snapshot, scene, width=640):
    return prepare_renderer(scene, width=width)(snapshot)


def verify_action_replay(directory, *, max_seconds=60):
    from weft.recording import load_recording

    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 180:
        raise ValueError("Camera replay requires a positive bound of at most 180 seconds")
    deadline = time.monotonic() + max_seconds
    loaded = load_recording(directory, deadline=deadline)
    scene, events = loaded["scene"], loaded["events"]
    if scene.get("profile") != PROFILE or not events:
        raise ValueError("Camera replay requires a nonempty camera task recording")
    if time.monotonic() >= deadline:
        raise TimeoutError("Camera replay exceeded its wall-time bound before engine construction")
    env = CameraSharedTaskEnv(scene["engine_scene"], scene["exercise_task"])
    try:
        if events[0]["snapshot"]["task_metrics"]["evaluator_sha256"] != env.evaluator_sha256:
            raise ValueError("Camera evaluator differs; use the pinned implementation")
        for event in events:
            if time.monotonic() >= deadline:
                raise TimeoutError("Camera replay exceeded its declared wall-time bound")
            validate_event(event, scene)
            if event["type"] in ("start", "reset"):
                env.reset(seed=event["seed"], options=event["snapshot"]["resolved_reset_options"])
            else:
                if event.get("controller") == "authored" and env.authored_action().tolist() != event["action"]:
                    raise ValueError(f"Camera authored action differs at event {event['index']}")
                _, reward, terminated, truncated, _ = env.step(event["action"])
                if (reward, terminated, truncated) != (event["reward"], event["terminated"], event["truncated"]):
                    raise ValueError(f"Camera replay outcome differs at event {event['index']}")
            if env.snapshot() != event["snapshot"]:
                raise ValueError(f"Camera full state/task replay differs at event {event['index']}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Camera replay exceeded its declared wall-time bound")
        task = scene["exercise_task"]
        return {"protocol": PROTOCOL, "events": len(events), "steps": sum(row["type"] == "step" for row in events),
                "exact_state_replay": True, "scene_sha256": task["scene_sha256"], "task_sha256": canonical_hash(task),
                **{key: task[key] for key in IDENTITIES}, "policy_sha256": None, "learned_actions_verified": None,
                "scope": "Exact simulated actual-world/action/task replay; not physical validation"}
    finally:
        env.close()
