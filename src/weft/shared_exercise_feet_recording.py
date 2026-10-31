"""Versioned feet recordings retain the actual world and unchanged base scoring provenance."""

from copy import deepcopy
import math
import time

from weft.io import canonical_hash
from weft.shared_exercise_feet_projection import scatter_task_action
from weft.shared_exercise_feet_task import (
    SharedExerciseFeetTaskEnv, prepare_shared_exercise_feet_task,
)
from weft.shared_exercise_feet_state import prepare_shared_exercise_feet_snapshot_validator


PROFILE = "shared-exercise-feet-recording-v1"


def recording_scene(scene, task):
    actual, _, task = prepare_shared_exercise_feet_task(task, scene)
    return {"schema_version": 1, "profile": PROFILE, "engine_scene": actual,
            "exercise_task": task, "dt": actual["dt"], "objects": deepcopy(actual["objects"])}


def validate_recording_scene(scene):
    if (not isinstance(scene, dict) or set(scene) != {
            "schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects"}
            or type(scene["schema_version"]) is not int or scene["schema_version"] != 1
            or scene["profile"] != PROFILE):
        raise ValueError("Feet recording fields differ from their versioned contract; policies and avatars are unsupported")
    expected = recording_scene(scene["engine_scene"], scene["exercise_task"])
    if scene != expected or canonical_hash(scene) != canonical_hash(expected):
        raise ValueError("Feet recording aliases must preserve the exact actual world and task")
    return expected


def validate_action(action, scene):
    control = scene["exercise_task"]["base_task"]["control"]["root_assistance"]
    if control not in ("disabled", "explicit_3_axis"):
        raise ValueError("Feet recording requires an explicit root-assistance setting")
    if not isinstance(action, (list, tuple)):
        raise ValueError("Recorded feet actions require an explicit numeric vector")
    scatter_task_action(action, engine_action_count=len(scene["engine_scene"]["action_contract"]["names"]),
                        root_assistance=control == "explicit_3_axis")


def prepare_snapshot_validator(scene):
    return prepare_shared_exercise_feet_snapshot_validator(scene["engine_scene"], scene["exercise_task"])


def validate_snapshot(snapshot, scene):
    prepare_snapshot_validator(scene)(snapshot)


def validate_event(event, scene):
    if not isinstance(event, dict) or event.get("type") not in ("start", "reset", "step"):
        raise ValueError("Feet recording events require start, reset or step")
    if any(key in event for key in ("policy_sha256", "reference_controller_sha256")):
        raise ValueError("Feet recordings cannot claim an unqualified learned or proposed controller")
    step = event["type"] == "step"
    if not step and "controller" in event:
        raise ValueError("Feet controller identity belongs only to action events")
    required = ({"type", "action", "reward", "terminated", "truncated", "snapshot"} if step
                else {"type", "seed", "snapshot"})
    optional = {"index", "controller"} if step else {"index"}
    if not required <= event.keys() or event.keys() - required - optional:
        raise ValueError("Feet fields differ from the versioned event contract")
    if "index" in event and (type(event["index"]) is not int or event["index"] < 0):
        raise ValueError("Feet journal index must be a nonnegative integer")
    state = event["snapshot"]
    if event["type"] == "step":
        validate_action(event.get("action"), scene)
        controller = event.get("controller", "manual")
        if controller not in ("manual", "coast", "authored"):
            raise ValueError("Feet recordings support manual, coast or authored actions only")
        if controller == "coast" and any(event["action"]):
            raise ValueError("Feet coast actions must be zero")
        if list(event["action"]) != state["last_action"]:
            raise ValueError("Feet event action differs from its task snapshot")
        if type(event.get("reward")) not in (int, float) or not math.isfinite(event["reward"]):
            raise ValueError("Feet event reward must be finite")
        if (any(type(event.get(key)) is not bool for key in ("terminated", "truncated"))
                or any(event[key] != state[key] for key in ("terminated", "truncated"))):
            raise ValueError("Feet event termination must agree with its task snapshot")
        return
    if type(event.get("seed")) is not int or not 0 <= event["seed"] <= 2**32 - 1:
        raise ValueError("Feet reset seed must be a bounded integer")
    engine, metrics, ankles = state["engine_state"], state["task_metrics"], state["ankle_diagnostics"]
    zero_metrics = ("observed_substeps", "valid_substeps", "simulated_duration_s", "valid_observed_duration_s",
                    "reward_integral", "mean_reward", "failure_reward_penalty")
    if (state["engine_state_status"] != "current" or any(state["last_action"])
            or any(engine["normalized_action"])
            or any(engine[key] != 0 for key in ("steps", "episode_steps", "time_s"))
            or any(metrics[key] != 0 for key in zero_metrics)
            or any(state[key] or engine[key] for key in ("terminated", "truncated"))
            or metrics["failure"] is not None or metrics["success"] or metrics["completed_interval"]
            or ankles["observed_substeps"] != 0 or ankles["latest_substep"] is not None):
        raise ValueError("Feet start/reset requires an empty, genuine initial control boundary")


def replay_view(scene, events):
    return {"scene": scene["engine_scene"], "exercise_task": scene["exercise_task"],
            "feet_recording_scope": "Actual coupled feet; original body scoring, unscored ankle diagnostics; not physical validation",
            "events": [{**event, "snapshot": event["snapshot"]["engine_state"],
                        "exercise_metrics": event["snapshot"]["task_metrics"],
                        "ankle_diagnostics": event["snapshot"]["ankle_diagnostics"],
                        "engine_state_status": event["snapshot"]["engine_state_status"]}
                       for event in events]}


def prepare_renderer(scene, width=640):
    from weft.shared_exercise_recording import _annotate_frame
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
    """Replay full v5 state and base scores; this does not qualify a learned policy."""
    from weft.recording import load_recording

    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 180:
        raise ValueError("Feet action replay requires a positive bound of at most 180 seconds")
    deadline = time.monotonic() + max_seconds
    loaded = load_recording(directory, deadline=deadline)
    scene, events = loaded["scene"], loaded["events"]
    if scene.get("profile") != PROFILE or not events:
        raise ValueError("Action replay requires a nonempty feet exercise recording")
    env = SharedExerciseFeetTaskEnv(scene["engine_scene"], scene["exercise_task"])
    try:
        if events[0]["snapshot"]["task_metrics"]["evaluator_sha256"] != env.evaluator_sha256:
            raise ValueError("Recorded base evaluator differs; use the pinned implementation")
        for event in events:
            if time.monotonic() >= deadline:
                raise TimeoutError("Feet action replay exceeded its declared wall-time bound")
            if event["type"] in ("start", "reset"):
                env.reset(seed=event["seed"], options=event["snapshot"]["resolved_reset_options"])
            else:
                if event.get("controller") == "authored" and env.authored_action().tolist() != event["action"]:
                    raise ValueError(f"Feet authored action differs at event {event['index']}")
                _, reward, terminated, truncated, _ = env.step(event["action"])
                if (reward, terminated, truncated) != (event["reward"], event["terminated"], event["truncated"]):
                    raise ValueError(f"Feet replay outcome differs at event {event['index']}")
            if env.snapshot() != event["snapshot"]:
                raise ValueError(f"Feet full state/base score replay differs at event {event['index']}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Feet action replay exceeded its declared wall-time bound")
        return {"events": len(events), "steps": sum(event["type"] == "step" for event in events),
                "exact_state_replay": True, "scene_sha256": canonical_hash(scene["engine_scene"]),
                "task_sha256": canonical_hash(scene["exercise_task"]),
                "base_scene_sha256": scene["exercise_task"]["base_scene_sha256"],
                "base_task_sha256": scene["exercise_task"]["base_task_sha256"],
                "projection_contract_sha256": scene["exercise_task"]["projection_contract_sha256"],
                "adapter_implementation_sha256": scene["exercise_task"]["adapter_implementation_sha256"],
                "foot_reference_sha256": scene["exercise_task"]["foot_reference_sha256"],
                "scope": "Exact simulated actual-world/actions/base-scores/ankle-diagnostics replay; not physical validation"}
    finally:
        env.close()
