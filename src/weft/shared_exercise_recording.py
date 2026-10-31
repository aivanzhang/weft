"""A distinct recording envelope keeps exercise gates separate from legacy scores."""

from copy import deepcopy
import hashlib
import math
import time

import numpy as np
from PIL import Image, ImageDraw

from weft.io import canonical_hash
from weft.shared_exercise_task import (
    HUMAN_ONLY_TASK_PROFILE,
    SharedExerciseTaskEnv,
    validate_shared_exercise_snapshot,
    validate_shared_exercise_task,
)
from weft.shared_world import validate_shared_world


PROFILE = "shared-exercise-recording-v1"


def recording_scene(scene, task, *, policy_document=None):
    scene = validate_shared_world(scene)
    task = validate_shared_exercise_task(task, scene)
    result = {"schema_version": 1, "profile": PROFILE, "engine_scene": scene, "exercise_task": task,
              "dt": scene["dt"], "objects": deepcopy(scene["objects"])}
    if policy_document is not None:
        from weft.shared_exercise_lab import _json
        from weft.shared_exercise_policies import validate_policy

        if not isinstance(policy_document, bytes) or not 1 <= len(policy_document) <= 64 * 1024:
            raise ValueError("Exercise recording policy requires bounded exact document bytes")
        validate_policy(_json(policy_document), scene, task)
        result["learned_policy"] = {"document_utf8": policy_document.decode("utf-8"),
                                    "sha256": hashlib.sha256(policy_document).hexdigest()}
    return result


def validate_recording_scene(scene):
    keys = {"schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects"}
    if (not isinstance(scene, dict) or set(scene) not in (keys, keys | {"learned_policy"})
            or type(scene["schema_version"]) is not int or scene["schema_version"] != 1 or scene["profile"] != PROFILE):
        raise ValueError("Exercise recording envelope differs from its versioned contract")
    policy_document = None
    if "learned_policy" in scene:
        policy = scene["learned_policy"]
        if (not isinstance(policy, dict) or set(policy) != {"document_utf8", "sha256"}
                or not isinstance(policy["document_utf8"], str) or len(policy["document_utf8"]) > 64 * 1024):
            raise ValueError("Exercise recording learned-policy envelope is invalid")
        try:
            policy_document = policy["document_utf8"].encode("utf-8")
        except UnicodeError as error:
            raise ValueError("Exercise recording policy must be UTF-8") from error
        if hashlib.sha256(policy_document).hexdigest() != policy["sha256"]:
            raise ValueError("Exercise recording policy hash differs from its exact document")
    result = recording_scene(scene["engine_scene"], scene["exercise_task"], policy_document=policy_document)
    if scene != result:
        raise ValueError("Exercise recording aliases must preserve the exact engine scene and task")
    return result


def validate_action(action, scene):
    control = scene["exercise_task"]["control"]["root_assistance"]
    count = 22 if control == "explicit_3_axis" else 19
    if (not isinstance(action, (tuple, list)) or len(action) != count
            or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1 for value in action)):
        raise ValueError(f"Exercise recording requires {count} finite human-only actions in [-1,1]")


def validate_snapshot(snapshot, scene):
    validate_shared_exercise_snapshot(snapshot, scene["engine_scene"], scene["exercise_task"])


def prepare_snapshot_validator(scene):
    from weft.shared_exercise_task import prepare_shared_exercise_snapshot_validator

    return prepare_shared_exercise_snapshot_validator(scene["engine_scene"], scene["exercise_task"])


def validate_event(event, scene):
    state = event["snapshot"]
    learned = event.get("controller") == "learned"
    if learned:
        if (event["type"] != "step" or "learned_policy" not in scene
                or event.get("policy_sha256") != scene["learned_policy"]["sha256"]):
            raise ValueError("Exercise learned action requires its pinned policy hash and document")
    elif "policy_sha256" in event:
        raise ValueError("Exercise policy identity belongs only to learned actions")
    if event["type"] == "step":
        validate_action(event.get("action"), scene)
        if "controller" in event:
            if event["controller"] not in ("manual", "coast", "authored", "learned"):
                raise ValueError("Exercise controller metadata must be manual, coast, authored or learned")
            if event["controller"] == "coast" and any(event["action"]):
                raise ValueError("Exercise coast metadata requires zero human actions")
        if list(event["action"]) != state["last_action"]:
            raise ValueError("Exercise event action must match the task snapshot")
        if type(event.get("reward")) not in (int, float) or not math.isfinite(event["reward"]):
            raise ValueError("Exercise recorded reward must be finite")
        if any(type(event.get(name)) is not bool for name in ("terminated", "truncated")):
            raise ValueError("Exercise recorded step requires explicit termination flags")
        if any(event[name] != state[name] for name in ("terminated", "truncated")):
            raise ValueError("Exercise event termination must agree with its task snapshot")
    else:
        if type(event.get("seed")) is not int or not 0 <= event["seed"] <= 2**32 - 1:
            raise ValueError("Exercise reset seed must be a bounded integer")
        if any(state["last_action"]) or state["engine_state_status"] != "current":
            raise ValueError("Exercise start/reset requires a current state with zero action")
        engine, metrics = state["engine_state"], state["task_metrics"]
        zero_metrics = ("observed_substeps", "valid_substeps", "simulated_duration_s", "valid_observed_duration_s",
                        "joint_effort_mean_squared", "root_assistance_mean_squared", "reward_integral", "mean_reward",
                        "failure_reward_penalty")
        if (any(engine[key] != 0 for key in ("steps", "episode_steps", "time_s"))
                or any(metrics[key] != 0 for key in zero_metrics)
                or any(state[key] or engine[key] for key in ("terminated", "truncated"))
                or metrics["failure"] is not None or metrics["clock_status"] != "observed"
                or metrics["success"] or metrics["completed_interval"]):
            raise ValueError("Exercise start/reset requires a genuine initial boundary, not an advanced coast state")
        # Reset jitter can create real equipment velocities; only accumulated history must be empty.
        for row in metrics["contact_requirements"]:
            if any(row[key] != 0 for key in ("observed_duration_s", "qualifying_duration_s", "current_gap_s", "max_gap_s")):
                raise ValueError("Exercise start/reset cannot carry accumulated contact observations")


def replay_view(scene, events):
    return {"scene": scene["engine_scene"], "exercise_task": scene["exercise_task"],
            "events": [{**event, "snapshot": event["snapshot"]["engine_state"],
                        "exercise_metrics": event["snapshot"]["task_metrics"],
                        "engine_state_status": event["snapshot"]["engine_state_status"]}
                       for event in events]}


def render_snapshot(snapshot, scene, width=640):
    return prepare_renderer(scene, width=width)(snapshot)


def prepare_renderer(scene, width=640):
    from weft.shared_world import prepare_shared_renderer

    scene = validate_recording_scene(scene)
    validate = prepare_snapshot_validator(scene)
    render = prepare_shared_renderer(scene["engine_scene"], width=width, show_target=False)

    def frame(snapshot):
        validate(snapshot)
        return _annotate_frame(render(snapshot["engine_state"]), snapshot)

    return frame


def _annotate_frame(frame, snapshot):
    image = Image.new("RGB", (frame.shape[1], frame.shape[0] + 64), "#F5EADC")
    image.paste(Image.fromarray(frame), (0, 0))
    draw = ImageDraw.Draw(image)
    metrics = snapshot["task_metrics"]
    result = "PASSED" if metrics["success"] else "NOT PASSED" if metrics["completed_interval"] or metrics["failure"] else "IN PROGRESS"
    person_only = metrics["profile"] == HUMAN_ONLY_TASK_PROFILE
    if person_only:
        heading = f"PERSON-ONLY TASK | {result} | equipment/contact: N/A"
    else:
        drift = max((row["displacement_model_units"] for row in metrics["equipment"]), default=0.)
        heading = f"HUMAN-ONLY TASK | {result} | maximum equipment drift {drift:.3f} model units"
    draw.text((10, frame.shape[0] + 4), heading, fill="#214F50")
    note = (f"Last valid geometry; simulation failed at {metrics['simulated_duration_s']:.3f}s."
            if snapshot["engine_state_status"] != "current" else
            "Person and ground only. Simulated task success is not verified biomechanics." if person_only else
            "Passive equipment; no object forces. Simulated task success is not verified grasp or biomechanics.")
    draw.text((10, frame.shape[0] + 27), note, fill="#965644")
    return np.asarray(image)


def verify_action_replay(directory, *, max_seconds=60):
    """Re-execute saved human actions and compare every task/engine state and outcome."""
    from weft.recording import load_recording

    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 180:
        raise ValueError("Action replay requires a positive bound of at most 180 seconds")
    deadline = time.monotonic() + max_seconds
    loaded = load_recording(directory, deadline=deadline)
    scene, events = loaded["scene"], loaded["events"]
    if scene.get("profile") != PROFILE:
        raise ValueError("Action replay requires a shared exercise recording")
    env = SharedExerciseTaskEnv(scene["engine_scene"], scene["exercise_task"])
    try:
        policy_action = None
        if "learned_policy" in scene:
            from weft.shared_exercise_lab import _json
            from weft.shared_exercise_policies import prepare_policy
            policy_action = prepare_policy(_json(scene["learned_policy"]["document_utf8"].encode("utf-8")), env)
        if events[0]["snapshot"]["task_metrics"]["evaluator_sha256"] != env.evaluator_sha256:
            raise ValueError("Recorded evaluator differs from this runtime; use the pinned implementation for action replay")
        for event in events:
            if time.monotonic() > deadline:
                raise TimeoutError("Exercise action replay exceeded its declared wall-time bound")
            if event["type"] in ("start", "reset"):
                env.reset(seed=event["seed"], options=event["snapshot"]["resolved_reset_options"])
            else:
                if event.get("controller") == "authored" and env.authored_action().tolist() != event["action"]:
                    raise ValueError(f"Exercise authored controller replay differs at event {event['index']}")
                if event.get("controller") == "learned" and (policy_action is None or policy_action().tolist() != event["action"]):
                    raise ValueError(f"Exercise learned controller replay differs at event {event['index']}")
                _, reward, terminated, truncated, _ = env.step(event["action"])
                if (reward, terminated, truncated) != (event["reward"], event["terminated"], event["truncated"]):
                    raise ValueError(f"Exercise outcome replay differs at event {event['index']}")
            if env.snapshot() != event["snapshot"]:
                raise ValueError(f"Exercise state replay differs at event {event['index']}")
        if time.monotonic() > deadline:
            raise TimeoutError("Exercise action replay exceeded its declared wall-time bound")
        return {"events": len(events), "steps": sum(event["type"] == "step" for event in events),
                "exact_state_replay": True, "scene_sha256": canonical_hash(scene["engine_scene"]),
                "task_sha256": canonical_hash(scene["exercise_task"]),
                **({"policy_sha256": scene["learned_policy"]["sha256"],
                    "learned_actions_verified": sum(event.get("controller") == "learned" for event in events)}
                   if "learned_policy" in scene else {}),
                "scope": "Exact simulated action/state/task replay; not physical validation"}
    finally:
        env.close()
