"""Exact external-policy decisions and replay on the unchanged actual camera task."""

from copy import deepcopy
from hashlib import sha256
import math
import time

from weft import camera_shared_recording as camera
from weft.camera_shared_policy_admission import prepare_camera_policy_sampler, validate_camera_policy_admission
from weft.camera_shared_policy_receipt import PROFILE, PROTOCOL
from weft.camera_shared_receipt import IDENTITIES
from weft.camera_shared_task import CameraSharedTaskEnv, prepare_camera_shared_task
from weft.io import canonical_hash


POLICY_EVENT_FIELDS = frozenset({"policy_sha256", "policy_admission_sha256", "policy_sample"})
ENVELOPE_FIELDS = frozenset({"schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects",
                            "learned_policy", "policy_admission"})


def _policy_document(scene):
    embedded = scene.get("learned_policy")
    if (not isinstance(embedded, dict) or set(embedded) != {"document_utf8", "sha256"}
            or not isinstance(embedded["document_utf8"], str) or len(embedded["document_utf8"]) > 64 << 10):
        raise ValueError("Camera policy recording needs its bounded exact policy document")
    try:
        raw = embedded["document_utf8"].encode("utf-8")
    except UnicodeError as error:
        raise ValueError("Camera policy document must be valid UTF-8") from error
    if len(raw) > 64 << 10 or sha256(raw).hexdigest() != embedded["sha256"]:
        raise ValueError("Camera recording policy bytes differ from their identity")
    return raw


def recording_scene(scene, task, *, policy_document, policy_admission):
    if type(policy_document) is not bytes or not 0 < len(policy_document) <= 64 << 10:
        raise ValueError("Camera policy recording requires bounded original policy bytes")
    actual, task = prepare_camera_shared_task(task, scene)
    admitted = validate_camera_policy_admission(policy_admission, actual, task, policy_document)
    return {"schema_version": 1, "profile": PROFILE, "engine_scene": actual, "exercise_task": task,
            "dt": actual["dt"], "objects": [], "policy_admission": admitted,
            "learned_policy": {"document_utf8": policy_document.decode("utf-8"),
                               "sha256": sha256(policy_document).hexdigest()}}


def validate_recording_scene(scene):
    if (not isinstance(scene, dict) or set(scene) != ENVELOPE_FIELDS
            or type(scene["schema_version"]) is not int or scene["schema_version"] != 1 or scene["profile"] != PROFILE):
        raise ValueError("Camera policy recording requires its distinct envelope; proposals and avatars are unsupported")
    expected = recording_scene(scene["engine_scene"], scene["exercise_task"],
                               policy_document=_policy_document(scene), policy_admission=scene["policy_admission"])
    if canonical_hash(scene) != canonical_hash(expected):
        raise ValueError("Camera policy recording aliases changed the actual world or task")
    return expected


validate_action = camera.validate_action
prepare_snapshot_validator = camera.prepare_snapshot_validator
validate_snapshot = camera.validate_snapshot


def validate_event(event, scene):
    if not isinstance(event, dict) or event.get("controller") != "learned":
        camera.validate_event(event, scene)
        return
    if event.get("type") != "step" or not POLICY_EVENT_FIELDS <= event.keys():
        raise ValueError("Camera learned controls require their policy decision metadata")
    admission = scene["policy_admission"]
    if (event["policy_sha256"] != scene["learned_policy"]["sha256"]
            or event["policy_admission_sha256"] != canonical_hash(admission)):
        raise ValueError("Camera learned control policy or admission differs")
    sample = event["policy_sample"]
    if (not isinstance(sample, dict) or set(sample) != {"phase", "authored_action", "applied_joint_delta"}
            or type(sample["phase"]) not in (int, float) or not 0 <= sample["phase"] <= 1
            or not math.isfinite(sample["phase"])):
        raise ValueError("Camera learned control requires its finite original phase and realized action delta")
    validate_action(sample["authored_action"], scene)
    validate_action(event.get("action"), scene)
    delta, authored, action = sample["applied_joint_delta"], sample["authored_action"], event["action"]
    if (not isinstance(delta, list) or len(delta) != 19
            or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > .25 + 1e-12 for value in delta)
            or delta != [float(a - b) for a, b in zip(action[:19], authored[:19], strict=True)]
            or list(action[19:]) != list(authored[19:])):
        raise ValueError("Camera learned action delta or unchanged authored root controls differ")
    # The validation-only view reuses outcome/reset rules; saved controller identity stays learned.
    ordinary = {key: value for key, value in event.items() if key not in POLICY_EVENT_FIELDS}
    camera.validate_event({**ordinary, "controller": "manual"}, scene)


def replay_view(scene, events):
    result = camera.replay_view(scene, events)
    admission = scene["policy_admission"]
    result.update(camera_policy_admission=deepcopy(admission), policy_admission_sha256=canonical_hash(admission),
                  policy_sha256=scene["learned_policy"]["sha256"], camera_policy_summary=deepcopy(admission["summary"]))
    return result


def prepare_renderer(scene, width=640):
    from weft.shared_world import prepare_shared_renderer

    scene = validate_recording_scene(scene)
    validate = prepare_snapshot_validator(scene)
    render = prepare_shared_renderer(scene["engine_scene"], width=width, show_target=False)

    def frame(snapshot):
        validate(snapshot)
        return camera._annotate_frame(render(snapshot["engine_state"]), snapshot)

    return frame


def render_snapshot(snapshot, scene, width=640):
    return prepare_renderer(scene, width=width)(snapshot)


def verify_action_replay(directory, *, max_seconds=60):
    from weft.recording import load_recording

    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 180:
        raise ValueError("Camera policy replay requires a positive bound of at most 180 seconds")
    deadline = time.monotonic() + max_seconds
    loaded = load_recording(directory, deadline=deadline)
    scene, events = loaded["scene"], loaded["events"]
    if scene.get("profile") != PROFILE or not events:
        raise ValueError("Camera policy replay requires its nonempty distinct recording")
    if time.monotonic() >= deadline:
        raise TimeoutError("Camera policy replay exceeded its bound before engine construction")
    env = CameraSharedTaskEnv(scene["engine_scene"], scene["exercise_task"])
    try:
        sample = prepare_camera_policy_sampler(_policy_document(scene), scene["policy_admission"], env)
        if events[0]["snapshot"]["task_metrics"]["evaluator_sha256"] != env.evaluator_sha256:
            raise ValueError("Camera policy replay evaluator differs from the pinned implementation")
        learned = 0
        for event in events:
            if time.monotonic() >= deadline:
                raise TimeoutError("Camera policy replay exceeded its cooperative deadline")
            validate_event(event, scene)
            if event["type"] in ("start", "reset"):
                env.reset(seed=event["seed"], options=event["snapshot"]["resolved_reset_options"])
            else:
                if event.get("controller") == "learned":
                    action, diagnostic = sample()
                    if action.tolist() != event["action"] or diagnostic != event["policy_sample"]:
                        raise ValueError(f"Camera learned action or realized delta differs at event {event.get('index')}")
                    learned += 1
                elif event.get("controller") == "authored" and env.authored_action().tolist() != event["action"]:
                    raise ValueError(f"Camera authored action differs at event {event.get('index')}")
                _, reward, terminated, truncated, _ = env.step(event["action"])
                if (reward, terminated, truncated) != (event["reward"], event["terminated"], event["truncated"]):
                    raise ValueError(f"Camera policy replay outcome differs at event {event.get('index')}")
            if env.snapshot() != event["snapshot"]:
                raise ValueError(f"Camera policy full state/task replay differs at event {event.get('index')}")
        if time.monotonic() >= deadline:
            raise TimeoutError("Camera policy replay exceeded its cooperative deadline")
        task = scene["exercise_task"]
        return {"protocol": PROTOCOL, "events": len(events), "steps": sum(row["type"] == "step" for row in events),
                "exact_state_replay": True, "scene_sha256": task["scene_sha256"], "task_sha256": canonical_hash(task),
                **{key: task[key] for key in IDENTITIES}, "policy_sha256": scene["learned_policy"]["sha256"],
                "policy_admission_sha256": canonical_hash(scene["policy_admission"]), "learned_actions_verified": learned,
                "scope": "Exact external-policy decisions and simulated camera task replay; not physical validation"}
    finally:
        env.close()
