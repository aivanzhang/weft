"""Record camera-angle control alternatives without changing the original task."""

from functools import partial
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from weft import camera_reference_controller as controller
from weft import camera_shared_recording as camera
from weft.camera_shared_receipt import IDENTITIES, RUNTIME_REQUIRED
from weft.camera_shared_task import CameraSharedTaskEnv, prepare_camera_shared_task
from weft.io import canonical_hash
from weft.shared_exercise_reference_recording import _document, _trusted_velocity_index
from weft.shared_world_packaging import read_package_file


PROFILE = "camera-shared-reference-recording-v1"
PROTOCOL = "camera-shared-reference-full-state-replay-v1"
REQUIRED_RUNTIME_FILES = RUNTIME_REQUIRED | controller.REQUIRED_RUNTIME_FILES | frozenset({
    "camera_reference_recording.py", "camera_shared_recording.py", "camera_shared_state.py",
    "shared_exercise_reference_recording.py", "shared_world_packaging.py", "recording.py"})
MAX_RUNTIME_BYTES = 8 << 20
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = frozenset({"schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects",
                     "reference_controller"})


def _require(value, message):
    if not value:
        raise ValueError("Camera reference recording: " + message)


def _runtime_hashes(declared):
    _require(type(declared) is dict and REQUIRED_RUNTIME_FILES <= declared.keys() and len(declared) <= 128
             and all(type(name) is str and re.fullmatch(r"[A-Za-z0-9_]+\.py", name)
                     and type(pin) is str and _SHA.fullmatch(pin) for name, pin in declared.items()),
             "bounded complete runtime declarations required")
    package = Path(__file__).absolute().parent
    _require(Path(controller.__file__).absolute() == package / "camera_reference_controller.py",
             "controller must come from the current package")
    hashes, total = {}, 0
    for name in sorted(declared):
        _require(total < MAX_RUNTIME_BYTES, "runtime byte reservation exhausted")
        raw = read_package_file(package, name, max_bytes=min(1 << 20, MAX_RUNTIME_BYTES - total))
        total += len(raw)
        _require(total <= MAX_RUNTIME_BYTES, "runtime exceeds eight MiB")
        hashes[name] = sha256(raw).hexdigest()
    _require(hashes == declared, "recorded runtime differs from current code")
    return hashes


def _prepare(raw, scene, task, *, expected=None, action_contract=None):
    text, recipe = _document(raw)
    hashes = _runtime_hashes(recipe.get("runtime_sha256"))
    _require(expected is None or hashes == expected, "envelope runtime differs from its recipe")
    action = controller.camera_reference_action_contract(scene, task) if action_contract is None else action_contract
    provider = controller.prepare_camera_reference_recipe(recipe, scene=scene, task=task,
        action_contract=action, runtime_sha256=hashes, reference_velocity_index=_trusted_velocity_index())
    return {"document_utf8": text, "sha256": sha256(raw).hexdigest(), "implementation_sha256": hashes}, provider


def _envelope(value):
    _require(type(value) is dict and set(value) == {"document_utf8", "sha256", "implementation_sha256"},
             "exact recipe envelope required")
    text, pin = value["document_utf8"], value["sha256"]
    _require(type(text) is str and 0 < len(text) <= controller.MAX_RECIPE_BYTES
             and type(pin) is str and _SHA.fullmatch(pin), "invalid bounded recipe identity")
    raw = text.encode("utf-8")
    _require(len(raw) <= controller.MAX_RECIPE_BYTES and sha256(raw).hexdigest() == pin,
             "recipe bytes differ from their identity")
    _require(type(value["implementation_sha256"]) is dict, "recorded implementation identities required")
    return raw, value["implementation_sha256"]


def recording_scene(scene, task, *, reference_controller_document=None):
    if reference_controller_document is None:
        return camera.recording_scene(scene, task)
    envelope, _ = _prepare(reference_controller_document, scene, task)
    actual, admitted_task = prepare_camera_shared_task(task, scene)
    _require(canonical_hash(actual) == canonical_hash(scene) and canonical_hash(admitted_task) == canonical_hash(task),
             "admission changed the original scene or task")
    return {"schema_version": 1, "profile": PROFILE, "engine_scene": actual, "exercise_task": task,
            "dt": actual["dt"], "objects": [], "reference_controller": envelope}


def validate_recording_scene(scene):
    _require(type(scene) is dict and set(scene) == _FIELDS
             and type(scene["schema_version"]) is int and scene["schema_version"] == 1
             and scene["profile"] == PROFILE, "distinct envelope required; policies and avatars are unsupported")
    raw, hashes = _envelope(scene["reference_controller"])
    envelope, _ = _prepare(raw, scene["engine_scene"], scene["exercise_task"], expected=hashes)
    actual, task = prepare_camera_shared_task(scene["exercise_task"], scene["engine_scene"])
    expected = {"schema_version": 1, "profile": PROFILE, "engine_scene": actual, "exercise_task": task,
                "dt": actual["dt"], "objects": [], "reference_controller": envelope}
    _require(canonical_hash(scene) == canonical_hash(expected), "recording aliases changed the original world or task")
    return expected


def prepare_recorded_reference_controller(envelope, env):
    raw, hashes = _envelope(envelope)
    _, provider = _prepare(raw, env.scene, env.task, expected=hashes, action_contract=env.action_contract())
    return partial(provider.action, env)


validate_action = camera.validate_action
prepare_snapshot_validator = camera.prepare_snapshot_validator
validate_snapshot = camera.validate_snapshot


def validate_event(event, scene):
    if not isinstance(event, dict) or event.get("controller") != "reference_proposal":
        camera.validate_event(event, scene)
        return
    _require(event.get("type") == "step"
             and event.get("reference_controller_sha256") == scene["reference_controller"]["sha256"],
             "proposal action requires the exact recorded recipe identity")
    # Reuse unchanged outcome/reset checks without relabeling the saved control.
    ordinary = {key: value for key, value in event.items() if key != "reference_controller_sha256"}
    camera.validate_event({**ordinary, "controller": "manual"}, scene)


def replay_view(scene, events):
    result = camera.replay_view(scene, events)
    task = scene["exercise_task"]
    result["reference_controller"] = {"profile": "camera-source-fit-controller-view-v1",
        "sha256": scene["reference_controller"]["sha256"], "scene_sha256": task["scene_sha256"],
        "task_sha256": canonical_hash(task), "gates_sha256": task["gates_sha256"],
        "kind": "source_fit_authored_controller", "physical_validation": "unverified",
        "scope": "Fitted joint controls; original camera, body, task and scoring remain unchanged"}
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

    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 180,
             "replay requires a positive deadline of at most 180 seconds")
    deadline = time.monotonic() + max_seconds
    # Dispatch only; the selected loader still authenticates the complete recording.
    envelope = json.loads(read_package_file(Path(directory), "scene.json", max_bytes=16 << 20))
    _require(type(envelope) is dict, "saved recording envelope must be an object")
    if envelope.get("profile") == camera.PROFILE:
        _require(time.monotonic() < deadline, "replay deadline before base dispatch")
        return camera.verify_action_replay(directory, max_seconds=deadline - time.monotonic())
    loaded = load_recording(directory, deadline=deadline)
    scene, events = loaded["scene"], loaded["events"]
    _require(scene.get("profile") == PROFILE and events, "nonempty distinct camera-reference recording required")
    _require(time.monotonic() < deadline, "replay deadline before engine construction")
    env = CameraSharedTaskEnv(scene["engine_scene"], scene["exercise_task"])
    try:
        action = prepare_recorded_reference_controller(scene["reference_controller"], env)
        _require(events[0]["snapshot"]["task_metrics"]["evaluator_sha256"] == env.evaluator_sha256,
                 "replay evaluator differs from the original task")
        verified = 0
        for event in events:
            _require(time.monotonic() < deadline, "replay deadline exceeded")
            validate_event(event, scene)
            if event["type"] in ("start", "reset"):
                env.reset(seed=event["seed"], options=event["snapshot"]["resolved_reset_options"])
            else:
                if event.get("controller") == "reference_proposal":
                    _require(action().tolist() == event["action"], "recorded proposal action differs")
                    verified += 1
                elif event.get("controller") == "authored":
                    _require(env.authored_action().tolist() == event["action"], "recorded authored action differs")
                _, reward, terminated, truncated, _ = env.step(event["action"])
                _require((reward, terminated, truncated) == (event["reward"], event["terminated"], event["truncated"]),
                         "replay outcome differs")
            _require(env.snapshot() == event["snapshot"], "full camera state/task replay differs")
        _require(time.monotonic() < deadline, "replay deadline after final state")
        task = scene["exercise_task"]
        return {"protocol": PROTOCOL, "events": len(events), "steps": sum(row["type"] == "step" for row in events),
            "exact_state_replay": True, "scene_sha256": task["scene_sha256"], "task_sha256": canonical_hash(task),
            **{key: task[key] for key in IDENTITIES}, "reference_controller_sha256": scene["reference_controller"]["sha256"],
            "reference_actions_verified": verified, "policy_sha256": None, "learned_actions_verified": None,
            "scope": "Exact simulated source-fit control decisions; not learned control or physical validation"}
    finally:
        env.close()
