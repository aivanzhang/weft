"""Durable export receipts; inspecting a receipt never replays or decodes a recording."""

from contextlib import contextmanager
from datetime import datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import re
import stat
import time
from zoneinfo import ZoneInfo

from weft.camera_shared_receipt import (
    IDENTITIES as CAMERA_IDENTITIES, IMPLEMENTATIONS as CAMERA_IMPLEMENTATIONS,
    PROFILE as CAMERA_PROFILE, PROTOCOL as CAMERA_PROTOCOL, saved_camera_bindings,
)
from weft.camera_shared_policy_receipt import (
    IMPLEMENTATIONS as CAMERA_POLICY_IMPLEMENTATIONS, PROFILE as CAMERA_POLICY_PROFILE,
    PROTOCOL as CAMERA_POLICY_PROTOCOL, saved_camera_policy_bindings,
)
from weft.camera_reference_receipt import (
    IMPLEMENTATIONS as CAMERA_REFERENCE_IMPLEMENTATIONS, PROFILE as CAMERA_REFERENCE_PROFILE,
    PROTOCOL as CAMERA_REFERENCE_PROTOCOL, reference_runtime_hashes, saved_reference_bindings,
)


PROFILE = "recording-export-evidence-v1"
SEAL_PROFILE = "recording-evidence-seal-v1"
VIEW_PROFILE = "recording-evidence-v1"
PROTOCOL = "shared-exercise-exact-action-state-task-v1"
SHARED_PROFILE = "shared-exercise-recording-v1"
FEET_PROFILE = "shared-exercise-feet-recording-v1"
FEET_PROTOCOL = "shared-exercise-feet-full-state-replay-v1"
FEET_IDENTITIES = ("base_scene_sha256", "base_task_sha256", "projection_contract_sha256",
                   "adapter_implementation_sha256", "foot_reference_sha256")
FEET_ADAPTER_FILES = ("shared_exercise_feet_task.py", "shared_exercise_feet_projection.py", "shared_exercise_feet_state.py")
FEET_IMPLEMENTATIONS = frozenset((*FEET_ADAPTER_FILES, "shared_exercise_feet_recording.py"))
# Saved inspection cannot import the lab: these versioned prerequisites match its v5 closure.
FEET_RUNTIME_REQUIRED = frozenset({
    "shared_exercise_task.py", "shared_exercise_reset.py", "shared_world.py", "shared_world_observation.py",
    "shared_world_evidence.py", "source_equivalence.py", "world_evidence.py", "observations.py", "human.py",
    "mobile_human.py", "whole_body.py", "spatial_rigid.py", "body_frame.py", "spatial_pose.py", "root_motion.py",
    "errors.py", "io.py", "shared_world_feet.py", "foot_body.py", "foot_reference.py", "whole_body_feasibility.py",
    "shared_exercise_recording.py", "recording.py", "shared_support_task.py",
}) | FEET_IMPLEMENTATIONS
LEGACY_SCENE_BYTES = 4 * 1024 * 1024
FEET_SCENE_BYTES = 16 * 1024 * 1024
SPATIAL_PROTOCOL = "spatial-rigid-exact-action-state-v1"
SPATIAL_POLICY_PROTOCOL = "spatial-rigid-exact-action-state-policy-v2"
MAX_JOURNAL_BYTES = 16 * 1024 * 1024
MAX_READ_BYTES = 40 * 1024 * 1024
MAX_SECONDS = 10
MAX_RECEIPT_BYTES = 64 * 1024
MAX_VIEW_BYTES = 32 * 1024
BASE_MEMBERS = {"scene.json", "events.jsonl", "recording.json", "video.mp4", "replay.html", "preview.gif"}
SCOPE = ("Saved export/replay receipt bound to this run and journal; inspection hashes the journal without "
         "parsing or replaying it and does not read media. Not physical validation or fresh-final evidence.")
_SESSION = re.compile(r"session-[0-9a-f]{12}\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE = re.compile(r"[a-z][a-z0-9_]*\.py\Z")
_PACKAGE = Path(__file__).resolve().parent


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(value):
    _require(isinstance(value, str) and _HASH.fullmatch(value), "Evidence requires an exact SHA-256 identity")
    return value


def _integer(value, name, maximum=18001, minimum=0):
    _require(type(value) is int and minimum <= value <= maximum, f"Invalid bounded {name}")
    return value


def _text(value, maximum=256):
    _require(isinstance(value, str) and 1 <= len(value) <= maximum
             and all(ord(char) >= 32 and ord(char) != 127 for char in value), "Invalid bounded evidence text")
    return value


def _encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate evidence JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite evidence JSON value")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    _require(isinstance(result, dict), "Evidence JSON must be an object")
    # Also reject overflowing exponents, which parse_constant does not see.
    json.dumps(result, allow_nan=False)
    return result


@contextmanager
def _directory(path):
    path = Path(path).absolute()
    _require(".." not in path.parts, "Evidence paths must not traverse parents")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            following = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        yield descriptor
    finally:
        os.close(descriptor)


class _Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.bytes_read = 0
        self.journal_bytes_hashed = 0

    def check(self):
        _require(time.monotonic() - self.started <= MAX_SECONDS, "Evidence read exceeded its time bound")
        _require(self.bytes_read <= MAX_READ_BYTES, "Evidence read exceeded its total byte bound")

    def read(self, root, name, maximum, *, keep=True):
        _require(isinstance(name, str) and "/" not in name and name not in {"", ".", ".."}, "Unsafe evidence member")
        self.check()
        with _directory(root) as directory:
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, dir_fd=directory)
            try:
                before = os.fstat(fd)
                _require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= maximum,
                         "Evidence member is not a bounded regular file")
                _require(self.bytes_read + before.st_size <= MAX_READ_BYTES, "Evidence read exceeded its total byte bound")
                digest, blocks, size = sha256(), [], 0
                while True:
                    self.check()
                    block = os.read(fd, min(64 * 1024, maximum + 1 - size))
                    if not block:
                        break
                    size += len(block)
                    self.bytes_read += len(block)
                    _require(size <= maximum, "Evidence member exceeded its byte bound")
                    digest.update(block)
                    if keep:
                        blocks.append(block)
                after = os.fstat(fd)
                identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                _require(identity(before) == identity(after) and size == before.st_size,
                         "Evidence member changed while being read")
                if name == "events.jsonl":
                    self.journal_bytes_hashed += size
                return (b"".join(blocks) if keep else None), {"sha256": digest.hexdigest(), "bytes": size}
            finally:
                os.close(fd)

    def measurements(self):
        self.check()
        return {"bytes_read": self.bytes_read, "journal_bytes_hashed": self.journal_bytes_hashed,
                "wall_seconds": round(time.monotonic() - self.started, 6), "replay_performed": False,
                "journal_parsed": False, "media_read": False}


def _hashes(value):
    _require(isinstance(value, dict) and 1 <= len(value) <= 64, "Invalid bounded runtime identity map")
    for name, digest in value.items():
        _require(isinstance(name, str) and _SOURCE.fullmatch(name), "Unsafe runtime identity name")
        _hash(digest)
    return dict(value)


def _versions(value):
    _require(isinstance(value, dict) and set(value) == {"python", "mujoco", "numpy", "gymnasium"},
             "Runtime versions require the four recorded dependencies")
    return {name: _text(item, 100) for name, item in value.items()}


def _feet_bindings(scene, task, envelope, contract, native, members):
    """Check saved identity relationships, not fresh native task/geometry qualification."""
    profiles = {"shared-world-v2": "shared-exercise-task-v1", "shared-world-v3": "shared-exercise-task-v2",
                "shared-world-v4": "shared-support-task-v1"}
    base_profile = scene.get("base_profile")
    _require(scene.get("profile") == "shared-world-v5" and isinstance(base_profile, str) and base_profile in profiles,
             "Feet receipts require an explicit v5 scene and original base profile")
    _require(type(envelope.get("schema_version")) is int and envelope["schema_version"] == 1
             and set(envelope) == {"schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects"},
             "Feet recording fields differ; policies and avatars are unsupported")
    _require(set(task) == {"schema_version", "profile", "scene_sha256", "base_scene_sha256", "base_task_sha256",
                           "base_task", "projection_profile", "projection_contract_sha256", "adapter_implementation_sha256",
                           "foot_reference_sha256", "interval", "control", "scope"}
             and type(task.get("schema_version")) is int and task["schema_version"] == 1
             and task.get("profile") == "shared-exercise-feet-task-v1"
             and task.get("projection_profile") == "shared-exercise-feet-projection-v1",
             "Feet receipt requires its distinct task/projection envelope")
    _require(contract.get("policy_sha256") is None and "policy.json" not in members
             and not any(key in document for document in (scene, contract)
                         for key in ("avatar", "avatar_asset", "avatar_sha256", "learned_policy")),
             "Feet receipts do not support policies or avatars")
    required = FEET_RUNTIME_REQUIRED | ({"person_inventory.py"} if base_profile == "shared-world-v3" else set())
    _require(required <= native.keys(), "Feet runtime lacks its complete task/recording/transitive identity closure")
    identities = {key: _hash(task.get(key)) for key in FEET_IDENTITIES}
    _require(all(contract[key] == value for key, value in identities.items() if key in contract),
             "Feet lab metadata contradicts the original task identities")
    original = task.get("base_task")
    _require(isinstance(original, dict) and type(original.get("schema_version")) is int and original["schema_version"] == 1
             and original.get("profile") == profiles[base_profile]
             and _canonical(original) == identities["base_task_sha256"]
             and original.get("scene_sha256") == identities["base_scene_sha256"],
             "Feet original task or its base-scoring scene identity differs")
    original_fields = {"schema_version", "profile", "scene_sha256", "gate_profile", "gates_sha256", "interval",
                       "control", "source_scope"}
    original_fields.update({"support_requirements", "reward_profile", "authored_addition"} if base_profile == "shared-world-v4" else
                           {"equipment_ids", "contact_requirements"})
    _require(set(original) == original_fields, "Feet original scorer task fields differ; policies and avatars are unsupported")
    _require(base_profile != "shared-world-v4" or original["authored_addition"] is True,
             "Feet support task must retain its authored scope")
    control = original.get("control")
    interval = scene.get("task")
    _require(isinstance(control, dict) and set(control) == {"root_assistance", "object_forces"}
             and control.get("root_assistance") in ("disabled", "explicit_3_axis") and control.get("object_forces") == "forbidden"
             and task.get("control") == {**control, "ankle_target_mode": "authored_initial_angle_hold"}
             and isinstance(interval, dict)
             and task.get("interval") == original.get("interval") == {
                 "start_offset_s": interval.get("start_offset_s"), "horizon_steps": interval.get("horizon_steps")},
             "Feet receipt changed original controls or interval")
    _require(_canonical({name: native[name] for name in FEET_ADAPTER_FILES}) == identities["adapter_implementation_sha256"],
             "Feet adapter identity differs from the frozen runtime triple")
    extension, body = scene.get("foot_extension"), scene.get("human_spec")
    _require(isinstance(extension, dict) and isinstance(body, dict)
             and extension.get("profile") == "shared-world-feet-v1"
             and isinstance(extension.get("reference_json"), str), "Feet source reference binding is missing")
    raw_reference = extension["reference_json"].encode("utf-8")
    _require(0 < len(raw_reference) <= 8 << 20
             and sha256(raw_reference).hexdigest() == extension.get("reference_sha256") == identities["foot_reference_sha256"],
             "Feet original reference bytes differ from their pinned identity")
    reference = _json(raw_reference)
    _require(extension.get("body_model_sha256") == _canonical(body) == reference.get("body_model_sha256")
             and _hash(extension.get("source_sha256")) == body.get("source_sha256") == reference.get("source_sha256")
             and extension.get("reference_canonical_sha256") == _canonical(reference),
             "Feet reference source/body/canonical identities differ")
    return identities, base_profile


def _bindings(run, session, budget, *, expected_manifest=None, expected_scene=None, expected_task=None,
              sealed=False):
    metadata = {}

    def read(root, name, maximum, prefix, expected=None, *, keep=True):
        raw, identity = budget.read(root, name, maximum, keep=keep)
        if expected is not None:
            _require(identity["sha256"] == _hash(expected), "Selected evidence member differs from its seal")
        metadata[prefix + name] = identity
        return _json(raw) if keep else identity

    session_prefix = f"sessions/{session.name}/"
    envelope = read(session, "scene.json", FEET_SCENE_BYTES, session_prefix)
    feet = envelope.get("profile") == FEET_PROFILE
    camera_policy = envelope.get("profile") == CAMERA_POLICY_PROFILE
    camera_reference = envelope.get('profile') == CAMERA_REFERENCE_PROFILE
    camera = envelope.get("profile") == CAMERA_PROFILE or camera_policy or camera_reference
    _require(feet or camera or metadata[session_prefix + "scene.json"]["bytes"] <= LEGACY_SCENE_BYTES,
             "Legacy recording scene exceeds its unchanged 4 MiB bound")
    closed = read(session, "recording.json", 64 * 1024, session_prefix)
    _require(type(closed.get("schema_version")) is int and closed["schema_version"] == 1 and closed.get("status") == "closed",
             "Recording must be closed before evidence is captured")
    recording_scene = _canonical(envelope)
    _require(closed.get("scene_sha256") == recording_scene, "Closed recording differs from its scene")
    events = _integer(closed.get("events"), "recorded event count", minimum=1)
    journal = read(session, "events.jsonl", MAX_JOURNAL_BYTES, session_prefix, keep=False)
    _require(_integer(closed.get("journal_bytes"), "journal byte count", MAX_JOURNAL_BYTES, 1) == journal["bytes"],
             "Closed recording differs from its journal size")
    manifest = read(run, "manifest.json", 1024 * 1024, "run/", expected_manifest)
    members = manifest.get("files")
    _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
             and isinstance(members, dict) and 1 <= len(members) <= 10000,
             "Source run requires a bounded sealed member map")

    def source(name, maximum):
        _require(name in members, "Required source metadata is not sealed")
        return read(run, name, maximum, "run/", _hash(members[name]))

    scene = source("scene.json", FEET_SCENE_BYTES if feet or camera else LEGACY_SCENE_BYTES)
    engine_scene = _canonical(scene)
    _text(scene.get("profile"), 100)
    _require(scene["profile"] != "shared-world-v5" or feet, "Feet scenes require their explicit recording profile")
    _require(scene["profile"] != "shared-world-v6" or camera, "Camera scenes require their explicit recording profile")
    native, versions, task_sha, policy_sha = {}, {}, None, None
    spatial_contract, spatial_proof = None, None
    exploration_contract, exploration_proof = None, None
    shared = envelope.get("profile") == SHARED_PROFILE or feet or camera
    foot_identities, base_profile = {}, None
    if shared:
        _require(_canonical(envelope.get("engine_scene")) == engine_scene
                 and envelope.get("dt") == scene.get("dt") and envelope.get("objects") == scene.get("objects"),
                 "Recording envelope differs from the selected engine scene")
        task = source("exercise-task.json", (128 if feet else 64) * 1024)
        _require(task.get("profile") != "shared-exercise-feet-task-v1" or feet,
                 "Feet tasks cannot use the legacy recording protocol")
        _require(task.get("profile") != "camera-shared-task-v1" or camera,
                 "Camera tasks cannot use the legacy recording protocol")
        task_sha = _canonical(task)
        _require(_canonical(envelope.get("exercise_task")) == task_sha and task.get("scene_sha256") == engine_scene,
                 "Recording envelope differs from the selected task")
        contract = source("exercise-lab.json", 128 * 1024)
        _require(contract.get("scene_sha256") == engine_scene and contract.get("task_sha256") == task_sha,
                 "Exercise lab identities differ from selected source metadata")
        native = (reference_runtime_hashes if camera_reference else _hashes)(contract.get("runtime_sha256"))
        versions = _versions(contract.get("runtime_versions"))
        _require("shared_exercise_task.py" in native, "Exercise runtime lacks its evaluator identity")
        if feet:
            foot_identities, base_profile = _feet_bindings(scene, task, envelope, contract, native, members)
            budget.check()
        if camera_reference:
            recipe = source('reference-controller.json', 1 << 20)
            custody = source('reference-attachment.json', 128 << 10)
            def trial_source(name, maximum):
                member = 'reference-trial/' + name
                _require(member in members, 'Fitted trial metadata is not sealed')
                return read(run / 'reference-trial', name, maximum, 'run/reference-trial/', _hash(members[member]))
            trial_manifest = trial_source('manifest.json', 256 << 10)
            trial_report = trial_source('report.json', 1 << 20)
            foot_identities = saved_reference_bindings(scene, task, envelope, contract, native, members,
                recipe=recipe, recipe_file_sha256=metadata['run/reference-controller.json']['sha256'],
                custody=custody, trial_manifest=trial_manifest, trial_report=trial_report)
            budget.check()
        elif camera_policy:
            policy = source("policy.json", 64 * 1024)
            admission = source("camera-policy-admission.json", 128 * 1024)
            foot_identities = saved_camera_policy_bindings(scene, task, envelope, contract, native, members,
                                                           policy=policy, admission=admission)
            budget.check()
        elif camera:
            foot_identities = saved_camera_bindings(scene, task, envelope, contract, native, members)
            budget.check()
        if scene["profile"] == "shared-world-v4":
            _require(task.get("profile") == "shared-support-task-v1" and "shared_support_task.py" in native,
                     "Static-support runtime lacks its distinct task/evaluator identity")
        embedded = envelope.get("learned_policy")
        policy_sha = contract.get("policy_sha256")
        if policy_sha is not None:
            policy_sha = _hash(policy_sha)
            policy = policy if camera_policy else source("policy.json", 64 * 1024)
            _require(metadata["run/policy.json"]["sha256"] == policy_sha and isinstance(embedded, dict)
                     and embedded.get("sha256") == policy_sha and isinstance(embedded.get("document_utf8"), str)
                     and len(embedded["document_utf8"]) <= 64 * 1024
                     and sha256(embedded["document_utf8"].encode()).hexdigest() == policy_sha,
                     "Recording embedded policy differs from the selected exact policy bytes")
            _require(policy.get("scene_sha256") == engine_scene and policy.get("task_sha256") == task_sha
                     and _versions(policy.get("runtime_versions")) == versions,
                     "Policy identities differ from the selected task/runtime")
            if not camera_policy:
                for name, digest in _hashes(policy.get("runtime_sha256")).items():
                    _require(name not in native or native[name] == digest, "Policy and lab runtime identities conflict")
                    native[name] = digest
        else:
            _require(embedded is None and "policy.json" not in members, "Unbound recording policy")
    elif "shared_world_action_replay_contract" in closed:
        from weft.shared_world_action_replay import validate_contract

        exploration_contract = validate_contract(closed["shared_world_action_replay_contract"], envelope)
        lab_contract = source("exploration-lab.json", 256 << 10)
        source("motion-evidence.json", 16 << 20)
        _require(manifest.get("profile") == "shared-world-exploration-lab-v1"
                 and manifest.get("control_profile") == "shared_world_exploration"
                 and lab_contract.get("profile") == "shared-world-exploration-lab-v1"
                 and lab_contract.get("scene_sha256") == engine_scene
                 and lab_contract.get("evaluation_status") == "not_configured"
                 and lab_contract.get("task_success_available") is False
                 and lab_contract.get("learned_policy_available") is False
                 and exploration_contract["source_run_manifest_sha256"] == metadata["run/manifest.json"]["sha256"]
                 and exploration_contract["motion_ledger_sha256"] == lab_contract.get("motion_ledger_sha256")
                 == metadata["run/motion-evidence.json"]["sha256"]
                 and recording_scene == engine_scene
                 and "policy.json" not in members and "exercise-task.json" not in members,
                 "Exploration recording differs from its unscored source/ledger contract")
        native = _hashes(exploration_contract["runtime_sha256"])
        versions = dict(exploration_contract["runtime_versions"])
        _require(lab_contract.get("replay_runtime") == {"runtime_sha256": native, "runtime_versions": versions},
                 "Exploration recording runtime differs from its admitted lab")
        exporter_hash = _hash(lab_contract.get("implementation_sha256", {}).get("shared_world_exploration.py"))
        _require(members.get("implementation/shared_world_exploration.py") == exporter_hash,
                 "Exploration exporter is not sealed to its lab")
        read(run / "implementation", "shared_world_exploration.py", 2 << 20, "run/implementation/", exporter_hash, keep=False)
        native["shared_world_exploration.py"] = exporter_hash
    elif scene["profile"] == "spatial-rigid-v1" and "action_replay_contract" in closed:
        from weft.spatial_recording import MAX_POLICY_BYTES, policy_document, validate_contract

        spatial_contract = validate_contract(closed["action_replay_contract"], envelope)
        actor = spatial_contract["controlled_id"]
        objects = scene.get("objects")
        _require(isinstance(objects, list) and sum(obj.get("id") == actor and obj.get("role") in {"active", "passive"}
                 for obj in objects) == 1, "Spatial recording must select one dynamic source object")
        selected = {**scene, "controlled_id": actor, "objects": [
            {**obj, "role": "active" if obj["id"] == actor else "passive"} if obj["role"] != "static" else obj
            for obj in objects]}
        _require(recording_scene == engine_scene or _canonical(selected) == recording_scene,
                 "Spatial recording changed source fields beyond actor selection")
        native, versions = _hashes(spatial_contract["runtime_sha256"]), _versions(spatial_contract["runtime_versions"])
        if spatial_contract["schema_version"] == 2 and spatial_contract["learned_policy"] is not None:
            policy = source("policy.json", MAX_POLICY_BYTES)
            embedded = spatial_contract["learned_policy"]
            policy_sha = _hash(embedded["sha256"])
            _require(metadata["run/policy.json"]["sha256"] == policy_sha
                     and policy == policy_document(spatial_contract, envelope)
                     and policy.get("scene_sha256") == engine_scene,
                     "Spatial recording policy differs from the selected exact source policy")
    else:
        _require(recording_scene == engine_scene, "Recording does not belong to the selected source scene")
    for expected, actual in ((expected_scene, engine_scene), (expected_task, task_sha)):
        if expected is not None:
            _require(_hash(expected) == actual, "Expected source identity differs from selected run")
    identities = {"source_run_manifest_sha256": metadata["run/manifest.json"]["sha256"],
                  "recording_scene_sha256": recording_scene, "engine_scene_sha256": engine_scene,
                  "task_sha256": task_sha, "policy_sha256": policy_sha, **foot_identities}
    if sealed:
        base = read(session, "manifest.json", 64 * 1024, session_prefix)
        files = base.get("files")
        _require(type(base.get("schema_version")) is int and base["schema_version"] == 1
                 and isinstance(files, dict) and set(files) == BASE_MEMBERS,
                 "Recording requires its complete original six-member seal")
        for name, digest in files.items():
            _hash(digest)
            if name in {"scene.json", "recording.json", "events.jsonl"}:
                _require(digest == metadata[session_prefix + name]["sha256"], "Recording member differs from its original seal")
        _require(type(base.get("video_frames")) is int and base["video_frames"] == events,
                 "Recording frame count differs from its closed event count")
        if exploration_contract is not None:
            from weft.shared_world_action_replay import validate_replay_result
            exploration_proof = base.get("shared_world_action_replay")
            _require(isinstance(exploration_proof, dict), "Exploration export lacks exact replay proof")
            validate_replay_result(exploration_proof, exploration_contract, envelope,
                {name: metadata[session_prefix + name]["sha256"] for name in ("scene.json", "recording.json", "events.jsonl")},
                events=events, steps=exploration_proof.get("steps"), resets=exploration_proof.get("resets"))
            _require(exploration_proof["expected_contract_pin_supplied"] is True,
                     "Exploration export needs its caller-pinned start contract")
        if spatial_contract is not None and base.get("action_replay") is not None:
            from weft.spatial_recording import validate_replay_result

            spatial_proof = base["action_replay"]
            _require(isinstance(spatial_proof, dict), "Invalid spatial replay proof")
            validate_replay_result(spatial_proof, spatial_contract, envelope,
                {name: metadata[session_prefix + name]["sha256"] for name in ("scene.json", "recording.json", "events.jsonl")},
                events=events, steps=spatial_proof.get("steps"), resets=spatial_proof.get("resets"))
        identities["recording_manifest_sha256"] = metadata[session_prefix + "manifest.json"]["sha256"]
    return {"identities": identities, "metadata": metadata, "native_sha256": native, "runtime_versions": versions,
            "events": events, "shared": shared, "feet": feet, "camera": camera, "camera_policy": camera_policy,
            'camera_reference': camera_reference,
            "base_profile": base_profile, "engine_profile": scene["profile"],
            "spatial_contract": spatial_contract, "spatial_proof": spatial_proof,
            "exploration_contract": exploration_contract, "exploration_proof": exploration_proof}


def _evaluator_identity(binding):
    hashes = binding["native_sha256"]
    if binding.get("camera"):
        return _canonical({name: hashes[name] for name in ("camera_shared_task.py", "shared_exercise_task.py")})
    if binding.get("feet") and binding["base_profile"] == "shared-world-v4":
        return _canonical({name: hashes[name] for name in ("shared_exercise_task.py", "shared_support_task.py")})
    if binding["shared"] and binding["engine_profile"] == "shared-world-v4":
        from weft.shared_support_task import evaluator_identity

        return evaluator_identity({name: hashes.get(name) for name in
                                   ("shared_exercise_task.py", "shared_support_task.py")})
    return hashes.get("shared_exercise_task.py")


def _runtime(binding, budget):
    names = {"recording.py", "recording_receipt.py", "io.py"}
    if binding["shared"]:
        names.update({"shared_exercise_lab.py", "shared_exercise_recording.py"})
        if binding.get("feet"):
            names.update(FEET_IMPLEMENTATIONS)
        if binding.get("camera"):
            names.update(CAMERA_IMPLEMENTATIONS)
        if binding.get("camera_policy"):
            names.update(CAMERA_POLICY_IMPLEMENTATIONS)
        if binding.get('camera_reference'):
            names.update(CAMERA_REFERENCE_IMPLEMENTATIONS)
    elif binding["engine_profile"] == "spatial-rigid-v1":
        names.add("spatial_rigid.py")
        if binding.get("spatial_contract") is not None:
            names.update(binding["native_sha256"])
    if binding.get("exploration_contract") is not None:
        names.add("shared_world_exploration.py")
    observed = {}
    for name in sorted(names | set(binding["native_sha256"])):
        _, item = budget.read(_PACKAGE, name, 1024 * 1024, keep=False)
        observed[name] = item["sha256"]
    _require(all(observed[name] == expected for name, expected in binding["native_sha256"].items()),
             "Export runtime differs from the source lab/policy pins")
    versions = {"python": platform.python_version()}
    for name in ("mujoco", "numpy", "gymnasium"):
        budget.check()
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = "unavailable"
    if binding.get("exploration_contract") is not None:
        versions.update(system=platform.system(), machine=platform.machine())
    if binding["shared"] or binding.get("spatial_contract") is not None or binding.get("exploration_contract") is not None:
        _require(versions == binding["runtime_versions"], "Export dependency versions differ from the source lab")
    return {"implementation_sha256": {name: observed[name] for name in sorted(names)},
            "native_sha256": dict(binding["native_sha256"]), "versions": versions,
            "evaluator_sha256": _evaluator_identity(binding)}


def _counts(value):
    _require(isinstance(value, dict) and set(value) == {"events", "actions", "resets"}, "Invalid recorded count summary")
    result = {name: _integer(value[name], name, minimum=1 if name == "events" else 0) for name in value}
    _require(result["events"] == 1 + result["actions"] + result["resets"], "Recorded counts contradict one start event")
    return result


def _replay(value, recorded, binding):
    keys = {"status", "protocol", "events_checked", "actions_checked", "resets_checked", "learned_actions_verified"}
    if binding.get('camera_reference'):
        keys.add('reference_actions_verified')
    _require(isinstance(value, dict) and set(value) == keys, "Invalid saved replay summary")
    if value["status"] == "not_performed":
        _require(all(value[name] is None for name in keys - {"status"}), "Unperformed replay cannot claim checked counts")
    else:
        shared = binding["shared"] and value["protocol"] == (
            CAMERA_REFERENCE_PROTOCOL if binding.get('camera_reference') else
            CAMERA_POLICY_PROTOCOL if binding.get("camera_policy") else
            CAMERA_PROTOCOL if binding.get("camera") else FEET_PROTOCOL if binding.get("feet") else PROTOCOL)
        spatial = (binding.get("spatial_proof") is not None
                   and value["protocol"] in {SPATIAL_PROTOCOL, SPATIAL_POLICY_PROTOCOL}
                   and value["protocol"] == binding["spatial_proof"]["protocol"])
        exploration = binding.get("exploration_proof") is not None and value["protocol"] == "shared-world-exploration-recording-v1"
        _require(value["status"] == "passed" and (shared or spatial or exploration), "Exact replay requires its bound supported protocol")
        for name in ("events", "actions", "resets"):
            _require(_integer(value[name + "_checked"], "replay count") == recorded[name], "Replay count differs from recording")
        learned = value["learned_actions_verified"]
        if binding.get('camera_reference'):
            _integer(value['reference_actions_verified'], 'fitted replay actions', recorded['actions'])
            _require(learned is None, 'Fitted controls cannot claim learned replay decisions')
        if binding.get("camera_policy"):
            _require(type(learned) is int, "Camera policy replay must report its checked decision count")
        if spatial:
            proof = binding["spatial_proof"]
            _require(proof["events"] == recorded["events"] and proof["steps"] == recorded["actions"]
                     and proof["resets"] == recorded["resets"], "Spatial proof counts differ from the receipt")
            _require(learned == proof["learned_actions_verified"] and type(learned) is type(proof["learned_actions_verified"]),
                     "Spatial learned count differs from its bound replay/decision proof")
        if exploration:
            _require(learned is None, "Manual exploration cannot claim learned decisions")
        if learned is not None:
            _require(binding["identities"]["policy_sha256"] is not None, "Learned replay count requires a bound policy")
            _integer(learned, "learned replay actions", recorded["actions"])
    return dict(value)


def _from_export(result, binding, export, session):
    _require(isinstance(result, dict), "Exporter did not return a verified summary")
    _require(isinstance(result.get("directory"), str) and Path(result["directory"]).absolute() == session,
             "Exporter returned a different recording directory")
    recorded = _counts({"events": result.get("events"), "actions": result.get("steps"), "resets": result.get("resets")})
    _require(result.get("manifest_sha256") == binding["identities"]["recording_manifest_sha256"]
             and type(result.get("verified_files")) is int and result["verified_files"] == 6
             and type(result.get("video_frames")) is int and result["video_frames"] == recorded["events"]
             and recorded["events"] == binding["events"], "Exporter summary differs from sealed recording")
    replay = {"status": "not_performed", "protocol": None, "events_checked": None, "actions_checked": None,
              "resets_checked": None, "learned_actions_verified": None}
    if binding.get('camera_reference'):
        replay['reference_actions_verified'] = None
    returned = result.get("action_replay")
    if binding.get("exploration_proof") is not None:
        function = getattr(export, "func", None)
        _require(returned == binding["exploration_proof"] and function is not None
                 and getattr(function, "__module__", None) == "weft.shared_world_exploration"
                 and getattr(function, "__qualname__", None) == "export_exploration_session"
                 and export.keywords == {"expected_contract_sha256": _canonical(binding["exploration_contract"])},
                 "Exploration exact replay needs its trusted exporter and start-time contract pin")
        replay = {"status": "passed", "protocol": "shared-world-exploration-recording-v1",
                  "events_checked": returned["events"], "actions_checked": returned["steps"],
                  "resets_checked": returned["resets"], "learned_actions_verified": None}
    elif binding.get("spatial_proof") is not None:
        _require(returned == binding["spatial_proof"] and getattr(export, "__module__", None) == "weft.recording"
                 and getattr(export, "__qualname__", None) == "export_recording",
                 "Spatial exact replay requires its sealed proof and trusted exporter")
        replay = {"status": "passed", "protocol": returned["protocol"], "events_checked": returned["events"],
                  "actions_checked": returned["steps"], "resets_checked": returned["resets"],
                  "learned_actions_verified": returned["learned_actions_verified"]}
    elif returned is not None:
        _require(binding["shared"] and isinstance(returned, dict) and returned.get("exact_state_replay") is True,
                 "Exporter did not return complete exact replay success")
        _require(getattr(export, "__module__", None) == "weft.shared_exercise_lab"
                 and getattr(export, "__qualname__", None) == "export_shared_exercise_session",
                 "Exact replay result requires its existing trusted exporter")
        for name in ("scene_sha256", "task_sha256", "policy_sha256",
                     *(("reference_controller_sha256",) if binding.get('camera_reference') else ()),
                     *(("policy_admission_sha256",) if binding.get("camera_policy") else ()),
                     *(CAMERA_IDENTITIES if binding.get("camera") else FEET_IDENTITIES if binding.get("feet") else ())):
            target = "engine_scene_sha256" if name == "scene_sha256" else name
            _require(returned.get(name) == binding["identities"][target], "Returned replay identities differ from recording")
        _require(type(returned.get("events")) is int and returned["events"] == recorded["events"]
                 and type(returned.get("steps")) is int and returned["steps"] == recorded["actions"],
                 "Returned replay counts differ from recording")
        protocol = (CAMERA_REFERENCE_PROTOCOL if binding.get('camera_reference') else
                    CAMERA_POLICY_PROTOCOL if binding.get("camera_policy") else CAMERA_PROTOCOL if binding.get("camera")
                    else FEET_PROTOCOL if binding.get("feet") else PROTOCOL)
        if binding.get("camera_policy") or binding.get('camera_reference'):
            _require(returned.get("protocol") == protocol, "Returned replay protocol differs from its recording")
        replay = {"status": "passed", "protocol": protocol, "events_checked": returned["events"],
                  "actions_checked": returned["steps"], "resets_checked": recorded["resets"],
                  "learned_actions_verified": returned.get("learned_actions_verified")}
        if binding.get('camera_reference'):
            replay['reference_actions_verified'] = returned.get('reference_actions_verified')
    return recorded, _replay(replay, recorded, binding)


def _atomic_member(directory, name, raw):
    _require(len(raw) <= MAX_RECEIPT_BYTES, "Receipt output exceeded its byte bound")
    pending = "." + name + ".pending"
    fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
    try:
        with os.fdopen(fd, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(fd)
    finally:
        os.close(fd)
    # Link publishes a complete file without ever replacing a competing result.
    os.link(pending, name, src_dir_fd=directory, dst_dir_fd=directory, follow_symlinks=False)
    os.unlink(pending, dir_fd=directory)
    os.fsync(directory)


def _publish(session, receipt):
    raw = _encoded(receipt)
    seal = {"schema_version": 1, "profile": SEAL_PROFILE, "status": "completed",
            "source_run_manifest_sha256": receipt["identities"]["source_run_manifest_sha256"],
            "recording_manifest_sha256": receipt["identities"]["recording_manifest_sha256"],
            "files": {"receipt.json": sha256(raw).hexdigest()}}
    seal_raw = _encoded(seal)
    _require(len(raw) <= MAX_RECEIPT_BYTES and len(seal_raw) <= MAX_RECEIPT_BYTES, "Receipt output exceeded its byte bound")
    with _directory(session) as root:
        os.mkdir("evidence", mode=0o700, dir_fd=root)
        fd = os.open("evidence", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=root)
        try:
            _atomic_member(fd, "receipt.json", raw)
            _atomic_member(fd, "manifest.json", seal_raw)
            os.fsync(root)
        finally:
            os.close(fd)
    return {"receipt.json": {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)},
            "manifest.json": {"sha256": sha256(seal_raw).hexdigest(), "bytes": len(seal_raw)}}


def _unavailable(session, reason, code="unavailable"):
    return {"schema_version": 1, "profile": VIEW_PROFILE, "status": "unavailable", "session": session,
            "reason": reason, "code": code, "scope": SCOPE}


def _failure(session, error):
    if isinstance(error, FileNotFoundError):
        return _unavailable(session, "No complete bound recording receipt is available", "missing_receipt_or_member")
    if isinstance(error, OSError):
        return _unavailable(session, "A required evidence file could not be safely read or exclusively published", "unsafe_or_unreadable_member")
    clean = "".join(char if ord(char) >= 32 and ord(char) != 127 else " " for char in str(error))
    return _unavailable(session, " ".join(clean.split())[:300] or "Recording evidence validation failed", "invalid_evidence")


def _view(session, receipt, metadata, seals, budget):
    prefix = f"sessions/{session}/evidence/"
    pointers = {**metadata, **{prefix + name: value for name, value in seals.items()}}
    identities = {**receipt["identities"], "receipt_manifest_sha256": seals["manifest.json"]["sha256"],
                  "receipt_sha256": seals["receipt.json"]["sha256"]}
    result = {"schema_version": 1, "profile": VIEW_PROFILE, "status": "available", "session": session,
              "recorded": receipt["recorded"], "replay": receipt["replay"], "identities": identities,
              "verified_metadata": [{"source_name": name, "path": name, **item} for name, item in sorted(pointers.items())],
              "partition": "exposed_development", "physical_validation": "unverified", "scope": SCOPE,
              "inspection": budget.measurements()}
    _require(len(_encoded(result)) <= MAX_VIEW_BYTES, "Recording evidence view exceeds its byte bound")
    return result


def export_recording_with_receipt(directory, *, export, run_directory, expected_run_manifest_sha256=None):
    """Call the existing exporter once; receipt failure never invalidates a successful base export."""
    session, run = Path(directory).absolute(), Path(run_directory).absolute()
    name, before, problem, old = session.name, None, None, False
    try:
        _require(_SESSION.fullmatch(name) is not None, "Invalid recording session name")
        _require(session.parent.name == "sessions" and session.parent.parent == run.parent and run.name == "run",
                 "Recording and source run must belong to one explicit lab")
        with _directory(session) as root:
            try:
                os.stat("manifest.json", dir_fd=root, follow_symlinks=False)
                old = True
            except FileNotFoundError:
                pass
        if not old:
            budget = _Budget()
            before = _bindings(run, session, budget, expected_manifest=expected_run_manifest_sha256)
            runtime_before = _runtime(before, budget)
            budget.check()
    except Exception as error:
        problem = error
    result = export(directory)
    if old:
        return {**result, "recording_evidence": _unavailable(name, "Existing exports are not retroactively backfilled", "legacy_not_backfilled")}
    try:
        if problem is not None:
            raise problem
        budget = _Budget()
        after = _bindings(run, session, budget, expected_manifest=expected_run_manifest_sha256, sealed=True)
        comparison = {**after, "identities": {key: value for key, value in after["identities"].items()
                                              if key != "recording_manifest_sha256"},
                      "metadata": {key: value for key, value in after["metadata"].items()
                                   if key != f"sessions/{name}/manifest.json"}, "spatial_proof": None, "exploration_proof": None}
        _require(comparison == before, "Source or recording changed during export")
        runtime_after = _runtime(after, budget)
        _require(runtime_before == runtime_after, "Export runtime changed while generating the recording")
        recorded, replay = _from_export(result, after, export, session)
        exporter = getattr(export, "func", export) if after.get("exploration_contract") is not None else export
        receipt = {"schema_version": 1, "profile": PROFILE, "status": "completed", "session": name,
                   "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
                   "identities": after["identities"], "recorded": recorded, "replay": replay,
                   "metadata": after["metadata"], "runtime": runtime_after,
                   "exporter": {"module": _text(getattr(exporter, "__module__", "unknown")),
                                "name": _text(getattr(exporter, "__qualname__", "unknown"))},
                   "partition": "exposed_development", "physical_validation": "unverified", "scope": SCOPE}
        budget.check()
        seals = _publish(session, receipt)
        evidence = _view(name, receipt, after["metadata"], seals, budget)
    except Exception as error:
        evidence = _failure(name if _SESSION.fullmatch(name) else None, error)
    return {**result, "recording_evidence": evidence}


def load_recording_evidence(lab_root, session_name, *, expected_run_manifest_sha256=None,
                            scene_sha256=None, task_sha256=None):
    """Inspect one selected sealed receipt with bounded hashes, without native or media imports."""
    name = session_name if isinstance(session_name, str) and _SESSION.fullmatch(session_name) else None
    try:
        _require(name is not None, "Invalid recording session name")
        budget = _Budget()
        lab = Path(lab_root).absolute()
        session, run = lab / "sessions" / name, lab / "run"
        seal_raw, seal_identity = budget.read(session / "evidence", "manifest.json", MAX_RECEIPT_BYTES)
        seal = _json(seal_raw)
        _require(type(seal.get("schema_version")) is int and seal["schema_version"] == 1
                 and seal.get("profile") == SEAL_PROFILE and seal.get("status") == "completed"
                 and isinstance(seal.get("files"), dict) and set(seal["files"]) == {"receipt.json"},
                 "Recording evidence seal is incomplete or unsupported")
        raw, identity = budget.read(session / "evidence", "receipt.json", MAX_RECEIPT_BYTES)
        _require(identity["sha256"] == _hash(seal["files"]["receipt.json"]), "Recording receipt differs from its seal")
        receipt = _json(raw)
        _require(type(receipt.get("schema_version")) is int and receipt["schema_version"] == 1
                 and receipt.get("profile") == PROFILE and receipt.get("status") == "completed"
                 and receipt.get("session") == name and receipt.get("partition") == "exposed_development"
                 and receipt.get("physical_validation") == "unverified", "Recording receipt identity or scope is invalid")
        bound = _bindings(run, session, budget, expected_manifest=expected_run_manifest_sha256,
                          expected_scene=scene_sha256, expected_task=task_sha256, sealed=True)
        _require(receipt.get("identities") == bound["identities"] and receipt.get("metadata") == bound["metadata"],
                 "Recording receipt does not match the selected run and recording bytes")
        for key in ("source_run_manifest_sha256", "recording_manifest_sha256"):
            _require(seal.get(key) == bound["identities"][key], "Recording evidence seal binding differs from its receipt")
        recorded = _counts(receipt.get("recorded"))
        _require(recorded["events"] == bound["events"], "Saved receipt count differs from the closed recording")
        _replay(receipt.get("replay"), recorded, bound)
        runtime = receipt.get("runtime")
        _require(isinstance(runtime, dict) and runtime.get("native_sha256") == bound["native_sha256"],
                 "Saved replay runtime differs from the sealed lab/policy")
        implementations = _hashes(runtime.get("implementation_sha256"))
        required = {"recording.py", "recording_receipt.py", "io.py"}
        if bound["shared"]:
            required.update({"shared_exercise_lab.py", "shared_exercise_recording.py"})
            if bound.get("feet"):
                required.update(FEET_IMPLEMENTATIONS)
            if bound.get("camera"):
                required.update(CAMERA_IMPLEMENTATIONS)
            if bound.get("camera_policy"):
                required.update(CAMERA_POLICY_IMPLEMENTATIONS)
            if bound.get('camera_reference'):
                required.update(CAMERA_REFERENCE_IMPLEMENTATIONS)
        elif bound["engine_profile"] == "spatial-rigid-v1":
            required.add("spatial_rigid.py")
            if bound.get("spatial_contract") is not None:
                required.update(bound["native_sha256"])
        if bound.get("exploration_contract") is not None:
            required.add("shared_world_exploration.py")
        _require(set(implementations) == required, "Saved export implementation identity is incomplete")
        if bound.get("feet") or bound.get("camera") or bound.get("exploration_contract") is not None:
            _require(all(implementations[name] == bound["native_sha256"][name]
                         for name in implementations.keys() & bound["native_sha256"].keys()),
                     "Saved implementation identities differ from the frozen lab runtime")
        if bound.get("exploration_contract") is not None:
            versions = runtime.get("versions")
            _require(versions == bound["runtime_versions"], "Saved exploration dependency versions differ")
        else:
            versions = _versions(runtime.get("versions"))
        _require(runtime.get("evaluator_sha256") == _evaluator_identity(bound),
                 "Saved evaluator identity differs from source runtime")
        if bound["shared"] or bound.get("spatial_contract") is not None or bound.get("exploration_contract") is not None:
            _require(versions == bound["runtime_versions"], "Saved dependency versions differ from the sealed lab")
        if receipt["replay"]["status"] == "passed":
            expected_exporter = ({"module": "weft.shared_world_exploration", "name": "export_exploration_session"}
                                 if bound.get("exploration_contract") is not None else {"module": "weft.shared_exercise_lab", "name": "export_shared_exercise_session"}
                                 if bound["shared"] else {"module": "weft.recording", "name": "export_recording"})
            _require(receipt.get("exporter") == expected_exporter,
                     "Saved exact replay requires its existing trusted exporter")
        seals = {"manifest.json": seal_identity, "receipt.json": identity}
        return _view(name, receipt, bound["metadata"], seals, budget)
    except Exception as error:
        return _failure(name, error)
