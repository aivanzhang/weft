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


PROFILE = "recording-export-evidence-v1"
SEAL_PROFILE = "recording-evidence-seal-v1"
VIEW_PROFILE = "recording-evidence-v1"
PROTOCOL = "shared-exercise-exact-action-state-task-v1"
SHARED_PROFILE = "shared-exercise-recording-v1"
SPATIAL_PROTOCOL = "spatial-rigid-exact-action-state-v1"
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
    envelope = read(session, "scene.json", 4 * 1024 * 1024, session_prefix)
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

    scene = source("scene.json", 4 * 1024 * 1024)
    engine_scene = _canonical(scene)
    _text(scene.get("profile"), 100)
    native, versions, task_sha, policy_sha = {}, {}, None, None
    spatial_contract, spatial_proof = None, None
    shared = envelope.get("profile") == SHARED_PROFILE
    if shared:
        _require(_canonical(envelope.get("engine_scene")) == engine_scene
                 and envelope.get("dt") == scene.get("dt") and envelope.get("objects") == scene.get("objects"),
                 "Recording envelope differs from the selected engine scene")
        task = source("exercise-task.json", 64 * 1024)
        task_sha = _canonical(task)
        _require(_canonical(envelope.get("exercise_task")) == task_sha and task.get("scene_sha256") == engine_scene,
                 "Recording envelope differs from the selected task")
        contract = source("exercise-lab.json", 128 * 1024)
        _require(contract.get("scene_sha256") == engine_scene and contract.get("task_sha256") == task_sha,
                 "Exercise lab identities differ from selected source metadata")
        native = _hashes(contract.get("runtime_sha256"))
        versions = _versions(contract.get("runtime_versions"))
        _require("shared_exercise_task.py" in native, "Exercise runtime lacks its evaluator identity")
        embedded = envelope.get("learned_policy")
        policy_sha = contract.get("policy_sha256")
        if policy_sha is not None:
            policy_sha = _hash(policy_sha)
            policy = source("policy.json", 64 * 1024)
            _require(metadata["run/policy.json"]["sha256"] == policy_sha and isinstance(embedded, dict)
                     and embedded.get("sha256") == policy_sha and isinstance(embedded.get("document_utf8"), str)
                     and len(embedded["document_utf8"]) <= 64 * 1024
                     and sha256(embedded["document_utf8"].encode()).hexdigest() == policy_sha,
                     "Recording embedded policy differs from the selected exact policy bytes")
            _require(policy.get("scene_sha256") == engine_scene and policy.get("task_sha256") == task_sha
                     and _versions(policy.get("runtime_versions")) == versions,
                     "Policy identities differ from the selected task/runtime")
            for name, digest in _hashes(policy.get("runtime_sha256")).items():
                _require(name not in native or native[name] == digest, "Policy and lab runtime identities conflict")
                native[name] = digest
        else:
            _require(embedded is None and "policy.json" not in members, "Unbound recording policy")
    elif scene["profile"] == "spatial-rigid-v1" and "action_replay_contract" in closed:
        from weft.spatial_recording import validate_contract

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
    else:
        _require(recording_scene == engine_scene, "Recording does not belong to the selected source scene")
    for expected, actual in ((expected_scene, engine_scene), (expected_task, task_sha)):
        if expected is not None:
            _require(_hash(expected) == actual, "Expected source identity differs from selected run")
    identities = {"source_run_manifest_sha256": metadata["run/manifest.json"]["sha256"],
                  "recording_scene_sha256": recording_scene, "engine_scene_sha256": engine_scene,
                  "task_sha256": task_sha, "policy_sha256": policy_sha}
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
        if spatial_contract is not None and base.get("action_replay") is not None:
            from weft.spatial_recording import validate_replay_result

            spatial_proof = base["action_replay"]
            _require(isinstance(spatial_proof, dict), "Invalid spatial replay proof")
            validate_replay_result(spatial_proof, spatial_contract, envelope,
                {name: metadata[session_prefix + name]["sha256"] for name in ("scene.json", "recording.json", "events.jsonl")},
                events=events, steps=spatial_proof.get("steps"), resets=spatial_proof.get("resets"))
        identities["recording_manifest_sha256"] = metadata[session_prefix + "manifest.json"]["sha256"]
    return {"identities": identities, "metadata": metadata, "native_sha256": native, "runtime_versions": versions,
            "events": events, "shared": shared, "engine_profile": scene["profile"],
            "spatial_contract": spatial_contract, "spatial_proof": spatial_proof}


def _runtime(binding, budget):
    names = {"recording.py", "recording_receipt.py", "io.py"}
    if binding["shared"]:
        names.update({"shared_exercise_lab.py", "shared_exercise_recording.py"})
    elif binding["engine_profile"] == "spatial-rigid-v1":
        names.add("spatial_rigid.py")
        if binding.get("spatial_contract") is not None:
            names.update(binding["native_sha256"])
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
    if binding["shared"] or binding.get("spatial_contract") is not None:
        _require(versions == binding["runtime_versions"], "Export dependency versions differ from the source lab")
    return {"implementation_sha256": {name: observed[name] for name in sorted(names)},
            "native_sha256": dict(binding["native_sha256"]), "versions": versions,
            "evaluator_sha256": binding["native_sha256"].get("shared_exercise_task.py")}


def _counts(value):
    _require(isinstance(value, dict) and set(value) == {"events", "actions", "resets"}, "Invalid recorded count summary")
    result = {name: _integer(value[name], name, minimum=1 if name == "events" else 0) for name in value}
    _require(result["events"] == 1 + result["actions"] + result["resets"], "Recorded counts contradict one start event")
    return result


def _replay(value, recorded, binding):
    keys = {"status", "protocol", "events_checked", "actions_checked", "resets_checked", "learned_actions_verified"}
    _require(isinstance(value, dict) and set(value) == keys, "Invalid saved replay summary")
    if value["status"] == "not_performed":
        _require(all(value[name] is None for name in keys - {"status"}), "Unperformed replay cannot claim checked counts")
    else:
        shared = binding["shared"] and value["protocol"] == PROTOCOL
        spatial = binding.get("spatial_proof") is not None and value["protocol"] == SPATIAL_PROTOCOL
        _require(value["status"] == "passed" and (shared or spatial), "Exact replay requires its bound supported protocol")
        for name in ("events", "actions", "resets"):
            _require(_integer(value[name + "_checked"], "replay count") == recorded[name], "Replay count differs from recording")
        learned = value["learned_actions_verified"]
        if spatial:
            _require(learned is None, "Spatial action replay does not verify policy decisions")
            proof = binding["spatial_proof"]
            _require(proof["events"] == recorded["events"] and proof["steps"] == recorded["actions"]
                     and proof["resets"] == recorded["resets"], "Spatial proof counts differ from the receipt")
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
    returned = result.get("action_replay")
    if binding.get("spatial_proof") is not None:
        _require(returned == binding["spatial_proof"] and getattr(export, "__module__", None) == "weft.recording"
                 and getattr(export, "__qualname__", None) == "export_recording",
                 "Spatial exact replay requires its sealed proof and trusted exporter")
        replay = {"status": "passed", "protocol": SPATIAL_PROTOCOL, "events_checked": returned["events"],
                  "actions_checked": returned["steps"], "resets_checked": returned["resets"], "learned_actions_verified": None}
    elif returned is not None:
        _require(binding["shared"] and isinstance(returned, dict) and returned.get("exact_state_replay") is True,
                 "Exporter did not return complete exact replay success")
        _require(getattr(export, "__module__", None) == "weft.shared_exercise_lab"
                 and getattr(export, "__qualname__", None) == "export_shared_exercise_session",
                 "Exact replay result requires its existing trusted exporter")
        for name in ("scene_sha256", "task_sha256", "policy_sha256"):
            target = "engine_scene_sha256" if name == "scene_sha256" else name
            _require(returned.get(name) == binding["identities"][target], "Returned replay identities differ from recording")
        _require(type(returned.get("events")) is int and returned["events"] == recorded["events"]
                 and type(returned.get("steps")) is int and returned["steps"] == recorded["actions"],
                 "Returned replay counts differ from recording")
        replay = {"status": "passed", "protocol": PROTOCOL, "events_checked": returned["events"],
                  "actions_checked": returned["steps"], "resets_checked": recorded["resets"],
                  "learned_actions_verified": returned.get("learned_actions_verified")}
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
                                   if key != f"sessions/{name}/manifest.json"}, "spatial_proof": None}
        _require(comparison == before, "Source or recording changed during export")
        runtime_after = _runtime(after, budget)
        _require(runtime_before == runtime_after, "Export runtime changed while generating the recording")
        recorded, replay = _from_export(result, after, export, session)
        receipt = {"schema_version": 1, "profile": PROFILE, "status": "completed", "session": name,
                   "created_at": datetime.now(ZoneInfo("America/Los_Angeles")).isoformat(),
                   "identities": after["identities"], "recorded": recorded, "replay": replay,
                   "metadata": after["metadata"], "runtime": runtime_after,
                   "exporter": {"module": _text(getattr(export, "__module__", "unknown")),
                                "name": _text(getattr(export, "__qualname__", "unknown"))},
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
        elif bound["engine_profile"] == "spatial-rigid-v1":
            required.add("spatial_rigid.py")
            if bound.get("spatial_contract") is not None:
                required.update(bound["native_sha256"])
        _require(set(implementations) == required, "Saved export implementation identity is incomplete")
        versions = _versions(runtime.get("versions"))
        _require(runtime.get("evaluator_sha256") == bound["native_sha256"].get("shared_exercise_task.py"),
                 "Saved evaluator identity differs from source runtime")
        if bound["shared"] or bound.get("spatial_contract") is not None:
            _require(versions == bound["runtime_versions"], "Saved dependency versions differ from the sealed lab")
        if receipt["replay"]["status"] == "passed":
            expected_exporter = ({"module": "weft.shared_exercise_lab", "name": "export_shared_exercise_session"}
                                 if bound["shared"] else {"module": "weft.recording", "name": "export_recording"})
            _require(receipt.get("exporter") == expected_exporter,
                     "Saved exact replay requires its existing trusted exporter")
        seals = {"manifest.json": seal_identity, "receipt.json": identity}
        return _view(name, receipt, bound["metadata"], seals, budget)
    except Exception as error:
        return _failure(name, error)
