"""Exact spatial action replay for journals pinned when recording starts."""

from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import time

from weft.io import canonical_hash


PROTOCOL = "spatial-rigid-exact-action-state-v1"
RUNTIME_FILES = frozenset({"spatial_recording.py", "spatial_rigid.py", "recording.py", "io.py",
                           "shared_world_packaging.py", "shared_world_evidence.py"})
MAX_EVENTS, MAX_STEPS = 18001, 18000
MAX_EVENT_BYTES, MAX_JOURNAL_BYTES = 65536, 16 * 1024 * 1024
SCOPE = "Saved exact simulated action, reset, reward, flags and full-state replay; not policy or physical validation"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _seed(value):
    _require(type(value) is int and 0 <= value < 2**32, "Spatial replay requires a bounded integer seed")
    return value


def reset_options(value=None):
    value = {"jitter": False} if value is None else value
    _require(isinstance(value, dict) and not set(value) - {"jitter", "position_jitter", "velocity_jitter"},
             "Spatial recording reset options are unsupported")
    jitter = value.get("jitter", False)
    _require(type(jitter) is bool, "Spatial reset jitter must be a boolean")
    result = {"jitter": jitter, "position_jitter": value.get("position_jitter", .03 if jitter else 0.),
              "velocity_jitter": value.get("velocity_jitter", .15 if jitter else 0.)}
    for name, maximum in (("position_jitter", .15), ("velocity_jitter", 1.)):
        number = result[name]
        _require(type(number) in (int, float) and math.isfinite(number) and 0 <= number <= maximum,
                 "Spatial recording jitter must be finite and bounded")
        _require(jitter or number == 0, "Nonzero spatial jitter requires jitter=True")
    return result


def runtime_identity():
    package = Path(__file__).resolve().parent
    hashes = {}
    for name in sorted(RUNTIME_FILES):
        path = package / name
        _require(path.is_file() and path.stat().st_size <= 1024 * 1024, "Spatial runtime source exceeds its bound")
        hashes[name] = sha256(path.read_bytes()).hexdigest()
    return {"runtime_sha256": hashes, "runtime_versions": {"python": platform.python_version(),
            **{name: version(name) for name in ("mujoco", "numpy", "gymnasium")}}}


def recording_contract(scene):
    return {"schema_version": 1, "protocol": PROTOCOL, "scene_sha256": canonical_hash(scene),
            "controlled_id": scene["controlled_id"], **runtime_identity()}


def validate_contract(contract, scene):
    keys = {"schema_version", "protocol", "scene_sha256", "controlled_id", "runtime_sha256", "runtime_versions"}
    _require(isinstance(contract, dict) and set(contract) == keys and type(contract["schema_version"]) is int
             and contract["schema_version"] == 1 and contract["protocol"] == PROTOCOL,
             "Spatial recording has no supported start-time replay contract")
    _require(scene.get("profile") == "spatial-rigid-v1" and contract["scene_sha256"] == canonical_hash(scene)
             and contract["controlled_id"] == scene.get("controlled_id"), "Spatial recording actor or scene identity changed")
    hashes = contract["runtime_sha256"]
    _require(isinstance(hashes, dict) and set(hashes) == RUNTIME_FILES
             and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()),
             "Spatial recording runtime identity is incomplete")
    versions = contract["runtime_versions"]
    _require(isinstance(versions, dict) and set(versions) == {"python", "mujoco", "numpy", "gymnasium"}
             and all(isinstance(v, str) and 0 < len(v) <= 100 for v in versions.values()),
             "Spatial recording dependency identity is incomplete")
    return contract


def validate_replay_result(proof, contract, scene, sources, *, events, steps, resets):
    validate_contract(contract, scene)
    keys = {"schema_version", "protocol", "exact_state_replay", "events", "steps", "resets", "scene_sha256",
            "controlled_id", "source_sha256", "runtime_sha256", "runtime_versions", "learned_actions_verified", "scope"}
    _require(isinstance(proof, dict) and set(proof) == keys and type(proof["schema_version"]) is int
             and proof["schema_version"] == 1 and proof["protocol"] == PROTOCOL
             and proof["exact_state_replay"] is True and proof["learned_actions_verified"] is None,
             "Spatial exact replay proof is incomplete")
    for name, count in (("events", events), ("steps", steps), ("resets", resets)):
        _require(type(proof[name]) is int and proof[name] == count, "Spatial replay count differs from journal")
    _require(events == 1 + steps + resets and 1 <= events <= MAX_EVENTS and 0 <= steps <= MAX_STEPS
             and 0 <= resets < MAX_EVENTS,
             "Spatial replay exceeds its event bound")
    for name in ("scene_sha256", "controlled_id", "runtime_sha256", "runtime_versions"):
        _require(proof[name] == contract[name], "Spatial replay differs from its start-time identity")
    _require(proof["source_sha256"] == sources and proof["scope"] == SCOPE,
             "Spatial replay differs from its exact source bytes or scope")
    return proof


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate spatial replay JSON key")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs)
    json.dumps(value, allow_nan=False)
    _require(isinstance(value, dict), "Spatial replay JSON must be an object")
    return value


def verify_action_replay(directory, *, max_seconds=60, max_steps=MAX_STEPS, deadline=None):
    """Re-execute one bounded journal using the existing spatial environment."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 180,
             "Spatial replay requires a positive bound of at most 180 seconds")
    _require(type(max_steps) is int and 0 <= max_steps <= MAX_STEPS, "Spatial replay step bound is invalid")
    if deadline is not None:
        _require(type(deadline) in (int, float) and math.isfinite(deadline), "Spatial replay deadline must be finite")
    stop = min(time.monotonic() + max_seconds, deadline) if deadline is not None else time.monotonic() + max_seconds

    def check():
        if time.monotonic() > stop:
            raise TimeoutError("Spatial action replay exceeded its cooperative wall-time bound")

    check()
    from weft.shared_world_packaging import read_package_file

    bounds = {"scene.json": 250000, "recording.json": 65536, "events.jsonl": MAX_JOURNAL_BYTES}
    raw = {name: read_package_file(directory, name, max_bytes=bound) for name, bound in bounds.items()}
    sources = {name: sha256(value).hexdigest() for name, value in raw.items()}
    scene, metadata = _json(raw["scene.json"]), _json(raw["recording.json"])
    contract = validate_contract(metadata.get("action_replay_contract"), scene)
    _require(type(metadata.get("schema_version")) is int and metadata["schema_version"] == 1
             and metadata.get("status") == "closed" and metadata.get("scene_sha256") == canonical_hash(scene),
             "Spatial replay requires a closed, scene-bound recording")
    _require(metadata.get("journal_bytes") == len(raw["events.jsonl"]), "Spatial journal byte count changed")
    _require(raw["events.jsonl"].endswith(b"\n"), "Spatial journal is incomplete")
    lines = raw["events.jsonl"].splitlines()
    _require(1 <= len(lines) <= MAX_EVENTS and all(len(line) + 1 <= MAX_EVENT_BYTES for line in lines),
             "Spatial journal exceeds its event bounds")
    events = [_json(line) for line in lines]
    _require(type(metadata.get("events")) is int and metadata["events"] == len(events), "Spatial journal event count changed")
    steps = sum(event.get("type") == "step" for event in events)
    resets = sum(event.get("type") == "reset" for event in events)
    _require(steps <= max_steps, "Spatial replay exceeds its step bound")
    for index, event in enumerate(events):
        check()
        kind = event.get("type")
        _require(type(event.get("index")) is int and event["index"] == index and
                 kind in ({"start"} if index == 0 else {"step", "reset"}), "Spatial journal event sequence changed")
        keys = {"index", "type", "snapshot"} | ({"action", "reward", "terminated", "truncated"} if kind == "step"
                                               else {"seed", "reset_options"})
        _require(set(event) == keys, "Spatial replay event fields differ from its protocol")
        if kind == "step":
            _require(type(event["action"]) is int and 0 <= event["action"] <= 7, "Spatial action must be an integer 0–7")
            _require(type(event["reward"]) in (int, float) and math.isfinite(event["reward"]), "Spatial reward must be finite")
            _require(all(type(event[k]) is bool for k in ("terminated", "truncated")), "Spatial outcome flags must be booleans")
        else:
            _seed(event["seed"])
            _require(event["reset_options"] == reset_options(event["reset_options"]), "Spatial reset options must be explicit")
    _require(_seed(metadata.get("seed")) == events[0]["seed"], "Spatial start seed differs from metadata")
    check()
    _require(runtime_identity() == {k: contract[k] for k in ("runtime_sha256", "runtime_versions")},
             "Spatial replay runtime differs from its recording-time pins")
    check()
    from weft.spatial_rigid import SpatialRigidEnv, validate_scene, validate_snapshot

    validate_scene(scene)
    for event in events:
        check()
        validate_snapshot(event["snapshot"], scene)
    check()
    env = SpatialRigidEnv(scene)
    try:
        for event in events:
            check()
            if event["type"] in ("start", "reset"):
                env.reset(seed=event["seed"], options=event["reset_options"])
            else:
                _, reward, terminated, truncated, _ = env.step(event["action"])
                if canonical_hash([reward, terminated, truncated]) != canonical_hash(
                        [event["reward"], event["terminated"], event["truncated"]]):
                    raise ValueError(f"Spatial outcome replay differs at event {event['index']}")
            if canonical_hash(env.snapshot()) != canonical_hash(event["snapshot"]):
                raise ValueError(f"Spatial full-state replay differs at event {event['index']}")
        check()
    finally:
        env.close()
    _require(runtime_identity() == {k: contract[k] for k in ("runtime_sha256", "runtime_versions")},
             "Spatial runtime changed during replay")
    for name, bound in bounds.items():
        check()
        read_package_file(directory, name, max_bytes=bound, expected_sha256=sources[name])
    check()
    return {"schema_version": 1, "protocol": PROTOCOL, "exact_state_replay": True, "events": len(events),
            "steps": steps, "resets": resets, "scene_sha256": canonical_hash(scene), "controlled_id": scene["controlled_id"],
            "source_sha256": sources, "runtime_sha256": contract["runtime_sha256"],
            "runtime_versions": contract["runtime_versions"], "learned_actions_verified": None, "scope": SCOPE}
