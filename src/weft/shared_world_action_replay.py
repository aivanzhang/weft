"""Exact manual exploration replay; source admission and exercise scoring stay separate."""

from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import time

from weft.io import canonical_hash

PROFILE = "shared-world-exploration-recording-v1"
METADATA_KEY = "shared_world_action_replay_contract"
MAX_STEPS = 1800
MAX_EVENTS = 3601
BOUNDS = {"scene.json": 16 << 20, "recording.json": 64 << 10, "events.jsonl": 16 << 20}
SCOPE = "Exact manual simulated action/state replay only; legacy engine reward is diagnostic, not exercise success or source admission."


def _require(value, message):
    if not value:
        raise ValueError(message)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate exploration JSON key")
            result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs)
    pending, count = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _require(depth <= 48 and count <= 500_000, "Exploration JSON exceeds structural bounds")
        if isinstance(item, dict):
            pending.extend((x, depth + 1) for x in item.values())
        elif isinstance(item, list):
            pending.extend((x, depth + 1) for x in item)
        elif isinstance(item, float):
            _require(math.isfinite(item), "Nonfinite exploration JSON")
    _require(isinstance(value, dict), "Exploration JSON must be an object")
    return value


def _runtime_modules(scene):
    from weft.shared_world_lab import _lab_runtime_modules

    names = set(_lab_runtime_modules(scene)) | {"shared_world_action_replay.py", "recording.py", "io.py"}
    if "native_clock" in scene.get("human_spec", {}):
        names.add("source_clock.py")
    return names


def runtime_identity(scene):
    """Prepare in a bounded worker, never during UI rendering or request dispatch."""
    names = _runtime_modules(scene)
    package = Path(__file__).parent
    from weft.shared_world_packaging import read_package_file

    hashes = {}
    for name in sorted(names):
        raw = read_package_file(package, name, max_bytes=2 << 20)
        hashes[name] = sha256(raw).hexdigest()
    return {"runtime_sha256": hashes, "runtime_versions": {
        "python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
        **{name: version(name) for name in ("mujoco", "numpy", "gymnasium")}}}


def recording_contract(scene, *, source_run_manifest_sha256, motion_ledger_sha256, seed=0):
    from weft.shared_world import validate_shared_world

    scene = validate_shared_world(scene)
    result = {"schema_version": 1, "profile": PROFILE, "scene_sha256": canonical_hash(scene),
        "source_binding_sha256": canonical_hash(scene["source_binding"]),
        "native_interval_sha256": canonical_hash(scene["task"].get("native_interval")),
        "source_run_manifest_sha256": source_run_manifest_sha256, "motion_ledger_sha256": motion_ledger_sha256,
        "controlled_id": scene["controlled_id"], "action_names": scene["action_contract"]["names"],
        "initial_seed": seed, "reset_options": {"jitter": False}, "controller": "manual", "evaluation_status": "not_configured",
        **runtime_identity(scene)}
    return validate_contract(result, scene)


def validate_contract(contract, scene):
    keys = {"schema_version", "profile", "scene_sha256", "source_binding_sha256", "native_interval_sha256",
        "source_run_manifest_sha256", "motion_ledger_sha256", "controlled_id", "action_names", "reset_options",
        "initial_seed", "controller", "evaluation_status", "runtime_sha256", "runtime_versions"}
    _require(isinstance(contract, dict) and set(contract) == keys and type(contract["schema_version"]) is int
        and contract["schema_version"] == 1 and contract["profile"] == PROFILE, "Unsupported exploration replay contract")
    _require(scene.get("profile") in {"shared-world-v1", "shared-world-v2"}
        and scene.get("learned_policy_available") is False, "Exploration replay requires a no-policy coupled world")
    _require(contract["scene_sha256"] == canonical_hash(scene)
        and contract["source_binding_sha256"] == canonical_hash(scene["source_binding"])
        and contract["native_interval_sha256"] == canonical_hash(scene["task"].get("native_interval"))
        and contract["controlled_id"] == scene["controlled_id"]
        and contract["action_names"] == scene["action_contract"]["names"], "Exploration scene, clock or actor identity changed")
    _require(all(_hash(contract[k]) for k in ("source_run_manifest_sha256", "motion_ledger_sha256")),
        "Exploration source/ledger byte pins are required")
    _require(type(contract["initial_seed"]) is int and 0 <= contract["initial_seed"] < 2**31, "Exploration initial seed is invalid")
    _require(contract["controller"] == "manual" and contract["evaluation_status"] == "not_configured"
        and contract["reset_options"] == {"jitter": False}
        and type(contract["reset_options"].get("jitter")) is bool, "Exploration allows only explicit manual actions and fixed resets")
    names = _runtime_modules(scene)
    hashes, versions = contract["runtime_sha256"], contract["runtime_versions"]
    _require(isinstance(hashes, dict) and set(hashes) == names and all(_hash(v) for v in hashes.values()),
        "Exploration runtime closure is incomplete")
    _require(isinstance(versions, dict) and set(versions) == {"python", "system", "machine", "mujoco", "numpy", "gymnasium"}
        and all(isinstance(v, str) and 0 < len(v) <= 100 for v in versions.values()), "Exploration dependency identity is incomplete")
    return _json(json.dumps(contract, allow_nan=False).encode())


def validate_recording_event(event, contract, scene):
    """Reject implicit controllers and non-reset starts before a journal write."""
    kind = event.get("type") if isinstance(event, dict) else None
    _require(kind in {"start", "reset", "step"}, "Unsupported exploration event")
    keys = {"type", "snapshot"} | ({"action", "reward", "terminated", "truncated", "controller"} if kind == "step" else {"seed"})
    _require(set(event) == keys, "Exploration recording requires exact manual event fields")
    state = event["snapshot"]
    _require(isinstance(state, dict) and state.get("controlled_id") == contract["controlled_id"], "Exploration actor cannot change during recording")
    if kind in {"start", "reset"}:
        _require(type(event["seed"]) is int and 0 <= event["seed"] < 2**31, "Exploration seed is invalid")
        _require(state.get("steps") == 0 and state.get("time_s") == 0 and state.get("terminated") is False
            and state.get("truncated") is False, "Exploration recording requires a reset-bound initial state")
        if kind == "start":
            _require(event["seed"] == contract["initial_seed"], "Exploration initial seed changed")
    else:
        action = event["action"]
        _require(event["controller"] == "manual" and isinstance(action, list)
            and len(action) == len(contract["action_names"])
            and all(type(v) in (int, float) and math.isfinite(v) and -1 <= v <= 1 for v in action),
            "Exploration requires explicit finite manual full-vector actions")
        _require(type(event["reward"]) in (int, float) and math.isfinite(event["reward"])
            and all(type(event[k]) is bool for k in ("terminated", "truncated")), "Exploration diagnostic outcome is invalid")


def verify_action_replay(directory, *, max_seconds=60, max_steps=MAX_STEPS, expected_contract_sha256=None, deadline=None):
    """Verify exact states without interpreting legacy reward as a moving-equipment task."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 60,
        "Exploration replay requires a positive deadline of at most 60 seconds")
    _require(type(max_steps) is int and 0 <= max_steps <= MAX_STEPS, "Exploration replay step bound is invalid")
    _require(expected_contract_sha256 is None or _hash(expected_contract_sha256), "Exploration expected contract pin is invalid")
    _require(deadline is None or type(deadline) in (int, float) and math.isfinite(deadline), "Exploration deadline must be finite")
    stop = min(time.monotonic() + max_seconds, deadline) if deadline is not None else time.monotonic() + max_seconds

    def check():
        if time.monotonic() >= stop:
            raise TimeoutError("Exploration replay exceeded its cooperative wall-time bound")

    from weft.shared_world_packaging import read_package_file
    check()
    raw = {}
    for name, limit in BOUNDS.items():
        check()
        raw[name] = read_package_file(directory, name, max_bytes=limit)
    sources = {name: sha256(data).hexdigest() for name, data in raw.items()}
    scene, metadata = _json(raw["scene.json"]), _json(raw["recording.json"])
    contract = validate_contract(metadata.get(METADATA_KEY), scene)
    contract_hash = canonical_hash(contract)
    _require(expected_contract_sha256 is None or expected_contract_sha256 == contract_hash,
        "Exploration start-time contract pin changed")
    _require(metadata.get("schema_version") == 1 and type(metadata.get("schema_version")) is int
        and metadata.get("profile") == "shared-world" and metadata.get("status") == "closed"
        and metadata.get("scene_sha256") == canonical_hash(scene) and "failed_attempt" not in metadata,
        "Exploration requires a closed ordinary journal without an unreplayed failed attempt")
    _require("action_replay_contract" not in metadata, "Exploration cannot reuse a spatial replay contract")
    data = raw["events.jsonl"]
    _require(type(metadata.get("journal_bytes")) is int and metadata["journal_bytes"] == len(data)
        and data.endswith(b"\n"), "Exploration journal byte count or final event changed")
    lines = data.splitlines()
    _require(1 <= len(lines) <= MAX_EVENTS and all(len(line) <= 1 << 20 for line in lines), "Exploration journal exceeds its event bounds")
    _require(type(metadata.get("events")) is int and metadata["events"] == len(lines), "Exploration event count changed")
    events, steps, resets = [], 0, 0
    for index, line in enumerate(lines):
        check()
        event = _json(line)
        kind = event.get("type")
        _require(type(event.get("index")) is int and event["index"] == index
            and kind in ({"start"} if index == 0 else {"step", "reset"}), "Exploration event order changed")
        keys = {"index", "type", "snapshot"} | ({"action", "reward", "terminated", "truncated", "controller"}
            if kind == "step" else {"seed"})
        _require(set(event) == keys, "Exploration event fields differ from its manual protocol")
        if kind == "step":
            action = event["action"]
            _require(event["controller"] == "manual" and isinstance(action, list)
                and len(action) == len(contract["action_names"])
                and all(type(v) in (int, float) and math.isfinite(v) and -1 <= v <= 1 for v in action),
                "Exploration requires explicit finite manual full-vector actions")
            _require(type(event["reward"]) in (int, float) and math.isfinite(event["reward"])
                and all(type(event[k]) is bool for k in ("terminated", "truncated")), "Exploration outcome flags or diagnostic reward changed")
            steps += 1
        else:
            _require(type(event["seed"]) is int and 0 <= event["seed"] < 2**31, "Exploration seed is invalid")
            resets += kind == "reset"
        events.append(event)
    _require(steps <= max_steps and type(metadata.get("seed")) is int
        and metadata["seed"] == events[0]["seed"] == contract["initial_seed"], "Exploration step limit or initial seed changed")
    expected_runtime = {k: contract[k] for k in ("runtime_sha256", "runtime_versions")}
    check()
    _require(runtime_identity(scene) == expected_runtime, "Exploration runtime differs from its recording-time pins")
    from weft.shared_world import SharedWorldEnv, validate_shared_world, prepare_shared_snapshot_validator
    scene = validate_shared_world(scene)
    validate_snapshot = prepare_shared_snapshot_validator(scene)
    for event in events:
        check()
        validate_snapshot(event["snapshot"])
    check()
    env = SharedWorldEnv(scene)
    try:
        for event in events:
            check()
            if event["type"] in {"start", "reset"}:
                env.reset(seed=event["seed"], options=contract["reset_options"])
            else:
                _, reward, terminated, truncated, _ = env.step(event["action"])
                _require(canonical_hash([reward, terminated, truncated]) == canonical_hash(
                    [event["reward"], event["terminated"], event["truncated"]]),
                    f"Exploration diagnostic outcome differs at event {event['index']}")
            _require(canonical_hash(env.snapshot()) == canonical_hash(event["snapshot"]),
                f"Exploration full-state replay differs at event {event['index']}")
        check()
    finally:
        env.close()
    _require(runtime_identity(scene) == expected_runtime, "Exploration runtime changed during replay")
    for name, limit in BOUNDS.items():
        check()
        read_package_file(directory, name, max_bytes=limit, expected_sha256=sources[name])
    check()
    return {"schema_version": 1, "profile": PROFILE, "exact_state_replay": True, "events": len(events),
        "steps": steps, "resets": resets, "scene_sha256": canonical_hash(scene), "contract_sha256": contract_hash,
        "source_sha256": sources, "runtime_sha256": contract["runtime_sha256"],
        "runtime_versions": contract["runtime_versions"], "controlled_id": contract["controlled_id"],
        "expected_contract_pin_supplied": expected_contract_sha256 is not None,
        "qualification": "caller_pinned_simulation_replay" if expected_contract_sha256 else "unbound_self_consistency",
        "source_admission_verified": False, "task_success_available": False, "scope": SCOPE}
