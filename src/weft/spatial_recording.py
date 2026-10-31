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
POLICY_PROTOCOL = "spatial-rigid-exact-action-state-policy-v2"
RUNTIME_FILES = frozenset({"spatial_recording.py", "spatial_rigid.py", "recording.py", "io.py",
                           "shared_world_packaging.py", "shared_world_evidence.py"})
POLICY_RUNTIME_FILES = RUNTIME_FILES | {"spatial_learning.py", "lab.py"}
MAX_POLICY_BYTES = 32 * 1024
MAX_EVENTS, MAX_STEPS = 18001, 18000
MAX_EVENT_BYTES, MAX_JOURNAL_BYTES = 65536, 16 * 1024 * 1024
SCOPE = "Saved exact simulated action, reset, reward, flags and full-state replay; not policy or physical validation"
POLICY_SCOPE = "Saved exact simulated action/state replay and learned-decision consistency; not policy quality or physical validation"
DECISION_SCOPE = "Policy decisions regenerated from preceding saved simulated states; no physics execution or physical validation"
_UNSET = object()


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


def runtime_identity(*, controller_provenance=False):
    package = Path(__file__).resolve().parent
    hashes = {}
    for name in sorted(POLICY_RUNTIME_FILES if controller_provenance else RUNTIME_FILES):
        path = package / name
        _require(path.is_file() and path.stat().st_size <= 1024 * 1024, "Spatial runtime source exceeds its bound")
        hashes[name] = sha256(path.read_bytes()).hexdigest()
    return {"runtime_sha256": hashes, "runtime_versions": {"python": platform.python_version(),
            **{name: version(name) for name in ("mujoco", "numpy", "gymnasium")}}}


def policy_document(contract, scene):
    """Check exact policy/source binding without importing an engine or controller."""
    embedded = contract.get("learned_policy")
    if embedded is None:
        return None
    _require(isinstance(embedded, dict) and set(embedded) == {"sha256", "document_utf8"}
             and isinstance(embedded["document_utf8"], str), "Spatial policy envelope is invalid")
    raw = embedded["document_utf8"].encode("utf-8")
    _require(0 < len(raw) <= MAX_POLICY_BYTES and sha256(raw).hexdigest() == embedded["sha256"],
             "Spatial policy bytes differ from their exact identity")
    policy = _json(raw)
    _require(type(policy.get("schema_version")) is int and
             (policy["schema_version"], policy.get("profile"), policy.get("algorithm")) in {
                 (1, "spatial-policy-v1", "spatial_cem_v1"), (2, "spatial-policy-v2", "spatial_cem_v2")},
             "Unsupported spatial policy version")
    objects, actor = scene.get("objects"), policy.get("controlled_id")
    _require(isinstance(objects, list) and all(isinstance(obj, dict) for obj in objects)
             and policy.get("object_ids") == [obj.get("id") for obj in objects]
             and type(policy.get("controlled_index")) is int and 0 <= policy["controlled_index"] < len(objects)
             and objects[policy["controlled_index"]].get("id") == actor
             and objects[policy["controlled_index"]].get("role") in {"active", "passive"},
             "Spatial policy actor or object ordering differs")
    # Actor selection changes only the active/passive roles, never geometry or the task.
    original = {**scene, "controlled_id": actor, "objects": [
        {**obj, "role": "active" if obj["id"] == actor else "passive"} if obj["role"] != "static" else obj
        for obj in objects]}
    _require(policy.get("scene_sha256") == canonical_hash(original)
             and policy.get("physical_validation") == "unverified"
             and policy.get("observation_dim") == 13 * len(objects) + 4
             and all(policy.get(key) == scene["task"][key] for key in ("position_tolerance", "speed_tolerance")),
             "Spatial policy belongs to another scene or task")
    return policy


def recording_contract(scene, *, controller_provenance=False, policy_document_bytes=None):
    _require(type(controller_provenance) is bool and (controller_provenance or policy_document_bytes is None),
             "Policy bytes require explicit spatial controller provenance")
    result = {"schema_version": 2 if controller_provenance else 1,
              "protocol": POLICY_PROTOCOL if controller_provenance else PROTOCOL,
              "scene_sha256": canonical_hash(scene), "controlled_id": scene["controlled_id"],
              **(runtime_identity(controller_provenance=True) if controller_provenance else runtime_identity())}
    if controller_provenance:
        _require(policy_document_bytes is None or type(policy_document_bytes) is bytes
                 and 0 < len(policy_document_bytes) <= MAX_POLICY_BYTES, "Spatial policy document exceeds its bound")
        result["learned_policy"] = (None if policy_document_bytes is None else {
            "sha256": sha256(policy_document_bytes).hexdigest(), "document_utf8": policy_document_bytes.decode("utf-8")})
    return validate_contract(result, scene)


def validate_contract(contract, scene):
    keys = {"schema_version", "protocol", "scene_sha256", "controlled_id", "runtime_sha256", "runtime_versions"}
    version = contract.get("schema_version") if isinstance(contract, dict) else None
    if version == 2:
        keys.add("learned_policy")
    _require(isinstance(contract, dict) and set(contract) == keys and type(contract["schema_version"]) is int
             and (version, contract["protocol"]) in {(1, PROTOCOL), (2, POLICY_PROTOCOL)},
             "Spatial recording has no supported start-time replay contract")
    _require(scene.get("profile") == "spatial-rigid-v1" and contract["scene_sha256"] == canonical_hash(scene)
             and contract["controlled_id"] == scene.get("controlled_id"), "Spatial recording actor or scene identity changed")
    hashes = contract["runtime_sha256"]
    _require(isinstance(hashes, dict) and set(hashes) == (POLICY_RUNTIME_FILES if version == 2 else RUNTIME_FILES)
             and all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()),
             "Spatial recording runtime identity is incomplete")
    versions = contract["runtime_versions"]
    _require(isinstance(versions, dict) and set(versions) == {"python", "mujoco", "numpy", "gymnasium"}
             and all(isinstance(v, str) and 0 < len(v) <= 100 for v in versions.values()),
             "Spatial recording dependency identity is incomplete")
    if version == 2:
        policy_document(contract, scene)
    return contract


def validate_controller_event(event, contract, scene, *, policy=_UNSET):
    if contract["schema_version"] != 2:
        return
    if event["type"] != "step":
        _require("controller" not in event and "policy_sha256" not in event,
                 "Spatial start/reset cannot claim a controller decision")
        return
    _require(isinstance(event.get("controller"), str) and event["controller"] in {"manual", "learned"},
             "Spatial step requires explicit manual/learned mode")
    if event["controller"] == "learned":
        if policy is _UNSET:
            policy = policy_document(contract, scene)
        _require(policy is not None and policy["controlled_id"] == scene["controlled_id"]
                 and event.get("policy_sha256") == contract["learned_policy"]["sha256"],
                 "Spatial learned action requires its bound policy and trained actor")
    else:
        _require("policy_sha256" not in event, "Manual action cannot claim a learned policy identity")


def validate_decision_result(result, contract, scene, sources, *, events, steps, resets):
    keys = {"schema_version", "profile", "status", "events", "steps", "resets", "manual_actions_recorded",
            "learned_actions_verified", "policy_sha256", "contract_sha256", "source_sha256", "scope"}
    _require(isinstance(result, dict) and set(result) == keys and type(result["schema_version"]) is int
             and result["schema_version"] == 1 and result["profile"] == "spatial-policy-decisions-v1"
             and result["status"] == "passed" and contract["schema_version"] == 2,
             "Spatial decision proof is incomplete")
    for name, count in (("events", events), ("steps", steps), ("resets", resets)):
        _require(type(result[name]) is int and result[name] == count, "Spatial decision counts differ")
    embedded = contract["learned_policy"]
    learned = result["learned_actions_verified"]
    _require((learned is None if embedded is None else type(learned) is int and 0 <= learned <= steps)
             and type(result["manual_actions_recorded"]) is int
             and result["manual_actions_recorded"] == steps - (learned or 0), "Spatial decision count is invalid")
    if learned:
        _require(policy_document(contract, scene)["controlled_id"] == scene["controlled_id"],
                 "Spatial learned proof targets a different actor")
    _require(result["policy_sha256"] == (embedded["sha256"] if embedded else None)
             and result["contract_sha256"] == canonical_hash(contract) and result["source_sha256"] == sources
             and result["scope"] == DECISION_SCOPE, "Spatial decision identity or scope differs")
    return result


def validate_replay_result(proof, contract, scene, sources, *, events, steps, resets):
    validate_contract(contract, scene)
    keys = {"schema_version", "protocol", "exact_state_replay", "events", "steps", "resets", "scene_sha256",
            "controlled_id", "source_sha256", "runtime_sha256", "runtime_versions", "learned_actions_verified", "scope"}
    policy_proof = contract["schema_version"] == 2
    if policy_proof:
        keys.add("policy_decisions")
    _require(isinstance(proof, dict) and set(proof) == keys and type(proof["schema_version"]) is int
             and proof["schema_version"] == contract["schema_version"] and proof["protocol"] == contract["protocol"]
             and proof["exact_state_replay"] is True,
             "Spatial exact replay proof is incomplete")
    for name, count in (("events", events), ("steps", steps), ("resets", resets)):
        _require(type(proof[name]) is int and proof[name] == count, "Spatial replay count differs from journal")
    _require(events == 1 + steps + resets and 1 <= events <= MAX_EVENTS and 0 <= steps <= MAX_STEPS
             and 0 <= resets < MAX_EVENTS,
             "Spatial replay exceeds its event bound")
    for name in ("scene_sha256", "controlled_id", "runtime_sha256", "runtime_versions"):
        _require(proof[name] == contract[name], "Spatial replay differs from its start-time identity")
    _require(proof["source_sha256"] == sources and proof["scope"] == (POLICY_SCOPE if policy_proof else SCOPE),
             "Spatial replay differs from its exact source bytes or scope")
    if policy_proof:
        decision = validate_decision_result(proof["policy_decisions"], contract, scene, sources,
                                           events=events, steps=steps, resets=resets)
        _require(proof["learned_actions_verified"] == decision["learned_actions_verified"]
                 and type(proof["learned_actions_verified"]) is type(decision["learned_actions_verified"]),
                 "Spatial replay learned count differs from its decision proof")
    else:
        _require(proof["learned_actions_verified"] is None, "Legacy spatial replay cannot claim learned decisions")
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


def _read_journal(directory, max_steps, check):
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
    policy = policy_document(contract, scene)
    for index, event in enumerate(events):
        check()
        kind = event.get("type")
        _require(type(event.get("index")) is int and event["index"] == index and
                 kind in ({"start"} if index == 0 else {"step", "reset"}), "Spatial journal event sequence changed")
        keys = {"index", "type", "snapshot"} | ({"action", "reward", "terminated", "truncated"} if kind == "step"
                                               else {"seed", "reset_options"})
        if contract["schema_version"] == 2 and kind == "step":
            keys.add("controller")
            if event.get("controller") == "learned":
                keys.add("policy_sha256")
        _require(set(event) == keys, "Spatial replay event fields differ from its protocol")
        validate_controller_event(event, contract, scene, policy=policy)
        if kind == "step":
            _require(type(event["action"]) is int and 0 <= event["action"] <= 7, "Spatial action must be an integer 0–7")
            _require(type(event["reward"]) in (int, float) and math.isfinite(event["reward"]), "Spatial reward must be finite")
            _require(all(type(event[k]) is bool for k in ("terminated", "truncated")), "Spatial outcome flags must be booleans")
        else:
            _seed(event["seed"])
            _require(event["reset_options"] == reset_options(event["reset_options"]), "Spatial reset options must be explicit")
    _require(_seed(metadata.get("seed")) == events[0]["seed"], "Spatial start seed differs from metadata")
    return scene, contract, events, sources, bounds, steps, resets


def _runtime_for(contract):
    return runtime_identity(controller_provenance=True) if contract["schema_version"] == 2 else runtime_identity()


def observation_from_snapshot(scene, snapshot):
    """Preserve the engine's object ordering and float32 observation boundary."""
    import numpy as np

    objects = snapshot["objects"]
    _require([obj["id"] for obj in objects] == [obj["id"] for obj in scene["objects"]],
             "Spatial observation object ordering differs")
    controlled = next(obj for obj in objects if obj["id"] == scene["controlled_id"])
    state = [value for obj in objects for field in ("position", "quaternion", "velocity", "angular_velocity")
             for value in obj[field]]
    state += (np.asarray(scene["target"]) - controlled["position"]).tolist()
    state += [snapshot["steps"] / scene["task"]["max_steps"]]
    result = np.asarray(state, dtype=np.float32)
    _require(result.shape == (13 * len(objects) + 4,) and np.isfinite(result).all(),
             "Spatial policy observation is nonfinite or malformed")
    return result


def _regenerate_decisions(scene, contract, events, sources, check):
    from weft.spatial_learning import policy_action, validate_spatial_policy

    policy = policy_document(contract, scene)
    if policy is not None:
        validate_spatial_policy(policy)
    previous, learned, manual, resets = None, 0, 0, 0
    for event in events:
        check()
        state = event["snapshot"]
        if event["type"] in {"start", "reset"}:
            _require(state["steps"] == 0 and not state["terminated"] and not state["truncated"],
                     "Spatial controller state must restart at a nonterminal reset")
            resets += event["type"] == "reset"
        else:
            _require(previous is not None and not previous["terminated"] and not previous["truncated"]
                     and state["steps"] == previous["steps"] + 1,
                     "Spatial controller events cannot skip steps or cross a terminal state without reset")
            _require(all(event[key] == state[key] for key in ("terminated", "truncated")),
                     "Spatial controller outcome flags differ from the saved state")
            if event["controller"] == "learned":
                _require(policy_action(policy, observation_from_snapshot(scene, previous)) == event["action"],
                         f"Spatial learned decision differs at event {event['index']}")
                learned += 1
            else:
                manual += 1
        previous = state
    result = {"schema_version": 1, "profile": "spatial-policy-decisions-v1", "status": "passed",
              "events": len(events), "steps": learned + manual, "resets": resets,
              "manual_actions_recorded": manual, "learned_actions_verified": learned if policy is not None else None,
              "policy_sha256": contract["learned_policy"]["sha256"] if policy is not None else None,
              "contract_sha256": canonical_hash(contract), "source_sha256": sources, "scope": DECISION_SCOPE}
    return validate_decision_result(result, contract, scene, sources,
                                    events=len(events), steps=learned + manual, resets=resets)


def verify_policy_decisions(directory, *, expected_manifest_sha256=None, max_seconds=5):
    """Check selected sealed states/policy bytes without importing an environment instance or reading media."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 10,
             "Spatial decision verification requires at most ten seconds")
    stop = time.monotonic() + max_seconds

    def check():
        if time.monotonic() > stop:
            raise TimeoutError("Spatial policy decisions exceeded their cooperative time bound")

    from weft.shared_world_packaging import read_package_file

    manifest_raw = read_package_file(directory, "manifest.json", max_bytes=65536,
                                     expected_sha256=expected_manifest_sha256)
    manifest_sha = sha256(manifest_raw).hexdigest()
    manifest = _json(manifest_raw)
    scene, contract, events, sources, bounds, steps, resets = _read_journal(directory, MAX_STEPS, check)
    _require(isinstance(manifest.get("files"), dict) and all(manifest["files"].get(k) == v for k, v in sources.items()),
             "Spatial decision inputs differ from their selected seal")
    if contract["schema_version"] == 1:
        return {"status": "unavailable", "learned_actions_verified": None,
                "reason": "Legacy spatial journals do not record controller provenance"}
    _require(_runtime_for(contract) == {k: contract[k] for k in ("runtime_sha256", "runtime_versions")},
             "Spatial decision runtime differs from its recording-time pins")
    from weft.spatial_rigid import validate_scene, validate_snapshot

    validate_scene(scene)
    for event in events:
        check()
        validate_snapshot(event["snapshot"], scene)
    result = _regenerate_decisions(scene, contract, events, sources, check)
    _require(_runtime_for(contract) == {k: contract[k] for k in ("runtime_sha256", "runtime_versions")},
             "Spatial decision runtime changed during verification")
    for name, bound in {**bounds, "manifest.json": 65536}.items():
        check()
        read_package_file(directory, name, max_bytes=bound,
                          expected_sha256=manifest_sha if name == "manifest.json" else sources[name])
    check()
    return {**result, "manifest_sha256": manifest_sha, "external_manifest_pin_supplied": expected_manifest_sha256 is not None}


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

    from weft.shared_world_packaging import read_package_file

    scene, contract, events, sources, bounds, steps, resets = _read_journal(directory, max_steps, check)
    check()
    _require(_runtime_for(contract) == {k: contract[k] for k in ("runtime_sha256", "runtime_versions")},
             "Spatial replay runtime differs from its recording-time pins")
    check()
    from weft.spatial_rigid import SpatialRigidEnv, validate_scene, validate_snapshot

    validate_scene(scene)
    for event in events:
        check()
        validate_snapshot(event["snapshot"], scene)
    decisions = _regenerate_decisions(scene, contract, events, sources, check) if contract["schema_version"] == 2 else None
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
    _require(_runtime_for(contract) == {k: contract[k] for k in ("runtime_sha256", "runtime_versions")},
             "Spatial runtime changed during replay")
    for name, bound in bounds.items():
        check()
        read_package_file(directory, name, max_bytes=bound, expected_sha256=sources[name])
    check()
    return {"schema_version": contract["schema_version"], "protocol": contract["protocol"], "exact_state_replay": True, "events": len(events),
            "steps": steps, "resets": resets, "scene_sha256": canonical_hash(scene), "controlled_id": scene["controlled_id"],
            "source_sha256": sources, "runtime_sha256": contract["runtime_sha256"],
            "runtime_versions": contract["runtime_versions"],
            "learned_actions_verified": decisions["learned_actions_verified"] if decisions else None,
            **({"policy_decisions": decisions} if decisions else {}), "scope": POLICY_SCOPE if decisions else SCOPE}
