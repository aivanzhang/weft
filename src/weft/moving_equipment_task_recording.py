"""Task-bound journals and exact replay, composed with the unchanged media recorder."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from weft.io import canonical_hash
from weft.shared_world_action_replay import _hash, _json, _require, _runtime_modules
from weft.shared_world_packaging import read_package_file, write_package_files

PROFILE = "moving-equipment-task-recording-v1"
METADATA_KEY = "moving_equipment_task_recording"
RECEIPT = "task-export-receipt.json"
MAX_STEPS = 1800
MAX_EVENTS = 3601
MAX_JOURNAL_BYTES = 16 << 20
BOUNDS = {"scene.json": 16 << 20, "recording.json": 64 << 10, "events.jsonl": MAX_JOURNAL_BYTES,
          "placement-task.json": 64 << 10, "moving-proposals.json": 16 << 20,
          "task-recording-contract.json": 64 << 10}
MEDIA = {"video.mp4", "preview.gif", "replay.html"}
MAX_EXPORT_BYTES = 160 << 20
SCOPE = "Exact replay of an authored simulated placement task; not recovered intent, physical accuracy, grasp or independent source admission."


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _names(scene):
    return _runtime_modules(scene) | {"moving_equipment_task.py", "moving_equipment_task_lab.py",
        "moving_equipment_task_recording.py", "shared_exercise_task.py"}


def runtime_identity(scene):
    """Capture once off the request thread; replay checks the same closure again."""
    from weft.shared_world_action_replay import runtime_identity as engine_identity
    result = engine_identity(scene)
    for name in sorted(_names(scene) - result["runtime_sha256"].keys()):
        result["runtime_sha256"][name] = sha256(read_package_file(Path(__file__).parent, name, max_bytes=2 << 20)).hexdigest()
    return result


def _runtime_shape(value, scene):
    _require(isinstance(value, dict) and set(value) == {"runtime_sha256", "runtime_versions"}, "Task runtime fields differ")
    hashes, versions = value["runtime_sha256"], value["runtime_versions"]
    _require(isinstance(hashes, dict) and set(hashes) == _names(scene) and all(_hash(v) for v in hashes.values()),
             "Task replay runtime closure is incomplete")
    _require(isinstance(versions, dict) and set(versions) == {"python", "system", "machine", "mujoco", "numpy", "gymnasium"}
        and all(isinstance(v, str) and 0 < len(v) <= 100 for v in versions.values()), "Task dependency pins differ")


def recording_contract(scene, task, bundle, *, task_lab_manifest_sha256, parent_manifest_sha256,
                       task_lab_contract_sha256, seed=0, runtime_snapshot=None):
    from weft.moving_equipment_task import validate_moving_equipment_task
    validate_moving_equipment_task(task, scene, bundle)
    result = {"schema_version": 1, "profile": PROFILE, "scene_sha256": canonical_hash(scene),
        "task_sha256": canonical_hash(task), "bundle_sha256": canonical_hash(bundle),
        "family_sha256": task["binding"]["family_sha256"], "selection_sha256": task["binding"]["selection_sha256"],
        "source_binding_sha256": canonical_hash(scene["source_binding"]),
        "native_interval_sha256": canonical_hash(scene["task"].get("native_interval")),
        "scorer_sha256": deepcopy(task["scorer_sha256"]), "task_lab_manifest_sha256": task_lab_manifest_sha256,
        "parent_manifest_sha256": parent_manifest_sha256, "task_lab_contract_sha256": task_lab_contract_sha256,
        "controlled_id": scene["controlled_id"], "action_names": scene["action_contract"]["names"],
        "initial_seed": seed, "reset_options": {"jitter": False}, "controller": "manual", "scope": SCOPE,
        **(runtime_identity(scene) if runtime_snapshot is None else runtime_snapshot)}
    return validate_contract(result, scene, task, bundle)


def validate_contract(contract, scene, task, bundle):
    """Pure identity checks; source qualification belongs to the sealed task-child loader."""
    keys = {"schema_version", "profile", "scene_sha256", "task_sha256", "bundle_sha256", "family_sha256",
        "selection_sha256", "source_binding_sha256", "native_interval_sha256", "scorer_sha256",
        "task_lab_manifest_sha256", "parent_manifest_sha256", "task_lab_contract_sha256", "controlled_id",
        "action_names", "initial_seed", "reset_options", "controller", "scope", "runtime_sha256", "runtime_versions"}
    _require(isinstance(contract, dict) and set(contract) == keys and type(contract["schema_version"]) is int
        and contract["schema_version"] == 1 and contract["profile"] == PROFILE, "Unsupported task recording contract")
    joins = {"scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        "bundle_sha256": canonical_hash(bundle), "family_sha256": canonical_hash(bundle.get("depth_family")),
        "selection_sha256": canonical_hash(bundle.get("depth_selection")),
        "source_binding_sha256": canonical_hash(scene.get("source_binding")),
        "native_interval_sha256": canonical_hash(scene.get("task", {}).get("native_interval"))}
    _require(all(contract[k] == v for k, v in joins.items()), "Task recording scene, task, family or clock identity changed")
    _require(all(_hash(contract[k]) for k in ("task_lab_manifest_sha256", "parent_manifest_sha256", "task_lab_contract_sha256")),
             "Task recording requires exact child and parent pins")
    _require(contract["scorer_sha256"] == task.get("scorer_sha256") and isinstance(contract["scorer_sha256"], dict)
        and all(_hash(v) for v in contract["scorer_sha256"].values()), "Task scorer identity changed")
    _runtime_shape({k: contract[k] for k in ("runtime_sha256", "runtime_versions")}, scene)
    _require(all(contract["runtime_sha256"].get(k) == v for k, v in contract["scorer_sha256"].items()),
             "Task scorer and replay runtime disagree")
    _require(contract["controlled_id"] == scene.get("controlled_id") and contract["action_names"] == scene.get("action_contract", {}).get("names")
        and contract["controller"] == "manual" and contract["scope"] == SCOPE
        and contract["reset_options"] == {"jitter": False} and type(contract["reset_options"].get("jitter")) is bool,
        "Task recording actor, controls or reset scope differs")
    _require(type(contract["initial_seed"]) is int and 0 <= contract["initial_seed"] < 2**32, "Task recording seed is invalid")
    return _json(_bytes(contract))


def validate_event(event, contract):
    from weft.moving_equipment_task_lab import compose_task_snapshot
    kind = event.get("type") if isinstance(event, dict) else None
    keys = {"type", "snapshot"} | ({"action", "reward", "terminated", "truncated", "controller"} if kind == "step" else {"seed"})
    _require(kind in {"start", "reset", "step"} and set(event) == keys, "Task recording event fields differ")
    state = event["snapshot"]
    _require(isinstance(state, dict) and isinstance(state.get("task"), dict) and isinstance(state.get("engine_diagnostics"), dict),
             "Task recording requires separate task and engine snapshots")
    _require("simulation_failure" not in state, "An unreplayed failed action cannot become a successful journal event")
    _require(canonical_hash(state) == canonical_hash(compose_task_snapshot(state["engine_diagnostics"], state["task"]))
        and state["task"].get("task_sha256") == contract["task_sha256"]
        and state.get("controlled_id") == contract["controlled_id"], "Task snapshot identity or projection changed")
    if kind in {"start", "reset"}:
        _require(type(event["seed"]) is int and 0 <= event["seed"] < 2**32, "Task reset seed is invalid")
        _require(state.get("steps") == 0 and state.get("time_s") == 0 and state.get("terminated") is False
            and state.get("truncated") is False, "Task recording must begin at an exact reset")
        _require(kind != "start" or event["seed"] == contract["initial_seed"], "Task initial seed changed")
    else:
        action = event["action"]
        _require(event["controller"] == "manual" and isinstance(action, list) and len(action) == len(contract["action_names"])
            and all(type(x) in (int, float) and math.isfinite(x) and -1 <= x <= 1 for x in action),
            "Task recording requires a finite manual full-vector action")
        _require(type(event["reward"]) in (int, float) and math.isfinite(event["reward"])
            and all(type(event[k]) is bool for k in ("terminated", "truncated")), "Task reward or flags are invalid")


class TaskRecorder:
    """Use the original durable journal and size checks; add only task-bound sidecars."""

    def __init__(self, root, scene, task, bundle, contract):
        from weft.moving_equipment_task import validate_moving_equipment_task
        from weft.recording import Recorder
        validate_moving_equipment_task(task, scene, bundle)
        self._contract = validate_contract(contract, scene, task, bundle)
        self._sidecars = {"placement-task.json": _bytes(task), "moving-proposals.json": _bytes(bundle),
                          "task-recording-contract.json": _bytes(self._contract)}
        _require(all(len(raw) <= BOUNDS[name] for name, raw in self._sidecars.items()), "Task recording sidecar bound exceeded")
        self._recorder = Recorder(root, scene, profile="shared-world")
        self.root, self._closed, self._failure = self._recorder.root, False, None

    @property
    def contract(self): return deepcopy(self._contract)

    def start(self, snapshot, seed=0):
        validate_event({"type": "start", "snapshot": snapshot, "seed": seed}, self._contract)
        result = self._recorder.start(snapshot, seed=seed)
        try:
            write_package_files(self.root, self._sidecars)
        except BaseException:
            self._recorder.close()
            raise
        return result

    def check_capacity(self):
        self._recorder.check_capacity()
        state = self.summary()
        _require(self._failure is None and state["events"] < MAX_EVENTS and state["steps"] < MAX_STEPS
            and state["bytes"] + 65536 <= MAX_JOURNAL_BYTES, "Task replay journal capacity reached or action failed")

    def append(self, event):
        self.check_capacity()
        _require(event.get("type") != "start", "A task journal has only one start")
        validate_event(event, self._contract)
        return self._recorder.append(event)

    def mark_failed(self, failure):
        raw = _bytes(failure)
        _require(isinstance(failure, dict) and len(raw) <= 16 << 10 and not self._closed, "Task failure receipt is invalid")
        self._failure = _json(raw)

    def summary(self): return self._recorder.summary()

    def close(self):
        from weft.recording import _atomic_json
        result = self._recorder.close()
        if not self._closed:
            metadata = _json(read_package_file(self.root, "recording.json", max_bytes=BOUNDS["recording.json"]))
            metadata[METADATA_KEY] = {"profile": PROFILE, "contract_sha256": canonical_hash(self._contract)}
            if self._failure is not None:
                metadata["failed_attempt"] = self._failure
            _require(len(_bytes(metadata)) <= BOUNDS["recording.json"], "Task metadata bound exceeded")
            _atomic_json(self.root / "recording.json", metadata)
            self._closed = True
        return result


def _stop(seconds, deadline):
    _require(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 120,
             "Task recording work requires a positive bound up to 120 seconds")
    _require(deadline is None or type(deadline) in (int, float) and math.isfinite(deadline), "Task deadline must be finite")
    return min(time.monotonic() + seconds, deadline) if deadline is not None else time.monotonic() + seconds


def _check(stop):
    if time.monotonic() >= stop:
        raise TimeoutError("Task recording exceeded its aggregate deadline")


def _read(directory, stop):
    raw = {}
    for name, maximum in BOUNDS.items():
        _check(stop)
        raw[name] = read_package_file(directory, name, max_bytes=maximum)
    docs = {name: _json(value) for name, value in raw.items() if name != "events.jsonl"}
    scene, task, bundle, contract = (docs[name] for name in ("scene.json", "placement-task.json", "moving-proposals.json", "task-recording-contract.json"))
    validate_contract(contract, scene, task, bundle)
    metadata, data = docs["recording.json"], raw["events.jsonl"]
    _require(metadata.get("status") == "closed" and metadata.get("profile") == "shared-world"
        and type(metadata.get("schema_version")) is int and metadata["schema_version"] == 1
        and metadata.get("scene_sha256") == canonical_hash(scene)
        and metadata.get(METADATA_KEY) == {"profile": PROFILE, "contract_sha256": canonical_hash(contract)}
        and not {"failed_attempt", "action_replay_contract", "shared_world_action_replay_contract"}.intersection(metadata),
        "Task replay requires a closed task-bound journal without an unreplayed failed attempt")
    _require(type(metadata.get("journal_bytes")) is int and metadata["journal_bytes"] == len(data)
        and data.endswith(b"\n"), "Task journal byte count or final newline differs")
    lines = data.splitlines()
    _require(1 <= len(lines) <= MAX_EVENTS and all(len(line) + 1 <= 65536 for line in lines)
        and type(metadata.get("events")) is int and metadata["events"] == len(lines), "Task journal count or size differs")
    events, steps, resets = [], 0, 0
    for index, line in enumerate(lines):
        _check(stop)
        event = _json(line)
        _require(type(event.get("index")) is int and event.pop("index") == index
            and event.get("type") in ({"start"} if index == 0 else {"step", "reset"}), "Task journal event order differs")
        validate_event(event, contract)
        steps += event["type"] == "step"; resets += event["type"] == "reset"
        events.append(event)
    _require(steps <= MAX_STEPS and type(metadata.get("seed")) is int
        and metadata["seed"] == events[0]["seed"] == contract["initial_seed"], "Task journal seed or step bound differs")
    return raw, docs, events, steps, resets


def _proof(contract, sources, events, steps, resets, pinned):
    metrics = events[-1]["snapshot"]["task"]
    completed, success = metrics.get("completed_interval"), metrics.get("success")
    _require(type(completed) is bool and type(success) is bool and (not success or completed), "Task final metrics are inconsistent")
    return {"schema_version": 1, "profile": PROFILE, "exact_state_replay": True, "exact_task_reward_replay": True,
        "contract_sha256": canonical_hash(contract), "source_sha256": sources, "events": len(events), "steps": steps,
        "resets": resets, "final_episode_metrics": deepcopy(metrics), "completed_interval": completed, "success": success,
        "outcome": "completed_success" if success else "completed_failed_criteria" if completed else "stopped_or_terminated_partial",
        "expected_contract_pin_supplied": pinned, "qualification": "caller_pinned_simulation_replay" if pinned else "unbound_self_consistency",
        "source_admission_verified": False, "scope": SCOPE}


def verify_task_replay(directory, *, expected_contract_sha256=None, max_seconds=60, deadline=None):
    stop = _stop(max_seconds, deadline)
    _require(expected_contract_sha256 is None or _hash(expected_contract_sha256), "Task expected contract pin is invalid")
    raw, docs, events, steps, resets = _read(directory, stop)
    scene, task, bundle, contract = (docs[name] for name in ("scene.json", "placement-task.json", "moving-proposals.json", "task-recording-contract.json"))
    _require(expected_contract_sha256 is None or canonical_hash(contract) == expected_contract_sha256, "Task start-time contract pin changed")
    expected_runtime = {k: contract[k] for k in ("runtime_sha256", "runtime_versions")}
    _require(runtime_identity(scene) == expected_runtime, "Task replay runtime differs from recording-time pins")
    from weft.moving_equipment_task import MovingEquipmentTaskEnv
    from weft.moving_equipment_task_lab import compose_task_snapshot
    from weft.shared_world import prepare_shared_snapshot_validator
    validate_snapshot = prepare_shared_snapshot_validator(scene)
    for event in events:
        _check(stop)
        validate_snapshot(event["snapshot"]["engine_diagnostics"])
    _check(stop)
    env = MovingEquipmentTaskEnv(scene, task, bundle)
    try:
        for index, event in enumerate(events):
            _check(stop)
            if event["type"] in {"start", "reset"}:
                env.reset(seed=event["seed"], options={"jitter": False})
            else:
                _, reward, terminated, truncated, _ = env.step(event["action"])
                _require(canonical_hash([reward, terminated, truncated]) == canonical_hash([event[k] for k in ("reward", "terminated", "truncated")]),
                         f"Task reward or termination replay differs at event {index}")
            state = env.snapshot()
            _require(canonical_hash(compose_task_snapshot(state["engine_diagnostics"], state["task"])) == canonical_hash(event["snapshot"]),
                     f"Task full-state or metrics replay differs at event {index}")
    finally:
        env.close()
    _check(stop)
    _require(runtime_identity(scene) == expected_runtime, "Task runtime changed during replay")
    sources = {name: sha256(value).hexdigest() for name, value in raw.items()}
    for name, pin in sources.items():
        _check(stop)
        read_package_file(directory, name, max_bytes=BOUNDS[name], expected_sha256=pin)
    return _proof(contract, sources, events, steps, resets, expected_contract_sha256 is not None)


def export_task_recording(directory, *, expected_contract_sha256, expected_receipt_sha256=None, max_seconds=120, deadline=None):
    """Export ordinary geometry previews plus a distinct task proof; never relabel engine scoring."""
    stop = _stop(max_seconds, deadline)
    _require(_hash(expected_contract_sha256), "Task export requires its exact start-time contract pin")
    root = Path(directory)
    if (root / RECEIPT).exists():
        return load_task_recording_evidence(root, expected_contract_sha256=expected_contract_sha256,
            expected_receipt_sha256=expected_receipt_sha256, deadline=stop)
    proof = verify_task_replay(root, expected_contract_sha256=expected_contract_sha256, deadline=stop)
    from weft.recording import export_recording
    export_recording(root, deadline=stop, max_output_bytes=128 << 20)
    files = {}
    for name in {*BOUNDS, *MEDIA, "manifest.json"}:
        _check(stop)
        limit = BOUNDS.get(name, 64 << 20 if name in MEDIA else 64 << 10)
        raw = read_package_file(root, name, max_bytes=limit)
        files[name] = {"sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
    _require(sum(row["bytes"] for row in files.values()) <= MAX_EXPORT_BYTES, "Task export aggregate byte bound exceeded")
    _require(all(files[name]["sha256"] == pin for name, pin in proof["source_sha256"].items()), "Task source changed during media export")
    receipt = {"schema_version": 1, "profile": PROFILE, "contract_sha256": expected_contract_sha256,
        "files": files, "proof": proof, "media_scope": "Geometry previews; engine diagnostics are not the authored task score. Use the task proof for outcome.",
        "scope": SCOPE}
    _require(len(_bytes(receipt)) <= 256 << 10, "Task receipt byte bound exceeded")
    receipt_raw = _bytes(receipt)
    write_package_files(root, {RECEIPT: receipt_raw})
    return load_task_recording_evidence(root, expected_contract_sha256=expected_contract_sha256,
        expected_receipt_sha256=sha256(receipt_raw).hexdigest(), deadline=stop)


def load_task_recording_evidence(directory, *, expected_contract_sha256, expected_receipt_sha256=None, deadline=None):
    """Bounded saved-proof inspection only: no simulator, inference, rendering or source re-admission."""
    stop = _stop(30, deadline)
    _require(_hash(expected_contract_sha256) and (expected_receipt_sha256 is None or _hash(expected_receipt_sha256)), "Task receipt pins are invalid")
    raw_receipt = read_package_file(directory, RECEIPT, max_bytes=256 << 10, expected_sha256=expected_receipt_sha256)
    receipt = _json(raw_receipt)
    _require(set(receipt) == {"schema_version", "profile", "contract_sha256", "files", "proof", "media_scope", "scope"}
        and type(receipt["schema_version"]) is int and receipt["schema_version"] == 1 and receipt["profile"] == PROFILE
        and receipt["contract_sha256"] == expected_contract_sha256 and receipt["scope"] == SCOPE,
        "Saved task export receipt differs")
    files = receipt["files"]
    _require(isinstance(files, dict) and set(files) == {*BOUNDS, *MEDIA, "manifest.json"}, "Task export closure differs")
    total = 0
    for name, row in files.items():
        _check(stop)
        _require(isinstance(row, dict) and set(row) == {"sha256", "bytes"} and _hash(row["sha256"])
            and type(row["bytes"]) is int and 0 < row["bytes"] <= BOUNDS.get(name, 64 << 20 if name in MEDIA else 64 << 10),
            "Task export member bound differs")
        total += row["bytes"]
    _require(total <= MAX_EXPORT_BYTES, "Task saved export exceeds its aggregate bound")
    for name, row in files.items():
        _check(stop)
        value = read_package_file(directory, name, max_bytes=row["bytes"], expected_sha256=row["sha256"])
        _require(len(value) == row["bytes"], "Task saved export member length differs")
    raw, docs, events, steps, resets = _read(directory, stop)
    contract, scene = docs["task-recording-contract.json"], docs["scene.json"]
    _require(canonical_hash(contract) == expected_contract_sha256, "Task saved start-time pin differs")
    _require(runtime_identity(scene) == {k: contract[k] for k in ("runtime_sha256", "runtime_versions")}, "Task saved exporter/runtime differs")
    sources = {name: sha256(value).hexdigest() for name, value in raw.items()}
    _require(all(files[name]["sha256"] == pin for name, pin in sources.items())
        and canonical_hash(receipt["proof"]) == canonical_hash(_proof(contract, sources, events, steps, resets, True)),
        "Saved task replay proof differs from its bound journal")
    manifest = _json(read_package_file(directory, "manifest.json", max_bytes=64 << 10, expected_sha256=files["manifest.json"]["sha256"]))
    _require(manifest.get("files") == {name: files[name]["sha256"] for name in {*MEDIA, "scene.json", "events.jsonl", "recording.json"}}
        and type(manifest.get("video_frames")) is int and manifest["video_frames"] == len(events), "Task geometry export manifest differs")
    _check(stop)
    read_package_file(directory, RECEIPT, max_bytes=256 << 10, expected_sha256=sha256(raw_receipt).hexdigest())
    trusted = expected_receipt_sha256 is not None
    # Recomputed hashes cannot authenticate a journal and receipt that were rewritten together.
    return {"directory": str(directory), "receipt_sha256": sha256(raw_receipt).hexdigest(), "contract_sha256": expected_contract_sha256,
        ("proof" if trusted else "claimed_proof"): receipt["proof"], "verified_files": len(files),
        "scope": SCOPE, "media_scope": receipt["media_scope"], "exact_replay_verified": trusted,
        "verification": "trusted_saved_receipt_not_a_new_physics_replay" if trusted else "self_consistent_saved_claim_only",
        "receipt_pin_supplied": trusted, "source_admission_verified": False}
