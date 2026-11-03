"""Source-qualified authored placement children; original exploration packages stay unchanged."""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import stat
import time

from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files

PROFILE = "moving-equipment-task-lab-v1"
CONTROL_PROFILE = "moving_equipment_task"
MAX_LAB_BYTES = 256 << 20
MAX_LAB_FILES = 2080
MAX_SECONDS = 90
IMPLEMENTATIONS = ("moving_equipment_task_lab.py", "moving_equipment_task.py", "moving_equipment_task_recording.py",
                   "shared_exercise_task.py", "whole_body.py")
PARAMETERS = {"target_object_id", "target_position_world", "position_tolerance_world", "speed_tolerance_world_s", "hold_duration_s"}
SCOPE = "Authored placement task with explicit virtual-force controls; not recovered exercise intent, physical grasp or independent changed-start evaluation."
_TOKEN = object()


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _json(raw):
    from weft.shared_world_action_replay import _json as parse
    return parse(raw)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Authored task lab exceeded its aggregate deadline")


def _deadline(seconds, deadline=None):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= MAX_SECONDS:
        raise ValueError("Task lab needs a finite positive bound up to 90 seconds")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("Task lab deadline must be finite")
    return min(time.monotonic() + seconds, deadline) if deadline is not None else time.monotonic() + seconds


def _signature(path):
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Task lab members must be ordinary nonlinked files")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _runtime(parent_contract):
    own = {"implementation/" + name: read_package_file(Path(__file__).parent, name, max_bytes=2 << 20)
           for name in IMPLEMENTATIONS}
    pins = dict(parent_contract["implementation_sha256"])
    pins.update({name.removeprefix("implementation/"): sha256(raw).hexdigest() for name, raw in own.items()})
    return own, pins


def _contract(parent_config_raw, parent_manifest_raw, parent_contract, task, request, runtime):
    return {"schema_version": 1, "profile": PROFILE, "control_profile": CONTROL_PROFILE,
        "parent_manifest_sha256": sha256(parent_manifest_raw).hexdigest(), "parent_config_sha256": sha256(parent_config_raw).hexdigest(),
        "parent_contract_sha256": canonical_hash(parent_contract), "scene_sha256": task["binding"]["scene_sha256"],
        "task_sha256": canonical_hash(task), "task_request_sha256": canonical_hash(request),
        "candidate_id": parent_contract["candidate_id"], "family_sha256": task["binding"]["family_sha256"],
        "bundle_sha256": task["binding"]["bundle_sha256"], "source_sha256": parent_contract["source_sha256"],
        "source_file": parent_contract["source_file"], "runtime_sha256": runtime, "controller_modes": ["manual"],
        "reset_options": {"jitter": False}, "learned_policy_available": False, "scope": SCOPE,
        "recording_scope": "Task recording requires an independently validated task-bound replay contract; this package alone is not replay proof."}


def prepare_moving_equipment_task_lab(parent_lab, task_parameters, out, *, parent_manifest_sha256, max_seconds=90, deadline=None):
    from weft.shared_world_exploration import load_exploration_lab
    from weft.moving_equipment_task import build_moving_equipment_task

    stop = _deadline(max_seconds, deadline)
    parent, out = Path(parent_lab).absolute(), Path(out).absolute()
    if (out.exists() or out.is_relative_to(parent) or ".." in out.parts or ".." in parent.parts
            or any(p.is_symlink() for p in (out, parent, *out.parents, *parent.parents))):
        raise ValueError("Choose a new nonlinked task child outside its original exploration lab")
    if not isinstance(task_parameters, dict) or set(task_parameters) != PARAMETERS or len(_bytes(task_parameters)) > 16 << 10:
        raise ValueError("Task parameters require only the explicit target, tolerances and hold duration")
    request = _json(_bytes(task_parameters))
    config_raw = read_package_file(parent, "lab-config.json", max_bytes=64 << 10)
    config = _json(config_raw)
    manifest_raw, pins = read_package_manifest(parent / "run")
    if sha256(manifest_raw).hexdigest() != parent_manifest_sha256 or config.get("manifest_sha256") != parent_manifest_sha256:
        raise ValueError("Parent exploration manifest differs from the caller's exact pin")
    # Capture own code before source/native admission, not after its work has executed.
    before = {name: read_package_file(Path(__file__).parent, name, max_bytes=2 << 20) for name in IMPLEMENTATIONS}
    admitted = load_exploration_lab(parent / "run", config, max_seconds=max(.01, min(60, stop-time.monotonic())))
    scene = admitted["scene"]
    bundle_raw = read_package_file(parent / "run", "moving-proposals.json", max_bytes=16 << 20, expected_sha256=pins["moving-proposals.json"])
    bundle = _json(bundle_raw)
    task = build_moving_equipment_task(scene, bundle, candidate_id=admitted["contract"]["candidate_id"], **request)
    own, runtime = _runtime(admitted["contract"])
    if any(own["implementation/" + name] != raw for name, raw in before.items()):
        raise ValueError("Authored task implementation changed during parent admission")
    contract = _contract(config_raw, manifest_raw, admitted["contract"], task, request, runtime)
    files = {"parent/lab-config.json": config_raw, "parent/run/manifest.json": manifest_raw,
        "placement-task.json": _bytes(task), "task-request.json": _bytes(request), "task-lab.json": _bytes(contract), **own}
    signatures, total = {}, sum(map(len, files.values()))
    if len(pins) + len(files) + 6 > MAX_LAB_FILES:
        raise ValueError("Task child exceeds its complete file bound")
    for name, pin in pins.items():
        _check(stop)
        path = parent / "run" / name
        signature = _signature(path)
        if total + signature[2] + (2 << 20) > MAX_LAB_BYTES:
            raise ValueError("Task child exceeds its complete byte bound")
        raw = read_package_file(parent / "run", name, max_bytes=64 << 20, expected_sha256=pin)
        if _signature(path) != signature:
            raise ValueError("Parent member changed during child preparation")
        files["parent/run/" + name] = raw; total += len(raw); signatures[name] = signature
    for name in ("scene.json", "human-reference.json", contract["source_file"]):
        raw = files["parent/run/" + name]
        files[name] = raw; total += len(raw)
    if total + (2 << 20) > MAX_LAB_BYTES:
        raise ValueError("Task child aliases exceed the unchanged 256 MiB lab bound")
    manifest = {"schema_version": 1, "profile": PROFILE, "control_profile": CONTROL_PROFILE, "status": "ready",
        "files": {name: sha256(raw).hexdigest() for name, raw in sorted(files.items())}}
    encoded = _bytes(manifest)
    child_config = {"schema_version": 1, "profile": "shared-world", "control_profile": CONTROL_PROFILE,
        "manifest_sha256": sha256(encoded).hexdigest(), "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task),
        "learned_policy_available": False}
    (out / "run").mkdir(parents=True, mode=0o700)
    for name, raw in files.items():
        _check(stop)
        write_package_files(out / "run", {name: raw})
    _check(stop)
    if (read_package_file(parent, "lab-config.json", max_bytes=64 << 10) != config_raw
            or read_package_file(parent / "run", "manifest.json", max_bytes=8 << 20) != manifest_raw
            or any(_signature(parent / "run" / name) != signature for name, signature in signatures.items())
            or any(read_package_file(Path(__file__).parent, name, max_bytes=2 << 20) != raw for name, raw in before.items())):
        raise ValueError("Parent or task runtime changed before child publication")
    write_package_files(out / "run", {"manifest.json": encoded})
    write_package_files(out, {"lab-config.json": _bytes(child_config), "status.json": _bytes({"status": "ready", "events": [
        {"stage": "authored_task", "message": SCOPE, "time": local_time()}]})})
    return {"status": "ready", "lab_directory": str(out), "manifest_sha256": child_config["manifest_sha256"],
        "parent_manifest_sha256": parent_manifest_sha256, "scene_sha256": child_config["scene_sha256"],
        "task_sha256": child_config["task_sha256"], "control_profile": CONTROL_PROFILE, "learned_policy_available": False,
        "scope": SCOPE, "bytes_written": total + len(encoded) + len(_bytes(child_config)) + (out / "status.json").stat().st_size}


@dataclass(frozen=True, init=False, slots=True)
class TaskLabAdmission:
    _token: object
    scene_bytes: bytes
    task_bytes: bytes
    bundle_bytes: bytes
    contract_bytes: bytes
    config_bytes: bytes
    source_check_bytes: bytes
    runtime_bytes: bytes
    artifacts_bytes: bytes
    manifest_sha256: str
    parent_manifest_sha256: str

    @property
    def scene(self): return _json(self.scene_bytes)
    @property
    def task(self): return _json(self.task_bytes)
    @property
    def bundle(self): return _json(self.bundle_bytes)
    @property
    def contract(self): return _json(self.contract_bytes)
    @property
    def config(self): return _json(self.config_bytes)
    @property
    def source_check(self): return _json(self.source_check_bytes)
    @property
    def runtime_sha256(self): return _json(self.runtime_bytes)
    @property
    def artifacts(self): return _json(self.artifacts_bytes)["items"]


def validate_task_lab_admission(value, *, deadline=None):
    if deadline is not None:
        _check(deadline)
    if not isinstance(value, TaskLabAdmission) or value._token is not _TOKEN:
        raise ValueError("Task use requires an immutable source-admitted child capability")
    return value


def load_moving_equipment_task_lab(run, config, *, max_seconds=90, deadline=None):
    from weft.shared_world_exploration import load_exploration_lab
    from weft.moving_equipment_task import build_moving_equipment_task

    stop = _deadline(max_seconds, deadline); run = Path(run)
    manifest_raw, pins = read_package_manifest(run)
    manifest = _json(manifest_raw)
    if (config.get("profile") != "shared-world" or config.get("control_profile") != CONTROL_PROFILE
            or config.get("manifest_sha256") != sha256(manifest_raw).hexdigest() or config.get("learned_policy_available") is not False
            or manifest.get("profile") != PROFILE or manifest.get("control_profile") != CONTROL_PROFILE
            or manifest.get("status") != "ready" or len(pins) + 4 > MAX_LAB_FILES):
        raise ValueError("Authored task config or manifest identity differs")
    required = {"parent/lab-config.json", "parent/run/manifest.json", "placement-task.json", "task-request.json", "task-lab.json", "scene.json", "human-reference.json"}
    if not required <= pins.keys():
        raise ValueError("Authored task parent/definition closure is missing")
    signatures, total = {}, len(manifest_raw)
    for name in pins:
        _check(stop)
        signature = _signature(run / name)
        total += signature[2]; signatures[name] = signature
        if signature[2] > 64 << 20 or total + (2 << 20) > MAX_LAB_BYTES:
            raise ValueError("Authored task complete input bound exceeded")
    def read(name, limit=16 << 20):
        return read_package_file(run, name, max_bytes=limit, expected_sha256=pins[name])
    parent_config_raw = read("parent/lab-config.json", 64 << 10)
    parent_config = _json(parent_config_raw)
    parent_manifest_raw, parent_pins = read_package_manifest(run / "parent/run")
    if sha256(parent_manifest_raw).hexdigest() != pins["parent/run/manifest.json"]:
        raise ValueError("Nested parent manifest differs from child custody")
    own_names = {"implementation/" + name for name in IMPLEMENTATIONS}
    contract = _json(read("task-lab.json", 128 << 10))
    expected_names = required | {contract["source_file"]} | own_names | {"parent/run/" + name for name in parent_pins}
    if set(pins) != expected_names or any(pins["parent/run/" + name] != pin for name, pin in parent_pins.items()):
        raise ValueError("Task child must preserve the exact complete parent inventory")
    before = {name: read_package_file(Path(__file__).parent, name, max_bytes=2 << 20) for name in IMPLEMENTATIONS}
    if any(read("implementation/" + name, 2 << 20) != raw for name, raw in before.items()):
        raise ValueError("Authored task requires its exact current trusted runtime")
    # Parent loader hashes and validates its complete source once; no duplicate parent-byte dictionary here.
    parent = load_exploration_lab(run / "parent/run", parent_config, max_seconds=max(.01, min(60, stop-time.monotonic())))
    scene = parent["scene"]
    bundle_raw = read("parent/run/moving-proposals.json")
    bundle = _json(bundle_raw)
    request = _json(read("task-request.json", 16 << 10))
    if not isinstance(request, dict) or set(request) != PARAMETERS:
        raise ValueError("Task child request has unsupported fields")
    task = build_moving_equipment_task(scene, bundle, candidate_id=parent["contract"]["candidate_id"], **request)
    own, runtime = _runtime(parent["contract"])
    expected = _contract(parent_config_raw, parent_manifest_raw, parent["contract"], task, request, runtime)
    if (contract != expected or read("placement-task.json", 64 << 10) != _bytes(task)
            or config.get("task_sha256") != canonical_hash(task) or config.get("scene_sha256") != canonical_hash(scene)
            or any(own["implementation/" + name] != raw for name, raw in before.items())):
        raise ValueError("Task definition, source binding, controls or runtime differs from rederivation")
    for name in ("scene.json", "human-reference.json", contract["source_file"]):
        if pins[name] != pins["parent/run/" + name]:
            raise ValueError("Task viewer aliases differ from the admitted parent")
        read(name, 64 << 20 if name == contract["source_file"] else 16 << 20)
    _check(stop)
    if (read_package_file(run, "manifest.json", max_bytes=8 << 20) != manifest_raw
            or any(_signature(run / name) != signature for name, signature in signatures.items())
            or any(read_package_file(Path(__file__).parent, name, max_bytes=2 << 20) != raw for name, raw in before.items())):
        raise ValueError("Task child or runtime changed during admission")
    result = object.__new__(TaskLabAdmission)
    values = {"_token": _TOKEN, "scene_bytes": _bytes(scene), "task_bytes": _bytes(task), "bundle_bytes": bundle_raw,
        "contract_bytes": _bytes(contract), "config_bytes": _bytes(config), "source_check_bytes": _bytes(parent["source_check"]),
        "runtime_bytes": _bytes(runtime), "artifacts_bytes": _bytes({"items": [{"name": name, "bytes": sig[2]} for name, sig in signatures.items()]}),
        "manifest_sha256": sha256(manifest_raw).hexdigest(), "parent_manifest_sha256": sha256(parent_manifest_raw).hexdigest()}
    for key, value in values.items():
        object.__setattr__(result, key, value)
    return result


def compose_task_snapshot(engine, metrics, failure=None):
    """Keep render state and untouched engine diagnostics distinct from authored task success."""
    result = {**deepcopy(engine), "task": deepcopy(metrics), "engine_diagnostics": deepcopy(engine),
        "success": metrics.get("success") is True, "control_profile": CONTROL_PROFILE}
    if failure is not None:
        result.update(terminated=True, simulation_failure=deepcopy(failure))
    return result


class MovingEquipmentTaskSession:
    control_profile = CONTROL_PROFILE

    def __init__(self, admission):
        from weft.moving_equipment_task import MovingEquipmentTaskEnv
        self.admission = validate_task_lab_admission(admission)
        self.scene, self.task, self.bundle = admission.scene, admission.task, admission.bundle
        self.source_binding_verification = admission.source_check
        self.artifacts, self.policy = admission.artifacts, None
        self.env = MovingEquipmentTaskEnv(self.scene, self.task, self.bundle)
        try:
            self.reset(0)
        except BaseException:
            self.env.close(); raise

    def reset(self, seed):
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError("Task reset requires a bounded integer seed")
        self.observation, _ = self.env.reset(seed=seed, options={"jitter": False})
        self.seed, self.done, self.failure = seed, False, None
        self._state = self.env.snapshot()
        return self.snapshot()

    def resolve_action(self, action):
        if (not isinstance(action, list) or len(action) != len(self.env.action_names)
                or any(type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1 for value in action)):
            raise ValueError("Authored task accepts only its full finite normalized manual action vector")
        return [float(value) for value in action]

    def step(self, action):
        vector = self.resolve_action(action)
        try:
            self.observation, reward, terminated, truncated, info = self.env.step(vector)
            self._state = self.env.snapshot()
        except RuntimeError as error:
            self.done = True
            self.failure = {"reason": str(error), "attempted_action": vector, "scope": "Last valid engine state retained; task failure is separate."}
            self._state["task"] = self.env.metrics()
            raise
        self.done = bool(terminated or truncated)
        return reward, terminated, truncated, info

    def snapshot(self):
        return compose_task_snapshot(self._state["engine_diagnostics"], self._state["task"], self.failure)

    def engine_snapshot(self): return deepcopy(self._state["engine_diagnostics"])
    def select_actor(self, name): raise ValueError("Task actor is fixed; use the exact named action channels")
    def reference_action(self): raise ValueError("This authored task has no qualified reference controller")
    def policy_action(self): raise ValueError("This authored task has no admitted learned policy")
    def close(self): self.env.close()


def create_session(run, config, *, max_seconds=90, deadline=None):
    stop = _deadline(max_seconds, deadline)
    session = MovingEquipmentTaskSession(load_moving_equipment_task_lab(run, config, max_seconds=max_seconds, deadline=stop))
    try:
        _check(stop)
        return session
    except BaseException:
        session.close(); raise


_LAB_TYPE = None


def _task_lab_type():
    """Import native/UI dependencies only when opening a lab, never for saved-receipt projection."""
    global _LAB_TYPE
    if _LAB_TYPE is not None:
        return _LAB_TYPE
    from uuid import uuid4
    from weft.lab import Lab

    class TaskLab(Lab):
        def __init__(self, root, *, max_seconds=90, deadline=None):
            self._load_deadline = _deadline(max_seconds, deadline)
            self._record_start_future = None
            self._record_start_state = None
            super().__init__(root)

        def load_session(self):
            if self.session is None:
                if self.status().get("status") != "ready":
                    raise ValueError("Authored task child is not ready")
                session = create_session(self.root / "run", self.config, deadline=self._load_deadline)
                try:
                    from weft.moving_equipment_task_recording import runtime_identity
                    self._recording_runtime = runtime_identity(session.scene)
                    expected = session.admission.runtime_sha256
                    if any(expected.get(name) != pin for name, pin in self._recording_runtime["runtime_sha256"].items()):
                        raise ValueError("Task recording runtime is not sealed by this child")
                    self.session, self.artifacts = session, session.artifacts
                    self._trusted_receipts = {}
                    self._task_capabilities = self._prepare_capabilities(session)
                    self.evidence_provenance(self._task_capabilities)
                    _check(self._load_deadline)
                except BaseException:
                    self.session = None; session.close(); raise
            return self.session

        def _prepare_capabilities(self, session):
            value = super().capabilities(session)
            value["controllers"].update(learned_policy_available=False, authored_reference_controller=False,
                fitted_reference_controller=False, coast=False)
            value["task_contract"] = {"task_type": session.task["profile"], "authored_addition": True,
                "reward_summary": "Authored placement proximity and settling with original whole-interval body/floor limits; not the engine parking reward.",
                "success_criteria": {"goal": deepcopy(session.task["goal"]), "body_gates": deepcopy(session.task["body_gates"]), "complete_interval": True},
                "termination_conditions": value["task_contract"]["termination_conditions"],
                "truncation_steps": session.task["interval"]["horizon_steps"]}
            value["reset_contract"]["supports_jitter"] = False
            value["observation_contract"]["dimension"] = int(session.env.observation_space.shape[0])
            value["observation_contract"]["components"].append("authored_placement_target_tolerances_hold_elapsed_8")
            return value

        def capabilities(self, session):
            return deepcopy(self._task_capabilities)

        def summary(self):
            self._complete_record_start()
            if self.status().get("status") == "ready":
                self.load_session()
            result = super().summary()
            result.update(control_profile=CONTROL_PROFILE, task_recording_available=self.session is not None, learned_policy_available=False,
                recording_preparation=deepcopy(self._record_start_state))
            if self.session is not None:
                result.update(placement_task=deepcopy(self.session.task), placement_task_sha256=self.config["task_sha256"],
                    scope=SCOPE, parent_manifest_sha256=self.session.admission.parent_manifest_sha256)
            return result

        def control(self, path, data):
            if not isinstance(data, dict):
                raise ValueError("Authored task controls require a JSON object")
            session = self.load_session()
            self._complete_record_start()
            if self._record_start_future is not None:
                raise ValueError("Task recording is preparing; controls remain paused until its reset-bound start is sealed")
            if path == "/api/reset":
                if set(data) - {"seed"}:
                    raise ValueError("Task resets accept only an exact-start seed")
                seed = data.get("seed", self.seed)
                if self.recorder:
                    self.recorder.check_capacity()
                snapshot = session.reset(seed); self.seed = seed
                if self.recorder:
                    self.recorder.append({"type": "reset", "seed": seed, "snapshot": snapshot})
                return {"snapshot": snapshot}
            if path == "/api/step":
                if set(data) != {"action", "steps"} or type(data["steps"]) is not int or not 1 <= data["steps"] <= 120:
                    raise ValueError("Authored task accepts only an explicit action vector and 1–120 steps")
                action = session.resolve_action(data["action"])
                transitions = []
                for _ in range(data["steps"]):
                    if session.done:
                        break
                    if self.recorder:
                        self.recorder.check_capacity()
                    try:
                        reward, terminated, truncated, _ = session.step(action)
                    except RuntimeError:
                        if self.recorder:
                            recorder, self.recorder = self.recorder, None
                            recorder.mark_failed(session.failure)
                            recorder.close()
                            self._queue_export(recorder, session)
                        return {"snapshot": session.snapshot(), "transitions": transitions, "recording": False,
                                "warning": "Failed action retained; no complete task replay is claimed."}
                    event = {"type": "step", "action": list(action), "reward": float(reward), "terminated": bool(terminated),
                             "truncated": bool(truncated), "snapshot": session.snapshot(), "controller": "manual"}
                    if self.recorder:
                        self.recorder.append(event)
                    transitions.append(event)
                return {"snapshot": session.snapshot(), "transitions": transitions}
            if path in {"/api/record/start", "/api/record/stop"}:
                if data:
                    raise ValueError("Task recording cannot override its source, task or controls")
                if path.endswith("start"):
                    if self.recorder or (self.export_future is not None and not self.export_future.done()):
                        raise ValueError("Finish the current task recording/export first")
                    name = "session-" + uuid4().hex[:12]
                    snapshot, seed = session.snapshot(), self.seed
                    if snapshot.get("steps") != 0 or snapshot.get("time_s") != 0 or snapshot.get("terminated") is not False or snapshot.get("truncated") is not False:
                        raise ValueError("Reset the task before recording; arbitrary mid-run starts are unsupported")
                    self._record_start_state = {"status": "running", "directory": name}
                    def prepare_recording():
                        from weft.moving_equipment_task_recording import recording_contract, TaskRecorder
                        contract = recording_contract(session.scene, session.task, session.bundle,
                            task_lab_manifest_sha256=session.admission.manifest_sha256,
                            parent_manifest_sha256=session.admission.parent_manifest_sha256,
                            task_lab_contract_sha256=canonical_hash(session.admission.contract), seed=seed,
                            runtime_snapshot=self._recording_runtime)
                        recorder = TaskRecorder(self.root / "sessions" / name, session.scene, session.task, session.bundle, contract)
                        try:
                            recorder.start(snapshot, seed=seed)
                        except BaseException:
                            try:
                                if recorder.summary().get("events", 0):
                                    recorder.close()
                            except (ValueError, OSError):
                                pass
                            raise
                        return recorder, canonical_hash(snapshot), seed
                    self._record_start_future = self.executor.submit(prepare_recording)
                    return {"recording": False, "preparing": True, "directory": name, "snapshot": snapshot}
                if self.recorder is None:
                    raise ValueError("No task recording is active")
                recorder, self.recorder = self.recorder, None
                recorder.close(); self._queue_export(recorder, session)
                return {"recording": False, "export": "running", "directory": self.export_path}
            raise ValueError("Authored task supports manual vectors, exact-start resets and task recording only; actor, avatar, reference and learned controls are unavailable")

        def _complete_record_start(self):
            future = self._record_start_future
            if future is None or not future.done():
                return
            self._record_start_future = None
            try:
                recorder, snapshot_sha, seed = future.result()
                if self.seed != seed or canonical_hash(self.session.snapshot()) != snapshot_sha:
                    recorder.close()
                    raise ValueError("Task reset or state changed during recording preparation")
                self.recorder = recorder
                self._record_start_state = {"status": "ready", "directory": recorder.root.name}
            except Exception as error:
                self._record_start_state = {**self._record_start_state, "status": "failed", "error": str(error)[:1000]}

        def close(self):
            future = self._record_start_future
            if future is not None:
                if not future.cancel():
                    try:
                        recorder, _, _ = future.result(timeout=90)
                        recorder.close()
                    except Exception:
                        pass
                self._record_start_future = None
            super().close()

        def _queue_export(self, recorder, session):
            from weft.moving_equipment_task_recording import export_task_recording
            self.export_path = recorder.root.name
            self.recording_evidence_name = None
            pin = canonical_hash(recorder.contract)
            def export():
                result = export_task_recording(recorder.root, expected_contract_sha256=pin)
                if result.get("exact_replay_verified") is not True:
                    raise ValueError("Fresh task export did not return an independently pinned replay receipt")
                self._trusted_receipts[recorder.root.name] = result["receipt_sha256"]
                while len(self._trusted_receipts) > 100:
                    self._trusted_receipts.pop(next(iter(self._trusted_receipts)))
                return {**result, "recording_evidence": {"status": "available", **result}}
            self.export_future = self.executor.submit(export)

        def recording_evidence(self, name):
            from weft.lab_inspection import recording_session_name
            name = recording_session_name(name)
            if self.export_path == name and self.export_future is not None and not self.export_future.done():
                return {"status": "loading", "session": name}
            now = time.monotonic()
            if self.recording_evidence_future is not None:
                if self.recording_evidence_name == name and (not self.recording_evidence_future.done() or now-self.recording_evidence_started < 600):
                    return self.recording_evidence_future.result() if self.recording_evidence_future.done() else {"status": "loading", "session": name}
                if not self.recording_evidence_future.done() and not self.recording_evidence_future.cancel():
                    return {"status": "loading", "session": name}
            self.recording_evidence_name, self.recording_evidence_started = name, now
            session = self.load_session()
            def inspect():
                from weft.moving_equipment_task_recording import load_task_recording_evidence, recording_contract
                try:
                    directory = self.root / "sessions" / name
                    contract = _json(read_package_file(directory, "task-recording-contract.json", max_bytes=64 << 10))
                    expected = recording_contract(session.scene, session.task, session.bundle,
                        task_lab_manifest_sha256=session.admission.manifest_sha256,
                        parent_manifest_sha256=session.admission.parent_manifest_sha256,
                        task_lab_contract_sha256=canonical_hash(session.admission.contract),
                        seed=contract["initial_seed"], runtime_snapshot=self._recording_runtime)
                    if contract != expected:
                        raise ValueError("Saved task recording differs from this exact source-qualified task child")
                    result = load_task_recording_evidence(directory, expected_contract_sha256=canonical_hash(expected),
                        expected_receipt_sha256=self._trusted_receipts.get(name))
                    return {"status": "available", "session": name, **result}
                except (ValueError, OSError, KeyError) as error:
                    return {"status": "unavailable", "session": name, "reason": str(error)[:1000]}
            self.recording_evidence_future = self.executor.submit(inspect)
            return {"status": "loading", "session": name}

        def exercise_coverage(self): return {"status": "not_applicable"}
        def inspection(self): return {"status": "not_applicable", "reason": "Inspect the sealed placement task and nested parent artifacts; stationary-equipment diagnosis does not apply."}
        def recording_inspection(self, name): return {"status": "not_applicable", "reason": "Use the task-bound saved recording receipt."}
        def start_worker(self): raise ValueError("Authored task labs are prepared explicitly; no generic training worker is admitted")

    _LAB_TYPE = TaskLab
    return TaskLab


def __getattr__(name):
    if name == "TaskLab":
        return _task_lab_type()
    raise AttributeError(name)


from weft.shared_world_exploration import LoadingExplorationLab


class LoadingTaskLab(LoadingExplorationLab):
    """Reuse the single bounded source-loader executor across task and exploration labs."""

    @property
    def loading(self):
        if not self._future.done():
            return True
        if self._future.cancelled() or self._future.exception() is not None:
            return False
        # Pending and completed-but-unadopted starts own a recorder; cache eviction must not strand it.
        return self._future.result()._record_start_future is not None

    def _load(self):
        lab = _task_lab_type()(self.root)
        try:
            lab.load_session()
            return lab
        except BaseException:
            lab.close(); raise

    def summary(self):
        if not self._future.done():
            return {"status": "running", "profile": "shared-world", "control_profile": CONTROL_PROFILE,
                "task_recording_available": False, "recording": False, "events": [{"stage": "source_validation", "message": "Checking the original world and authored task"}]}
        try:
            return self._ready().summary()
        except Exception as error:
            return {"status": "failed", "profile": "shared-world", "control_profile": CONTROL_PROFILE,
                    "task_recording_available": False, "recording": False, "error": {"message": str(error)[:1000]}, "events": []}
