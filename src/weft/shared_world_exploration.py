"""Sealed manual exploration of conditional moving worlds, without an exercise scorer."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import threading
import time

from weft.io import canonical_hash, local_time, write_json
from weft.shared_world_action_replay import runtime_identity as replay_runtime_identity
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files, validate_package_layout

PROFILE = "shared-world-exploration-lab-v1"
CONTROL_PROFILE = "shared_world_exploration"
MAX_LAB_BYTES = 256 << 20
MAX_LAB_FILES = 2048
MAX_BUNDLE_BYTES = 8 << 20
SCOPE = "Manual simulated exploration of conditional body/equipment hypotheses; no configured exercise-success task, learned policy, grasp or physical validation."
IMPLEMENTATIONS = ("shared_world_exploration.py", "moving_equipment_inputs.py", "moving_equipment_evidence.py",
    "moving_equipment_proposal.py", "moving_equipment_world.py", "lab.py", "capabilities.py", "recording_receipt.py",
    "shared_exercise_lab.py", "shared_exercise_reset.py", "dynamic_native_interval.py", "native_interval.py", "source_clock.py")


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _json(raw):
    from weft.shared_world_action_replay import _json as parse
    return parse(raw)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Exploration preparation exceeded its deadline")


def _deadline(seconds, deadline=None):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 180:
        raise ValueError("Exploration requires a positive bound of at most 180 seconds")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
        raise ValueError("Exploration deadline must be finite")
    return min(time.monotonic() + seconds, deadline) if deadline is not None else time.monotonic() + seconds


def _implementation(scene, inputs, *, depth_selected=False):
    identity = replay_runtime_identity(scene)
    names = set(identity["runtime_sha256"]) | set(IMPLEMENTATIONS) | set(_json(inputs.admission_receipt_bytes)["runtime_sha256"])
    if depth_selected:
        names.update(("moving_depth_family.py", "moving_depth_geometry.py", "moving_depth_intervals.py"))
    files = {"implementation/" + name: read_package_file(Path(__file__).parent, name, max_bytes=2 << 20)
             for name in sorted(names)}
    return files, identity


def _media(inputs):
    source = inputs.source_binding["source_sha256"]
    matches = [(name, raw) for name, raw in inputs.source_files.items()
               if re.search(r"(^|/)source\.(mp4|webm|mov|m4v|avi|mkv)$", name)
               and sha256(raw).hexdigest() == source]
    if not matches:
        raise ValueError("Exploration needs the exact original source video in its admitted inputs")
    name, raw = min(matches)
    return "source" + Path(name).suffix, raw


def _inventory(inputs, bundle, scene, candidate_id, deadline):
    files = {"source-inputs/" + name: raw for name, raw in inputs.source_files.items()}
    for digest, path in scene["source_binding"]["evidence_documents"].items():
        if path in files or path in {"scene.json", "human.json", "task.json", "motion-evidence.json", "source-admission.json",
                                    "moving-proposals.json", "exploration-lab.json", "manifest.json"}:
            raise ValueError("Moving source observation path collides with exploration outputs")
        raw = inputs.observation_documents[digest]
        if sha256(raw).hexdigest() != digest:
            raise ValueError("Moving observation document hash differs")
        files[path] = raw
    media_name, media = _media(inputs)
    if media_name in files:
        raise ValueError("Exploration video path collides with observation evidence")
    files[media_name] = media
    from weft.shared_exercise_lab import _reference_frames
    reference_bytes = (json.dumps(_reference_frames(scene, deadline), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if len(reference_bytes) > 16 << 20:
        raise ValueError("Complete kinematic reference exceeds the exploration preview bound; no poses were clipped")
    files.update({"human-reference.json": reference_bytes, "scene.json": _bytes(scene), "human.json": inputs.human_bytes,
        "task.json": _bytes(scene["task"]), "motion-evidence.json": inputs.motion_evidence_bytes,
        "source-admission.json": inputs.admission_receipt_bytes, "moving-proposals.json": _bytes(bundle)})
    implementation, runtime = _implementation(scene, inputs, depth_selected=bundle.get("profile") == "moving-depth-selected-proposals-v1")
    files.update(implementation)
    if len(files["moving-proposals.json"]) > MAX_BUNDLE_BYTES:
        raise ValueError("Moving proposal bundle exceeds exploration bound")
    contract = {"schema_version": 1, "profile": PROFILE, "scene_sha256": canonical_hash(scene),
        "candidate_id": candidate_id, "bundle_sha256": sha256(files["moving-proposals.json"]).hexdigest(),
        "motion_ledger_sha256": sha256(inputs.motion_evidence_bytes).hexdigest(),
        "source_admission_sha256": sha256(inputs.admission_receipt_bytes).hexdigest(),
        "source_file": media_name, "source_sha256": scene["source_binding"]["source_sha256"],
        "source_files": {name: sha256(raw).hexdigest() for name, raw in inputs.source_files.items()},
        "implementation_sha256": {name.removeprefix("implementation/"): sha256(raw).hexdigest() for name, raw in implementation.items()},
        "replay_runtime": runtime, "controller_modes": ["manual"], "reset_options": {"jitter": False},
        "evaluation_status": "not_configured", "task_success_available": False, "learned_policy_available": False,
        "physical_validation": "unverified", "scope": SCOPE}
    files["exploration-lab.json"] = _bytes(contract)
    validate_package_layout(files)
    if len(files) + 4 > MAX_LAB_FILES or sum(map(len, files.values())) + (2 << 20) > MAX_LAB_BYTES:
        raise ValueError("Exploration lab exceeds its complete file/byte bound")
    return files, contract


def prepare_shared_world_exploration(inputs, bundle, candidate_id, out, *, max_seconds=60, deadline=None, depth_family=None):
    from weft.moving_equipment_inputs import validate_moving_equipment_inputs
    from weft.moving_equipment_world import construct_moving_equipment_world

    stop = _deadline(max_seconds, deadline)
    validate_moving_equipment_inputs(inputs, deadline=stop)
    out = Path(out).absolute()
    if out.exists() or ".." in out.parts or any(p.is_symlink() for p in (out, *out.parents)):
        raise ValueError("Choose a new nonlinked exploration lab directory")
    minimum_bytes = sum(map(len, inputs.source_files.values())) + len(inputs.motion_evidence_bytes) + len(inputs.admission_receipt_bytes) + len(_bytes(bundle))
    if minimum_bytes + (2 << 20) > MAX_LAB_BYTES or len(inputs.source_files) + 40 > MAX_LAB_FILES:
        raise ValueError("Exploration inputs exceed the complete lab bound before construction")
    # The immutable admitted packet is shared across candidates; no repeated source reads.
    before_implementation, _ = _implementation(inputs.base_scene, inputs, depth_selected=depth_family is not None)
    options = {"depth_family": depth_family} if depth_family is not None else {}
    scene = construct_moving_equipment_world(inputs, bundle, candidate_id, deadline=stop, **options)
    _check(stop)
    files, contract = _inventory(inputs, bundle, scene, candidate_id, stop)
    if any(files.get(name) != raw for name, raw in before_implementation.items()):
        raise ValueError("Exploration implementation changed during construction")
    manifest = {"schema_version": 1, "profile": PROFILE, "control_profile": CONTROL_PROFILE, "status": "ready",
                "files": {name: sha256(raw).hexdigest() for name, raw in sorted(files.items())}}
    manifest_raw = _bytes(manifest)
    config = {"schema_version": 1, "profile": "shared-world", "control_profile": CONTROL_PROFILE,
        "manifest_sha256": sha256(manifest_raw).hexdigest(), "scene_sha256": contract["scene_sha256"],
        "learned_policy_available": False, "task_success_available": False}
    _check(stop)
    (out / "run").mkdir(parents=True, mode=0o700)
    for name, raw in files.items():
        _check(stop)
        write_package_files(out / "run", {name: raw})
    _check(stop)
    write_package_files(out / "run", {"manifest.json": manifest_raw})
    write_package_files(out, {"lab-config.json": _bytes(config)})
    write_json(out / "status.json", {"status": "ready", "events": [{"stage": "exploration", "message": SCOPE, "time": local_time()}]})
    _check(stop)
    return {"status": "ready", "lab_directory": str(out), "run_directory": str(out / "run"),
        "manifest_sha256": config["manifest_sha256"], "scene_sha256": config["scene_sha256"],
        "control_profile": CONTROL_PROFILE, "physical_validation": "unverified", "task_success_available": False,
        "bytes_written": sum(map(len, files.values())) + len(manifest_raw) + len(_bytes(config))
            + (out / "status.json").stat().st_size, "scope": SCOPE}


def load_exploration_lab(run, config, *, max_seconds=60):
    from weft.moving_equipment_inputs import admit_moving_equipment_files
    from weft.moving_equipment_world import construct_moving_equipment_world
    from weft.shared_world_evidence import verify_shared_world_sources

    stop = _deadline(max_seconds)
    manifest_raw, pins = read_package_manifest(run)
    manifest = _json(manifest_raw)
    if (config.get("control_profile") != CONTROL_PROFILE or config.get("profile") != "shared-world"
            or config.get("manifest_sha256") != sha256(manifest_raw).hexdigest()
            or config.get("learned_policy_available") is not False or config.get("task_success_available") is not False
            or manifest.get("profile") != PROFILE or manifest.get("control_profile") != CONTROL_PROFILE
            or manifest.get("status") != "ready" or len(pins) + 4 > MAX_LAB_FILES):
        raise ValueError("Exploration lab config or manifest identity differs")
    files, signatures, total = {}, {}, len(manifest_raw)
    for name, pin in pins.items():
        _check(stop)
        path = Path(run) / name
        before = path.lstat()
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        raw = read_package_file(run, name, max_bytes=min(64 << 20, MAX_LAB_BYTES - total), expected_sha256=pin)
        after = path.lstat()
        if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("Exploration member changed while loading")
        signatures[name] = signature
        total += len(raw); files[name] = raw
    if any(name in files for name in ("exercise-lab.json", "exercise-task.json", "policy.json", "training.json")):
        raise ValueError("Scored or learned lab artifacts cannot become manual exploration")
    required = {"exploration-lab.json", "scene.json", "human.json", "task.json", "motion-evidence.json", "source-admission.json", "moving-proposals.json"}
    if not required <= files.keys():
        raise ValueError("Exploration source/contract closure is incomplete")
    contract = _json(files["exploration-lab.json"])
    scene, bundle = _json(files["scene.json"]), _json(files["moving-proposals.json"])
    _check(stop)
    implementation_names = contract.get("implementation_sha256", {})
    if (not isinstance(implementation_names, dict) or not 1 <= len(implementation_names) <= 128
            or any(not re.fullmatch(r"[a-z][a-z0-9_]*\.py", name) for name in implementation_names)):
        raise ValueError("Exploration implementation closure is unsafe or unbounded")
    implementation = {"implementation/" + name: read_package_file(Path(__file__).parent, name, max_bytes=2 << 20)
                      for name in implementation_names}
    if any(files.get(name) != raw for name, raw in implementation.items()):
        raise ValueError("Exploration requires its exact current trusted runtime; archived execution is unsupported")
    source_files = {name.removeprefix("source-inputs/"): raw for name, raw in files.items() if name.startswith("source-inputs/")}
    packet = admit_moving_equipment_files(source_files, max_seconds=max(.01, min(30., stop - time.monotonic())))
    options = {}
    if bundle.get("profile") == "moving-depth-selected-proposals-v1":
        from weft.moving_depth_family import admit_moving_depth_family, materialize_moving_depth_selection
        family = admit_moving_depth_family(packet, bundle["depth_family"], deadline=stop)
        if materialize_moving_depth_selection(family, bundle["depth_selection"], deadline=stop) != bundle:
            raise ValueError("Exploration explicit depth selection differs from the complete source family")
        options["depth_family"] = family
    rebuilt = construct_moving_equipment_world(packet, bundle, contract.get("candidate_id"), deadline=stop, **options)
    expected_files, expected_contract = _inventory(packet, bundle, rebuilt, contract.get("candidate_id"), stop)
    if (files != expected_files or contract != expected_contract or scene != rebuilt
            or config.get("scene_sha256") != canonical_hash(scene)):
        raise ValueError("Exploration world or retained source/proposal contract differs from rederivation")
    source_check = verify_shared_world_sources(scene, run, deadline=stop)
    _check(stop)
    if read_package_file(run, "manifest.json", max_bytes=8 << 20) != manifest_raw:
        raise ValueError("Exploration manifest changed during admission")
    for name, signature in signatures.items():
        _check(stop)
        now = (Path(run) / name).lstat()
        if signature != (now.st_dev, now.st_ino, now.st_size, now.st_mtime_ns, now.st_ctime_ns):
            raise ValueError("Exploration source member changed during admission")
    return {"scene": scene, "source_check": source_check, "contract": contract, "config": deepcopy(config), "runtime": replay_runtime_identity(scene),
        "artifacts": [{"name": name, "bytes": len(raw)} for name, raw in files.items()]}


def create_session(run, config):
    from weft.lab import SharedWorldSession

    admitted = load_exploration_lab(run, config)

    class ExplorationSession(SharedWorldSession):
        control_profile = CONTROL_PROFILE

        def __init__(self):
            from weft.shared_world import SharedWorldEnv
            self.scene = deepcopy(admitted["scene"])
            self.source_binding_verification = admitted["source_check"]
            self.policy = None
            self.env = SharedWorldEnv(self.scene)
            self.scene = self.env.scene
            try:
                if self.scene != admitted["scene"]:
                    raise ValueError("Exploration runtime changed the admitted scene")
                self.reset(0)
            except BaseException:
                self.env.close()
                raise
            self.exploration_contract = admitted["contract"]
            self.replay_runtime = admitted["runtime"]
            self.artifacts = admitted["artifacts"]

        def reference_action(self):
            raise ValueError("Manual exploration has no admitted reference controller")

        def policy_action(self):
            raise ValueError("Manual exploration has no learned policy")

        def exploration_summary(self):
            return {"profile": PROFILE, "evaluation_status": "not_configured", "task_success_available": False,
                    "manual_only": True, "physical_validation": "unverified", "scope": SCOPE}

    return ExplorationSession()


_LOAD_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weft-exploration-load")


class LoadingExplorationLab:
    """One worker per open lab; repeated status polling cannot repeat admission."""

    def __init__(self, root, *, project):
        self.root, self.project = Path(root), project
        self.lock = threading.RLock()
        self.token = ""
        self._closed = False
        self._disposed = False
        self._executor = _LOAD_EXECUTOR
        self._future = self._executor.submit(self._load)
        self._future.add_done_callback(self._loaded)

    def _load(self):
        from weft.lab import Lab
        lab = Lab(self.root)
        try:
            lab.load_session()
            return lab
        except BaseException:
            lab.close()
            raise

    def _dispose(self, future):
        if not self._disposed and not future.cancelled() and future.exception() is None:
            self._disposed = True
            future.result().close()

    def _loaded(self, future):
        with self.lock:
            if self._closed:
                self._dispose(future)

    def _ready(self):
        if self._closed:
            raise ValueError("Exploration lab is closed")
        if not self._future.done():
            raise ValueError("Exploration source validation is still loading")
        return self._future.result()

    @property
    def loading(self):
        return not self._future.done()

    @property
    def recorder(self):
        return self._future.result().recorder if self._future.done() and not self._future.cancelled() and self._future.exception() is None else None

    @property
    def export_future(self):
        return self._future.result().export_future if self._future.done() and not self._future.cancelled() and self._future.exception() is None else None

    @property
    def avatars(self):
        return self._ready().avatars

    def summary(self):
        if not self._future.done():
            return {"status": "running", "profile": "shared-world", "control_profile": CONTROL_PROFILE,
                    "events": [{"stage": "source_validation", "message": "Checking saved source evidence and loading manual exploration"}],
                    "recording": False}
        try:
            return self._ready().summary()
        except Exception as error:
            return {"status": "failed", "profile": "shared-world", "control_profile": CONTROL_PROFILE,
                    "error": {"message": str(error)[:1000]}, "events": [], "recording": False}

    def control(self, path, data): return self._ready().control(path, data)
    def exercise_coverage(self): return {"status": "not_applicable"}
    def inspection(self): return self._ready().inspection() if self._future.done() else {"status": "loading"}
    def recording_inspection(self, name): return self._ready().recording_inspection(name)
    def recording_evidence(self, name): return self._ready().recording_evidence(name)

    def close(self):
        with self.lock:
            if self._closed:
                return
            self._closed = True
            if self._future.done() and not self._future.cancelled() and self._future.exception() is None:
                self._dispose(self._future)
            else:
                self._future.cancel()
            # The shared single worker bounds loading across cache eviction and portal reopen.


def export_exploration_session(directory, *, expected_contract_sha256):
    from weft.recording import export_recording
    from weft.shared_world_action_replay import verify_action_replay

    stop = _deadline(180)
    proof = verify_action_replay(directory, expected_contract_sha256=expected_contract_sha256, deadline=stop)
    exported = export_recording(directory, deadline=stop, max_output_bytes=128 << 20)
    manifest_path = Path(directory) / "manifest.json"
    manifest = _json(read_package_file(directory, "manifest.json", max_bytes=64 << 10))
    if any(manifest["files"].get(name) != pin for name, pin in proof["source_sha256"].items()):
        raise ValueError("Exploration journal changed between exact replay and export")
    manifest["shared_world_action_replay"] = proof
    from weft.recording import _atomic_json
    _atomic_json(manifest_path, manifest)
    _check(stop)
    return {**exported, "manifest_sha256": sha256(manifest_path.read_bytes()).hexdigest(), "action_replay": proof}
