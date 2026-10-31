"""Compare finer solver steps with a sealed fixed-action coupled-world baseline.

This trusted local diagnostic never reruns the baseline or generates controller
actions. Completion means measurements were saved, not numerical convergence,
task success, physical accuracy or permission to promote a solver setting.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, validate_package_layout, write_package_files


ROOT = Path(__file__).resolve().parents[2]
PROFILE = "shared-solver-resolution-diagnostic-v1"
BASE_PROFILE = "shared-exercise-reference-trial-v1"
DEFAULT_RUN = ROOT / "artifacts/reference-controller-trial-cli-01/comparison"
DEFAULT_SEAL = "ee5ce972b231c3850fa1454072d4ca3e78946e4eb3fb5e989c324a741c1757f1"
RESOLUTIONS = (32, 64)
MAX_HORIZON = 254
MAX_CONTROLS = 508
MAX_NATIVE_STEPS = 24384
MAX_METADATA_READ_BYTES = 64 << 20
MAX_RUNTIME_READ_BYTES = 8 << 20
MAX_OUTPUT_BYTES = 64 << 20
MAX_OUTPUT_FILES = 96
MAX_FILE_BYTES = 16 << 20
RESET = {"jitter": False, "jitter_scale": 0.}
RUNTIME_FILES = (
    "__init__.py", "io.py", "shared_exercise_task.py", "shared_exercise_reset.py",
    "shared_world.py", "shared_world_observation.py", "shared_world_evidence.py",
    "shared_world_packaging.py", "world_evidence.py", "source_equivalence.py",
    "observations.py", "person_inventory.py", "human.py", "mobile_human.py",
    "whole_body.py", "spatial_rigid.py", "spatial_pose.py", "body_frame.py", "root_motion.py", "errors.py",
)
SELECTED = (
    "plan.json", "report.json", "run/scene.json", "run/exercise-task.json", "run/manifest.json",
    "run/source-verification.json", "recordings/authored/manifest.json", "recordings/authored/scene.json",
    "recordings/authored/recording.json", "recordings/authored/events.jsonl", "proofs/authored-action-replay.json",
)
DIFF_PATHS = {"substeps", "action_contract/substeps_per_action", "coordinate_frame/clock_binding/physics_substeps"}
_SHA = re.compile(r"[0-9a-f]{64}")
_SCOPE = ("Fixed normalized actions at the original control rate; only native solver resolution changes. "
          "Saved source admission is inherited, not independently repeated. No policy transfer, source fit, "
          "training, media work, replay verification, default promotion or physical validation.")


def _require(condition, message):
    if not condition:
        raise ValueError("Solver resolution: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Solver-resolution cooperative deadline exceeded; partial artifacts retained")


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _path(value):
    value = Path(value).absolute()
    _require(".." not in value.parts and not any(p.is_symlink() for p in (value, *value.parents)), "unsafe path")
    return value


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _hash(value):
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _files(document, maximum=2048):
    _require(isinstance(document, dict) and type(document.get("schema_version")) is int
             and document["schema_version"] == 1, "invalid manifest")
    files = document.get("files")
    _require(isinstance(files, dict) and 1 <= len(files) <= maximum
             and all(_hash(value) for value in files.values()), "invalid manifest members")
    validate_package_layout(files)
    return files


class _Reader:
    def __init__(self, root, deadline):
        self.root, self.deadline, self.raw, self.bytes_read = root, deadline, {}, 0

    def read(self, name, expected=None):
        _check(self.deadline)
        if name not in self.raw:
            budget = MAX_METADATA_READ_BYTES // 2 - self.bytes_read
            _require(budget > 0, "metadata read budget exhausted")
            raw = read_package_file(self.root, name, max_bytes=min(MAX_FILE_BYTES, budget), expected_sha256=expected)
            self.bytes_read += len(raw)
            self.raw[name] = raw
        raw = self.raw[name]
        _require(expected is None or sha256(raw).hexdigest() == expected, "selected identity mismatch: " + name)
        _check(self.deadline)
        return raw

    def recheck(self, deadline):
        for name, raw in self.raw.items():
            _check(deadline)
            self.bytes_read += len(read_package_file(self.root, name, max_bytes=max(1, len(raw)),
                                                    expected_sha256=sha256(raw).hexdigest()))
            _require(self.bytes_read <= MAX_METADATA_READ_BYTES, "metadata readback budget exceeded")


def _versions():
    return {"python": sys.version.split()[0], "executable": sys.executable,
            "packages": {name: version(name) for name in ("numpy", "mujoco", "gymnasium", "opencv-contrib-python", "Pillow")}}


def _package_root():
    return ROOT / "src/weft"


def _runtime(reader, manifest, plan, deadline):
    pins = plan.get("runtime_sha256")
    _require(isinstance(pins, dict), "missing baseline runtime pins")
    current, total = {}, 0
    for name in RUNTIME_FILES:
        _check(deadline)
        expected = pins.get(name)
        archived = "runtime/weft/" + name
        _require(_hash(expected) and manifest.get(archived) == expected, "baseline runtime binding: " + name)
        saved = reader.read(archived, expected)
        raw = read_package_file(_package_root(), name, max_bytes=min(1 << 20, MAX_RUNTIME_READ_BYTES // 3 - total),
                                expected_sha256=expected)
        _require(raw == saved, "current implementation differs from baseline: " + name)
        current[name] = raw
        total += len(raw)
    versions = _versions()
    _require(versions == plan.get("runtime_versions"), "native package/Python versions differ from baseline")
    return current, versions


def _runtime_recheck(runtime, versions, deadline):
    for name, raw in runtime.items():
        _check(deadline)
        read_package_file(_package_root(), name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
    _require(_versions() == versions, "runtime versions changed")


def _vector(value, length):
    _require(isinstance(value, list) and len(value) == length and all(_finite(x) for x in value), "invalid state vector")
    return list(value)


def _physical(snapshot, scene):
    """Saved public coordinates cover the articulated and free-object degrees of freedom."""
    engine = snapshot["engine_state"]
    human = engine["human"]
    names = scene["action_contract"]["names"][:19]
    _require(len(names) == len(set(names)) == 19 and set(human["angles"]) == set(names)
             and set(human["angular_velocities"]) == set(names), "human coordinate identity changed")
    items = engine["objects"]
    _require(isinstance(items, list) and len(items) == len(scene["objects"]) <= 12
             and len({row["id"] for row in items}) == len(items)
             and {row["id"] for row in items} == {row["id"] for row in scene["objects"]}, "object identity changed")
    objects = {}
    for item in items:
        objects[item["id"]] = {key: _vector(item[key], length) for key, length in
                               (("position", 3), ("quaternion", 4), ("velocity", 3), ("angular_velocity", 3))}
        _require(abs(sum(x * x for x in item["quaternion"]) - 1.) <= 1e-8, "nonunit object quaternion")
    joints = human["joints"]
    _require(isinstance(joints, dict) and 1 <= len(joints) <= 32, "invalid named joint positions")
    return {"angles": _vector([human["angles"][key] for key in names], 19),
            "angular_velocities": _vector([human["angular_velocities"][key] for key in names], 19),
            "root_translation_model_units": _vector(human["mobility"]["translation"], 3),
            "root_velocity_model_units_s": _vector(human["mobility"]["linear_velocity"], 3),
            "joint_positions_world": {key: _vector(value, 3) for key, value in joints.items()}, "objects": objects}


def _state_identity(snapshot, scene, task, *, index, identities, complete_boundary=True):
    metrics, engine = snapshot["task_metrics"], snapshot["engine_state"]
    for document in (snapshot, metrics):
        _require((document["scene_sha256"], document["task_sha256"]) == identities,
                 "snapshot scene/task binding mismatch")
    _require(metrics["gates_sha256"] == task["gates_sha256"] and snapshot["resolved_reset_options"] == RESET,
             "snapshot gates/reset mismatch")
    if complete_boundary:
        _require(snapshot["engine_state_status"] == "current" and engine["steps"] == index
                 and engine["substeps"] == scene["substeps"] and metrics["observed_substeps"] == index * scene["substeps"]
                 and metrics["valid_substeps"] == metrics["observed_substeps"] and metrics["clock_status"] == "observed",
                 "snapshot is not a complete observed control boundary")
        for value in (metrics["simulated_duration_s"], engine["time_s"]):
            _require(_finite(value) and abs(value - index * scene["dt"]) <= 1e-8, "control clock mismatch")
        _require(abs(engine["source_time_s"] - scene["coordinate_frame"]["clock_binding"]["source_time_offset_s"]
                     - index * scene["dt"]) <= 1e-8, "source clock mismatch")


def _load_baseline(reader, expected):
    _require(_hash(expected), "an exact expected comparison seal is required")
    manifest_raw = reader.read("manifest.json", expected)
    outer = _json(manifest_raw)
    files = _files(outer)
    _require(outer.get("profile") == BASE_PROFILE and outer.get("status") == "completed", "baseline is not completed")
    values = {}
    for name in SELECTED:
        _require(name in files, "missing selected sealed member: " + name)
        raw = reader.read(name, files[name])
        if not name.endswith(".jsonl"):
            values[name] = _json(raw)
    scene, task, plan = (values[name] for name in ("run/scene.json", "run/exercise-task.json", "plan.json"))
    horizon = task["interval"]["horizon_steps"]
    _require(type(horizon) is int and 1 <= horizon <= MAX_HORIZON and scene["task"]["horizon_steps"] == horizon,
             "baseline horizon exceeds the 508-control/24,384-native-step study bound")
    _require(scene["profile"] in ("shared-world-v1", "shared-world-v2") and scene["substeps"] == 16
             and type(scene["substeps"]) is int and scene["dt"] == 1 / 30
             and scene["action_contract"]["substeps_per_action"] == 16
             and scene["coordinate_frame"]["clock_binding"]["physics_substeps"] == 16,
             "baseline must be a 30 Hz coupled-world scene with 16 substeps")
    _require(task["scene_sha256"] == canonical_hash(scene) and outer["scene_sha256"] == canonical_hash(scene)
             and outer["task_sha256"] == canonical_hash(task) and task["control"]["object_forces"] == "forbidden",
             "baseline scene/task/action binding mismatch")
    _require(scene["controlled_id"] == "human" and task["interval"]["start_offset_s"] == 0., "unsupported baseline interval/actor")
    _require(plan.get("profile") == BASE_PROFILE + "-plan" and plan.get("status") == "frozen"
             and plan["reset_options"] == RESET and plan["horizon_steps_each"] == horizon,
             "unsupported baseline plan/reset")
    seed = plan["seed"]
    _require(type(seed) is int and 0 <= seed <= 2**31 - 1, "invalid baseline seed")
    source_files = _files(values["run/manifest.json"])
    _require(all(files.get("run/" + name) == value for name, value in source_files.items()),
             "inherited source package differs from the completed comparison seal")
    _require(values["run/source-verification.json"].get("mode") in ("source_qualified", "analytic_fixture"),
             "missing saved source-admission receipt")
    recording_files = _files(values["recordings/authored/manifest.json"], 6)
    _require(set(recording_files) == {"scene.json", "recording.json", "events.jsonl", "video.mp4", "preview.gif", "replay.html"}
             and all(files.get("recordings/authored/" + k) == v for k, v in recording_files.items()),
             "recording seal differs from completed comparison")
    recording_scene, meta = values["recordings/authored/scene.json"], values["recordings/authored/recording.json"]
    _require(recording_scene.get("profile") == "shared-exercise-recording-v1"
             and canonical_hash(recording_scene["engine_scene"]) == canonical_hash(scene)
             and canonical_hash(recording_scene["exercise_task"]) == canonical_hash(task)
             and meta["scene_sha256"] == canonical_hash(recording_scene) and meta["status"] == "closed"
             and meta["seed"] == seed and meta["events"] == horizon + 1,
             "recording metadata differs from exact original scene/task")
    proof = values["proofs/authored-action-replay.json"]
    _require(proof["exact_state_replay"] is True and proof["steps"] == horizon and proof["events"] == horizon + 1
             and proof["scene_sha256"] == canonical_hash(scene) and proof["task_sha256"] == canonical_hash(task),
             "baseline lacks exact saved replay proof")
    journal = reader.raw["recordings/authored/events.jsonl"]
    _require(len(journal) == meta["journal_bytes"] and journal.endswith(b"\n"), "recording journal size/truncation mismatch")
    lines = journal.splitlines()
    _require(len(lines) == horizon + 1 and all(0 < len(line) <= 128 << 10 for line in lines), "journal row bound exceeded")
    events = [_json(line) for line in lines]
    identities = (canonical_hash(scene), canonical_hash(task))
    count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    _require(events[0]["type"] == "start" and events[0]["seed"] == seed, "recording initial event mismatch")
    for index, event in enumerate(events):
        _check(reader.deadline)
        _require(type(event["index"]) is int and event["index"] == index, "event order mismatch")
        state = event["snapshot"]
        _state_identity(state, scene, task, index=index, identities=identities)
        _physical(state, scene)
        if index:
            action = _vector(event["action"], count)
            _require(event["type"] == "step" and event["controller"] == "authored" and max(map(abs, action)) <= 1
                     and state["last_action"] == action and _finite(event["reward"])
                     and type(event["terminated"]) is bool and type(event["truncated"]) is bool,
                     "invalid literal baseline action or event")
            _require(event["terminated"] == state["terminated"] and event["truncated"] == state["truncated"]
                     and ((event["terminated"] or event["truncated"]) == (index == horizon)), "early/nonterminal baseline")
    result = values["report.json"]
    trials = result.get("trials")
    _require(isinstance(trials, list) and len(trials) == 2, "unexpected baseline trial count")
    rows = [row for row in trials if row.get("mode") == "authored"]
    _require(len(rows) == 1, "ambiguous authored baseline")
    authored = rows[0]
    _require(result["status"] == "completed" and result["seed"] == seed and result["same_initial_state"] is True
             and authored["status"] == "verified" and authored["steps"] == horizon
             and authored["recording"]["manifest_sha256"] == files["recordings/authored/manifest.json"]
             and authored["initial_state_sha256"] == canonical_hash(events[0]["snapshot"])
             and authored["final_state_sha256"] == canonical_hash(events[-1]["snapshot"])
             and canonical_hash(authored["metrics"]) == canonical_hash(events[-1]["snapshot"]["task_metrics"])
             and authored["action_replay"] == dict(proof, path="proofs/authored-action-replay.json", status="passed")
             and authored["metrics"]["completed_interval"] is True, "baseline report/proof/state inconsistency")
    return SimpleNamespace(scene=scene, task=task, plan=plan, files=files, events=events, seed=seed, horizon=horizon)


def _diff(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b):
        return set().union(*(_diff(a[k], b[k], prefix + "/" + k if prefix else k) for k in a)) if a else set()
    return set() if canonical_hash(a) == canonical_hash(b) else {prefix}


def make_variant(scene, task, resolution):
    _require(type(resolution) is int and resolution in RESOLUTIONS, "unsupported solver resolution")
    derived, rebound = deepcopy(scene), deepcopy(task)
    derived["substeps"] = resolution
    derived["action_contract"]["substeps_per_action"] = resolution
    derived["coordinate_frame"]["clock_binding"]["physics_substeps"] = resolution
    rebound["scene_sha256"] = canonical_hash(derived)
    _require(_diff(scene, derived) == DIFF_PATHS and _diff(task, rebound) == {"scene_sha256"}, "variant changed non-solver fields")
    return derived, rebound


def _distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _quaternion_distance(a, b):
    a = [x / math.sqrt(sum(y * y for y in a)) for x in a]
    b = [x / math.sqrt(sum(y * y for y in b)) for x in b]
    minus, plus = _distance(a, b), _distance(a, [-x for x in b])
    return 4 * math.atan2(min(minus, plus), max(minus, plus))


def compare_boundaries(left, right, scene):
    """Separate units and sampled measurements; no combined score or convergence threshold."""
    a, b = _physical(left, scene), _physical(right, scene)
    _require(set(a["joint_positions_world"]) == set(b["joint_positions_world"]), "joint position identity changed")
    objects = {}
    for key in a["objects"]:
        first, second = a["objects"][key], b["objects"][key]
        objects[key] = {"position_world_units": _distance(first["position"], second["position"]),
                        "velocity_world_units_s": _distance(first["velocity"], second["velocity"]),
                        "orientation_rad": _quaternion_distance(first["quaternion"], second["quaternion"]),
                        "angular_velocity_rad_s": _distance(first["angular_velocity"], second["angular_velocity"])}
    penetration = [item["task_metrics"].get("max_floor_penetration_model_units") for item in (left, right)]
    metrics = {}
    for key in ("angular_rmse_rad", "root_rmse_model_units", "peak_angular_rmse_rad", "peak_root_rmse_model_units", "mean_reward"):
        values = [item["task_metrics"].get(key) for item in (left, right)]
        metrics[key] = abs(values[0] - values[1]) if all(_finite(x) for x in values) else None
    return {"joint_angle_max_abs_rad": max(abs(x - y) for x, y in zip(a["angles"], b["angles"], strict=True)),
            "joint_velocity_max_abs_rad_s": max(abs(x - y) for x, y in zip(a["angular_velocities"], b["angular_velocities"], strict=True)),
            "root_position_model_units": _distance(a["root_translation_model_units"], b["root_translation_model_units"]),
            "root_velocity_model_units_s": _distance(a["root_velocity_model_units_s"], b["root_velocity_model_units_s"]),
            "joint_position_max_world_units": max(_distance(value, b["joint_positions_world"][key]) for key, value in a["joint_positions_world"].items()),
            "cumulative_floor_penetration_difference_model_units": abs(penetration[0] - penetration[1]) if all(_finite(x) for x in penetration) else None,
            "equipment": objects, "task_metric_absolute_differences": metrics}


def _maximum(rows):
    if not rows:
        return None
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        result[key] = _maximum(values) if isinstance(values[0], dict) else max(values) if all(_finite(v) for v in values) else None
    return result


def _engine_api():
    # No native import or model construction occurs before the recorded runtime is authenticated.
    import mujoco
    from weft.shared_exercise_task import SharedExerciseTaskEnv
    return SimpleNamespace(Env=SharedExerciseTaskEnv, native=mujoco)


@contextmanager
def _native_budget(api, counts, deadline):
    original_step, original_reset = api.native.mj_step, api.native.mj_resetData
    def step(model, data, *args, **kwargs):
        _check(deadline)
        _require(not args and not kwargs, "batched/native step arguments are forbidden")
        _require(counts["active_resolution"] in RESOLUTIONS and counts["native_steps"] < MAX_NATIVE_STEPS,
                 "native step work bound exceeded")
        counts["native_steps"] += 1
        return original_step(model, data)
    def reset(model, data):
        _check(deadline)
        _require(counts["native_resets"] < 2, "native reset work bound exceeded")
        counts["native_resets"] += 1
        return original_reset(model, data)
    api.native.mj_step, api.native.mj_resetData = step, reset
    try:
        yield
    finally:
        api.native.mj_step, api.native.mj_resetData = original_step, original_reset


class _Output:
    def __init__(self, root):
        self.root, self.sizes, self.protected = root, {}, {}

    def reserve(self, name, count):
        _require(len(self.sizes) + (name not in self.sizes) < MAX_OUTPUT_FILES, "output file bound exceeded")
        _require(count <= MAX_FILE_BYTES and sum(self.sizes.values()) - self.sizes.get(name, 0) + count <= MAX_OUTPUT_BYTES - (256 << 10),
                 "output byte bound exceeded")
        self.sizes[name] = count

    def write(self, name, raw):
        _require(name not in self.sizes, "output member already exists")
        self.reserve(name, len(raw))
        write_package_files(self.root, {name: raw})
        self.protected[name] = sha256(raw).hexdigest()

    @contextmanager
    def journal(self, name):
        self.reserve(name, 0)
        fd = os.open(self.root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            def append(event):
                raw = _encode(event)
                _require(len(raw) <= 128 << 10, "output event row bound exceeded")
                self.reserve(name, self.sizes[name] + len(raw))
                stream.write(raw)
                stream.flush()
            yield append

    def seal(self, status, deadline):
        found, pending, directories = {}, [self.root], 0
        while pending:
            _check(deadline)
            directory = pending.pop()
            directories += 1
            _require(directories <= MAX_OUTPUT_FILES, "output directory bound exceeded")
            with os.scandir(directory) as entries:
                for entry in entries:
                    _check(deadline)
                    _require(not entry.is_symlink(), "unexpected output symlink")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                        _require(directories + len(pending) <= MAX_OUTPUT_FILES, "output directory bound exceeded")
                        continue
                    _require(entry.is_file(follow_symlinks=False), "output must contain regular files")
                    name = Path(entry.path).relative_to(self.root).as_posix()
                    _require(name in self.sizes and len(found) < MAX_OUTPUT_FILES, "unexpected output member")
                    found[name] = entry.stat(follow_symlinks=False).st_size
        _require(found == self.sizes, "output members or sizes changed before sealing")
        files = {}
        for name, size in self.sizes.items():
            _check(deadline)
            raw = read_package_file(self.root, name, max_bytes=max(1, size))
            digest = sha256(raw).hexdigest()
            _require(len(raw) == size and (name not in self.protected or self.protected[name] == digest), "output changed before sealing")
            files[name] = digest
        raw = _encode({"schema_version": 1, "profile": PROFILE, "status": status, "files": files,
                       "physical_validation": "unverified"})
        _require(len(files) + 1 <= MAX_OUTPUT_FILES and sum(self.sizes.values()) + len(raw) <= MAX_OUTPUT_BYTES, "seal bound exceeded")
        write_package_files(self.root, {"manifest.json": raw})
        return sha256(raw).hexdigest()


def run_study(run, out, *, expected_manifest_sha256, max_seconds=30):
    """Run off UI threads; the caller owns hard process termination and native-library custody."""
    _require(_finite(max_seconds) and 1 <= max_seconds <= 30, "cooperative bound must be 1–30 seconds")
    run, out = _path(run), _path(out)
    _require(out != run and not out.is_relative_to(run) and not run.is_relative_to(out), "output overlaps immutable baseline")
    if out.exists():
        raise FileExistsError("Solver-resolution study requires a new output directory")
    started = time.monotonic()
    deadline = started + max_seconds
    work_deadline = deadline - min(3., max_seconds * .15)
    out.mkdir(parents=True, exist_ok=False)
    output, reader = _Output(out), _Reader(run, work_deadline)
    counts = {"live_controls": 0, "native_steps": 0, "native_resets": 0, "environments": 0, "active_resolution": None}
    report = {"schema_version": 1, "profile": PROFILE, "status": "failed", "created_at": local_time(),
              "baseline_manifest_sha256": expected_manifest_sha256, "scope": _SCOPE, "physical_validation": "unverified",
              "promoted": False, "counts": counts, "variants": [], "comparisons": {}, "error": None,
              "baseline_substep_traces": None, "baseline_native_qpos_qvel_arrays": None,
              "limits": {"seconds": max_seconds, "new_controls": MAX_CONTROLS, "native_steps": MAX_NATIVE_STEPS,
                         "one_live_environment": True, "metadata_read_bytes": MAX_METADATA_READ_BYTES,
                         "runtime_read_bytes": MAX_RUNTIME_READ_BYTES, "output_bytes": MAX_OUTPUT_BYTES,
                         "output_files": MAX_OUTPUT_FILES, "hard_rss_cap": False, "hard_process_timeout_owned_by_caller": True}}
    env = None
    try:
        baseline = _load_baseline(reader, expected_manifest_sha256)
        runtime, versions = _runtime(reader, baseline.files, baseline.plan, work_deadline)
        evaluator = sha256(runtime["shared_exercise_task.py"]).hexdigest()
        _require(all(event["snapshot"]["task_metrics"]["evaluator_sha256"] == evaluator for event in baseline.events),
                 "baseline evaluator differs from current authenticated scorer")
        report.update(scene_sha256=canonical_hash(baseline.scene), task_sha256=canonical_hash(baseline.task),
                      seed=baseline.seed, horizon_steps=baseline.horizon, baseline_metrics=baseline.events[-1]["snapshot"]["task_metrics"])
        for name, raw in reader.raw.items():
            output.write("baseline/" + name, raw)
        for name, raw in runtime.items():
            output.write("runtime/weft/" + name, raw)
        script = read_package_file(Path(__file__).parent, Path(__file__).name, max_bytes=128 << 10)
        output.write("runtime/check_shared_solver_resolution.py", script)
        plan = {"schema_version": 1, "profile": PROFILE + "-plan", "status": "frozen", "scope": _SCOPE,
                "baseline_directory": str(run), "baseline_manifest_sha256": expected_manifest_sha256,
                "selected_input_sha256": {k: sha256(v).hexdigest() for k, v in reader.raw.items()},
                "runtime_sha256": {k: sha256(v).hexdigest() for k, v in runtime.items()}, "runtime_versions": versions,
                "script_sha256": sha256(script).hexdigest(), "source_admission": "inherited_exact_completed_comparison_seal",
                "source_media_read": False, "baseline_rerun": False, "resolutions": list(RESOLUTIONS),
                "normalized_actions_sha256": canonical_hash([row["action"] for row in baseline.events[1:]]),
                "scene_diff_whitelist": sorted(DIFF_PATHS), "task_diff_whitelist": ["scene_sha256"],
                "limits": report["limits"], "comparison_scope": "Common observed control boundaries; no inferred old within-step trace"}
        output.write("plan.json", _encode(plan))
        _runtime_recheck(runtime, versions, work_deadline)
        _check(work_deadline)
        api = _engine_api()
        initial_physical = canonical_hash(_physical(baseline.events[0]["snapshot"], baseline.scene))
        previous = None
        initial_native = None
        with _native_budget(api, counts, work_deadline):
            for resolution in RESOLUTIONS:
                _check(work_deadline)
                scene, task = make_variant(baseline.scene, baseline.task, resolution)
                prefix = f"substeps-{resolution}"
                output.write(prefix + "/scene.json", _encode(scene))
                output.write(prefix + "/exercise-task.json", _encode(task))
                row = {"substeps": resolution, "status": "started", "controls": 0, "native_steps": 0,
                       "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "initial_physical_state_matches": False}
                report["variants"].append(row)
                counts["active_resolution"] = resolution
                native_start = counts["native_steps"]
                counts["environments"] += 1
                env = api.Env(scene, task)
                _check(work_deadline)
                _require(canonical_hash(env.scene) == canonical_hash(scene) and canonical_hash(env.task) == canonical_hash(task),
                         "native admission rewrote a variant")
                env.reset(seed=baseline.seed, options=dict(RESET))
                initial = env.snapshot()
                identities = (canonical_hash(scene), canonical_hash(task))
                _state_identity(initial, scene, task, index=0, identities=identities)
                row["initial_physical_state_matches"] = canonical_hash(_physical(initial, scene)) == initial_physical
                _require(row["initial_physical_state_matches"], "solver variant changed initial positions or velocities")
                native = {"qpos": env._engine.data.qpos.tolist(), "qvel": env._engine.data.qvel.tolist()}
                _require(all(_finite(x) for values in native.values() for x in values), "nonfinite initial native state")
                if initial_native is None:
                    initial_native = native
                _require(canonical_hash(native) == canonical_hash(initial_native), "variants have different initial qpos/qvel")
                row["initial_native_state_sha256"] = canonical_hash(native)
                output.write(prefix + "/initial-native-state.json", _encode(native))
                samples, errors, cross_errors, comparisons, cross_comparisons = [], [], [], [], []
                with output.journal(prefix + "/events.jsonl") as append:
                    append({"index": 0, "type": "start", "seed": baseline.seed, "snapshot": initial})
                    for index, saved in enumerate(baseline.events[1:], 1):
                        _check(work_deadline)
                        _require(counts["live_controls"] < MAX_CONTROLS, "control work bound exceeded")
                        action = deepcopy(saved["action"])
                        before = counts["native_steps"]
                        counts["live_controls"] += 1
                        row["controls"] += 1
                        _, reward, terminated, truncated, _ = env.step(action)
                        _require(counts["native_steps"] - before <= resolution, "one action exceeded declared solver resolution")
                        state = env.snapshot()
                        _require(_finite(reward) and type(terminated) is bool and type(truncated) is bool
                                 and state["last_action"] == saved["action"], "invalid action/result")
                        current = (state["engine_state_status"] == "current" and state["task_metrics"]["observed_substeps"] == index * resolution)
                        _state_identity(state, scene, task, index=index, identities=identities, complete_boundary=current)
                        _require(current or terminated or truncated, "partial nonterminal control boundary")
                        append({"index": index, "type": "step", "controller": "literal_saved_authored_action", "action": action,
                                "reward": float(reward), "terminated": terminated, "truncated": truncated,
                                "common_control_boundary": current, "snapshot": state})
                        samples.append(state if current else None)
                        if current:
                            errors.append(compare_boundaries(saved["snapshot"], state, scene))
                            comparisons.append({"index": index, "nominal_control_time_s": index * scene["dt"],
                                "saved_native_time_s": saved["snapshot"]["task_metrics"]["simulated_duration_s"],
                                "variant_native_time_s": state["task_metrics"]["simulated_duration_s"], "differences": errors[-1],
                                "saved_failed_gates": saved["snapshot"]["task_metrics"]["failed_gates"],
                                "variant_failed_gates": state["task_metrics"]["failed_gates"]})
                            if previous is not None and index <= len(previous) and previous[index - 1] is not None:
                                cross_errors.append(compare_boundaries(previous[index - 1], state, scene))
                                cross_comparisons.append({"index": index, "differences": cross_errors[-1],
                                    "substeps_32_failed_gates": previous[index - 1]["task_metrics"]["failed_gates"],
                                    "substeps_64_failed_gates": state["task_metrics"]["failed_gates"]})
                        if terminated or truncated:
                            break
                row.update(status="observed", native_steps=counts["native_steps"] - native_start,
                           metrics=env.metrics(), terminated=bool(terminated), truncated=bool(truncated),
                           common_control_boundaries=len(errors), unavailable_control_boundaries=baseline.horizon - len(errors),
                           maximum_difference_from_saved_16=_maximum(errors), final_state_sha256=canonical_hash(state))
                _require(terminated or truncated, "variant failed to terminate by original horizon")
                _require(row["native_steps"] <= baseline.horizon * resolution, "per-variant native work exceeded")
                report["comparisons"][f"16_to_{resolution}"] = {"common_boundaries": len(errors), "maximum_differences": _maximum(errors),
                    "baseline_final_failed_gates": report["baseline_metrics"]["failed_gates"],
                    "variant_final_failed_gates": row["metrics"]["failed_gates"], "convergence_certified": False}
                output.write(prefix + "/comparison-to-16.json", _encode(comparisons))
                if previous is not None:
                    report["comparisons"]["32_to_64"] = {"common_boundaries": len(cross_errors), "maximum_differences": _maximum(cross_errors),
                                                           "convergence_certified": False}
                    output.write("comparison-32-to-64.json", _encode(cross_comparisons))
                output.write(prefix + "/report.json", _encode(row))
                env.close()
                env = None
                previous = samples
                counts["active_resolution"] = None
        reader.recheck(deadline)
        _runtime_recheck(runtime, versions, deadline)
        read_package_file(Path(__file__).parent, Path(__file__).name, max_bytes=len(script), expected_sha256=sha256(script).hexdigest())
        report["readback"] = {"selected_files": len(reader.raw), "metadata_bytes": reader.bytes_read,
                              "runtime_bytes": 3 * sum(map(len, runtime.values())), "source_admission_recomputed": False,
                              "scope": "Selected files only; library/import I/O and unrelated source/media bytes are excluded"}
        report["status"] = "completed"
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
        if report["variants"] and report["variants"][-1]["status"] == "started":
            report["variants"][-1].update(status="failed", native_steps=counts["native_steps"] - native_start)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                report["cleanup_error"] = str(error)[:1000]
        counts["active_resolution"] = None
    report["seconds"] = time.monotonic() - started
    output.write("report.json", _encode(report))
    try:
        report["manifest_sha256"] = output.seal(report["status"], deadline)
    except Exception as error:
        output.write("finalization-error.json", _encode({"type": type(error).__name__, "message": str(error)[:1000]}))
        report["status"] = "failed"
    return dict(report, directory=str(out))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/shared-solver-resolution-01")
    parser.add_argument("--expected-manifest-sha256", default=DEFAULT_SEAL)
    parser.add_argument("--max-seconds", type=float, default=30.)
    args = parser.parse_args(argv)
    result = run_study(args.run, args.out, expected_manifest_sha256=args.expected_manifest_sha256, max_seconds=args.max_seconds)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
