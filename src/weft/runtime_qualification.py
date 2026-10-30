"""Bounded cross-profile execution evidence; neither policy training nor physical validation."""

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import time

import numpy as np

from weft.capabilities import describe_capabilities
from weft.io import canonical_hash, local_time
from weft.recording import MAX_EVENT_BYTES, Recorder, load_recording
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


PROFILE = "weft-runtime-qualification-v1"
MAX_CASES, MAX_STEPS, MAX_SECONDS = 5, 12000, 120
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_INPUT_BYTES, MAX_INPUT_FILES = 256 * 1024 * 1024, 4096
MAX_JSON_BYTES = 8 * 1024 * 1024
PROFILES = frozenset({"rolling", "planar-rigid-v1", "planar-rigid-v2", "spatial-rigid-v1",
    "exercise-v1", "exercise-v2", "exercise-v3", "exercise-imitation-v1", "exercise-feet-posture-v1",
    "shared-world-v1", "shared-world-v2"})
INPUT_JSON = frozenset({"scene.json", "human.json", "world.json", "task.json", "exercise-task.json",
    "exercise-lab.json", "policy.json", "settings.json", "foot-reference.json", "source.json",
    "report.json", "reconstruction.json", "status.json"})
RUNTIME_FILES = ("runtime_qualification.py", "capabilities.py", "recording.py", "env.py", "world.py",
    "scene.py", "spatial_rigid.py", "human.py", "spatial_pose.py", "body_frame.py", "mobile_human.py",
    "root_motion.py", "whole_body.py", "foot_body.py", "foot_sandbox.py", "shared_world.py",
    "shared_world_observation.py", "shared_exercise_task.py", "shared_exercise_reset.py",
    "shared_exercise_recording.py", "shared_world_packaging.py", "shared_world_evidence.py",
    "source_equivalence.py", "observations.py", "world_evidence.py", "io.py", "errors.py",
    "shared_exercise_lab.py", "shared_exercise_policies.py", "shared_exercise_learning.py",
    "shared_exercise_residual_learning.py", "shared_exercise_feasibility_learning.py",
    "shared_exercise_feasibility_evaluation.py", "spatial_learning.py", "shared_world_learning.py",
    "foot_reference.py", "whole_body_feasibility.py", "camera_body.py", "camera_ground.py",
    "camera_pose.py", "multiview_body.py", "multiview_root.py", "brand.py")
_TERMINAL_REJECTIONS = {"failed", "error", "unsupported", "unsupported_input", "rejected", "cancelled",
                        "running", "queued", "construction_only"}


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    def invalid(_):
        raise ValueError("Nonfinite JSON value")

    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid(value)
        return parsed

    try:
        return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid, parse_float=number)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Qualification requires bounded strict JSON") from error


def _bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Qualification exceeded its cooperative wall-time bound")


def _path(path):
    path = Path(os.path.abspath(path))
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("Qualification paths cannot traverse symbolic links")
    return path


def _runtime():
    package = Path(__file__).parent
    return {"files": {name: _sha(read_package_file(package, name, max_bytes=MAX_JSON_BYTES))
                       for name in RUNTIME_FILES},
            "versions": {"python": platform.python_version(),
                         **{name: version(name) for name in ("numpy", "mujoco", "pymunk", "gymnasium")}}}


def _runtime_pins(document):
    if not isinstance(document, dict) or "runtime_sha256" not in document:
        return
    pins = document["runtime_sha256"]
    if not isinstance(pins, dict) or not pins or len(pins) > 128:
        raise ValueError("Invalid declared runtime pins")
    for name, expected in pins.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z_]+\.py", name):
            raise ValueError("Unsupported declared runtime path")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("Declared runtime pins require exact SHA256 values")
        read_package_file(Path(__file__).parent, name, max_bytes=MAX_JSON_BYTES, expected_sha256=expected)
    if "runtime_versions" in document:
        from weft.shared_exercise_lab import _runtime_versions
        if document["runtime_versions"] != _runtime_versions():
            raise ValueError("Declared runtime dependency versions differ")


def _task_document(document, scene):
    if document is None:
        return
    if any(document.get(key) != value for key, value in scene["task"].items()):
        raise ValueError("Sealed task document differs from the executable task")
    if any(key in document and document[key] != scene.get(key) for key in ("profile", "controlled_id", "target")):
        raise ValueError("Sealed task target differs from the executable scene")


@dataclass
class _Input:
    root: Path
    manifest: bytes
    hashes: dict
    raw: dict
    docs: dict
    config: dict | None
    outer: dict
    kind: str
    scene: dict | None
    human: dict | None
    task: dict | None
    settings: dict
    identity: dict


def _admit(path, deadline):
    root = _path(path)
    config = None
    outer = {}
    if (root / "lab-config.json").exists():
        outer = {name: read_package_file(root, name, max_bytes=MAX_JSON_BYTES)
                 for name in ("lab-config.json", "status.json")}
        config, status = _json(outer["lab-config.json"]), _json(outer["status.json"])
        if not isinstance(config, dict) or status.get("status") != "ready":
            raise ValueError("Qualification requires a completed ready lab")
        root = _path(root / "run")
    manifest, hashes = read_package_manifest(root)
    header = _json(manifest)
    if header.get("status") in _TERMINAL_REJECTIONS or header.get("simulation_ready") is False:
        raise ValueError("Rejected or incomplete packages cannot enter runtime qualification")
    if len(hashes) > MAX_INPUT_FILES:
        raise ValueError("Qualification input exceeds its file-count bound")
    if "construction-seed.json" in hashes:
        raise ValueError("Construction-only scaffolding is not an admitted runtime")
    raw, total = {}, len(manifest)
    for name, expected in hashes.items():
        _deadline(deadline)
        content = read_package_file(root, name, max_bytes=min(64 * 1024 * 1024, max(1, MAX_INPUT_BYTES - total)),
                                    expected_sha256=expected)
        total += len(content)
        if total > MAX_INPUT_BYTES:
            raise ValueError("Qualification input exceeds its byte bound")
        if name in INPUT_JSON:
            if len(content) > MAX_JSON_BYTES:
                raise ValueError("Qualification input JSON exceeds its byte bound")
            raw[name] = content
    docs = {name: _json(value) for name, value in raw.items()}
    if any(not isinstance(doc, dict) for doc in docs.values()):
        raise ValueError("Qualification runtime documents must be JSON objects")
    for name in ("report.json", "reconstruction.json", "status.json"):
        document = docs.get(name, {})
        if document.get("status") in _TERMINAL_REJECTIONS or document.get("simulation_ready") is False:
            raise ValueError("Rejected or incomplete input evidence cannot enter runtime qualification")
    for document in docs.values():
        _runtime_pins(document)
    if config and config.get("manifest_sha256", _sha(manifest)) != _sha(manifest):
        raise ValueError("Lab configuration differs from its sealed run")
    scene, human, task = docs.get("scene.json"), docs.get("human.json"), None
    settings = {}
    policy = docs.get("policy.json", {})
    source_proof = None
    if scene and scene.get("profile") in {"shared-world-v1", "shared-world-v2"}:
        from weft.shared_world import validate_shared_world
        from weft.shared_world_packaging import collect_shared_source_files
        scene = validate_shared_world(scene)
        source_proof = collect_shared_source_files(scene, root, expected_manifest_sha256=_sha(manifest)).report
        if "exercise-lab.json" in docs:
            from weft.shared_exercise_lab import load_shared_exercise_lab
            loaded = load_shared_exercise_lab(root, config=config, deadline=deadline)
            scene, task = loaded["scene"], loaded["task"]
        elif "exercise-task.json" in docs:
            from weft.shared_exercise_task import validate_shared_exercise_task
            task = validate_shared_exercise_task(docs["exercise-task.json"], scene)
            if policy:
                # Policy admission includes the original sealed training closure, not just a JSON actor.
                from weft.shared_exercise_lab import _validate_policy_package
                learning_files = {name: read_package_file(root, name, max_bytes=64 * 1024 * 1024,
                                  expected_sha256=expected) for name, expected in hashes.items()}
                _validate_policy_package(manifest, learning_files, scene, task)
        elif policy:
            from weft.shared_world_learning import validate_shared_policy
            validate_shared_policy(policy, scene)
        kind = "shared_exercise" if task is not None else "shared_world"
    elif scene and scene.get("profile") in {"planar-rigid-v1", "planar-rigid-v2"}:
        from weft.scene import validate_scene
        scene, kind = validate_scene(scene), "planar"
    elif scene and scene.get("profile") == "spatial-rigid-v1":
        from weft.spatial_rigid import validate_scene
        scene, kind = validate_scene(scene), "spatial"
        if policy:
            from weft.spatial_learning import validate_spatial_policy
            validate_spatial_policy(policy, scene)
    elif human is not None:
        from weft.human import validate_human, joint_contract
        human = validate_human(human)
        if "foot-reference.json" in docs:
            from weft.foot_body import validate_reference
            validate_reference(human, docs["foot-reference.json"])
            kind = "feet"
            settings["horizon_steps"] = scene["task"]["horizon_steps"] if scene else 180
        elif (scene and scene.get("profile") == "exercise-imitation-v1") or policy.get("algorithm") == "whole_body_cem_v1":
            kind = "whole_body"
            partition = docs.get("settings.json", {}).get("partition", {})
            starts = partition.get("evaluation_start_times_s")
            if not isinstance(starts, list) or not starts:
                raise ValueError("Imitation input must preserve its selected evaluation interval")
            settings = {"start_time_s": starts[0], "horizon_steps": partition.get("horizon_steps")}
            if policy and policy.get("spec_sha256") != canonical_hash(human):
                raise ValueError("Whole-body policy body identity differs")
        else:
            kind = "human"
            selected = docs.get("task.json", {}).get("selected_joint", policy.get("selected_joint", "left_knee"))
            if selected not in joint_contract(human)[0]:
                raise ValueError("Unknown selected human joint")
            settings["selected_joint"] = selected
        if scene and scene.get("profile") not in {human["profile"], "exercise-imitation-v1", "exercise-feet-posture-v1"}:
            raise ValueError("Body and scene profiles disagree")
    elif "world.json" in docs:
        from weft.world import World
        World.from_dict(docs["world.json"])
        kind = "rolling"
    else:
        raise ValueError("Unsupported sealed runtime package")
    expected_lab = {"rolling": {"rolling"}, "planar": {"planar", "appearance"}, "spatial": {"spatial-rigid"},
                    "human": {"exercise"}, "whole_body": {"exercise"}, "feet": {"exercise"},
                    "shared_world": {"shared-world"}, "shared_exercise": {"shared-world"}}
    if config and config.get("profile") not in expected_lab[kind]:
        raise ValueError("Lab profile differs from its sealed runtime")
    if config and (config.get("control_profile") == "shared_exercise") != (kind == "shared_exercise"):
        raise ValueError("Lab control profile differs from its sealed task")
    if scene is not None:
        _task_document(docs.get("task.json"), scene)
    identity = {"directory": str(root), "manifest_sha256": _sha(manifest), "file_count": len(hashes),
        "verified_bytes": total, "runtime_documents": {name: _sha(value) for name, value in raw.items()},
        "lab_metadata": {name: _sha(value) for name, value in outer.items()},
        "source_binding": source_proof, "policy_executed": False,
        "policy_sha256": hashes.get("policy.json", hashes.get("policy.npz")),
        "custody_scope": "Every manifest-linked input verified before environment construction; no input writes."}
    return _Input(root, manifest, dict(hashes), raw, docs, config, outer, kind, scene, human, task, settings, identity)


class _Adapter:
    def __init__(self, data):
        self.data, self.done = data, False
        kind = data.kind
        if kind == "rolling":
            from weft.env import RollingEnv
            from weft.world import World
            self.env = RollingEnv(World.from_dict(data.docs["world.json"]))
            w = self.env.world
            self.scene = {"schema_version": 1, "profile": "rolling", "width": 1,
                "height": max(w.height_px / w.width_px, .4), "dt": self.env.task.dt, "controlled_id": "ball-0",
                "target": [.75, w.center_y(.75)], "task": self.env.task.to_dict(), "world": w.to_dict(),
                "support": {"slope": w.support_slope, "intercept": w.support_intercept},
                "objects": [{"id": "ball-0", "label": "Ball", "role": "active", "shape": "circle", "radius": w.radius}]}
        elif kind == "planar":
            from weft.scene import SceneEnv
            self.env = SceneEnv(data.scene)
        elif kind == "spatial":
            from weft.spatial_rigid import SpatialRigidEnv
            self.env = SpatialRigidEnv(data.scene)
        elif kind == "shared_world":
            from weft.shared_world import SharedWorldEnv
            self.env = SharedWorldEnv(data.scene)
        elif kind == "shared_exercise":
            from weft.shared_exercise_task import SharedExerciseTaskEnv
            self.env = SharedExerciseTaskEnv(data.scene, data.task)
            self.scene = data.scene
        elif kind == "human":
            from weft.human import HumanEnv
            self.env = HumanEnv(data.human)
            self.env.select_joint(data.settings["selected_joint"])
        elif kind == "whole_body":
            from weft.whole_body import WholeBodyEnv
            self.env = WholeBodyEnv(data.human, **data.settings)
        elif kind == "feet":
            from weft.foot_sandbox import FootEnv
            self.env = FootEnv(data.human, data.docs["foot-reference.json"], **data.settings)
        else:
            raise ValueError("Unsupported qualification adapter")
        try:
            if kind not in {"rolling", "shared_exercise"}:
                self.scene = self.env.scene
            if self.scene["profile"] not in PROFILES:
                raise ValueError("Adapter produced an unsupported capability profile")
            if data.scene is not None and data.scene.get("task") != self.scene.get("task"):
                raise ValueError("Constructed task differs from the sealed scene task")
            _task_document(data.docs.get("task.json"), self.scene)
            self.capabilities = describe_capabilities(self.scene, lab_profile="qualification",
                reference_available=bool(data.human or self.scene.get("human_spec")), exercise_task=data.task,
                learned_policy_available=False, source_video_present=any(name.startswith("source.") and
                    name.endswith((".mp4", ".webm", ".mov", ".avi", ".mkv")) for name in data.hashes))
            action_contract = self.capabilities["action_contract"]
            if self.env.observation_space.shape != (self.capabilities["observation_contract"]["dimension"],):
                raise ValueError("Capability observation dimension differs from the runtime")
            if action_contract["space_type"] == "continuous_box":
                if (self.env.action_space.shape != (action_contract["dimension"],) or
                        not np.all(self.env.action_space.low == action_contract["continuous_bounds"][0]) or
                        not np.all(self.env.action_space.high == action_contract["continuous_bounds"][1])):
                    raise ValueError("Capability motor space differs from the runtime")
            elif self.env.action_space.n != len(action_contract["discrete_actions"]):
                raise ValueError("Capability discrete actions differ from the runtime")
            self.horizon = self.capabilities["task_contract"]["truncation_steps"]
            if type(self.horizon) is not int or not 1 <= self.horizon <= 1800:
                raise ValueError("Qualification horizon must be within 1–1800 steps")
            self.reset_options = ({"jitter": False, "jitter_scale": 0.} if kind == "shared_exercise" else
                                  {"jitter": True} if kind == "whole_body" else
                                  {} if kind in {"rolling", "feet"} else {"jitter": False})
            self.recording_scene = self.scene
            if kind == "shared_exercise":
                from weft.shared_exercise_recording import recording_scene
                self.recording_scene = recording_scene(self.scene, data.task)
        except BaseException:
            self.env.close()
            raise

    def reset(self, seed):
        observation, _ = self.env.reset(seed=seed, options=self.reset_options)
        self.done = self.terminated = self.truncated = False
        self._observation(observation)
        return self.snapshot()

    def _observation(self, observation):
        if not self.env.observation_space.contains(observation) or not np.isfinite(observation).all():
            raise ValueError("Runtime observation violates its finite declared space")

    def snapshot(self):
        if self.data.kind != "rolling":
            return self.env.snapshot()
        from weft.env import task_success
        w, env = self.env.world, self.env
        return {"objects": [{**self.scene["objects"][0], "position": [env.x, w.center_y(env.x)],
            "velocity": [env.velocity, 0], "angle": 0}], "target": [env.target, w.center_y(env.target)],
            "controlled_id": "ball-0", "steps": env.steps, "time_s": env.steps * env.task.dt,
            "success": task_success(env.x, env.velocity, env.target, env.task),
            "terminated": self.terminated, "truncated": self.truncated}

    def action(self, controller):
        contract = self.capabilities["action_contract"]
        if controller == "authored":
            if self.data.kind in {"shared_world", "shared_exercise"}:
                return self.env.authored_action().tolist()
            if self.data.kind == "whole_body":
                return np.clip(self.env.feature_matrix().sum(axis=1), -1, 1).tolist()
            if self.data.kind == "feet":
                return self.env.hold_action().tolist()
            raise ValueError("This task has no authored controller with the same frozen task semantics")
        if contract["space_type"] == "continuous_box":
            return [0.] * contract["dimension"]
        return next(row["index"] for row in contract["discrete_actions"] if row["name"] == "coast")

    def changed_action(self, action, channel):
        contract = self.capabilities["action_contract"]
        if contract["space_type"] == "continuous_box":
            result = list(action)
            result[channel] = -.25 if result[channel] >= 0 else .25
            return result
        name = "positive" if self.data.kind == "human" else "right"
        return next(row["index"] for row in contract["discrete_actions"] if row["name"] == name)

    def validate_action(self, action):
        contract = self.capabilities["action_contract"]
        if contract["space_type"] == "discrete":
            if type(action) is not int or action not in {row["index"] for row in contract["discrete_actions"]}:
                raise ValueError("Action is outside the declared discrete contract")
            # Human actions >=5 bypass the selected-joint task; this protocol does not silently change it.
            if self.data.kind == "human" and action >= 5:
                raise ValueError("Reference/mobile intervention does not preserve the selected-joint task")
        elif (not isinstance(action, list) or len(action) != contract["dimension"] or
              any(type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1 for value in action)):
            raise ValueError("Action requires the declared finite normalized motor vector")

    def step(self, action):
        self.validate_action(action)
        if self.done:
            raise ValueError("Cannot advance an ended qualification episode")
        native = {0: 1, 1: 0, 2: 2}[action] if self.data.kind == "rolling" else action
        observation, reward, terminated, truncated, _ = self.env.step(native)
        self._observation(observation)
        if not math.isfinite(reward) or type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError("Runtime returned invalid reward or terminal flags")
        self.done, self.terminated, self.truncated = terminated or truncated, terminated, truncated
        event = {"type": "step", "action": action, "reward": float(reward), "terminated": terminated,
                 "truncated": truncated, "snapshot": self.snapshot()}
        if any(event["snapshot"].get(key) != event[key] for key in ("terminated", "truncated")):
            raise ValueError("Snapshot and transition terminal flags disagree")
        if self.data.kind == "shared_exercise":
            event["controller"] = "manual"
        _bytes(event)
        return event

    def close(self):
        self.env.close()


def _physical(snapshot):
    state = snapshot.get("engine_state", snapshot)
    result = {}

    def add(prefix, value):
        if isinstance(value, dict):
            for key, item in sorted(value.items()):
                add(prefix + "/" + key, item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                add(prefix + "/" + str(index), item)
        elif type(value) in (int, float) and math.isfinite(value):
            result[prefix] = float(value)
        else:
            raise ValueError("Nonfinite or malformed physical state")

    for obj in state.get("objects", []):
        for field in ("position", "velocity", "angle", "quaternion", "angular_velocity"):
            if field in obj:
                add("objects/" + obj["id"] + "/" + field, obj[field])
    human = state.get("human", {})
    for field in ("angles", "angular_velocities", "joints"):
        if field in human:
            add("human/" + field, human[field])
    for field in ("translation", "velocity", "linear_velocity"):
        if field in human.get("mobility", {}):
            add("human/mobility/" + field, human["mobility"][field])
    if not result:
        raise ValueError("No physical coordinates available for matched-action comparison")
    return result


def _task_result(snapshot):
    # Copy authoritative metrics; qualification never reconstructs or relaxes a success rule.
    for key in ("task_metrics", "whole_body_metrics", "posture_metrics", "coupled_metrics"):
        if isinstance(snapshot.get(key), dict):
            return {"origin": "snapshot." + key, "value": deepcopy(snapshot[key])}
    return {"origin": "snapshot", "value": {key: snapshot[key] for key in
        ("success", "terminated", "truncated", "steps", "episode_steps", "time_s") if key in snapshot}}


def _interval_coverage(adapter, initial, final, steps, failure):
    start, end = initial.get("engine_state", initial), final.get("engine_state", final)
    expected_seconds = adapter.horizon * adapter.scene["dt"]
    elapsed = end.get("time_s", 0.) - start.get("time_s", 0.)
    completion, engine_failure = True, None
    if adapter.data.kind == "shared_exercise":
        metrics = final["task_metrics"]
        completion = (metrics["completed_interval"] is True and final["engine_state_status"] == "current"
                      and metrics["clock_status"] == "observed")
        engine_failure = metrics["failure"]
    else:
        key = {"shared_world": "coupled_metrics", "whole_body": "whole_body_metrics", "feet": "posture_metrics"}.get(adapter.data.kind)
        if key:
            metrics = final[key]
            completion = metrics["completed_horizon"] is True
            # Completed-but-unsuccessful task gates are not engine failures.
            engine_failure = metrics.get("failure") if not completion else None
    evidence = {"recorded_steps_match_horizon": steps == adapter.horizon,
        "authoritative_completion": completion, "engine_failure": engine_failure,
        "expected_simulated_seconds": expected_seconds, "observed_simulated_seconds": elapsed,
        "complete_engine_clock": math.isclose(elapsed, expected_seconds, rel_tol=0., abs_tol=1e-8),
        "terminal_reported": bool(final.get("terminated") or final.get("truncated")),
        "scope": "Declared task duration and engine completion only; task success is unchanged and separate"}
    complete = (failure is None and engine_failure is None and all(evidence[key] for key in
                ("recorded_steps_match_horizon", "authoritative_completion", "complete_engine_clock", "terminal_reported")))
    return complete, evidence


class _GlobalLimit(ValueError):
    pass


class _Budget:
    def __init__(self, deadline, steps, output):
        self.deadline, self.steps, self.output = deadline, steps, output
        self.used_steps, self.used_bytes = 0, 0

    def step(self):
        _deadline(self.deadline)
        if self.used_steps >= self.steps:
            raise _GlobalLimit("Qualification total step bound reached")
        self.used_steps += 1

    def ensure(self, size):
        _deadline(self.deadline)
        if self.used_bytes + size > self.output:
            raise _GlobalLimit("Qualification output byte bound reached")

    def reserve(self, size):
        self.ensure(size)
        self.used_bytes += size


def _write(root, name, value, budget):
    raw = _bytes(value)
    budget.reserve(len(raw))
    write_package_files(root, {name: raw})


def _replay(adapter, recording, expected, budget):
    loaded = load_recording(recording, deadline=budget.deadline)
    events = loaded["events"]
    if loaded["scene"] != adapter.recording_scene:
        raise ValueError("Recorded scene differs from the admitted runtime")
    if len(events) != len(expected) + 1:
        raise ValueError("Recorded transition count differs from execution")
    initial = adapter.reset(events[0]["seed"])
    if canonical_hash(initial) != canonical_hash(events[0]["snapshot"]):
        raise ValueError("Same-seed reset differs from the recorded initial state")
    for saved, expected_hash in zip(events[1:], expected):
        event = {key: value for key, value in saved.items() if key != "index"}
        if canonical_hash(event) != expected_hash:
            raise ValueError("Recorded transition differs from authoritative execution")
        budget.step()
        actual = adapter.step(event["action"])
        if adapter.data.kind == "shared_exercise":
            actual["controller"] = event["controller"]
        if canonical_hash(actual) != expected_hash:
            raise ValueError("Action replay differs from authoritative reward, flags or state")
    return loaded


def _execute(data, case, root, budget):
    adapter = _Adapter(data)
    recorder = None
    try:
        contract = adapter.capabilities["action_contract"]
        channel = case["probe_channel"]
        if channel >= contract["dimension"]:
            raise ValueError("Probe channel is outside this capability contract")
        if case["controller"] == "authored" and data.kind not in {"whole_body", "feet", "shared_world", "shared_exercise"}:
            raise ValueError("Authored controller would change the selected task")
        maximum_steps = 2 * adapter.horizon + min(case["probe_steps"], adapter.horizon)
        if maximum_steps > budget.steps - budget.used_steps:
            raise ValueError("Complete interval and replay exceed the remaining step envelope")
        contracts = {"capabilities": adapter.capabilities, "scene_sha256": canonical_hash(adapter.scene),
            "task_sha256": canonical_hash(data.task or adapter.scene["task"]), "horizon_steps": adapter.horizon,
            "reset_options": adapter.reset_options, "maximum_steps_including_replay": maximum_steps,
            "controller": case["controller"], "probe_channel": channel, "probe_steps": case["probe_steps"],
            "probe_operator": "replace selected motor with -0.25 if baseline >=0 else +0.25; discrete positive/right",
            "other_channels": "Exact saved baseline commands, not recomputed feedback",
            "source_policy": "Bound but not executed; learned-policy performance is outside this protocol"}
        _write(root, "contracts.json", contracts, budget)
        initial = adapter.reset(case["seed"])
        _physical(initial)
        recorder = Recorder(root / "recording", adapter.recording_scene, profile=data.kind)
        budget.reserve(2 * len(_bytes(adapter.recording_scene)) + 8192 + len(_bytes(initial)))
        recorder.start(initial, seed=case["seed"])
        hashes, prefix, failure = [], [], None
        final = initial
        for _ in range(adapter.horizon):
            _deadline(budget.deadline)
            if adapter.done:
                break
            action = adapter.action(case["controller"])
            # Reserve room before physics: an exhausted output envelope must not lose a completed action.
            budget.ensure(2 * MAX_EVENT_BYTES + 128)
            recorder.check_capacity()
            budget.step()
            try:
                event = adapter.step(action)
            except (RuntimeError, ValueError) as error:
                failure = {"type": type(error).__name__, "reason": str(error), "attempted_action": action,
                           "completed_steps": len(hashes), "last_valid_snapshot_sha256": canonical_hash(final)}
                break
            if data.kind == "shared_exercise" and case["controller"] == "authored":
                event["controller"] = "authored"
            budget.reserve(len(_bytes(event)) + 128)
            recorder.append(event)
            hashes.append(canonical_hash(event))
            final = event["snapshot"]
            if len(prefix) < case["probe_steps"]:
                prefix.append(event)
        recorder.close()
        recorder = None
        _write(root, "execution.json", {"transition_hashes": hashes, "failure": failure,
            "initial_snapshot_sha256": canonical_hash(initial), "final_snapshot_sha256": canonical_hash(final)}, budget)
        adapter.close()
        adapter = _Adapter(data)
        _replay(adapter, root / "recording", hashes, budget)
        replay_failure = None
        if failure:
            budget.step()
            try:
                adapter.step(failure["attempted_action"])
            except (RuntimeError, ValueError) as error:
                replay_failure = {"type": type(error).__name__, "reason": str(error)}
            if replay_failure != {key: failure[key] for key in ("type", "reason")}:
                raise ValueError("Failed attempted action did not reproduce exactly")
        reset = adapter.reset(case["seed"])
        if canonical_hash(reset) != canonical_hash(initial):
            raise ValueError("Changed-action trial did not start from the exact same seeded state")
        changed, differences, probe_failure = [], {}, None
        budget.ensure((len(prefix) + 1) * 2 * MAX_EVENT_BYTES)
        for baseline in prefix:
            if adapter.done:
                break
            action = adapter.changed_action(baseline["action"], channel)
            budget.step()
            try:
                event = adapter.step(action)
            except (RuntimeError, ValueError) as error:
                probe_failure = {"type": type(error).__name__, "reason": str(error), "attempted_action": action}
                break
            before, after = _physical(baseline["snapshot"]), _physical(event["snapshot"])
            if before.keys() != after.keys():
                raise ValueError("Matched intervention changed physical state identity")
            for key in before:
                delta = abs(after[key] - before[key])
                if delta:
                    differences[key] = max(differences.get(key, 0.), delta)
            changed.append({**event, "baseline_action": baseline["action"],
                            "baseline_snapshot_sha256": canonical_hash(baseline["snapshot"])})
        response = any(value > 1e-10 for value in differences.values())
        _write(root, "changed-action.json", {"initial_snapshot_sha256": canonical_hash(reset), "transitions": changed,
            "maximum_absolute_differences_by_coordinate": differences, "failure": probe_failure,
            "response_observed": response, "numeric_detection_floor": 1e-10,
            "scope": "Matched simulated intervention only; coordinates retain their own units, never a physical accuracy score"}, budget)
        complete, coverage = _interval_coverage(adapter, initial, final, len(hashes), failure)
        operational = complete and response and probe_failure is None and len(changed) == len(prefix) and bool(prefix)
        result = {"id": case["id"], "status": "qualified" if operational else "not_qualified",
            "scope": "Operational runtime only", "physical_validation": "unverified", "input": data.identity,
            "profile": adapter.scene["profile"], "control_profile": data.kind, "steps": len(hashes),
            "declared_horizon_steps": adapter.horizon, "complete_interval": complete,
            "interval_coverage": coverage,
            "execution_outcome": ("complete_declared_interval" if complete else "partial_engine_failure" if failure or coverage["engine_failure"]
                                  else "valid_early_terminal" if final.get("terminated") or final.get("truncated")
                                  else "incomplete_without_terminal"),
            "exact_action_replay": True, "same_seed_reset": True, "changed_action_response": response,
            "changed_action_steps": len(changed), "simulation_failure": failure, "probe_failure": probe_failure,
            "terminal": {"terminated": bool(final.get("terminated")), "truncated": bool(final.get("truncated"))},
            "task_result": _task_result(final), "policy_evaluation": "not_performed"}
        _write(root, "result.json", result, budget)
        return result
    finally:
        if recorder is not None and recorder._started is not None and not recorder._closed:
            recorder.close()
        adapter.close()


def _cases(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise ValueError("Qualification requires 1–5 explicit cases")
    result, ids = [], set()
    for row in cases:
        if not isinstance(row, dict) or set(row) - {"id", "input", "manifest_sha256", "seed", "controller", "probe_steps", "probe_channel"}:
            raise ValueError("Unknown qualification case fields; task and policy overrides are forbidden")
        name = row.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,47}", name) or name in ids:
            raise ValueError("Case identifiers must be unique safe names")
        if not isinstance(row.get("input"), (str, Path)):
            raise ValueError("Case requires a sealed input path")
        resolved = {"id": name, "input": str(_path(row["input"])), "seed": row.get("seed", 1729),
                    "controller": row.get("controller", "coast"), "probe_steps": row.get("probe_steps", 6),
                    "probe_channel": row.get("probe_channel", 0)}
        for key, maximum in (("seed", 2**31 - 1), ("probe_steps", 12), ("probe_channel", 63)):
            if type(resolved[key]) is not int or not (1 if key == "probe_steps" else 0) <= resolved[key] <= maximum:
                raise ValueError("Qualification seed/probe bounds are invalid")
        if resolved["controller"] not in {"coast", "authored"}:
            raise ValueError("Qualification supports coast or task-preserving authored control, not learned evaluation")
        if "manifest_sha256" in row:
            if not isinstance(row["manifest_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["manifest_sha256"]):
                raise ValueError("Expected input manifest must be an exact SHA256")
            resolved["manifest_sha256"] = row["manifest_sha256"]
        ids.add(name)
        result.append(resolved)
    return result


def qualify_runtimes(cases, out, *, max_seconds=60, max_steps=MAX_STEPS, max_output_bytes=64 * 1024 * 1024):
    """Qualify sealed runtimes without modifying inputs, training, encoding, servers or graphics."""
    cases = _cases(cases)
    for value, maximum in ((max_seconds, MAX_SECONDS), (max_steps, MAX_STEPS), (max_output_bytes, MAX_OUTPUT_BYTES)):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("Qualification work bounds must be positive bounded integers")
    if max_output_bytes < 1024 * 1024:
        raise ValueError("Qualification output bound must allow a failure receipt")
    out = _path(out)
    if out.exists() or any(out.is_relative_to(_path(row["input"])) for row in cases):
        raise ValueError("Choose a new output directory outside every inspected input")
    started = time.monotonic()
    budget = _Budget(started + max_seconds, max_steps, max_output_bytes - 512 * 1024)
    runtime = _runtime()
    inputs = [_admit(row["input"], budget.deadline) for row in cases]
    if any(row.get("manifest_sha256", data.identity["manifest_sha256"]) != data.identity["manifest_sha256"]
           for row, data in zip(cases, inputs)):
        raise ValueError("Input differs from the explicitly frozen manifest identity")
    if len({item.identity["manifest_sha256"] for item in inputs}) != len(inputs):
        raise ValueError("Duplicate input packages are not distinct qualification cases")
    plan = {"schema_version": 1, "profile": PROFILE, "created_at": local_time(), "cases": cases,
        "inputs": [item.identity for item in inputs], "runtime": runtime,
        "bounds": {"max_seconds": max_seconds, "max_total_env_steps_including_replay": max_steps,
                   "max_output_bytes": max_output_bytes, "workers": 1, "external_api_usd": 0},
        "scope": "Manual/authored software execution, same-seed fresh replay and matched simulated interventions; not learning or physical validation",
        "limits": ["Early terminal attempts remain recorded but do not qualify the complete declared horizon",
                   "No thresholds, scene, task, source interval or selected learned policy are changed",
                   "Only the input's active control target is tested; other actors are not automatically qualified",
                   "Wall-time checks are cooperative; a stuck native physics call needs an external process deadline"]}
    plan_raw = _bytes(plan)
    budget.reserve(len(plan_raw))
    out.mkdir(parents=True, exist_ok=False)
    write_package_files(out, {"plan.json": plan_raw})
    results = []
    failure = None
    try:
        for data, case in zip(inputs, cases):
            _deadline(budget.deadline)
            # Recheck every consumed runtime document before construction; never execute archived source.
            for name, raw in data.raw.items():
                read_package_file(data.root, name, max_bytes=MAX_JSON_BYTES, expected_sha256=_sha(raw))
            for name, raw in data.outer.items():
                read_package_file(data.root.parent, name, max_bytes=MAX_JSON_BYTES, expected_sha256=_sha(raw))
            if _runtime() != runtime:
                raise ValueError("Qualification implementation changed after the plan freeze")
            target = out / case["id"]
            target.mkdir()
            try:
                results.append(_execute(data, case, target, budget))
            except (_GlobalLimit, TimeoutError):
                raise
            except (ValueError, RuntimeError, OSError) as error:
                result = {"id": case["id"], "status": "failed", "scope": "Operational runtime only",
                    "physical_validation": "unverified", "input": data.identity,
                    "failure": {"type": type(error).__name__, "reason": str(error)[:4096],
                                "partial_evidence_preserved": True}, "policy_evaluation": "not_performed"}
                _write(target, "result.json", result, budget)
                results.append(result)
        for data in inputs:
            read_package_file(data.root, "manifest.json", max_bytes=MAX_JSON_BYTES, expected_sha256=_sha(data.manifest))
            for name, raw in data.raw.items():
                read_package_file(data.root, name, max_bytes=MAX_JSON_BYTES, expected_sha256=_sha(raw))
            for name, raw in data.outer.items():
                read_package_file(data.root.parent, name, max_bytes=MAX_JSON_BYTES, expected_sha256=_sha(raw))
        if _runtime() != runtime:
            raise ValueError("Qualification implementation changed during execution")
    except (ValueError, RuntimeError, TimeoutError, OSError) as error:
        failure = {"type": type(error).__name__, "reason": str(error), "partial_evidence_preserved": True}
    report = {"schema_version": 1, "profile": PROFILE,
        "status": "failed" if failure or any(row["status"] == "failed" for row in results) else "completed",
        "plan_sha256": _sha((out / "plan.json").read_bytes()), "cases": results, "failure": failure,
        "operationally_qualified": sum(row["status"] == "qualified" for row in results),
        "physical_validation": "unverified", "policy_evaluation": "not_performed",
        "env_steps_including_replay": budget.used_steps, "seconds": time.monotonic() - started,
        "input_manifests_and_consumed_documents_unchanged": failure is None}
    # A reserved receipt remains writable when the work/output envelope is exhausted.
    write_package_files(out, {"report.json": _bytes(report)})
    files, total, entry_count, pending = {}, 0, 0, [(out, 0)]
    while pending:
        directory, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_count += 1
                if entry.is_symlink() or depth > 3 or entry_count > 128:
                    raise ValueError("Qualification output contains unsafe or excessive entries")
                if entry.is_dir(follow_symlinks=False):
                    pending.append((Path(entry.path), depth + 1))
                    continue
                name = Path(entry.path).relative_to(out).as_posix()
                raw = read_package_file(out, name, max_bytes=min(64 * 1024 * 1024, max(1, max_output_bytes - total)))
                total += len(raw)
                files[name] = _sha(raw)
    manifest = _bytes({"schema_version": 1, "profile": PROFILE, "status": report["status"], "files": files})
    if total + len(manifest) > max_output_bytes:
        raise ValueError("Preserved qualification evidence exceeds its output envelope; no successful seal published")
    write_package_files(out, {"manifest.json": manifest})
    return {"directory": str(out), "manifest_sha256": _sha(manifest), **report}
