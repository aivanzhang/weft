"""Turn saved action/hash trials into inspectable recordings without choosing new actions."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from weft.io import canonical_hash, digest, local_time, write_json
from weft.recording import Recorder, export_recording
from weft.shared_exercise_diagnostics import _SubstepDiagnosis, _StudyOutput
from weft.shared_exercise_lab import _json
from weft.shared_exercise_learning import _runtime_hashes, _runtime_versions
from weft.shared_exercise_recording import recording_scene
from weft.shared_exercise_task import SharedExerciseTaskEnv, validate_shared_exercise_task, prepare_shared_exercise_snapshot_validator
from weft.shared_world import validate_shared_world
from weft.shared_world_packaging import write_package_files


PROFILE = "shared-exercise-saved-trial-replay-v1"
MAX_TRIAL_BYTES = 8 << 20
MAX_OUTPUT_BYTES = 112 << 20


def _require(condition, message):
    if not condition:
        raise ValueError("Saved trial replay: " + message)


def replay_runtime_hashes(scene):
    names = set(_runtime_hashes(scene)) | {"recording.py", "shared_exercise_recording.py", "shared_exercise_diagnostics.py"}
    return {name: digest(Path(__file__).parent / name) for name in sorted(names)}


def _validate_trial(scene, task, raw, expected):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_TRIAL_BYTES and sha256(raw).hexdigest() == expected,
             "exact bounded original trial bytes required")
    trial = _json(raw)
    _require(trial.get("status") == "completed" and type(trial.get("control_steps")) is int
             and 1 <= trial["control_steps"] == task["interval"]["horizon_steps"] <= 1800,
             "complete original trial required; interrupted trials cannot be filled in")
    _require(type(trial.get("seed")) is int and 0 <= trial["seed"] <= 2**31 - 1,
             "bounded original reset seed required")
    validate = prepare_shared_exercise_snapshot_validator(scene, task)
    for name in ("initial_snapshot", "final_snapshot"):
        validate(trial[name])
    _require(trial["initial_state_sha256"] == canonical_hash(trial["initial_snapshot"]["engine_state"])
             and trial["reset_options"] == trial["initial_snapshot"]["resolved_reset_options"]
             and trial["initial_metrics"] == trial["initial_snapshot"]["task_metrics"]
             and trial["metrics"] == trial["final_snapshot"]["task_metrics"], "original state/task/reset binding differs")
    actions = trial.get("actions")
    count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    _require(type(actions) is list and len(actions) == trial["control_steps"], "complete saved action/hash trace required")
    for i, row in enumerate(actions):
        _require(type(row) is dict and set(row) == {"action", "reward", "terminated", "truncated", "snapshot_sha256"},
                 "saved action trace fields differ")
        _require(type(row["action"]) is list and len(row["action"]) == count
                 and all(type(v) in (int, float) and math.isfinite(v) and abs(v) <= 1 for v in row["action"]),
                 "saved actions must stay inside the original human action space")
        _require(type(row["reward"]) in (int, float) and math.isfinite(row["reward"])
                 and type(row["terminated"]) is bool and type(row["truncated"]) is bool
                 and type(row["snapshot_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["snapshot_sha256"]),
                 "saved outcomes and state hashes required")
        _require(i == len(actions) - 1 or not (row["terminated"] or row["truncated"]), "trace continues after termination")
    _require(actions[-1]["snapshot_sha256"] == canonical_hash(trial["final_snapshot"])
             and actions[-1]["terminated"] == trial["terminated"] and actions[-1]["truncated"] == trial["truncated"]
             and (trial["terminated"] or trial["truncated"]), "original terminal state differs")
    return trial


def replay_saved_trial(scene, task, trial_bytes, out, *, expected_trial_sha256,
                       expected_runtime_sha256, expected_runtime_versions, controller=None, max_seconds=60):
    """Verify every saved boundary while observing substeps; never run a new policy search."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 60,
             "positive deadline of at most 60 seconds required")
    started, root = time.monotonic(), Path(out).absolute()
    deadline = started + max_seconds
    scene = validate_shared_world(scene)
    _require(scene["profile"] in ("shared-world-v1", "shared-world-v2", "shared-world-v3"), "unsupported scene profile")
    task = validate_shared_exercise_task(task, scene)
    _require(controller in (None, "authored"), "controller origin must be unspecified or independently checked authored actions")
    runtime = replay_runtime_hashes(scene)
    _require(expected_runtime_sha256 == runtime and expected_runtime_versions == _runtime_versions(),
             "original physics/evaluator/runtime identities differ")
    trial = _validate_trial(scene, task, trial_bytes, expected_trial_sha256)
    _require(not root.exists() and not root.is_symlink(), "new output directory required")
    implementation = {"implementation/shared_exercise_trial_replay.py": Path(__file__).read_bytes(),
        **{"implementation/" + name: (Path(__file__).parent / name).read_bytes() for name in runtime}}
    inputs = {"original-trial.json": trial_bytes, **implementation,
              "scene.json": (json.dumps(scene, allow_nan=False) + "\n").encode(),
              "exercise-task.json": (json.dumps(task, allow_nan=False) + "\n").encode()}
    _require(sum(map(len, inputs.values())) <= 15 << 20, "input/runtime copies exceed reserved output space")

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Saved trial replay deadline")

    check()
    root.mkdir(parents=True, mode=0o700)
    diagnostic_output = _StudyOutput(root / "diagnostic")
    recorder, env, completed = None, None, False
    try:
        plan = {"schema_version": 1, "profile": PROFILE, "created_at": local_time(),
                "trial_sha256": expected_trial_sha256, "scene_sha256": canonical_hash(scene),
                "task_sha256": canonical_hash(task), "runtime_sha256": runtime, "runtime_versions": _runtime_versions(),
                "implementation_sha256": digest(Path(__file__)), "controller": controller or "unspecified_saved_actions",
                "control_steps": trial["control_steps"], "max_seconds": max_seconds, "maximum_output_bytes": MAX_OUTPUT_BYTES,
                "scope": "One exact saved-action re-execution, not a new trial, policy search or physical observation. Substep telemetry is simulated; original source custody belongs to the caller's sealed run."}
        write_json(root / "plan.json", plan)
        write_package_files(root, inputs)
        diagnostic = _SubstepDiagnosis(scene, task, diagnostic_output.append, deadline)

        class ObservedEnv(SharedExerciseTaskEnv):
            def _observe_substep(self, sample):
                super()._observe_substep(sample)
                diagnostic.consume(sample)

        env = ObservedEnv(scene, task)
        env.reset(seed=trial["seed"], options=trial["reset_options"])
        diagnostic.reset(env._accumulator.initial)
        first = env.snapshot()
        _require(first == trial["initial_snapshot"], "initial snapshot differs; original evidence retained")
        recorder = Recorder(root / "recording", recording_scene(scene, task), profile="shared-exercise")
        recorder._journal_limit = min(recorder._journal_limit, 16 << 20)
        recorder.start(first, seed=trial["seed"])
        total = 0.
        for index, row in enumerate(trial["actions"]):
            check()
            if controller == "authored":
                _require(env.authored_action().tolist() == row["action"], "claimed authored actions differ")
            _, reward, terminated, truncated, _ = env.step(row["action"])
            snapshot = env.snapshot()
            _require((reward, terminated, truncated) == (row["reward"], row["terminated"], row["truncated"])
                     and canonical_hash(snapshot) == row["snapshot_sha256"], f"saved boundary differs at action {index}; no later actions executed")
            event = {"type": "step", "action": row["action"], "reward": reward,
                     "terminated": terminated, "truncated": truncated, "snapshot": snapshot}
            if controller is not None:
                event["controller"] = controller
            recorder.append(event)
            total += float(reward)
        _require(env.snapshot() == trial["final_snapshot"] and env.metrics() == trial["metrics"] and total == trial["return"],
                 "original final outcome differs")
        recorder.close()
        env.close()
        env = None
        diagnostic_output.write("summary.json", (json.dumps(diagnostic.summary(), allow_nan=False) + "\n").encode())
        diagnostic_output.seal("verified_saved_trial")
        exported = export_recording(recorder.root, deadline=deadline, max_output_bytes=64 << 20)
        check()
        _require(replay_runtime_hashes(scene) == runtime and digest(Path(__file__)) == plan["implementation_sha256"],
                 "runtime changed during saved trial replay")
        result = {"schema_version": 1, "profile": PROFILE, "status": "verified_saved_trial", "exact_state_replay": True,
                  "trial_sha256": expected_trial_sha256, "control_steps": len(trial["actions"]),
                  "substeps": diagnostic.samples, "substep_diagnosis": diagnostic.summary(), "recording": exported,
                  "seconds": time.monotonic() - started, "inference_calls": 0, "api_calls": 0,
                  "scope": plan["scope"]}
        write_json(root / "result.json", result)
        completed = True
        return result
    except Exception as error:
        write_json(root / "failure.json", {"type": type(error).__name__, "message": str(error)[:2000],
                   "partial_evidence_preserved": True, "status": "failed"})
        raise
    finally:
        if recorder is not None and recorder._started is not None and not recorder._closed:
            recorder.close()
        if env is not None:
            env.close()
        if not completed and diagnostic_output.trace is not None:
            diagnostic_output.seal("failed")
        files = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*") if p.is_file() and p != root / "manifest.json"}
        _require(sum((root / name).stat().st_size for name in files) <= MAX_OUTPUT_BYTES, "output exceeds declared bound")
        write_json(root / "manifest.json", {"schema_version": 1, "status": "completed" if completed else "failed", "files": files})
