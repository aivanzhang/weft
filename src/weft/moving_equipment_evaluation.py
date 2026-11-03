"""Changed-object-start development evaluation using the unchanged engine and placement scorer."""

from copy import deepcopy
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import numpy as np

from weft.io import canonical_hash
from weft.moving_equipment_task import MovingEquipmentTaskEnv, PlacementMetrics
from weft import moving_equipment_learning as learning

PROFILE = "moving-equipment-object-reset-evaluation-v1"
PROTOCOL_PROFILE = "moving-equipment-object-reset-protocol-v1"
MAX_CONTROLS = 20_000
MAX_BYTES = 64 << 10
CONTROLLERS = ("zero", "authored_feedback", "random", "frozen_policy")
SCOPE = ("Development evaluation on explicitly authored changed object starts in the same simulated world. "
         "Human/reference, goal, scorer and limits are unchanged. The nested original task is a frozen "
         "scorer template; its exact-start scope does not describe this separate reset variant. "
         "This is not a final holdout, new-video validation, recovered grasp or physical accuracy.")


def _admitted(value):
    from weft.moving_equipment_task_lab import validate_task_lab_admission
    return validate_task_lab_admission(value)


def _bytes(value):
    return learning._bytes(value, MAX_BYTES)


def _context(admission):
    admission = _admitted(admission)
    scene, task = admission.scene, admission.task
    if (task.get("profile") != "moving-equipment-placement-task-v1"
            or task.get("reset") != {"jitter": False} or type(task["reset"].get("jitter")) is not bool
            or task["interval"]["start_offset_s"] != 0):
        raise ValueError("Evaluation requires the original exact-start scorer template")
    return admission, scene, task


def _pins(admission, scene, task):
    return learning._pins(admission, scene, task) | {"task_binding": deepcopy(task["binding"])}


def build_changed_start_protocol(admission, *, evaluation_seeds, position_jitter=.025, velocity_jitter=.1):
    """Freeze bounded object-only jitter; seeds are requests, not proof of distinct physical starts."""
    admission, scene, task = _context(admission)
    if (not isinstance(evaluation_seeds, list) or not 2 <= len(evaluation_seeds) <= 16
            or any(type(x) is not int or not 0 <= x < 2**31 for x in evaluation_seeds)
            or len(set(evaluation_seeds)) != len(evaluation_seeds)):
        raise ValueError("Evaluation needs 2–16 unique bounded integer seeds")
    for value, maximum in ((position_jitter, .05), (velocity_jitter, .1)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError("Object jitter exceeds its authored evaluation bound")
    if position_jitter == velocity_jitter == 0:
        raise ValueError("Different seeds without a changed state are not evaluation")
    horizon = task["interval"]["horizon_steps"]
    if type(horizon) is not int or horizon < 1 or horizon*len(evaluation_seeds)*4 > MAX_CONTROLS:
        raise ValueError("Matched changed-start evaluation exceeds 20000 controls")
    result = {"schema_version": 1, "profile": PROTOCOL_PROFILE, "source": _pins(admission, scene, task),
        "evaluation_seeds": list(evaluation_seeds),
        "reset_options": {"jitter": True, "jitter_scale": 0., "position_jitter": float(position_jitter),
                          "velocity_jitter": float(velocity_jitter)},
        "training_start_scope": "existing_nominal_only_feedback_policy",
        "interval": deepcopy(task["interval"]), "controllers": list(CONTROLLERS),
        "maximum_control_steps": horizon*len(evaluation_seeds)*4, "maximum_seconds": 60,
        "reset_sampling": "existing_engine_up_to_32_position_attempts_with_explicit_fallback",
        "fallback_policy": "exclude_whole_start_without_replacement",
        "duplicate_policy": "exclude_nominal_or_previously_seen_actual_state_without_replacement",
        "scope": SCOPE}
    _bytes(result)
    return result


def validate_changed_start_protocol(protocol, admission):
    _bytes(protocol)
    if not isinstance(protocol, dict) or type(protocol.get("schema_version")) is not int:
        raise ValueError("Evaluation protocol requires its closed versioned contract")
    options = protocol.get("reset_options")
    if not isinstance(options, dict):
        raise ValueError("Evaluation reset options are missing")
    expected = build_changed_start_protocol(admission, evaluation_seeds=protocol.get("evaluation_seeds"),
        position_jitter=options.get("position_jitter"), velocity_jitter=options.get("velocity_jitter"))
    if canonical_hash(expected) != canonical_hash(protocol):
        raise ValueError("Evaluation source, scorer interval, reset protocol or bounds differ")
    return expected


def _policy(raw, admission):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("Frozen policy requires 1–65536 exact bytes")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Frozen policy repeats JSON keys")
            result[key] = value
        return result
    def constant(_):
        raise ValueError("Frozen policy contains nonfinite JSON")
    try:
        policy = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Frozen policy requires bounded JSON") from error
    policy = learning.validate_moving_equipment_policy(policy, admission)
    if policy["algorithm"] != learning.ALGORITHM:
        raise ValueError("Changed-start evaluator supports only the nominal-only training family")
    return policy


class _ChangedStartPlacementEnv(MovingEquipmentTaskEnv):
    """Separate reset variant; inherited action/scoring path is unchanged, and old recording is not supported."""

    def __init__(self, scene, task, bundle, protocol):
        self._evaluation_protocol = deepcopy(protocol)
        super().__init__(scene, task, bundle)

    def reset(self, *, seed=None, options=None):
        if options is not None or seed not in self._evaluation_protocol["evaluation_seeds"]:
            raise ValueError("Evaluation reset must use an explicit protocol seed without caller overrides")
        self._done, self._metrics = True, None
        observation, info = self._engine.reset(seed=seed, options=deepcopy(self._evaluation_protocol["reset_options"]))
        if (info["initial_angular_offset_rmse_rad"] != 0 or info["initial_root_offset_rmse_model_units"] != 0
                or info["jitter_scale"] != 0):
            raise ValueError("Evaluation reset unexpectedly changed the original human start")
        state = self._engine.snapshot()
        self._metrics = PlacementMetrics(self._scene, self._task, state["human"]["mobility"]["max_penetration"])
        self._done = False
        return self._observation(observation), info


def _state(env):
    # Complete generalized positions/velocities distinguish physical starts, not their requested seeds.
    return {"qpos": env._engine.data.qpos.tolist(), "qvel": env._engine.data.qvel.tolist()}


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Changed-start evaluation reached its aggregate deadline")


def evaluate_changed_starts(admission, protocol, policy_bytes, out, *, max_seconds=60):
    """Run matched full-horizon comparisons; refused starts remain in the reported denominator."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 60:
        raise ValueError("Changed-start evaluation requires a finite bound up to 60 seconds")
    started = time.monotonic(); deadline = started+max_seconds
    admission, scene, task = _context(admission)
    protocol = validate_changed_start_protocol(protocol, admission)
    policy = _policy(policy_bytes, admission)
    if policy["training_seed"] in protocol["evaluation_seeds"]:
        raise ValueError("Evaluation seed must not reuse the frozen policy training seed")
    target = Path(out).absolute()
    if target.exists() or ".." in target.parts or any(p.is_symlink() for p in (target, *target.parents)):
        raise ValueError("Choose a new nonlinked evaluation output")
    _check(deadline)
    hashes, versions, runtime = learning._runtime(admission, scene, deadline)
    own = Path(__file__).read_bytes(); hashes[Path(__file__).name] = sha256(own).hexdigest()
    runtime["implementation/"+Path(__file__).name] = own
    if len(runtime) > 160 or sum(map(len, runtime.values())) > 16 << 20:
        raise ValueError("Evaluation runtime exceeds its 160-file/16-MiB bound")
    output = learning._Output(target)
    env = None; rows = []; controls = 0
    try:
        # Freeze policy/protocol before any nominal or perturbed state is exposed.
        output.save("policy.json", policy_bytes); output.json("protocol.json", protocol)
        output.json("scene.json", scene); output.json("scorer-template.json", task)
        for name, raw in runtime.items():
            output.save(name, raw)
        output.json("plan.json", {"profile": PROFILE, "policy_sha256": sha256(policy_bytes).hexdigest(),
            "protocol_sha256": canonical_hash(protocol), "runtime_sha256": hashes, "runtime_versions": versions,
            "maximum_control_steps": protocol["maximum_control_steps"], "maximum_seconds": max_seconds,
            "scope": SCOPE, "recording": "old_exact_start_task_recording_not_supported"})
        _check(deadline)
        env = _ChangedStartPlacementEnv(scene, task, admission.bundle, protocol)
        env._engine.reset(seed=policy["training_seed"], options={"jitter": False})
        nominal = _state(env); seen = {canonical_hash(nominal)}
        output.json("nominal-start.json", nominal)
        # The policy was fully validated before exposure; do not scan its runtime again per controller.
        frozen = learning._Feedback(env, scene, task, policy["weights"])
        dimension = len(policy["weights"])
        authored = learning._Feedback(env, scene, task, np.ones(dimension))
        for index, seed in enumerate(protocol["evaluation_seeds"]):
            _check(deadline)
            row = {"seed": seed, "status": "pending", "trials": []}; rows.append(row)
            try:
                _, reset_info = env.reset(seed=seed)
            except (ValueError, RuntimeError) as error:
                row.update(status="reset_refused", reason=str(error)[:300]); output.json(f"start-{index+1}.json", row); continue
            physical = _state(env); state_sha = canonical_hash(physical)
            row.update(actual_start=physical, actual_start_sha256=state_sha, reset_receipt=reset_info)
            if any(value["fallback_unperturbed"] for value in reset_info["object_perturbations"].values()):
                row.update(status="excluded_fallback")
            elif state_sha in seen:
                row.update(status="excluded_duplicate_or_nominal")
            else:
                seen.add(state_sha); row["status"] = "admitted"
            if row["status"] != "admitted":
                output.json(f"start-{index+1}.json", row); continue
            for controller in CONTROLLERS:
                _check(deadline)
                if controller != "zero":
                    _, reset_again = env.reset(seed=seed)
                    if _state(env) != physical or reset_again != reset_info:
                        raise ValueError("Matched controller reset differs from its original admitted start")
                rng = np.random.default_rng(seed ^ 0x5A17)
                trial = {"controller": controller, "control_steps": 0, "status": "running"};row["trials"].append(trial)
                while not env._done:
                    _check(deadline)
                    if controls >= protocol["maximum_control_steps"]:
                        raise RuntimeError("Changed-start aggregate control reservation exhausted")
                    action = (np.zeros(len(env.action_names)) if controller == "zero" else authored() if controller == "authored_feedback"
                              else rng.uniform(-1., 1., len(env.action_names)) if controller == "random" else frozen())
                    controls += 1; trial["control_steps"] += 1
                    try:
                        env.step(action)
                    except RuntimeError as error:
                        trial.update(status="engine_failed", failure=str(error)[:300]); break
                trial.update(metrics=env.metrics())
                if trial["status"] == "running": trial["status"] = "complete"
            output.json(f"start-{index+1}.json", row)
        _check(deadline)
        after, after_versions, _ = learning._runtime(admission, scene, deadline)
        after[Path(__file__).name] = sha256(Path(__file__).read_bytes()).hexdigest()
        if hashes != after or versions != after_versions or (target/"policy.json").read_bytes() != policy_bytes:
            raise ValueError("Evaluation runtime or frozen policy changed")
        _check(deadline)
        admitted = [row for row in rows if row["status"] == "admitted"]
        comparisons = [{"controller": name, "successes": sum(bool(trial["metrics"]["success"]) for row in admitted
                       for trial in row["trials"] if trial["controller"] == name),
                       "requested_starts": len(rows), "admitted_starts": len(admitted)} for name in CONTROLLERS]
        result = {"profile": PROFILE, "status": "complete", "scope": SCOPE, "source": protocol["source"],
            "policy_sha256": sha256(policy_bytes).hexdigest(), "protocol_sha256": canonical_hash(protocol),
            "requested_starts": len(rows), "admitted_changed_starts": len(admitted), "excluded_or_refused_starts": len(rows)-len(admitted),
            "comparisons": comparisons, "control_steps": controls, "maximum_control_steps": protocol["maximum_control_steps"],
            "changed_start_evidence": "matched_simulated_development_starts" if admitted else "unavailable_no_admitted_changed_starts",
            "final_holdout_independence": "not_established", "physical_validation": "unverified",
            "trace_scope": "initial states and final all-substep metrics; no action-journal replay claim",
            "source_reprocessing_calls": 0, "inference_calls": 0, "api_calls": 0, "seconds": time.monotonic()-started}
        output.json("result.json", result)
        output.json("manifest.json", {"profile": PROFILE, "status": "complete", "files": dict(output.files)})
        return result
    except Exception as error:
        partial = {"status": "failed", "reason": str(error)[:500], "control_steps": controls, "starts": rows, "scope": SCOPE}
        if env is not None and env._metrics is not None:
            partial["last_metrics"] = env.metrics()
        output.json("failure.json", partial)
        output.json("manifest.json", {"profile": PROFILE, "status": "failed", "files": dict(output.files)})
        raise
    finally:
        if env is not None: env.close()


def evaluate_saved_changed_starts(task_lab, protocol_file, policy_file, out, *, max_seconds=90,
        evaluation_seeds=None, position_jitter=.025, velocity_jitter=.1):
    """Admit the sealed task once, then evaluate within one admission-plus-rollout deadline."""
    from weft.moving_equipment_task_lab import load_moving_equipment_task_lab
    from weft.shared_world_packaging import read_package_file
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 90:
        raise ValueError("Saved evaluation requires a finite aggregate bound up to 90 seconds")
    started = time.monotonic(); deadline = started+max_seconds
    if (protocol_file is None) == (evaluation_seeds is None):
        raise ValueError("Choose exactly one saved protocol or explicit evaluation seeds")
    if protocol_file is not None and (position_jitter != .025 or velocity_jitter != .1):
        raise ValueError("Saved protocol jitter cannot be overridden")
    source, policy_path, target = [Path(p).absolute() for p in (task_lab, policy_file, out)]
    protocol_path = Path(protocol_file).absolute() if protocol_file is not None else None
    for path in (source, policy_path, target, *((protocol_path,) if protocol_path is not None else ())):
        if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError("Saved evaluation paths must be nonlinked and traversal-free")
    if (target.exists() or target.is_relative_to(source) or source.is_relative_to(target)
            or (protocol_path is not None and protocol_path.is_relative_to(target)) or policy_path.is_relative_to(target)):
        raise ValueError("Choose a new evaluation output isolated from all inputs")
    protocol_raw = (read_package_file(protocol_path.parent, protocol_path.name, max_bytes=MAX_BYTES)
                    if protocol_path is not None else None)
    policy_raw = read_package_file(policy_path.parent, policy_path.name, max_bytes=MAX_BYTES)
    from weft.shared_world_action_replay import _json
    protocol = _json(protocol_raw) if protocol_raw is not None else None
    config = _json(read_package_file(source, "lab-config.json", max_bytes=MAX_BYTES))
    _check(deadline)
    admitted = load_moving_equipment_task_lab(source/"run", config, max_seconds=min(30., deadline-time.monotonic()), deadline=deadline)
    _check(deadline)
    if ((protocol_path is not None and read_package_file(protocol_path.parent, protocol_path.name, max_bytes=MAX_BYTES) != protocol_raw)
            or read_package_file(policy_path.parent, policy_path.name, max_bytes=MAX_BYTES) != policy_raw):
        raise ValueError("Evaluation protocol or policy changed during source admission")
    if protocol is None:
        protocol = build_changed_start_protocol(admitted, evaluation_seeds=evaluation_seeds,
            position_jitter=position_jitter, velocity_jitter=velocity_jitter)
        protocol_raw = _bytes(protocol)
    result = evaluate_changed_starts(admitted, protocol, policy_raw, target, max_seconds=min(60., deadline-time.monotonic()))
    return result | {"aggregate_seconds": time.monotonic()-started, "source_admissions": 1,
                     "protocol_input_sha256": sha256(protocol_raw).hexdigest()}
