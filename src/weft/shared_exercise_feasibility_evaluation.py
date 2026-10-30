"""Freeze feasibility-first policies before independent changed-start comparisons."""

from pathlib import Path
import math
import time

from weft.io import canonical_hash, digest, local_time
from weft.shared_exercise_feasibility_learning import SCOPE, _runtime_hashes
from weft.shared_exercise_lab import _json, _verified_inputs, _validate_policy_package
from weft.shared_exercise_learning import _context, _integer, _output_path, _score, _seconds, _seed_plan, _sha, _write, _bytes
from weft.shared_exercise_reset import human_jitter_options
from weft.shared_exercise_residual_learning import (
    CONTROLLERS, MAX_EVALUATION_STEPS, MAX_FULLSPAN_EVALUATION_STEPS,
    _Budget, _check, _coefficients, _excluded_seeds, _failure, _initialize, _input_unchanged,
    _seal, _source_views, _trial,
)
from weft.shared_exercise_task import SharedExerciseTaskEnv, prepare_shared_exercise_snapshot_validator
from weft.shared_world_packaging import collect_shared_source_files, read_package_file


def evaluate_shared_exercise_feasibility_policy(learning_dir, out, *, seed=924017, trials=3,
                                           jitter_scale=.10, max_seconds=30, excluded_seeds=()):
    """Freeze selected policy bytes and fresh reset seeds before any comparison state is observed."""
    _integer(seed, "evaluation seed", 0, 2**31 - 1)
    _integer(trials, "evaluation trials", 1, 3)
    _seconds(max_seconds, 60)
    excluded = _excluded_seeds(excluded_seeds)
    if type(jitter_scale) not in (int, float) or not math.isfinite(jitter_scale) or not 0 < jitter_scale <= .25:
        raise ValueError("Changed-start evaluation jitter_scale must be explicit in (0,.25]")
    source, out = Path(learning_dir).resolve(), _output_path(learning_dir, out)
    started = time.monotonic()
    deadline = started + max_seconds
    manifest, files = _verified_inputs(source, deadline=deadline)
    required = {"policy.json", "training.json", "settings.json", "scene.json", "exercise-task.json"}
    if _json(manifest).get("status") != "trained" or not required <= set(files):
        raise ValueError("Feasibility evaluation requires completed sealed training, never a failed checkpoint")
    scene_raw, task_raw, policy_raw = files["scene.json"], files["exercise-task.json"], files["policy.json"]
    scene, task, _ = _context(_json(scene_raw), _json(task_raw))
    fullspan = scene["profile"] == "shared-world-v2"
    _seconds(max_seconds, 60 if fullspan else 30)
    hashes = _runtime_hashes()
    policy, _, previous = _validate_policy_package(manifest, files, scene, task, include_baseline=True)
    if policy["profile"] != "shared-exercise-policy-v3" or previous is None:
        raise ValueError("Feasibility evaluation requires a current v3 policy and verified previous baseline")
    settings, training = _json(files["settings.json"]), _json(files["training.json"])
    training_seeds = settings.get("training_seeds")
    if (training.get("status") != "trained" or training.get("policy_sha256") != _sha(policy_raw)
            or not isinstance(training_seeds, list) or len(training_seeds) != 2
            or any(type(value) is not int or not 0 <= value < 2**31 for value in training_seeds)
            or len(set(training_seeds)) != 2 or seed == settings.get("seed")
            or any(settings.get(name) != policy[name] for name in ("scene_sha256", "task_sha256", "gates_sha256", "runtime_sha256"))):
        raise ValueError("Feasibility evaluation requires a frozen policy and independent seed stream")
    maximum_steps = len(CONTROLLERS) * trials * task["interval"]["horizon_steps"]
    baseline_steps = trials * task["interval"]["horizon_steps"]
    maximum_allowed = (MAX_FULLSPAN_EVALUATION_STEPS if fullspan else MAX_EVALUATION_STEPS) * 5 // 4
    if maximum_steps + baseline_steps > maximum_allowed:
        raise ValueError(f"Feasibility evaluation exceeds {maximum_allowed} full-interval actions")
    seeds = _seed_plan(seed, trials, [*training_seeds, *excluded])
    train_states = {trial["initial_state_sha256"] for generation in training["history"]
                    for row in generation["candidates"] for trial in row["trials"] if trial["initial_state_sha256"] is not None}
    proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(manifest))
    views = _source_views(source, manifest)
    reset = human_jitter_options(float(jitter_scale))
    evaluation = {"schema_version": 1, "status": "running", "created_at": local_time(), "policy_sha256": _sha(policy_raw),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "runtime_sha256": hashes,
        "source_views_sha256": {name: _sha(raw) for name, raw in views.items()},
        "seed": seed, "evaluation_seeds": seeds, "training_seeds": training_seeds, "excluded_seeds": excluded,
        "reset_options": reset, "horizon_steps": task["interval"]["horizon_steps"], "maximum_control_steps": maximum_steps,
        "max_seconds": max_seconds, "workers": 1, "external_api_usd": 0, "control_steps": 0,
        "evaluation_feedback_to_optimizer": False, "source_scope": task["source_scope"], "scope": SCOPE,
        "controllers": {}, "initial_states": {}}
    if fullspan:
        evaluation["evaluation_protocol"] = "shared-exercise-full-reference-evaluation-v1"
    evaluation["policy_profile"] = policy["profile"]
    evaluation["training_algorithm"] = policy["algorithm"]
    evaluation["actor_algorithm"] = policy["actor_algorithm"]
    baseline_plan = {key: previous[key] for key in (
        "policy_sha256", "training_manifest_sha256", "actor_algorithm", "training_algorithm", "import_proof_sha256")}
    baseline_plan.update(maximum_control_steps=baseline_steps, evaluation_seeds=seeds,
        reset_options=reset, scope="Original reward-selected coefficients through the identical actor/core; archived v2 policy is not admitted as a current policy.")
    evaluation["previous_reward_baseline"] = baseline_plan
    evaluation["aggregate_maximum_control_steps"] = maximum_steps + baseline_steps
    freeze = {key: value for key, value in evaluation.items()
              if key not in ("status", "control_steps", "controllers", "initial_states")}
    _check(deadline)
    _initialize(out, scene_raw, task_raw, manifest, proof,
                {**views, "policy.json": policy_raw, "pre-evaluation-freeze.json": _bytes(freeze),
                 "previous-policy.json": files[previous["source_path"]],
                 "previous-baseline-proof.json": _bytes(previous)})
    env, budget = None, _Budget(maximum_steps)
    baseline_budget = _Budget(baseline_steps)
    comparison = {"schema_version": 1, "status": "running", "plan": baseline_plan, "trials": [],
                  "evaluation_feedback_to_optimizer": False, "control_steps": 0}
    try:
        _write(out, "settings.json", freeze)
        env = SharedExerciseTaskEnv(scene, task)
        validator = prepare_shared_exercise_snapshot_validator(scene, task)
        coefficients = _coefficients(policy["residual_coefficients"])
        for controller in CONTROLLERS:
            rows = []
            evaluation["controllers"][controller] = {"trials": rows}
            for trial_seed in seeds:
                try:
                    row = _trial(env, controller, trial_seed, reset, coefficients, deadline, budget,
                                 trace=True, validator=validator)
                except Exception as error:
                    if hasattr(error, "partial_trial"):
                        rows.append(error.partial_trial)
                    raise
                rows.append(row)
                evaluation["control_steps"] = budget.used
                initial_hash = row["initial_state_sha256"]
                if initial_hash is None:
                    continue
                if initial_hash in train_states:
                    raise ValueError("Feasibility evaluation repeated a training state; do not replace its seed")
                previous_initial_hash = evaluation["initial_states"].setdefault(str(trial_seed), initial_hash)
                if previous_initial_hash != initial_hash:
                    raise RuntimeError("Matched residual controllers received different seeded initial states")
            if len(set(evaluation["initial_states"].values())) != len(evaluation["initial_states"]):
                raise ValueError("Feasibility evaluation seeds did not produce distinct changed initial states")
            evaluation["controllers"][controller].update(
                successes=sum(bool(row["metrics"] and row["metrics"]["success"]) for row in rows),
                completed_intervals=sum(bool(row["metrics"] and row["metrics"]["completed_interval"]) for row in rows),
                reset_failures=sum(row["status"] == "reset_failed" for row in rows), mean_score=_score(rows))
            _write(out, "evaluation.json", evaluation)
        previous_coefficients = _coefficients(previous["policy"]["residual_coefficients"])
        for trial_seed in seeds:
            try:
                row = _trial(env, "learned", trial_seed, reset, previous_coefficients, deadline, baseline_budget,
                             trace=True, validator=validator)
            except Exception as error:
                if hasattr(error, "partial_trial"):
                    comparison["trials"].append(error.partial_trial)
                raise
            comparison["trials"].append(row)
            comparison["control_steps"] = baseline_budget.used
            if row["initial_state_sha256"] != evaluation["initial_states"].get(str(trial_seed)):
                raise RuntimeError("Previous reward baseline did not receive the matched independent start")
            _write(out, "comparison.json", comparison)
        comparison.update(status="evaluated", successes=sum(bool(row["metrics"] and row["metrics"]["success"])
            for row in comparison["trials"]),
            completed_intervals=sum(bool(row["metrics"] and row["metrics"]["completed_interval"])
                for row in comparison["trials"]),
            mean_score=_score(comparison["trials"]), physical_validation="unverified")
        _write(out, "comparison.json", comparison)
        _input_unchanged(source, manifest, deadline)
        if _runtime_hashes() != hashes:
            raise RuntimeError("Frozen residual policy or evaluator runtime changed during evaluation")
        read_package_file(out, "policy.json", max_bytes=64 * 1024, expected_sha256=_sha(policy_raw))
        read_package_file(out, "previous-policy.json", max_bytes=64 * 1024, expected_sha256=previous["policy_sha256"])
        read_package_file(out, "pre-evaluation-freeze.json", max_bytes=64 * 1024, expected_sha256=_sha(_bytes(freeze)))
        _check(deadline)
        evaluation.update(status="evaluated", seconds=time.monotonic() - started,
            aggregate_control_steps=budget.used + baseline_budget.used,
            previous_reward_comparison_sha256=digest(out / "comparison.json"),
            independent_changed_states=len(evaluation["initial_states"]), policy_unchanged=True,
            comparison_complete=all(row["initial_state_sha256"] is not None
                for summary in evaluation["controllers"].values() for row in summary["trials"]),
            physical_validation="unverified")
        _write(out, "evaluation.json", evaluation)
        return {"directory": str(out), "manifest_sha256": _seal(out, "evaluated", deadline), **evaluation}
    except Exception as error:
        comparison.update(status="failed", control_steps=baseline_budget.used,
                          error=f"{type(error).__name__}: {error}"[:2000])
        _write(out, "comparison.json", comparison)
        evaluation.update(control_steps=budget.used, aggregate_control_steps=budget.used + baseline_budget.used,
                          seconds=time.monotonic() - started)
        _failure(out, "evaluation.json", evaluation, error)
        raise
    finally:
        if env is not None:
            env.close()


def validate_previous_reward_comparison(manifest_raw, files, learning_files, policy, evaluation, scene, task, *, standard_control_steps):
    """Attest the fifth paired baseline without admitting its archived policy or replaying physics."""
    import re
    from weft.shared_exercise_selection import _trial as validate_trial
    from weft.shared_exercise_feasibility_learning import ACTOR_ALGORITHM, ALGORITHM, POLICY_PROFILE
    from weft.shared_exercise_residual_learning import ALGORITHM as ORIGINAL_ALGORITHM

    required = {"comparison.json", "previous-policy.json", "previous-baseline-proof.json", "pre-evaluation-freeze.json", "settings.json"}
    manifest = _json(manifest_raw)
    if (policy.get("profile") != POLICY_PROFILE or not isinstance(manifest, dict) or manifest.get("status") != "evaluated"
            or not isinstance(manifest.get("files"), dict) or not required <= set(files) or not required <= set(manifest["files"])
            or any(len(files[name]) > (8 * 1024 * 1024 if name == "comparison.json" else 512 * 1024)
                   or _sha(files[name]) != manifest["files"][name] for name in required)):
        raise ValueError("V3 evaluation requires the complete sealed previous-reward comparison")
    imported = "imported-training/"
    needed = {imported + "policy.json", imported + "manifest.json", "import-proof.json"}
    if not needed <= set(learning_files):
        raise ValueError("Previous-reward comparison lacks the already-validated learning import proof")
    original_raw = learning_files[imported + "policy.json"]
    original_policy = _json(original_raw)
    import_proof = _json(learning_files["import-proof.json"])
    import_hash, training_hash = _sha(learning_files["import-proof.json"]), _sha(learning_files[imported + "manifest.json"])
    if (files["previous-policy.json"] != original_raw or policy["import_proof_sha256"] != import_hash
            or policy["source_training_manifest_sha256"] != training_hash
            or import_proof.get("original_policy_sha256") != _sha(original_raw)
            or import_proof.get("original_training_manifest_sha256") != training_hash
            or original_policy.get("algorithm") != ORIGINAL_ALGORITHM):
        raise ValueError("Previous-reward comparison differs from the admitted original policy and import proof")
    expected_proof = {"policy": original_policy, "policy_sha256": _sha(original_raw), "training_manifest_sha256": training_hash,
        "actor_algorithm": ACTOR_ALGORITHM, "training_algorithm": ORIGINAL_ALGORITHM,
        "source_path": imported + "policy.json", "import_proof_sha256": import_hash,
        "actor_equivalence": {"actor_ast_sha256": import_proof["actor_ast_sha256"],
            "unchanged_training_execution_runtime_sha256": import_proof["unchanged_training_execution_runtime_sha256"],
            "old_policy_admitted": False,
            "comparison_method": "Original coefficients through the current identical actor, not old-policy runtime migration"}}
    if _json(files["previous-baseline-proof.json"]) != expected_proof:
        raise ValueError("Previous-reward baseline provenance cannot replace the verified historical-actor proof")
    seeds, reset, initial = evaluation["evaluation_seeds"], evaluation["reset_options"], evaluation["initial_states"]
    if (not isinstance(seeds, list) or not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in seeds)
            or set(seeds) & set(evaluation["training_seeds"]) or set(seeds) & set(evaluation.get("excluded_seeds", []))
            or not isinstance(initial, dict) or set(initial) != {str(seed) for seed in seeds}
            or len(set(initial.values())) != len(seeds)):
        raise ValueError("Previous-reward comparison requires the same fresh distinct evaluation starts")
    maximum = len(seeds) * task["interval"]["horizon_steps"]
    plan = {name: expected_proof[name] for name in ("policy_sha256", "training_manifest_sha256", "actor_algorithm",
                                                  "training_algorithm", "import_proof_sha256")}
    plan.update(maximum_control_steps=maximum, evaluation_seeds=seeds, reset_options=reset,
        scope="Original reward-selected coefficients through the identical actor/core; archived v2 policy is not admitted as a current policy.")
    freeze, settings, comparison = (_json(files[name]) for name in ("pre-evaluation-freeze.json", "settings.json", "comparison.json"))
    if (not isinstance(freeze, dict) or freeze != settings or any(evaluation.get(name) != value for name, value in freeze.items())
            or evaluation.get("previous_reward_baseline") != plan or freeze.get("previous_reward_baseline") != plan
            or freeze.get("aggregate_maximum_control_steps") != (len(CONTROLLERS) + 1) * maximum
            or freeze.get("maximum_control_steps") != len(CONTROLLERS) * maximum
            or freeze.get("policy_profile") != POLICY_PROFILE or freeze.get("training_algorithm") != ALGORITHM
            or freeze.get("actor_algorithm") != ACTOR_ALGORITHM or freeze.get("evaluation_feedback_to_optimizer") is not False
            or evaluation.get("previous_reward_comparison_sha256") != _sha(files["comparison.json"])):
        raise ValueError("Previous-reward comparison hash, seeds or aggregate budget differs from its pre-evaluation freeze")
    keys = {"schema_version", "status", "plan", "trials", "evaluation_feedback_to_optimizer", "control_steps",
            "successes", "completed_intervals", "mean_score", "physical_validation"}
    if (not isinstance(comparison, dict) or set(comparison) != keys or type(comparison["schema_version"]) is not int
            or comparison["schema_version"] != 1 or comparison["status"] != "evaluated" or comparison["plan"] != plan
            or comparison["evaluation_feedback_to_optimizer"] is not False or comparison["physical_validation"] != "unverified"
            or not isinstance(comparison["trials"], list) or len(comparison["trials"]) != len(seeds)):
        raise ValueError("Previous-reward comparison requires every completed paired trial and explicit unverified physical scope")
    validate = prepare_shared_exercise_snapshot_validator(scene, task)
    action_count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
    steps = 0
    for seed, row in zip(seeds, comparison["trials"], strict=True):
        if not isinstance(row, dict) or not {"initial_snapshot", "final_snapshot", "actions"} <= set(row):
            raise ValueError("Previous-reward trial lacks full boundary snapshots and action trace")
        basic = {name: value for name, value in row.items() if name not in {"initial_snapshot", "final_snapshot", "actions"}}
        validate_trial(basic, scene, task, policy, seed, reset)
        first, final, actions = row["initial_snapshot"], row["final_snapshot"], row["actions"]
        validate(first)
        validate(final)
        if (row["initial_state_sha256"] != initial[str(seed)] or canonical_hash(first["engine_state"]) != initial[str(seed)]
                or first["engine_state_status"] != "current" or first["engine_state"]["steps"] != 0
                or first["engine_state"]["episode_steps"] != 0 or first["engine_state"]["time_s"] != 0
                or any(first["last_action"]) or first["terminated"] or first["truncated"]
                or row["initial_metrics"] != first["task_metrics"] or row["metrics"] != final["task_metrics"]
                or first["resolved_reset_options"] != reset or final["resolved_reset_options"] != reset
                or row["final_snapshot_sha256"] != canonical_hash(final)
                or row["terminated"] is not final["terminated"] or row["truncated"] is not final["truncated"]
                or not isinstance(actions, list) or len(actions) != row["control_steps"]
                or not 1 <= len(actions) <= task["interval"]["horizon_steps"]):
            raise ValueError("Previous-reward trial does not preserve its paired reset, final state and complete action count")
        reward_sum = 0.
        for index, action in enumerate(actions):
            if (not isinstance(action, dict) or set(action) != {"action", "reward", "terminated", "truncated", "snapshot_sha256"}
                    or not isinstance(action["action"], list) or len(action["action"]) != action_count
                    or any(type(value) not in (int, float) or not math.isfinite(value) or abs(value) > 1 for value in action["action"])
                    or type(action["reward"]) not in (int, float) or not math.isfinite(action["reward"])
                    or type(action["terminated"]) is not bool or type(action["truncated"]) is not bool
                    or action["terminated"] and action["truncated"]
                    or index < len(actions) - 1 and (action["terminated"] or action["truncated"])
                    or not isinstance(action["snapshot_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", action["snapshot_sha256"])):
                raise ValueError("Previous-reward action trace contains invalid actions, clocks or terminal flags")
            reward_sum += action["reward"]
        if (actions[-1]["snapshot_sha256"] != canonical_hash(final) or actions[-1]["action"] != final["last_action"]
                or actions[-1]["terminated"] is not final["terminated"] or actions[-1]["truncated"] is not final["truncated"]
                or not math.isclose(reward_sum, row["return"], rel_tol=0, abs_tol=1e-8)):
            raise ValueError("Previous-reward final action or accumulated reward differs from its validated trial")
        steps += len(actions)
    successes = sum(row["metrics"]["success"] for row in comparison["trials"])
    completed = sum(row["metrics"]["completed_interval"] for row in comparison["trials"])
    expected_counts = {"control_steps": steps, "successes": successes, "completed_intervals": completed}
    if (any(type(comparison.get(name)) is not int or comparison[name] != value for name, value in expected_counts.items())
            or type(comparison["mean_score"]) not in (int, float) or not math.isfinite(comparison["mean_score"])
            or comparison["mean_score"] != _score(comparison["trials"])
            or type(standard_control_steps) is not int or standard_control_steps != evaluation.get("control_steps")
            or not 0 <= standard_control_steps <= len(CONTROLLERS) * maximum or not 0 <= steps <= maximum
            or type(evaluation.get("aggregate_control_steps")) is not int
            or evaluation["aggregate_control_steps"] != standard_control_steps + steps
            or evaluation["aggregate_control_steps"] > freeze["aggregate_maximum_control_steps"]):
        raise ValueError("Previous-reward comparison summaries or aggregate action accounting differ from its validated trials")
    return {"previous_reward_comparison_sha256": _sha(files["comparison.json"]),
            "previous_reward_comparison_verified": True,
            "previous_reward_baseline": {"successes": successes, "episodes": len(seeds), "completed_intervals": completed,
                                         "mean_score": comparison["mean_score"]}}
