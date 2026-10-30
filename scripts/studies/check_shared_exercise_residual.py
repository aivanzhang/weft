"""One frozen residual-policy experiment on unchanged equipment and the complete declared task interval."""

import argparse
import json
from pathlib import Path
import time

from weft.io import canonical_hash, local_time
from weft.shared_exercise_residual_learning import (
    ALGORITHM, BASIS_PROFILE, MAX_CONTROL_STEPS, MAX_EVALUATION_STEPS, RESIDUAL_BOUND, TRAIN_RESETS,
    _check, _excluded_seeds, _failure, _input_unchanged, _integer, _runtime_hashes, _seal,
    _seed_plan, _sha, _source, _terminal_manifest, _write,
    evaluate_shared_exercise_residual_policy, train_shared_exercise_residual_policy,
)
from weft.shared_exercise_reset import human_jitter_options


def run(source, out, *, task_path=None, root_assistance=False, training_seed=1729,
        evaluation_seed=924017, excluded_seeds=()):
    _integer(training_seed, "training seed", 0, 2**31 - 1)
    _integer(evaluation_seed, "evaluation seed", 0, 2**31 - 1)
    excluded = _excluded_seeds(excluded_seeds)
    if training_seed == evaluation_seed:
        raise ValueError("Choose distinct training and evaluation seed streams before the residual study")
    source, out = Path(source).resolve(), Path(out).resolve()
    task_path = Path(task_path).absolute() if task_path is not None else None
    inputs = [source] + ([task_path] if task_path is not None else [])
    if out.exists() or any(out == path or out.is_relative_to(path) or path.is_relative_to(out) for path in inputs):
        raise ValueError("Choose a new residual study directory outside all immutable inputs")
    started = time.monotonic()
    deadline = started + 95
    inputs, scene, task, scene_raw, task_raw, manifest, _ = _source(source, task_path, root_assistance, deadline)
    _terminal_manifest(manifest)
    horizon = task["interval"]["horizon_steps"]
    if 144 * horizon > MAX_CONTROL_STEPS or 12 * horizon > MAX_EVALUATION_STEPS:
        raise ValueError("The complete task interval exceeds this fixed residual study's action bound")
    training_seeds = _seed_plan(training_seed, 2)
    evaluation_seeds = _seed_plan(evaluation_seed, 3, [*training_seeds, *excluded])
    hashes = _runtime_hashes()
    plan = {"schema_version": 1, "created_at": local_time(), "algorithm": ALGORITHM,
        "source": str(inputs), "source_manifest_sha256": _sha(manifest), "source_scene_file_sha256": _sha(scene_raw),
        "scene_sha256": canonical_hash(scene), "task_sha256": canonical_hash(task), "task_file_sha256": _sha(task_raw),
        "runtime_sha256": hashes, "training_seed": training_seed, "evaluation_seed": evaluation_seed,
        "training_seeds": training_seeds, "evaluation_seeds": evaluation_seeds, "excluded_seeds": excluded,
        "generations": 6, "population": 12, "training_reset_trials": 2, "changed_start_seeds": 3,
        "training_reset_options": list(TRAIN_RESETS), "evaluation_reset_options": human_jitter_options(.10),
        "basis_profile": BASIS_PROFILE, "coefficient_shape": [2, 19], "residual_bound": RESIDUAL_BOUND,
        "maximum_training_control_steps": 144 * horizon, "maximum_evaluation_control_steps": 12 * horizon,
        "maximum_training_physics_substeps": 144 * horizon * scene["substeps"],
        "maximum_evaluation_physics_substeps": 12 * horizon * scene["substeps"],
        "horizon_steps": horizon, "declared_interval_s": horizon * scene["dt"], "workers": 1,
        "max_training_seconds": 60, "max_evaluation_seconds": 30, "max_study_seconds": 95,
        "maximum_study_bytes": 256 * 1024 * 1024, "maximum_study_files": 1100, "external_api_usd": 0,
        "scope": "One frozen base-world comparison. All anchors are inside population; no retries, task shortening, "
                 "equipment changes, evaluation-driven tuning, new-video generalization or physical success claim."}
    _check(deadline)
    out.mkdir(parents=True, mode=0o700)
    result = {"status": "running", "plan": plan}
    try:
        _write(out, "pre-training-freeze.json", plan)
        (out / "study-driver.py").write_bytes(Path(__file__).read_bytes())
        training = train_shared_exercise_residual_policy(source, out / "training", task_path=task_path,
            root_assistance=root_assistance, seed=training_seed, generations=6, population=12,
            max_seconds=min(60., deadline - time.monotonic()))
        result["training"] = {key: training[key] for key in (
            "manifest_sha256", "policy_sha256", "seconds", "control_steps", "evaluated_episodes", "cache_hits",
            "selected_feedback_kind", "improvement_over_authored_training", "training_score_improved_over_authored")}
        _check(deadline)
        evaluation = evaluate_shared_exercise_residual_policy(out / "training", out / "evaluation",
            seed=evaluation_seed, trials=3, jitter_scale=.10, max_seconds=min(30., deadline - time.monotonic()),
            excluded_seeds=excluded)
        if evaluation["evaluation_seeds"] != evaluation_seeds or evaluation["training_seeds"] != training_seeds:
            raise ValueError("Residual evaluation changed the pre-training frozen seed plan")
        result["evaluation"] = {key: evaluation[key] for key in (
            "manifest_sha256", "policy_sha256", "seconds", "control_steps", "comparison_complete", "independent_changed_states")}
        result["evaluation"]["controllers"] = {name: {key: value for key, value in summary.items() if key != "trials"}
                                               for name, summary in evaluation["controllers"].items()}
        _input_unchanged(inputs, manifest, deadline, task_path=task_path, task_raw=task_raw)
        if _runtime_hashes() != hashes:
            raise RuntimeError("Residual study changed its pre-training frozen runtime")
        _check(deadline)
        result.update(status="completed", seconds=time.monotonic() - started)
        _write(out, "report.json", result)
        _seal(out, "completed", deadline, maximum_bytes=plan["maximum_study_bytes"],
              maximum_files=plan["maximum_study_files"])
    except Exception as error:
        result.update(seconds=time.monotonic() - started,
                      scope="Stopped without retries or replacement seeds; all partial child evidence retained.")
        _failure(out, "report.json", result, error)
        raise
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--training-seed", type=int, default=1729)
    parser.add_argument("--evaluation-seed", type=int, default=924017)
    parser.add_argument("--exclude-seed", type=int, action="append", default=[],
                        help="Previously observed reset seed to exclude from the frozen evaluation plan")
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--task")
    task.add_argument("--root-assistance", action="store_true")
    args = parser.parse_args()
    run(args.run, args.out, task_path=args.task, root_assistance=args.root_assistance,
        training_seed=args.training_seed, evaluation_seed=args.evaluation_seed, excluded_seeds=args.exclude_seed)
