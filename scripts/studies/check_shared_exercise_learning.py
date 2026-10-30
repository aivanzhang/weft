"""One frozen local gain-search run and one independent changed-start comparison."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from weft.io import digest, local_time, write_json
from weft.shared_exercise_learning import train_shared_exercise_policy, evaluate_shared_exercise_policy
from weft.shared_world_packaging import read_package_file, read_package_manifest


def run(source, out, *, task_path=None, root_assistance=False):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists() or out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Choose a new study directory outside the immutable source")
    inputs = source / "run" if (source / "run/manifest.json").is_file() else source
    manifest, files = read_package_manifest(inputs)
    if "scene.json" not in files:
        raise ValueError("Study source must seal its scene.json")
    scene_raw = read_package_file(inputs, "scene.json", max_bytes=8 * 1024 * 1024, expected_sha256=files.get("scene.json"))
    scene = json.loads(scene_raw)
    horizon = scene["task"]["horizon_steps"]
    if type(horizon) is not int or 48 * horizon > 20_000 or 12 * horizon > 3_000:
        raise ValueError("The complete task interval exceeds this fixed study's control-step bound")
    plan = {"schema_version": 1, "created_at": local_time(), "source_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "source_scene_sha256": hashlib.sha256(scene_raw).hexdigest(), "training_seed": 1729, "evaluation_seed": 924017,
            "generations": 3, "population": 8, "training_reset_trials": 2, "changed_start_seeds": 3,
            "training_reset_jitter_scales": [0., .05], "evaluation_reset_jitter_scale": .10,
            "maximum_training_control_steps": 48 * horizon, "maximum_evaluation_control_steps": 12 * horizon,
            "horizon_steps": horizon, "declared_interval_s": horizon * scene["dt"], "workers": 1,
            "max_training_seconds": 60, "max_evaluation_seconds": 30, "max_study_seconds": 95,
            "maximum_study_bytes": 256 * 1024 * 1024, "external_api_usd": 0,
            "scope": "One development comparison on the unchanged declared interval; no retries, seed selection, "
                     "evaluation-driven tuning, full-exercise claim or physical transfer claim."}
    out.mkdir(parents=True, mode=0o700)
    write_json(out / "pre-training-freeze.json", plan)
    (out / "study-driver.py").write_bytes(Path(__file__).read_bytes())
    started = time.monotonic()
    result = {"status": "running", "plan": plan}
    try:
        training = train_shared_exercise_policy(source, out / "training", task_path=task_path,
            root_assistance=root_assistance, seed=1729, generations=3, population=8, max_seconds=60)
        result["training"] = {key: training[key] for key in (
            "manifest_sha256", "policy_sha256", "seconds", "control_steps", "evaluated_episodes", "cache_hits",
            "selected_feedback_kind", "improvement_over_authored_training")}
        evaluation = evaluate_shared_exercise_policy(out / "training", out / "evaluation",
            seed=924017, trials=3, jitter_scale=.10, max_seconds=30)
        result["evaluation"] = {key: evaluation[key] for key in (
            "manifest_sha256", "policy_sha256", "seconds", "control_steps", "comparison_complete", "independent_changed_states")}
        result["evaluation"]["controllers"] = {name: {key: value for key, value in summary.items() if key != "trials"}
                                               for name, summary in evaluation["controllers"].items()}
        read_package_file(inputs, "manifest.json", max_bytes=8 * 1024 * 1024,
                          expected_sha256=plan["source_manifest_sha256"])
        read_package_file(inputs, "scene.json", max_bytes=8 * 1024 * 1024, expected_sha256=plan["source_scene_sha256"])
        if time.monotonic() - started > plan["max_study_seconds"]:
            raise TimeoutError("Study exceeded its fixed total wall-time bound")
        result["status"] = "completed"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}",
                      scope="Stopped without retries or replacement seeds; all partial child evidence retained.")
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        write_json(out / "report.json", result)
        paths = [path for path in out.rglob("*") if path.is_file() and path != out / "manifest.json"]
        if len(paths) <= 1100 and sum(path.stat().st_size for path in paths) <= plan["maximum_study_bytes"]:
            write_json(out / "manifest.json", {"schema_version": 1, "status": result["status"],
                "files": {path.relative_to(out).as_posix(): digest(path) for path in sorted(paths)}})
        else:
            result.update(status="failed", output_bound_exceeded=True)
            write_json(out / "report.json", result)
            write_json(out / "failure.json", {"status": "failed", "reason": "Study exceeded its output bound; files retained unsealed"})
        print(json.dumps(result, indent=2))
        if result.get("output_bound_exceeded"):
            raise ValueError("Study output exceeded its bound; retained as a failed unsealed run")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", required=True)
    task = parser.add_mutually_exclusive_group()
    task.add_argument("--task")
    task.add_argument("--root-assistance", action="store_true")
    args = parser.parse_args()
    run(args.run, args.out, task_path=args.task, root_assistance=args.root_assistance)
