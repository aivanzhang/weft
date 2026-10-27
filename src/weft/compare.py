import json
import math
from pathlib import Path

from weft.io import canonical_hash, digest
from weft.pipeline import verify_artifacts


def _read_run(directory):
    root = Path(directory).resolve()
    verification = verify_artifacts(root)
    files = json.loads((root / "manifest.json").read_text())["files"]
    required = {"report.json", "evidence.json", "world.json", "task.json",
                "rl-evaluation.json", "implementation.json", "source.mp4"}
    if not required <= files.keys():
        raise ValueError("Comparison requires a complete run with all scoring inputs in its manifest")
    records = {name: json.loads((root / name).read_text()) for name in required if name.endswith(".json")}
    report = records["report.json"]
    if (report["world_hash"] != canonical_hash(records["world.json"])
            or report["task_hash"] != canonical_hash(records["task.json"])
            or report["source"]["sha256"] != digest(root / "source.mp4")):
        raise ValueError("Report identity disagrees with its scoring inputs")
    return root, records, verification


def _audit_rmse(report):
    values = [episode["after"]["metrics"] for episode in report["audit"]]
    count = sum(v["observations"] for v in values)
    return math.sqrt(sum(v["rmse_px"] ** 2 * v["observations"] for v in values) / count) if count else None


def compare_runs(baseline, candidate):
    left_path, left, left_verification = _read_run(baseline)
    right_path, right, right_verification = _read_run(candidate)
    a, b = left["report.json"], right["report.json"]
    ia, ib = left["implementation.json"], right["implementation.json"]

    def same_code(name):
        return isinstance(ia.get(name), str) and ia[name] == ib.get(name)

    replay_reasons, rl_reasons = [], []
    for key in ("source_sha256", "width", "height", "fps", "generation", "development_validation", "audit_episodes"):
        if left["evidence.json"].get(key) != right["evidence.json"].get(key):
            replay_reasons.append(f"Evidence differs: {key}")
    if a["evaluator"] != b["evaluator"] or not same_code("evaluate.py") or not same_code("world.py"):
        replay_reasons.append("Scorer or shared physics implementation differs; explicitly establish a new benchmark")
    if a["world_hash"] != b["world_hash"]:
        rl_reasons.append("Policies were evaluated in different fitted worlds, not a shared environment")
    if a["task_hash"] != b["task_hash"]:
        rl_reasons.append("Task or success definition differs")
    for name in ("env.py", "world.py", "learning.py"):
        if not same_code(name):
            rl_reasons.append(f"RL benchmark implementation differs: {name}")
    ea, eb = left["rl-evaluation.json"], right["rl-evaluation.json"]
    if ea["matched_episode_seeds"] != eb["matched_episode_seeds"] or ea["seed"] != eb["seed"]:
        rl_reasons.append("RL evaluation seeds or trial count differ")

    def metric(before, after, comparable, direction):
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value))
               for value in (before, after)):
            raise ValueError("Comparison metrics must be finite numbers or unavailable")
        return {"baseline": before, "candidate": after,
                "change": after - before if before is not None and after is not None else None,
                "comparable": comparable and before is not None and after is not None,
                "preferred_direction": direction}

    policies_a, policies_b = a["rl"]["policies"], b["rl"]["policies"]
    metrics = {
        "development_rmse_px": metric(a["reconstruction"]["after"]["rmse_px"], b["reconstruction"]["after"]["rmse_px"], not replay_reasons, "lower"),
        "audit_rmse_px": metric(_audit_rmse(a), _audit_rmse(b), not replay_reasons, "lower"),
        "rl_success_rate": metric(policies_a["q_learning"]["success_rate"], policies_b["q_learning"]["success_rate"], not rl_reasons, "higher"),
        "rl_lift_over_random": metric(policies_a["q_learning"]["success_rate"] - policies_a["random"]["success_rate"],
                                      policies_b["q_learning"]["success_rate"] - policies_b["random"]["success_rate"], not rl_reasons, "higher"),
        "elapsed_seconds": metric(a["timings"]["total_seconds"], b["timings"]["total_seconds"], False, "lower"),
        "inference_usd": metric(a["cost"]["inference_usd"], b["cost"]["inference_usd"], False, "lower"),
    }
    return {
        "schema_version": 1, "baseline": str(left_path), "candidate": str(right_path),
        "integrity": {"baseline": left_verification, "candidate": right_verification},
        "comparability": {"reconstruction": {"comparable": not replay_reasons, "reasons": replay_reasons},
                          "rl": {"comparable": not rl_reasons, "reasons": rl_reasons}},
        "metrics": metrics,
        "physics_passes": {"baseline": a["physics"]["passes"], "candidate": b["physics"]["passes"]},
        "changed_implementation_files": [name for name in sorted(ia.keys() | ib.keys()) if ia.get(name) != ib.get(name)],
        "notes": ["No overall winner is inferred from unlike metrics or benchmarks",
                  "Same-video audit is development evidence, not proof of real-world transfer",
                  "Repeatedly inspected evaluation seeds are a development benchmark, not a fresh final test",
                  "Time and cost are descriptive single-run observations, not controlled performance measurements"],
    }


def save_comparison(path, comparison):
    path = Path(path).resolve()
    if any(path.is_relative_to(Path(comparison[key])) for key in ("baseline", "candidate")):
        raise ValueError("Save comparisons outside immutable input run directories")
    with path.open("x") as stream:
        stream.write(json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n")
