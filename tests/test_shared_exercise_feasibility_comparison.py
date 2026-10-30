"""A fifth paired baseline needs its own provenance, trace and aggregate validation."""

from copy import deepcopy
import time

import mujoco
import pytest

from weft import shared_exercise_feasibility_evaluation as module
from weft import shared_exercise_feasibility_learning as learner
from weft.shared_exercise_lab import _json, _verified_inputs, _validate_policy_evaluation
from weft.shared_exercise_residual_learning import train_shared_exercise_residual_policy
from test_shared_exercise_run import source_fixture


@pytest.fixture(scope="module")
def compared(tmp_path_factory):
    root = tmp_path_factory.mktemp("paired-baseline")
    source, _ = source_fixture(root)
    old, new, evaluation = root / "old", root / "new", root / "evaluation"
    train_shared_exercise_residual_policy(source, old, seed=4001, generations=2, population=8, root_assistance=True)
    learner.train_shared_exercise_feasibility_policy(old, new)
    module.evaluate_shared_exercise_feasibility_policy(new, evaluation, seed=4002, trials=2)
    learning_manifest, learning = _verified_inputs(new, deadline=time.monotonic() + 10)
    manifest, files = _verified_inputs(evaluation, deadline=time.monotonic() + 10)
    return learning_manifest, learning, manifest, files


def inspect(compared, files=None, manifest=None, evaluation=None):
    _, learning, original_manifest, original_files = compared
    files = original_files if files is None else files
    evaluation = _json(files["evaluation.json"]) if evaluation is None else evaluation
    return module.validate_previous_reward_comparison(original_manifest if manifest is None else manifest, files, learning,
        _json(learning["policy.json"]), evaluation, _json(learning["scene.json"]), _json(learning["exercise-task.json"]),
        standard_control_steps=evaluation["control_steps"])


def reseal(manifest, files):
    document = _json(manifest)
    document["files"] = {name: module._sha(raw) for name, raw in files.items()}
    return module._bytes(document)


def test_paired_baseline_is_verified_without_physics_or_archive_revalidation(compared, monkeypatch):
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("Admission cannot run physics"))
    monkeypatch.setattr(learner, "_verify_original", lambda *args: pytest.fail("Do not reverify the whole admitted archive"))
    result = inspect(compared)
    files = compared[3]
    comparison = _json(files["comparison.json"])
    assert result["previous_reward_comparison_verified"] is True
    assert result["previous_reward_comparison_sha256"] == module._sha(files["comparison.json"])
    assert result["previous_reward_baseline"] == {"successes": comparison["successes"], "episodes": 2,
        "completed_intervals": comparison["completed_intervals"], "mean_score": comparison["mean_score"]}


def test_generic_admission_attests_standard_and_previous_comparisons(compared):
    learning_manifest, learning, manifest, files = compared
    identity, successes, _ = _validate_policy_evaluation(manifest, files, learning_manifest, learning, _json(learning["policy.json"]))
    assert set(successes) == {"coast", "random", "authored", "learned"}
    assert identity["previous_reward_comparison_verified"] is True
    assert identity["previous_reward_comparison_sha256"] == module._sha(files["comparison.json"])


@pytest.mark.parametrize("kind", ["missing", "failed", "few_trials", "seed", "reset", "initial", "metrics", "last_hash",
    "action", "early_terminal", "reward", "control_count", "success_count", "mean_score", "physical", "old_policy", "old_proof", "hash"])
def test_resealed_false_baseline_cannot_be_admitted(compared, kind):
    files = dict(compared[3])
    comparison = _json(files["comparison.json"])
    evaluation = _json(files["evaluation.json"])
    trial = comparison["trials"][0]
    if kind == "missing":
        files.pop("comparison.json")
    elif kind == "failed":
        comparison["status"] = "failed"
    elif kind == "few_trials":
        comparison["trials"].pop()
    elif kind == "seed":
        trial["seed"] += 1
    elif kind == "reset":
        trial["reset_options"]["jitter_scale"] += .01
    elif kind == "initial":
        trial["initial_state_sha256"] = "0" * 64
    elif kind == "metrics":
        trial["metrics"]["equipment"][0]["speed_model_units_s"] = 100
    elif kind == "last_hash":
        trial["final_snapshot_sha256"] = "0" * 64
    elif kind == "action":
        trial["actions"][0]["action"].append(1.)
    elif kind == "early_terminal":
        trial["actions"][0]["terminated"] = True
    elif kind == "reward":
        trial["actions"][0]["reward"] += .5
    elif kind == "control_count":
        comparison["control_steps"] += 1
    elif kind == "success_count":
        comparison["successes"] += 1
    elif kind == "mean_score":
        comparison["mean_score"] += .1
    elif kind == "physical":
        comparison["physical_validation"] = "verified"
    elif kind == "old_policy":
        policy = _json(files["previous-policy.json"])
        policy["residual_coefficients"][0][0] += .01
        files["previous-policy.json"] = module._bytes(policy)
    elif kind == "old_proof":
        proof = _json(files["previous-baseline-proof.json"])
        proof["actor_equivalence"]["old_policy_admitted"] = True
        files["previous-baseline-proof.json"] = module._bytes(proof)
    else:
        evaluation["previous_reward_comparison_sha256"] = "0" * 64
    if kind != "missing":
        files["comparison.json"] = module._bytes(comparison)
        if kind != "hash":
            evaluation["previous_reward_comparison_sha256"] = module._sha(files["comparison.json"])
    files["evaluation.json"] = module._bytes(evaluation)
    with pytest.raises(ValueError):
        inspect(compared, files, reseal(compared[2], files))


@pytest.mark.parametrize("kind", ["budget", "aggregate", "frozen_plan", "source_identity", "reused_seed"])
def test_pairing_and_frozen_work_bounds_cannot_be_rewritten(compared, kind):
    files = dict(compared[3])
    evaluation, freeze, comparison = [_json(files[name]) for name in ("evaluation.json", "pre-evaluation-freeze.json", "comparison.json")]
    if kind == "budget":
        freeze["aggregate_maximum_control_steps"] += 1
        evaluation["aggregate_maximum_control_steps"] += 1
        files["settings.json"] = module._bytes(freeze)
    elif kind == "aggregate":
        evaluation["aggregate_control_steps"] += 1
    elif kind == "frozen_plan":
        freeze["previous_reward_baseline"]["maximum_control_steps"] += 1
    elif kind == "source_identity":
        evaluation["previous_reward_baseline"]["training_manifest_sha256"] = "0" * 64
    else:
        evaluation["training_seeds"].append(evaluation["evaluation_seeds"][0])
    files["evaluation.json"], files["pre-evaluation-freeze.json"] = module._bytes(evaluation), module._bytes(freeze)
    with pytest.raises(ValueError):
        inspect(compared, files, reseal(compared[2], files))


def test_unsealed_comparison_mutation_rejects_before_parsing(compared):
    files = dict(compared[3])
    files["comparison.json"] = b"not JSON"
    with pytest.raises(ValueError, match="sealed"):
        inspect(compared, files)


def test_baseline_trial_full_completion_is_checked_by_unchanged_gates(compared):
    files = dict(compared[3])
    comparison, evaluation = _json(files["comparison.json"]), _json(files["evaluation.json"])
    trial = comparison["trials"][0]
    trial["metrics"]["completed_interval"] = False
    trial["status"] = "failed"
    comparison["completed_intervals"] -= 1
    files["comparison.json"] = module._bytes(comparison)
    evaluation["previous_reward_comparison_sha256"] = module._sha(files["comparison.json"])
    files["evaluation.json"] = module._bytes(evaluation)
    with pytest.raises(ValueError):
        inspect(compared, files, reseal(compared[2], files))
