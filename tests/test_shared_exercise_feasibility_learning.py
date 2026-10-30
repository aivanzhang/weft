"""Feasibility refinement preserves original evidence, actions, gates and distinct optimizer identity."""

from copy import deepcopy
import json
from pathlib import Path
import time

import mujoco
import numpy as np
import pytest

from weft import shared_exercise_feasibility_learning as module
from weft import shared_exercise_residual_learning as original
from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
from weft.shared_exercise_lab import _json, _validate_policy_package, _verified_inputs
from weft.shared_exercise_task import GATES, SharedExerciseTaskEnv
from test_shared_exercise_run import source_fixture


@pytest.fixture(scope="module")
def original_training(tmp_path_factory):
    root = tmp_path_factory.mktemp("feasibility-original")
    source, _ = source_fixture(root)
    out = root / "training"
    original.train_shared_exercise_residual_policy(source, out, generations=2, population=8, root_assistance=True)
    manifest, files = _verified_inputs(out, deadline=time.monotonic() + 15)
    return out, manifest, files


@pytest.fixture(scope="module")
def refinement(tmp_path_factory, original_training):
    source, _, _ = original_training
    out = tmp_path_factory.mktemp("feasibility-refinement") / "run"
    report = module.train_shared_exercise_feasibility_policy(source, out)
    manifest, files = _verified_inputs(out, deadline=time.monotonic() + 15)
    return out, report, manifest, files


def modified_trials(original_training, *, ratio=2., reward=.7, completed=True):
    _, _, files = original_training
    trials = deepcopy(_json(files["training.json"])["history"][0]["candidates"][0]["trials"])
    for trial in trials:
        m = trial["metrics"]
        for key in module.BODY_GATES:
            m[key] = 0.
        for row in m["equipment"]:
            for key in module.EQUIPMENT_GATES:
                row[key] = 0.
        m["equipment"][0]["speed_model_units_s"] = GATES["equipment_speed_model_units_s"] * ratio
        m["mean_reward"] = reward
        m["completed_interval"] = completed
        m["success"] = completed and ratio <= 1
        trial["status"] = "completed" if completed else "failed"
    return trials


def test_continuous_rank_precedes_reward_and_does_not_rename_score(original_training):
    lower = module.optimization_rank(modified_trials(original_training, ratio=2., reward=.5))
    higher = module.optimization_rank(modified_trials(original_training, ratio=10., reward=.9))
    feasible = module.optimization_rank(modified_trials(original_training, ratio=1., reward=.1))
    incomplete = module.optimization_rank(modified_trials(original_training, ratio=0., reward=1., completed=False))
    assert module._rank_key(feasible) < module._rank_key(lower) < module._rank_key(higher) < module._rank_key(incomplete)
    assert lower["max_normalized_excess"] == pytest.approx(1) and higher["max_normalized_excess"] == pytest.approx(9)
    assert lower["reward_score"] == .5 and incomplete["complete"] is False
    assert incomplete["max_normalized_excess"] is None


def test_all_equipment_and_body_gates_both_starts_participate(original_training):
    trials = modified_trials(original_training, ratio=.5)
    trials[1]["metrics"]["peak_root_rmse_model_units"] = 4 * GATES["peak_root_rmse_model_units"]
    trials[1]["metrics"]["success"] = False
    rank = module.optimization_rank(trials)
    assert rank["max_normalized_excess"] == 3 and rank["violated_terms"] == 1
    assert rank["gate_terms"] == 2 * (5 + 4 * len(trials[0]["metrics"]["equipment"]))
    assert rank["mean_normalized_excess"] == pytest.approx(3 / rank["gate_terms"])


def test_contact_lower_bound_coverage_and_gap_are_not_ignored(original_training):
    trials = modified_trials(original_training, ratio=.5)
    for trial in trials:
        trial["metrics"]["success"] = False
        trial["metrics"]["contact_requirements"] = [{"required_duration_s": 1., "observed_duration_s": .5,
            "qualifying_fraction": GATES["contact_duration_fraction"] / 2,
            "max_gap_s": 3 * GATES["max_contact_gap_s"]}]
    rank = module.optimization_rank(trials)
    assert rank["max_normalized_excess"] == pytest.approx(2) and rank["violated_terms"] == 6


def test_gate_tolerance_and_nonfinite_values_are_explicit(original_training):
    trials = modified_trials(original_training, ratio=1.)
    trials[0]["metrics"]["equipment"][0]["speed_model_units_s"] += 5e-13
    assert module.optimization_rank(trials)["feasible"]
    trials[0]["metrics"]["equipment"][0]["speed_model_units_s"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        module.optimization_rank(trials)
    with pytest.raises(ValueError, match="No complete"):
        module._selected([{"optimization_rank": module.optimization_rank(modified_trials(original_training, completed=False))}])


def test_original_import_is_pure_and_retains_exact_full_archive(original_training, monkeypatch):
    _, manifest, files = original_training
    monkeypatch.setattr(mujoco, "mj_step", lambda *args: pytest.fail("Import cannot run physics"))
    monkeypatch.setattr(original, "validate_shared_exercise_residual_policy", lambda *args: pytest.fail("Old policy must not be admitted"))
    result = module._verify_original(manifest, files, time.monotonic() + 10)
    proof = result[-1]
    assert proof["original_training_manifest_sha256"] == module._sha(manifest)
    assert proof["imported_episodes"] == 16 and proof["imported_control_steps"] == 48
    assert proof["new_physics_steps_during_import"] == 0 and proof["evaluation_used"] is False
    assert set(proof["actor_ast_sha256"]) == {"_trial", "_action", "phase_basis", "_coefficients"}
    assert "shared_exercise_policies.py" not in proof["unchanged_training_execution_runtime_sha256"]
    assert "not executed or admitted" in proof["dispatcher_scope"]


def test_changed_dispatcher_does_not_admit_old_policy_or_change_direct_training_actor(original_training, monkeypatch):
    _, manifest, files = original_training
    old_digest = module.digest
    monkeypatch.setattr(module, "digest", lambda path: "f" * 64 if Path(path).name == "shared_exercise_policies.py" else old_digest(path))
    proof = module._verify_original(manifest, files, time.monotonic() + 10)[-1]
    assert proof["current_dispatcher_sha256"] == "f" * 64
    assert proof["original_dispatcher_sha256"] != proof["current_dispatcher_sha256"]
    assert proof["original_full_history_verified"]


def resealed(files, status="trained"):
    return module._bytes({"schema_version": 1, "status": status,
        "files": {name: module._sha(raw) for name, raw in files.items()}})


@pytest.mark.parametrize("kind", ["rng", "gate", "actor", "reset", "source", "reward", "labels", "policy", "evaluation"])
def test_unproven_original_trials_never_enter_cache(original_training, kind):
    _, _, original_files = original_training
    files = dict(original_files)
    if kind == "actor":
        name = "implementation/shared_exercise_residual_learning.py"
        files[name] += b"\n# changed original actor closure\n"
    elif kind == "evaluation":
        files["evaluation.json"] = b'{}'
    elif kind in {"policy", "source"}:
        data = _json(files["policy.json"])
        data["algorithm" if kind == "policy" else "scene_sha256"] = "wrong"
        files["policy.json"] = module._bytes(data)
    else:
        data = _json(files["training.json"])
        row = data["history"][0]["candidates"][1]
        if kind == "rng":
            row["residual_coefficients"][0][0] += .001
        elif kind == "gate":
            row["trials"][0]["metrics"]["gates_sha256"] = "0" * 64
        elif kind == "reset":
            row["trials"][0]["seed"] += 1
        elif kind == "labels":
            data["selected_coast_baseline"] = True
        else:
            row["score"] += .1
        files["training.json"] = module._bytes(data)
    with pytest.raises(ValueError):
        module._verify_original(resealed(files), files, time.monotonic() + 10)


def test_refinement_only_spends_six_new_candidates_and_retains_old_score(refinement, original_training):
    out, report, manifest, files = refinement
    source, old_manifest, old_files = original_training
    assert report["status"] == "trained" and report["evaluated_episodes"] == 12
    assert report["control_steps"] == 36 and report["imported_control_steps"] == 48 and report["cache_hits"] == 2
    assert files["imported-training/manifest.json"] == old_manifest
    for name, raw in old_files.items():
        assert files["imported-training/" + name] == raw
        assert (source / name).read_bytes() == raw
    settings, policy = _json(files["settings.json"]), _json(files["policy.json"])
    assert settings["maximum_control_steps"] == 36 and settings["maximum_new_episodes"] == 12
    assert policy["profile"] == "shared-exercise-policy-v3" and policy["algorithm"] != original.ALGORITHM
    assert policy["actor_algorithm"] == module.ACTOR_ALGORITHM
    assert policy["training_score"] == policy["optimization_rank"]["reward_score"]
    for generation in report["history"]:
        for row in generation["candidates"]:
            assert row["score"] == module._score(row["trials"]) and row["loss"] == -row["score"]
            assert row["optimization_rank"] == module.optimization_rank(row["trials"])
    assert verify_artifacts(out)["manifest_sha256"] == report["manifest_sha256"]
    assert sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) <= module.MAX_OUTPUT_BYTES


def test_current_v3_admission_and_original_baseline_proof_are_distinct(refinement):
    _, _, manifest, files = refinement
    scene, task = _json(files["scene.json"]), _json(files["exercise-task.json"])
    policy, _ = _validate_policy_package(manifest, files, scene, task)
    baseline = module.verify_imported_policy_baseline(files, scene, task)
    assert policy["profile"] == module.POLICY_PROFILE and baseline["policy"]["profile"] == original.POLICY_PROFILE
    assert baseline["actor_equivalence"]["old_policy_admitted"] is False
    assert baseline["policy_sha256"] == module._sha(files[baseline["source_path"]])
    assert baseline["import_proof_sha256"] == policy["import_proof_sha256"]
    assert baseline["training_manifest_sha256"] == policy["source_training_manifest_sha256"]


def test_v3_actor_is_exact_old_action_family_and_detached(refinement, monkeypatch):
    _, _, _, files = refinement
    scene, task, policy = [_json(files[name]) for name in ("scene.json", "exercise-task.json", "policy.json")]
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=17)
        action = module.prepare_shared_exercise_feasibility_policy(policy, env)
        coefficients = original._coefficients(policy["residual_coefficients"])
        policy["residual_coefficients"][0][0] = .25
        monkeypatch.setattr(module, "_runtime_hashes", lambda: pytest.fail("Do not hash runtime per action"))
        for _ in range(3):
            current = action()
            assert current.tobytes() == original._action(coefficients, env).tobytes()
            assert current[19:].tobytes() == env.authored_action()[19:].tobytes()
            assert np.all(np.abs(current) <= 1)
            env.step(current)
            assert not any(env.snapshot()["engine_state"]["normalized_action"][22:])
    finally:
        env.close()


@pytest.mark.parametrize("kind", ["actor", "trainer", "objective", "bound", "rank", "runtime", "import_hash"])
def test_v3_contract_rejects_changed_execution_or_optimization_identity(refinement, kind):
    _, _, _, files = refinement
    scene, task, policy = [_json(files[name]) for name in ("scene.json", "exercise-task.json", "policy.json")]
    if kind == "rank":
        policy["optimization_rank"]["complete"] = False
    elif kind == "runtime":
        policy["runtime_sha256"].pop("shared_exercise_feasibility_evaluation.py")
    else:
        policy[{"actor": "actor_algorithm", "trainer": "algorithm", "objective": "selection_objective",
                "bound": "residual_bound", "import_hash": "import_proof_sha256"}[kind]] = "wrong"
    with pytest.raises(ValueError):
        module.validate_shared_exercise_feasibility_policy(policy, scene, task)


@pytest.mark.parametrize("kind", ["rank", "policy_import", "proof", "objective", "history", "cache", "count"])
def test_generic_admission_cannot_trust_resealed_false_training_claims(refinement, kind):
    _, _, _, old_files = refinement
    files = dict(old_files)
    scene, task = _json(files["scene.json"]), _json(files["exercise-task.json"])
    if kind == "proof":
        proof = _json(files["import-proof.json"])
        proof["imported_control_steps"] += 1
        files["import-proof.json"] = module._bytes(proof)
    elif kind == "policy_import":
        policy = _json(files["policy.json"])
        policy["source_training_manifest_sha256"] = "0" * 64
        files["policy.json"] = module._bytes(policy)
    elif kind == "objective":
        settings = _json(files["settings.json"])
        settings["selection_objective"] = "reward only"
        files["settings.json"] = module._bytes(settings)
    else:
        training = _json(files["training.json"])
        if kind == "rank":
            training["history"][1]["candidates"][2]["optimization_rank"]["mean_normalized_excess"] += .1
        elif kind == "history":
            training["history"][0]["next_mean"][0][0] += .01
        elif kind == "cache":
            training["history"][1]["candidates"][0]["cached"] = False
        else:
            training["control_steps"] += 1
        files["training.json"] = module._bytes(training)
    with pytest.raises(ValueError):
        module.verify_imported_policy_baseline(files, scene, task)


def test_evaluation_files_are_rejected_before_any_payload_read(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "manifest.json", {"schema_version": 1, "status": "trained", "files": {"evaluation.json": "a" * 64}})
    monkeypatch.setattr(module, "_verified_inputs", lambda *args, **kwargs: pytest.fail("Evaluation payload must not be read"))
    with pytest.raises(ValueError, match="cannot read evaluation"):
        module.train_shared_exercise_feasibility_policy(source, tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_interrupted_refinement_seals_partial_evidence_without_rerun(tmp_path, original_training, monkeypatch):
    source, _, _ = original_training
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise TimeoutError("Bounded fixture interruption")

    monkeypatch.setattr(original, "_trial", fail)
    out = tmp_path / "interrupted"
    with pytest.raises(TimeoutError):
        module.train_shared_exercise_feasibility_policy(source, out)
    assert len(calls) == 1
    training = _json((out / "training.json").read_bytes())
    assert training["status"] == "failed" and training["evaluated_episodes"] == 1 and training["control_steps"] == 0
    assert not (out / "policy.json").exists()
    assert _json((out / "manifest.json").read_bytes())["status"] == "failed"
    verify_artifacts(out)


def test_output_budget_is_reserved_before_metadata_write(tmp_path, monkeypatch):
    out = tmp_path / "output"
    out.mkdir()
    writer = module._Writer(out)
    writer.sizes["archive.bin"] = module.MAX_OUTPUT_BYTES - 65536
    monkeypatch.setattr(module, "_write", lambda *args: pytest.fail("Excessive bytes must not be written"))
    with pytest.raises(ValueError, match="before write"):
        writer.write("training.json", {"new": True})


@pytest.mark.parametrize("seconds", [0, True, float("nan"), 61])
def test_invalid_work_bounds_have_no_output(tmp_path, seconds):
    with pytest.raises(ValueError):
        module.train_shared_exercise_feasibility_policy(tmp_path / "missing", tmp_path / "unused", max_seconds=seconds)
    assert not (tmp_path / "unused").exists()
