"""Saved-candidate selection validates custody and gates without observing new physics."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from weft.io import canonical_hash, digest, write_json
from weft.pipeline import verify_artifacts
import weft.shared_exercise_residual_learning as generator
import weft.shared_exercise_selection as module
import weft.shared_exercise_task as task_module
from weft.shared_exercise_reset import describe_reset_protocol
from weft.shared_world_observation import EquipmentState
from test_shared_exercise_run import source_fixture


class NoPhysicsEnv:
    """Expose policy metadata only; fixture creation cannot reset or step physics."""

    def __init__(self, scene, task):
        self.scene, self.task = scene, task
        self.scene_sha256, self.task_sha256 = canonical_hash(scene), canonical_hash(task)
        self.evaluator_sha256 = digest(task_module.__file__)
        self.horizon_steps = task["interval"]["horizon_steps"]
        self.action_count = 22 if task["control"]["root_assistance"] == "explicit_3_axis" else 19
        self.action_names = scene["action_contract"]["names"][:self.action_count]

    def close(self):
        pass


def learning_fixture(tmp_path, monkeypatch, *, feasible=(2,), generations=2, root_assistance=False,
                     contacts=False, tied=False, rotated=False):
    source, scene = source_fixture(tmp_path)
    if rotated:
        quaternion = np.asarray([.1, .2, .3, .4])
        scene["objects"][0]["quaternion"] = (quaternion / np.linalg.norm(quaternion)).tolist()
        write_json(source / "scene.json", scene)
    requirements = [{"object_id": "ball", "wrist": "left", "kind": "upward_support", "start_step": 0,
                     "end_step": 3}] if contacts else []
    task = task_module.build_shared_exercise_task(scene, root_assistance=root_assistance,
                                                 contact_requirements=requirements)
    write_json(source / "exercise-task.json", task)
    write_json(source / "source.json", {"url": "https://example.invalid/fixture", "scope": "synthetic metadata"})
    (source / "source.webm").write_bytes(b"Exact fixture bytes, not a reconstructed video")
    reseal(source, status="ready")
    candidates = {}

    def trial(env, controller, seed, reset, coefficients, deadline, budget, **kwargs):
        assert controller == "learned"
        key = canonical_hash(coefficients.tolist())
        index = candidates.setdefault(key, len(candidates))
        score = .7 if index == 0 else .9 if index == 1 else .8 if index == 2 or tied and index == 3 else .65
        equipment = tuple(EquipmentState(obj["id"], tuple(obj["position"]), tuple(obj["quaternion"]),
            tuple(obj["velocity"]), tuple(obj["angular_velocity"]), (0., 0., 0.)) for obj in scene["objects"])
        metrics = task_module._ExerciseMetrics(scene, task, equipment, 0., 1.)
        fixed = {"scene_sha256": env.scene_sha256, "task_sha256": env.task_sha256,
                 "evaluator_sha256": env.evaluator_sha256, "reset_protocol": describe_reset_protocol(reset)}
        initial = metrics.metrics() | fixed
        for _ in range(env.horizon_steps):
            budget.reserve()
        duration = env.horizon_steps * scene["dt"]
        metrics.samples = metrics.valid_samples = env.horizon_steps * scene["substeps"]
        metrics.elapsed = metrics.valid_duration = duration
        metrics.angular_squared, metrics.root_squared = .2**2 * duration, .03**2 * duration
        metrics.angular_peak, metrics.root_peak = .2, .03
        metrics.reward_integral = score * duration
        if index not in feasible:
            metrics.equipment[task["equipment_ids"][0]]["speed_model_units_s"] = .25
        for contact in metrics.contacts:
            contact.update(observed_duration_s=duration, qualifying_duration_s=duration,
                           active_now=True, qualifying_now=True, load_body_weight_fraction=.1)
        final = metrics.metrics() | fixed
        return {"seed": seed, "reset_options": deepcopy(reset), "initial_state_sha256": canonical_hash({"seed": seed}),
            "status": "completed", "control_steps": env.horizon_steps, "return": score * env.horizon_steps,
            "initial_metrics": initial, "metrics": final, "terminated": final["success"], "truncated": not final["success"],
            "final_snapshot_sha256": canonical_hash({"candidate": key, "seed": seed, "metrics": final})}

    # The frozen generator supplies exact CEM identities and packaging; all trial data are synthetic.
    with monkeypatch.context() as patch:
        patch.setattr(generator, "SharedExerciseTaskEnv", NoPhysicsEnv)
        patch.setattr(generator, "_trial", trial)
        generator.train_shared_exercise_residual_policy(source, tmp_path / "learning", generations=generations, population=4)
    return tmp_path / "learning", scene, task


def reseal(path, *, status="trained"):
    write_json(path / "manifest.json", {"schema_version": 1, "status": status,
        "files": {file.relative_to(path).as_posix(): digest(file) for file in sorted(path.rglob("*"))
                  if file.is_file() and file != path / "manifest.json"}})


def read(path):
    return json.loads(path.read_text())


def forbid_physics(*args, **kwargs):
    pytest.fail("No-retrain selection must not construct physics or invoke training/evaluation")


@pytest.mark.parametrize("root_assistance", [False, True])
def test_selects_highest_training_feasible_candidate_without_physics_or_evaluation(tmp_path, monkeypatch, root_assistance):
    learning, scene, task = learning_fixture(tmp_path, monkeypatch, root_assistance=root_assistance)
    before = {p.relative_to(learning).as_posix(): p.read_bytes() for p in learning.rglob("*") if p.is_file()}
    monkeypatch.setattr(generator, "SharedExerciseTaskEnv", forbid_physics)
    monkeypatch.setattr(task_module, "SharedExerciseTaskEnv", forbid_physics)
    monkeypatch.setattr(generator, "train_shared_exercise_residual_policy", forbid_physics)
    monkeypatch.setattr(generator, "evaluate_shared_exercise_residual_policy", forbid_physics)
    out = tmp_path / "selected"
    result = module.select_feasible_shared_exercise_policy(learning, out)
    selected, original = read(out / "policy.json"), read(learning / "policy.json")
    training, old = read(out / "training.json"), read(learning / "training.json")
    selection = read(out / "selection.json")
    candidate = old["history"][0]["candidates"][2]
    assert result["selection_status"] == "selected_feasible_candidate"
    assert result["selected_candidate"] == {"generation": 0, "candidate": 2,
                                           "coefficients_sha256": canonical_hash(candidate["residual_coefficients"])}
    assert selected["residual_coefficients"] == candidate["residual_coefficients"]
    assert selected["training_score"] == candidate["score"] < original["training_score"]
    assert selected == original | {"residual_coefficients": candidate["residual_coefficients"], "training_score": candidate["score"]}
    assert result["new_training_episodes"] == result["new_control_steps"] == 0
    assert selection["evaluation_used"] is False and not (out / "evaluation.json").exists()
    assert selection["independent_evaluation_status"] == "not_performed_for_this_selection"
    assert selection["score_change_from_original_selection"] < 0
    assert training["history"] == old["history"]
    for name in ("evaluated_episodes", "control_steps", "cache_hits", "seconds"):
        assert training[name] == old[name]
    assert training["selected_feedback_kind"] == "searched_joint_residuals"
    assert training["improvement_over_authored_training"] == selected["training_score"] - selection["authored_training_score"]
    assert set(selection["generator_runtime_sha256"]) == set(selected["runtime_sha256"])
    assert "shared_exercise_selection.py" not in selected["runtime_sha256"]
    assert digest(out / selection["selector_source_path"]) == selection["selector_source_sha256"] == digest(module.__file__)
    for name, raw in before.items():
        preserved = selection["original_manifest_path"] if name == "manifest.json" else selection["original_file_replacements"].get(name, name)
        assert (out / preserved).read_bytes() == raw
    assert (out / "settings.json").read_bytes() == before["settings.json"]
    assert {p.relative_to(learning).as_posix(): p.read_bytes() for p in learning.rglob("*") if p.is_file()} == before
    manifest, files = module._verified_inputs(out, deadline=module.time.monotonic() + 10)
    assert module._validate_policy_package(manifest, files, scene, task)[0] == selected
    assert verify_artifacts(out)["manifest_sha256"] == result["manifest_sha256"]
    assert digest(out / "selection.json") == result["selection_sha256"]


@pytest.mark.parametrize("feasible,status,index", [
    ((), "retained_original_no_feasible_candidate", 1),
    ((1, 2), "retained_original_feasible", 1),
    ((0,), "selected_feasible_candidate", 0),
])
def test_fallback_original_or_zero_anchor_remains_truthful(tmp_path, monkeypatch, feasible, status, index):
    learning, _, _ = learning_fixture(tmp_path, monkeypatch, feasible=feasible, generations=1)
    out = tmp_path / "selected"
    result = module.select_feasible_shared_exercise_policy(learning, out)
    assert result["selection_status"] == status and result["selected_candidate"]["candidate"] == index
    if index == 1:
        assert (out / "policy.json").read_bytes() == (learning / "policy.json").read_bytes()
    training = read(out / "training.json")
    assert training["selected_authored_baseline"] is (index == 0)
    assert training["selected_feedback_kind"] == ("authored_unchanged" if index == 0 else "searched_joint_residuals")
    assert training["selected_coast_baseline"] is False
    if index == 0:
        assert training["improvement_over_authored_training"] == 0
        assert training["training_score_improved_over_authored"] is False


def test_equal_feasible_reward_uses_first_original_candidate(tmp_path, monkeypatch):
    learning, _, _ = learning_fixture(tmp_path, monkeypatch, feasible=(2, 3), generations=1, tied=True)
    result = module.select_feasible_shared_exercise_policy(learning, tmp_path / "selected")
    assert result["feasible_candidates"] == 2 and result["selected_candidate"]["candidate"] == 2


def test_initial_equipment_accepts_same_quaternion_roundoff(tmp_path, monkeypatch):
    learning, _, _ = learning_fixture(tmp_path, monkeypatch, generations=1, rotated=True)
    initial = read(learning / "training.json")["history"][0]["candidates"][0]["trials"][0]["initial_metrics"]
    assert initial["equipment"][0]["orientation_rad"] > 0
    assert module.select_feasible_shared_exercise_policy(learning, tmp_path / "selected")["feasible_candidates"] == 1


@pytest.mark.parametrize("mutation", [
    "false_success", "gate_value", "missing_equipment", "substeps", "short_interval", "terminal", "return",
    "initial_equipment", "coefficient", "score", "cached_trial", "cache_count", "episode_count", "step_count",
    "seed", "reset", "boolean_reset_scale", "missing_trial", "missing_generation", "next_mean", "initial_state",
])
def test_resealed_malformed_history_cannot_select_a_candidate(tmp_path, monkeypatch, mutation):
    learning, _, _ = learning_fixture(tmp_path, monkeypatch)
    path = learning / "training.json"
    training = read(path)
    candidate = training["history"][0]["candidates"][2]
    trial, metrics = candidate["trials"][0], candidate["trials"][0]["metrics"]
    if mutation == "false_success":
        training["history"][0]["candidates"][1]["trials"][0]["metrics"]["success"] = True
    elif mutation == "gate_value":
        metrics["equipment"][0]["speed_model_units_s"] = .3
    elif mutation == "missing_equipment":
        metrics["equipment"].pop()
    elif mutation == "substeps":
        metrics["valid_substeps"] -= 1
    elif mutation == "short_interval":
        trial["control_steps"] = 2
    elif mutation == "terminal":
        trial["terminated"] = False
    elif mutation == "return":
        trial["return"] += .01
    elif mutation == "initial_equipment":
        trial["initial_metrics"]["equipment"][0]["speed_model_units_s"] = .01
    elif mutation == "coefficient":
        candidate["residual_coefficients"][0][0] += .001
    elif mutation == "score":
        candidate["score"] += .001
    elif mutation == "cached_trial":
        training["history"][1]["candidates"][0]["trials"][0]["return"] += .01
    elif mutation in ("cache_count", "episode_count", "step_count"):
        training[{"cache_count": "cache_hits", "episode_count": "evaluated_episodes", "step_count": "control_steps"}[mutation]] += 1
    elif mutation == "seed":
        trial["seed"] += 1
    elif mutation == "reset":
        trial["reset_options"]["jitter_scale"] = .2
    elif mutation == "boolean_reset_scale":
        trial["reset_options"]["jitter_scale"] = False
    elif mutation == "missing_trial":
        candidate["trials"].pop()
    elif mutation == "missing_generation":
        training["history"].pop()
    elif mutation == "next_mean":
        training["history"][0]["next_mean"][0][0] += .001
    elif mutation == "initial_state":
        trial["initial_state_sha256"] = "0" * 64
    write_json(path, training)
    reseal(learning)
    out = tmp_path / "selected"
    with pytest.raises(ValueError):
        module.select_feasible_shared_exercise_policy(learning, out)
    assert not out.exists()


def test_contact_gates_are_checked_even_when_success_flags_claim_pass(tmp_path, monkeypatch):
    learning, _, _ = learning_fixture(tmp_path, monkeypatch, contacts=True, generations=1)
    module.select_feasible_shared_exercise_policy(learning, tmp_path / "valid")
    path = learning / "training.json"
    training = read(path)
    contact = training["history"][0]["candidates"][2]["trials"][0]["metrics"]["contact_requirements"][0]
    contact.update(qualifying_duration_s=0., qualifying_fraction=0.)
    write_json(path, training)
    reseal(learning)
    with pytest.raises(ValueError, match="contact success"):
        module.select_feasible_shared_exercise_policy(learning, tmp_path / "invalid")


@pytest.mark.parametrize("tamper", ["stored_runtime", "current_runtime", "scene", "source_view", "settings"])
def test_resealed_identity_or_runtime_changes_are_rejected(tmp_path, monkeypatch, tamper):
    learning, _, _ = learning_fixture(tmp_path, monkeypatch, generations=1)
    if tamper == "stored_runtime":
        path = learning / "implementation/shared_exercise_task.py"
        path.write_bytes(path.read_bytes() + b"\n# changed snapshot\n")
    elif tamper == "current_runtime":
        hashes = generator._runtime_hashes() | {"shared_exercise_task.py": "0" * 64}
        monkeypatch.setattr(generator, "_runtime_hashes", lambda: hashes)
    elif tamper == "scene":
        scene = read(learning / "scene.json")
        scene["objects"][0]["velocity"][0] = .01
        write_json(learning / "scene.json", scene)
    elif tamper == "source_view":
        (learning / "source.webm").write_bytes(b"changed original source")
    elif tamper == "settings":
        settings = read(learning / "settings.json")
        settings["maximum_control_steps"] += 1
        write_json(learning / "settings.json", settings)
    reseal(learning)
    with pytest.raises(ValueError):
        module.select_feasible_shared_exercise_policy(learning, tmp_path / "selected")
    assert not (tmp_path / "selected").exists()


@pytest.mark.parametrize("reserved", ["evaluation.json", "nested/evaluation.json", "pre-evaluation-freeze.json",
                                     "policy-evaluation-run/result.json", "selection.json", "selection-source/policy.json"])
def test_reserved_inputs_reject_before_reading_evaluation_or_prior_selection(tmp_path, monkeypatch, reserved):
    learning, _, _ = learning_fixture(tmp_path, monkeypatch, generations=1)
    path = learning / reserved
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"Must never be read")
    reseal(learning)
    monkeypatch.setattr(module, "_verified_inputs", lambda *a, **k: pytest.fail("Read rejected package contents"))
    with pytest.raises(ValueError, match="never evaluation"):
        module.select_feasible_shared_exercise_policy(learning, tmp_path / "selected")


@pytest.mark.parametrize("bound", [0, 16, True, float("nan")])
def test_invalid_time_bound_rejects_before_input_reads(tmp_path, monkeypatch, bound):
    monkeypatch.setattr(module, "read_package_manifest", lambda *a, **k: pytest.fail("Read before bound check"))
    with pytest.raises(ValueError):
        module.select_feasible_shared_exercise_policy(tmp_path / "missing", tmp_path / "out", max_seconds=bound)


def test_output_overlap_and_custody_bound_reject_before_writes(tmp_path, monkeypatch):
    learning, _, _ = learning_fixture(tmp_path, monkeypatch, generations=1)
    with pytest.raises(ValueError, match="immutable input"):
        module.select_feasible_shared_exercise_policy(learning, learning / "selected")
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 1)
    with pytest.raises(ValueError, match="custody exceeds"):
        module.select_feasible_shared_exercise_policy(learning, tmp_path / "selected")
    assert not (tmp_path / "selected").exists()


@pytest.mark.parametrize("fault", ["changed_input", "changed_runtime", "after_terminal_manifest"])
def test_publication_fault_keeps_exact_custody_and_failed_manifest(tmp_path, monkeypatch, fault):
    learning, _, _ = learning_fixture(tmp_path, monkeypatch, generations=1)
    out = tmp_path / "selected"
    if fault == "changed_input":
        monkeypatch.setattr(module, "_input_unchanged", lambda *a, **k: (_ for _ in ()).throw(ValueError("Changed input")))
    elif fault == "changed_runtime":
        monkeypatch.setattr(module, "_runtime_hashes", lambda: {})
    else:
        original = module._seal

        def after_manifest(*args):
            original(*args)
            raise TimeoutError("Injected post-manifest deadline")

        monkeypatch.setattr(module, "_seal", after_manifest)
    with pytest.raises((ValueError, TimeoutError)):
        module.select_feasible_shared_exercise_policy(learning, out)
    assert read(out / "manifest.json")["status"] == "failed"
    assert read(out / "selection.json")["status"] == "failed"
    assert (out / module.ARCHIVE / "manifest.json").read_bytes() == (learning / "manifest.json").read_bytes()
    assert (out / module.ARCHIVE / "policy.json").read_bytes() == (learning / "policy.json").read_bytes()
    verify_artifacts(out)
    with pytest.raises(ValueError):
        module._verified_inputs(out, deadline=module.time.monotonic() + 10)
