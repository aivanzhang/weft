"""Real bounded orchestration and scoring with analytical dynamics, never native training."""

from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import pytest

from test_camera_shared_task import analytical_only, _deny
from test_camera_shared_recording import RecordingEngine
from test_camera_shared_policy import environment
from weft import camera_shared_learning as learner
from weft import camera_shared_policy as actor
from weft import camera_shared_task as task_api
from weft.io import canonical_hash, digest
from weft.shared_world_packaging import PinnedSourceFiles


class LearningEngine(RecordingEngine):
    reset_failure_scale = None
    repeated_state = False
    prefer_anchor = False
    reset_observer = None

    def reset(self, *, seed, options):
        if self.reset_observer is not None:
            self.reset_observer(seed, options)
        if self.reset_failure_scale is not None and options["jitter_scale"] == self.reset_failure_scale:
            raise ValueError("authored analytical reset failure")
        self.mode = None
        self.offset = 0. if self.repeated_state else options["jitter_scale"] * ((seed % 997) + 1) / 1000
        self.angular_error = self.root_error = self.penetration = 0.
        return super().reset(seed=seed, options=options)

    def feature_matrix(self):
        return np.tile([.2 + self.offset, .05, .02], (22, 1))

    def authored_human_action(self):
        return np.clip(self.feature_matrix().sum(axis=1), -1., 1.)

    def snapshot(self):
        result = super().snapshot()
        result["human"]["segments"][0]["a"][0] = float(self.offset)
        result["human"]["segments"][0]["b"][0] = float(self.offset)
        return result

    def step(self, action):
        delta = float(np.mean(np.asarray(action)[:19] - self.authored_human_action()[:19]))
        self.angular_error = .25 + abs(delta) if self.prefer_anchor else .25 - .8 * delta
        if np.max(abs(action)) > .7:
            self.mode = "finite_failure"
        return super().step(action)


@pytest.fixture(autouse=True)
def boundary(analytical_only, monkeypatch):
    monkeypatch.setattr(task_api, "SharedWorldEnv", LearningEngine)
    monkeypatch.setattr(LearningEngine, "reset_failure_scale", None)
    monkeypatch.setattr(LearningEngine, "repeated_state", False)
    monkeypatch.setattr(LearningEngine, "prefer_anchor", False)
    monkeypatch.setattr(LearningEngine, "reset_observer", None)
    # These authored fixtures contain no media or observed-world source package.
    monkeypatch.setattr(learner, "collect_shared_source_files", lambda scene, source, **kw:
        PinnedSourceFiles({"scope": "analytical fixture; source semantics substituted"},
                         {name: Path(source, name).read_bytes() for name in scene["source_binding"]["evidence_documents"].values()},
                         Path(source, "manifest.json").read_bytes(), {}, kw.get("expected_manifest_sha256")))


def write_source(root):
    env = environment(True)
    evidence = learner._bytes({"kind": "authored_fixture_descriptor_not_observed_media"})
    evidence_sha = learner._sha(evidence)
    evidence_name = "evidence/" + evidence_sha + ".json"
    env.scene["source_binding"]["evidence_documents"] = {evidence_sha: evidence_name}
    env.task = task_api.build_camera_shared_task(env.scene, root_assistance=True)
    root.mkdir()
    files = {"scene.json": learner._bytes(env.scene), "exercise-task.json": learner._bytes(env.task),
             "camera-coupling.json": learner._bytes(env.scene["camera_coupling"]),
             "source.json": learner._bytes({"kind": "authored_analytical_descriptor_not_video", "sha256": env.task["source_sha256"]}),
             evidence_name: evidence}
    for name, raw in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(raw)
    (root / "manifest.json").write_bytes(learner._bytes({"schema_version": 1, "status": "ready",
        "files": {name: learner._sha(raw) for name, raw in files.items()}}))
    return env.scene, env.task


def package(path):
    manifest = Path(path, "manifest.json").read_bytes()
    return manifest, {name: Path(path, name).read_bytes() for name in json.loads(manifest)["files"]}


def reseal(manifest, files):
    document = json.loads(manifest)
    document["files"] = {name: learner._sha(raw) for name, raw in files.items()}
    return learner._bytes(document)


_COMPLETED = None


@pytest.fixture
def completed(tmp_path_factory):
    global _COMPLETED
    if _COMPLETED is None:
        root = tmp_path_factory.mktemp("camera-learning-analytical")
        scene, task = write_source(root / "source")
        trained = learner.train_camera_shared_policy(root / "source", root / "training", generations=2, population=4)
        evaluated = learner.evaluate_camera_shared_policy(root / "training", root / "evaluation")
        _COMPLETED = root, scene, task, trained, evaluated
    return _COMPLETED


def test_actual_cem_changes_weights_and_actions_without_changing_task_or_source(completed):
    root, scene, task, trained, evaluated = completed
    assert trained["status"] == "trained" and trained["nonzero_coefficients"] > 0
    assert trained["selected_feedback_kind"] == "searched_phase_feedback_residuals"
    assert trained["selected_applied_residual"]["changed_actions"] > 0
    assert trained["training_score_improved_over_authored"] is True and trained["cache_hits"] == 2
    assert trained["evaluated_episodes"] == 12 and trained["control_steps"] == 36
    for name in ("scene.json", "exercise-task.json", "camera-coupling.json", "source.json"):
        assert (root / "source" / name).read_bytes() == (root / "training" / name).read_bytes() == (root / "evaluation" / name).read_bytes()
    assert scene["learned_policy_available"] is False and task["scope"]["learning"] == "unqualified"
    assert evaluated["selected_feedback_kind"] == trained["selected_feedback_kind"]
    for output, verify in (("training", learner.verify_camera_shared_training), ("evaluation", learner.verify_camera_shared_evaluation)):
        raw, files = package(root / output)
        result = verify(raw, files, scene, task, deadline=time.monotonic() + 10)
        assert result["verified"] and result["native_model_readmission"] is False


def test_frozen_matched_evaluation_retains_random_terminal_failures_and_new_states(completed):
    root, _, _, trained, evaluated = completed
    settings = json.loads((root / "training/settings.json").read_bytes())
    assert evaluated["policy_unchanged"] and evaluated["evaluation_feedback_to_optimizer"] is False
    assert evaluated["independent_changed_states"] == 3 and evaluated["comparison_complete"]
    assert not set(evaluated["evaluation_seeds"]) & set(settings["training_seeds"])
    initial = {trial["initial_state_sha256"] for entry in trained["history"] for row in entry["candidates"] for trial in row["trials"]}
    assert not initial & set(evaluated["initial_states"].values())
    assert len(evaluated["controllers"]["random"]["trials"]) == 3
    assert all(row["status"] == "failed" and row["metrics"]["failure"] for row in evaluated["controllers"]["random"]["trials"])
    assert evaluated["control_steps"] == 30
    assert all(pair["mean_reward_delta"] > 0 for pair in evaluated["paired_vs_authored"])


@pytest.mark.parametrize("mutation", ["sample", "label", "action", "policy", "runtime", "reward"])
def test_saved_training_rejects_resealed_search_and_claim_tampering_without_native(completed, monkeypatch, mutation):
    root, scene, task, _, _ = completed
    raw, files = package(root / "training")
    monkeypatch.setattr(task_api, "CameraSharedTaskEnv", _deny)
    monkeypatch.setattr(task_api, "prepare_camera_shared_task", _deny)
    training = json.loads(files["training.json"])
    if mutation == "sample":
        training["history"][0]["candidates"][1]["feedback_coefficients"][0][0][0] += .01
    elif mutation == "label":
        training["selected_applied_residual"]["changed_actions"] = 0
    elif mutation in {"action", "reward"}:
        row = training["history"][0]["candidates"][0]["trials"][0]
        name = row["diagnostics"]["artifact"]
        records = [json.loads(line) for line in files[name].splitlines()]
        if mutation == "action":
            records[0]["action"][0] += .01
        else:
            records[1]["reward"] += .01
        files[name] = b"".join((json.dumps(record) + "\n").encode() for record in records)
        row["diagnostics"]["sha256"] = learner._sha(files[name])
    elif mutation == "policy":
        policy = json.loads(files["policy.json"])
        policy["feedback_coefficients"][0][0][0] += .01
        files["policy.json"] = learner._bytes(policy)
        training["policy_sha256"] = learner._sha(files["policy.json"])
    else:
        files["implementation/camera_shared_policy.py"] += b"\n"
    files["training.json"] = learner._bytes(training)
    with pytest.raises((ValueError, KeyError)):
        learner.verify_camera_shared_training(reseal(raw, files), files, scene, task, deadline=time.monotonic() + 10)


def test_saved_view_cannot_be_reauthorized_by_resealed_settings(completed, monkeypatch):
    root, scene, task, _, _ = completed
    raw, files = package(root / "training")
    monkeypatch.setattr(task_api, "CameraSharedTaskEnv", _deny)
    monkeypatch.setattr(task_api, "prepare_camera_shared_task", _deny)
    files["source.json"] += b" "
    settings = json.loads(files["settings.json"])
    settings["source_views_sha256"]["source.json"] = learner._sha(files["source.json"])
    files["settings.json"] = learner._bytes(settings)
    with pytest.raises(ValueError, match="immutable parent custody"):
        learner.verify_camera_shared_training(reseal(raw, files), files, scene, task, deadline=time.monotonic() + 10)


@pytest.mark.parametrize("member", ["source.json", "camera-coupling.json", "bound_evidence"])
def test_saved_parent_member_cannot_be_deleted_with_resealed_settings(completed, monkeypatch, member):
    root, scene, task, _, _ = completed
    raw, files = package(root / "training")
    monkeypatch.setattr(task_api, "CameraSharedTaskEnv", _deny)
    monkeypatch.setattr(task_api, "prepare_camera_shared_task", _deny)
    if member == "bound_evidence":
        member = next(iter(scene["source_binding"]["evidence_documents"].values()))
    files.pop(member)
    settings = json.loads(files["settings.json"])
    settings["source_views_sha256"].pop(member, None)
    files["settings.json"] = learner._bytes(settings)
    with pytest.raises(ValueError, match="inventory|custody"):
        learner.verify_camera_shared_training(reseal(raw, files), files, scene, task, deadline=time.monotonic() + 10)


def test_original_task_alias_cannot_be_deleted_under_explicit_authority(tmp_path, monkeypatch):
    scene, task = write_source(tmp_path / "source")
    explicit = tmp_path / "explicit-task.json"
    explicit.write_bytes((tmp_path / "source/exercise-task.json").read_bytes() + b" ")
    learner.train_camera_shared_policy(tmp_path / "source", tmp_path / "training", task_path=explicit,
                                      generations=1, population=4)
    raw, files = package(tmp_path / "training")
    assert files["input-original-task.json"] != files["exercise-task.json"]
    monkeypatch.setattr(task_api, "CameraSharedTaskEnv", _deny)
    monkeypatch.setattr(task_api, "prepare_camera_shared_task", _deny)
    files.pop("input-original-task.json")
    settings = json.loads(files["settings.json"])
    settings["source_views_sha256"].pop("input-original-task.json")
    files["settings.json"] = learner._bytes(settings)
    with pytest.raises(ValueError, match="inventory|custody"):
        learner.verify_camera_shared_training(reseal(raw, files), files, scene, task, deadline=time.monotonic() + 10)


@pytest.mark.parametrize("missing", ["manifest.json", "source-descriptor.json"])
def test_source_equivalence_requires_its_complete_pinned_inventory(completed, missing):
    root, scene, _, _, _ = completed
    _, files = package(root / "training")
    scene = deepcopy(scene)
    parent = json.loads(files["input-manifest.json"])
    member = learner._bytes({"kind": "authored_mapping_descriptor_not_media"})
    manifest = learner._bytes({"schema": 1, "files": {"source-descriptor.json": learner._sha(member)}})
    pin = learner._sha(manifest)
    prefix = "source-mappings/" + pin
    scene["source_binding"]["source_equivalences"] = [{"original_source_sha256": scene["source_binding"]["source_sha256"],
        "derivative_source_sha256": learner._sha(b"authored derivative identity"),
        "bundle_manifest_sha256": pin, "bundle_path": prefix}]
    for name, raw in (("manifest.json", manifest), ("source-descriptor.json", member)):
        files[prefix + "/" + name] = raw
        parent["files"][prefix + "/" + name] = learner._sha(raw)
    raw_parent = learner._bytes(parent)
    learner._source_custody(scene, raw_parent, files, "sealed_source", learner._sha(files["input-task.json"]))
    files.pop(prefix + "/" + missing)
    with pytest.raises(ValueError, match="custody"):
        learner._source_custody(scene, raw_parent, files, "sealed_source", learner._sha(files["input-task.json"]))


@pytest.mark.parametrize("mutation", ["seed", "summary", "omit_failure", "freeze", "source"])
def test_saved_evaluation_rejects_resealed_partition_and_result_tampering(completed, mutation, monkeypatch):
    root, scene, task, _, _ = completed
    raw, files = package(root / "evaluation")
    monkeypatch.setattr(task_api, "CameraSharedTaskEnv", _deny)
    monkeypatch.setattr(task_api, "prepare_camera_shared_task", _deny)
    result = json.loads(files["evaluation.json"])
    if mutation == "seed":
        result["evaluation_seeds"][0] = result["training_seeds"][0]
    elif mutation == "summary":
        result["paired_vs_authored"][0]["mean_reward_delta"] += .1
    elif mutation == "omit_failure":
        result["controllers"]["random"]["trials"].pop()
    elif mutation == "freeze":
        files["pre-evaluation-freeze.json"] += b" "
    else:
        files["source.json"] += b" "
    files["evaluation.json"] = learner._bytes(result)
    with pytest.raises(ValueError):
        learner.verify_camera_shared_evaluation(reseal(raw, files), files, scene, task, deadline=time.monotonic() + 10)


def test_evaluation_freezes_before_reset_and_keeps_all_failed_starts(completed, tmp_path, monkeypatch):
    root, _, _, _, _ = completed
    out = tmp_path / "evaluation"
    calls = []

    def check(seed, options):
        frozen = json.loads((out / "pre-evaluation-freeze.json").read_bytes())
        assert seed in frozen["evaluation_seeds"] and options["jitter_scale"] == .1
        assert digest(out / "policy.json") == frozen["policy_sha256"]
        calls.append(seed)

    monkeypatch.setattr(LearningEngine, "reset_observer", staticmethod(check))
    monkeypatch.setattr(LearningEngine, "reset_failure_scale", .1)
    result = learner.evaluate_camera_shared_policy(root / "training", out)
    assert len(calls) == 12 and result["control_steps"] == 0 and not result["comparison_complete"]
    assert result["independent_changed_states"] == 0
    assert all(summary["reset_failures"] == 3 and len(summary["trials"]) == 3 for summary in result["controllers"].values())
    manifest, files = package(out)
    assert learner.verify_camera_shared_evaluation(manifest, files, completed[1], completed[2], deadline=time.monotonic() + 10)["verified"]


def test_unchanged_anchor_is_not_learning_and_failed_training_cannot_publish_policy(tmp_path, monkeypatch):
    source = tmp_path / "source"
    write_source(source)
    monkeypatch.setattr(LearningEngine, "prefer_anchor", True)
    result = learner.train_camera_shared_policy(source, tmp_path / "anchor", generations=1, population=4)
    assert result["selected_feedback_kind"] == "authored_unchanged" and result["nonzero_coefficients"] == 0
    assert result["selected_applied_residual"]["changed_actions"] == 0
    monkeypatch.setattr(LearningEngine, "reset_failure_scale", .05)
    with pytest.raises(ValueError, match="No candidate"):
        learner.train_camera_shared_policy(source, tmp_path / "failed", generations=1, population=4)
    manifest, files = package(tmp_path / "failed")
    assert json.loads(manifest)["status"] == "failed" and "policy.json" not in files
    saved = json.loads(files["training.json"])
    assert len(saved["active_candidates"]) == 4
    assert all(row["trials"][1]["status"] == "reset_failed" for row in saved["active_candidates"])


def test_action_budget_and_interruption_preserve_partial_trials_before_extra_step(tmp_path, monkeypatch):
    source = tmp_path / "source"
    write_source(source)
    original = learner._Budget

    def tiny(_maximum):
        return original(1)

    monkeypatch.setattr(learner, "_Budget", tiny)
    with pytest.raises(ValueError, match="budget exhausted"):
        learner.train_camera_shared_policy(source, tmp_path / "interrupted", generations=1, population=4)
    manifest, files = package(tmp_path / "interrupted")
    saved = json.loads(files["training.json"])
    assert json.loads(manifest)["status"] == "failed" and saved["control_steps"] == 1
    trial = saved["active_candidates"][0]["trials"][0]
    assert trial["status"] == "interrupted" and trial["control_steps"] == 1 and trial["diagnostics"]["attempts"] == 1
    assert LearningEngine.instances[-1].closed


def test_inputs_options_and_full_interval_budget_reject_before_engine_or_output(tmp_path, monkeypatch):
    source = tmp_path / "source"
    write_source(source)
    monkeypatch.setattr(task_api, "CameraSharedTaskEnv", _deny)
    for kwargs in ({"population": 3}, {"generations": 0}, {"max_seconds": 61}, {"seed": True}):
        with pytest.raises(ValueError):
            learner.train_camera_shared_policy(source, tmp_path / "rejected", **kwargs)
        assert not (tmp_path / "rejected").exists()
    monkeypatch.setattr(learner, "MAX_CONTROL_STEPS", 1)
    with pytest.raises(ValueError, match="8640"):
        learner.train_camera_shared_policy(source, tmp_path / "over_budget")
    assert not (tmp_path / "over_budget").exists()
    with pytest.raises(ValueError, match="outside"):
        learner.train_camera_shared_policy(source, source / "nested")
    manifest = json.loads((source / "manifest.json").read_bytes())
    manifest["files"].pop("exercise-task.json")
    (source / "manifest.json").write_bytes(learner._bytes(manifest))
    with pytest.raises(ValueError, match="exact sealed task"):
        learner.train_camera_shared_policy(source, tmp_path / "missing_task")
    assert not (tmp_path / "missing_task").exists()
