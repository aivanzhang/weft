"""Bounded analytical custody and unchanged-actor compatibility; no native or network work."""

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pytest

from test_camera_shared_task import analytical_only, _deny
from test_camera_shared_policy import environment
from test_camera_shared_learning import boundary, package, write_source
from weft import camera_shared_learning as learner
from weft import camera_shared_policy as actor
from weft import camera_shared_policy_admission as admission
from weft.camera_shared_policy_receipt import (
    ADMISSION_PROFILE, ALLOWED_CHANGED_FILES, COMPATIBILITY_PROFILE, ORIGINAL_RUNTIME_REQUIRED, SCOPE,
)
from weft.camera_shared_receipt import IDENTITIES
from weft.io import canonical_hash


def raw(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def fixture_admission(env, *, zero=False):
    current = admission.execution_runtime_hashes(env.scene)
    original = {name: current[name] for name in ORIGINAL_RUNTIME_REQUIRED}
    weights = np.zeros(actor.SHAPE) if zero else np.full(actor.SHAPE, .1)
    policy = actor.make_policy(env, weights, 1729, .5, original)
    document = raw(policy)
    compatibility = {"profile": COMPATIBILITY_PROFILE, "allowed_changed_files": list(ALLOWED_CHANGED_FILES),
                     "changed_files": {}, "verifier_delta": None}
    result = {"schema_version": 1, "profile": ADMISSION_PROFILE, "status": "admitted",
        "scene_sha256": env.task["scene_sha256"], "task_sha256": canonical_hash(env.task),
        **{key: env.task[key] for key in IDENTITIES}, "policy_sha256": sha256(document).hexdigest(),
        "training_manifest_sha256": "1" * 64, "evaluation_manifest_sha256": None,
        "original_runtime_sha256": original, "execution_runtime_sha256": current,
        "runtime_versions": policy["runtime_versions"], "compatibility": compatibility,
        "compatibility_sha256": canonical_hash(compatibility),
        "summary": {"selected_feedback_kind": actor.selection_kind(policy), "nonzero_coefficients": 0 if zero else 12,
                    "root_assistance": env.task["control"]["root_assistance"], "learning_qualification": "unqualified_research"},
        "privileged_simulator_features": True, "scope": SCOPE}
    return document, result


@pytest.fixture(scope="module")
def historical_source():
    current = Path(learner.__file__).read_bytes()
    # An authored predecessor fixture contains only the old verifier signatures and dispatch.
    previous = current.replace(b", historical_runtime_sha256=None):", b"):")
    previous = previous.replace(b"actor.runtime_hashes(scene) if historical_runtime_sha256 is None else historical_runtime_sha256",
                                b"actor.runtime_hashes(scene)")
    previous = previous.replace(b",\n                                            historical_runtime_sha256=historical_runtime_sha256", b"")
    assert b"historical_runtime_sha256" not in previous
    return previous


def test_saved_verifier_delta_is_exact_and_excludes_optimizer_changes(historical_source):
    current = Path(learner.__file__).read_bytes()
    delta = admission._verifier_delta(historical_source, current)
    assert delta["original_sha256"] == sha256(historical_source).hexdigest()
    assert delta["current_sha256"] == sha256(current).hexdigest()
    for changed in (current.replace(b"maximum = generations * population * 2", b"maximum = generations * population * 3"),
                    current.replace(b"historical_runtime_sha256 is None", b"historical_runtime_sha256 == None")):
        assert changed != current
        with pytest.raises(ValueError):
            admission._verifier_delta(historical_source, changed)


@pytest.fixture
def saved_evidence(tmp_path, monkeypatch, historical_source):
    scene, task = write_source(tmp_path / "source")
    sources = admission._runtime_sources(scene)
    historical = {name: sources[name] for name in ORIGINAL_RUNTIME_REQUIRED}
    historical["camera_shared_learning.py"] = historical_source
    historical["recording.py"] = b'"""Analytical original integration wrapper; never executed."""\n'
    pins = {name: sha256(value).hexdigest() for name, value in historical.items()}
    monkeypatch.setattr(actor, "runtime_hashes", lambda _: pins)
    original_read = learner.read_package_file

    def implementation(root, name, **kwargs):
        if Path(root).resolve() == Path(learner.__file__).parent.resolve() and name in historical:
            assert kwargs.get("expected_sha256") == pins[name]
            return historical[name]
        return original_read(root, name, **kwargs)

    monkeypatch.setattr(learner, "read_package_file", implementation)
    # Original three-step starts exercise real saved verification with analytical dynamics only.
    learner.train_camera_shared_policy(tmp_path / "source", tmp_path / "trained", generations=1, population=4)
    learner.evaluate_camera_shared_policy(tmp_path / "trained", tmp_path / "evaluated")
    training, train_files = package(tmp_path / "trained")
    evaluation, eval_files = package(tmp_path / "evaluated")
    monkeypatch.setattr(actor, "runtime_hashes", _deny)
    return scene, task, training, train_files, evaluation, eval_files


def test_saved_evaluation_is_checked_once_with_original_runtime_and_exact_embedded_training(saved_evidence, monkeypatch):
    scene, task, training, train_files, evaluation, eval_files = saved_evidence
    verify = learner.verify_camera_shared_training
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs["historical_runtime_sha256"])
        return verify(*args, **kwargs)

    monkeypatch.setattr(learner, "verify_camera_shared_training", observed)
    result = admission.prepare_camera_policy_admission(scene, task, training_manifest_raw=training,
        training_files=train_files, evaluation_manifest_raw=evaluation, evaluation_files=eval_files,
        deadline=time.monotonic() + 10)
    assert len(calls) == 1 and calls[0] == json.loads(train_files["policy.json"])["runtime_sha256"]
    assert result["training_manifest_sha256"] == sha256(training).hexdigest()
    assert result["evaluation_manifest_sha256"] == sha256(evaluation).hexdigest()
    assert set(result["compatibility"]["changed_files"]) == {"camera_shared_learning.py", "recording.py"}
    assert admission.validate_camera_policy_admission(result, scene, task, train_files["policy.json"]) == result
    assert scene["learned_policy_available"] is False and task["scope"]["learning"] == "unqualified"
    training_only = admission.prepare_camera_policy_admission(scene, task, training_manifest_raw=training,
        training_files=train_files, deadline=time.monotonic() + 10)
    assert len(calls) == 2 and training_only["evaluation_manifest_sha256"] is None
    assert {key: value for key, value in result.items() if key != "evaluation_manifest_sha256"} == {
        key: value for key, value in training_only.items() if key != "evaluation_manifest_sha256"}
    with pytest.raises(AssertionError, match="forbidden"):
        verify(training, train_files, scene, task, deadline=time.monotonic() + 10)
    altered = dict(eval_files)
    altered["policy.json"] += b" "
    with pytest.raises(ValueError, match="training evidence"):
        admission.prepare_camera_policy_admission(scene, task, training_manifest_raw=training,
            training_files=train_files, evaluation_manifest_raw=evaluation, evaluation_files=altered,
            deadline=time.monotonic() + 10)


def test_no_unapproved_runtime_delta_or_missing_original_can_reach_saved_verifiers(monkeypatch):
    env = environment()
    document, _ = fixture_admission(env)
    policy = json.loads(document)
    sources = admission._runtime_sources(env.scene)
    files = {"policy.json": document, **{"implementation/" + name: sources[name] for name in policy["runtime_sha256"]}}
    monkeypatch.setattr(learner, "verify_camera_shared_training", _deny)
    for name in ("camera_shared_policy.py", "shared_world.py", "shared_exercise_task.py"):
        changed = dict(sources)
        changed[name] += b"\n# changed\n"
        monkeypatch.setattr(admission, "_runtime_sources", lambda _: changed)
        with pytest.raises(ValueError, match="runtime changed"):
            admission.prepare_camera_policy_admission(env.scene, env.task, training_manifest_raw=b"{}",
                training_files=files, deadline=time.monotonic() + 10)
    monkeypatch.setattr(admission, "_runtime_sources", lambda _: sources)
    del files["implementation/camera_shared_policy.py"]
    with pytest.raises(ValueError, match="every exact original"):
        admission.prepare_camera_policy_admission(env.scene, env.task, training_manifest_raw=b"{}",
            training_files=files, deadline=time.monotonic() + 10)


@pytest.mark.parametrize("roots,zero", [(False, False), (True, False), (True, True)])
def test_sampler_is_detached_zero_io_and_preserves_authored_roots_and_realized_delta(roots, zero, monkeypatch):
    env = environment(roots)
    document, admitted = fixture_admission(env, zero=zero)
    sample = admission.prepare_camera_policy_sampler(document, admitted, env)
    admitted["summary"]["selected_feedback_kind"] = "invented"
    for owner, name in ((admission, "_runtime_sources"), (admission, "canonical_hash"),
                         (actor, "runtime_hashes"), (actor, "canonical_hash"), (actor, "prepare_camera_shared_task")):
        monkeypatch.setattr(owner, name, _deny)
    action, diagnostic = sample()
    authored = np.clip(env.features.sum(axis=1), -1., 1.)
    assert env.calls == 1 and diagnostic == {"phase": .5, "authored_action": authored.tolist(),
        "applied_joint_delta": (action[:19] - authored[:19]).tolist()}
    assert np.array_equal(action[19:], authored[19:])
    assert bool(np.any(action[:19] != authored[:19])) is (not zero)


def test_deadline_and_oversized_policy_fail_before_runtime_reads(monkeypatch):
    env = environment()
    monkeypatch.setattr(admission, "_runtime_sources", _deny)
    with pytest.raises(TimeoutError):
        admission.prepare_camera_policy_admission(env.scene, env.task, training_manifest_raw=b"{}",
            training_files={"policy.json": b"{}"}, deadline=time.monotonic() - 1)
    with pytest.raises(ValueError, match="64 KiB"):
        admission.prepare_camera_policy_admission(env.scene, env.task, training_manifest_raw=b"{}",
            training_files={"policy.json": b" " * (65536 + 1)}, deadline=time.monotonic() + 10)
