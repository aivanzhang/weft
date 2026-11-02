"""Native timing remains pinned across policies and immutable source wrappers; no training."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_exercise_body_source import forbid_runtime, pose_run
from test_native_clock_runtime import clocked_scene, native_binding
from weft import exercise_body_source, shared_exercise_lab as lab
from weft import shared_exercise_learning as gains
from weft import shared_exercise_residual_learning as residual
from weft import shared_exercise_feasibility_learning as feasibility
from weft import shared_exercise_feedback_residual as feedback
from weft import shared_exercise_equipment_feedback_policy as equipment
from weft.io import canonical_hash, digest, write_json
from weft.person_inventory import make_person_inventory
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_world import HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE, validate_shared_world
from weft.shared_world_human import construct_human_only_run
from weft.shared_world_packaging import collect_shared_source_files


@pytest.mark.parametrize("profile", ["shared-world-v2", HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE])
def test_workflow_backend_keeps_native_timing_in_policy_closure(profile):
    from weft.exercise_learning_pipeline import _learning_backend

    scene = {"profile": profile, "human_spec": {"native_clock": {}}}
    backend = _learning_backend({"policy_family": "phase-residual"}, scene)
    assert backend[2]() == residual._scene_runtime_hashes(scene)
    assert "source_clock.py" in backend[2]()


@pytest.mark.parametrize("runtime", [gains._runtime_hashes, residual._scene_runtime_hashes,
    feasibility._scene_runtime_hashes, feedback.runtime_hashes, equipment.runtime_hashes])
@pytest.mark.parametrize("profile", ["shared-world-v2", HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE])
def test_native_policy_closure_adds_only_clock_and_preserves_legacy(runtime, profile):
    legacy = {"profile": profile, "human_spec": {}}
    native = {"profile": profile, "human_spec": {"native_clock": {}}}
    previous, current = runtime(legacy), runtime(native)
    assert "source_clock.py" not in previous
    assert current == {**previous, "source_clock.py": digest(Path(gains.__file__).with_name("source_clock.py"))}


@pytest.mark.parametrize("profile", ["shared-world-v2", HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE])
def test_lab_closure_preserves_legacy_and_pins_native_clock_once(profile):
    legacy = {"profile": profile, "human_spec": {}}
    native = {"profile": profile, "human_spec": {"native_clock": {}}}
    old, new = lab.runtime_files(legacy), lab.runtime_files(native)
    assert "source_clock.py" not in old
    assert new.count("source_clock.py") == 1
    assert tuple(name for name in new if name != "source_clock.py") == old


@pytest.mark.parametrize("kind", ["gains", "residual", "feasibility", "feedback"])
@pytest.mark.parametrize("tamper", ["replace", "omit"])
def test_policy_validator_rejects_native_clock_codepin_tampering_without_training(kind, tamper):
    scene = validate_shared_world(clocked_scene())
    task = build_shared_exercise_task(scene)
    runtime = {"gains": gains._runtime_hashes, "residual": residual._scene_runtime_hashes,
               "feasibility": feasibility._scene_runtime_hashes, "feedback": feedback.runtime_hashes}[kind]
    hashes = runtime(scene)
    env = SimpleNamespace(scene=scene, task=task, scene_sha256=canonical_hash(scene), task_sha256=canonical_hash(task),
        evaluator_sha256=hashes["shared_exercise_task.py"], action_names=scene["action_contract"]["names"][:19], action_count=19)
    if kind == "gains":
        policy = gains._policy(scene, task, env, np.ones((3, 3)), 1729, .5, hashes)
        validate = gains.validate_shared_exercise_policy
    elif kind == "residual":
        policy = residual._policy(env, np.zeros((2, 19)), 1729, .5, hashes)
        validate = residual.validate_shared_exercise_residual_policy
    else:
        rank = {"profile": feasibility.RANK_PROFILE, "complete": True, "feasible": True,
                "max_normalized_excess": 0., "mean_normalized_excess": 0., "violated_terms": 0,
                "gate_terms": 2 * (len(feasibility.BODY_GATES) + len(feasibility.EQUIPMENT_GATES) * len(task["equipment_ids"])
                                   + 3 * len(task["contact_requirements"])), "reward_score": .5}
        if kind == "feedback":
            policy = feedback.make_policy(env, np.zeros((3, 2)), 1729, .5, rank, hashes)
            validate = feedback.validate_shared_exercise_feedback_policy
        else:
            selected = {"residual_coefficients": np.zeros((2, 19)).tolist(), "score": .5, "optimization_rank": rank}
            policy = feasibility._policy(env, selected, 1729, hashes, {"original_training_manifest_sha256": "a" * 64})
            validate = feasibility.validate_shared_exercise_feasibility_policy
    assert validate(policy, scene, task) == policy
    corrupted = deepcopy(policy)
    if tamper == "replace":
        corrupted["runtime_sha256"]["source_clock.py"] = "0" * 64
    else:
        del corrupted["runtime_sha256"]["source_clock.py"]
    with pytest.raises(ValueError, match="runtime"):
        validate(corrupted, scene, task)


def test_person_wrapper_and_learning_package_retain_exact_native_evidence(tmp_path, monkeypatch):
    original = pose_run(tmp_path / "poses")
    evidence = json.loads((original / "evidence.json").read_bytes())
    binding = native_binding(evidence, [60000 + i * 1000 for i in range(63)],
                             time_base="1/30000", offset=-1.997)
    binding["evidence_sha256"] = digest(original / "evidence.json")
    clock = tmp_path / "clock.json"
    write_json(clock, binding)
    forbid_runtime(monkeypatch)
    body = tmp_path / "body"
    exercise_body_source.prepare_exercise_body_source_from_evidence(original, body, native_clock=clock)
    kept = {name: (body / name).read_bytes() for name in (
        "native-clock.json", "reconstruction-evidence.json", "evidence.json", "source-clock.json")}
    candidate = tmp_path / "candidate"
    result = construct_human_only_run(body, make_person_inventory(binding["source_sha256"]), candidate)
    assert all((candidate / name).read_bytes() == raw for name, raw in kept.items())
    scene_raw = (candidate / "scene.json").read_bytes()
    scene = json.loads(scene_raw)
    proof = collect_shared_source_files(scene, candidate, expected_manifest_sha256=result["manifest_sha256"])
    manifest = (candidate / "manifest.json").read_bytes()
    views = residual._source_views(candidate, manifest, scene)
    assert all(views[name] == raw for name, raw in kept.items())
    task = build_shared_exercise_task(scene)
    packaged = tmp_path / "packaged"
    gains._package(packaged, scene_raw, gains._bytes(task), manifest, proof, extra=views)
    assert (packaged / "implementation/source_clock.py").read_bytes() == Path(gains.__file__).with_name("source_clock.py").read_bytes()
    assert all(raw in {p.read_bytes() for p in packaged.rglob("*.json")} for raw in kept.values())


@pytest.mark.parametrize("reader", [gains._native_source_views, residual._source_views])
@pytest.mark.parametrize("mutation", [*sorted(lab.NATIVE_SOURCE_VIEWS), "changed_original", "changed_derived", "changed_binding"])
def test_native_source_readers_reject_resealed_missing_or_changed_receipts(tmp_path, reader, mutation):
    from test_source_clock import clock_fixture
    from weft.source_clock import clock_summary, derive_evidence

    evidence, binding, raw = clock_fixture()
    scene = {"profile": HUMAN_ONLY_PROFILE, "human_spec": {"native_clock": clock_summary(binding)}}
    documents = {"evidence.json": evidence, "native-clock.json": binding,
                 "reconstruction-evidence.json": derive_evidence(evidence, binding), "source-clock.json": {"samples": []}}
    files = {name: gains._bytes(doc) for name, doc in documents.items()}
    files["evidence.json"] = raw
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    manifest = gains._bytes({"files": {name: digest(tmp_path / name) for name in files}})
    assert reader(tmp_path, manifest, scene) == files
    if mutation in lab.NATIVE_SOURCE_VIEWS:
        files.pop(mutation)
    else:
        name = {"changed_original": "evidence.json", "changed_derived": "reconstruction-evidence.json",
                "changed_binding": "native-clock.json"}[mutation]
        changed = deepcopy(documents[name])
        if mutation == "changed_binding":
            changed["rows"][1]["pts"] += 1
        else:
            changed["frames"][1]["time_s"] += .001
        files[name] = gains._bytes(changed)
        (tmp_path / name).write_bytes(files[name])
    manifest = gains._bytes({"files": {name: digest(tmp_path / name) for name in files}})
    with pytest.raises(ValueError):
        reader(tmp_path, manifest, scene)


@pytest.mark.parametrize("learner", ["human", "whole_body", "shared_world"])
def test_legacy_learners_reject_native_input_before_training_or_output(tmp_path, monkeypatch, learner):
    from weft import human_learning, whole_body_learning, shared_world_learning

    scene = validate_shared_world(clocked_scene())
    out = tmp_path / "forbidden-training"
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported native learner must stop before constructing a physics environment")
    monkeypatch.setattr(human_learning, "HumanEnv", forbidden)
    monkeypatch.setattr(whole_body_learning, "WholeBodyEnv", forbidden)
    monkeypatch.setattr(shared_world_learning, "SharedWorldEnv", forbidden)
    with pytest.raises(ValueError, match="source-bound shared-exercise"):
        if learner == "human":
            human_learning.train_human(scene["human_spec"], out)
        elif learner == "whole_body":
            whole_body_learning.train_whole_body(scene["human_spec"], out)
        else:
            shared_world_learning.train_shared_world_policy(scene, out)
    assert not out.exists()


def test_diagnostics_exposes_detached_verified_native_elapsed_clock_not_capture_clock():
    from weft.shared_exercise_diagnostics import _ReferenceClock

    scene = validate_shared_world(clocked_scene())
    clock = _ReferenceClock(scene)
    summary = deepcopy(scene["human_spec"]["native_clock"])
    assert clock.at(0)["source_frame_bracket"]["native_video_time_available"] is True
    assert clock.native_clock == summary
    assert clock.native_clock["capture_clock_status"] == "unknown"
    scene["human_spec"]["native_clock"]["binding_sha256"] = "0" * 64
    assert clock.native_clock == summary
