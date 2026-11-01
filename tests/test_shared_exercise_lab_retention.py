"""Loader ownership and proof failures, with document validators and physics mocked."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import time
import weakref

import pytest

from weft import shared_exercise_lab as lab


class TrackedBytes(bytearray):
    pass


class TrackedFiles(dict):
    pass


def setup_packet(monkeypatch, mode="evaluated"):
    camera = mode in {"camera_policy", "reference"}
    learned = mode in {"trained", "evaluated", "camera_policy"}
    scene = {"profile": lab.CAMERA_PROFILE if camera else "shared-world-v1",
             "source_binding": {"mode": "analytic_fixture"}}
    task, policy, status = {"authored_task": True}, {"fixture_policy": True}, {"fixture_status": True}
    source = {"scene.json": lab._bytes(scene), "source-payload.bin": b"s" * 4096}
    seal = lambda files: lab._bytes({"files": {name: lab._sha(raw) for name, raw in files.items()}})
    source_manifest = seal(source)
    runtime = {"retention-fixture.py": lab._sha(b"runtime fixture")}
    contract = {"schema_version": 1, "profile": "shared-world", "control_profile": lab.CONTROL_PROFILE,
        "scene_sha256": lab.canonical_hash(scene), "task_sha256": lab.canonical_hash(task),
        "reset_options": dict(lab.RESET_OPTIONS), "source_manifest_sha256": lab._sha(source_manifest),
        "source_files_verified": len(source), "learned_policy_available": learned,
        "implementation_sha256": runtime, "runtime_sha256": runtime, "runtime_versions": {"fixture": True}}
    files = {"scene.json": source["scene.json"], "exercise-task.json": lab._bytes(task),
        "source-run/manifest.json": source_manifest, "lab-implementation/retention-fixture.py": b"runtime fixture",
        "root-payload.bin": b"r" * 4096, **{"source-run/" + name: raw for name, raw in source.items()}}
    expected_checks, events, file_refs, payload_refs, proof_refs = [], [], [], [], []
    root = Path("/authored-lab/run")
    if learned:
        files["policy.json"] = lab._bytes(policy)
    if mode in {"trained", "evaluated"}:
        learning = {"policy.json": files["policy.json"], "training.json": b"{}", "learning-payload.bin": b"l" * 4096}
        learning_manifest = seal(learning)
        files.update({"learning-run/" + name: raw for name, raw in learning.items()})
        files["learning-run/manifest.json"] = learning_manifest
        identity = {"learning_manifest_sha256": lab._sha(learning_manifest), "policy_sha256": lab._sha(files["policy.json"])}
        contract.update(identity, exercise_policy=status)
        expected_checks.append((root / "learning-run", {"expected_manifest_sha256": identity["learning_manifest_sha256"]}))

        def validate_policy(raw, members, actual_scene, actual_task, **kwargs):
            assert raw == learning_manifest and members == learning
            assert actual_scene == scene and actual_task == task
            assert len(file_refs) == 1 and file_refs[0]() is not None
            events.append("policy-metadata")
            return deepcopy(policy), dict(identity)

        monkeypatch.setattr(lab, "_validate_policy_package", validate_policy)
    if mode == "evaluated":
        evaluation = {"evaluation.json": b"{}", "evaluation-payload.bin": b"e" * 4096}
        evaluation_manifest = seal(evaluation)
        files.update({"policy-evaluation-run/" + name: raw for name, raw in evaluation.items()})
        files.update({"policy-evaluation-run/manifest.json": evaluation_manifest, "policy-evaluation.json": b"{}"})
        evaluation_identity = {"policy_evaluation_manifest_sha256": lab._sha(evaluation_manifest),
                               "policy_evaluation_sha256": lab._sha(b"{}")}
        contract.update(evaluation_identity)
        expected_checks.append((root / "policy-evaluation-run", {"expected_manifest_sha256": evaluation_identity["policy_evaluation_manifest_sha256"]}))

        def validate_evaluation(raw, members, learning_raw, learning_members, actual_policy):
            assert raw == evaluation_manifest and members == evaluation
            assert learning_raw == learning_manifest and learning_members == learning and actual_policy == policy
            events.append("evaluation-metadata")
            return dict(evaluation_identity), {"learned": {"successes": 0}}, {"fixture": True}

        monkeypatch.setattr(lab, "_validate_policy_evaluation", validate_evaluation)
    if mode == "camera_policy":
        from weft import camera_shared_policy_admission, camera_shared_policy_lab

        monkeypatch.setattr(camera_shared_policy_admission, "execution_runtime_hashes", lambda _scene: runtime)
        admission = {"fixture_admission": True}
        def camera_attachment(members, *_args, **_kwargs):
            events.append("camera-metadata")
            return {"policy": deepcopy(policy), "admission": admission,
                "summary": {"evaluation_successes": None}, "files": dict(members)}
        monkeypatch.setattr(camera_shared_policy_lab, "load_camera_attachment", camera_attachment)
    if mode == "reference":
        from weft import camera_reference_lab

        recipe = b'{"authored_recipe":true}'
        contract["reference_controller_sha256"] = lab._sha(recipe)
        files["reference-controller.json"] = recipe
        def reference_attachment(members, _scene, _task, source_raw, originals, *_args, **_kwargs):
            assert originals == source and source_raw == source_manifest
            events.append("reference-metadata")
            return {"recipe_document": recipe, "runtime_sha256": runtime,
                "identity": {"reference_controller_sha256": lab._sha(recipe)}, "files": dict(members)}
        monkeypatch.setattr(camera_reference_lab, "load_reference_attachment", reference_attachment)
    files["exercise-lab.json"] = lab._bytes(contract)
    manifest = lab._bytes({"control_profile": lab.CONTROL_PROFILE, "files": {name: lab._sha(raw) for name, raw in files.items()}})
    deadline = time.monotonic() + 10
    expected_checks.append((root, {"expected_manifest_sha256": lab._sha(manifest), "deadline": deadline}))
    config = {key: deepcopy(contract[key]) for key in ("schema_version", "profile", "control_profile", "scene_sha256", "task_sha256",
        "reset_options", "learned_policy_available")}
    config.update(manifest_sha256=lab._sha(manifest))
    for key in ("policy_sha256", "learning_manifest_sha256", "reference_controller_sha256",
                "policy_evaluation_manifest_sha256", "policy_evaluation_sha256"):
        if key in contract:
            config[key] = contract[key]
    def verified(*_args, **_kwargs):
        packet = TrackedFiles(files)
        file_refs.append(weakref.ref(packet))
        for name in packet:
            if name.endswith("-payload.bin"):
                packet[name] = TrackedBytes(packet[name])
                payload_refs.append(weakref.ref(packet[name]))
        return manifest, packet
    monkeypatch.setattr(lab, "_verified_inputs", verified)
    monkeypatch.setattr(lab, "validate_shared_world", lambda value: value)
    monkeypatch.setattr(lab, "_validate_task", lambda value, _scene: value)
    monkeypatch.setattr(lab, "runtime_files", lambda _scene: tuple(runtime))
    monkeypatch.setattr(lab, "_runtime_versions", lambda: {"fixture": True})
    monkeypatch.setattr(lab, "read_package_file", lambda *_args, **_kwargs: b"runtime fixture")
    monkeypatch.setattr(lab, "_policy_status", lambda *_args, **_kwargs: deepcopy(status))
    def forbidden(*_args, **_kwargs):
        pytest.fail("Loader must not create a live environment or controller")
    monkeypatch.setattr(lab, "SharedExerciseTaskEnv", forbidden)
    monkeypatch.setattr(lab, "SharedWorldEnv", forbidden)
    from weft import shared_exercise_policies
    monkeypatch.setattr(shared_exercise_policies, "prepare_policy", forbidden)
    return SimpleNamespace(root=root, files=files, manifest=manifest, config=config, deadline=deadline, scene=scene,
        task=task, policy=policy, contract=contract, expected_checks=expected_checks, events=events,
        file_refs=file_refs, payload_refs=payload_refs, proof_refs=proof_refs)


def source_checker(data, *, failing_index=None):
    calls = []
    def collect(scene, root, **kwargs):
        assert scene == data.scene
        assert all(reference() is None for reference in data.file_refs + data.payload_refs + data.proof_refs)
        assert (root, kwargs) == data.expected_checks[len(calls)]
        calls.append((root, kwargs)); data.events.append("source-proof:" + root.name)
        if len(calls)-1 == failing_index:
            raise ValueError("Corrupt source proof: " + root.name)
        payload = TrackedBytes(b"source proof buffer")
        data.proof_refs.append(weakref.ref(payload))
        return SimpleNamespace(report={"verified_root": str(root)}, files={"payload.bin": payload})
    return collect, calls


@pytest.mark.parametrize("mode", ["untrained", "trained", "evaluated", "camera_policy", "reference"])
def test_archive_and_all_aliases_are_released_before_each_unchanged_source_proof(monkeypatch, mode):
    data = setup_packet(monkeypatch, mode)
    collect, calls = source_checker(data)
    monkeypatch.setattr(lab, "collect_shared_source_files", collect)
    result = lab.load_shared_exercise_lab(data.root, deadline=data.deadline)
    assert calls == data.expected_checks
    expected = {"scene": data.scene, "task": data.task, "contract": data.contract,
        "source_verification": {"verified_root": str(data.root)},
        "artifacts": [{"name": name, "bytes": len(raw)} for name, raw in sorted(data.files.items()) if "/" not in name]
                     + [{"name": "manifest.json", "bytes": len(data.manifest)}]}
    if mode in {"trained", "evaluated", "camera_policy"}:
        expected.update(policy=data.policy, policy_document=data.files["policy.json"])
    if mode == "camera_policy":
        expected["camera_policy_admission"] = {"fixture_admission": True}
    if mode == "reference":
        expected["reference_document"] = data.files["reference-controller.json"]
    assert result == expected
    assert all(reference() is None for reference in data.proof_refs)


@pytest.mark.parametrize("failing_index", [0, 1, 2])
def test_corrupt_learning_evaluation_or_root_proof_exposes_no_session(monkeypatch, failing_index):
    data = setup_packet(monkeypatch)
    monkeypatch.setattr(lab, "time", SimpleNamespace(monotonic=lambda: data.deadline - 60))
    collect, calls = source_checker(data, failing_index=failing_index)
    monkeypatch.setattr(lab, "collect_shared_source_files", collect)
    with pytest.raises(ValueError, match="Corrupt source proof"):
        lab.SharedExerciseSession(data.root)
    assert calls == data.expected_checks[:failing_index+1]


def test_invalid_config_rejects_before_deferred_reads_without_a_partial_result(monkeypatch):
    data = setup_packet(monkeypatch)
    data.config["manifest_sha256"] = "0" * 64
    collect, calls = source_checker(data)
    monkeypatch.setattr(lab, "collect_shared_source_files", collect)
    with pytest.raises(ValueError, match="configuration differs"):
        lab.load_shared_exercise_lab(data.root, config=data.config, deadline=data.deadline)
    assert calls == []
    assert data.events == ["policy-metadata", "evaluation-metadata"]


def test_changed_source_payload_is_still_rejected_by_original_hash_check(monkeypatch):
    data = setup_packet(monkeypatch)
    data.files["source-run/source-payload.bin"] = b"changed after seal"
    collect, calls = source_checker(data)
    monkeypatch.setattr(lab, "collect_shared_source_files", collect)
    with pytest.raises(ValueError, match="complete sealed source candidate"):
        lab.load_shared_exercise_lab(data.root, deadline=data.deadline)
    assert not calls and not data.events


def test_expired_deadline_after_proofs_still_exposes_no_result(monkeypatch):
    data = setup_packet(monkeypatch, "untrained")
    collect, calls = source_checker(data)
    def expire(*args, **kwargs):
        value = collect(*args, **kwargs)
        monkeypatch.setattr(lab, "time", SimpleNamespace(monotonic=lambda: data.deadline + 1))
        return value
    monkeypatch.setattr(lab, "collect_shared_source_files", expire)
    with pytest.raises(TimeoutError):
        lab.load_shared_exercise_lab(data.root, deadline=data.deadline)
    assert calls == data.expected_checks
