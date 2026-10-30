"""Installed policies retain exact custody, human-only controls and reproducible intent."""

from copy import deepcopy
import json
import shutil

import pytest

from weft.capabilities import describe_run_capabilities
from weft.io import digest, write_json
from weft.lab import Lab
from weft.recording import Recorder, load_recording
from weft.shared_exercise_lab import load_shared_exercise_lab, prepare_shared_exercise_lab
from weft.shared_exercise_learning import evaluate_shared_exercise_policy, train_shared_exercise_policy
from weft.shared_exercise_recording import recording_scene, validate_event, validate_recording_scene, verify_action_replay
from weft.shared_exercise_task import SharedExerciseTaskEnv
from test_shared_exercise_run import source_fixture
from test_shared_world_packaging import candidate


def seal(directory, *, status=None):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_bytes()) if path.exists() else {"schema_version": 1}
    if status is not None:
        manifest["status"] = status
    manifest["files"] = {item.relative_to(directory).as_posix(): digest(item)
                         for item in directory.rglob("*") if item.is_file() and item != path}
    write_json(path, manifest)


def mutate(path, operation):
    doc = json.loads(path.read_bytes())
    operation(doc)
    write_json(path, doc)


@pytest.fixture(scope="module")
def bundles(tmp_path_factory):
    # One tiny real training/comparison per contract, reused by all packaging/tamper checks.
    cache = {}

    def get(root_assistance=False, qualified=False):
        key = (root_assistance, qualified)
        if key not in cache:
            root = tmp_path_factory.mktemp("exercise_policy_bundle")
            if qualified:
                source = root / "source"
                scene, observation = candidate(source, equivalent=True)
                scene["task"]["horizon_steps"] = 3
                write_json(source / "scene.json", scene)
                seal(source)
            else:
                source, scene = source_fixture(root)
                observation = None
            scene["learned_policy_available"] = False
            write_json(source / "scene.json", scene)
            seal(source)
            learning = root / "learning"
            train_shared_exercise_policy(source, learning, root_assistance=root_assistance,
                                          generations=1, population=4, max_seconds=20)
            evaluation = root / "evaluation"
            evaluate_shared_exercise_policy(learning, evaluation, trials=1, max_seconds=20)
            cache[key] = {"source": source, "scene": scene, "learning": learning, "evaluation": evaluation,
                          "observation": observation}
        return cache[key]

    return get


@pytest.mark.parametrize("root_assistance", [False, True])
def test_policy_lab_pins_complete_learning_and_separate_evaluation(tmp_path, bundles, root_assistance):
    bundle = bundles(root_assistance)
    learning, evaluation = bundle["learning"], bundle["evaluation"]
    before = {directory: {path.relative_to(directory): digest(path) for path in directory.rglob("*") if path.is_file()}
              for directory in (bundle["source"], learning, evaluation)}
    result = prepare_shared_exercise_lab(bundle["source"], tmp_path / "lab", root_assistance=root_assistance,
                                        policy_run=learning, evaluation_run=evaluation)
    target = tmp_path / "lab"
    assert result["learned_policy_available"]
    assert (target / "run/policy.json").read_bytes() == (learning / "policy.json").read_bytes()
    assert (target / "run/policy-evaluation.json").read_bytes() == (evaluation / "evaluation.json").read_bytes()
    assert (target / "run/scene.json").read_bytes() == (bundle["source"] / "scene.json").read_bytes()
    for directory, prefix in ((learning, "learning-run"), (evaluation, "policy-evaluation-run")):
        for relative, sha in before[directory].items():
            assert digest(target / "run" / prefix / relative) == sha
    lab = Lab(target)
    try:
        summary = lab.summary()
        metadata = summary["exercise_policy"]
        assert summary["learned_policy_available"] is True
        assert summary["scene"] == bundle["scene"]
        assert summary["scene"]["learned_policy_available"] is False
        assert metadata["policy_sha256"] == digest(learning / "policy.json")
        assert metadata["evaluation_status"] == "evaluated"
        assert set(metadata["evaluation_successes"]) == {"coast", "random", "authored", "learned"}
        actual = json.loads((evaluation / "evaluation.json").read_bytes())
        assert metadata["evaluation_successes"] == {name: {"successes": row["successes"], "episodes": 1}
                                                    for name, row in actual["controllers"].items()}
        assert metadata["selected_feedback_kind"] == json.loads((learning / "training.json").read_bytes())["selected_feedback_kind"]
        metadata["evaluation_successes"]["learned"]["successes"] = 999
        assert lab.summary()["exercise_policy"]["evaluation_successes"]["learned"]["successes"] <= 1
        count = 22 if root_assistance else 19
        assert summary["capabilities"]["action_contract"]["dimension"] == count
        assert summary["capabilities"]["action_contract"]["object_force_channels"] == []
        assert summary["capabilities"]["controllers"]["trained_target_id"] == "human"
        for payload in ({"policy": True, "action": [0.] * count}, {"policy": True, "controller": "coast"},
                        {"policy": True, "actor": "ball"}, {"policy": 1}):
            with pytest.raises(ValueError):
                lab.control("/api/step", payload)
        event = lab.control("/api/step", {"policy": True})["transitions"][0]
        assert event["controller"] == "learned" and event["policy_sha256"] == metadata["policy_sha256"]
        assert len(event["action"]) == count
        assert not any(event["snapshot"]["engine_state"]["normalized_action"][22:])
    finally:
        lab.close()
    assert describe_run_capabilities(target / "run")["controllers"]["learned_policy_available"]
    for directory, hashes in before.items():
        assert hashes == {path.relative_to(directory): digest(path) for path in directory.rglob("*") if path.is_file()}


def test_training_only_has_no_evaluation_claim_and_reuses_cached_factory(tmp_path, bundles, monkeypatch):
    bundle = bundles()
    prepare_shared_exercise_lab(bundle["source"], tmp_path / "lab", policy_run=bundle["learning"])
    lab = Lab(tmp_path / "lab")
    try:
        summary = lab.summary()
        assert summary["exercise_policy"]["evaluation_status"] == "not_attached"
        assert summary["exercise_policy"]["evaluation_successes"] is None
        assert not (tmp_path / "lab/run/policy-evaluation.json").exists()
        session = lab.load_session()
        expected = session.policy_action()
        session.policy["weights"][0][0] = -3

        def no_hot_hash():
            raise AssertionError("Controller actions must reuse the validated factory")

        monkeypatch.setattr("weft.shared_exercise_learning._runtime_hashes", no_hot_hash)
        assert session.policy_action() == expected
        session.step(expected)
        session.reset(12)
        assert session.policy_action() == expected
        summary["exercise_policy"]["policy_sha256"] = "0" * 64
        assert session.exercise_policy()["policy_sha256"] == digest(bundle["learning"] / "policy.json")
    finally:
        lab.close()


@pytest.mark.parametrize("mutation", ["status", "policy_hash", "scene", "task", "runtime", "selection", "source_manifest", "assistance"])
def test_training_mismatches_reject_before_environment_or_output(tmp_path, bundles, monkeypatch, mutation):
    bundle = bundles()
    learning = tmp_path / "learning"
    shutil.copytree(bundle["learning"], learning)
    if mutation == "status":
        seal(learning, status="running")
    else:
        if mutation == "policy_hash":
            mutate(learning / "training.json", lambda doc: doc.update(policy_sha256="0" * 64))
        elif mutation == "scene":
            mutate(learning / "scene.json", lambda doc: doc["task"].update(horizon_steps=4))
        elif mutation == "task":
            mutate(learning / "exercise-task.json", lambda doc: doc["control"].update(root_assistance="explicit_3_axis"))
        elif mutation == "runtime":
            with (learning / "implementation/shared_world.py").open("ab") as stream:
                stream.write(b"\n# changed frozen runtime\n")
        elif mutation == "selection":
            mutate(learning / "training.json", lambda doc: doc.update(selected_feedback_kind="fabricated_grasp"))
        elif mutation == "source_manifest":
            mutate(learning / "settings.json", lambda doc: doc.update(source_manifest_sha256="0" * 64))
        else:
            mutate(learning / "policy.json", lambda doc: doc.update(root_assistance="explicit_3_axis"))
            mutate(learning / "training.json", lambda doc: doc.update(policy_sha256=digest(learning / "policy.json")))
        seal(learning)

    def no_environment(*args, **kwargs):
        raise AssertionError("Invalid learning package reached an environment")

    monkeypatch.setattr("weft.shared_exercise_lab.SharedWorldEnv", no_environment)
    monkeypatch.setattr("weft.shared_exercise_lab.SharedExerciseTaskEnv", no_environment)
    with pytest.raises(ValueError):
        prepare_shared_exercise_lab(bundle["source"], tmp_path / "unused", policy_run=learning)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("mutation", ["incomplete", "policy_bytes", "input_manifest", "aggregate", "metrics", "snapshot", "repeated_seed"])
def test_evaluation_attachment_rejects_resealed_misleading_results(tmp_path, bundles, monkeypatch, mutation):
    bundle = bundles()
    evaluation = tmp_path / "evaluation"
    shutil.copytree(bundle["evaluation"], evaluation)
    report = json.loads((evaluation / "evaluation.json").read_bytes())
    if mutation == "incomplete":
        report["comparison_complete"] = False
    elif mutation == "policy_bytes":
        with (evaluation / "policy.json").open("ab") as stream:
            stream.write(b"\n")
    elif mutation == "input_manifest":
        (evaluation / "input-manifest.json").write_text("{}\n")
    elif mutation == "aggregate":
        report["controllers"]["learned"]["successes"] += 1
    elif mutation in ("metrics", "snapshot"):
        row = report["controllers"]["learned"]["trials"][0]
        row["metrics"]["success"] = True
        row["metrics"]["completed_interval"] = False
        report["controllers"]["learned"].update(successes=1, completed_intervals=0)
        if mutation == "snapshot":
            row["final_snapshot"]["task_metrics"] = deepcopy(row["metrics"])
    else:
        report["evaluation_seeds"] = report["training_seeds"][:1]
        mutate(evaluation / "settings.json", lambda doc: doc.update(evaluation_seeds=report["evaluation_seeds"]))
    write_json(evaluation / "evaluation.json", report)
    seal(evaluation)

    def no_environment(*args, **kwargs):
        raise AssertionError("Invalid evaluation package reached an environment")

    monkeypatch.setattr("weft.shared_exercise_lab.SharedWorldEnv", no_environment)
    monkeypatch.setattr("weft.shared_exercise_lab.SharedExerciseTaskEnv", no_environment)
    with pytest.raises(ValueError):
        prepare_shared_exercise_lab(bundle["source"], tmp_path / "unused", policy_run=bundle["learning"], evaluation_run=evaluation)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("root_assistance", [False, True])
def test_mixed_controller_recording_pins_raw_policy_and_replays_exactly(tmp_path, bundles, root_assistance):
    bundle = bundles(root_assistance)
    prepare_shared_exercise_lab(bundle["source"], tmp_path / "lab", root_assistance=root_assistance,
                                policy_run=bundle["learning"])
    lab = Lab(tmp_path / "lab")
    try:
        session = lab.load_session()
        lab.control("/api/record/start", {})
        recorded = lab.recorder.root
        count = 22 if root_assistance else 19
        for payload in ({"policy": True}, {"controller": "coast"}, {"controller": "reference"}):
            lab.control("/api/step", payload)
        lab.control("/api/reset", {"seed": 32})
        lab.control("/api/step", {"action": [0.125] + [0.] * (count - 1)})
        lab.control("/api/step", {"policy": True})
        lab.recorder.close()
        lab.recorder = None
        saved = load_recording(recorded)
        assert saved["scene"]["learned_policy"]["document_utf8"].encode() == (bundle["learning"] / "policy.json").read_bytes()
        assert [row.get("controller") for row in saved["events"]] == [None, "learned", "coast", "authored", None, "manual", "learned"]
        assert all(("policy_sha256" in row) == (row.get("controller") == "learned") for row in saved["events"])
        replay = verify_action_replay(recorded)
        assert replay["exact_state_replay"] and replay["learned_actions_verified"] == 2
        assert replay["policy_sha256"] == digest(bundle["learning"] / "policy.json")
        assert session.exercise_policy()["evaluation_status"] == "not_attached"
    finally:
        lab.close()


def test_learned_identity_and_actions_cannot_be_forged(tmp_path, bundles):
    bundle = bundles()
    task = json.loads((bundle["learning"] / "exercise-task.json").read_bytes())
    envelope = recording_scene(bundle["scene"], task, policy_document=(bundle["learning"] / "policy.json").read_bytes())
    env = SharedExerciseTaskEnv(bundle["scene"], task)
    recorder = Recorder(tmp_path / "recording", envelope, profile="shared-exercise")
    try:
        env.reset(seed=0, options={"jitter": False, "jitter_scale": 0.})
        recorder.start(env.snapshot(), seed=0)
        from weft.shared_exercise_learning import prepare_shared_exercise_policy
        action = prepare_shared_exercise_policy(json.loads(envelope["learned_policy"]["document_utf8"]), env)()
        action[0] = .123 if action[0] != .123 else -.123
        _, reward, terminated, truncated, _ = env.step(action)
        event = {"type": "step", "action": action.tolist(), "reward": reward, "terminated": terminated,
                 "truncated": truncated, "snapshot": env.snapshot(), "controller": "learned",
                 "policy_sha256": envelope["learned_policy"]["sha256"]}
        for bad in ({**event, "policy_sha256": "0" * 64}, {**event, "controller": "manual"},
                    {key: value for key, value in event.items() if key != "policy_sha256"}):
            with pytest.raises(ValueError, match="policy"):
                validate_event(bad, envelope)
        with pytest.raises(ValueError, match="policy"):
            validate_event(event, recording_scene(bundle["scene"], task))
        recorder.append(event)
        recorder.close()
        with pytest.raises(ValueError, match="learned controller replay"):
            verify_action_replay(recorder.root)
        changed = deepcopy(envelope)
        changed["learned_policy"]["document_utf8"] += " "
        with pytest.raises(ValueError, match="hash"):
            validate_recording_scene(changed)
    finally:
        recorder.close()
        env.close()


def test_qualified_learning_closure_and_narrow_mapping_log_allowlist(tmp_path, bundles, monkeypatch):
    bundle = bundles(qualified=True)
    target = tmp_path / "lab"
    prepare_shared_exercise_lab(bundle["source"], target, policy_run=bundle["learning"], evaluation_run=bundle["evaluation"])
    scene = bundle["scene"]
    mapping = scene["source_binding"]["source_equivalences"][0]["bundle_path"]
    for prefix in ("learning-run", "policy-evaluation-run"):
        assert (target / "run" / prefix / mapping / "source-frames.log").is_file()
    assert load_shared_exercise_lab(target / "run")["source_verification"]["source_evidence_verified"]
    learning = tmp_path / "incomplete-learning"
    shutil.copytree(bundle["learning"], learning)
    mutate(learning / "manifest.json", lambda doc: doc["files"].pop(bundle["observation"]))

    def no_environment(*args, **kwargs):
        raise AssertionError("Incomplete evidence reached an environment")

    monkeypatch.setattr("weft.shared_exercise_lab.SharedWorldEnv", no_environment)
    with pytest.raises(ValueError, match="manifest must cover|source"):
        prepare_shared_exercise_lab(bundle["source"], tmp_path / "unused", policy_run=learning)
    (learning / "random.log").write_text("not source mapping evidence\n")
    seal(learning)
    with pytest.raises(ValueError, match="logs"):
        prepare_shared_exercise_lab(bundle["source"], tmp_path / "unused-log", policy_run=learning)


def test_no_policy_contracts_stay_unchanged_and_evaluation_requires_policy(tmp_path, bundles):
    bundle = bundles()
    task = json.loads((bundle["learning"] / "exercise-task.json").read_bytes())
    assert set(recording_scene(bundle["scene"], task)) == {"schema_version", "profile", "engine_scene", "exercise_task", "dt", "objects"}
    with pytest.raises(ValueError, match="requires.*learning"):
        prepare_shared_exercise_lab(bundle["source"], tmp_path / "unused", evaluation_run=bundle["evaluation"])
    prepare_shared_exercise_lab(bundle["source"], tmp_path / "lab")
    contract = json.loads((tmp_path / "lab/run/exercise-lab.json").read_bytes())
    config = json.loads((tmp_path / "lab/lab-config.json").read_bytes())
    assert not any("policy" in key for key in contract if key != "learned_policy_available")
    assert not any("policy" in key for key in config if key != "learned_policy_available")
    lab = Lab(tmp_path / "lab")
    try:
        assert lab.summary()["exercise_policy"] is None
        assert lab.summary()["learned_policy_available"] is False
        with pytest.raises(ValueError, match="no learned"):
            lab.control("/api/step", {"policy": True})
    finally:
        lab.close()


@pytest.mark.parametrize("mutation", ["root_policy", "missing_training_member", "root_evaluation", "selection_metadata"])
def test_resealed_policy_lab_tampering_blocks_resume_before_environment(tmp_path, bundles, monkeypatch, mutation):
    from weft.shared_exercise_lab import SharedExerciseSession

    bundle = bundles()
    target = tmp_path / "lab"
    prepare_shared_exercise_lab(bundle["source"], target, policy_run=bundle["learning"], evaluation_run=bundle["evaluation"])
    root = target / "run"
    if mutation == "root_policy":
        with (root / "policy.json").open("ab") as stream:
            stream.write(b"\n")
    elif mutation == "root_evaluation":
        mutate(root / "policy-evaluation.json", lambda doc: doc.update(comparison_complete=False))
    elif mutation == "selection_metadata":
        mutate(root / "exercise-lab.json", lambda doc: doc["exercise_policy"].update(selected_feedback_kind="unbound"))
    seal(root)
    if mutation == "missing_training_member":
        mutate(root / "manifest.json", lambda doc: doc["files"].pop("learning-run/implementation/shared_world.py"))

    def no_environment(*args, **kwargs):
        raise AssertionError("Tampered ready lab reached an environment")

    monkeypatch.setattr("weft.shared_exercise_lab.SharedExerciseTaskEnv", no_environment)
    with pytest.raises(ValueError):
        SharedExerciseSession(root)


def test_oversized_exact_policy_document_rejects_before_environment(tmp_path, bundles, monkeypatch):
    bundle = bundles()
    learning = tmp_path / "learning"
    shutil.copytree(bundle["learning"], learning)
    with (learning / "policy.json").open("ab") as stream:
        stream.write(b" " * (64 * 1024))
    mutate(learning / "training.json", lambda doc: doc.update(policy_sha256=digest(learning / "policy.json")))
    seal(learning)

    def no_environment(*args, **kwargs):
        raise AssertionError("Oversized policy reached an environment")

    monkeypatch.setattr("weft.shared_exercise_lab.SharedWorldEnv", no_environment)
    with pytest.raises(ValueError, match="document bound"):
        prepare_shared_exercise_lab(bundle["source"], tmp_path / "unused", policy_run=learning)
    assert not (tmp_path / "unused").exists()


def test_exact_utf8_bom_policy_bytes_survive_recording_and_action_replay(tmp_path, bundles):
    from weft.shared_exercise_learning import prepare_shared_exercise_policy

    bundle = bundles()
    task = json.loads((bundle["learning"] / "exercise-task.json").read_bytes())
    policy_raw = b"\xef\xbb\xbf" + (bundle["learning"] / "policy.json").read_bytes()
    envelope = recording_scene(bundle["scene"], task, policy_document=policy_raw)
    env = SharedExerciseTaskEnv(bundle["scene"], task)
    recorder = Recorder(tmp_path / "recording", envelope, profile="shared-exercise")
    try:
        env.reset(seed=0, options={"jitter": False, "jitter_scale": 0.})
        recorder.start(env.snapshot(), seed=0)
        action = prepare_shared_exercise_policy(json.loads(policy_raw), env)().tolist()
        _, reward, terminated, truncated, _ = env.step(action)
        recorder.append({"type": "step", "action": action, "reward": reward, "terminated": terminated,
                         "truncated": truncated, "snapshot": env.snapshot(), "controller": "learned",
                         "policy_sha256": envelope["learned_policy"]["sha256"]})
        recorder.close()
        assert load_recording(recorder.root)["scene"]["learned_policy"]["document_utf8"].encode() == policy_raw
        assert verify_action_replay(recorder.root)["learned_actions_verified"] == 1
    finally:
        recorder.close()
        env.close()
