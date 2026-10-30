"""One protocol preserves differing action/task contracts and complete replay evidence."""

from copy import deepcopy
import json
import time

import pytest

from weft import runtime_qualification as qualification
from weft.human import JOINT_NAMES, build_human
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts
from weft.recording import load_recording
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_world_lab import authored_shared_world_scene
from weft.spatial_lab import authored_spatial_scene
from weft.world import World
from test_capabilities import make_planar_scene
from test_mobile_integration import evidence as mobile_evidence


def seal(root, files):
    root.mkdir(parents=True)
    for name, document in files.items():
        write_json(root / name, document)
    write_json(root / "manifest.json", {"schema_version": 1,
        "files": {name: digest(root / name) for name in files}})
    return root


def rigid(tmp_path, profile="spatial-rigid-v1", horizon=10):
    if profile.startswith("planar"):
        scene = make_planar_scene(profile, multi_object=True)
        scene["task"]["max_steps"] = horizon
    else:
        scene = authored_spatial_scene()
        scene["task"]["max_steps"] = horizon
    return seal(tmp_path, {"scene.json": scene})


def shared(tmp_path, *, exercise=False, full=False, assistance=False, horizon=8):
    scene = authored_shared_world_scene()
    scene["task"]["horizon_steps"] = horizon
    if full:
        from weft.shared_world import full_reference_horizon
        scene["profile"] = "shared-world-v2"
        scene["task"]["horizon_steps"] = full_reference_horizon(scene["human_spec"]["reference"], scene["dt"])
    files = {"scene.json": scene}
    if exercise:
        files["exercise-task.json"] = build_shared_exercise_task(scene, root_assistance=assistance)
    return seal(tmp_path, files)


def body(tmp_path, mode, *, direct=None):
    if mode == "hinge":
        evidence = {"source_sha256": "a" * 64,
            "lengths": dict(torso=.45, upper_arm=.3, forearm=.27, thigh=.43, shin=.43, shoulder_width=.38, hip_width=.28),
            "reference": [{"time_s": t, "joints": {name: .1 + t * .2 for name in JOINT_NAMES}} for t in (0., .5, 1.)]}
    else:
        evidence = mobile_evidence()
    spec = build_human(evidence, tmp_path / "generated", body_model=mode)
    files = {"human.json": spec}
    if direct == "whole_body":
        from weft.whole_body import WholeBodyEnv
        env = WholeBodyEnv(spec, horizon_steps=6)
        try:
            files["scene.json"] = env.scene
        finally:
            env.close()
        files["settings.json"] = {"partition": {"evaluation_start_times_s": [0.], "horizon_steps": 6}}
    elif direct == "feet":
        from weft.foot_reference import build_foot_reference
        from weft.foot_sandbox import FootEnv
        from test_foot_reference import fixture
        _, evidence, metadata = fixture()
        reference = build_foot_reference(spec, evidence, metadata)
        env = FootEnv(spec, reference, horizon_steps=6)
        try:
            files["scene.json"] = env.scene
        finally:
            env.close()
        files["foot-reference.json"] = reference
    return seal(tmp_path / "sealed", files)


def case(root, name="trial", controller="coast", **kwargs):
    return {"id": name, "input": str(root), "controller": controller, "seed": 37, "probe_steps": 3, **kwargs}


def test_one_protocol_records_mixed_action_spaces_and_does_not_claim_task_or_physical_success(tmp_path):
    roots = [rigid(tmp_path / "planar", "planar-rigid-v2"), rigid(tmp_path / "spatial"),
             shared(tmp_path / "human-equipment", exercise=True)]
    before = [{path.name: digest(path) for path in root.iterdir()} for root in roots]
    result = qualification.qualify_runtimes([case(root, f"case-{index}", "authored" if index == 2 else "coast")
        for index, root in enumerate(roots)], tmp_path / "out")
    assert result["status"] == "completed", result
    assert result["physical_validation"] == "unverified"
    assert result["policy_evaluation"] == "not_performed"
    assert result["env_steps_including_replay"] == 65
    assert [row["control_profile"] for row in result["cases"]] == ["planar", "spatial", "shared_exercise"]
    assert all(row["exact_action_replay"] and row["same_seed_reset"] for row in result["cases"])
    assert all(row["status"] == "qualified" for row in result["cases"])
    assert result["cases"][1]["task_result"]["value"]["success"] is False
    for index, root in enumerate(roots):
        assert before[index] == {path.name: digest(path) for path in root.iterdir()}
        recording = load_recording(tmp_path / "out" / f"case-{index}" / "recording")
        assert len(recording["events"]) == (9 if index == 2 else 11)
        actions = [row["action"] for row in recording["events"][1:]]
        if index == 2:
            assert all(len(action) == 19 for action in actions)
            assert all(not any(row["snapshot"]["engine_state"]["normalized_action"][22:])
                       for row in recording["events"])
        else:
            assert all(type(action) is int for action in actions)
    assert verify_artifacts(tmp_path / "out")["manifest_sha256"] == result["manifest_sha256"]


@pytest.mark.parametrize("profile", ["rolling", "planar-rigid-v1", "exercise-v1", "exercise-v2", "exercise-v3",
    "exercise-imitation-v1", "exercise-feet-posture-v1", "shared-world-v1", "shared-world-v2", "human-only-root"])
def test_all_capability_profiles_have_explicit_recordable_adapters(tmp_path, profile):
    controller = "coast"
    if profile == "rolling":
        world = World(.2, 0., .5, .04, 0., .75, 640, 360, 0)
        root = seal(tmp_path / "input", {"world.json": world.to_dict()})
    elif profile.startswith("planar"):
        root = rigid(tmp_path / "input", profile)
    elif profile.startswith("shared") or profile == "human-only-root":
        root = shared(tmp_path / "input", exercise=profile == "human-only-root",
                      full=profile == "shared-world-v2", assistance=True)
        controller = "authored"
    else:
        mode = {"exercise-v1": "hinge", "exercise-v2": "spatial"}.get(profile, "mobile")
        direct = {"exercise-imitation-v1": "whole_body", "exercise-feet-posture-v1": "feet"}.get(profile)
        root = body(tmp_path / "body", mode, direct=direct)
        if direct:
            controller = "authored"
    result = qualification.qualify_runtimes([case(root, controller=controller)], tmp_path / "out")
    assert result["status"] == "completed", result
    trial = result["cases"][0]
    expected = "shared-world-v1" if profile == "human-only-root" else profile
    assert trial["profile"] == expected
    assert trial["exact_action_replay"] and trial["changed_action_response"]
    contract = json.loads((tmp_path / "out/trial/contracts.json").read_text())
    assert contract["capabilities"]["scene_profile"] == expected
    assert trial["steps"] <= contract["horizon_steps"]
    if profile == "human-only-root":
        assert contract["capabilities"]["action_contract"]["dimension"] == 22
        assert contract["capabilities"]["action_contract"]["object_force_channels"] == []


@pytest.mark.parametrize("mutation", ["content", "missing", "stale_runtime", "null_runtime", "null_runtime_map", "failed", "unsafe_path", "symlink"])
def test_invalid_inputs_reject_before_environment_construction(tmp_path, monkeypatch, mutation):
    root = rigid(tmp_path / "input")
    manifest = json.loads((root / "manifest.json").read_text())
    if mutation == "content":
        (root / "scene.json").write_text("{}")
    elif mutation == "missing":
        manifest["files"]["absent.json"] = "0" * 64
    elif mutation in {"stale_runtime", "null_runtime", "null_runtime_map"}:
        pins = None if mutation == "null_runtime_map" else {"shared_world.py": None if mutation == "null_runtime" else "0" * 64}
        write_json(root / "settings.json", {"runtime_sha256": pins})
        manifest["files"]["settings.json"] = digest(root / "settings.json")
    elif mutation == "failed":
        manifest["status"] = "rejected"
    elif mutation == "unsafe_path":
        manifest["files"]["../outside"] = "0" * 64
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        root = alias
    if mutation != "symlink":
        write_json(root / "manifest.json", manifest)
    monkeypatch.setattr(qualification, "_Adapter", lambda *_: pytest.fail("Environment constructed before admission"))
    with pytest.raises(ValueError):
        qualification.qualify_runtimes([case(root)], tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("change", [{"task": {}}, {"controller": "learned"}, {"seed": True}, {"probe_steps": 0},
                                   {"probe_channel": 64}, {"id": "../escape"}])
def test_no_hidden_task_policy_or_probe_overrides(tmp_path, change):
    with pytest.raises(ValueError):
        qualification.qualify_runtimes([case(tmp_path, **change)], tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_same_seed_replay_catches_changed_reward_not_just_state(tmp_path, monkeypatch):
    root = rigid(tmp_path / "input")
    original = qualification._replay

    def changed_reward(adapter, recording, expected, budget):
        path = recording / "events.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[1]["reward"] += 1
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return original(adapter, recording, expected, budget)

    monkeypatch.setattr(qualification, "_replay", changed_reward)
    result = qualification.qualify_runtimes([case(root)], tmp_path / "out")
    assert result["status"] == "failed"
    assert "authoritative execution" in result["cases"][0]["failure"]["reason"]
    assert result["operationally_qualified"] == 0


def test_seed_mismatch_cannot_pass_reset_replay(tmp_path, monkeypatch):
    world = World(.2, 0., .5, .04, 0., .75, 640, 360, 0)
    root = seal(tmp_path / "input", {"world.json": world.to_dict()})
    original, calls = qualification._Adapter.reset, 0

    def changed_reset(self, seed):
        nonlocal calls
        calls += 1
        return original(self, seed + (1 if calls == 2 else 0))

    monkeypatch.setattr(qualification._Adapter, "reset", changed_reset)
    result = qualification.qualify_runtimes([case(root)], tmp_path / "out")
    assert result["status"] == "failed"
    assert "Same-seed" in result["cases"][0]["failure"]["reason"]


def test_late_runtime_failure_is_saved_and_never_full_interval_qualification(tmp_path, monkeypatch):
    root = rigid(tmp_path / "input", horizon=10)
    original = qualification._Adapter.step

    def fail(self, action):
        if self.env.steps == 6:
            raise RuntimeError("injected late engine failure")
        return original(self, action)

    monkeypatch.setattr(qualification._Adapter, "step", fail)
    result = qualification.qualify_runtimes([case(root)], tmp_path / "out")
    assert result["status"] == "completed", result
    row = result["cases"][0]
    assert row["steps"] == 6 and not row["complete_interval"]
    assert row["status"] == "not_qualified" and row["exact_action_replay"]
    assert row["simulation_failure"]["reason"] == "injected late engine failure"
    assert len(load_recording(tmp_path / "out/trial/recording")["events"]) == 7


def test_work_bound_rejects_complete_interval_before_first_reset(tmp_path, monkeypatch):
    root = rigid(tmp_path / "input", horizon=10)
    monkeypatch.setattr(qualification._Adapter, "reset", lambda *_: pytest.fail("Reset beyond frozen work envelope"))
    result = qualification.qualify_runtimes([case(root)], tmp_path / "out", max_steps=2)
    assert result["status"] == "failed" and result["env_steps_including_replay"] == 0
    assert not (tmp_path / "out/trial/recording").exists()


@pytest.mark.parametrize("action", [0, [0.] * 25, [0.] * 18 + [True], [0.] * 18 + [float("nan")], [2.] * 19])
def test_human_only_actions_are_validated_before_stepping(tmp_path, action):
    root = shared(tmp_path / "input", exercise=True)
    data = qualification._admit(root, time.monotonic() + 10)
    adapter = qualification._Adapter(data)
    try:
        initial = adapter.reset(37)
        with pytest.raises(ValueError):
            adapter.step(action)
        assert adapter.snapshot() == initial
    finally:
        adapter.close()


def test_physical_comparison_excludes_action_reward_phase_and_muscle_labels():
    left = {"objects": [{"id": "body", "position": [1., 2.], "velocity": [0., 0.]}],
            "human": {"angles": {"knee": .1}, "muscles": [{"activation": 0.}]},
            "action": [0.], "reward": 0., "time_s": 0.}
    right = deepcopy(left)
    right.update(action=[1.], reward=10., time_s=100.)
    right["human"]["muscles"][0]["activation"] = 1.
    assert qualification._physical(left) == qualification._physical(right)
    right["human"]["mobility"] = {"linear_velocity": [1., 0., 0.]}
    assert qualification._physical(right)["human/mobility/linear_velocity/0"] == 1.


def test_duplicate_packages_do_not_fake_diversity(tmp_path, monkeypatch):
    root = rigid(tmp_path / "input")
    monkeypatch.setattr(qualification, "_Adapter", lambda *_: pytest.fail("Duplicate package executed"))
    with pytest.raises(ValueError, match="Duplicate"):
        qualification.qualify_runtimes([case(root, "one"), case(root, "two")], tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_failed_case_does_not_block_an_independent_admitted_case(tmp_path, monkeypatch):
    first = rigid(tmp_path / "first")
    second = rigid(tmp_path / "second", horizon=12)
    original = qualification._replay

    def fail_one(adapter, recording, expected, budget):
        if adapter.horizon == 10:
            raise ValueError("injected replay mismatch")
        return original(adapter, recording, expected, budget)

    monkeypatch.setattr(qualification, "_replay", fail_one)
    result = qualification.qualify_runtimes([case(first, "first"), case(second, "second")], tmp_path / "out")
    assert result["status"] == "failed" and result["failure"] is None
    assert [row["status"] for row in result["cases"]] == ["failed", "qualified"]
    assert result["cases"][1]["steps"] == 12
    assert result["operationally_qualified"] == 1


def test_valid_early_task_success_is_not_full_interval_coverage(tmp_path):
    scene = authored_spatial_scene()
    scene["target"] = scene["objects"][0]["position"].copy()
    scene["task"].update(max_steps=10, speed_tolerance=.5)
    root = seal(tmp_path / "input", {"scene.json": scene})
    result = qualification.qualify_runtimes([case(root)], tmp_path / "out")
    row = result["cases"][0]
    assert row["status"] == "not_qualified" and row["steps"] < row["declared_horizon_steps"]
    assert row["execution_outcome"] == "valid_early_terminal"
    assert row["task_result"]["value"]["success"] is True
    assert row["exact_action_replay"] and row["same_seed_reset"]


def test_explicit_manifest_pin_rejects_resealed_input_before_construction(tmp_path, monkeypatch):
    root = rigid(tmp_path / "input")
    monkeypatch.setattr(qualification, "_Adapter", lambda *_: pytest.fail("Manifest pin bypassed"))
    with pytest.raises(ValueError, match="frozen manifest"):
        qualification.qualify_runtimes([case(root, manifest_sha256="0" * 64)], tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_output_exhaustion_stops_before_unrecorded_physics_and_seals_failure(tmp_path, monkeypatch):
    root = rigid(tmp_path / "input")
    original = qualification._Budget.ensure

    def insufficient(self, size):
        if size == 2 * qualification.MAX_EVENT_BYTES + 128:
            raise qualification._GlobalLimit("injected exhausted evidence bound")
        return original(self, size)

    monkeypatch.setattr(qualification._Budget, "ensure", insufficient)
    result = qualification.qualify_runtimes([case(root)], tmp_path / "out")
    assert result["status"] == "failed" and result["env_steps_including_replay"] == 0
    assert "evidence bound" in result["failure"]["reason"]
    assert len(load_recording(tmp_path / "out/trial/recording")["events"]) == 1
    assert verify_artifacts(tmp_path / "out")["manifest_sha256"] == result["manifest_sha256"]


def test_exponent_overflow_is_not_finite_json():
    with pytest.raises(ValueError, match="Nonfinite"):
        qualification._json(b'{"value":1e999}')


@pytest.mark.parametrize("mutation", [None, "max_steps", "target"])
def test_enriched_task_documents_bind_the_exact_executable_task(tmp_path, monkeypatch, mutation):
    scene = authored_spatial_scene()
    scene["task"]["max_steps"] = 10
    task = {**scene["task"], "profile": scene["profile"], "controlled_id": scene["controlled_id"],
            "target": scene["target"], "scope": "Authored analytic fixture"}
    if mutation == "max_steps":
        task["max_steps"] = 11
    elif mutation == "target":
        task["target"] = [.1, .1, .1]
    root = seal(tmp_path / "input", {"scene.json": scene, "task.json": task})
    if mutation:
        monkeypatch.setattr(qualification, "_Adapter", lambda *_: pytest.fail("Mismatched task reached the engine"))
        with pytest.raises(ValueError, match="task"):
            qualification.qualify_runtimes([case(root)], tmp_path / "unused")
    else:
        result = qualification.qualify_runtimes([case(root)], tmp_path / "out")
        assert result["status"] == "completed" and result["cases"][0]["status"] == "qualified"


def test_public_study_driver_accepts_relative_frozen_input_without_server_or_export(tmp_path, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts/studies/qualify_runtime.py"
    spec = importlib.util.spec_from_file_location("qualification_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    root = rigid(tmp_path / "input")
    write_json(tmp_path / "cases.json", [case("input", manifest_sha256=digest(root / "manifest.json"))])
    monkeypatch.setattr(sys, "argv", [str(path), "--cases", str(tmp_path / "cases.json"),
                                     "--out", str(tmp_path / "out"), "--max-steps", "25"])
    assert driver.main() == 0
    assert "1/1 operationally qualified" in capsys.readouterr().out
    assert not list((tmp_path / "out").rglob("*.mp4"))
    assert not (root / "sessions").exists()


def test_final_action_returned_engine_failure_is_not_complete_duration(tmp_path, monkeypatch):
    from weft.shared_world import SharedWorldEnv

    root = shared(tmp_path / "input", exercise=True, assistance=True, horizon=8)
    original = SharedWorldEnv.step

    def fail_last(self, action):
        if self.steps == 7:
            raise RuntimeError("injected final-action engine failure")
        return original(self, action)

    monkeypatch.setattr(SharedWorldEnv, "step", fail_last)
    result = qualification.qualify_runtimes([case(root, controller="authored")], tmp_path / "out")
    assert result["status"] == "completed", result
    row = result["cases"][0]
    assert row["steps"] == row["declared_horizon_steps"] == 8
    assert row["exact_action_replay"] and row["simulation_failure"] is None
    assert row["execution_outcome"] == "partial_engine_failure"
    assert not row["complete_interval"] and row["status"] == "not_qualified"
    assert not row["interval_coverage"]["authoritative_completion"]
    assert not row["interval_coverage"]["complete_engine_clock"]
    assert row["task_result"]["value"]["completed_interval"] is False
    final = load_recording(tmp_path / "out/trial/recording")["events"][-1]
    assert final["snapshot"]["engine_state_status"] == "last_valid_control_boundary"


def test_complete_interval_with_failed_frozen_task_gate_still_qualifies_operation(tmp_path):
    scene = authored_shared_world_scene()
    scene["task"]["horizon_steps"] = 8
    next(obj for obj in scene["objects"] if obj["role"] != "static")["velocity"] = [.5, 0., 0.]
    task = build_shared_exercise_task(scene, root_assistance=True)
    root = seal(tmp_path / "input", {"scene.json": scene, "exercise-task.json": task})
    result = qualification.qualify_runtimes([case(root, controller="authored")], tmp_path / "out")
    assert result["status"] == "completed", result
    row = result["cases"][0]
    assert row["complete_interval"] and row["status"] == "qualified"
    assert row["interval_coverage"]["authoritative_completion"]
    assert row["interval_coverage"]["engine_failure"] is None
    metrics = row["task_result"]["value"]
    assert metrics["completed_interval"] and not metrics["success"]
    assert any(name.startswith("equipment_stability:") for name in metrics["failed_gates"])
    assert row["physical_validation"] == "unverified"


@pytest.mark.parametrize("name", ["lab-config.json", "status.json"])
def test_outer_lab_configuration_and_readiness_bytes_are_pinned_and_rechecked(tmp_path, monkeypatch, name):
    lab = tmp_path / "lab"
    run = rigid(lab / "run")
    write_json(lab / "lab-config.json", {"profile": "spatial-rigid", "manifest_sha256": digest(run / "manifest.json")})
    write_json(lab / "status.json", {"status": "ready"})
    before = digest(lab / name)
    original = qualification._replay

    def mutate_outer(*args):
        result = original(*args)
        value = json.loads((lab / name).read_text())
        value["unrecorded_change"] = True
        write_json(lab / name, value)
        return result

    monkeypatch.setattr(qualification, "_replay", mutate_outer)
    result = qualification.qualify_runtimes([case(lab)], tmp_path / "out")
    assert result["status"] == "failed"
    assert not result["input_manifests_and_consumed_documents_unchanged"]
    assert result["cases"][0]["input"]["lab_metadata"][name] == before
    assert "SHA256" in result["failure"]["reason"]


def test_recorder_own_capacity_is_checked_before_the_next_action(tmp_path, monkeypatch):
    root = rigid(tmp_path / "input")
    original = qualification.Recorder.check_capacity

    def exhausted(self):
        if self._steps >= 2:
            raise ValueError("injected independent recorder journal limit")
        return original(self)

    monkeypatch.setattr(qualification.Recorder, "check_capacity", exhausted)
    result = qualification.qualify_runtimes([case(root)], tmp_path / "out")
    assert result["status"] == "failed" and result["env_steps_including_replay"] == 2
    assert "recorder journal limit" in result["cases"][0]["failure"]["reason"]
    assert len(load_recording(tmp_path / "out/trial/recording")["events"]) == 3


def test_post_admission_deadline_does_not_create_an_unsealed_output(tmp_path, monkeypatch):
    root = rigid(tmp_path / "input")
    original = qualification._admit

    def exhaust_after_admission(*args):
        result = original(*args)
        monkeypatch.setattr(qualification, "_deadline", lambda _: (_ for _ in ()).throw(TimeoutError("expired")))
        return result

    monkeypatch.setattr(qualification, "_admit", exhaust_after_admission)
    monkeypatch.setattr(qualification, "_Adapter", lambda *_: pytest.fail("Expired plan constructed an environment"))
    with pytest.raises(TimeoutError, match="expired"):
        qualification.qualify_runtimes([case(root)], tmp_path / "unused")
    assert not (tmp_path / "unused").exists()


def test_pre_start_recording_budget_failure_is_not_masked_by_cleanup(tmp_path, monkeypatch):
    root = rigid(tmp_path / "input")
    original_recorder, original_reserve = qualification.Recorder, qualification._Budget.reserve
    created = []

    def recorder(*args, **kwargs):
        value = original_recorder(*args, **kwargs)
        created.append(value)
        return value

    def reserve(self, size):
        if created and created[-1]._started is None:
            raise qualification._GlobalLimit("injected pre-start evidence limit")
        return original_reserve(self, size)

    monkeypatch.setattr(qualification, "Recorder", recorder)
    monkeypatch.setattr(qualification._Budget, "reserve", reserve)
    result = qualification.qualify_runtimes([case(root)], tmp_path / "out")
    assert result["status"] == "failed" and result["env_steps_including_replay"] == 0
    assert result["failure"]["reason"] == "injected pre-start evidence limit"
    assert not (tmp_path / "out/trial/recording").exists()


def test_study_driver_rejects_overlarge_plan_before_admission(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts/studies/qualify_runtime.py"
    spec = importlib.util.spec_from_file_location("qualification_large_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    plan = tmp_path / "cases.json"
    plan.write_bytes(b" " * (1024 * 1024) + b"[]")
    monkeypatch.setattr(driver, "qualify_runtimes", lambda *_args, **_kwargs: pytest.fail("Overlarge plan admitted"))
    monkeypatch.setattr(sys, "argv", [str(path), "--cases", str(plan), "--out", str(tmp_path / "unused")])
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
    assert not (tmp_path / "unused").exists()
