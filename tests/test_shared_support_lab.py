"""Support lab contracts remain distinct from legacy equipment goals and learned policies."""

from copy import deepcopy
import json

import pytest

from weft.capabilities import describe_capabilities
from weft.io import canonical_hash
from weft.recording_receipt import _evaluator_identity
from weft.shared_support_fixture import static_support_fixture, write_static_support_fixture
from weft.shared_support_task import evaluator_identity


def test_fixture_pair_changes_only_support_height_and_preserves_task_criteria(tmp_path):
    first, task = static_support_fixture()
    lowered, other = static_support_fixture(lowered=True)
    changed = deepcopy(first)
    changed["objects"][0]["position"][2] -= .6
    assert changed == lowered
    assert first["human_spec"] == lowered["human_spec"]
    for key in task.keys() - {"scene_sha256", "source_scope"}:
        assert task[key] == other[key]
    assert task["source_scope"]["supports"] != other["source_scope"]["supports"]
    result = write_static_support_fixture(tmp_path / "fixture")
    assert result["scene_sha256"] == canonical_hash(first)
    assert json.loads((tmp_path / "fixture/exercise-task.json").read_text()) == task
    assert result["simulation_ready"] is False
    with pytest.raises(ValueError, match="new fixture"):
        write_static_support_fixture(tmp_path / "fixture")


def test_support_capabilities_describe_real_controls_loads_and_no_equipment_reward():
    scene, task = static_support_fixture()
    result = describe_capabilities(scene, lab_profile="shared_exercise", exercise_task=task)
    assert result["entities"]["static_count"] == 1
    assert result["entities"]["mixed_rigid_articulated"] is True
    assert result["entities"]["multi_target_selection_endpoint"] is None
    assert result["action_contract"]["dimension"] == 19
    assert result["action_contract"]["root_assistance_channels"] == []
    assert result["action_contract"]["object_force_channels"] == []
    assert result["observation_contract"]["dimension"] == 162
    assert result["task_contract"]["success_criteria"]["support_requirements"] == task["support_requirements"]
    assert result["task_contract"]["success_criteria"]["equipment_stability"] == "not_applicable"
    assert result["controllers"]["learned_policy_available"] is False
    with pytest.raises(ValueError, match="learned-policy"):
        describe_capabilities(scene, exercise_task=task, learned_policy_available=True)


def test_support_learning_rejects_before_native_scene_admission(monkeypatch):
    from weft import shared_exercise_learning as learner
    from weft.shared_exercise_policies import validate_policy
    from weft.shared_exercise_residual_learning import validate_shared_exercise_residual_policy

    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported support policies must stop before native scene admission")

    monkeypatch.setattr(learner, "validate_shared_world", forbidden)
    for validate in (learner.validate_shared_exercise_policy, validate_policy, validate_shared_exercise_residual_policy):
        with pytest.raises(ValueError, match="separately qualified"):
            validate({}, {"profile": "shared-world-v4"}, {})


def test_support_receipt_pins_both_evaluators_without_reinterpreting_legacy_identity():
    hashes = {"shared_exercise_task.py": "a" * 64, "shared_support_task.py": "b" * 64}
    bound = {"shared": True, "engine_profile": "shared-world-v4", "native_sha256": hashes}
    assert _evaluator_identity(bound) == evaluator_identity(hashes)
    assert _evaluator_identity(dict(bound, engine_profile="shared-world-v3")) == "a" * 64
    for name in hashes:
        with pytest.raises(ValueError):
            _evaluator_identity(dict(bound, native_sha256={key: value for key, value in hashes.items() if key != name}))


def test_console_exposes_support_load_units_and_preserves_unobserved_nulls():
    from weft.lab_console import _brief

    row = {"object_id": "bench", "human_segment": "torso", "direction_world": [0., 0., 1.],
           "minimum_load_body_weight_fraction": .2, "load_body_weight_fraction": None,
           "qualifying_fraction": None, "max_gap_s": 0., "satisfied": False}
    assert _brief({"task_metrics": {"profile": "shared-support-task-v1", "support_requirements": [row]}})[
        "simulated_support_loads"] == [row]


def test_support_inspection_preserves_task_clock_and_links_authored_artifacts(tmp_path):
    from test_lab_inspection import seal, write
    from weft.lab_inspection import load_inspection

    source = tmp_path / "source"
    write_static_support_fixture(source)
    write(source, "support-config.json", {"scope": "authored; not observed"})
    result = load_inspection(source, profile="shared-world")
    assert result["timeline"]["task_steps"] == 60
    assert result["timeline"]["task_duration_s"] == 2.
    assert result["timeline"]["media_sync_verified"] is False
    assert not any(row["name"] == "support-config.json" for row in result["artifact_links"])
    seal(source)
    result = load_inspection(source, profile="shared-world")
    assert result["status"] == "available"
    links = {row["name"]: row["label"] for row in result["artifact_links"]}
    assert "Authored support" in links["support-config.json"]
    assert "not measured forces" in links["exercise-task.json"]


def test_support_avatar_option_stops_before_creating_a_recording(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from weft.lab import Lab
    from weft import shared_exercise_lab

    closed = []
    session = SimpleNamespace(scene_profile="shared-world-v4", env=SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(shared_exercise_lab, "SharedExerciseSession", lambda *args, **kwargs: session)
    lab = Lab.__new__(Lab)
    lab.session, lab.root = None, tmp_path
    lab.config = {"profile": "shared-world", "control_profile": "shared_exercise"}
    lab.avatars = SimpleNamespace(packages={"test-avatar": object()})
    lab.status = lambda: {"status": "ready"}
    with pytest.raises(ValueError, match="without avatar options"):
        lab.load_session()
    assert closed == [True] and lab.session is None
    assert not list(tmp_path.iterdir())
