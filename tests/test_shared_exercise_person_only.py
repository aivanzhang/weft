"""Person-only tasks score human motion without vacuous equipment passes or reward."""

from copy import deepcopy
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import pytest

from weft.io import canonical_hash
from weft.shared_exercise_task import (
    GATES, GATES_SHA256, HUMAN_ONLY_GATE_APPLICABILITY, HUMAN_ONLY_REWARD_PROFILE,
    HUMAN_ONLY_TASK_PROFILE, PROFILE, SharedExerciseTaskEnv, _ExerciseMetrics,
    build_shared_exercise_task, validate_shared_exercise_snapshot, validate_shared_exercise_task,
)
from weft.shared_world_lab import authored_shared_world_scene
from weft.shared_world_observation import EquipmentState, SubstepSample
from test_shared_world_human_only import person_scene


def person_accumulator(*, root=False):
    scene = person_scene(steps=2)
    task = build_shared_exercise_task(scene, root_assistance=root)
    return scene, task, _ExerciseMetrics(scene, task, (), 0., 1.)


def motion_sample(acc, index, *, angle=0., root_error=0., joint_action=0., root_action=0., penetration=0.):
    tick = acc.dt / acc.substeps
    return SubstepSample(
        index // acc.substeps + 1, index % acc.substeps + 1, index * tick, (index + 1) * tick, index * tick,
        True, None, (angle,) * 19 + (root_error,) * 3, (0.,) * 22, (0.,) * 22,
        (joint_action,) * 19 + (root_action,) * 3, (), (),
        MappingProxyType({"left": (0., 0., 0.), "right": (1., 0., 0.)}), penetration,
    )


def test_person_task_version_empty_equipment_and_full_interval_are_explicit():
    scene, task, _ = person_accumulator()
    legacy = build_shared_exercise_task(authored_shared_world_scene())
    assert set(task) == set(legacy)
    assert task["profile"] == HUMAN_ONLY_TASK_PROFILE and legacy["profile"] == PROFILE
    assert task["equipment_ids"] == task["contact_requirements"] == []
    assert task["source_scope"]["equipment_initialization"] == []
    assert task["interval"] == {"start_offset_s": 0., "horizon_steps": 2}
    assert task["gates_sha256"] == legacy["gates_sha256"] == GATES_SHA256
    assert validate_shared_exercise_task(task, scene) == task
    for bad_task, source in (({**task, "profile": PROFILE}, scene),
                             ({**legacy, "profile": HUMAN_ONLY_TASK_PROFILE}, authored_shared_world_scene())):
        with pytest.raises(ValueError, match="frozen gate"):
            validate_shared_exercise_task(bad_task, source)
    with pytest.raises(ValueError, match="equipment"):
        build_shared_exercise_task(scene, equipment_ids=["invented"])
    with pytest.raises(ValueError, match="contact"):
        build_shared_exercise_task(scene, contact_requirements=[{}])


@pytest.mark.parametrize("angle,root_error", [(0., 0.), (.4, 0.), (0., .2)])
def test_person_reward_has_no_absent_equipment_bonus_and_unchanged_human_gates(angle, root_error):
    _, _, acc = person_accumulator()
    with np.errstate(invalid="raise", divide="raise"):
        for index in range(acc.horizon * acc.substeps):
            acc.consume(motion_sample(acc, index, angle=angle, root_error=root_error))
    metrics = acc.metrics()
    expected = np.exp(-angle ** 2 / GATES["angular_rmse_rad"] ** 2
                      - root_error ** 2 / GATES["root_rmse_model_units"] ** 2)
    assert metrics["mean_reward"] == pytest.approx(expected)
    assert metrics["reward_integral"] == pytest.approx(expected * acc.duration)
    assert metrics["equipment"] == metrics["contact_requirements"] == []
    assert metrics["gate_applicability"] == dict(HUMAN_ONLY_GATE_APPLICABILITY)
    assert metrics["reward_profile"] == HUMAN_ONLY_REWARD_PROFILE
    assert metrics["completed_interval"] is True
    assert metrics["success"] is (angle == root_error == 0.)
    assert not any("equipment" in gate or "contact" in gate for gate in metrics["failed_gates"])


@pytest.mark.parametrize("root", [False, True])
def test_person_effort_penalties_and_root_assistance_remain_explicit(root):
    _, _, acc = person_accumulator(root=root)
    for index in range(acc.horizon * acc.substeps):
        acc.consume(motion_sample(acc, index, joint_action=.5, root_action=.5 if root else 0.))
    metrics = acc.metrics()
    assert metrics["joint_effort_mean_squared"] == pytest.approx(.25)
    assert metrics["root_assistance_mean_squared"] == pytest.approx(.25 if root else 0.)
    assert metrics["mean_reward"] == pytest.approx(1. - .01 * .25 - (.02 * .25 if root else 0.))


@pytest.mark.parametrize("field,value,gate", [
    ("angle", .8, "peak_angular_rmse_rad"),
    ("root_error", .4, "peak_root_rmse_model_units"),
    ("penetration", .026, "max_floor_penetration_model_units"),
])
def test_person_transient_human_failure_cannot_disappear_by_final_pose(field, value, gate):
    _, _, acc = person_accumulator()
    for index in range(acc.horizon * acc.substeps):
        acc.consume(motion_sample(acc, index, **({field: value} if index == 1 else {})))
    metrics = acc.metrics()
    assert metrics["completed_interval"] and not metrics["success"]
    assert gate in metrics["failed_gates"]


def test_person_accumulator_rejects_fabricated_equipment_and_disabled_root_force():
    scene, task, acc = person_accumulator()
    equipment = EquipmentState("invented", (0.,) * 3, (1., 0., 0., 0.), (0.,) * 3, (0.,) * 3, (0.,) * 3)
    with pytest.raises(ValueError, match="equipment"):
        _ExerciseMetrics(scene, task, (equipment,), 0., 1.)
    with pytest.raises(ValueError, match="equipment"):
        acc.consume(replace(motion_sample(acc, 0), equipment=(equipment,)))
    with pytest.raises(ValueError, match="contacts"):
        acc.consume(replace(motion_sample(acc, 0), contacts=(object(),)))
    acc.consume(motion_sample(acc, 0, root_action=.5))
    assert acc.metrics()["failure"] == "forbidden_human_actuation"
    assert not acc.metrics()["success"]


@pytest.mark.parametrize("root", [False, True])
def test_person_task_real_engine_has_156_observations_and_exact_seeded_replay(root):
    scene = person_scene(first=2 / 30)
    task = build_shared_exercise_task(scene, root_assistance=root)
    env = SharedExerciseTaskEnv(scene, task)
    try:
        observation, _ = env.reset(seed=81)
        assert observation.shape == env.observation_space.shape == (156,)
        assert env.action_space.shape == ((22,) if root else (19,))
        assert np.array_equal(observation[:155], env._engine.observation())
        assert env.action_contract()["object_force_channels"] == []
        assert env.observation_contract()["size"] == 156
        actions, expected = [], []
        for _ in range(scene["task"]["horizon_steps"]):
            action = env._engine.authored_human_action()[:env.action_count]
            observation, reward, _, _, metrics = env.step(action)
            assert np.isfinite(observation).all() and np.isfinite(reward)
            state = env.snapshot()
            validate_shared_exercise_snapshot(state, scene, task)
            assert state["engine_state"]["objects"] == []
            assert metrics["equipment"] == metrics["contact_requirements"] == []
            assert metrics["gate_applicability"]["equipment_stability"] == "not_applicable"
            actions.append(action)
            expected.append((reward, canonical_hash(state)))
        assert metrics["completed_interval"]
        assert env._engine.data.time == pytest.approx(3 / 30)
        env.reset(seed=81)
        for action, (reward, state_hash) in zip(actions, expected):
            _, actual_reward, _, _, _ = env.step(action)
            assert actual_reward == reward and canonical_hash(env.snapshot()) == state_hash
    finally:
        env.close()


@pytest.mark.parametrize("change", [
    lambda m: m.update(profile=PROFILE),
    lambda m: m.pop("gate_applicability"),
    lambda m: m["gate_applicability"].update(equipment_stability="passed"),
    lambda m: m.update(reward_profile="coupled"),
    lambda m: m.update(equipment=[{"object_id": "invented", "within_limits": True}]),
    lambda m: m.update(contact_requirements=[{"satisfied": True}]),
    lambda m: m.update(success=True),
])
def test_person_snapshot_rejects_invented_checks_and_success(change):
    scene = person_scene()
    task = build_shared_exercise_task(scene)
    env = SharedExerciseTaskEnv(scene, task)
    try:
        env.reset(seed=1)
        state = deepcopy(env.snapshot())
        change(state["task_metrics"])
        with pytest.raises(ValueError):
            validate_shared_exercise_snapshot(state, scene, task)
    finally:
        env.close()
