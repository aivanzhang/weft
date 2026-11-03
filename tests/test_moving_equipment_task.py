"""Analytic task/scorer checks; source admission is fixture-stubbed, never a video qualification."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from test_moving_depth_family import settings, choice
from test_moving_equipment_proposal import packet
from weft.io import canonical_hash
from weft.moving_depth_family import prepare_moving_depth_family, materialize_moving_depth_selection
from weft.moving_equipment_world import construct_moving_equipment_world
from weft.moving_equipment_task import (build_moving_equipment_task, validate_moving_equipment_task,
    MovingEquipmentTaskEnv, PlacementMetrics, BODY_GATES)
from weft.shared_world import SharedWorldEnv
from weft.shared_world_observation import EquipmentState, SubstepSample


@pytest.fixture(scope="module")
def example():
    patch = pytest.MonkeyPatch()
    try:
        inputs, _ = packet(patch, raw_tracks=1, image_shift=(300, 190))
        family = prepare_moving_depth_family(inputs, settings())
        bundle = materialize_moving_depth_selection(family, choice(family, ["raw-0"]))
        scene = construct_moving_equipment_world(inputs, bundle, "candidate-1", depth_family=family)
        task = build_moving_equipment_task(scene, bundle, candidate_id="candidate-1", target_object_id="raw-0",
            target_position_world=list(scene["objects"][0]["position"]), position_tolerance_world=.05,
            speed_tolerance_world_s=.1, hold_duration_s=.04)
        yield scene, bundle, task
    finally:
        patch.undo()


def samples(scene, task):
    elapsed = 0.
    count = scene["action_contract"]["human_motor_count"]
    for step in range(task["interval"]["horizon_steps"]):
        dt = (task["interval"]["native_interval"]["terminal_duration_s"]
              if step == task["interval"]["horizon_steps"] - 1 else scene["dt"])/scene["substeps"]
        for substep in range(scene["substeps"]):
            end = elapsed + dt
            item = EquipmentState("raw-0", tuple(task["goal"]["position_world"]), (1., 0., 0., 0.),
                                  (0., 0., 0.), (0., 0., 0.), (0., 0., 1.))
            yield SubstepSample(step+1, substep+1, elapsed, end, elapsed, True, None,
                (0.,)*count, (0.,)*count, (0.,)*count, (0.,)*count, (item,), (), {}, 0.)
            elapsed = end


def test_contract_pins_complete_selected_world_and_original_interval(example):
    scene, bundle, task = example
    assert task["binding"]["family_sha256"] == canonical_hash(bundle["depth_family"])
    assert task["binding"]["selection_sha256"] == canonical_hash(bundle["depth_selection"])
    assert task["binding"]["scene_sha256"] == canonical_hash(scene)
    assert task["interval"]["native_interval"]["duration_s"] == .45
    assert task["body_gates"] == BODY_GATES
    assert task["action_contract"] == scene["action_contract"]
    assert task["reset"] == {"jitter": False}
    assert "changed-start" in task["scope"] and "virtual object forces" in task["scope"]
    assert validate_moving_equipment_task(task, scene, bundle) == task


@pytest.mark.parametrize("damage", ["family", "selection", "scene", "interval", "gate", "scorer", "action", "reset", "extra"])
def test_pins_gates_controls_and_interval_cannot_be_rewritten(example, damage):
    scene, bundle, original = example
    task = deepcopy(original)
    if damage in {"family", "selection", "scene"}: task["binding"][damage+"_sha256"] = "0"*64
    elif damage == "interval": task["interval"]["horizon_steps"] -= 1
    elif damage == "gate": task["body_gates"]["angular_rmse_rad"] = 100
    elif damage == "scorer": task["scorer_sha256"]["moving_equipment_task.py"] = "0"*64
    elif damage == "action": task["action_contract"]["physical_limits"][0] *= 2
    elif damage == "reset": task["reset"]["jitter"] = 0
    else: task["allow_grasp"] = True
    with pytest.raises(ValueError): validate_moving_equipment_task(task, scene, bundle)


@pytest.mark.parametrize("field,value", [("position_world", [float("nan"),0,0]),
    ("position_world", [-1,0,0]), ("position_world", [0,0]), ("position_tolerance_world", True),
    ("speed_tolerance_world_s", 0), ("terminal_consecutive_hold_s", 1), ("object_id", "absent")])
def test_invalid_goals_refuse(example, field, value):
    scene, bundle, original = example
    task = deepcopy(original);task["goal"][field] = value
    with pytest.raises(ValueError): validate_moving_equipment_task(task, scene, bundle)


def test_observed_full_native_interval_and_terminal_hold_required(example):
    scene, _, task = example
    scorer = PlacementMetrics(scene, task)
    rows = list(samples(scene, task))
    for sample in rows[:-1]: scorer.consume(sample)
    assert not scorer.metrics()["success"]
    scorer.consume(rows[-1])
    result = scorer.metrics()
    assert result["success"] and result["elapsed_s"] == pytest.approx(.45, abs=1e-12)
    assert result["mean_reward"] == pytest.approx(1.)
    assert result["valid_substeps"] == 14*16
    assert rows[-1].interval_end_s-rows[-1].interval_start_s == pytest.approx((.45-13/30)/16)


@pytest.mark.parametrize("damage", ["gap", "missing_clock", "nominal_terminal", "invalid", "floor", "angular_peak", "root_peak", "lost_object", "speed", "late_miss"])
def test_no_endpoint_recovery_masks_observation_or_body_failures(example, damage):
    scene, _, task = example
    scorer = PlacementMetrics(scene, task)
    rows = list(samples(scene, task))
    i = len(rows)-1 if damage in {"nominal_terminal", "speed", "late_miss"} else 2
    row = rows[i]
    if damage == "gap": rows.pop(i)
    elif damage == "missing_clock": rows[i] = replace(row, interval_end_s=None)
    elif damage == "nominal_terminal": rows[i] = replace(row, interval_end_s=row.interval_start_s+scene["dt"]/scene["substeps"])
    elif damage == "invalid": rows[i] = replace(row, valid=False, failure="engine_problem")
    elif damage == "floor": rows[i] = replace(row, max_human_floor_penetration_model_units=.026)
    elif damage in {"angular_peak", "root_peak"}:
        q = list(row.human_qpos)
        indices = scorer.angular if damage == "angular_peak" else scorer.roots
        for index in indices: q[index] = .71 if damage == "angular_peak" else .31*scorer.scale
        rows[i] = replace(row, human_qpos=tuple(q))
    elif damage == "lost_object": rows[i] = replace(row, equipment=())
    else:
        item = row.equipment[0]
        item = replace(item, velocity=(.11,0.,0.)) if damage == "speed" else replace(item, position=(99.,0.,0.))
        rows[i] = replace(row, equipment=(item,))
    for row in rows: scorer.consume(row)
    assert not scorer.metrics()["success"]
    if damage in {"speed", "late_miss"}: assert scorer.metrics()["terminal_consecutive_hold_s"] == 0


def test_wrapper_passes_actions_and_physics_exactly_and_keeps_diagnostics_separate(example):
    scene, bundle, task = example
    before = deepcopy((scene, bundle, task))
    wrapped, direct = MovingEquipmentTaskEnv(scene, task, bundle), SharedWorldEnv(scene)
    try:
        obs, info = wrapped.reset(seed=7)
        direct_obs, _ = direct.reset(seed=7, options={"jitter": False})
        assert np.array_equal(obs[:-8], direct_obs)
        assert "success" not in info and "task" in info
        for _ in range(3):
            action = direct.authored_action(); action[-1] = .2
            got = wrapped.step(action)
            expected = direct.step(action)
            assert np.array_equal(got[0][:-8], expected[0])
            assert got[2:4] == expected[2:4]
            assert wrapped.snapshot()["engine_diagnostics"] == direct.snapshot()
            assert got[4]["engine_diagnostics"] == expected[4]
            assert "success" not in got[4] and got[4]["task"]["scope"] == task["scope"]
        assert wrapped.metrics()["valid_substeps"] == 48
        assert tuple(wrapped.action_names) == tuple(scene["action_contract"]["names"])
        assert (scene, bundle, task) == before
        for options in ({"jitter": True}, {"jitter": 0}, {"jitter": False, "position_jitter": .1}):
            with pytest.raises(ValueError, match="exact original start"): wrapped.reset(options=options)
    finally:
        wrapped.close();direct.close()


def test_structural_joins_do_not_accept_changed_proposal_or_candidate(example):
    scene, bundle, task = example
    changed = deepcopy(bundle);changed["depth_family"]["scope"] += " changed"
    with pytest.raises(ValueError): validate_moving_equipment_task(task, scene, changed)
    changed = deepcopy(scene);changed["objects"][0]["provenance"]["candidate_id"] = "candidate-2"
    with pytest.raises(ValueError): validate_moving_equipment_task(task, changed, bundle)


def test_old_stationary_task_and_learning_guards_still_reject(example):
    from weft.shared_exercise_task import build_shared_exercise_task
    from weft.shared_world_learning import validate_shared_policy
    scene, _, _ = example
    with pytest.raises(ValueError, match="separately qualified"): build_shared_exercise_task(scene)
    with pytest.raises(ValueError, match="not qualified"): validate_shared_policy({}, scene)


def test_postintegration_boundary_floor_failure_cannot_hide_behind_last_substep(example, monkeypatch):
    scene, bundle, task = example
    wrapped = MovingEquipmentTaskEnv(scene, task, bundle)
    try:
        observation, _ = wrapped.reset(seed=7)
        def boundary_only(_action):
            for row in samples(scene, task): wrapped._observe(row)
            return observation[:-8], 999., False, True, {"success": True, "max_floor_penetration_model_units": .026}
        monkeypatch.setattr(wrapped._engine, "step", boundary_only)
        _, reward, _, _, info = wrapped.step(np.zeros(len(wrapped.action_names)))
        assert reward == pytest.approx(1.)
        assert info["engine_diagnostics"]["success"] is True
        assert info["task"]["success"] is False
        assert info["task"]["max_floor_penetration_model_units"] == .026
        assert "max_floor_penetration_model_units" in info["task"]["failed_criteria"]
    finally:
        wrapped.close()


def test_frozen_gates_and_detached_task_cannot_change_active_scoring(example):
    scene, bundle, task = example
    with pytest.raises(TypeError): BODY_GATES["max_floor_penetration_model_units"] = 10
    wrapped = MovingEquipmentTaskEnv(scene, task, bundle)
    try:
        wrapped.reset(seed=7)
        changed = wrapped.task
        changed["body_gates"]["max_floor_penetration_model_units"] = 10
        changed["goal"]["position_world"][0] += 1
        assert wrapped.task == task
        for row in samples(scene, task): wrapped._observe(row)
        wrapped._metrics.consume_boundary({"max_floor_penetration_model_units": .026})
        assert not wrapped.metrics()["success"]
        assert wrapped.metrics()["task_sha256"] == canonical_hash(task)
    finally:
        wrapped.close()
