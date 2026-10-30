"""The full-reference profile changes duration, not physics, clocks or legacy limits."""

from copy import deepcopy

import mujoco
import pytest

from weft.shared_world import (
    FULLSPAN_PROFILE, SharedWorldEnv, full_reference_horizon,
    validate_shared_snapshot, validate_shared_world,
)
from weft.shared_world_lab import authored_shared_world_scene


def fullspan_scene(*, steps=254, first=0.):
    scene = authored_shared_world_scene()
    body = scene["human_spec"]
    for key in ("reference", "root_reference"):
        rows = deepcopy(body[key])
        body[key] = []
        for index in range(steps + 1):
            row = deepcopy(rows[min(index, len(rows) - 1)])
            row["time_s"] = first + index / 30
            body[key].append(row)
    scene["profile"] = FULLSPAN_PROFILE
    body["mobility"]["floor_fit"]["reference_time_s"] = first
    scene["task"].update(start_offset_s=0., horizon_steps=steps)
    scene["coordinate_frame"]["clock_binding"]["source_time_offset_s"] = first
    return validate_shared_world(scene)


def test_fullspan_reaches_final_reference_with_one_continuous_model_clock(monkeypatch):
    scene = fullspan_scene(first=2 / 30)
    env = SharedWorldEnv(scene)
    resets, advances = [], []
    original_reset, original_step = mujoco.mj_resetData, mujoco.mj_step

    def reset(model, data):
        resets.append(id(data))
        return original_reset(model, data)

    def step(model, data):
        advances.append(id(data))
        return original_step(model, data)

    monkeypatch.setattr(mujoco, "mj_resetData", reset)
    monkeypatch.setattr(mujoco, "mj_step", step)
    try:
        env.reset(seed=817, options={"jitter": False, "jitter_scale": 0.})
        model_id, data_id = id(env.model), id(env.data)
        reset_count = len(resets)
        for index in range(254):
            _, _, terminated, truncated, _ = env.step(env.authored_action())
            assert id(env.model) == model_id and id(env.data) == data_id
            assert not terminated
            assert truncated is (index == 253)
            if index in (179, 180, 253):
                validate_shared_snapshot(env.snapshot(), scene)
        final = env.snapshot()
        assert len(resets) == reset_count == 1
        assert len(advances) == 254 * scene["substeps"]
        assert set(advances) == {data_id}
        assert final["steps"] == final["episode_steps"] == 254
        assert final["time_s"] == pytest.approx(254 / 30)
        assert env.data.time == pytest.approx(254 / 30)
        assert final["source_time_s"] == pytest.approx(scene["human_spec"]["reference"][-1]["time_s"])
        assert final["profile"] == FULLSPAN_PROFILE
        with pytest.raises(RuntimeError, match="Reset"):
            env.step(env.authored_action())
    finally:
        env.close()


def test_v1_and_v2_keep_identical_physics_and_actions_over_the_shared_prefix():
    scene = fullspan_scene()
    legacy = deepcopy(scene)
    legacy["profile"] = "shared-world-v1"
    legacy["task"]["horizon_steps"] = 180
    old, new = SharedWorldEnv(legacy), SharedWorldEnv(scene)
    try:
        old.reset(seed=54, options={"jitter": False, "jitter_scale": 0.})
        new.reset(seed=54, options={"jitter": False, "jitter_scale": 0.})
        for index in range(180):
            action = old.authored_action()
            assert action.tobytes() == new.authored_action().tobytes()
            old.step(action)
            new.step(action)
            assert old.data.qpos.tobytes() == new.data.qpos.tobytes()
            assert old.data.qvel.tobytes() == new.data.qvel.tobytes()
            assert old.data.time == new.data.time
            assert old.snapshot()["contact_details"] == new.snapshot()["contact_details"]
        assert old.truncated and not new.truncated
        new.step(new.authored_action())
        assert new.steps == 181
    finally:
        old.close()
        new.close()


def test_partial_failure_after_legacy_limit_uses_actual_substep_clock(monkeypatch):
    scene = fullspan_scene(first=2 / 30)
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=7, options={"jitter": False, "jitter_scale": 0.})
        for _ in range(180):
            env.step(env.authored_action())
        original = env._instability

        def failure():
            return "object_left_bounds" if env.data.time > 6.01 else original()

        monkeypatch.setattr(env, "_instability", failure)
        _, _, terminated, truncated, _ = env.step(env.authored_action())
        state = env.snapshot()
        assert terminated and not truncated
        assert state["steps"] == 181
        assert 0 < state["partial_action"]["completed_substeps"] < scene["substeps"]
        assert state["time_s"] == env.data.time < 181 / 30
        assert state["source_time_s"] == env.source_start + env.data.time
        validate_shared_snapshot(state, scene)
        wrong = deepcopy(state)
        wrong["source_time_s"] = scene["human_spec"]["reference"][0]["time_s"] + 181 / 30
        with pytest.raises(ValueError, match="source_time|clocks"):
            validate_shared_snapshot(wrong, scene)
    finally:
        env.close()


@pytest.mark.parametrize("change", ["legacy_181", "clipped", "offset", "fractional_end", "too_long", "boolean_horizon"])
def test_profile_interval_constraints_never_round_clip_or_retime(change):
    scene = fullspan_scene()
    if change == "legacy_181":
        scene["profile"] = "shared-world-v1"
        scene["task"]["horizon_steps"] = 181
    elif change == "clipped":
        scene["task"]["horizon_steps"] = 180
    elif change == "offset":
        scene["task"]["start_offset_s"] = 1 / 30
    elif change == "fractional_end":
        for key in ("reference", "root_reference"):
            scene["human_spec"][key][-1]["time_s"] += .001
    elif change == "too_long":
        scene["task"]["horizon_steps"] = 1801
    else:
        scene["task"]["horizon_steps"] = True
    with pytest.raises(ValueError):
        validate_shared_world(scene)


def test_fullspan_grid_limit_is_explicit_and_nonfinite_is_not_accepted():
    assert full_reference_horizon([{"time_s": .2}, {"time_s": 60.2}], 1 / 30) == 1800
    for end in (.2, 60.3, 8.001, float("nan"), float("inf")):
        with pytest.raises((ValueError, OverflowError)):
            full_reference_horizon([{"time_s": .2}, {"time_s": end}], 1 / 30)


@pytest.mark.parametrize("field", ["profile", "source_time_s", "time_s", "steps", "reference_marker"])
def test_fullspan_reference_snapshots_preserve_original_irregular_clock(field):
    scene = fullspan_scene(first=2 / 30)
    # Missing observations keep their identity; reference indices are not control steps.
    for key in ("reference", "root_reference"):
        scene["human_spec"][key] = scene["human_spec"][key][::2]
    env = SharedWorldEnv(scene)
    try:
        state = env.reference_snapshot(100)
        validate_shared_snapshot(state, scene)
        assert state["steps"] == 100 and state["source_time_s"] == pytest.approx(202 / 30)
        wrong = deepcopy(state)
        if field == "profile":
            wrong[field] = "shared-world-v1"
        elif field == "steps":
            wrong[field] = len(scene["human_spec"]["reference"])
        elif field == "reference_marker":
            wrong["human"]["reference"] = 1
        else:
            wrong[field] += .01
        with pytest.raises(ValueError):
            validate_shared_snapshot(wrong, scene)
    finally:
        env.close()
