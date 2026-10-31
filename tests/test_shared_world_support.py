"""Static supports share the existing world without introducing a movable-object goal."""

from copy import deepcopy

import numpy as np
import pytest

from weft.shared_world import (
    STATIC_SUPPORT_PROFILE, SharedWorldEnv, full_reference_horizon,
    validate_shared_snapshot, validate_shared_world,
)
from weft.shared_world_lab import authored_shared_world_scene


@pytest.fixture(scope="module")
def support_scene():
    scene = authored_shared_world_scene()
    scene.pop("action_contract")
    scene.update(profile=STATIC_SUPPORT_PROFILE, target=None)
    scene["objects"] = [obj for obj in scene["objects"] if obj["role"] == "static"]
    scene["task"].update(type="human_support_v1", object_goal=None,
                         horizon_steps=full_reference_horizon(scene["human_spec"]["reference"], scene["dt"]))
    return validate_shared_world(scene)


def test_static_support_world_preserves_joint_clock_and_has_no_object_goal(support_scene):
    scene = support_scene
    assert scene["profile"] == "shared-world-v4" and scene["target"] is None
    assert scene["task"]["horizon_steps"] == 60
    assert scene["action_contract"]["object_force_channels"] == 0
    assert scene["action_contract"]["dynamic_object_ids"] == []
    env = SharedWorldEnv(scene)
    try:
        initial, _ = env.reset(seed=31, options={"jitter": False, "jitter_scale": 0.})
        assert initial.shape == (155 + 13 * len(scene["objects"]),)
        assert env.action_space.shape == (22,)
        before = env.snapshot()
        states = []
        for _ in range(3):
            env.step(np.zeros(22))
            states.append(env.snapshot())
            validate_shared_snapshot(states[-1], scene)
        assert states[-1]["target"] is None
        assert states[-1]["coupled_metrics"]["object_goal_status"] == "not_applicable"
        assert states[-1]["objects"] == before["objects"]
        env.reset(seed=31, options={"jitter": False, "jitter_scale": 0.})
        for expected in states:
            env.step(np.zeros(22))
            assert env.snapshot() == expected
        reference = env.reference_snapshot(0)
        validate_shared_snapshot(reference, scene)
        assert reference["objects"] == before["objects"]
    finally:
        env.close()


@pytest.mark.parametrize("change", ["dynamic", "empty", "target", "goal", "shorten", "offset", "inventory"])
def test_static_support_contract_rejects_ambiguous_or_legacy_task_fields(support_scene, change):
    scene = deepcopy(support_scene)
    scene.pop("action_contract")
    if change == "dynamic":
        scene["objects"][0]["role"] = "passive"
    elif change == "empty":
        scene["objects"] = []
    elif change == "target":
        scene["target"] = [1., 1., 1.]
    elif change == "goal":
        scene["task"]["object_goal"] = {}
    elif change == "shorten":
        scene["task"]["horizon_steps"] -= 1
    elif change == "offset":
        scene["task"]["start_offset_s"] = scene["dt"]
    else:
        scene["inventory"] = {"support": "ground_only"}
    with pytest.raises(ValueError):
        validate_shared_world(scene)


def test_old_coupled_profiles_still_require_a_dynamic_object(support_scene):
    for profile in ("shared-world-v1", "shared-world-v2"):
        scene = deepcopy(support_scene)
        scene.update(profile=profile)
        with pytest.raises(ValueError, match="non-static"):
            validate_shared_world(scene)


@pytest.mark.parametrize("field,value", [("target", [1., 1., 1.]), ("virtual_force", [1., 0., 0.])])
def test_support_snapshot_cannot_smuggle_an_object_target_or_force(support_scene, field, value):
    env = SharedWorldEnv(support_scene)
    try:
        env.reset(seed=31, options={"jitter": False, "jitter_scale": 0.})
        state = env.snapshot()
        state[field] = value
        with pytest.raises(ValueError, match="object target or virtual object force"):
            validate_shared_snapshot(state, support_scene)
    finally:
        env.close()
