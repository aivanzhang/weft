"""Analytic equipment-proxy boundaries; these fixtures do not recover a real kettlebell."""

from copy import deepcopy

import numpy as np
import pytest

from weft.io import canonical_hash
from weft.shared_world import SharedWorldEnv, validate_shared_world
from weft.shared_world_lab import authored_shared_world_scene


def proxy_scene():
    scene = authored_shared_world_scene()
    scene.pop("action_contract")
    scene["objects"] = [
        {
            "id": name, "label": f"{name} bounding-box proxy", "role": "passive", "shape": "box",
            "position": [x, 0.25, 0.09], "size": [0.12, 0.12, 0.16],
            "quaternion": [1.0, 0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0], "mass": 2.0,
            "friction": [0.65, 0.02, 0.018], "color": "#E4C439",
            "provenance": {"scope": "authored_analytic_equipment_proxy", "unmodeled": ["handle_aperture"],
                           "depth_mass_material": "assumed"},
        }
        for name, x in (("working_bell", 0.25), ("idle_bell", 1.50))
    ]
    scene["task"]["object_goal"]["target_object_id"] = "working_bell"
    scene["task"]["object_goal"]["require_human_contact"] = False
    scene["target"] = [0.25, 0.25, 0.09]
    return validate_shared_world(scene)


def test_two_equipment_proxies_preserve_body_floor_and_shared_dynamics_contract():
    baseline = authored_shared_world_scene()
    candidate = proxy_scene()
    assert canonical_hash(candidate["human_spec"]) == canonical_hash(baseline["human_spec"])
    assert candidate["coordinate_frame"] == baseline["coordinate_frame"]
    assert candidate["task"]["human_limits"] == baseline["task"]["human_limits"]
    base_env, env = SharedWorldEnv(baseline), SharedWorldEnv(candidate)
    try:
        base_floor = base_env.model.geom("mobile_floor").id
        floor = env.model.geom("mobile_floor").id
        np.testing.assert_array_equal(env.model.geom_pos[floor], base_env.model.geom_pos[base_floor])
        np.testing.assert_array_equal(env.model.geom_quat[floor], base_env.model.geom_quat[base_floor])
        np.testing.assert_array_equal(env.model.opt.gravity, base_env.model.opt.gravity)
        env.reset(seed=0)
        assert env.model.nu == 22
        assert env.action_space.shape == (28,)
        assert env.observation_space.shape == (184,)
        assert {obj["id"] for obj in env.snapshot()["objects"]} == {"working_bell", "idle_bell"}
        assert all(obj["role"] == "passive" for obj in env.snapshot()["objects"])
    finally:
        base_env.close()
        env.close()


def test_reconstruction_baseline_does_not_apply_equipment_forces_or_follow_provenance():
    scene = proxy_scene()
    changed_annotations = deepcopy(scene)
    for obj in changed_annotations["objects"]:
        obj["provenance"]["synthetic_image_center_samples"] = [[0.0, 0.0], [10000.0, -10000.0]]
    original, annotated = SharedWorldEnv(scene), SharedWorldEnv(changed_annotations)
    try:
        original.reset(seed=7)
        annotated.reset(seed=7)
        for _ in range(8):
            for env in (original, annotated):
                action = env.authored_action()
                np.testing.assert_array_equal(action[22:], np.zeros(6))
                env.step(action)
                assert all(obj["virtual_force"] == [0.0, 0.0, 0.0] for obj in env.snapshot()["objects"])
            np.testing.assert_array_equal(original.data.qpos, annotated.data.qpos)
            np.testing.assert_array_equal(original.data.qvel, annotated.data.qvel)
    finally:
        original.close()
        annotated.close()


def test_equipment_interaction_uses_explicit_forces_not_a_hidden_position_script():
    env = SharedWorldEnv(proxy_scene())
    try:
        env.reset(seed=7)
        env.select_target("working_bell")
        action = env.manual_action_vector(object_id="working_bell", object_action_code=2)
        np.testing.assert_array_equal(action[:22], np.zeros(22))
        np.testing.assert_array_equal(action[25:], np.zeros(3))
        before = {obj["id"]: obj for obj in env.snapshot()["objects"]}
        for _ in range(4):
            env.step(action)
        after = {obj["id"]: obj for obj in env.snapshot()["objects"]}
        assert after["working_bell"]["virtual_force"][0] > 0.0
        assert after["working_bell"]["position"][0] > before["working_bell"]["position"][0]
        assert after["idle_bell"]["virtual_force"] == [0.0, 0.0, 0.0]
    finally:
        env.close()


@pytest.mark.parametrize("shape", ["capsule", "compound", "mesh"])
def test_unimplemented_equipment_geometry_rejects_instead_of_silent_box_conversion(shape):
    scene = proxy_scene()
    scene["objects"][0]["shape"] = shape
    with pytest.raises(ValueError, match="object fields differ"):
        validate_shared_world(scene)


def test_static_only_equipment_requires_an_explicit_future_task_contract():
    scene = proxy_scene()
    for obj in scene["objects"]:
        obj["role"] = "static"
    with pytest.raises(ValueError, match="at least one non-static"):
        validate_shared_world(scene)
