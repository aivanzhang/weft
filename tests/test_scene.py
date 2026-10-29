from copy import deepcopy
import json

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from weft.scene import SceneEnv, rollout, validate_scene
from weft.scene_learning import policy_action, train_scene


@pytest.fixture
def scene():
    return {"schema_version": 1, "profile": "planar-rigid-v1", "width": 1.0, "height": 0.6,
            "dt": 1 / 30, "gravity": [0, 0], "damping": 0.9, "restitution": 0.65,
            "objects": [
                {"id": "actor", "label": "Controlled disc", "role": "active", "shape": "circle",
                 "position": [0.2, 0.3], "velocity": [0, 0], "radius": 0.035, "color": "#E0B620", "mass": 1,
                 "provenance": {"geometry": "synthetic", "mass": "assumed"}},
                {"id": "obstacle", "label": "Fixed obstacle", "role": "static", "shape": "box",
                 "position": [0.5, 0.45], "velocity": [0, 0], "size": [0.15, 0.08], "color": "#567799", "mass": 1,
                 "provenance": {"geometry": "synthetic"}}],
            "controlled_id": "actor", "target": [0.8, 0.3],
            "task": {"position_tolerance": 0.06, "speed_tolerance": 0.15, "max_steps": 150, "acceleration": 2},
            "relationships": [{"type": "shared_plane", "objects": ["actor", "obstacle"], "status": "assumed"}],
            "assumptions": ["Synthetic software fixture, not a measured physical scene", "Authored walls and actuator"]}


def test_scene_contract_is_detached_and_rejects_invalid_values(scene):
    validated = validate_scene(scene)
    validated["objects"][0]["position"][0] = 0.4
    assert scene["objects"][0]["position"][0] == 0.2
    for mutate in (
        lambda s: s.update(damping=float("nan")),
        lambda s: s.update(typo=1),
        lambda s: s["objects"][0].update(position=[0, 0]),
        lambda s: s["objects"][1].update(id="actor"),
        lambda s: s["relationships"][0].update(objects=["missing"]),
        lambda s: s["task"].update(max_steps=True),
        lambda s: s["objects"][1].update(position=[0.2, 0.3]),
        lambda s: s.update(target=[0.5, 0.45]),
    ):
        candidate = deepcopy(scene)
        mutate(candidate)
        with pytest.raises(ValueError):
            validate_scene(candidate)


def test_gym_contract_reset_and_render_do_not_change_physics(scene):
    env = SceneEnv(scene)
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=91)
    first_state = env.snapshot()
    rendered = env.render()
    assert rendered.shape == (384, 640, 3)
    assert env.snapshot() == first_state
    env.step(2)
    second, _ = env.reset(seed=91)
    assert np.array_equal(first, second)
    assert env.snapshot() == first_state
    with pytest.raises(ValueError):
        env.step(True)
    with pytest.raises(ValueError):
        env.step(6)
    env.close()


def test_dynamic_contact_transfers_momentum_and_wall_contains_bodies(scene):
    scene["objects"][1] = {
        "id": "passive", "label": "Passive disc", "role": "passive", "shape": "circle",
        "position": [0.43, 0.3], "velocity": [0, 0], "radius": 0.035, "color": "#3377AA", "mass": 1,
        "provenance": {"geometry": "synthetic"}}
    scene["relationships"] = []
    scene["objects"][0]["velocity"] = [1.2, 0]
    env = SceneEnv(scene)
    env.reset(seed=0, options={"jitter": False})
    contacts = []
    maximum_passive_speed = 0
    for _ in range(120):
        state = env.advance_physics()
        contacts.extend(state["contacts"])
        maximum_passive_speed = max(maximum_passive_speed, abs(env.bodies["passive"].velocity.x))
        assert all(0.02 < obj["position"][0] < 0.98 for obj in state["objects"])
    assert ["actor", "passive"] in contacts
    assert maximum_passive_speed > 0.5
    env.close()


def test_static_obstacle_is_not_crossed_at_high_speed(scene):
    scene["objects"][1]["position"] = [0.5, 0.3]
    scene["objects"][0]["velocity"] = [4, 0]
    env = SceneEnv(scene)
    env.reset(seed=0, options={"jitter": False})
    for _ in range(30):
        env.advance_physics()
        assert env.bodies["actor"].position.x < 0.45
        assert list(env.bodies["obstacle"].position) == [0.5, 0.3]
    env.close()


def test_replay_is_deterministic_and_actions_change_outcomes(scene):
    actions = [2] * 12 + [5] * 12 + [4] * 6
    assert rollout(scene, actions, seed=3) == rollout(scene, actions, seed=3)
    coast = rollout(scene, [0] * 30)
    controlled = rollout(scene, actions)
    assert controlled[-1]["objects"][0]["position"] != coast[-1]["objects"][0]["position"]


def test_success_requires_low_speed_and_physics_replay_ignores_authored_goal(scene):
    scene["objects"][0]["position"] = scene["target"][:]
    scene["objects"][0]["velocity"] = [0.3, 0]
    env = SceneEnv(scene)
    env.reset(seed=0, options={"jitter": False})
    assert not env.step(0)[2]
    scene["objects"][0]["velocity"] = [0, 0]
    env = SceneEnv(scene)
    env.reset(seed=0, options={"jitter": False})
    assert env.step(0)[2]
    with pytest.raises(RuntimeError):
        env.step(0)
    assert not env.advance_physics()["terminated"]
    assert env.steps == 2


def test_training_preserves_actual_updates_matched_trials_and_bounds(scene, tmp_path):
    scene["task"]["max_steps"] = 30
    progress = []
    summary = train_scene(scene, tmp_path, episodes=8, eval_episodes=3, seed=42,
                          progress=lambda stage, detail: progress.append((stage, detail)))
    policy = json.loads((tmp_path / "policy.json").read_text())
    evaluation = json.loads((tmp_path / "rl-evaluation.json").read_text())
    assert summary["training"]["completed_episodes"] == 8
    assert any(any(value != 0 for value in row) for row in policy["q_values"].values())
    for result in evaluation["policies"].values():
        assert [trial["seed"] for trial in result["trials"]] == evaluation["matched_episode_seeds"]
        assert all(trial["steps"] <= 30 for trial in result["trials"])
    assert 0 <= policy_action(policy, [0.1, 0, 0, 0]) < 6
    assert progress[-1][0] == "evaluation"
    with pytest.raises(FileExistsError):
        train_scene(scene, tmp_path, episodes=8, eval_episodes=3)
    with pytest.raises(ValueError):
        train_scene(scene, tmp_path / "invalid", episodes=5001)
