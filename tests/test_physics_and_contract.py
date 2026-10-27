from dataclasses import replace
import json
from pathlib import Path
import subprocess

from gymnasium.utils.env_checker import check_env
import numpy as np
import pytest

from weft.env import RollingEnv, Task, task_success
from weft.evaluate import physics_checks
from weft.world import World, advance, passive_position


def test_numerical_engine_matches_passive_closed_form(world):
    x, velocity = world.x0, world.v0
    for step in range(1, 151):
        x, velocity = advance(world, x, velocity, 0, 0.05)
        expected = float(passive_position(world, [step * 0.05])[0])
        assert x == pytest.approx(expected, abs=1e-9)
    assert velocity == pytest.approx(0, abs=1e-9)


def test_all_physics_invariants(world):
    assert physics_checks(world)["passes"]


@pytest.mark.parametrize("patch", [{"drag": -1}, {"drag": float("nan")}, {"drag": True},
                                    {"v0": 99}, {"v0": "1"}, {"reward": 100}, {"width_px": 9}, {}])
def test_untrusted_patch_cannot_escape_schema(world, patch):
    with pytest.raises((ValueError, TypeError)):
        world.patch(patch)


def test_world_rejects_extra_fields(world):
    with pytest.raises(ValueError):
        World.from_dict(world.to_dict() | {"code": "do not execute"})


def test_gymnasium_contract_and_reproducibility(world):
    env = RollingEnv(world)
    check_env(env, skip_render_check=True)
    first, _ = env.reset(seed=91)
    second, _ = env.reset(seed=91)
    np.testing.assert_array_equal(first, second)
    with pytest.raises(ValueError):
        env.step(9)


def test_terminal_and_truncated_are_distinct(world):
    env = RollingEnv(world, task=Task(max_steps=1))
    env.reset(seed=0, options={"x": 0.3, "velocity": 0, "target": 0.8})
    _, _, terminated, truncated, _ = env.step(1)
    assert not terminated and truncated
    with pytest.raises(RuntimeError):
        env.step(1)
    env.reset(seed=0, options={"x": 0.75, "velocity": 0, "target": 0.75})
    _, _, terminated, truncated, info = env.step(1)
    assert terminated and not truncated and info["is_success"]


def test_reward_is_not_success_and_velocity_matters():
    assert not task_success(0.75, 1, 0.75, Task())
    assert task_success(0.75, 0, 0.75, Task())


def test_visual_engine_matches_python_for_braking_crossings(world):
    engine = Path(__file__).parents[1] / "src/weft/assets/engine.js"
    scenarios = []
    for velocity in (-0.7, -0.001, 0.0, 0.001, 0.7):
        for action in (-1, 0, 1):
            scenarios.append({"world": world.to_dict(), "x": 0.5, "v": velocity, "action": action, "dt": 0.05})
    js = "require(process.argv[1]);let text='';process.stdin.on('data',c=>text+=c);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(text).map(s=>WeftPhysics.advance(s.world,s.x,s.v,s.action,s.dt)))));"
    result = subprocess.run(["node", "-e", js, str(engine)], input=json.dumps(scenarios),
                            capture_output=True, text=True, check=True, timeout=10)
    expected = [advance(world, s["x"], s["v"], s["action"], s["dt"]) for s in scenarios]
    np.testing.assert_allclose(json.loads(result.stdout), expected, atol=1e-12)
