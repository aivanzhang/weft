"""Analytic inclined-plane checks; no video labels or fitted physical parameters."""

from copy import deepcopy
import math

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from weft.scene import SCENE_UNITS, SceneEnv, _extent, _overlap, validate_scene


def ramp_scene(degrees=20, pair_friction=0.25):
    angle = math.radians(degrees)
    tangent, normal = np.array([math.cos(angle), math.sin(angle)]), np.array([math.sin(angle), -math.cos(angle)])
    center = np.array([0.5, 0.4])
    common = {"shape": "box", "velocity": [0, 0], "angle": angle, "angular_velocity": 0,
              "friction": math.sqrt(pair_friction), "mass": 1, "provenance": {"all": "analytic fixture"}}
    return {"schema_version": 2, "profile": "planar-rigid-v2", "units": deepcopy(SCENE_UNITS),
            "boundary_friction": 0, "width": 1, "height": 0.8, "dt": 1 / 240,
            "gravity": [0, 9.81], "damping": 1, "restitution": 0,
            "objects": [common | {"id": "block", "label": "Test block", "role": "active",
                "size": [0.06, 0.03], "position": (center - 0.18 * tangent + 0.02501 * normal).tolist(), "color": "#AB7E48"},
                common | {"id": "ramp", "label": "Fixed support", "role": "static",
                "size": [0.6, 0.02], "position": center.tolist(), "color": "#555555"}],
            "controlled_id": "block", "target": [0.85, 0.12],
            "task": {"position_tolerance": 0.02, "speed_tolerance": 0.05, "max_steps": 900, "acceleration": 2},
            "relationships": [{"type": "supported_by", "objects": ["block", "ramp"], "status": "assumed"}],
            "assumptions": ["Analytic software fixture, not a video reconstruction",
                            "Normalized lengths and relative masses; one friction coefficient per shape"]}


@pytest.mark.parametrize("mutate", [
    lambda s: s.pop("units"), lambda s: s["units"].update(length="meter"),
    lambda s: s.update(schema_version=True), lambda s: s.update(profile="planar-rigid-v1"),
    lambda s: s.update(boundary_friction=-0.1), lambda s: s["objects"][0].pop("angle"),
    lambda s: s["objects"][0].update(angle=4), lambda s: s["objects"][0].update(angle=float("nan")),
    lambda s: s["objects"][0].update(angular_velocity=21), lambda s: s["objects"][0].update(friction=True),
    lambda s: s["objects"][1].update(angular_velocity=1), lambda s: s["objects"][0].update(size=[0.9, 0.02]),
])
def test_oriented_schema_rejects_implicit_units_and_unsafe_state(mutate):
    scene = ramp_scene()
    mutate(scene)
    with pytest.raises(ValueError):
        validate_scene(scene)


def test_rotated_geometry_rejects_corner_escape_and_actual_intersection():
    scene = ramp_scene()
    scene["objects"][0].update(position=[0.03, 0.2], angle=math.pi / 4)
    with pytest.raises(ValueError, match="boundary"):
        validate_scene(scene)
    scene = ramp_scene()
    scene["objects"][0]["position"] = scene["objects"][1]["position"][:]
    with pytest.raises(ValueError, match="overlap"):
        validate_scene(scene)
    scene = ramp_scene()
    scene["target"] = scene["objects"][1]["position"][:]
    with pytest.raises(ValueError, match="Target overlaps"):
        validate_scene(scene)


def test_separating_axes_do_not_confuse_bounding_boxes_with_contact():
    box = {"shape": "box", "size": [0.4, 0.02], "position": [0.5, 0.4], "angle": math.pi / 4}
    other = box | {"position": [0.5 + 0.03 / math.sqrt(2), 0.4 - 0.03 / math.sqrt(2)]}
    assert all(abs(a-b) < 2 * extent for a, b, extent in zip(box["position"], other["position"], _extent(box)))
    assert not _overlap(box, other)
    assert _overlap(box, other, padding=0.02)
    touching = box | {"position": [0.5 + 0.02 / math.sqrt(2), 0.4 - 0.02 / math.sqrt(2)]}
    assert not _overlap(box, touching)
    circle = {"shape": "circle", "radius": 0.01, "position": other["position"]}
    assert not _overlap(circle, box)
    circle["position"] = box["position"]
    assert _overlap(circle, box) and _overlap(box, circle)


@pytest.mark.parametrize("degrees,friction,steps", [(10, 0.25, 120), (20, 0, 48), (20, 0.25, 72)])
def test_inclined_plane_matches_static_and_sliding_solutions(degrees, friction, steps):
    scene = ramp_scene(degrees, friction)
    env = SceneEnv(scene)
    env.reset(options={"jitter": False})
    start = np.array(env.bodies["block"].position)
    angle = math.radians(degrees)
    tangent, normal = np.array([math.cos(angle), math.sin(angle)]), np.array([math.sin(angle), -math.cos(angle)])
    acceleration = max(0, 9.81 * (math.sin(angle) - friction * math.cos(angle)))
    contacts = set()
    for _ in range(steps):
        state = env.advance_physics()
        contacts.update(tuple(pair) for pair in state["contacts"])
        assert not any(state[name] for name in ("speed_limited", "angular_speed_limited", "substep_limited"))
        assert state["objects"][1]["position"] == scene["objects"][1]["position"]
        assert state["objects"][1]["angle"] == angle
    duration = steps * scene["dt"]
    displacement = np.array(env.bodies["block"].position) - start
    assert float(displacement @ tangent) == pytest.approx(0.5 * acceleration * duration**2, abs=0.002)
    assert float(np.array(env.bodies["block"].velocity) @ tangent) == pytest.approx(acceleration * duration, abs=0.02)
    assert abs(float(displacement @ normal)) < 0.0006
    assert env.bodies["block"].angle == pytest.approx(angle, abs=0.003)
    assert ("block", "ramp") in contacts
    env.close()


def test_rotated_equal_mass_collision_conserves_linear_momentum():
    scene = ramp_scene(pair_friction=0)
    scene.update(gravity=[0, 0], restitution=1)
    angle = scene["objects"][0]["angle"]
    tangent = np.array([math.cos(angle), math.sin(angle)])
    actor = scene["objects"][0]
    actor["velocity"] = tangent.tolist()
    passive = deepcopy(actor) | {"id": "passive", "label": "Passive block", "role": "passive",
        "velocity": [0, 0], "position": (np.array(actor["position"]) + 0.16 * tangent).tolist()}
    scene["objects"].append(passive)
    env = SceneEnv(scene)
    env.reset(options={"jitter": False})
    contacts = set()
    for _ in range(48):
        contacts.update(tuple(pair) for pair in env.advance_physics()["contacts"])
    v1, v2 = np.array(env.bodies["block"].velocity), np.array(env.bodies["passive"].velocity)
    assert np.allclose(v1 + v2, tangent, atol=1e-6)
    assert np.linalg.norm(v1) < 0.01
    assert np.allclose(v2, tangent, atol=0.01)
    assert float(v1 @ v1 + v2 @ v2) == pytest.approx(1, abs=0.002)
    assert ("block", "passive") in contacts
    env.close()


def test_initial_rotation_and_spin_survive_reset_and_snapshot():
    scene = ramp_scene()
    scene["gravity"] = [0, 0]
    scene["objects"][0].update(position=[0.25, 0.15], angle=0.25, angular_velocity=2)
    env = SceneEnv(scene)
    check_env(env, skip_render_check=True)
    env.reset(seed=21, options={"jitter": False})
    initial = env.snapshot()
    for _ in range(60):
        state = env.advance_physics()
    assert state["objects"][0]["angle"] == pytest.approx(0.75, abs=1e-12)
    assert state["objects"][0]["angular_velocity"] == 2
    before = env.snapshot()
    assert env.render().shape == (512, 640, 3)
    assert env.snapshot() == before
    env.reset(seed=21, options={"jitter": False})
    assert env.snapshot() == initial
    env.close()


def test_rotated_jitter_stays_inside_and_outside_supports():
    env = SceneEnv(ramp_scene())
    for seed in range(20):
        env.reset(seed=seed)
        objects = env.snapshot()["objects"]
        assert not _overlap(*objects)
        for obj in objects:
            assert all(extent + 0.002 <= coordinate <= limit - extent - 0.002
                       for extent, coordinate, limit in zip(_extent(obj), obj["position"], [1, 0.8]))
    env.close()


def test_fast_disc_cannot_cross_tilted_thin_support():
    scene = ramp_scene()
    angle = scene["objects"][1]["angle"]
    normal = np.array([math.sin(angle), -math.cos(angle)])
    actor = scene["objects"][0]
    actor.pop("size")
    actor.update(shape="circle", radius=0.008, position=(np.array([0.5, 0.4]) + 0.12 * normal).tolist(),
                 velocity=(-4 * normal).tolist())
    scene.update(gravity=[0, 0], dt=1 / 20, restitution=0.5)
    scene["objects"][1]["size"][1] = 0.016
    env = SceneEnv(scene)
    env.reset(options={"jitter": False})
    for _ in range(5):
        env.advance_physics()
        offset = np.array(env.bodies["block"].position) - np.array([0.5, 0.4])
        assert float(offset @ normal) >= 0.014
    env.close()
