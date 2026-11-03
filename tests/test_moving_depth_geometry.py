"""Actual initial model geometry; synthetic source fixture stubs only original body admission."""

from copy import deepcopy
import time

import mujoco
import numpy as np
import pytest

from test_moving_equipment_proposal import packet
from weft.io import canonical_hash
from weft.moving_depth_geometry import extract_depth_geometry, validate_depth_geometry, _geometry


def analytic_geometry():
    """Structural solver fixture only; its placeholder pins are not source admission."""
    from weft.moving_depth_geometry import PROFILE, FILTER, GATES, SCOPE
    bounds = {"low": [0., 0., 0.], "high": [2., 2., 2.], "planes": []}
    for axis, length in enumerate(bounds["high"]):
        normal = [0., 0., 0.]; normal[axis] = 1.
        bounds["planes"] += [{"name": f"axis-{axis}-low", "normal": normal, "offset": 0.},
                             {"name": f"axis-{axis}-high", "normal": [-v for v in normal], "offset": -length}]
    return {"schema_version": 1, "profile": PROFILE,
        "input_pins": {name: "a" * 64 for name in ("admission_receipt_sha256", "base_scene_sha256", "full_human_sha256",
            "source_sha256", "motion_evidence_sha256", "body_binding_sha256")},
        "runtime_sha256": {"moving_depth_geometry.py": "b" * 64},
        "runtime_versions": {name: "analytical-fixture" for name in ("python", "system", "machine", "mujoco", "numpy", "gymnasium")},
        "source_reference": {"first": {"index": 0, "source_frame": 0, "evidence_id": "analytic:0", "time_s": 0.},
            "count": 3, **{name: "c" * 64 for name in ("reference_sha256", "root_reference_sha256", "human_qpos_sha256", "native_clock_sha256")}},
        "coordinate_frame": {"scope": "Analytical solver fixture; no physical or original-source qualification"},
        "world_bounds": bounds, "primitives": [{"name": "analytic-body", "type": "sphere", "center": [1., 1., 1.],
            "radius": .1, "contype": 2, "conaffinity": 5, "margin": 0., "gap": 0.}],
        "floor": {"name": "mobile_floor", "type": "plane", "normal": [0., 0., 1.], "offset": 0.,
            "contype": 1, "conaffinity": 6, "margin": 0., "gap": 0.},
        "candidate_collision_filter": dict(FILTER), "geometric_gates": dict(GATES),
        "ignored_authored_object_ids": [], "physics_steps": 0, "model_compilations": 1, "scope": SCOPE}


def test_pure_fixture_and_structure_bounds():
    fixture = analytic_geometry()
    assert validate_depth_geometry(fixture) == fixture
    for _ in range(20): fixture["coordinate_frame"] = {"nested": fixture["coordinate_frame"]}
    with pytest.raises(ValueError, match="structural bounds"): validate_depth_geometry(fixture)


def test_exact_initial_primitives_match_native_distances_once_without_steps(monkeypatch):
    from weft import shared_world
    inputs, _ = packet(monkeypatch)
    original = inputs.base_scene
    created = []
    Native = shared_world.SharedWorldEnv
    class Watched(Native):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)
        def reset(self, *args, **kwargs): pytest.fail("No authored-prop reset or pose adjustment")
        def step(self, *args, **kwargs): pytest.fail("No physics steps")
        def close(self): self.was_closed = True
    monkeypatch.setattr(shared_world, "SharedWorldEnv", Watched)
    value = extract_depth_geometry(inputs)
    assert len(created) == 1 and created[0].was_closed
    env = created[0]
    assert value["physics_steps"] == 0 and value["model_compilations"] == 1
    assert env.steps == env.data.time == 0
    assert inputs.base_scene == original
    assert value["input_pins"]["base_scene_sha256"] == canonical_hash(original)
    assert value["source_reference"]["count"] == len(original["human_spec"]["reference"])
    assert value["coordinate_frame"] == original["coordinate_frame"]
    assert {p["name"] for p in value["primitives"]} == set(env._human_geom_to_name.values())
    assert value["ignored_authored_object_ids"] == sorted(obj["id"] for obj in original["objects"])
    assert {p["type"] for p in value["primitives"]} == {"sphere", "capsule"}
    ball = env.model.geom("ball").id
    center = env.data.geom_xpos[ball].copy()
    radius = float(env.model.geom_size[ball, 0])
    for row in value["primitives"]:
        if row["type"] == "sphere":
            nearest = np.asarray(row["center"])
        else:
            start, end = np.asarray(row["endpoints"])
            segment = end - start
            nearest = start + np.clip(np.dot(center-start, segment) / np.dot(segment, segment), 0, 1) * segment
        analytic = float(np.linalg.norm(center - nearest)) - radius - row["radius"]
        native = mujoco.mj_geomDistance(env.model, env.data, ball, env.model.geom(row["name"]).id, 10., None)
        assert analytic == pytest.approx(native, abs=1e-12)
    floor = value["floor"]
    assert np.dot(floor["normal"], center) - floor["offset"] - radius == pytest.approx(
        mujoco.mj_geomDistance(env.model, env.data, ball, env._floor_geom, 10., None), abs=1e-12)
    assert value["world_bounds"]["low"] == [0, 0, 0] and value["world_bounds"]["high"] == original["bounds"]
    assert value["geometric_gates"]["domain_clearance_world"] == 0
    assert {"shared_world.py", "source_clock.py", "shared_human_projection.py", "moving_depth_geometry.py"} <= value["runtime_sha256"].keys()


@pytest.mark.parametrize("damage", ["box", "filter", "lost_geom", "contact_disabled"])
def test_unsupported_exact_geometry_or_filters_refuse(monkeypatch, damage):
    from weft.shared_world import SharedWorldEnv
    inputs, _ = packet(monkeypatch)
    env = SharedWorldEnv(inputs.base_scene)
    try:
        mujoco.mj_forward(env.model, env.data)
        gid = next(iter(env._human_geom_to_name))
        if damage == "box": env.model.geom_type[gid] = int(mujoco.mjtGeom.mjGEOM_BOX)
        elif damage == "filter": env.model.geom_contype[gid] = 0
        elif damage == "lost_geom": del env._human_geom_to_name[gid]
        else: env.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        with pytest.raises(ValueError, match="unsupported|retained"):
            _geometry(env)
    finally:
        env.close()


def test_deadline_and_original_packet_required_before_compile(monkeypatch):
    from weft import shared_world
    monkeypatch.setattr(shared_world, "SharedWorldEnv", lambda *_: pytest.fail("No compile before admission"))
    with pytest.raises(TimeoutError): extract_depth_geometry(object(), deadline=time.monotonic() - 1)
    with pytest.raises(ValueError, match="immutable packet"): extract_depth_geometry(object())
    with pytest.raises(ValueError, match="finite deadline"): extract_depth_geometry(object(), deadline=float("nan"))


def test_receipt_is_detached_and_shape_tampering_refuses(monkeypatch):
    import weft.moving_depth_geometry as module
    inputs, _ = packet(monkeypatch)
    value = extract_depth_geometry(inputs)
    original = deepcopy(value)
    detached = validate_depth_geometry(value)
    detached["primitives"][0]["radius"] *= 2
    assert value == original
    for changed in (dict(value, physics_steps=True), dict(value, extra=True),
                    dict(value, geometric_gates={**value["geometric_gates"], "domain_clearance_world": -1e-5}),
                    dict(value, geometric_gates={**value["geometric_gates"], "mask_filtering_applied": 0})):
        with pytest.raises(ValueError): validate_depth_geometry(changed)
    changed = deepcopy(value); changed["world_bounds"]["planes"][0]["offset"] += .1
    with pytest.raises(ValueError, match="containment"): validate_depth_geometry(changed)
    changed = deepcopy(value); changed["primitives"][0]["radius"] = float("nan")
    with pytest.raises(ValueError): validate_depth_geometry(changed)
    monkeypatch.setattr(module, "MAX_BYTES", 100)
    with pytest.raises(ValueError, match="byte bound"): validate_depth_geometry(value)


def test_runtime_change_after_compile_refuses_and_closes(monkeypatch):
    import weft.moving_depth_geometry as module
    from weft import shared_world
    inputs, _ = packet(monkeypatch)
    original = module._runtime
    calls, closed = [], []
    old_close = shared_world.SharedWorldEnv.close
    def close(env):
        closed.append(1)
        old_close(env)
    monkeypatch.setattr(shared_world.SharedWorldEnv, "close", close)
    def runtime(names, deadline):
        result = original(names, deadline)
        calls.append(1)
        if len(calls) == 2: result["shared_world.py"] = "0" * 64
        return result
    monkeypatch.setattr(module, "_runtime", runtime)
    with pytest.raises(ValueError, match="runtime changed during extraction"):
        extract_depth_geometry(inputs)
    assert len(calls) == 2 and closed == [1]
