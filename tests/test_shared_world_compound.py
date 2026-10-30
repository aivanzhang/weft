"""Authored compound geometry remains one rigid actor and preserves legacy worlds."""

from copy import deepcopy
import hashlib
import math

import mujoco
import numpy as np
import pytest

from weft.recording import Recorder, load_recording
from weft.shared_exercise_recording import recording_scene, verify_action_replay
from weft.shared_exercise_task import SharedExerciseTaskEnv, build_shared_exercise_task
from weft.shared_world import (
    SharedWorldEnv, _capsule_endpoints, _world_primitives, audit_shared_world_feasibility,
    render_snapshot, shared_world_xml, validate_shared_snapshot, validate_shared_world,
)
from weft.shared_world_lab import authored_shared_world_scene


def part(name, shape, position, **dimensions):
    return {"id": name, "shape": shape, "position": position, "quaternion": [1., 0., 0., 0.], **dimensions}


def compound(obj, parts, *, diagonal=(.004, .004, .004)):
    obj.pop("size", None)
    obj.pop("radius", None)
    obj["shape"] = "compound"
    obj["geometry"] = {"profile": "rigid-compound-v1", "origin": "authored_hypothesis",
                       "assumptions": ["Analytic rigid union and centered inertia prior; no observed cavity or grasp."],
                       "inertia": {"mode": "authored_centered_diagonal", "diagonal": list(diagonal)},
                       "parts": deepcopy(parts)}
    return obj


def mixed_scene():
    scene = authored_shared_world_scene()
    obj = compound(scene["objects"][1], [
        part("block", "box", [-.12, 0., 0.], size=[.08, .06, .08]),
        part("round", "sphere", [.12, 0., 0.], radius=.035),
        part("bar", "capsule", [0., .05, .04], radius=.02, length=.16),
    ])
    obj["position"] = [.35, 1.65, 1.5]
    obj["quaternion"] = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
    obj["geometry"]["parts"][2]["quaternion"] = [math.sqrt(.5), math.sqrt(.5), 0., 0.]
    obj["color"] = "#F10F59"
    return validate_shared_world(scene)


def open_shape_scene():
    scene = authored_shared_world_scene()
    scene["gravity"] = [0., 0., 0.]
    obj = compound(scene["objects"][1], [
        part("left", "box", [-.09, 0., 0.], size=[.04, .12, .18]),
        part("right", "box", [.09, 0., 0.], size=[.04, .12, .18]),
        part("base", "box", [0., 0., -.09], size=[.22, .12, .04]),
    ])
    obj["position"] = [.35, 1.65, 1.4]
    ball = scene["objects"][0]
    ball["position"], ball["radius"] = [.35, 1.65, 1.43], .025
    return validate_shared_world(scene)


def test_legacy_xml_and_nominal_dynamics_remain_frozen():
    scene = authored_shared_world_scene()
    assert hashlib.sha256(shared_world_xml(scene).encode()).hexdigest() == "050321d77477c66f914681862cee2c7b87761d84ee7397cd2fde65ee805dbdf1"
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        for _ in range(3):
            env.step(env.authored_action())
        expected = [-.0030469065158009057, .00008031238961403892, -.000894681098269681,
                    .00004536338469374563, .00014430710862609544, .1799464194139328, .3500252693116878,
                    .000019569655997491034, -.00002524482315950465, .08259316091869401, .16498562349261836,
                    .00010104033574464389, .00016310047476003067, .18948100847822588, 1.2003857901339234,
                    .000019576422057204226, -.000024738899446450787, .08352770384146861, .1648509735341347,
                    -.00003159824961464928, -.0004898354810111498, -.0002887573984108877]
        np.testing.assert_allclose(env.data.qpos[env._q], expected, rtol=0, atol=1e-12)
        snapshot = env.snapshot()
        assert snapshot["contact_count"] == 5
        assert all("parts" not in row for row in snapshot["contact_details"])
        assert set(env._object_geom_ids) == {obj["id"] for obj in scene["objects"]}
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ["radius", "size", "label", "role", "extra_geometry", "unknown_actor"])
def test_snapshot_cannot_substitute_geometry_or_static_identity(mutation):
    scene = authored_shared_world_scene()
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        state = env.snapshot()
        if mutation == "radius":
            state["objects"][0]["radius"] = .49
        elif mutation == "size":
            state["objects"][1]["size"] = [.9] * 3
        elif mutation == "label":
            state["objects"][0]["label"] = "A different object"
        elif mutation == "role":
            state["objects"][0]["role"] = "static"
        elif mutation == "extra_geometry":
            state["objects"][0]["size"] = [.2] * 3
        else:
            state["controlled_id"] = "table"
        with pytest.raises(ValueError):
            validate_shared_snapshot(state, scene)
    finally:
        env.close()


def test_valid_actor_switching_preserves_pinned_geometry_and_static_roles():
    scene = mixed_scene()
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        for selected in ("ball", "marker_box", "human"):
            env.select_target(selected)
            state = env.snapshot()
            validate_shared_snapshot(state, scene)
            assert [obj["id"] for obj in state["objects"] if obj["role"] == "active"] == ([] if selected == "human" else [selected])
        state["objects"][2]["role"] = "passive"
        with pytest.raises(ValueError, match="role"):
            validate_shared_snapshot(state, scene)
    finally:
        env.close()


@pytest.mark.parametrize("mutation", ["version", "measured_origin", "assumptions", "inertia_mode", "zero_inertia",
    "inertia_triangle", "nonfinite", "quaternion", "zero_capsule", "recursive", "part_mass", "duplicate_id",
    "duplicate_geometry", "nine_parts", "world_bounds", "extent"])
def test_compound_admission_rejects_unsupported_or_unbounded_hypotheses(mutation):
    scene = mixed_scene()
    obj = scene["objects"][1]
    geometry, parts = obj["geometry"], obj["geometry"]["parts"]
    if mutation == "version":
        geometry["profile"] = "rigid-compound-v2"
    elif mutation == "measured_origin":
        geometry["origin"] = "measured_cavity"
    elif mutation == "assumptions":
        geometry["assumptions"] = []
    elif mutation == "inertia_mode":
        geometry["inertia"]["mode"] = "automatic"
    elif mutation == "zero_inertia":
        geometry["inertia"]["diagonal"][0] = 0.
    elif mutation == "inertia_triangle":
        geometry["inertia"]["diagonal"] = [.01, .001, .001]
    elif mutation == "nonfinite":
        parts[0]["position"][0] = float("nan")
    elif mutation == "quaternion":
        parts[0]["quaternion"] = [0., 0., 0., 0.]
    elif mutation == "zero_capsule":
        parts[2]["length"] = 0.
    elif mutation == "recursive":
        parts[0]["shape"] = "compound"
    elif mutation == "part_mass":
        parts[0]["mass"] = 1.
    elif mutation == "duplicate_id":
        parts[1]["id"] = parts[0]["id"]
    elif mutation == "duplicate_geometry":
        parts.append(deepcopy(parts[0]) | {"id": "redundant"})
    elif mutation == "nine_parts":
        geometry["parts"] = [part(f"part-{i}", "sphere", [i * .02, 0., 0.], radius=.01) for i in range(9)]
    elif mutation == "world_bounds":
        parts[1]["position"][0] = 1.
    else:
        parts[1]["position"][0] = 1.2
    with pytest.raises(ValueError):
        validate_shared_world(scene)


def test_total_primitive_count_is_bounded_without_adding_actor_channels():
    scene = authored_shared_world_scene()
    template = deepcopy(scene["objects"][1])
    for i in range(4):
        obj = compound(deepcopy(template), [part(f"p{j}", "sphere", [j * .02, 0., 0.], radius=.01) for j in range(8)])
        obj["id"] = f"cluster-{i}"
        scene["objects"].append(obj)
    scene.pop("action_contract")
    with pytest.raises(ValueError, match="32 rigid"):
        validate_shared_world(scene)


def test_exact_32_primitive_boundary_keeps_one_action_triplet_per_dynamic_object():
    scene = authored_shared_world_scene()
    scene["objects"].append(deepcopy(scene["objects"][1]) | {"id": "fourth"})
    for obj in scene["objects"]:
        compound(obj, [part(f"p{j}", "sphere", [j * .02, 0., 0.], radius=.01) for j in range(8)])
    scene.pop("action_contract")
    accepted = validate_shared_world(scene)
    assert len(accepted["action_contract"]["names"]) == 22 + 3 * 3
    assert accepted["action_contract"]["dynamic_object_ids"] == ["ball", "marker_box", "fourth"]
    extra = deepcopy(scene["objects"][0])
    extra["id"] = "thirty-third"
    extra["geometry"]["parts"] = extra["geometry"]["parts"][:1]
    scene["objects"].append(extra)
    with pytest.raises(ValueError, match="32 rigid"):
        validate_shared_world(scene)


@pytest.mark.parametrize("change", ["position", "boolean", "inertia", "origin", "part_id"])
def test_recorded_compound_geometry_cannot_change_or_use_boolean_coordinates(change):
    scene = mixed_scene()
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        state = env.snapshot()
        geometry = state["objects"][1]["geometry"]
        if change == "position":
            geometry["parts"][0]["position"][1] = .01
        elif change == "boolean":
            geometry["parts"][0]["position"][1] = False
        elif change == "inertia":
            geometry["inertia"]["diagonal"][0] *= 2
        elif change == "origin":
            geometry["origin"] = "measured"
        else:
            geometry["parts"][0]["id"] = "another-part"
        with pytest.raises(ValueError, match="geometry changed"):
            validate_shared_snapshot(state, scene)
    finally:
        env.close()


def test_compound_compiles_one_body_mass_and_rigid_local_transforms():
    scene = mixed_scene()
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        obj = scene["objects"][1]
        body = env.model.body(obj["id"]).id
        assert env.model.body_mass[body] == obj["mass"]
        np.testing.assert_allclose(env.model.body_ipos[body], [0., 0., 0.], atol=0, rtol=0)
        np.testing.assert_allclose(env.model.body_inertia[body], obj["geometry"]["inertia"]["diagonal"], atol=0, rtol=0)
        assert env.model.body_jntnum[body] == 1
        assert list(env.action_names) == authored_shared_world_scene()["action_contract"]["names"]
        assert len(env.snapshot()["objects"]) == 3
        ids = env.object_geometry_ids()
        assert len(ids["marker_box"]) == 3
        with pytest.raises(ValueError, match="all object_geometry_ids"):
            _ = env._object_geom_ids
        transformed = _world_primitives(obj)
        for gid, primitive in zip(ids[obj["id"]], transformed):
            np.testing.assert_allclose(env.data.geom_xpos[gid], primitive["position"], atol=1e-14)
            assert env.model.geom_bodyid[gid] == body
            if primitive["shape"] == "capsule":
                first, second = _capsule_endpoints(primitive)
                axis = env.data.geom_xmat[gid].reshape(3, 3)[:, 2]
                np.testing.assert_allclose(second - first, axis * primitive["length"], atol=1e-14)
        np.testing.assert_allclose(transformed[2]["position"], [.3, 1.65, 1.54], atol=1e-14)
        np.testing.assert_allclose(_capsule_endpoints(transformed[2]), [[.22, 1.65, 1.54], [.38, 1.65, 1.54]], atol=1e-14)
        env.data.qpos[env.model.jnt_qposadr[env.model.joint("marker_box_joint").id]:][:3] += [.05, -.03, .02]
        mujoco.mj_forward(env.model, env.data)
        snapshot = env.snapshot()
        assert snapshot["objects"][1]["geometry"] == obj["geometry"]
        validate_shared_snapshot(snapshot, scene)
    finally:
        env.close()


def test_empty_authored_gap_is_not_filled_and_later_part_overlap_is_rejected():
    scene = open_shape_scene()
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=0, options={"jitter": False})
        ids = env.object_geometry_ids()
        assert all(mujoco.mj_geomDistance(env.model, env.data, ids["ball"][0], gid, 1., None) > .01
                   for gid in ids["marker_box"])
        assert env.steps == 0
    finally:
        env.close()
    scene["objects"][0]["position"] = [.44, 1.65, 1.43]
    report = audit_shared_world_feasibility(scene)
    assert report["summary"]["dynamic_object_count"] == 2
    assert any(row.get("parts") == [None, "right"] for row in report["initial_intersections"])
    env = SharedWorldEnv(scene)
    try:
        with pytest.raises(ValueError, match="intersect"):
            env.reset(seed=0, options={"jitter": False})
    finally:
        env.close()


def test_contact_impulse_moves_one_compound_and_retains_exact_part_identity():
    scene = open_shape_scene()
    scene["objects"][0]["velocity"] = [.5, 0., 0.]
    env = SharedWorldEnv(scene)
    contacts = []
    try:
        env.reset(seed=0, options={"jitter": False})
        for _ in range(6):
            env.step(np.zeros(len(env.action_names)))
            state = env.snapshot()
            validate_shared_snapshot(state, scene)
            contacts.extend(row for row in state["contact_details"] if row["objects"] == ["ball", "marker_box"])
        assert contacts and any(row["normal_force"] > 0 for row in contacts)
        assert all(row["parts"] == [None, "right"] for row in contacts)
        assert next(obj for obj in state["objects"] if obj["id"] == "marker_box")["velocity"][0] > 0
        bad = deepcopy(state)
        if bad["contact_details"]:
            bad["contact_details"][0]["parts"] = ["not-a-part", None]
            with pytest.raises(ValueError, match="part identity"):
                validate_shared_snapshot(bad, scene)
    finally:
        env.close()


def test_human_contact_telemetry_remains_object_level_for_a_later_compound_part():
    scene = authored_shared_world_scene()
    ball = scene["objects"][0]
    radius, inertia = ball["radius"], .4 * ball["mass"] * ball["radius"] ** 2
    compound(ball, [part("raised", "sphere", [0., .12, .12], radius=.01),
                    part("shell", "sphere", [0., 0., 0.], radius=radius)], diagonal=[inertia] * 3)
    ball["velocity"] = [0., -1.8, 0.]
    samples = []
    env = SharedWorldEnv(scene, substep_observer=samples.append)
    raw = []
    try:
        env.reset(seed=0, options={"jitter": False})
        for _ in range(8):
            env.step(np.zeros(len(env.action_names)))
            raw.extend(row for row in env.snapshot()["contact_details"] if row["kind"] == "human_object")
        assert raw and any("shell" in row["parts"] for row in raw)
        contacts = [contact for sample in samples for contact in sample.contacts if contact.object_id == "ball"]
        assert contacts and all(contact.human_segment for contact in contacts)
        assert all(item.object_id in {"ball", "marker_box", "table"} for sample in samples for item in sample.equipment)
    finally:
        env.close()


def test_existing_reset_jitter_checks_every_part_and_ignores_same_body_junctions():
    scene = open_shape_scene()
    env = SharedWorldEnv(scene)
    try:
        for seed in range(3):
            env.reset(seed=seed, options={"jitter": True, "jitter_scale": .2, "position_jitter": .005, "velocity_jitter": .01})
            ids = env.object_geometry_ids()
            assert all(mujoco.mj_geomDistance(env.model, env.data, first, second, 1., None) >= -1e-5
                       for first in ids["marker_box"] for other, geoms in ids.items() if other != "marker_box" for second in geoms)
            assert env.steps == 0
    finally:
        env.close()


@pytest.mark.parametrize("root_assistance", [False, True])
def test_compound_human_only_recordings_render_and_replay_without_object_forces(tmp_path, root_assistance):
    scene = mixed_scene()
    scene["task"]["horizon_steps"] = 3
    task = build_shared_exercise_task(scene, root_assistance=root_assistance)
    env = SharedExerciseTaskEnv(scene, task)
    recorder = Recorder(tmp_path / "recording", recording_scene(scene, task), profile="shared-exercise")
    try:
        env.reset(seed=3, options={"jitter": False, "jitter_scale": 0.})
        recorder.start(env.snapshot(), seed=3)
        for _ in range(3):
            action = env.authored_action()
            _, reward, terminated, truncated, _ = env.step(action)
            recorder.append({"type": "step", "action": action.tolist(), "reward": reward, "terminated": terminated,
                             "truncated": truncated, "snapshot": env.snapshot(), "controller": "authored"})
        recorder.close()
        data = load_recording(recorder.root)
        assert verify_action_replay(recorder.root)["exact_state_replay"]
        for row in data["events"]:
            engine = row["snapshot"]["engine_state"]
            assert not any(engine["normalized_action"][22:])
            assert engine["objects"][1]["geometry"] == scene["objects"][1]["geometry"]
        image = render_snapshot(data["events"][0]["snapshot"]["engine_state"], scene, width=960)
        for pane in np.split(image, 3, axis=1):
            assert np.count_nonzero(np.all(pane == [241, 15, 89], axis=2)) > 2
        changed = deepcopy(data["events"][0]["snapshot"]["engine_state"])
        changed["objects"][1]["geometry"]["parts"][2]["length"] += .1
        with pytest.raises(ValueError, match="geometry changed"):
            render_snapshot(changed, scene)
    finally:
        recorder.close()
        env.close()
