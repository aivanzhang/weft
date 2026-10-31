"""Tiny analytical v5 worlds verify ankle actions in the common physics engine."""

from copy import deepcopy
from hashlib import sha256
import json
import socket
import subprocess
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from test_shared_world_feet_contract import inputs as foot_inputs, raw
from weft.foot_body import ANKLE_NAMES, FOOT_BODY_CONTRACT
from weft.io import canonical_hash
from weft.mobile_human import ROOT_AXES
from weft.person_inventory import make_person_inventory
from weft.shared_world import (
    FEET_PROFILE, FULLSPAN_PROFILE, HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE,
    SharedWorldEnv, add_feet_to_shared_world, prepare_shared_renderer, prepare_shared_snapshot_validator,
    validate_shared_world,
)
from weft.shared_world_lab import authored_shared_world_scene
from weft.spatial_pose import SPATIAL_JOINT_NAMES, rotation_zyx


SOURCE = "a" * 64
STEPS = 8


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Shared-feet fixtures must not render, spawn workers, or use the network")

    monkeypatch.setattr(mujoco, "Renderer", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)


@pytest.fixture(scope="module")
def scene_template():
    return authored_shared_world_scene()


def fixture_scene(template, profile=HUMAN_ONLY_PROFILE, *, scale=1., contact=None):
    scene = deepcopy(template)
    scene.pop("action_contract")
    spec, reference = foot_inputs()
    first = 2 / 30
    pose, root_pose = deepcopy(spec["reference"][0]), deepcopy(spec["root_reference"][0])
    spec["reference"] = [{**deepcopy(pose), "time_s": first + i / 30} for i in range(STEPS + 1)]
    spec["root_reference"] = [{**deepcopy(root_pose), "time_s": first + i / 30} for i in range(STEPS + 1)]
    if contact:
        surface_height = .2 if contact == "static" else 0.
        spec["mobility"]["base_height"] = (.42 + .4 + .08 + FOOT_BODY_CONTRACT["radius_model_units"]
                                                + surface_height + .001)
    scene.update(profile=profile, human_spec=spec, controlled_id="human")
    scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"] = scale
    scene["coordinate_frame"]["clock_binding"]["source_time_offset_s"] = first
    scene["task"].update(start_offset_s=0., horizon_steps=STEPS)
    for obj, position in zip(scene["objects"], ([1.5, 1.5, .6], [1.2, 1.5, .6], [1.5, 1.5, .1])):
        obj["position"] = position
    table = next(obj for obj in scene["objects"] if obj["id"] == "table")
    table["size"] = [.3, .3, .2]
    if profile == HUMAN_ONLY_PROFILE:
        scene.update(objects=[], target=None, inventory=make_person_inventory(SOURCE))
        scene["task"].update(type="human_motion_v1", object_goal=None)
    elif profile == STATIC_SUPPORT_PROFILE:
        scene.update(objects=[table], target=None)
        scene["task"].update(type="human_support_v1", object_goal=None)
        if contact == "static":
            table.update(position=[.9, .6, .1], size=[.8, .4, .2])
    else:
        scene["target"] = [1.5, 1.7, .6]
    scene["source_binding"] = {
        "mode": "source_qualified", "source_sha256": SOURCE, "total_source_frames": STEPS + 3,
        "evidence_documents": {"d" * 64: "evidence/analytical-human.json"},
        "entity_bindings": [{"world_entity_id": "human", "entity_kind": "articulated_human",
            "observation_entity_id": SOURCE + ":human:0", "observation_document_sha256": "d" * 64,
            "provenance_mode": "observed_track", "observed_slots": STEPS + 1, "missing_slots": 2, "pre_init_slots": 0},
            *[{"world_entity_id": obj["id"], "entity_kind": "rigid_object", "observation_entity_id": None,
                "observation_document_sha256": None, "provenance_mode": "authored_addition", "observed_slots": 0,
                "missing_slots": STEPS + 3, "pre_init_slots": 0} for obj in scene["objects"]]],
    }
    scene["assumptions"].append("Synthetic source identities and authored geometry for engine tests only.")
    scene = validate_shared_world(scene)
    reference.update(body_model_sha256=canonical_hash(scene["human_spec"]), source_sha256=SOURCE)
    start = deepcopy(reference["frames"][0])
    reference["frames"] = [{**deepcopy(start), "time_s": first + i / 30, "frame": i + 2,
                            "evidence_id": SOURCE + f":frame:{i + 2}"} for i in range(STEPS + 1)]
    # Later source orientations and gaps must not become ankle tracking targets.
    for row in reference["frames"][1:]:
        for foot in row["feet"].values():
            foot["reference"]["heel_to_index_axis"] = [0., 1., 0.]
            foot["endpoint_geometry"] = {"heel_relative": [0., -.04, -.08], "foot_index_relative": [0., .16, -.08]}
    for foot in reference["frames"][2]["feet"].values():
        foot.update(status="unavailable", reference=None, endpoint_geometry=None)
    original = raw(reference)
    extended = add_feet_to_shared_world(scene, original)
    return scene, extended, original


@pytest.mark.parametrize("profile", [FULLSPAN_PROFILE, HUMAN_ONLY_PROFILE, STATIC_SUPPORT_PROFILE])
def test_feet_extend_each_fullspan_profile_without_changing_source_task_or_motor_prefix(scene_template, profile):
    base, scene, original = fixture_scene(scene_template, profile)
    assert scene["profile"] == FEET_PROFILE and scene["base_profile"] == profile
    for key in ("human_spec", "source_binding", "coordinate_frame", "task", "gravity", "objects", "target"):
        assert scene[key] == base[key]
    assert scene["foot_extension"]["reference_json"].encode() == original
    assert scene["foot_extension"]["reference_sha256"] == sha256(original).hexdigest()
    assert scene["learned_policy_available"] is False
    env = SharedWorldEnv(scene)
    try:
        observation, _ = env.reset(seed=13)
        names = list(SPATIAL_JOINT_NAMES) + list(ROOT_AXES) + list(ANKLE_NAMES)
        assert list(env.human_action_names) == names
        assert env.human_motor_count == env.model.nu == 28
        for index, name in enumerate(names):
            actuator = env.model.actuator(name + "_motor").id
            assert actuator == index
            assert env.model.actuator_trnid[actuator, 0] == env.model.joint(name).id
        dynamic = [obj["id"] for obj in scene["objects"] if obj["role"] != "static"]
        assert list(env.action_names[28:]) == [name + "_" + axis for name in dynamic for axis in ("fx", "fy", "fz")]
        assert env.action_space.shape == (28 + 3 * len(dynamic),)
        assert observation.shape == env.observation_space.shape == (197 + 13 * len(scene["objects"]) + (3 if dynamic else 0),)
        assert env.observation_space.contains(observation)
        np.testing.assert_array_equal(env.model.actuator_ctrlrange[22:28], np.tile([-12., 12.], (6, 1)))
        assert env.feature_matrix().shape == (28, 3)
        validate = prepare_shared_snapshot_validator(scene)
        snapshot = json.loads(json.dumps(env.snapshot(), allow_nan=False))
        validate(snapshot)
        assert len(snapshot["human"]["angles"]) == 25
        assert len(snapshot["human"]["muscles"]) == 50
        if profile == HUMAN_ONLY_PROFILE:
            for change in ("reference", "mode", "layout", "angles", "targets", "hold_error", "force"):
                wrong = deepcopy(snapshot)
                if change == "reference":
                    wrong["human"]["feet"]["reference_sha256"] = "f" * 64
                elif change == "mode":
                    wrong["human"]["feet"]["target_mode"] = "video_foot_interpolation"
                elif change == "layout":
                    wrong["human"]["action_names"][22:24] = reversed(wrong["human"]["action_names"][22:24])
                elif change == "angles":
                    wrong["human"]["angles"].pop(ANKLE_NAMES[0])
                elif change == "targets":
                    wrong["human"]["feet"]["initial_ankle_targets_rad"][0] += .1
                elif change == "hold_error":
                    wrong["human"]["feet"]["current_hold_error_rmse_rad"] += .5
                else:
                    wrong["normalized_action"][22] = wrong["human"]["normalized_action"][22] = .25
                with pytest.raises(ValueError):
                    validate(wrong)
    finally:
        env.close()


@pytest.mark.parametrize("scale", [.5, 1.5])
def test_feet_scaled_mass_matches_actual_human_subtree_not_rigid_object_mass(scene_template, scale):
    base, scene, _ = fixture_scene(scene_template, FULLSPAN_PROFILE, scale=scale)
    env = SharedWorldEnv(scene)
    try:
        human_root = env.model.body("mobile_root").id
        assert env.model.body_subtreemass[human_root] == pytest.approx(env.human_mass, rel=1e-12)
        old_limit = base["action_contract"]["physical_limits"][19]
        mobility = scene["human_spec"]["mobility"]
        assert env.human_limits[19] == pytest.approx(min(mobility["maximum_force"],
            old_limit + scale**3 * mobility["force_limit_weight_multiple"] * 9.81))
        for side in ("left", "right"):
            foot = env.model.body(side + "_foot_body").id
            geom = env.model.geom(side + "_foot_geom").id
            assert env.model.body_mass[foot] == pytest.approx(.5 * scale**3)
            assert env.model.geom_size[geom, 0] == pytest.approx(FOOT_BODY_CONTRACT["radius_model_units"] * scale)
            assert env.model.geom_contype[geom] & 2
            assert env.model.geom_conaffinity[geom] & 1 and env.model.geom_conaffinity[geom] & 4
        assert env.human_mass < np.sum(env.model.body_mass)
    finally:
        env.close()


def test_feet_targets_use_original_nominal_start_and_seeded_reset_is_repeatable(scene_template):
    _, scene, _ = fixture_scene(scene_template)
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=27)
        original = env.snapshot()
        initial = np.asarray(original["human"]["feet"]["initial_ankle_targets_rad"])
        np.testing.assert_array_equal(env.data.qpos[env._q[22:]], initial)
        for side, angles in zip(("left", "right"), initial.reshape(2, 3)):
            knee = env.data.xmat[env.model.body(side + "_knee_body").id].reshape(3, 3)
            np.testing.assert_allclose(knee @ rotation_zyx(angles), np.eye(3), atol=1e-12, rtol=0)
        validate = prepare_shared_snapshot_validator(scene)
        for index in (0, 2, STEPS):
            reference = env.reference_snapshot(index)
            validate(reference)
            assert [reference["human"]["angles"][name] for name in ANKLE_NAMES] == pytest.approx(initial)
        assert env.snapshot() == original
        first, _ = env.reset(seed=174, options={"jitter": True, "jitter_scale": .2})
        qpos, qvel = env.data.qpos.copy(), env.data.qvel.copy()
        assert env.snapshot()["human"]["feet"]["initial_ankle_targets_rad"] == initial.tolist()
        second, _ = env.reset(seed=174, options={"jitter": True, "jitter_scale": .2})
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(env.data.qpos, qpos)
        np.testing.assert_array_equal(env.data.qvel, qvel)
        model_id, data_id = id(env.model), id(env.data)
        for _ in range(3):
            env.step(np.zeros(28))
            target, velocity = env.target_state()
            np.testing.assert_array_equal(target[22:], initial)
            np.testing.assert_array_equal(velocity[22:], np.zeros(6))
        assert id(env.model) == model_id and id(env.data) == data_id
        assert env.data.time == pytest.approx(3 * scene["dt"])
        assert env.snapshot()["source_time_s"] == pytest.approx(2 / 30 + env.data.time)
    finally:
        env.close()


def test_ankle_motor_has_causal_motion_without_expanding_the_19_joint_tracking_score(scene_template):
    _, scene, _ = fixture_scene(scene_template)
    env = SharedWorldEnv(scene)
    try:
        outcomes = {}
        for amount in (0., .1, -.1):
            env.reset(seed=31)
            action = env.manual_action_vector(selected_motor="left_ankle_y", motor_nudge=amount)
            assert np.flatnonzero(action).tolist() == ([23] if amount else [])
            env.step(action)
            outcomes[amount] = float(env.data.qpos[env._q[23]])
            assert env.data.ctrl[23] == pytest.approx(amount * 12)
            error = env.data.qpos[env._q] - env.target_state()[0]
            assert env.metrics()["angular_rmse_rad"] == pytest.approx(np.sqrt(np.mean(error[:19] ** 2)), abs=1e-12)
            if amount:
                assert not np.isclose(np.mean(error[:19] ** 2), np.mean(error[np.r_[:19, 22:28]] ** 2), atol=1e-12)
        assert outcomes[.1] > outcomes[0.] + 1e-5
        assert outcomes[-.1] < outcomes[0.] - 1e-5
    finally:
        env.close()


def test_object_forces_begin_after_ankles_and_do_not_drive_human_motors(scene_template):
    _, scene, _ = fixture_scene(scene_template, FULLSPAN_PROFILE)
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=91)
        env.step(np.zeros(env.action_space.shape))
        unforced = env.snapshot()
        env.reset(seed=91)
        action = env.manual_action_vector(object_id="ball", object_action_code=2)
        assert np.flatnonzero(action).tolist() == [28]
        env.step(action)
        state = env.snapshot()
        validate = prepare_shared_snapshot_validator(scene)
        validate(json.loads(json.dumps(state, allow_nan=False)))
        np.testing.assert_array_equal(env.data.ctrl, np.zeros(28))
        ball = next(obj for obj in state["objects"] if obj["id"] == "ball")
        baseline = next(obj for obj in unforced["objects"] if obj["id"] == "ball")
        assert ball["virtual_force"] == pytest.approx([scene["objects"][0]["mass"] * 6, 0., 0.])
        assert ball["velocity"][0] > baseline["velocity"][0]
        assert all(obj.get("virtual_force", [0., 0., 0.]) == [0., 0., 0.] for obj in state["objects"] if obj["id"] != "ball")
        wrong = deepcopy(state)
        next(obj for obj in wrong["objects"] if obj["id"] == "ball")["virtual_force"] = [0., 0., 0.]
        with pytest.raises(ValueError, match="virtual_force"):
            validate(wrong)
    finally:
        env.close()


def test_substep_observer_requires_explicit_28_motor_opt_in_before_compilation(scene_template, monkeypatch):
    _, scene, _ = fixture_scene(scene_template)

    def compile_forbidden(*_args, **_kwargs):
        pytest.fail("An incompatible observer must be rejected before native compilation")

    with monkeypatch.context() as guard:
        guard.setattr(mujoco, "MjModel", SimpleNamespace(from_xml_string=compile_forbidden))
        for count in (None, 22):
            with pytest.raises((TypeError, ValueError), match="observer|motor"):
                SharedWorldEnv(scene, substep_observer=lambda sample: None, substep_observer_motor_count=count)
    samples = []
    env = SharedWorldEnv(scene, substep_observer=samples.append, substep_observer_motor_count=28)
    try:
        env.reset(seed=33)
        env.step(np.zeros(28))
        assert len(samples) == scene["substeps"]
        assert all(sample.valid for sample in samples)
        for sample in samples:
            assert len(sample.human_action) == len(sample.human_qpos) == len(sample.human_qvel) == len(sample.target_qpos) == 28
        assert samples[-1].interval_end_s == pytest.approx(env.data.time)
    finally:
        env.close()


@pytest.mark.parametrize("surface,profile,kind", [("floor", HUMAN_ONLY_PROFILE, "human_floor"),
                                                 ("static", STATIC_SUPPORT_PROFILE, "human_static")])
def test_added_capsules_report_real_floor_and_static_contacts_with_foot_endpoints(scene_template, surface, profile, kind):
    _, scene, _ = fixture_scene(scene_template, profile, contact=surface)
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=49)
        validate = prepare_shared_snapshot_validator(scene)
        for _ in range(3):
            env.step(np.zeros(28))
        state = json.loads(json.dumps(env.snapshot(), allow_nan=False))
        validate(state)
        contacts = [row for row in state["contact_details"] if row["kind"] == kind
                    and any(name in row["objects"] for name in ("left_foot_geom", "right_foot_geom"))]
        assert contacts and any(row["normal_force"] > 0 for row in contacts)
        assert state["physical_validation"] == "unverified"
        for side in ("left", "right"):
            segment = next(row for row in state["human"]["segments"] if row["id"] == side + "_foot")
            assert segment["a"] == state["human"]["joints"][side + "_heel"]
            assert segment["b"] == state["human"]["joints"][side + "_foot_index"]
            assert np.linalg.norm(np.asarray(segment["a"]) - segment["b"]) == pytest.approx(.2, abs=1e-12)
        if surface == "floor":
            floor = state["human"]["mobility"]
            assert floor["contact_count"] == len([row for row in state["contact_details"] if row["kind"] == kind])
            assert floor["total_normal_force"] == pytest.approx(sum(row["normal_force"] for row in floor["contacts"]))
        else:
            assert state["objects"][0]["position"] == scene["objects"][0]["position"]
    finally:
        env.close()


def test_feet_render_reuses_prepared_and_live_models_but_rechecks_each_frame(scene_template, monkeypatch):
    _, scene, _ = fixture_scene(scene_template)
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=81)
        state = env.snapshot()
        render = prepare_shared_renderer(scene, width=480)

        def native_construction_forbidden(*_args, **_kwargs):
            pytest.fail("Rendering a prepared v5 frame must not construct a native model/data or advance physics")

        monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(from_xml_string=native_construction_forbidden,
                                                               from_xml_path=native_construction_forbidden))
        monkeypatch.setattr(mujoco, "MjData", native_construction_forbidden)
        monkeypatch.setattr(mujoco, "mj_step", native_construction_forbidden)
        first, second = render(state), render(deepcopy(state))
        assert first.shape == (254, 480, 3) and first.dtype == np.uint8
        assert np.count_nonzero(np.any(first != [245, 234, 220], axis=2)) > 1000
        np.testing.assert_array_equal(first, second)
        wrong = deepcopy(state)
        wrong["human"]["feet"]["initial_ankle_targets_rad"][0] += .1
        with pytest.raises(ValueError):
            render(wrong)
        live_first, live_second = env.render(), env.render()
        assert live_first.shape == (438, 960, 3) and live_first.dtype == np.uint8
        assert np.count_nonzero(np.any(live_first != [245, 234, 220], axis=2)) > 1000
        np.testing.assert_array_equal(live_first, live_second)
        assert env.snapshot() == state and env.data.time == 0.
    finally:
        env.close()
