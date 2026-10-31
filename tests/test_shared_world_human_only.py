"""Person-only worlds are explicit inventories, not empty legacy object scenes."""

from copy import deepcopy

import mujoco
import numpy as np
import pytest

from weft.io import canonical_hash
from weft.person_inventory import make_person_inventory
from weft.shared_world import (
    FULLSPAN_PROFILE, FULLSPAN_PROFILES, HUMAN_ONLY_PROFILE, PROFILE, SharedWorldEnv,
    render_snapshot, validate_shared_snapshot, validate_shared_world,
)
from weft.shared_world_lab import authored_shared_world_scene


def person_scene(*, steps=3, first=0., scale=1.):
    scene = authored_shared_world_scene()
    scene.pop("action_contract")
    scene.update(profile=HUMAN_ONLY_PROFILE, objects=[], target=None, controlled_id="human",
                 inventory=make_person_inventory(None, analytic=True))
    body = scene["human_spec"]
    for key in ("reference", "root_reference"):
        original = body[key]
        rows = [deepcopy(original[min(index, len(original) - 1)]) for index in range(steps + 1)]
        for index, row in enumerate(rows):
            row["time_s"] = first + index / 30
        body[key] = rows
    body["mobility"]["floor_fit"]["reference_time_s"] = first
    scene["coordinate_frame"]["clock_binding"]["source_time_offset_s"] = first
    scene["coordinate_frame"]["scale_bridge"]["body_height_to_world_scale"] = scale
    scene["task"].update(type="human_motion_v1", object_goal=None, start_offset_s=0., horizon_steps=steps)
    return validate_shared_world(scene)


def test_person_scene_has_one_body_no_equipment_or_ghost_goal():
    scene = person_scene(first=2 / 30)
    assert HUMAN_ONLY_PROFILE in FULLSPAN_PROFILES and FULLSPAN_PROFILE in FULLSPAN_PROFILES
    assert scene["objects"] == [] and scene["target"] is None and scene["task"]["object_goal"] is None
    assert scene["inventory"]["absence_verified"] is False
    assert scene["action_contract"]["human_motor_count"] == 22
    assert scene["action_contract"]["dynamic_object_ids"] == []
    assert scene["action_contract"]["object_force_channels"] == 0
    assert len(scene["action_contract"]["names"]) == 22
    assert validate_shared_world(scene) == scene
    assert scene["human_spec"]["reference"][0]["time_s"] == 2 / 30


@pytest.mark.parametrize("profile", [PROFILE, FULLSPAN_PROFILE])
def test_legacy_profiles_do_not_accept_person_inventory_or_empty_objects(profile):
    scene = person_scene()
    scene["profile"] = profile
    with pytest.raises(ValueError, match="fields"):
        validate_shared_world(scene)
    scene.pop("inventory")
    with pytest.raises(ValueError, match="1–12"):
        validate_shared_world(scene)


@pytest.mark.parametrize("change", [
    lambda s: s.pop("inventory"),
    lambda s: s["inventory"].update(absence_verified=True),
    lambda s: s["inventory"].update(task_relevant_equipment="unknown"),
    lambda s: s.update(objects=authored_shared_world_scene()["objects"]),
    lambda s: s.update(controlled_id="ball"),
    lambda s: s.update(target=[1., 1., 1.]),
    lambda s: s["task"].update(object_goal={}),
    lambda s: s["task"].update(type="coupled_human_object_v1"),
    lambda s: s["task"].update(start_offset_s=1 / 30),
    lambda s: s["task"].update(horizon_steps=2),
    lambda s: s["task"]["human_limits"].update(angular_rmse_rad=.5),
    lambda s: s["human_spec"].update(source_sha256="a" * 64),
])
def test_person_scene_rejects_ambiguous_inventory_goal_and_clipped_interval(change):
    scene = person_scene()
    change(scene)
    with pytest.raises(ValueError):
        validate_shared_world(scene)


def test_person_inventory_is_bound_to_exact_source_and_all_slots():
    scene = person_scene(first=2 / 30)
    sha = "a" * 64
    scene["human_spec"]["source_sha256"] = sha
    scene["inventory"] = make_person_inventory(sha)
    scene["source_binding"] = {
        "mode": "source_qualified", "source_sha256": sha, "total_source_frames": 7,
        "evidence_documents": {"d" * 64: "evidence/human.json"},
        "entity_bindings": [{"world_entity_id": "human", "entity_kind": "articulated_human",
                             "observation_entity_id": sha + ":human:0", "observation_document_sha256": "d" * 64,
                             "provenance_mode": "observed_track", "observed_slots": 4,
                             "missing_slots": 3, "pre_init_slots": 0}],
    }
    assert validate_shared_world(scene) == scene
    for change in (lambda s: s["inventory"].update(source_sha256="b" * 64),
                   lambda s: s["human_spec"].pop("source_sha256"),
                   lambda s: s["source_binding"]["entity_bindings"][0].update(missing_slots=0)):
        bad = deepcopy(scene)
        change(bad)
        with pytest.raises(ValueError):
            validate_shared_world(bad)


def test_person_full_reference_can_exceed_legacy_window_without_clipping():
    scene = person_scene(steps=181, first=2 / 30)
    assert scene["task"]["horizon_steps"] == 181
    assert scene["human_spec"]["reference"][-1]["time_s"] == pytest.approx(183 / 30)
    bad = deepcopy(scene)
    bad["task"]["horizon_steps"] = 180
    with pytest.raises(ValueError, match="complete original"):
        validate_shared_world(bad)


def test_person_engine_reset_step_exact_replay_and_render_have_no_object_channels():
    scene = person_scene(first=2 / 30)
    samples = []
    env = SharedWorldEnv(scene, substep_observer=samples.append)
    try:
        initial, _ = env.reset(seed=19, options={"jitter": False})
        assert initial.shape == env.observation_space.shape == (155,)
        assert env.action_space.shape == (22,) and env.model.nq == env.model.nv == env.model.nu == 22
        assert env.model.ngeom == 13 and env.object_geometry_ids() == {}
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "ball") == -1
        state = env.snapshot()
        validate_shared_snapshot(state, scene)
        assert np.array_equal(render_snapshot(state, scene, width=480),
                              render_snapshot(state, scene, width=480, show_target=False))
        with pytest.raises(ValueError, match="Unknown"):
            env.select_target("ball")
        with pytest.raises(ValueError, match="Unknown"):
            env.discrete_object_action_3d("ball", 0)
        with pytest.raises(ValueError, match="22 finite"):
            env.step([0.] * 25)
        actions, snapshots = [], []
        for step in range(scene["task"]["horizon_steps"]):
            action = env.authored_action()
            actions.append(action)
            _, reward, terminated, truncated, info = env.step(action)
            assert np.isfinite(reward) and not terminated
            assert truncated is (step == scene["task"]["horizon_steps"] - 1)
            assert info["object_goal_score"] is None
            assert info["object_parked"] is None and info["human_object_contact_satisfied"] is None
            assert info["object_goal_status"] == info["human_object_contact_status"] == "not_applicable"
            state = env.snapshot()
            validate_shared_snapshot(state, scene)
            snapshots.append(canonical_hash(state))
        assert all(sample.equipment == () and sample.contacts == () for sample in samples)
        assert not np.any(env.data.xfrc_applied) and not np.any(env.data.qfrc_applied)
        assert env.data.time == pytest.approx(3 / 30)
        assert state["source_time_s"] == pytest.approx(5 / 30)
        env.reset(seed=19, options={"jitter": False})
        for action, expected in zip(actions, snapshots):
            env.step(action)
            assert canonical_hash(env.snapshot()) == expected
        reference = env.reference_snapshot(len(scene["human_spec"]["reference"]) - 1)
        validate_shared_snapshot(reference, scene)
        assert reference["target"] is None and reference["objects"] == []
    finally:
        env.close()


@pytest.mark.parametrize("change", [
    lambda s: s.update(target=[0., 0., 0.]),
    lambda s: s.update(virtual_force=[1., 0., 0.]),
    lambda s: s["contact_summary"].update(human_object=1),
    lambda s: s["coupled_metrics"].update(object_parked=True),
    lambda s: s["coupled_metrics"].pop("human_object_contact_status"),
    lambda s: s.update(source_time_s=1.),
])
def test_person_snapshot_rejects_phantom_equipment_success_and_wrong_clock(change):
    scene = person_scene()
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=4)
        state = env.snapshot()
        change(state)
        with pytest.raises(ValueError):
            validate_shared_snapshot(state, scene)
    finally:
        env.close()
