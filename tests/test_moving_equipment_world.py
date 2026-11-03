"""Real analytic native geometry and dynamics, with the detector/body fixture scope explicit."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_moving_equipment_proposal import packet, settings
from weft.moving_equipment_proposal import derive_moving_proposals
from weft.moving_equipment_world import construct_moving_equipment_world
from weft.shared_world import SharedWorldEnv, validate_shared_snapshot


def test_free_body_constructor_keeps_native_body_and_uses_physics_not_image_positions(monkeypatch):
    admitted, calls = packet(monkeypatch)
    original = admitted.base_scene
    bundle = derive_moving_proposals(admitted, settings(free=True))
    preserved = deepcopy(bundle)
    scene = construct_moving_equipment_world(admitted, bundle, "candidate-1")
    assert len(calls) == 1 and bundle == preserved
    assert scene["profile"] == "shared-world-v2" and scene["learned_policy_available"] is False
    for key in ("human_spec", "coordinate_frame", "dt", "substeps", "gravity", "bounds", "contact"):
        assert scene[key] == original[key]
    assert scene["task"]["native_interval"]["duration_s"] == .45
    assert scene["task"]["horizon_steps"] == 14
    assert [obj["id"] for obj in scene["objects"]] == ["raw-0"]
    assert len(scene["source_binding"]["evidence_documents"]) == 2
    assert [row["world_entity_id"] for row in scene["source_binding"]["entity_bindings"]] == ["human", "raw-0"]
    provenance = scene["objects"][0]["provenance"]
    assert not provenance["trajectory_driving"] and not provenance["hidden_assistance"]
    assert provenance["initial_state_hypothesis"]["initial_state_status"] == "authored_prefix_hypothesis"
    assert provenance["initial_geometry_checks"][0]["physics_steps"] == 0
    env = SharedWorldEnv(scene)
    try:
        env.reset(seed=19, options={"jitter": False})
        first = env.snapshot()["objects"][0]
        action = env.authored_action()
        assert np.array_equal(action[22:], [0., 0., 0.])
        env.step(action)
        after = env.snapshot()
        validate_shared_snapshot(after, scene)
        assert after["objects"][0]["position"][2] < first["position"][2]
        assert after["objects"][0]["velocity"][2] < 0
        assert after["objects"][0]["virtual_force"] == [0., 0., 0.]
    finally:
        env.close()


def test_complete_candidate_family_kept_when_one_overlap_branch_refuses(monkeypatch):
    admitted, _ = packet(monkeypatch)
    bundle = derive_moving_proposals(admitted, settings(free=True))
    assert construct_moving_equipment_world(admitted, bundle, "candidate-2")["objects"][0]["id"] == "raw-1"
    with pytest.raises(ValueError, match="overlap|intersect"):
        construct_moving_equipment_world(admitted, bundle, "candidate-3")
    assert len(bundle["candidates"]) == 3
    assert all(item["status"] == "conditional_free_body_hypothesis" for item in bundle["candidates"])


@pytest.mark.parametrize("damage", ["radius", "initial_state", "source_pin", "remove_alternative", "clock", "slot"])
def test_constructor_independently_rederives_every_proposal_before_native_work(monkeypatch, damage):
    admitted, _ = packet(monkeypatch)
    bundle = derive_moving_proposals(admitted, settings(free=True))
    if damage == "radius": bundle["track_proposals"][0]["shape_prior"]["radius_model_units"] *= 1.1
    elif damage == "initial_state": bundle["track_proposals"][0]["initial_state_hypothesis"]["position_model_units"][0] += .1
    elif damage == "source_pin": bundle["input_pins"]["motion_evidence_sha256"] = "f" * 64
    elif damage == "remove_alternative": bundle["candidates"].pop()
    elif damage == "clock": bundle["track_proposals"][0]["rows"][0]["body_time_s"] += .01
    else: bundle["track_proposals"][0]["rows"].pop()
    from weft import shared_world_reconstruction
    monkeypatch.setattr(shared_world_reconstruction, "_check_initial_geometry", lambda *_: pytest.fail("No native work before exact rederivation"))
    with pytest.raises(ValueError, match="exact source/prior rederivation"):
        construct_moving_equipment_world(admitted, bundle, "candidate-1")


def test_out_of_bounds_free_prior_is_rejected_without_nudging(monkeypatch):
    admitted, _ = packet(monkeypatch)
    prior = settings(free=True)
    prior["initialization"]["position_model_units"] = [-2., 0., .8]
    bundle = derive_moving_proposals(admitted, prior)
    before = deepcopy(bundle)
    with pytest.raises(ValueError, match="position|bounds"):
        construct_moving_equipment_world(admitted, bundle, "candidate-1")
    assert bundle == before


def test_later_initialized_box_only_sets_authored_prefix_state_at_original_start(monkeypatch):
    admitted, _ = packet(monkeypatch, raw_tracks=1, image_shift=(300, 190))
    bundle = derive_moving_proposals(admitted, settings())
    scene = construct_moving_equipment_world(admitted, bundle, "candidate-1")
    initial = scene["objects"][0]["provenance"]["initial_state_hypothesis"]
    assert initial["basis_frame"] == 4 and initial["initial_state_status"] == "authored_prefix_hypothesis"
    assert scene["task"]["start_offset_s"] == 0 and scene["human_spec"]["reference"][0]["time_s"] == 0
    assert len(bundle["track_proposals"][0]["rows"]) == 10
    assert scene["objects"][0]["velocity"] == [0., 0., 0.]
    assert not initial["physical_state_measured"]


def test_grid_aligned_moving_provenance_also_blocks_old_learning_and_task(monkeypatch, tmp_path):
    from test_dynamic_native_interval import dynamic_scene
    from weft import shared_world_learning as learning
    from weft.shared_exercise_task import build_shared_exercise_task
    from weft.shared_world import validate_shared_world

    scene = dynamic_scene()
    scene["task"].pop("native_interval")
    for name in ("reference", "root_reference"):
        for index, row in enumerate(scene["human_spec"][name]):
            row["time_s"] = index / 30
    scene["objects"][0]["provenance"]["profile"] = "moving-equipment-free-body-v1"
    scene = validate_shared_world(scene)
    monkeypatch.setattr(learning, "SharedWorldEnv", lambda *_: pytest.fail("Unqualified native work"))
    for call in (lambda: build_shared_exercise_task(scene), lambda: learning.validate_shared_policy({}, scene),
                 lambda: learning.policy_action({}, SimpleNamespace(scene=scene)),
                 lambda: learning.train_shared_world_policy(scene, tmp_path / "forbidden")):
        with pytest.raises(ValueError, match="not qualified|separately qualified"):
            call()
    assert not (tmp_path / "forbidden").exists()
