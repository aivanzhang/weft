from copy import deepcopy

import numpy as np
import pytest

from weft.io import canonical_hash
from weft.spatial_lab import authored_spatial_scene
from weft.spatial_learning import policy_action, validate_spatial_policy
from weft.spatial_rigid import validate_scene


def policy(version, weights=None):
    scene = validate_scene(authored_spatial_scene())
    return {"schema_version": version, "profile": f"spatial-policy-v{version}", "algorithm": f"spatial_cem_v{version}",
            "scene_sha256": canonical_hash(scene), "object_ids": [obj["id"] for obj in scene["objects"]],
            "controlled_id": scene["controlled_id"], "controlled_index": 0,
            "observation_dim": 13 * len(scene["objects"]) + 4,
            "weights": weights or [1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.],
            "position_tolerance": scene["task"]["position_tolerance"],
            "speed_tolerance": scene["task"]["speed_tolerance"], "physical_validation": "unverified"}


def observation(doc, offset, velocity):
    result = np.zeros(doc["observation_dim"], dtype=np.float32)
    start = 13 * doc["controlled_index"] + 7
    result[start:start + 3] = velocity
    result[-4:-1] = offset
    return result


@pytest.mark.parametrize("offset,velocity,weights,legacy_action,new_action", [
    ([.2, 0, 0], [.3, 0, 0], None, 1, 2),
    ([0, .2, 0], [0, .3, 0], None, 4, 3),
    ([0, 0, .2], [0, 0, .3], None, 6, 5),
    ([.02, 0, 0], [.04, 0, 0], [1.] * 6 + [0., 0., 0., 1., 0.], 7, 0),
    ([.03, 0, 0], [0, 0, 0], [1.] * 6 + [0., 0., 0., 1., 0.], 0, 2),
    ([.3, 0, 0], [.8, 0, 0], [1.] * 6 + [0., 0., 0., 1., 0.], 1, 7),
    ([.3, 0, 0], [-.08, 0, 0], [1.] * 6 + [0., 0., 0., 1., 0.], 2, 7),
    ([.02, 0, 0], [.1, 0, 0], [0.] * 11, 7, 0),
    ([.2, 0, 0], [0, 0, 0], [0.] * 6 + [.15, 0., 0., 0., .4], 2, 0),
])
def test_same_weights_require_explicit_version_for_changed_controller(offset, velocity, weights, legacy_action, new_action):
    legacy, current = policy(1, weights), policy(2, weights)
    before = deepcopy(legacy), deepcopy(current)
    state = observation(legacy, offset, velocity)
    state_before = state.copy()
    assert policy_action(legacy, state) == legacy_action
    assert policy_action(current, state) == new_action
    assert (legacy, current) == before
    np.testing.assert_array_equal(state, state_before)


@pytest.mark.parametrize("version,profile_version,algorithm_version", [
    (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1), (3, 3, 3), (True, 1, 1),
])
def test_mixed_or_unknown_policy_version_triplets_are_rejected(version, profile_version, algorithm_version):
    doc = policy(1)
    doc.update(schema_version=version, profile=f"spatial-policy-v{profile_version}", algorithm=f"spatial_cem_v{algorithm_version}")
    with pytest.raises(ValueError, match="schema_version|profile/algorithm"):
        policy_action(doc, observation(doc, [.2, 0, 0], [0, 0, 0]))


@pytest.mark.parametrize("version", [1, 2])
def test_supported_policies_bind_scene_actor_and_task_thresholds(version):
    scene = authored_spatial_scene()
    doc = policy(version)
    original = deepcopy(doc)
    validated = validate_spatial_policy(doc, scene)
    validated["weights"][0] = 5
    assert doc == original
    mutations = [
        (dict(doc, scene_sha256="0" * 64), "different 3D scene"),
        (dict(doc, controlled_id="companion", controlled_index=1), "controlled actor"),
        (dict(doc, position_tolerance=.05), "position_tolerance differs"),
        (dict(doc, speed_tolerance=.1), "speed_tolerance differs"),
        (dict(doc, object_ids=[doc["object_ids"][1], doc["object_ids"][0], *doc["object_ids"][2:]], controlled_index=1), "ordering differs"),
    ]
    for altered, match in mutations:
        with pytest.raises(ValueError, match=match):
            validate_spatial_policy(altered, scene)


@pytest.mark.parametrize("version", [1, 2])
def test_controlled_object_velocity_is_not_taken_from_another_object(version):
    doc = policy(version)
    scene = authored_spatial_scene()
    scene["controlled_id"] = "companion"
    for obj in scene["objects"]:
        if obj["role"] != "static":
            obj["role"] = "active" if obj["id"] == "companion" else "passive"
    doc.update(controlled_id="companion", controlled_index=1, scene_sha256=canonical_hash(validate_scene(scene)))
    validate_spatial_policy(doc, scene)
    state = observation(doc, [0, .2, 0], [0, .3, 0])
    state[7:10] = [4, 0, 0]
    assert policy_action(doc, state) == (4 if version == 1 else 3)
