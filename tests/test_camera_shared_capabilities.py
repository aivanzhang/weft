"""Capabilities retain the actual camera contract and deny unqualified controllers."""

from copy import deepcopy

import pytest

from test_camera_shared_task import analytical_only, setup
from weft.capabilities import describe_capabilities
from weft.camera_shared_capabilities import IDENTITIES
from weft.io import canonical_hash


@pytest.mark.parametrize("roots", [False, True])
def test_camera_actions_pins_and_source_scope_remain_distinct(roots):
    scene, task = setup(root=roots)
    original = deepcopy((scene, task))
    result = describe_capabilities(scene, exercise_task=task, lab_profile="shared_exercise",
                                   reference_available=True, source_video_present=True)
    assert result["scene_profile"] == "shared-world-v6"
    assert result["action_contract"]["dimension"] == (22 if roots else 19)
    assert result["action_contract"]["root_assistance_channels"] == ([19, 20, 21] if roots else [])
    assert result["action_contract"]["object_force_channels"] == []
    assert result["observation_contract"]["dimension"] == 156
    assert result["task_contract"]["identity"] == {**{key: task[key] for key in IDENTITIES},
                                                      "task_sha256": canonical_hash(task)}
    assert result["task_contract"]["success_criteria"]["gate_applicability"]["contact_requirements"] == "not_applicable"
    assert result["controllers"]["learned_policy_available"] is False
    assert result["entities"]["mixed_rigid_articulated"] is False
    assert result["evidence_separation"]["ground_support_verified"] is False
    assert (scene, task) == original


@pytest.mark.parametrize("fault", ["no_task", "policy", "target", "wrong_task", "equipment"])
def test_camera_unsupported_capabilities_cannot_fall_back_to_legacy(fault):
    scene, task = setup()
    options = {"exercise_task": task}
    if fault == "no_task":
        options.clear()
    elif fault == "policy":
        options["learned_policy_available"] = True
    elif fault == "target":
        options["trained_target_id"] = "human"
    elif fault == "wrong_task":
        task["profile"] = "shared-exercise-task-v2"
    else:
        scene["objects"] = [{"id": "unsupported"}]
    with pytest.raises(ValueError):
        describe_capabilities(scene, **options)
