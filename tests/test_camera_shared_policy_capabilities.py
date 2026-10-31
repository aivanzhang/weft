"""Analytical capability joins; native task admission is replaced only by its existing fixture."""

from copy import deepcopy

import pytest

from test_camera_shared_task import analytical_only, setup
from test_camera_shared_policy_receipt import package
from weft.capabilities import describe_capabilities
from weft.camera_shared_policy_receipt import admission_view


@pytest.mark.parametrize("roots", [False, True])
def test_admitted_external_policy_preserves_original_control_and_source_scope(roots):
    scene, task = setup(root=roots)
    admission = package(scene=scene, task=task)["admission"]
    original = deepcopy((scene, task, admission))
    result = describe_capabilities(scene, exercise_task=task, lab_profile="shared_exercise",
        learned_policy_available=True, trained_target_id="human", camera_policy_admission=admission)
    assert result["controllers"]["camera_policy_admission"] == admission_view(admission)
    assert result["controllers"]["learned_policy_available"] is True
    assert result["action_contract"]["dimension"] == (22 if roots else 19)
    assert result["action_contract"]["root_assistance_channels"] == ([19, 20, 21] if roots else [])
    assert result["action_contract"]["object_force_channels"] == []
    assert result["evidence_separation"]["source_learning_qualified"] is False
    assert result["evidence_separation"]["external_policy_attached"] is True
    assert result["task_contract"]["source_scope"]["learning"] == "unqualified"
    assert (scene, task, admission) == original and scene["learned_policy_available"] is False


@pytest.mark.parametrize("fault", ["boolean_only", "wrong_target", "hidden_policy", "other_task", "changed_actor"])
def test_policy_claims_without_matching_admission_are_rejected(fault):
    scene, task = setup(root=True)
    admission = package(scene=scene, task=task)["admission"]
    options = {"exercise_task": task, "learned_policy_available": True, "trained_target_id": "human",
               "camera_policy_admission": admission}
    if fault == "boolean_only":
        options["camera_policy_admission"] = None
    elif fault == "wrong_target":
        options["trained_target_id"] = "body"
    elif fault == "hidden_policy":
        options["learned_policy_available"] = False
    elif fault == "other_task":
        admission["task_sha256"] = "0" * 64
    else:
        admission["execution_runtime_sha256"]["camera_shared_policy.py"] = "0" * 64
    with pytest.raises(ValueError):
        describe_capabilities(scene, **options)
