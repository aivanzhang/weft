import pytest

from weft.errors import UnsupportedVideo
from weft.human import JOINT_NAMES
from weft.human_task import propose_joint_task


def evidence(moving="left_elbow", confidence=1):
    return {"reference": [{"joints": {name: .2 + (i / 20 if name == moving else 0) for name in JOINT_NAMES},
                            "joint_confidence": {name: confidence if name == moving else 1 for name in JOINT_NAMES}}
                           for i in range(20)]}


def test_pushup_motion_selects_elbow_not_hardcoded_knee():
    proposal = propose_joint_task(evidence())
    assert proposal["selected_joint"] == "left_elbow"
    assert "Authored" in proposal["provenance"]


def test_uncertain_moving_joint_cannot_define_task():
    with pytest.raises(UnsupportedVideo, match="reliable"):
        propose_joint_task(evidence(confidence=.1))


def test_single_outlier_does_not_create_meaningful_motion():
    data = evidence()
    for row in data["reference"]:
        row["joints"]["left_elbow"] = .2
    data["reference"][0]["joints"]["left_elbow"] = 2
    with pytest.raises(UnsupportedVideo):
        propose_joint_task(data)


def test_motion_outside_model_limits_cannot_win_selection():
    data = evidence("left_knee")
    for index, row in enumerate(data["reference"]):
        row["joints"]["left_elbow"] = 3 + index
    assert propose_joint_task(data)["selected_joint"] == "left_knee"
