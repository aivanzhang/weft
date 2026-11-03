"""Role resolution must cover common rig conventions and refuse rather than guess."""

import math

import pytest

from avatar_assets.rig_mapping import RigMappingError, resolve_body_roles

CONVENTIONS = {
    "weft_creature": ["root", "spine", "chest", "arm_l_upper", "arm_l_lower", "hand_l",
                      "arm_r_upper", "arm_r_lower", "hand_r", "leg_l_upper", "leg_l_lower",
                      "foot_l", "leg_r_upper", "leg_r_lower", "foot_r"],
    "anatomical": ["body_root", "spine", "chest", "shoulder_L", "elbow_L", "wrist_L",
                   "shoulder_R", "elbow_R", "wrist_R", "hip_L", "knee_L", "ankle_L",
                   "hip_R", "knee_R", "ankle_R"],
    "dcc_common": ["hips", "spine", "chest", "upperarm_l", "forearm_l", "hand_l",
                   "upperarm_r", "forearm_r", "hand_r", "thigh_l", "shin_l", "foot_l",
                   "thigh_r", "shin_r", "foot_r"],
    "mixamo": ["Hips", "Spine", "Spine01", "Spine02", "LeftShoulder", "LeftArm", "LeftForeArm",
               "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
               "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot"],
}
EXPECTED_DISTAL = {"weft_creature": ("arm_l_lower", "leg_l_lower"),
                   "anatomical": ("elbow_L", "knee_L"),
                   "dcc_common": ("forearm_l", "shin_l"),
                   "mixamo": ("LeftForeArm", "LeftLeg")}


def skeleton(names):
    """Build {node: name} plus parents for a plain bipedal hierarchy."""
    bones = dict(enumerate(names))
    index = {name: node for node, name in bones.items()}
    parents = {}
    torso = next((name for name in names if name.lower() in ("chest", "spine02", "torso")), names[1])
    root = names[0]
    spine = next((name for name in names if name.lower() == "spine"), root)
    parents[index[spine]] = index[root]
    for name in names:
        lowered = name.lower()
        if name in (root, spine):
            continue
        if lowered in ("chest", "spine01", "spine02"):
            parents[index[name]] = index[spine]
        elif any(term in lowered for term in ("arm", "shoulder", "elbow", "hand", "wrist")):
            parents[index[name]] = index[torso]
        else:
            parents[index[name]] = index[root]
    # Chain each distal bone under its proximal partner.
    for upper, lower in (("arm_l_upper", "arm_l_lower"), ("arm_r_upper", "arm_r_lower"),
                         ("leg_l_upper", "leg_l_lower"), ("leg_r_upper", "leg_r_lower"),
                         ("shoulder_L", "elbow_L"), ("shoulder_R", "elbow_R"),
                         ("hip_L", "knee_L"), ("hip_R", "knee_R"),
                         ("upperarm_l", "forearm_l"), ("upperarm_r", "forearm_r"),
                         ("thigh_l", "shin_l"), ("thigh_r", "shin_r"),
                         ("LeftShoulder", "LeftArm"), ("RightShoulder", "RightArm"),
                         ("LeftArm", "LeftForeArm"), ("RightArm", "RightForeArm"),
                         ("LeftUpLeg", "LeftLeg"), ("RightUpLeg", "RightLeg")):
        if upper in index and lower in index:
            parents[index[lower]] = index[upper]
    return bones, parents


@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
def test_common_rig_conventions_resolve_to_the_same_semantic_roles(convention):
    bones, parents = skeleton(CONVENTIONS[convention])
    mapping = resolve_body_roles(bones, parents=parents, root_name="WeftAvatarRoot")
    assert set(mapping) == {"mobile_root", "torso", "left_shoulder_body", "left_elbow_body",
                            "left_hip_body", "left_knee_body", "right_shoulder_body",
                            "right_elbow_body", "right_hip_body", "right_knee_body"}
    elbow, knee = EXPECTED_DISTAL[convention]
    assert mapping["left_elbow_body"] == elbow and mapping["left_knee_body"] == knee
    assert mapping["mobile_root"] == "WeftAvatarRoot"
    assert len(set(mapping.values())) == len(mapping)
    assert "left" in mapping["left_shoulder_body"].lower() or "_l" in mapping["left_shoulder_body"].lower()


def test_mixamo_shoulder_role_uses_the_humerus_not_the_clavicle():
    bones, parents = skeleton(CONVENTIONS["mixamo"])
    mapping = resolve_body_roles(bones, parents=parents, root_name="WeftAvatarRoot")
    # LeftShoulder is the clavicle; the bone that actually drives the elbow is LeftArm.
    assert mapping["left_shoulder_body"] == "LeftArm"
    assert mapping["left_hip_body"] == "LeftUpLeg" and mapping["left_knee_body"] == "LeftLeg"


def test_rig_without_elbow_or_knee_joints_is_refused():
    names = ["Root", "Torso", "Head", "LeftArm", "LeftHand", "RightArm", "RightHand",
             "LeftLeg", "LeftFoot", "RightLeg", "RightFoot"]
    bones, parents = skeleton(names)
    with pytest.raises(RigMappingError, match="elbow|hip"):
        resolve_body_roles(bones, parents=parents, root_name="WeftAvatarRoot")


def test_structure_contradicting_its_names_is_refused():
    bones, parents = skeleton(CONVENTIONS["weft_creature"])
    index = {name: node for node, name in bones.items()}
    # Detach the elbow from its arm so the names no longer describe the skeleton.
    parents[index["arm_l_lower"]] = index["root"]
    with pytest.raises(RigMappingError, match="not below"):
        resolve_body_roles(bones, parents=parents, root_name="WeftAvatarRoot")


def test_sideless_and_duplicate_rigs_are_refused():
    bones, parents = skeleton(CONVENTIONS["weft_creature"])
    with pytest.raises(RigMappingError, match="unique"):
        resolve_body_roles({0: "chest", 1: "chest"}, parents={}, root_name="r")
    sideless = {index: name.replace("_l_", "_x_").replace("_r_", "_y_")
                for index, name in bones.items()}
    with pytest.raises(RigMappingError, match="no bone matched"):
        resolve_body_roles(sideless, parents=parents, root_name="WeftAvatarRoot")


def test_orientation_and_scale_are_derived_from_the_rig_not_assumed():
    """A Y-up rig must be turned to Weft's Z-up, and height measured along its own up."""
    import numpy as np
    from avatar_assets.weft_autopack import derive_orientation

    class FakeGLB:
        doc = {"nodes": [
            {"name": "root", "children": [1, 3, 4], "translation": [0, 0, 0]},
            {"name": "chest", "children": [2], "translation": [0, 1.0, 0]},
            {"name": "head", "translation": [0, 0.3, 0]},
            {"name": "thigh_l", "translation": [0.1, 0, 0]},
            {"name": "thigh_r", "translation": [-0.1, 0, 0]},
        ]}

    mapping = {"torso": "chest", "left_hip_body": "thigh_l", "right_hip_body": "thigh_r"}
    quaternion, up = derive_orientation(FakeGLB(), mapping)
    np.testing.assert_allclose(up, [0, 1, 0], atol=1e-9)
    # +90 degrees about X carries +Y onto +Z.
    np.testing.assert_allclose(quaternion, [math.sin(math.pi / 4), 0, 0, math.cos(math.pi / 4)], atol=1e-9)
    assert abs(sum(v * v for v in quaternion) - 1) < 1e-9


def test_already_z_up_rig_needs_no_turn():
    import numpy as np
    from avatar_assets.weft_autopack import derive_orientation

    class FakeGLB:
        doc = {"nodes": [
            {"name": "root", "children": [1, 2, 3], "translation": [0, 0, 0]},
            {"name": "chest", "translation": [0, 0, 1.0]},
            {"name": "thigh_l", "translation": [0.1, 0, 0]},
            {"name": "thigh_r", "translation": [-0.1, 0, 0]},
        ]}

    quaternion, up = derive_orientation(FakeGLB(), {"torso": "chest", "left_hip_body": "thigh_l",
                                                    "right_hip_body": "thigh_r"})
    np.testing.assert_allclose(up, [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(quaternion, [0, 0, 0, 1], atol=1e-9)
