import math

import numpy as np
import pytest

from weft.body_frame import CAMERA_TO_SIM
from weft.spatial_pose import (LIMB_ENDPOINTS, SPATIAL_JOINT_LIMITS, SPATIAL_JOINT_NAMES,
                               build_spatial_reference, rotation_to_zyx, rotation_zyx)


def pose_frame(index=0, torso=None, limb=None, flexion=.8):
    torso = np.eye(3) if torso is None else torso
    limb = np.eye(3) if limb is None else limb
    points = np.zeros((33, 3))
    points[11], points[12] = [-.2, 0, .5], [.2, 0, .5]
    points[23], points[24] = [-.13, 0, 0], [.13, 0, 0]
    for name, (base, middle, tip) in LIMB_ENDPOINTS.items():
        sign = -1 if name.endswith("hip") else 1
        points[middle] = points[base] + limb @ [0, 0, -.3]
        points[tip] = points[middle] + limb @ rotation_zyx([0, 0, sign * flexion]) @ [0, 0, -.25]
    points = points @ torso.T @ CAMERA_TO_SIM
    landmarks = [dict(x=x, y=y, z=z, presence=.9, visibility=.8) for x, y, z in points]
    image = [dict(row, x=.5 + .1 * row["x"], y=.5 + .1 * row["y"]) for row in landmarks]
    return {"id": f"source:frame:{index}", "time_s": index / 10, "status": "tracked",
            "world_landmarks": landmarks, "landmarks": image,
            "joint_confidence": {f"{side}_{joint}": .7 for side in ("left", "right")
                                 for joint in ("shoulder", "elbow", "hip", "knee")}}


def anchor(rotation=None):
    return {"schema_version": 1, "status": "inferred", "rotation": (np.eye(3) if rotation is None else rotation).tolist(),
            "evidence_ids": ["anchor-source"], "reference": []}


@pytest.mark.parametrize("torso_angles,limb_angles", [([0, 0, 0], [.4, .8, -.3]),
                                                       ([.7, -.4, 1.2], [-.5, -.9, 1.1]),
                                                       ([0, 0, math.pi / 2], [0, -math.pi / 2, .7])])
def test_spatial_forward_kinematics_reproduces_rotated_torso_and_bent_limb_endpoints(torso_angles, limb_angles):
    torso, limb = rotation_zyx(torso_angles), rotation_zyx(limb_angles)
    mean = rotation_zyx([-.2, .3, -.6])
    frame = pose_frame(torso=torso, limb=limb)
    reference, audit = build_spatial_reference({"frames": [frame], "source_sha256": "source"}, anchor(mean))
    joints = reference[0]["joints"]
    assert tuple(joints) == SPATIAL_JOINT_NAMES and len(joints) == 19
    recovered_torso = mean @ rotation_zyx([joints[name] for name in SPATIAL_JOINT_NAMES[:3]])
    np.testing.assert_allclose(recovered_torso, torso, atol=1e-8)
    world = np.array([[p[k] for k in ("x", "y", "z")] for p in frame["world_landmarks"]]) @ CAMERA_TO_SIM.T
    for name, (base, middle, end) in LIMB_ENDPOINTS.items():
        recovered = rotation_zyx([joints[name + "_twist"], joints[name + "_lateral"], joints[name]])
        bend = name.rsplit("_", 1)[0] + ("_knee" if name.endswith("hip") else "_elbow")
        sign = -1 if name.endswith("hip") else 1
        np.testing.assert_allclose(recovered_torso @ recovered @ [0, 0, -.3], world[middle] - world[base], atol=1e-8)
        np.testing.assert_allclose(recovered_torso @ recovered @ rotation_zyx([0, 0, sign * joints[bend]]) @ [0, 0, -.25],
                                   world[end] - world[middle], atol=1e-8)
    assert audit["frames"][0]["evidence_id"] == frame["id"]
    assert set(audit["frames"][0]["joint_confidence"]) == set(SPATIAL_JOINT_NAMES)


@pytest.mark.parametrize("pitch", [math.pi / 2, -math.pi / 2])
def test_euler_singularities_preserve_rotation_and_disclose_coupled_coordinates(pitch):
    previous = np.array([.4, pitch - .001, -.6])
    r = rotation_zyx([.4, pitch, -.6])
    angles, singular = rotation_to_zyx(r, previous)
    assert singular and angles[0] == pytest.approx(previous[0])
    np.testing.assert_allclose(rotation_zyx(angles), r, atol=1e-8)
    frame = pose_frame(torso=r)
    _, audit = build_spatial_reference({"frames": [frame]}, anchor())
    row = audit["frames"][0]
    assert "torso" in row["euler_singularities"]
    assert row["joint_confidence"]["torso_twist"] == row["joint_confidence"]["torso_flex"] == 0


def test_euler_branches_are_continuous_across_pi_and_pitch_chart_boundary():
    previous = None
    recovered = []
    for yaw, pitch in zip(np.linspace(3, 3.3, 31), np.linspace(1.4, 1.7, 31)):
        r = rotation_zyx([yaw, pitch, -.2])
        angles, _ = rotation_to_zyx(r, previous)
        np.testing.assert_allclose(rotation_zyx(angles), r, atol=1e-8)
        recovered.append(angles)
        previous = angles
    assert np.max(np.abs(np.diff(recovered, axis=0))) < .02
    assert np.max(np.abs(recovered)) <= math.tau


@pytest.mark.parametrize("limb_angles", [[0, 0, 0], [0, 0, math.pi], [.7, -.9, .3]])
def test_straight_limb_twist_is_assumed_without_fake_confidence(limb_angles):
    frame = pose_frame(limb=rotation_zyx(limb_angles), flexion=0)
    refs, audit = build_spatial_reference({"frames": [frame]}, anchor())
    row = audit["frames"][0]
    assert set(row["assumed_twist"]) == set(LIMB_ENDPOINTS)
    for name in LIMB_ENDPOINTS:
        assert row["joint_confidence"][name + "_twist"] == 0
        result = rotation_zyx([refs[0]["joints"][name + suffix] for suffix in ("_twist", "_lateral", "")])
        np.testing.assert_allclose(result @ [0, 0, -1], rotation_zyx(limb_angles) @ [0, 0, -1], atol=1e-8)


def test_near_straight_clamped_and_excluded_samples_remain_explicit():
    frames = [pose_frame(0, flexion=.01), {"id": "absent", "status": "no_pose"}, pose_frame(2, flexion=2.8)]
    refs, audit = build_spatial_reference({"frames": frames}, anchor())
    assert len(refs) == 2 and audit["excluded_frames"] == [{"evidence_id": "absent", "status": "no_pose"}]
    assert len(audit["frames"][0]["assumed_twist"]) == 4
    assert len(audit["frames"][1]["clamped_flexion"]) == 4
    for row in refs:
        assert all(SPATIAL_JOINT_LIMITS[n][0] <= value <= SPATIAL_JOINT_LIMITS[n][1] for n, value in row["joints"].items())


@pytest.mark.parametrize("index,proximal,distal,coordinate,value", [(15, "left_shoulder", "left_elbow", "x", 1.2),
                                                                  (28, "right_hip", "right_knee", "y", -.2)])
def test_cropped_endpoints_keep_estimates_but_zero_twist_and_distal_confidence(index, proximal, distal, coordinate, value):
    frame = pose_frame()
    before, _ = build_spatial_reference({"frames": [frame]}, anchor())
    frame["landmarks"][index][coordinate] = value
    after, audit = build_spatial_reference({"frames": [frame]}, anchor())
    assert after == before
    assert audit["frames"][0]["joint_confidence"][proximal + "_twist"] == 0
    assert audit["frames"][0]["joint_confidence"][distal] == 0
    assert audit["frames"][0]["joint_confidence"][proximal] == .7


def test_distal_confidence_never_exceeds_original_joint_confidence():
    frame = pose_frame()
    frame["joint_confidence"]["left_elbow"] = .15
    _, audit = build_spatial_reference({"frames": [frame]}, anchor())
    assert audit["frames"][0]["joint_confidence"]["left_elbow"] == .15


def test_missing_image_landmarks_do_not_promote_world_visibility_to_image_confidence():
    frame = pose_frame()
    del frame["landmarks"]
    refs, audit = build_spatial_reference({"frames": [frame]}, anchor())
    assert len(refs) == 1
    for name in ("torso_flex", "left_shoulder_twist", "right_knee"):
        assert audit["frames"][0]["joint_confidence"][name] == 0


@pytest.mark.parametrize("damage", ["time", "geometry", "confidence"])
def test_invalid_usable_samples_fail_instead_of_disappearing(damage):
    frame = pose_frame()
    if damage == "time":
        frame["time_s"] = float("nan")
    elif damage == "geometry":
        frame["world_landmarks"][13] = frame["world_landmarks"][11].copy()
    else:
        frame["joint_confidence"]["left_shoulder"] = float("nan")
    with pytest.raises(ValueError):
        build_spatial_reference({"frames": [frame]}, anchor())
