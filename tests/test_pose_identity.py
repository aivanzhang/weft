from copy import deepcopy
import math

import numpy as np
import pytest

from weft.pose_identity import DEFAULT_IDENTITY_CONFIG, IDENTITY_INDEX_MAPPING, repair_pose_identities


def sequence(count=12, *, turn=False, crossed_arms=False):
    rows = []
    for index in range(count):
        points = np.full((33, 3), [.5, .4, 0.])
        angle = index * math.pi / (count - 1) if turn else 0.
        for left, right, width, y in ((11, 12, .10, .3), (13, 14, .14, .42), (15, 16, .16, .54),
                                      (23, 24, .07, .6), (25, 26, .075, .72), (27, 28, .08, .84)):
            for point, sign in ((left, 1), (right, -1)):
                points[point] = [.5 + sign * width * math.cos(angle), y, sign * width * math.sin(angle)]
        if crossed_arms:
            fraction = index / (count - 1)
            points[15, 0] = .66 - .32 * fraction
            points[16, 0] = .34 + .32 * fraction
            points[13, 0] = .64 - .20 * fraction
            points[14, 0] = .36 + .20 * fraction
        points[:, 0] += index * .001
        image = [dict(x=x, y=y, z=z, visibility=.99, presence=.99, marker=i) for i, (x, y, z) in enumerate(points)]
        world = [dict(row, x=2 * (row["x"] - .5), y=2 * (row["y"] - .6)) for row in image]
        rows.append({"id": f"synthetic:{index}", "frame": index * 2, "time_s": index / 15,
                     "status": "tracked", "pose_count": 1, "landmarks": image, "world_landmarks": world,
                     "joints": {"left_elbow": .5}, "joint_confidence": {"left_elbow": .99},
                     "lengths": {"upper_arm": .3}, "out_of_plane_fraction": {"left_shoulder": .1},
                     "usable_required_landmarks": 12, "inference_seconds": .001})
    return rows


def swap(rows, indices):
    for index in indices:
        for key in ("landmarks", "world_landmarks"):
            rows[index][key] = [rows[index][key][source] for source in IDENTITY_INDEX_MAPPING]


@pytest.mark.parametrize("indices", [[4], [4, 5], [4, 5, 6]])
def test_short_swaps_repair_both_arrays_without_mutating_or_losing_original_fields(indices):
    pristine = sequence()
    observed = deepcopy(pristine)
    swap(observed, indices)
    before = deepcopy(observed)
    corrected, audit = repair_pose_identities(observed)
    assert observed == before
    assert audit["repaired_evidence_ids"] == [observed[i]["id"] for i in indices]
    assert audit["config"] == DEFAULT_IDENTITY_CONFIG
    for i in indices:
        assert corrected[i]["landmarks"] == pristine[i]["landmarks"]
        assert corrected[i]["world_landmarks"] == pristine[i]["world_landmarks"]
        assert corrected[i]["identity_raw"] == observed[i]
        assert corrected[i]["id"] == observed[i]["id"] and corrected[i]["time_s"] == observed[i]["time_s"]
        assert corrected[i]["identity_recompute_required"]
        assert all(key not in corrected[i] for key in ("joints", "lengths", "joint_confidence", "out_of_plane_fraction"))
    decision = next(row for row in audit["decisions"] if row["status"] == "repaired")
    assert len(decision["support_evidence_ids"]) == 4
    assert all(row["swapped_torso_error"] < 1e-10 for row in decision["comparisons"])


@pytest.mark.parametrize("kwargs", [{}, {"turn": True}, {"crossed_arms": True}])
def test_smooth_motion_real_body_turn_and_crossing_arms_are_not_relabelled(kwargs):
    observed = sequence(24, **kwargs)
    corrected, audit = repair_pose_identities(observed)
    assert corrected == observed and audit["repaired_evidence_ids"] == []


def test_initial_global_swap_and_permanent_turn_assignment_are_not_claimed_verified():
    rows = sequence()
    swap(rows, range(len(rows)))
    corrected, audit = repair_pose_identities(rows)
    assert corrected == rows and not audit["repaired_evidence_ids"]
    assert "unverified" in audit["scope"]
    rows = sequence()
    swap(rows, range(4, len(rows)))
    assert repair_pose_identities(rows)[0] == rows


@pytest.mark.parametrize("damage", ["gap", "long_island", "low_confidence", "ambiguous_torso", "multiple", "cropped", "no_after_anchors"])
def test_unsupported_identity_changes_abstain(damage):
    rows = sequence()
    swap(rows, [4, 5, 6, 7] if damage == "long_island" else [4, 5])
    if damage == "gap":
        for row in rows[4:]:
            row["time_s"] += 1
    elif damage == "low_confidence":
        rows[4]["landmarks"][11]["visibility"] = .4
    elif damage == "ambiguous_torso":
        rows[4]["landmarks"][12]["x"] = rows[4]["landmarks"][11]["x"]
    elif damage == "multiple":
        rows[4].update(pose_count=2, status="multiple_people", poses=[{"evidence": 1}, {"evidence": 2}])
    elif damage == "cropped":
        rows[4]["landmarks"][11]["x"] = 1.2
    elif damage == "no_after_anchors":
        rows = rows[:7]
    corrected, audit = repair_pose_identities(rows)
    assert corrected == rows and not audit["repaired_evidence_ids"]
    if damage == "gap":
        assert any("temporal_gap" in row["reasons"] for row in audit["decisions"])


def test_arm_only_identity_swap_does_not_override_consistent_torso():
    rows = sequence()
    for index in (4, 5):
        for key in ("landmarks", "world_landmarks"):
            for left, right in ((13, 14), (15, 16)):
                rows[index][key][left], rows[index][key][right] = rows[index][key][right], rows[index][key][left]
    assert repair_pose_identities(rows)[0] == rows


def test_all_33_landmark_mapping_is_an_involution_with_only_nose_fixed():
    assert len(IDENTITY_INDEX_MAPPING) == 33
    assert [i for i, j in enumerate(IDENTITY_INDEX_MAPPING) if i == j] == [0]
    assert [IDENTITY_INDEX_MAPPING[j] for j in IDENTITY_INDEX_MAPPING] == list(range(33))


@pytest.mark.parametrize("config", [{"max_span_frames": 4}, {"max_adjacent_gap_seconds": 1},
                                    {"maximum_error_ratio": float("nan")}, {"new_threshold": 1}])
def test_configuration_cannot_silently_expand_the_repair_envelope(config):
    with pytest.raises(ValueError):
        repair_pose_identities(sequence(), config=config)


def test_nonmonotonic_times_and_repairing_already_corrected_frames_fail():
    rows = sequence()
    rows[3]["time_s"] = rows[2]["time_s"]
    with pytest.raises(ValueError):
        repair_pose_identities(rows)
    rows = sequence()
    swap(rows, [4])
    corrected, _ = repair_pose_identities(rows)
    with pytest.raises(ValueError, match="previously corrected"):
        repair_pose_identities(corrected)
