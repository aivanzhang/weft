"""Locality pruning drops only anatomically distant influences and never empties a vertex."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from weight_locality import POLICY, prune_influences, segment_distance  # noqa: E402

SEGMENTS = {
    "pelvis": ((0.0, 0.0, 0.5), (0.0, 0.0, 0.6)),
    "chest": ((0.0, 0.0, 0.7), (0.0, 0.0, 0.9)),
    "arm_r_lower": ((0.25, 0.0, 0.8), (0.45, 0.0, 0.8)),
    "hand_r": ((0.45, 0.0, 0.8), (0.52, 0.0, 0.8)),
}
DIAMETER = 1.0


def test_cross_body_blend_loses_the_distal_influence():
    hip_vertex = (0.02, 0.0, 0.55)
    kept, pruned = prune_influences(hip_vertex, {"pelvis": 0.7, "arm_r_lower": 0.3},
                                    SEGMENTS, DIAMETER)
    assert kept == {"pelvis": 0.7} and pruned == 1


def test_legitimate_joint_blend_survives():
    wrist_vertex = (0.44, 0.0, 0.8)
    weights = {"arm_r_lower": 0.6, "hand_r": 0.4}
    kept, pruned = prune_influences(wrist_vertex, weights, SEGMENTS, DIAMETER)
    assert kept == weights and pruned == 0


def test_far_bones_prune_even_when_several():
    elbow_vertex = (0.26, 0.0, 0.8)
    weights = {"arm_r_lower": 0.6, "hand_r": 0.1, "chest": 0.3}
    kept, pruned = prune_influences(elbow_vertex, weights, SEGMENTS, DIAMETER)
    assert kept == {"arm_r_lower": 0.6} and pruned == 2


def test_distal_tube_is_tighter_than_proximal():
    torso_vertex = (0.0, 0.0, 0.78)
    kept, pruned = prune_influences(torso_vertex, {"chest": 0.8, "hand_r": 0.2},
                                    SEGMENTS, DIAMETER)
    assert kept == {"chest": 0.8} and pruned == 1


def test_sole_influence_always_survives_and_empty_stays_empty():
    far_vertex = (0.0, 0.0, 0.05)
    kept, pruned = prune_influences(far_vertex, {"hand_r": 1.0}, SEGMENTS, DIAMETER)
    assert kept == {"hand_r": 1.0} and pruned == 0
    kept, pruned = prune_influences(far_vertex, {}, SEGMENTS, DIAMETER)
    assert kept == {} and pruned == 0


def test_policy_is_closed_and_versioned():
    assert POLICY["profile"] == "authored-weight-locality-v1"
    assert POLICY["relative_factor"] > POLICY["distal_factor"] > 1
