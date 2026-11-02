"""Source-only selection must not infer anatomy from visibility or consume physical scores."""

from copy import deepcopy
import math

import pytest

from weft.pose_mode_recovery import CONTRACT, SEGMENTS, propose_pose_mode_recovery


def pair(samples=9, *, corrupted=True):
    positions = {11: (.4, .25), 12: (.6, .25), 13: (.32, .4), 14: (.68, .4),
                 15: (.25, .5), 16: (.75, .5), 23: (.4, .55), 24: (.6, .55),
                 25: (.4, .72), 26: (.6, .72), 27: (.4, .9), 28: (.6, .9)}
    model = {"sha256": "b" * 64, "bytes": 99, "model_version": "fixture", "mediapipe_version": "fixture",
             "delegate": "CPU", "network_inference": False, "running_mode": "VIDEO", "temporal_tracking": True,
             "maximum_pose_hypotheses": 1, "person_count_assumption": "single_person_unverified"}
    primary = {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": "a" * 64,
               "width": 640, "height": 480, "fps": 30, "decoded_frames": 2*samples-1,
               "frame_count": 2*samples-1, "inference_stride": 2, "model": model,
               "pose_identity": {"mode": "raw"}, "frames": []}
    for index in range(samples):
        points = [{"x": positions.get(joint, (.5, .5))[0], "y": positions.get(joint, (.5, .5))[1],
                   "z": 0., "visibility": 1., "presence": 1.} for joint in range(33)]
        primary["frames"].append({"frame": index*2, "id": f"{'a'*64}:frame:{index*2}",
            "time_s": index*2/30, "decoder_timestamp_s": index*2/30+.003, "pose_count": 1,
            "status": "tracked", "landmarks": points, "world_landmarks": deepcopy(points)})
    alternative = deepcopy(primary)
    alternative["model"].update(running_mode="IMAGE", temporal_tracking=False, maximum_pose_hypotheses=2)
    alternative["model"].pop("person_count_assumption")
    if corrupted:
        collapse(primary["frames"][3])
    return primary, alternative


def collapse(row):
    row["landmarks"][26]["y"] = .57
    row["world_landmarks"][26]["y"] = .57


def test_joint_rigidity_and_image_outlier_selects_actual_retained_alternative_without_mutation():
    primary, alternative = pair()
    original = deepcopy((primary, alternative))
    result = propose_pose_mode_recovery(primary, alternative)
    assert result["accepted_frames"] == [6] and result["simulation_ready"] is False
    assert (primary, alternative) == original
    assert [row["frame"] for row in result["rows"]] == list(range(0, 17, 2))
    assert all(set(row["segments"]) == set(SEGMENTS) for row in result["rows"])
    chosen = result["rows"][3]
    assert chosen["anchor_frames"] == [4, 8]
    assert chosen["segments"]["right_thigh"]["candidate_image_error_torso_ratio"] == pytest.approx(0)
    assert chosen["segments"]["right_thigh"]["candidate_length_to_expected"] == pytest.approx(1)
    assert chosen["original_sha256"] != chosen["candidate_sha256"]


@pytest.mark.parametrize("kind", ["world_only", "image_only", "rigid_translation"])
def test_single_domain_noise_or_whole_person_motion_is_not_a_joint_recovery_trigger(kind):
    primary, alternative = pair(corrupted=False)
    if kind == "rigid_translation":
        for index, row in enumerate(primary["frames"]):
            for field in ("landmarks", "world_landmarks"):
                for point in row[field]:
                    point["x"] += .02 * index
    else:
        field = "world_landmarks" if kind == "world_only" else "landmarks"
        primary["frames"][3][field][26]["y"] = .57
    assert propose_pose_mode_recovery(primary, alternative)["accepted_frames"] == []


@pytest.mark.parametrize("kind", ["competing", "missing", "lost_wrist", "bad_length", "bad_image", "moved_root"])
def test_unsupported_alternative_remains_rejected(kind):
    primary, alternative = pair()
    candidate = alternative["frames"][3]
    if kind == "competing":
        candidate.update(pose_count=2, status="multiple_people")
    elif kind == "missing":
        candidate.update(pose_count=0, status="no_pose", landmarks=[], world_landmarks=[])
    elif kind == "lost_wrist":
        candidate["landmarks"][15]["visibility"] = .4
    elif kind == "bad_length":
        candidate["world_landmarks"][26]["y"] = .57
    elif kind == "bad_image":
        candidate["landmarks"][26]["y"] = .57
    else:
        for joint in (23, 24):
            candidate["landmarks"][joint]["x"] += .2
    result = propose_pose_mode_recovery(primary, alternative)
    assert result["rows"][3]["eligible"] and result["rows"][3]["reasons"]
    assert result["accepted_frames"] == []


@pytest.mark.parametrize("kind", ["hidden_target", "hidden_neighbor", "missing_neighbor", "unstable_neighbor", "long_interval"])
def test_unknown_or_inconsistent_original_context_is_not_silently_interpolated(kind):
    primary, alternative = pair()
    if kind == "hidden_target":
        primary["frames"][3]["landmarks"][26]["visibility"] = .4
    elif kind == "hidden_neighbor":
        primary["frames"][2]["landmarks"][26]["visibility"] = .4
    elif kind == "missing_neighbor":
        primary["frames"][2].update(pose_count=0, status="no_pose", landmarks=[], world_landmarks=[])
    elif kind == "unstable_neighbor":
        primary["frames"][2]["world_landmarks"][26]["y"] = .62
    else:
        for evidence in (primary, alternative):
            evidence["fps"] = 10
            for row in evidence["frames"]:
                row.update(time_s=row["frame"]/10, decoder_timestamp_s=row["frame"]/10+.003)
    result = propose_pose_mode_recovery(primary, alternative)
    assert result["accepted_frames"] == []
    assert len(result["rows"]) == 9 and all(set(row["segments"]) == set(SEGMENTS) for row in result["rows"])


@pytest.mark.parametrize("kind", ["source", "dimensions", "model", "runtime", "capacity", "mode", "native_clock",
                                  "nominal_clock", "missing_slot", "duplicate_slot", "modified", "identity"])
def test_wrong_sources_clocks_models_or_modified_hypotheses_fail_before_selection(kind):
    primary, alternative = pair()
    if kind == "source":
        alternative["source_sha256"] = "c"*64
    elif kind == "dimensions":
        alternative["width"] = 800
    elif kind == "model":
        alternative["model"]["sha256"] = "c"*64
    elif kind == "runtime":
        alternative["model"]["mediapipe_version"] = "different"
    elif kind == "capacity":
        primary["model"]["maximum_pose_hypotheses"] = True
    elif kind == "mode":
        alternative["model"]["running_mode"] = "VIDEO"
    elif kind == "native_clock":
        alternative["frames"][3]["decoder_timestamp_s"] += .003
    elif kind == "nominal_clock":
        alternative["frames"][3]["time_s"] += .1
    elif kind == "missing_slot":
        alternative["frames"].pop()
    elif kind == "duplicate_slot":
        alternative["frames"][3] = deepcopy(alternative["frames"][2])
    elif kind == "modified":
        primary["frames"][3]["mode_refinement"] = {}
    else:
        primary["pose_identity"]["mode"] = "temporal"
    with pytest.raises(ValueError):
        propose_pose_mode_recovery(primary, alternative)


def test_excess_candidates_preserve_all_decisions_without_partial_acceptance():
    primary, alternative = pair(103, corrupted=False)
    for index in range(1, 102, 3):
        collapse(primary["frames"][index])
    result = propose_pose_mode_recovery(primary, alternative)
    assert sum(row["eligible"] for row in result["rows"]) == 34
    assert result["bound_exceeded"] and not result["accepted_frames"]
    assert len(result["rows"]) == 103
    assert all("candidate_bound_exceeded" in row["reasons"] for row in result["rows"] if row["eligible"])


@pytest.mark.parametrize("field", ["x", "visibility"])
def test_nonfinite_source_data_is_not_treated_as_invisible(field):
    primary, alternative = pair()
    primary["frames"][3]["landmarks"][26][field] = math.nan
    with pytest.raises(ValueError, match="finite"):
        propose_pose_mode_recovery(primary, alternative)


def test_recipe_is_source_only_and_does_not_accept_contact_or_score_arguments():
    primary, alternative = pair()
    with pytest.raises(TypeError):
        propose_pose_mode_recovery(primary, alternative, floor_limit=.1)
    with pytest.raises(TypeError):
        propose_pose_mode_recovery(primary, alternative, scored_frames=[6])
    with pytest.raises(TypeError):
        CONTRACT["collapse_ratio"] = .9


@pytest.mark.parametrize("kind", ["identity_type", "crop_marker", "mode_marker", "crop_raw", "mode_raw", "count", "unaudited_count"])
def test_malformed_original_metadata_or_orphaned_derivation_is_rejected(kind):
    primary, alternative = pair()
    if kind == "identity_type":
        primary["pose_identity"] = None
    elif kind in {"crop_marker", "mode_marker"}:
        primary["pose_reinference" if kind == "crop_marker" else "pose_mode_recovery"] = {}
    elif kind in {"crop_raw", "mode_raw"}:
        primary["frames"][3]["image_refinement_raw" if kind == "crop_raw" else "mode_refinement_raw"] = {}
    else:
        for document in (primary, alternative):
            document["frame_count"] = None if kind == "count" else 19
    with pytest.raises(ValueError):
        propose_pose_mode_recovery(primary, alternative)
