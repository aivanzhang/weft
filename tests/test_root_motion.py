"""Synthetic image coordinates test assumptions, not recovered physical accuracy."""

from copy import deepcopy
import time

import numpy as np
import pytest

from weft.root_motion import ROOT_MOTION_CONTRACT, build_root_reference


HASH = "a" * 64


def evidence(count=10, *, fps=20, origin=0, dx=0., dy=0.):
    rows = []
    for index in range(origin, origin+count):
        landmarks = [{"x": .5, "y": .5, "z": 99., "visibility": .9, "presence": .8} for _ in range(33)]
        for key, x, y in ((11, .4, .25), (12, .6, .25), (23, .44, .5), (24, .56, .5)):
            landmarks[key].update(x=x+dx*(index-origin)/640, y=y+dy*(index-origin)/480)
        rows.append({"id": f"{HASH}:frame:{index}", "frame": index, "time_s": index/fps,
                     "decoder_timestamp_s": index/fps+.001, "status": "tracked", "landmarks": landmarks})
    return {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": HASH,
            "width": 640, "height": 480, "fps": fps, "frame_count": max(10, origin+count),
            "frames": rows, "reference": [{"evidence_id": row["id"], "time_s": row["time_s"]} for row in rows],
            "origin_time_s": origin/fps, "lengths": {"torso": .6}}


def test_image_translation_maps_right_and_up_with_no_depth():
    source = evidence(dx=2, dy=-3)
    preserved = deepcopy(source)
    result = build_root_reference(source)
    assert result["status"] == "ready"
    assert result["scale"]["model_units_per_pixel"] == pytest.approx(.005)
    for index, row in enumerate(result["reference"]):
        assert row["image_displacement_px"] == pytest.approx([index*2, -index*3])
        assert row["translation"] == pytest.approx([index*.01, 0, index*.015])
        assert row["depth"] == {"status": "unobservable", "observed_value": None, "authored_model_offset": 0.}
        assert row["evidence_id"] == source["reference"][index]["evidence_id"]
        assert row["decoder_timestamp_s"] == source["frames"][index]["decoder_timestamp_s"]
        assert row["confidence"] == .8
    assert source == preserved


def test_origin_is_first_reference_not_source_zero_or_a_later_convenient_pose():
    source = evidence(origin=20, dx=3)
    result = build_root_reference(source)
    assert result["origin"]["time_s"] == 1.
    assert result["reference"][0]["translation"] == [0., 0., 0.]
    source["frames"][0]["landmarks"][23]["visibility"] = .1
    result = build_root_reference(source)
    assert result["status"] == "insufficient_reference"
    assert all(row["translation"] is None for row in result["reference"])
    assert result["origin"]["evidence_id"] == f"{HASH}:frame:20"


def test_robust_fixed_scale_does_not_follow_torso_projection_changes():
    source = evidence(dx=4)
    for index in (4, 8):
        for key in (11, 12):
            source["frames"][index]["landmarks"][key]["y"] = .15
    result = build_root_reference(source)
    assert result["status"] == "ready"
    assert result["scale"]["torso_image_median_px"] == 120
    assert result["scale"]["torso_image_p90_px"] > 120
    assert [row["translation"][0] for row in result["reference"]] == pytest.approx([.02*i for i in range(10)])
    assert "Weak perspective" in result["assumptions"][1]


def test_camera_motion_and_person_motion_are_indistinguishable_in_this_reference():
    camera_shift = evidence(dx=2)
    body_shift = deepcopy(camera_shift)
    camera_result = build_root_reference(camera_shift)
    body_result = build_root_reference(body_shift)
    assert camera_result["reference"] == body_result["reference"]
    assert camera_result["physical_translation_verified"] is False
    assert camera_result["contact_inference"] == "none"
    assert "not verified or motion-corrected" in camera_result["assumptions"][0]
    assert "Camera up is assumed gravity-up" in camera_result["assumptions"][2]


def test_short_missing_reference_is_explicitly_derived_not_observed():
    source = evidence(dx=2)
    source["frames"][4]["landmarks"][23]["visibility"] = .2
    result = build_root_reference(source)
    assert result["status"] == "ready"
    row = result["reference"][4]
    assert row["status"] == "interpolated_reference_only"
    assert row["observed_status"] == "low_confidence"
    assert row["confidence"] == .2
    assert row["pelvis_image_px"] is None and row["image_displacement_px"] is None
    assert row["translation"] == pytest.approx([.04, 0, 0])
    assert row["derivation_evidence_ids"] == [source["frames"][i]["id"] for i in (3, 5)]
    assert result["observations"][4]["pelvis_image_px"] is None
    assert result["quality"]["observed_reference_fraction"] == .9


@pytest.mark.parametrize("index,value", [(11, 0.), (12, 1.), (23, -.01), (24, 1.01)])
def test_cropped_shoulders_or_hips_never_publish_observed_root(index, value):
    source = evidence()
    source["frames"][4]["landmarks"][index]["x"] = value
    result = build_root_reference(source)
    assert result["observations"][4]["status"] == "cropped_landmark"
    assert result["observations"][4]["pelvis_image_px"] is None
    assert result["reference"][4]["status"] == "interpolated_reference_only"
    assert result["reference"][4]["image_displacement_px"] is None


def test_long_missing_run_and_missing_endpoint_fail_readiness():
    source = evidence()
    for index in (3, 4, 5):
        source["frames"][index]["landmarks"] = []
    result = build_root_reference(source)
    assert result["status"] == "insufficient_reference"
    assert all(result["reference"][index]["translation"] is None for index in (3, 4, 5))
    source = evidence()
    source["frames"][-1]["landmarks"] = []
    result = build_root_reference(source)
    assert result["status"] == "insufficient_reference"
    assert result["reference"][-1]["translation"] is None


def test_missing_raw_frames_omitted_by_joint_reference_still_leave_a_time_gap():
    source = evidence()
    source["reference"] = [source["reference"][i] for i in (0, 1, 7, 8, 9)]
    for index in range(2, 7):
        source["frames"][index]["status"] = "low_confidence"
    result = build_root_reference(source)
    assert result["status"] == "insufficient_reference"
    assert len(result["reference"]) == 5
    assert result["quality"]["long_reference_intervals"][0]["seconds"] == pytest.approx(.3)
    assert result["observations"][2]["confidence"] == .8
    assert result["observations"][2]["pelvis_image_px"] is None


def test_raw_identity_landmarks_are_preferred_without_overwriting_evidence():
    source = evidence(dx=2)
    raw = deepcopy(source["frames"][3])
    source["frames"][3]["identity_raw"] = raw
    for key in (11, 12, 23, 24):
        source["frames"][3]["landmarks"][key]["x"] += .1
    result = build_root_reference(source)
    assert result["observations"][3]["landmark_source"] == "identity_raw.landmarks"
    assert result["reference"][3]["translation"] == pytest.approx([.03, 0, 0])


def test_model_length_override_is_fixed_authored_scale():
    result = build_root_reference(evidence(dx=2), torso_length=.3)
    assert result["scale"]["model_torso_length"] == .3
    assert result["reference"][2]["translation"][0] == pytest.approx(.01)


@pytest.mark.parametrize("field,value", [("width", 4097), ("height", 2161), ("fps", 0),
                                        ("source_sha256", "unknown"), ("schema_version", True)])
def test_invalid_top_level_bounds_rejected(field, value):
    source = evidence()
    source[field] = value
    with pytest.raises(ValueError):
        build_root_reference(source)


def test_dangling_time_duplicate_and_nonfinite_references_rejected():
    for change in ("id", "time", "duplicate", "nan"):
        source = evidence()
        if change == "id":
            source["reference"][3]["evidence_id"] = "unrelated"
        elif change == "time":
            source["reference"][3]["time_s"] += .001
        elif change == "duplicate":
            source["frames"][3] = deepcopy(source["frames"][2])
        else:
            source["frames"][3]["time_s"] = float("nan")
        with pytest.raises(ValueError):
            build_root_reference(source)


def test_degenerate_or_invalid_landmarks_are_unknown_not_zero_positions():
    source = evidence()
    source["frames"][3]["landmarks"][11]["x"] = float("nan")
    for key in (11, 12):
        source["frames"][5]["landmarks"][key]["y"] = .5
    result = build_root_reference(source)
    assert result["observations"][3]["status"] == "invalid_landmark"
    assert result["observations"][5]["status"] == "degenerate_torso_projection"
    assert result["observations"][3]["pelvis_image_px"] is None
    assert result["observations"][5]["pelvis_image_px"] is None


def test_sparse_time_sampling_does_not_imply_supported_continuous_root():
    result = build_root_reference(evidence(fps=2))
    assert result["status"] == "insufficient_reference"
    assert len(result["quality"]["long_reference_intervals"]) == 9


def test_resource_bound_and_linear_reference_construction():
    source = evidence(count=900, fps=30)
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        result = build_root_reference(source)
        timings.append(time.perf_counter()-started)
        assert result["status"] == "ready"
        assert len(result["reference"]) == 900
    print(f"900-sample root reference median milliseconds: {np.median(timings)*1000:.3f}")
    with pytest.raises(ValueError, match="900"):
        build_root_reference(evidence(count=901, fps=30))
    assert ROOT_MOTION_CONTRACT["maximum_samples"] == 900


def test_omitted_source_ids_are_audited_even_inside_short_reference_intervals():
    source = evidence()
    source["reference"].pop(4)
    source["frames"][4]["status"] = "low_confidence"
    result = build_root_reference(source)
    assert result["status"] == "ready"
    interval = result["quality"]["reference_intervals"][3]
    assert interval["omitted_source_evidence_ids"] == [source["frames"][4]["id"]]
    assert interval["status"] == "bounded_reference_interval_not_continuous_observation"


def test_raw_identity_cannot_relabel_an_unrelated_source_frame():
    source = evidence()
    source["frames"][3]["identity_raw"] = deepcopy(source["frames"][4])
    with pytest.raises(ValueError, match="provenance"):
        build_root_reference(source)


def test_displacement_outside_envelope_is_not_hidden_by_interpolation():
    source = evidence()
    for row in source["frames"]:
        for key in (11, 12):
            row["landmarks"][key]["y"] = .48
    for key in (11, 12, 23, 24):
        source["frames"][4]["landmarks"][key]["x"] += .3
    result = build_root_reference(source)
    assert result["status"] == "insufficient_reference"
    assert result["reference"][4]["status"] == "translation_outside_envelope"
    assert result["reference"][4]["translation"] is None
    assert result["reference"][4]["image_displacement_px"] is not None
