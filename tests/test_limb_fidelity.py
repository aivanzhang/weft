from copy import deepcopy
from pathlib import Path

import pytest

from weft.io import digest
from weft.limb_fidelity import grade_limb_projection, load_annotations


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/limb-orientation/annotations.json"


@pytest.fixture
def labels():
    return load_annotations(LABELS)


def exact_predictions(labels):
    return [{"id": frame["id"], "source_sha256": frame["source_sha256"],
             "segments": {name: {"start_xy": frame["points"][row["start"]]["xy"],
                                  "end_xy": frame["points"][row["end"]]["xy"]}
                          for name, row in frame["segments"].items() if row["scorable"]}}
            for frame in labels["frames"]]


def test_exact_predictions_retain_source_occlusion(labels):
    result = grade_limb_projection(labels, exact_predictions(labels))
    summary = result["summary"]
    assert summary["expected_segment_slots"] == 96
    assert summary["source_scorable_segments"] == summary["within_tolerance_segments"] == 84
    assert summary["source_unavailable_segments"] == 12
    assert summary["all_source_scorable_within_tolerance"]
    assert len(result["observations"]) == 96


def test_absent_predictions_are_not_dropped(labels):
    result = grade_limb_projection(labels, [])
    assert result["summary"]["missing_predictions"] == 84
    assert result["summary"]["source_unavailable_segments"] == 12
    assert result["summary"]["within_tolerance_fraction_of_source_scorable"] == 0
    assert not result["summary"]["all_source_scorable_within_tolerance"]


def test_degenerate_projection_fails_without_hiding_coverage(labels):
    rows = exact_predictions(labels)
    rows[0]["segments"]["left_upper_arm"] = {"axis_xy": [0, 0]}
    result = grade_limb_projection(labels, rows)
    assert result["summary"]["degenerate_predictions"] == 1
    assert result["summary"]["scored_segments"] == 83
    assert not result["summary"]["all_source_scorable_within_tolerance"]


def test_direction_reversal_fails(labels):
    rows = exact_predictions(labels)
    for frame in rows:
        for segment in frame["segments"].values():
            segment["start_xy"], segment["end_xy"] = segment["end_xy"], segment["start_xy"]
    result = grade_limb_projection(labels, rows)
    assert result["summary"]["mean_angular_error_degrees"] == pytest.approx(180)
    assert result["summary"]["within_tolerance_segments"] == 0


def test_translation_and_uniform_scale_do_not_change_direction(labels):
    rows = deepcopy(exact_predictions(labels))
    for frame in rows:
        for segment in frame["segments"].values():
            for key in ("start_xy", "end_xy"):
                segment[key] = [3 * value + 42 for value in segment[key]]
    assert grade_limb_projection(labels, rows)["summary"]["mean_angular_error_degrees"] == pytest.approx(0)


@pytest.mark.parametrize("bad", [[float("nan"), 0], [0, float("inf")], [1]])
def test_nonfinite_or_malformed_axis_rejected(labels, bad):
    rows = exact_predictions(labels)
    rows[0]["segments"]["left_upper_arm"] = {"axis_xy": bad}
    with pytest.raises(ValueError):
        grade_limb_projection(labels, rows)


def test_wrong_source_duplicate_unknown_id_rejected(labels):
    rows = exact_predictions(labels)
    with pytest.raises(ValueError, match="Duplicate"):
        grade_limb_projection(labels, rows + [rows[0]])
    rows[0]["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="checksum"):
        grade_limb_projection(labels, rows)
    rows[0]["id"] = "unknown"
    with pytest.raises(ValueError, match="unknown"):
        grade_limb_projection(labels, rows)


def test_missing_source_slots_and_incorrect_visibility_rejected(labels):
    bad = deepcopy(labels)
    del bad["frames"][0]["segments"]["left_shin"]
    with pytest.raises(ValueError, match="retain all"):
        grade_limb_projection(bad, [])
    bad = deepcopy(labels)
    bad["frames"][0]["segments"]["left_shin"]["scorable"] = False
    with pytest.raises(ValueError, match="visibility"):
        grade_limb_projection(bad, [])


def test_frozen_label_and_excerpt_checksums(labels):
    assert digest(LABELS) == "f051db850105a77e0350cdfd7128f84d340ad1a18dafcfeda9c6f4ab722290bd"
    for frame in labels["frames"]:
        assert digest(ROOT / frame["image_path"]) == frame["image_sha256"]


def test_ambiguous_prediction_forms_rejected(labels):
    rows = exact_predictions(labels)
    rows[0]["segments"]["left_upper_arm"]["axis_xy"] = [1, 0]
    with pytest.raises(ValueError, match="either"):
        grade_limb_projection(labels, rows)
