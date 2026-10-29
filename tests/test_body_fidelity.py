from copy import deepcopy
from pathlib import Path

import pytest

from weft.body_fidelity import angular_error_degrees, grade_torso_projection, load_annotations


ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/body-orientation/annotations.json"


@pytest.fixture
def annotations():
    return load_annotations(LABELS)


def predictions(annotations):
    return [{key: row[key] for key in ("id", "source_sha256", "hip_xy", "shoulder_xy")}
            for row in annotations["frames"]]


@pytest.mark.parametrize("axis,error", [([0, -1], 0), ([1, 0], 90), ([0, 1], 180)])
def test_directed_angle_rejects_upside_down(axis, error):
    assert angular_error_degrees([0, -10], axis) == pytest.approx(error)


@pytest.mark.parametrize("axis", [[0, 0], [float("nan"), 1], [1, float("inf")], [1]])
def test_invalid_axes_are_rejected(axis):
    with pytest.raises(ValueError):
        angular_error_degrees([0, -1], axis)


def test_perfect_pixel_predictions_and_independent_position_metric(annotations):
    rows = predictions(annotations)
    result = grade_torso_projection(annotations, rows)
    assert result["summary"]["all_within_tolerance"]
    assert all(row["normalized_endpoint_error"] == 0 for row in result["frames"])
    shifted = deepcopy(rows)
    for row in shifted:
        for field in ("hip_xy", "shoulder_xy"):
            row[field] = [2 * value + 100 for value in row[field]]
    result = grade_torso_projection(annotations, shifted)
    assert result["summary"]["mean_angular_error_degrees"] == pytest.approx(0, abs=1e-12)
    assert all(row["normalized_endpoint_error"] > 0 for row in result["frames"])


def test_missing_frames_cannot_pass(annotations):
    result = grade_torso_projection(annotations, predictions(annotations)[:-1])
    assert result["summary"]["missing_frames"] == 1
    assert not result["summary"]["all_within_tolerance"]
    empty = grade_torso_projection(annotations, [])
    assert empty["summary"]["mean_angular_error_degrees"] is None
    assert empty["summary"]["missing_frames"] == 9


def test_wrong_source_and_duplicate_predictions_are_rejected(annotations):
    rows = predictions(annotations)
    with pytest.raises(ValueError, match="Duplicate"):
        grade_torso_projection(annotations, rows + [rows[0]])
    rows[0]["source_sha256"] = "another-video"
    with pytest.raises(ValueError, match="checksum"):
        grade_torso_projection(annotations, rows)


def test_annotation_bounds_and_duplicates_are_rejected(annotations):
    bad = deepcopy(annotations)
    bad["frames"][0]["hip_xy"] = [-1, 1]
    with pytest.raises(ValueError, match="original image pixels"):
        grade_torso_projection(bad, [])
    bad = deepcopy(annotations)
    bad["frames"].append(bad["frames"][0])
    with pytest.raises(ValueError, match="Duplicate annotation"):
        grade_torso_projection(bad, [])


def test_upright_baseline_exposes_horizontal_pushup(annotations):
    rows = [{"id": row["id"], "source_sha256": row["source_sha256"], "axis_xy": [0, -1]}
            for row in annotations["frames"]]
    result = grade_torso_projection(annotations, rows)
    pushup = result["clips"]["kettlebell-pushup"]
    assert pushup["mean_angular_error_degrees"] > 60
    assert pushup["within_tolerance_frames"] == 0
    assert all(row["normalized_endpoint_error"] is None for row in result["frames"])


def test_frozen_images_match_their_manifest():
    # Committed excerpts are required; full videos can remain local ignored assets.
    from weft.io import digest
    annotations = load_annotations(LABELS)
    for row in annotations["frames"]:
        assert digest(ROOT / row["image_path"]) == row["image_sha256"]
