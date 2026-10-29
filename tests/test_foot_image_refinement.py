"""Known image rotations test the proposal without real-video scoring feedback."""

from copy import deepcopy
import math

import numpy as np
import pytest

from weft.foot_body import FootBody, foot_rotation, validate_reference
from weft.foot_image_refinement import refine_foot_image
from weft.foot_reference import build_foot_reference
from weft.foot_sandbox import FootEnv
from weft.human import build_human
from weft.io import canonical_hash
from weft.recording import Recorder, export_recording
from test_foot_reference import fixture as raw_foot_fixture
from test_mobile_integration import evidence


@pytest.fixture
def inputs(tmp_path):
    spec = build_human(evidence(), tmp_path / "body", body_model="mobile")
    _, raw, metadata = raw_foot_fixture()
    return spec, raw, metadata


def rotate_image(raw, degrees):
    angle = math.radians(degrees)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    for frame in raw["frames"]:
        for heel, toe in ((29, 31), (30, 32)):
            a = np.array([frame["landmarks"][heel][key] for key in ("x", "y")])
            b = np.array([frame["landmarks"][toe][key] for key in ("x", "y")])
            value = a + rotation @ (b - a)
            frame["landmarks"][toe].update(x=float(value[0]), y=float(value[1]))


@pytest.mark.parametrize("degrees,expected", [(0, 0), (10, 10), (-10, -10), (40, 15), (-40, -15)])
def test_known_camera_plane_rotation_is_exact_and_bounded(inputs, degrees, expected):
    spec, raw, metadata = inputs
    rotate_image(raw, degrees)
    original = build_foot_reference(spec, raw, metadata)
    before = deepcopy((spec, original))
    candidate = refine_foot_image(spec, original)
    assert (spec, original) == before
    assert candidate["fixed_shapes"] == original["fixed_shapes"]
    assert candidate["coverage"] == original["coverage"]
    assert candidate["profile"] == "image-directed-foot-reference-v1"
    assert candidate["orientation_refinement"]["source_reference_sha256"] == canonical_hash(original)
    for old_row, row in zip(original["frames"], candidate["frames"]):
        for side in ("left", "right"):
            old, foot = old_row["feet"][side], row["feet"][side]
            assert foot["raw"] == old["raw"]
            assert math.degrees(foot["image_refinement"]["angle_change_rad"]) == pytest.approx(expected)
            np.testing.assert_allclose(foot_rotation(foot)[1], foot_rotation(old)[1], atol=1e-12)
            points = foot["endpoint_geometry"]
            prior = old["endpoint_geometry"]
            for name in ("heel_relative", "foot_index_relative"):
                assert np.linalg.norm(points[name]) == pytest.approx(np.linalg.norm(prior[name]))
                assert points[name][1] == pytest.approx(prior[name][1])
            assert np.linalg.norm(np.array(points["heel_relative"]) - points["foot_index_relative"]) == pytest.approx(
                original["fixed_shapes"][side]["fixed_parameters"]["length"])
    metrics = candidate["orientation_refinement"]["direction_agreement"]
    assert math.degrees(metrics["mean_before_rad"]) == pytest.approx(abs(degrees))
    assert math.degrees(metrics["mean_after_rad"]) == pytest.approx(abs(degrees - expected))


def test_source_dimensions_prevent_aspect_ratio_distortion(inputs):
    spec, raw, metadata = inputs
    rotate_image(raw, 10)
    metadata["width"] *= 2
    original = build_foot_reference(spec, raw, metadata)
    candidate = refine_foot_image(spec, original)
    observed = candidate["frames"][0]["feet"]["left"]["image_refinement"]["angle_change_rad"]
    assert observed == pytest.approx(math.atan(math.tan(math.radians(10)) / 2))


def test_missing_foot_remains_unavailable_and_uninterpolated(inputs):
    spec, raw, metadata = inputs
    raw["frames"][4]["landmarks"][29]["visibility"] = .1
    original = build_foot_reference(spec, raw, metadata)
    candidate = refine_foot_image(spec, original)
    assert candidate["frames"][4]["feet"]["left"] == original["frames"][4]["feet"]["left"]
    assert candidate["orientation_refinement"]["counts"]["unavailable_source_pose"] == 1
    assert len(candidate["frames"]) == len(original["frames"])


@pytest.mark.parametrize("point", [{"x": 1.1}, {"x": {"nonfinite_raw_value": "nan"}}, {"presence": .49}])
def test_invalid_image_axis_never_changes_available_world_pose(inputs, point):
    spec, raw, metadata = inputs
    original = build_foot_reference(spec, raw, metadata)
    original["frames"][0]["feet"]["left"]["raw"]["heel"]["landmarks"].update(point)
    candidate = refine_foot_image(spec, original)
    foot = candidate["frames"][0]["feet"]["left"]
    assert foot["image_refinement"]["status"] == "unavailable_image_axis"
    assert foot["image_refinement"]["angle_change_rad"] == 0
    np.testing.assert_array_equal(foot_rotation(foot), foot_rotation(original["frames"][0]["feet"]["left"]))


def test_nonfinite_unserialized_reference_is_rejected(inputs):
    spec, raw, metadata = inputs
    original = build_foot_reference(spec, raw, metadata)
    original["frames"][0]["feet"]["left"]["raw"]["heel"]["landmarks"]["x"] = float("nan")
    with pytest.raises(ValueError, match="JSON compliant"):
        refine_foot_image(spec, original)


def test_edge_on_projection_is_not_assigned_a_heading(inputs):
    spec, raw, metadata = inputs
    original = build_foot_reference(spec, raw, metadata)
    foot = original["frames"][0]["feet"]["left"]
    foot["reference"].update(heel_to_index_axis=[0., 1., 0.], axis_to_ankle_direction=[0., 0., 1.])
    candidate = refine_foot_image(spec, original)
    result = candidate["frames"][0]["feet"]["left"]
    assert result["image_refinement"]["status"] == "edge_on_model"
    assert result["image_refinement"]["angle_change_rad"] == 0


def test_native_projection_preserves_original_body_and_keeps_recording_contract(inputs, tmp_path):
    spec, raw, metadata = inputs
    rotate_image(raw, 20)
    original = build_foot_reference(spec, raw, metadata)
    refined = refine_foot_image(spec, original)
    baseline, candidate = FootBody(spec, original), FootBody(spec, refined)
    before, after = baseline.reference_snapshot(0), candidate.reference_snapshot(0)
    for name, point in before["human"]["joints"].items():
        if "heel" not in name and "foot_index" not in name:
            np.testing.assert_array_equal(after["human"]["joints"][name], point)
    np.testing.assert_array_equal(candidate.model.geom_pos[candidate.mobility.floor], baseline.model.geom_pos[baseline.mobility.floor])
    baseline.close()
    candidate.close()
    env = FootEnv(spec, refined)
    env.reset(seed=3)
    recorder = Recorder(tmp_path / "session", env.scene)
    recorder.start(env.snapshot())
    values = env.hold_action()
    _, reward, terminated, truncated, _ = env.step(values)
    recorder.append({"type": "step", "action": values.tolist(), "reward": reward,
                     "terminated": terminated, "truncated": truncated, "snapshot": env.snapshot()})
    recorder.close()
    assert export_recording(recorder.root)["video_frames"] == 2
    env.close()


def test_correction_cannot_accumulate_or_erase_provenance(inputs):
    spec, raw, metadata = inputs
    original = build_foot_reference(spec, raw, metadata)
    candidate = refine_foot_image(spec, original)
    with pytest.raises(ValueError, match="repeated corrections"):
        refine_foot_image(spec, candidate)
    candidate["orientation_refinement"]["floor_used"] = True
    with pytest.raises(ValueError, match="original reference hash"):
        validate_reference(spec, candidate)


@pytest.mark.parametrize("mutation", [
    lambda r: r["frames"][0]["feet"]["left"]["image_refinement"].update(angle_change_rad=1.),
    lambda r: r["frames"][0]["feet"]["left"]["reference"].update(heel_to_index_axis=[0., 1., 0.]),
    lambda r: r["frames"][0]["feet"]["left"].pop("image_refinement"),
    lambda r: r["contract"].update(maximum_orientation_change_rad=1.),
])
def test_bad_version_or_rotation_provenance_rejected(inputs, mutation):
    spec, raw, metadata = inputs
    candidate = refine_foot_image(spec, build_foot_reference(spec, raw, metadata))
    mutation(candidate)
    with pytest.raises(ValueError):
        validate_reference(spec, candidate)
