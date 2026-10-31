from copy import deepcopy
from hashlib import sha256
import json

import pytest

from weft import exercise_support_projection as projection


SOURCE = "a" * 64
TRANSLATION = [.1, -.05, 3.]
WORLD = {
    11: [-.2, -.45, .12], 12: [.2, -.45, .08], 13: [-.4, -.2, .1], 14: [.4, -.2, -.05],
    15: [-.45, .1, .15], 16: [.45, .1, -.12], 23: [-.15, 0., 0.], 24: [.15, 0., 0.],
    25: [-.16, .4, .1], 26: [.16, .4, .05], 27: [-.16, .8, .2], 28: [.16, .8, .1],
    29: [-.18, .82, .1], 30: [.18, .82, .1], 31: [-.12, .85, .35], 32: [.24, .85, .35],
}


def inputs(count=10, selected=(0, 1, 2)):
    frames = []
    for frame in range(count):
        world = [dict(x=0., y=0., z=0., presence=1., visibility=1.) for _ in range(33)]
        image = deepcopy(world)
        for index, coordinates in WORLD.items():
            world[index].update(zip(("x", "y", "z"), coordinates))
            x, y, z = [value + offset for value, offset in zip(coordinates, TRANSLATION)]
            image[index].update(x=(320 + 640 * x / z) / 640, y=(240 + 640 * y / z) / 480)
        frames.append({"id": f"{SOURCE}:frame:{frame}", "frame": frame, "time_s": frame / 30,
                       "decoder_timestamp_s": frame / 30, "status": "tracked", "pose_count": 1,
                       "landmarks": image, "world_landmarks": world})
    pose = {"source_sha256": SOURCE, "width": 640, "height": 480, "fps": 30.,
            "frame_count": count + 2, "decoded_frames": count + 1, "pose_identity": {"mode": "raw"}, "frames": frames}
    cutoff = int(.6 * count)
    audit = [{"frame": row["frame"], "evidence_id": row["id"], "status": "usable" if i in selected else "rejected",
              "reasons": [] if i in selected else ["frozen_selection_rejection"]} for i, row in enumerate(frames[:cutoff])]
    selection = {"source_sha256": SOURCE, "method": "stationary_candidate_foot_covariance_v1",
                 "supplied_pose_rows": count, "construction_rows": cutoff,
                 "selected_evidence_ids": [r["id"] for r in frames[:cutoff]],
                 "construction_source_frames": [r["frame"] for r in frames[:cutoff]],
                 "excluded_later_source_frames": [r["frame"] for r in frames[cutoff:]],
                 "construction_evidence_ids": [r["evidence_id"] for r in audit if r["status"] == "usable"],
                 "usable_frames": len(selected), "rejected_frames": cutoff - len(selected), "audit": audit,
                 "support_selection": {"profile": "exercise-support-evidence-v1", "source_sha256": SOURCE,
                                       "camera_prior": {"mode": "fixed_source_camera", "status": "authored_unverified",
                                                        "source_sha256": SOURCE}}}
    return pose, selection


def documents(pose, selection):
    selection["support_selection"]["pose_evidence_canonical_sha256"] = sha256(json.dumps(pose, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return json.dumps(pose, indent=2).encode(), json.dumps(selection, indent=2).encode()


def test_exact_camera_is_recovered_without_changing_source_geometry():
    pose, selection = inputs()
    before = deepcopy(pose)
    result = projection.audit_exercise_support_projection(*documents(pose, selection))
    assert pose == before
    assert result["status"] == "consistent_with_fixed_camera_hypothesis"
    assert result["metrics"]["median_foot_direction_error_deg"] < 1e-5
    assert result["metrics"]["median_core_direction_error_deg"] < 1e-5
    assert result["counts"]["linear_solves"] == 3
    assert result["counts"]["evaluable_foot_axes"] == 6
    assert result["counts"]["not_selected_construction_rows"] == 3
    assert result["counts"]["excluded_later_rows"] == 4
    assert result["counts"]["source_slot_status_counts"]["unsampled"] == 1
    assert result["counts"]["source_slot_status_counts"]["metadata_only_not_decoded"] == 1
    assert result["counts"]["retained_source_foot_axis_slots"] == 24
    for row in result["rows"][:3]:
        assert row["solve"]["translation_model_units"] == pytest.approx(TRANSLATION, abs=1e-12)
        assert row["core_endpoints_for_solve"] == list(projection.CORE_POINTS)
    assert result["ground"] == result["contact"] == result["gravity"] == "unknown"
    assert not result["physical_validation"] and not result["geometry_modified"] and not result["body_admission_changed"]


def test_held_out_foot_error_rejects_hypothesis_without_changing_core_translation():
    pose, selection = inputs()
    for row in pose["frames"][:3]:
        for index in (31, 32):
            row["world_landmarks"][index]["x"] -= .3
    result = projection.audit_exercise_support_projection(*documents(pose, selection))
    assert result["status"] == "fixed_camera_hypothesis_rejected"
    assert result["metrics"]["median_foot_direction_error_deg"] > 5
    assert result["metrics"]["evaluable_foot_axis_fraction"] == 1
    for row in result["rows"][:3]:
        assert row["solve"]["translation_model_units"] == pytest.approx(TRANSLATION, abs=1e-12)


def test_rejected_and_later_payloads_never_enter_geometry(monkeypatch):
    pose, selection = inputs()
    baseline = projection.audit_exercise_support_projection(*documents(pose, selection))
    for row in pose["frames"][3:]:
        row["landmarks"] = {"poison": "not a landmark array"}
        row["world_landmarks"] = "not evaluated"
    original = projection._point
    visited = []

    def selected_only(row, index):
        assert row["frame"] in (0, 1, 2)
        visited.append((row["frame"], index))
        return original(row, index)

    monkeypatch.setattr(projection, "_point", selected_only)
    result = projection.audit_exercise_support_projection(*documents(pose, selection))
    assert len(visited) == 48
    baseline.pop("bindings")
    result.pop("bindings")
    assert result == baseline


@pytest.mark.parametrize("space", ["landmarks", "world_landmarks"])
def test_missing_foot_confidence_preserves_denominator_and_is_inconclusive(space):
    pose, selection = inputs()
    pose["frames"][0][space][29]["visibility"] = .49
    result = projection.audit_exercise_support_projection(*documents(pose, selection))
    assert result["counts"]["selected_foot_axis_slots"] == 6
    assert result["counts"]["evaluable_foot_axes"] == 5
    assert result["status"] == "inconclusive"
    assert result["rows"][0]["foot_endpoints"][0]["reason"] == "low_endpoint_confidence"


def test_fewer_than_six_core_points_does_not_solve():
    pose, selection = inputs()
    for row in pose["frames"][:3]:
        for index in projection.CORE_POINTS[5:]:
            row["landmarks"][index]["presence"] = 0
    result = projection.audit_exercise_support_projection(*documents(pose, selection))
    assert result["status"] == "inconclusive"
    assert result["counts"]["linear_solves"] == 0
    assert all(row["reason"] == "insufficient_core_endpoints" for row in result["rows"][:3])


@pytest.mark.parametrize("spread,reason", [(0., "core_design_rank_unresolved"), (1e-9, "core_design_condition_excessive")])
def test_degenerate_or_ill_conditioned_core_is_retained(spread, reason):
    pose, selection = inputs()
    for row in pose["frames"][:3]:
        for i, index in enumerate(projection.CORE_POINTS):
            row["landmarks"][index].update(x=.5 + (i % 2) * spread, y=.5 + (i % 3) * spread)
    result = projection.audit_exercise_support_projection(*documents(pose, selection))
    assert result["status"] == "inconclusive"
    assert all(row["reason"] == reason for row in result["rows"][:3])
    json.dumps(result, allow_nan=False)


def test_one_held_out_point_behind_camera_invalidates_its_row():
    pose, selection = inputs()
    pose["frames"][0]["world_landmarks"][29]["z"] = -4
    result = projection.audit_exercise_support_projection(*documents(pose, selection))
    assert result["rows"][0]["reason"] == "row_has_invalid_camera_projection"
    assert result["rows"][0]["foot_endpoints"][0]["camera_depth_model_units"] == pytest.approx(-1)
    assert result["counts"]["evaluable_foot_axes"] == 4
    assert result["status"] == "inconclusive"


def test_zero_length_image_axis_is_not_an_accurate_direction():
    pose, selection = inputs()
    pose["frames"][0]["landmarks"][31] = deepcopy(pose["frames"][0]["landmarks"][29])
    result = projection.audit_exercise_support_projection(*documents(pose, selection))
    assert result["rows"][0]["foot_axes"][0]["reason"] == "zero_length_image_or_projected_axis"
    assert result["status"] == "inconclusive"


@pytest.mark.parametrize("change,match", [
    ("canonical", "different pose bytes"), ("source", "source/partition"),
    ("tail_clock", "clock differs"), ("selection_order", "ordered partition"), ("raw", "raw pose mode"),
])
def test_incompatible_provenance_is_rejected(change, match):
    pose, selection = inputs()
    if change == "source":
        selection["source_sha256"] = "b" * 64
    elif change == "tail_clock":
        pose["frames"][-1]["time_s"] += .01
    elif change == "selection_order":
        selection["excluded_later_source_frames"].reverse()
    elif change == "raw":
        pose["pose_identity"]["mode"] = "selected_crop"
    raw, saved = documents(pose, selection)
    if change == "canonical":
        pose["frames"][0]["world_landmarks"][29]["x"] += .01
        raw = json.dumps(pose).encode()
    with pytest.raises(ValueError, match=match):
        projection.audit_exercise_support_projection(raw, saved)


def test_oversized_bytes_are_rejected_before_json_parse(monkeypatch):
    monkeypatch.setitem(projection.CONTRACT, "maximum_input_bytes", 100)
    monkeypatch.setattr(projection.json, "loads", lambda *a, **k: pytest.fail("oversized JSON parsed"))
    with pytest.raises(ValueError, match="input byte bound"):
        projection.audit_exercise_support_projection(b" " * 100, b"{}")


def test_excessive_rows_rejected_before_canonical_serialization(monkeypatch):
    pose, selection = inputs(count=901)
    raw, saved = documents(pose, selection)
    monkeypatch.setattr(projection.json, "dumps", lambda *a, **k: pytest.fail("oversized rows serialized"))
    with pytest.raises(ValueError, match="900"):
        projection.audit_exercise_support_projection(raw, saved)


def test_deadline_is_checked_before_input_parsing(monkeypatch):
    times = iter((0., 11.))
    monkeypatch.setattr(projection.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(projection.json, "loads", lambda *a, **k: pytest.fail("expired JSON parsed"))
    with pytest.raises(TimeoutError, match="ten-second"):
        projection.audit_exercise_support_projection(b"{}", b"{}")


def test_output_byte_limit_is_enforced(monkeypatch):
    raw, saved = documents(*inputs())
    monkeypatch.setitem(projection.CONTRACT, "maximum_output_bytes", 100)
    with pytest.raises(ValueError, match="output byte bound"):
        projection.audit_exercise_support_projection(raw, saved)
