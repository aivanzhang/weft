from copy import deepcopy
from hashlib import sha256
import json
import math
import time

import numpy as np
import pytest

from weft.observations import COORDINATES
from weft import source_surface_anchors as anchors


SOURCE = "a" * 64


def fixture(count=4):
    unknown = {"value": None, "status": "unknown", "evidence_ids": []}
    source = {"sha256": SOURCE, "width_px": 100, "height_px": 80, "fps": 30., "metadata_frames": count, "decoded_frames": count}
    observations = {"schema_version": 1, "profile": "scene-observations-v1", "source": source,
                    "time": {"unit": "seconds", "origin": "video_start", "basis": "frame_index_divided_by_fps", "status": "assumed"},
                    "image": {"width_px": 100, "height_px": 80}, "coordinates": deepcopy(COORDINATES),
                    "camera": {"motion": deepcopy(unknown), "calibration": deepcopy(unknown), "screen": None},
                    "objects": [{"id": "person", "label": "person", "properties": {"kind": deepcopy(unknown), "role": deepcopy(unknown)}, "unknowns": []}],
                    "evidence": [], "observations": [], "relationships": [], "provenance": {}, "limits": []}
    annotations = {"schema_version": 1, "profile": "source-surface-annotations-v1", "source_sha256": SOURCE,
                   "quality": "authored", "provenance": "Synthetic pixel annotations, not physical evidence",
                   "channels": [{"id": "left_toe", "entity_ref": {"document_sha256": "pending", "object_id": "person"},
                                 "point_role": "authored toe surface point", "surface_id": "floor"}], "annotations": []}
    for frame in range(count):
        evidence_id = f"{SOURCE}:frame:{frame}:person"
        observation_id = f"{SOURCE}:observation:{frame}:person"
        observations["evidence"].append({"id": evidence_id, "frame": frame, "time_s": frame / 30,
                                         "object_id": "person", "kind": "person_hypothesis", "status": "inferred", "payload": {}})
        observations["observations"].append({"id": observation_id, "frame": frame, "time_s": frame / 30, "object_id": "person",
                                             "visibility": "detected", "partition": "construction", "evidence_ids": [evidence_id], "measurements": {}})
        annotations["annotations"].append({"id": f"{SOURCE}:frame:{frame}:surface-anchor:left_toe", "channel_id": "left_toe",
                                          "frame": frame, "time_s": frame / 30, "observation_id": observation_id,
                                          "status": "annotated", "pixel": [60., 45.], "uncertainty_px": 1.})
    camera = {"schema_version": 1, "source_sha256": SOURCE, "width": 100, "height": 80, "fps": 30., "frame_count": count,
              "world_axes": "right_handed_z_up", "units": "m", "provenance": "Authored analytical camera, not a measured device",
              "quality": "authored", "K": [[100., 0., 50.], [0., 100., 40.], [0., 0., 1.]], "radial_k1": 0.,
              "frames": [{"frame": frame, "time_s": frame / 30, "R": [[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]], "t": [0., 0., 2.]} for frame in range(count)]}
    surfaces = {"schema_version": 1, "profile": "source-horizontal-surfaces-v1", "source_sha256": SOURCE,
                "calibration_sha256": "pending", "world_frame": {"id": "supplied-room", "length_unit": "m", "axes": "x_right_y_away_z_up"},
                "calibration_evidence_refs": [], "surfaces": [{"id": "floor", "kind": "ground", "height_m": 0., "bounds_xy_m": None,
                    "quality": "authored", "provenance": "Analytical horizontal plane", "evidence_refs": []}]}
    return observations, camera, surfaces, annotations


def encoded(data):
    observations, camera, surfaces, annotations = deepcopy(data)
    raw_observations = json.dumps(observations).encode()
    digest = sha256(raw_observations).hexdigest()
    raw_camera = json.dumps(camera).encode()
    surfaces["calibration_sha256"] = sha256(raw_camera).hexdigest()
    for channel in annotations["channels"]:
        channel["entity_ref"]["document_sha256"] = digest
    for ref in surfaces["calibration_evidence_refs"] + [r for surface in surfaces["surfaces"] for r in surface["evidence_refs"]]:
        ref["document_sha256"] = digest
    return {"observation_documents": {digest: raw_observations}, "source_sha256": SOURCE,
            "camera_document": raw_camera, "surface_document": json.dumps(surfaces).encode(),
            "annotation_document": json.dumps(annotations).encode()}


def run(data=None, **changes):
    return anchors.reconstruct_source_surface_anchors(**(encoded(fixture() if data is None else data) | changes))


def test_known_plane_resolves_metric_coordinates_without_installation():
    result = run()
    point = result["rows"][0]["anchors"][0]
    assert point["status"] == "conditional_surface_anchor"
    assert point["position_world_m"] == pytest.approx([.2, -.1, 0.], abs=1e-12)
    assert point["sampled_pixel_box_world_range_m"]["minimum"] == pytest.approx([.18, -.12, 0.], abs=1e-12)
    assert point["sampled_pixel_box_world_range_m"]["maximum"] == pytest.approx([.22, -.08, 0.], abs=1e-12)
    assert point["camera_quality_claim"] == point["surface_quality_claim"] == "authored"
    assert result["counts"]["conditional_anchors"] == 4
    assert result["contact"] == result["load"] == result["grasp"] == "unknown"
    assert not result["geometry_modified"] and not result["body_or_floor_installed"] and not result["physical_validation"]
    alignment = result["alignments"]["camera-frame:0"]
    assert alignment["from_frame"]["length_unit"] == alignment["to_frame"]["length_unit"] == "m"
    assert alignment["spatial"]["scale"] == 1


def test_radial_distortion_reuses_existing_inverse_projection():
    data = fixture()
    data[1]["radial_k1"] = .1
    for row in data[3]["annotations"]:
        row.update(pixel=[60.0125, 45.00625], uncertainty_px=0.)
    assert run(data)["rows"][0]["anchors"][0]["position_world_m"] == pytest.approx([.2, -.1, 0.], abs=1e-10)


def test_oblique_translated_camera_and_raised_static_support():
    data = fixture()
    angle = .3
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0], [-math.sin(angle), -math.cos(angle), 0], [0, 0, -1.]])
    center, expected = np.array([1., 2., 3.]), np.array([1.2, 1.9, .5])
    translation = -rotation @ center
    camera_point = rotation @ expected + translation
    pixel = [50 + 100 * camera_point[0] / camera_point[2], 40 + 100 * camera_point[1] / camera_point[2]]
    for row in data[1]["frames"]:
        row.update(R=rotation.tolist(), t=translation.tolist())
    for row in data[3]["annotations"]:
        row.update(pixel=pixel, uncertainty_px=0)
    data[2]["surfaces"][0].update(kind="static_support", height_m=.5, bounds_xy_m=[1., 1., 2., 2.])
    assert run(data)["rows"][0]["anchors"][0]["position_world_m"] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("case,reason", [("parallel", "ray_parallel_to_surface"), ("behind", "intersection_behind_camera_or_beyond_bound"),
                                        ("outside", "intersection_uncertainty_outside_declared_surface"), ("image_edge", "pixel_uncertainty_box_outside_source_image")])
def test_invalid_intersections_remain_unavailable(case, reason):
    data = fixture()
    if case == "parallel":
        for row in data[1]["frames"]:
            row.update(R=[[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], t=[0., 2., 0.])
        for row in data[3]["annotations"]:
            row.update(pixel=[50., 40.], uncertainty_px=0)
    elif case == "behind":
        for row in data[1]["frames"]:
            row.update(R=np.eye(3).tolist(), t=[0., 0., -2.])
    elif case == "outside":
        data[2]["surfaces"][0].update(kind="static_support", bounds_xy_m=[0., 0., .1, .1])
    else:
        data[3]["annotations"][0]["pixel"] = [0., 40.]
    result = run(data)
    assert result["rows"][0]["anchors"][0]["reason"] == reason
    assert result["rows"][0]["anchors"][0]["position_world_m"] is None


def test_excluded_occluded_and_unsampled_slots_never_read_pixels(monkeypatch):
    data = fixture()
    data[3]["annotations"][1]["status"] = "occluded"
    data[0]["observations"][2]["partition"] = "audit"
    data[0]["observations"][3].update(partition="unsampled", visibility="not_sampled")
    for row in data[3]["annotations"][1:]:
        row.update(pixel="not geometry", uncertainty_px="not geometry")
    original = anchors._intersect
    calls = []

    def spy(*args):
        calls.append(args[0])
        return original(*args)

    monkeypatch.setattr(anchors, "_intersect", spy)
    result = run(data)
    assert calls == [[60., 45.]]
    assert [row["anchors"][0]["reason"] for row in result["rows"]] == [None, "annotation_occluded", "excluded_evidence_partition", "source_not_sampled"]
    assert result["counts"]["retained_anchor_slots"] == 4


@pytest.mark.parametrize("missing,reason", [("camera_document", "missing_camera_calibration"), ("surface_document", "missing_surface_geometry")])
def test_missing_additional_inputs_are_useful_unavailable_slots(missing, reason):
    result = run(**{missing: None})
    assert result["status"] == "unavailable"
    assert result["counts"]["retained_anchor_slots"] == 4
    assert result["rows"][0]["anchors"][0]["reason"] == reason


def test_missing_annotations_do_not_invent_channels_or_drop_original_frames():
    result = run(annotation_document=None)
    assert result["status"] == "unavailable" and result["missing_inputs"] == ["annotations"]
    assert result["counts"]["retained_source_frames"] == 4
    assert result["counts"]["declared_channels"] == 0


def test_unknown_surface_height_does_not_default_to_zero():
    data = fixture()
    data[2]["surfaces"][0].update(height_m=None, quality="unknown")
    assert run(data)["rows"][0]["anchors"][0]["reason"] == "unknown_surface_height"


@pytest.mark.parametrize("kind", ["camera", "surface"])
def test_metres_and_measured_label_are_not_measurement_evidence(kind):
    data = fixture()
    if kind == "camera":
        data[1]["quality"] = "measured"
    else:
        data[2]["surfaces"][0]["quality"] = "measured"
    result = run(data)
    assert result["counts"]["conditional_anchors"] == 0
    assert result["rows"][0]["anchors"][0]["reason"] == f"missing_supplied_{kind}_measurement_receipt"


@pytest.mark.parametrize("mismatch", [None, "calibration", "height"])
def test_supplied_measurement_receipts_stay_bound_claims_not_contact_validation(mismatch):
    data = fixture()
    data[1]["quality"] = data[2]["surfaces"][0]["quality"] = "measured"
    for kind, target in (("camera_calibration_measurement", data[2]["calibration_evidence_refs"]),
                         ("surface_height_measurement", data[2]["surfaces"][0]["evidence_refs"])):
        identifier = f"{SOURCE}:frame:0:{kind}"
        payload = {"source_sha256": SOURCE, "calibration_sha256": sha256(json.dumps(data[1]).encode()).hexdigest(), "units": "m",
                   "method": "Synthetic analytical measurement receipt", "uncertainty": "Controlled fixture; not a real measurement"}
        if kind == "surface_height_measurement":
            payload.update(surface_id="floor", world_frame_id="supplied-room", height_m=1. if mismatch == "height" else 0.)
        elif mismatch == "calibration":
            payload["calibration_sha256"] = "b" * 64
        data[0]["evidence"].append({"id": identifier, "frame": 0, "time_s": 0., "object_id": "person",
                                    "kind": kind, "status": "observed", "payload": payload})
        data[0]["observations"][0]["evidence_ids"].append(identifier)
        target.append({"document_sha256": "pending", "evidence_id": identifier})
    result = run(data)
    assert result["counts"]["conditional_anchors"] == (0 if mismatch else 4)
    assert not result["physical_validation"] and result["contact"] == "unknown"


def test_supplied_rotation_roundoff_uses_the_same_proper_mapping_for_ray_and_point():
    data = fixture()
    for row in data[1]["frames"]:
        row["R"][0][2] += 1e-6
    result = run(data)
    assert result["counts"]["conditional_anchors"] == 4
    assert result["rows"][0]["anchors"][0]["position_world_m"][2] == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("case,match", [("calibration", "calibration bytes"), ("source", "original source"),
                                      ("time", "identity/clock"), ("moving", "static-support"), ("units", "metres")])
def test_source_clock_units_and_geometry_identity_fail_closed(case, match):
    data = fixture()
    if case == "time":
        data[3]["annotations"][0]["time_s"] = .001
    elif case == "moving":
        data[2]["surfaces"][0]["kind"] = "handheld"
    elif case == "units":
        data[2]["world_frame"]["length_unit"] = "model_units"
    kwargs = encoded(data)
    if case == "calibration":
        doc = json.loads(kwargs["surface_document"])
        doc["calibration_sha256"] = "b" * 64
        kwargs["surface_document"] = json.dumps(doc).encode()
    elif case == "source":
        kwargs["source_sha256"] = "b" * 64
    with pytest.raises(ValueError, match=match):
        anchors.reconstruct_source_surface_anchors(**kwargs)


def test_native_observation_clock_is_not_silently_recast_as_nominal_camera_time():
    data = fixture()
    doc = data[0]
    doc["schema_version"] = 3
    doc["source"].pop("fps")
    doc["source"]["frame_times_s"] = [0., .034, .068, .102]
    doc["time"] = {"unit": "seconds", "origin": "submitted_video_start", "basis": "decoder_presentation_timestamps", "status": "observed"}
    for rows in (doc["evidence"], doc["observations"], data[3]["annotations"]):
        for row in rows:
            row["time_s"] = doc["source"]["frame_times_s"][row["frame"]]
    result = run(data)
    assert result["rows"][0]["anchors"][0]["status"] == "conditional_surface_anchor"
    assert all(row["anchors"][0]["reason"] == "camera_and_source_clocks_differ" for row in result["rows"][1:])


def test_900_annotation_limit_does_not_drop_a_longer_complete_source():
    data = fixture(count=1200)
    data[0]["source"]["metadata_frames"] = 1201
    data[3]["annotations"] = data[3]["annotations"][:900]
    for row in data[3]["annotations"]:
        row["status"] = "unknown"
    result = run(data, camera_document=None)
    assert result["counts"]["decoded_frames"] == 1200
    assert result["counts"]["retained_source_frames"] == result["counts"]["retained_anchor_slots"] == 1201
    assert result["counts"]["supplied_annotation_slots"] == 900
    assert result["rows"][-1]["anchors"][0]["reason"] == "metadata_only_not_decoded"
    assert result["rows"][1000]["anchors"][0]["reason"] == "missing_annotation"


def test_additional_byte_bound_precedes_json_loading(monkeypatch):
    monkeypatch.setitem(anchors.CONTRACT, "maximum_additional_input_bytes", 10)
    monkeypatch.setattr(anchors, "_sources", lambda *a: pytest.fail("oversized input entered registry"))
    with pytest.raises(ValueError, match="4 MiB"):
        anchors.reconstruct_source_surface_anchors({}, source_sha256=SOURCE, camera_document=b" " * 11)


def test_deadline_precedes_registry_loading(monkeypatch):
    monkeypatch.setitem(anchors.CONTRACT, "maximum_seconds", -1.)
    monkeypatch.setattr(anchors, "load_world_evidence", lambda *a: pytest.fail("expired input loaded"))
    with pytest.raises(TimeoutError, match="ten-second"):
        anchors.reconstruct_source_surface_anchors({}, source_sha256=SOURCE)


def test_output_bound_is_not_satisfied_by_dropping_slots(monkeypatch):
    monkeypatch.setitem(anchors.CONTRACT, "maximum_output_bytes", 100)
    with pytest.raises(ValueError, match="output byte bound"):
        run()


@pytest.mark.parametrize("index", [2, 3])
def test_boolean_version_is_not_a_schema_number(index):
    data = fixture()
    data[index]["schema_version"] = True
    with pytest.raises(ValueError, match="profile/source|source-bound"):
        run(data)


def test_bounded_full_length_synthetic_geometry(record_property):
    data = fixture(count=1800)
    data[3]["annotations"] = data[3]["annotations"][:900]
    kwargs = encoded(data)
    started = time.monotonic()
    result = anchors.reconstruct_source_surface_anchors(**kwargs)
    elapsed = time.monotonic() - started
    output_bytes = len(json.dumps(result, sort_keys=True, allow_nan=False).encode())
    assert result["counts"]["conditional_anchors"] == 900
    assert result["counts"]["retained_source_frames"] == result["counts"]["retained_anchor_slots"] == 1800
    assert result["counts"]["slot_status_counts"]["missing_annotation"] == 900
    assert result["rows"][899]["anchors"][0]["position_world_m"] == pytest.approx([.2, -.1, 0.], abs=1e-12)
    assert elapsed < 10 and output_bytes < 8 << 20
    record_property("geometry_seconds", elapsed)
    record_property("output_bytes", output_bytes)
    record_property("scope", "900 synthetic ray/plane anchors; 1800 complete source frames; not real-video accuracy")
