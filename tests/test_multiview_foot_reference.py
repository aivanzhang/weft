"""Analytical byte fixtures only; reconstruction, models and native physics are forbidden."""

import builtins
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
import socket
import subprocess

import numpy as np
import pytest


def _deny(*_args, **_kwargs):
    raise AssertionError("Native physics/model/network/subprocess or reconstruction is forbidden")


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    original = builtins.__import__
    forbidden = {"mujoco", "mediapipe", "tensorflow", "torch", "cv2", "requests", "httpx",
                 "weft.human", "weft.mobile_human", "weft.foot_body"}

    def guarded(name, *args, **kwargs):
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden):
            _deny()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for owner, names in ((socket, ("socket", "create_connection")),
                         (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                         (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, _deny)


@pytest.fixture
def api(monkeypatch):
    module = importlib.import_module("weft.multiview_foot_reference")
    pose = importlib.import_module("weft.multiview_pose")
    monkeypatch.setattr(pose, "triangulate_pair", _deny)
    monkeypatch.setattr(pose, "_triangulate_rays", _deny)
    return module


def _encode(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def _pin(fixture, api):
    """Rebuild declared fixture custody after an intentional analytical change."""
    for index, view in enumerate(fixture["predictions"]["views"]):
        view["camera_sha256"] = api.canonical_hash(fixture["cameras"][index])
        view["evidence_sha256"] = api.canonical_hash(fixture["evidence"][index])
        view["calibration_quality"] = fixture["cameras"][index]["quality"]
    body, ground, mapping = (fixture[key] for key in ("body", "ground", "mapping"))
    binding = body["camera_reference"]
    primary = fixture["predictions"]["views"][0]
    ground["calibration_sha256"] = primary["camera_sha256"]
    binding.update(calibration_sha256=primary["camera_sha256"], pose_evidence_sha256=primary["evidence_sha256"],
                   root_reference_sha256=api.canonical_hash(body["root_reference"]),
                   angular_reference_sha256=api.canonical_hash(body["reference"]),
                   body_rotation_sha256=api.canonical_hash(body["body_frame"]["rotation"]), ground=deepcopy(ground))
    root = importlib.import_module("weft.multiview_root")
    binding["root_refinement"].update(unchanged_shape_sha256=root.shape_hash(body),
                                      triangulation_sha256=api.canonical_hash(fixture["predictions"]),
                                      views=deepcopy(fixture["predictions"]["views"]))
    mapping["from_frame"]["id"] = "camera-world:" + primary["camera_sha256"]
    mapping["to_frame"]["id"] = "body:" + api.canonical_hash(body)
    return fixture


def _fixture(api, *, count=12, fps=10, sampled=(0, 2, 4, 6, 8, 10), ground_height=0.):
    sources = ("a" * 64, "b" * 64)
    identity = np.eye(3).tolist()
    cameras, evidence = [], []
    points = [[1., 2., 1.] for _ in range(33)]
    for shift, side in ((0., "left"), (.4, "right")):
        foot = ([1.1 + shift, 2., .14], [1.05 + shift, 2., .04], [1.25 + shift, 2., .04])
        for index, point in zip(api.LANDMARKS[side].values(), foot, strict=True):
            points[index] = point
    for number, source in enumerate(sources):
        translation = [-float(number), 0., 5.]
        camera = {"schema_version": 1, "source_sha256": source, "width": 640, "height": 480,
                  "fps": fps, "frame_count": count, "world_axes": "right_handed_z_up", "units": "m",
                  "provenance": "Authored analytical camera, not a source measurement", "quality": "authored",
                  "K": [[100., 0., 320.], [0., 100., 240.], [0., 0., 1.]], "radial_k1": 0.,
                  "frames": [{"frame": i, "time_s": i / fps, "R": identity, "t": translation} for i in range(count)]}
        image = []
        for world in points:
            position = np.asarray(world) + translation
            pixel = np.asarray(camera["K"]) @ (position / position[2])
            image.append({"x": float(pixel[0] / 640), "y": float(pixel[1] / 480), "z": 0.,
                          "visibility": .9, "presence": .9})
        document = {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": source,
                    "width": 640, "height": 480, "fps": fps, "frame_count": count, "decoded_frames": count,
                    "frames": [{"id": f"{source}:frame:{i}", "frame": i, "time_s": i / fps,
                                "decoder_timestamp_s": i / fps, "pose_count": 1, "status": "tracked",
                                "landmarks": deepcopy(image), "world_landmarks": "monocular depth MUST NOT be used"}
                               for i in sampled]}
        cameras.append(camera)
        evidence.append(document)
    prediction = {"schema_version": 1, "contract": dict(api.RAY_CONTRACT), "source_frames": count,
                  "joint_slots": count * 33, "monocular_world_landmarks_used": False,
                  "reference_body_or_contacts_used": False, "physical_validation": "unverified", "views": [], "frames": []}
    for number, source in enumerate(sources):
        prediction["views"].append({"name": f"view-{number}", "source_sha256": source,
                                    "camera_sha256": "0" * 64, "evidence_sha256": "0" * 64,
                                    "calibration_quality": "authored", "clock_provenance": "Authored equal-frame fixture clock"})
    for frame in range(count):
        present = frame in sampled
        prediction["frames"].append({"frame": frame, "time_s": frame / fps,
                                     "source_evidence_ids": [f"{source}:frame:{frame}" if present else None for source in sources],
                                     "joints": [{"status": "estimated" if present else "image_evidence_unavailable",
                                                 "world_point_m": deepcopy(point) if present else None,
                                                 "ray_angle_degrees": 10. if present else None,
                                                 "ray_separation_m": 0. if present else None,
                                                 "reprojection_px": [0., 0.] if present else None}
                                                for point in points]})
    ground = {"schema_version": 1, "source_sha256": sources[0], "calibration_sha256": "0" * 64,
              "world_axes": "right_handed_z_up", "units": "m", "height_m": ground_height,
              "quality": "authored", "provenance": "Authored horizontal plane, not observed contact"}
    body = {"schema_version": 1, "profile": "exercise-v3", "source_sha256": sources[0], "dt": 1 / 30,
            "controller": {}, "lengths": {}, "body_frame": {"rotation": identity}, "motor_limits": {},
            "mobility": {"base_height": 1. - ground_height},
            "reference": [{"time_s": i / fps, "joints": {}} for i in sampled],
            "reference_evidence": [{"time_s": i / fps, "evidence_id": f"{sources[0]}:frame:{i}"} for i in sampled],
            "root_reference": [{"time_s": i / fps, "translation": [0., 0., 0.]} for i in sampled]}
    body["camera_reference"] = {
        "schema_version": 1, "method": api.ROOT_METHOD, "source_sha256": sources[0],
        "calibration_sha256": "0" * 64, "pose_evidence_sha256": "0" * 64,
        "root_reference_sha256": "0" * 64, "angular_reference_sha256": "0" * 64, "body_rotation_sha256": "0" * 64,
        "world_axes": "right_handed_z_up", "calibration_quality": "authored",
        "scale_assumption": "supplied_camera_scale_for_root_monocular_scale_for_shape",
        "origin_world_m": [1., 2., 1.], "source_frame_ids": [f"{sources[0]}:frame:{i}" for i in sampled],
        "ground": deepcopy(ground), "root_refinement": {
            "schema_version": 1, "input_body_sha256": "d" * 64, "original_root_reference_sha256": "e" * 64,
            "unchanged_shape_sha256": "0" * 64, "triangulation_sha256": "0" * 64,
            "world_provenance": "Authored shared-world fixture, not measured camera alignment",
            "views": [], "fps": fps, "source_frames": count,
            "roots": [{"frame": i, "time_s": i / fps, "source_evidence_ids": [f"{source}:frame:{i}" for source in sources],
                       "hip_world_m": [[1., 2., 1.], [1., 2., 1.]], "root_world_m": [1., 2., 1.]} for i in sampled]}}
    mapping = {"schema_version": 1, "profile": "world-alignment-v1",
               "from_frame": {"id": "pending", "length_unit": "m", "axes": "x_right_y_away_z_up"},
               "to_frame": {"id": "pending", "length_unit": "human_model_unit", "axes": "x_right_y_away_z_up"},
               "spatial": {"scale": 1., "rotation_wxyz": [1., 0., 0., 0.], "translation": [-1., -2., -ground_height]},
               "clock": {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.},
               "claims": {name: {"status": "assumed", "method": "Authored analytical fixture binding",
                                  "uncertainty": "Not independently measured", "evidence_refs": []} for name in ("spatial", "clock")}}
    return _pin({"predictions": prediction, "evidence": evidence, "cameras": cameras,
                 "body": body, "ground": ground, "mapping": mapping}, api)


def _arguments(fixture):
    return {"predictions_document": _encode(fixture["predictions"]),
            "first_evidence_document": _encode(fixture["evidence"][0]),
            "second_evidence_document": _encode(fixture["evidence"][1]),
            "camera_documents": [_encode(value) for value in fixture["cameras"]],
            "body_document": _encode(fixture["body"]), "ground_document": _encode(fixture["ground"]),
            "world_to_body_document": _encode(fixture["mapping"])}


def _run(api, fixture):
    return api.build_multiview_foot_reference(**_arguments(fixture))


def _foot(result, frame=0, side="left"):
    return result["frames"][frame]["feet"][side]


def test_known_mapping_fixed_shape_and_complete_clock(api):
    fixture = _fixture(api)
    before = _encode(fixture)
    result = _run(api, fixture)
    assert result["status"] == "conditional_references_available"
    assert result["profile"] == "calibrated-two-view-foot-reference-v1"
    assert result["binding_unavailable_reasons"] == []
    assert result["construction_cutoff_time_s"] == pytest.approx(.66)
    assert result["coverage"] == {"original_source_frames": 12, "view_decoded_frames": [12, 12],
                                  "view_metadata_frames": [12, 12], "pose_samples_by_view": [6, 6],
                                  "retained_foot_slots": 24, "retained_endpoint_slots": 72,
                                  "foot_status_counts": {"conditional_kinematic_reference": 8,
                                                         "pose_not_sampled_in_both_views": 6,
                                                         "excluded_later_source": 10}}
    foot = _foot(result)
    assert foot["source_geometry"]["source_world_points_m"]["ankle"] == [1.1, 2., .14]
    assert foot["source_geometry"]["source_body_points"]["ankle"] == pytest.approx([.1, 0, .14])
    assert foot["endpoint_geometry"]["heel_relative"] == pytest.approx([-.05, 0, -.1])
    assert foot["endpoint_geometry"]["foot_index_relative"] == pytest.approx([.15, 0, -.1])
    assert foot["reference"]["heel_to_index_axis"] == pytest.approx([1, 0, 0])
    assert result["fixed_shapes"]["left"]["fixed_parameters"] == pytest.approx({"length": .2, "along": .05, "height": .1})
    assert result["fixed_shapes"]["left"]["source_frames"] == [0, 2, 4, 6]
    assert result["body_model_sha256"] == api.canonical_hash(fixture["body"])
    assert result["world_to_body"] == fixture["mapping"]
    assert _encode(fixture) == before
    assert all(result[key] is False for key in ("body_geometry_validated", "body_or_floor_installed", "monocular_world_landmarks_used"))
    assert all(result[key] == "unknown" for key in ("contact", "load", "grasp"))
    assert foot["reference"]["sole_normal"] is foot["reference"]["anatomical_axial_roll_rad"] is None
    assert foot["endpoint_geometry"]["collisions"] is foot["endpoint_geometry"]["physical_model_attached"] is False


@pytest.mark.parametrize("missing,reason", [("camera_documents", "missing_camera_1"), ("body_document", "missing_body"),
                                           ("ground_document", "missing_ground"), ("world_to_body_document", "missing_world_to_body")])
def test_missing_calibration_is_complete_unavailable(api, missing, reason):
    arguments = _arguments(_fixture(api))
    arguments[missing] = None
    result = api.build_multiview_foot_reference(**arguments)
    assert result["status"] == "unavailable" and reason in result["binding_unavailable_reasons"]
    assert len(result["frames"]) == 12 and result["coverage"]["retained_endpoint_slots"] == 72
    assert _foot(result)["reason"] == "source_or_body_binding_unavailable"
    assert _foot(result, 8)["reason"] == "excluded_later_source"


@pytest.mark.parametrize("section,key,value", [("spatial", "scale", 2.), ("spatial", "translation", [0, 0, 0]),
    ("spatial", "rotation_wxyz", [0., 0., 0., 1.]), ("clock", "rate", 2.), ("clock", "target_origin_s", .1),
    ("from_frame", "length_unit", "centimetres"), ("from_frame", "id", "different-world"),
    ("to_frame", "id", "different-body"), ("to_frame", "axes", "x_right_y_down_z_forward"),
    ("claims", "spatial", {"status": "observed", "method": "unsupported", "uncertainty": "unknown", "evidence_refs": []})])
def test_incompatible_mapping_cannot_align_implicitly(api, section, key, value):
    fixture = _fixture(api)
    fixture["mapping"][section][key] = value
    result = _run(api, fixture)
    assert result["status"] == "unavailable"
    assert result["binding_unavailable_reasons"] == ["incompatible_body_world_clock_binding"]
    assert result["world_to_body"] is None


@pytest.mark.parametrize("target", ["evidence", "camera", "body", "ground"])
def test_exact_source_camera_body_and_ground_pin_drift_fails_closed(api, target):
    fixture = _fixture(api)
    if target == "evidence":
        fixture["evidence"][0]["extra"] = "different exact source document"
    elif target == "camera":
        fixture["cameras"][0]["K"][0][0] = 101.
    elif target == "body":
        fixture["body"]["root_reference"][0]["translation"][0] = .01
    else:
        fixture["ground"]["height_m"] = .01
    result = _run(api, fixture)
    assert result["status"] == "unavailable" and result["binding_unavailable_reasons"]
    assert _foot(result)["source_geometry"] is None


@pytest.mark.parametrize("field,value", [("camera_reference", None), ("camera_reference", []),
                                      ("body_frame", None), ("body_frame", []), ("mobility", None)])
def test_malformed_optional_body_binding_is_unavailable(api, field, value):
    fixture = _fixture(api)
    fixture["body"][field] = value
    result = _run(api, fixture)
    assert result["status"] == "unavailable"
    assert result["binding_unavailable_reasons"] == ["incompatible_body_world_clock_binding"]
    assert _foot(result)["source_geometry"] is None
    assert len(result["frames"]) == 12


def test_raw_byte_pins_preserve_whitespace_separately_from_legacy_hash(api):
    fixture = _fixture(api)
    arguments = _arguments(fixture)
    arguments["first_evidence_document"] = json.dumps(fixture["evidence"][0], indent=1).encode()
    result = api.build_multiview_foot_reference(**arguments)
    pin = result["input_pins"]["evidence_1"]
    assert pin["sha256"] == sha256(arguments["first_evidence_document"]).hexdigest()
    assert pin["canonical_sha256"] == fixture["predictions"]["views"][0]["evidence_sha256"]
    assert pin["bytes"] == len(arguments["first_evidence_document"])
    assert pin["sha256"] != pin["canonical_sha256"]
    assert result["status"] == "conditional_references_available"


def test_later_coordinates_and_monocular_depth_never_evaluated(api):
    fixture = _fixture(api)
    for row in fixture["predictions"]["frames"][7:]:
        for point in row["joints"]:
            point.update(world_point_m="excluded geometry poison", ray_angle_degrees="not read", reprojection_px="not read")
    for evidence in fixture["evidence"]:
        for row in evidence["frames"]:
            row["world_landmarks"] = {"forbidden": "entire monocular body"}
            if row["frame"] > 6:
                row["landmarks"] = "later landmark payload must not be evaluated"
    result = _run(api, _pin(fixture, api))
    assert result["coverage"]["foot_status_counts"]["conditional_kinematic_reference"] == 8
    assert _foot(result, 8)["source_geometry"] is None
    assert _foot(result, 8)["original_endpoint_statuses"]["ankle"] == "estimated"


def test_metadata_only_and_one_view_unsampled_slots_remain(api):
    fixture = _fixture(api)
    fixture["evidence"][1]["decoded_frames"] = 11
    fixture["evidence"][1]["frames"].pop(1)
    fixture["predictions"]["frames"][2]["source_evidence_ids"][1] = None
    result = _run(api, _pin(fixture, api))
    assert result["coverage"]["retained_foot_slots"] == 24
    assert result["coverage"]["view_decoded_frames"] == [12, 11]
    assert _foot(result, 2)["reason"] == "pose_not_sampled_in_both_views"
    assert _foot(result, 11)["reason"] == "one_or_more_views_not_decoded"
    assert result["frames"][11]["view_slot_statuses"] == ["decoded", "metadata_only_not_decoded"]


def test_missing_bound_body_pose_does_not_interpolate(api):
    fixture = _fixture(api)
    body = fixture["body"]
    for key in ("reference", "reference_evidence", "root_reference"):
        body[key].pop(1)
    body["camera_reference"]["source_frame_ids"].pop(1)
    body["camera_reference"]["root_refinement"]["roots"].pop(1)
    result = _run(api, _pin(fixture, api))
    assert result["binding_unavailable_reasons"] == []
    assert _foot(result, 2)["reason"] == "body_pose_unavailable"
    assert result["fixed_shapes"]["left"]["source_frames"] == [0, 4, 6]


@pytest.mark.parametrize("field,value", [("visibility", .49), ("presence", 1.1), ("x", -0.01), ("x", "invalid")])
def test_bad_image_endpoint_remains_unavailable(api, field, value):
    fixture = _fixture(api)
    fixture["evidence"][1]["frames"][0]["landmarks"][29][field] = value
    result = _run(api, _pin(fixture, api))
    assert _foot(result)["reason"] == "source_endpoint_unavailable"
    assert _foot(result)["source_geometry"] is None
    assert _foot(result, side="right")["status"] == "conditional_kinematic_reference"


def test_small_image_axis_and_original_ray_rejection_retained(api):
    fixture = _fixture(api)
    image = fixture["evidence"][0]["frames"][0]["landmarks"]
    image[31] = deepcopy(image[29])
    fixture["predictions"]["frames"][2]["joints"][29]["status"] = "insufficient_ray_angle"
    result = _run(api, _pin(fixture, api))
    assert _foot(result)["reason"] == "projected_axis_degenerate"
    assert _foot(result, 2)["reason"] == "triangulated_endpoint_unavailable"
    assert _foot(result, 2)["original_endpoint_statuses"]["heel"] == "insufficient_ray_angle"
    assert result["fixed_shapes"]["left"]["eligible_prefix_samples"] == 2
    assert _foot(result, 4)["reason"] == "insufficient_eligible_prefix_samples"
    assert _foot(result, 4)["source_geometry"] is not None


@pytest.mark.parametrize("key,value", [("ray_angle_degrees", 1.99), ("ray_separation_m", -1),
                                     ("reprojection_px", [6.41, 0]), ("world_point_m", [2001, 0, 0])])
def test_estimated_point_cannot_bypass_original_quality_envelope(api, key, value):
    fixture = _fixture(api)
    fixture["predictions"]["frames"][0]["joints"][29][key] = value
    with pytest.raises(ValueError):
        _run(api, _pin(fixture, api))


@pytest.mark.parametrize("ankle,heel,toe,reason", [
    ([1.1, 2, .14], [1.05, 2, .04], [1.06, 2, .04], "foot_length_out_of_bounds"),
    ([1.1, 2, .04], [1.05, 2, .04], [1.25, 2, .04], "ankle_axis_plane_degenerate_or_out_of_bounds"),
    ([2., 2, .14], [1.05, 2, .04], [1.25, 2, .04], "ankle_attachment_out_of_bounds")])
def test_unchanged_shape_bounds_retain_rejected_source_geometry(api, ankle, heel, toe, reason):
    fixture = _fixture(api)
    for index, point in zip((27, 29, 31), (ankle, heel, toe), strict=True):
        fixture["predictions"]["frames"][0]["joints"][index]["world_point_m"] = point
    result = _run(api, _pin(fixture, api))
    assert _foot(result)["reason"] == reason
    assert _foot(result)["source_geometry"]["source_world_points_m"]["ankle"] == ankle
    assert _foot(result)["endpoint_geometry"] is None
    assert result["fixed_shapes"]["left"]["source_frames"] == [2, 4, 6]


def test_signed_clearance_does_not_repair_floor_or_change_shape(api):
    result = _run(api, _fixture(api, ground_height=.08))
    foot = _foot(result)
    assert foot["source_geometry"]["signed_ground_clearance_m"]["heel"] == pytest.approx(-.04)
    assert foot["source_geometry"]["source_body_points"]["heel"][2] == pytest.approx(-.04)
    assert foot["source_geometry"]["source_world_points_m"]["heel"][2] == .04
    assert result["fixed_shapes"]["left"]["fixed_parameters"] == pytest.approx({"length": .2, "along": .05, "height": .1})
    assert foot["status"] == "conditional_kinematic_reference"
    assert result["contact"] == "unknown"


def test_more_than_900_original_slots_are_not_truncated_to_pose_cap(api):
    result = _run(api, _fixture(api, count=1002, fps=30, sampled=(0, 2, 4, 6)))
    assert result["coverage"]["original_source_frames"] == len(result["frames"]) == 1002
    assert result["coverage"]["retained_endpoint_slots"] == 6012
    assert result["coverage"]["pose_samples_by_view"] == [4, 4]
    assert result["frames"][-1]["frame"] == 1001
    assert result["frames"][-1]["time_s"] == 1001 / 30


@pytest.mark.parametrize("document,field,value", [("predictions", "schema_version", True),
    ("predictions", "source_frames", True), ("predictions", "joint_slots", 395),
    ("predictions", "monocular_world_landmarks_used", True), ("predictions", "reference_body_or_contacts_used", True),
    ("evidence", "source_sha256", None), ("evidence", "schema_version", True), ("evidence", "fps", 0)])
def test_malformed_source_contract_rejected(api, document, field, value):
    fixture = _fixture(api)
    target = fixture[document] if document == "predictions" else fixture["evidence"][0]
    target[field] = value
    with pytest.raises(ValueError):
        _run(api, fixture)


@pytest.mark.parametrize("field,value", [("id", "wrong-source:frame:0"), ("frame", True),
                                      ("decoder_timestamp_s", .001), ("time_s", .001)])
def test_pose_identity_and_original_clock_cannot_drift(api, field, value):
    fixture = _fixture(api)
    fixture["evidence"][0]["frames"][0][field] = value
    with pytest.raises(ValueError):
        _run(api, fixture)


@pytest.mark.parametrize("raw", [b'{"frames":[],"x":1,"x":2}', b'{"frames":[],"x":NaN}', b'[]', b'\xff'])
def test_invalid_json_rejected(api, raw):
    arguments = _arguments(_fixture(api))
    arguments["predictions_document"] = raw
    with pytest.raises(ValueError):
        api.build_multiview_foot_reference(**arguments)


@pytest.mark.parametrize("name,limit", [("predictions", 1800), ("evidence_1", 900), ("camera_1", 1800)])
def test_excess_rows_rejected_before_canonical_serialization(api, monkeypatch, name, limit):
    monkeypatch.setattr(api, "canonical_hash", _deny)
    with pytest.raises(ValueError, match="before canonical hashing"):
        api._documents({name: _encode({"frames": [None] * (limit + 1)})}, float("inf"))


def test_input_output_and_cooperative_deadline_bounds(api, monkeypatch):
    arguments = _arguments(_fixture(api))
    with monkeypatch.context() as context:
        context.setitem(api.CONTRACT, "maximum_input_bytes", 10)
        context.setattr(api.json, "loads", _deny)
        with pytest.raises(ValueError, match="exact JSON bytes"):
            api.build_multiview_foot_reference(**arguments)
    with monkeypatch.context() as context:
        context.setitem(api.CONTRACT, "maximum_output_bytes", 10)
        with pytest.raises(ValueError, match="output exceeds"):
            api.build_multiview_foot_reference(**arguments)
    with monkeypatch.context() as context:
        context.setitem(api.CONTRACT, "maximum_seconds", 0.)
        context.setattr(api.json, "loads", _deny)
        with pytest.raises(TimeoutError, match="deadline"):
            api.build_multiview_foot_reference(**arguments)


def test_deadline_checked_after_final_output_chunk(api, monkeypatch):
    arguments = _arguments(_fixture(api))
    actual = api.json.JSONEncoder
    clock = [0.]

    class LateEncoder(actual):
        def iterencode(self, value, _one_shot=False):
            yield from super().iterencode(value, _one_shot)
            clock[0] = 11.

    monkeypatch.setattr(api.json, "JSONEncoder", LateEncoder)
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    with pytest.raises(TimeoutError):
        api.build_multiview_foot_reference(**arguments)
