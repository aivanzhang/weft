"""Analytical consumer checks; no saved source studies, physics or media work."""

import builtins
from copy import deepcopy
from dataclasses import FrozenInstanceError
from hashlib import sha256
import importlib
import json
import os
import socket
import subprocess

import numpy as np
import pytest


def _deny(*_args, **_kwargs):
    raise AssertionError("Native/model/media/network/subprocess or repeated preparation is forbidden")


@pytest.fixture(scope="module", autouse=True)
def local_only():
    with pytest.MonkeyPatch.context() as patch:
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            roots = {"mujoco", "cv2", "PIL", "mediapipe", "torch", "tensorflow", "requests", "httpx"}
            if name.split(".")[0] in roots or name in {"weft.human", "weft.mobile_human", "weft.foot_body", "weft.shared_world"}:
                _deny()
            return original(name, *args, **kwargs)

        patch.setattr(builtins, "__import__", guarded)
        for owner, names in ((socket, ("socket", "create_connection")),
                             (subprocess, ("Popen", "run", "check_call", "check_output")),
                             (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
            for name in names:
                if hasattr(owner, name):
                    patch.setattr(owner, name, _deny)
        yield


@pytest.fixture(scope="module")
def api(local_only):
    return importlib.import_module("weft.calibrated_foot_reference")


def _raw(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode()


def _analytical_inputs(*, mutate=None, ground_height=0., count=12, fps=10, sampled=(0, 2, 4, 6, 8, 10), missing_camera=False):
    producer = importlib.import_module("weft.multiview_foot_reference")
    fixture_tools = importlib.import_module("test_multiview_foot_reference")
    fixture = fixture_tools._fixture(producer, ground_height=ground_height, count=count, fps=fps, sampled=sampled)
    fixture["body"]["schema_version"] = 3
    if mutate:
        mutate(fixture)
    fixture_tools._pin(fixture, producer)
    arguments = fixture_tools._arguments(fixture)
    if missing_camera:
        arguments["camera_documents"][0] = None
    reference = producer.build_multiview_foot_reference(**arguments)
    return arguments["body_document"], _raw(reference)


@pytest.fixture(scope="module")
def inputs(local_only):
    return _analytical_inputs()


def _changed_reference(inputs, mutate):
    body, raw = inputs
    reference = json.loads(raw)
    mutate(reference)
    return body, _raw(reference)


def test_distinct_consumer_preserves_bytes_complete_slots_and_uncertainty(api, inputs):
    body, raw = inputs
    prepared = api.load_calibrated_foot_reference(body, raw)
    description, initial = prepared.describe(), prepared.initial_geometry()
    assert prepared.ready and prepared.reference_bytes is raw and prepared.body_bytes is body
    assert description["profile"] == "calibrated-foot-consumer-v1"
    assert description["source_profile"] == "calibrated-two-view-foot-reference-v1"
    assert description["reference_sha256"] == sha256(raw).hexdigest()
    assert description["body_sha256"] == sha256(body).hexdigest()
    original = json.loads(raw)
    assert description["views"] == original["views"]
    assert description["source_input_pins"] == original["input_pins"]
    assert description["world_to_body"] == original["world_to_body"]
    assert description["coverage"] == original["coverage"]
    assert description["coverage"]["retained_endpoint_slots"] == 72
    assert initial["frame"] == 0 and initial["time_s"] == 0.
    assert initial["source_evidence_ids"] == original["frames"][0]["source_evidence_ids"]
    for side in ("left", "right"):
        assert initial["feet"][side]["fixed_parameters"] == original["fixed_shapes"][side]["fixed_parameters"]
        assert initial["feet"][side]["endpoint_geometry"] == original["frames"][0]["feet"][side]["endpoint_geometry"]
        assert initial["feet"][side]["local_centerline"]["heel"] == pytest.approx([-.05, 0, -.1])
        assert initial["feet"][side]["physical_admission"] is False
    assert description["body_clock_binding_validated"] is True
    assert description["body_geometry_validated"] is description["coupled_world_admitted"] is False
    assert all(description[key] == "unknown" for key in ("contact", "load", "grasp"))
    with pytest.raises(FrozenInstanceError):
        prepared.reference_bytes = b"changed"
    initial["feet"]["left"]["fixed_parameters"]["length"] = 999
    description["world_to_body"]["spatial"]["scale"] = 999
    assert prepared.initial_geometry()["feet"]["left"]["fixed_parameters"]["length"] == pytest.approx(.2)
    assert prepared.describe()["world_to_body"]["spatial"]["scale"] == 1


@pytest.mark.parametrize("left,right", [([0, 0, 0], [0, 0, 0]), ([.8, -.3, .2], [-.6, .4, -.9]),
                                       ([.2, np.pi / 2, -.5], [-.3, -np.pi / 2, .8])])
def test_exact_body_frame_ankle_chart_roundtrip(api, inputs, left, right):
    rotations = importlib.import_module("weft.spatial_pose")
    prepared = api.load_calibrated_foot_reference(*inputs)
    knees = {"left": rotations.rotation_zyx(left), "right": rotations.rotation_zyx(right)}
    angles = prepared.initial_ankle_angles(knees, coordinate_frame=prepared.describe()["body_coordinate_frame"])
    assert len(angles) == 6 and all(-np.pi * 2 <= value <= np.pi * 2 for value in angles)
    initial = prepared.initial_geometry()
    for index, side in enumerate(("left", "right")):
        actual = knees[side] @ rotations.rotation_zyx(angles[index * 3:index * 3 + 3])
        assert np.allclose(actual, initial["feet"][side]["rotation_in_body_frame"], atol=1e-10, rtol=0)


def test_control_queries_never_reparse_or_copy_full_evidence(api, inputs, monkeypatch):
    prepared = api.load_calibrated_foot_reference(*inputs)
    frame = prepared.describe()["body_coordinate_frame"]
    monkeypatch.setattr(api.json, "loads", _deny)
    monkeypatch.setattr(api, "canonical_hash", _deny)
    monkeypatch.setattr(api, "_parse", _deny)
    for _ in range(3):
        angles = prepared.initial_ankle_angles({side: np.eye(3) for side in ("left", "right")}, coordinate_frame=frame)
        assert angles == pytest.approx([0.] * 6)
    assert prepared.reference_bytes is inputs[1]


@pytest.mark.parametrize("field,value", [("id", "different-body"), ("length_unit", "m"),
                                      ("axes", "x_right_y_depth_z_up")])
def test_knee_coordinate_frame_requires_exact_identity(api, inputs, field, value):
    prepared = api.load_calibrated_foot_reference(*inputs)
    frame = prepared.describe()["body_coordinate_frame"]
    frame[field] = value
    with pytest.raises(ValueError, match="exact body frame"):
        prepared.initial_ankle_angles({side: np.eye(3) for side in ("left", "right")}, coordinate_frame=frame)


@pytest.mark.parametrize("matrix", [np.eye(3) * 2, np.diag([1., 1., -1.]), [[1, 0], [0, 1]],
                                   [[True, False, False]] * 3, [[float("nan"), 0, 0]] * 3, "invalid"])
def test_improper_knee_rotations_never_normalized(api, inputs, matrix):
    prepared = api.load_calibrated_foot_reference(*inputs)
    with pytest.raises(ValueError, match="proper rotation"):
        prepared.initial_ankle_angles({"left": matrix, "right": np.eye(3)},
                                      coordinate_frame=prepared.describe()["body_coordinate_frame"])


def test_missing_original_foot_does_not_choose_later_available_pose(api):
    def missing(fixture):
        fixture["predictions"]["frames"][0]["joints"][29]["status"] = "insufficient_ray_angle"

    body, raw = _analytical_inputs(mutate=missing)
    prepared = api.load_calibrated_foot_reference(body, raw)
    assert not prepared.ready and prepared.initial_geometry() is None
    assert prepared.describe()["unavailable_reasons"] == ["both_feet_unavailable_at_original_body_start"]
    assert prepared.describe()["original_start"]["frame"] == 0
    assert json.loads(prepared.reference_bytes)["frames"][2]["feet"]["left"]["status"] == "conditional_kinematic_reference"
    with pytest.raises(ValueError, match="original body start"):
        prepared.initial_ankle_angles({side: np.eye(3) for side in ("left", "right")}, coordinate_frame={})


def test_original_nonzero_body_start_is_not_confused_with_video_start(api):
    def first_two(fixture):
        body = fixture["body"]
        for key in ("reference", "reference_evidence", "root_reference"):
            body[key].pop(0)
        body["camera_reference"]["source_frame_ids"].pop(0)
        body["camera_reference"]["root_refinement"]["roots"].pop(0)

    prepared = api.load_calibrated_foot_reference(*_analytical_inputs(mutate=first_two))
    assert prepared.ready and prepared.initial_geometry()["frame"] == 2
    assert prepared.initial_geometry()["time_s"] == .2
    assert json.loads(prepared.reference_bytes)["frames"][0]["feet"]["left"]["reason"] == "body_pose_unavailable"


def test_unavailable_fixed_shape_and_missing_calibration_are_complete(api):
    def too_few(fixture):
        for frame in (4, 6):
            fixture["predictions"]["frames"][frame]["joints"][29]["status"] = "insufficient_ray_angle"

    for inputs in (_analytical_inputs(mutate=too_few), _analytical_inputs(missing_camera=True)):
        prepared = api.load_calibrated_foot_reference(*inputs)
        assert not prepared.ready
        assert prepared.reference_bytes == inputs[1]
        assert prepared.describe()["coverage"]["retained_foot_slots"] == 24
        assert prepared.initial_geometry() is None


def test_signed_source_gaps_remain_negative_and_unqualified(api):
    body, raw = _analytical_inputs(ground_height=.08)
    prepared = api.load_calibrated_foot_reference(body, raw)
    assert prepared.ready
    foot = prepared.initial_geometry()["feet"]["left"]
    assert foot["source_signed_ground_clearance_m"]["heel"] == pytest.approx(-.04)
    assert foot["endpoint_geometry"]["heel_relative"] == pytest.approx([-.05, 0, -.1])
    assert prepared.describe()["ground"]["quality"] == "authored"
    assert prepared.describe()["world_to_body"]["claims"]["spatial"]["status"] == "assumed"
    assert prepared.describe()["coupled_world_admitted"] is False
    assert prepared.reference_bytes is raw


@pytest.mark.parametrize("kind", ["forged_rotation", "erased_negative_gap", "body_map_disagreement",
                                 "world_map_disagreement", "changed_source_measurement"])
def test_saved_geometry_must_match_retained_source_triplet(api, inputs, kind):
    original = _analytical_inputs(ground_height=.08) if kind == "erased_negative_gap" else inputs

    def change(reference):
        foot = reference["frames"][0]["feet"]["left"]
        source = foot["source_geometry"]
        if kind == "forged_rotation":
            quarter_turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            for key in ("heel_to_index_axis", "axis_to_ankle_direction"):
                foot["reference"][key] = (quarter_turn @ foot["reference"][key]).tolist()
            for key in ("heel_relative", "foot_index_relative"):
                foot["endpoint_geometry"][key] = (quarter_turn @ foot["endpoint_geometry"][key]).tolist()
        elif kind == "erased_negative_gap":
            assert source["signed_ground_clearance_m"]["heel"] < 0
            source["signed_ground_clearance_m"]["heel"] = 0.
        elif kind == "body_map_disagreement":
            source["source_body_points"]["heel"][0] += .001
        elif kind == "world_map_disagreement":
            source["source_world_points_m"]["heel"][0] += .001
        else:
            source["inferred_measurements"]["along"] += .001

    with pytest.raises(ValueError, match="source (triplet|gap|mapping)"):
        api.load_calibrated_foot_reference(*_changed_reference(original, change))


def test_preserve_more_original_slots_than_body_pose_cap(api):
    prepared = api.load_calibrated_foot_reference(*_analytical_inputs(count=1002, fps=30, sampled=(0, 2, 4, 6)))
    assert prepared.ready
    assert prepared.describe()["coverage"]["original_source_frames"] == 1002
    assert len(json.loads(prepared.reference_bytes)["frames"]) == 1002
    assert prepared.describe()["coverage"]["retained_endpoint_slots"] == 6012


def test_reference_whitespace_preserved_but_body_bytes_are_exact(api, inputs):
    body, raw = inputs
    spaced = json.dumps(json.loads(raw), indent=2).encode()
    prepared = api.load_calibrated_foot_reference(body, spaced)
    assert prepared.ready and prepared.reference_bytes is spaced
    assert prepared.describe()["reference_sha256"] == sha256(spaced).hexdigest()
    assert prepared.describe()["reference_canonical_sha256"] == api.canonical_hash(json.loads(raw))
    different_body = json.dumps(json.loads(body), indent=2).encode()
    unavailable = api.load_calibrated_foot_reference(different_body, raw)
    assert not unavailable.ready and unavailable.body_bytes is different_body
    assert unavailable.describe()["unavailable_reasons"] == ["exact_original_body_bytes_unavailable"]


@pytest.mark.parametrize("target,key,value", [("spatial", "scale", 2.), ("spatial", "translation", [0, 0, 0]),
    ("spatial", "rotation_wxyz", [0., 0., 0., 1.]), ("clock", "target_origin_s", .1),
    ("from_frame", "id", "camera-world:different"), ("to_frame", "length_unit", "m")])
def test_changed_world_body_mapping_is_unavailable_not_fitted(api, inputs, target, key, value):
    def changed(reference):
        reference["world_to_body"][target][key] = value
        reference["input_pins"]["world_to_body"]["canonical_sha256"] = api.canonical_hash(reference["world_to_body"])

    prepared = api.load_calibrated_foot_reference(*_changed_reference(inputs, changed))
    assert not prepared.ready
    assert prepared.describe()["unavailable_reasons"] == ["incompatible_body_camera_world_clock_binding"]
    assert prepared.describe()["world_to_body"][target][key] == value


@pytest.mark.parametrize("name", ["predictions", "camera_1", "camera_2", "evidence_1", "evidence_2", "ground"])
def test_upstream_canonical_pin_mismatch_is_unavailable(api, inputs, name):
    arguments = _changed_reference(inputs, lambda r: r["input_pins"][name].update(canonical_sha256="f" * 64))
    prepared = api.load_calibrated_foot_reference(*arguments)
    assert not prepared.ready
    assert prepared.describe()["unavailable_reasons"] == ["incompatible_body_camera_world_clock_binding"]


@pytest.mark.parametrize("key,value", [("profile", "kinematic-foot-reference-v1"), ("schema_version", True),
    ("physical_validation", "verified"), ("contact", "measured"), ("load", "supported"),
    ("body_geometry_validated", True), ("body_or_floor_installed", True), ("monocular_world_landmarks_used", True)])
def test_old_profiles_and_promoted_claims_rejected(api, inputs, key, value):
    with pytest.raises(ValueError):
        api.load_calibrated_foot_reference(*_changed_reference(inputs, lambda r: r.update({key: value})))


@pytest.mark.parametrize("kind", ["frame_removed", "clock_drift", "decoder_drift", "source_id", "counter",
                                 "missing_endpoint_status", "later_geometry", "shape_later_frame"])
def test_complete_original_slot_partition_and_clock_guards(api, inputs, kind):
    def change(reference):
        row = reference["frames"][0]
        if kind == "frame_removed":
            reference["frames"].pop(1)
        elif kind == "clock_drift":
            row["time_s"] = .001
        elif kind == "decoder_drift":
            row["source_clocks"][0]["decoder_timestamp_s"] = .001
        elif kind == "source_id":
            row["source_evidence_ids"][0] = "wrong-source:frame:0"
        elif kind == "counter":
            reference["coverage"]["foot_status_counts"]["conditional_kinematic_reference"] += 1
        elif kind == "missing_endpoint_status":
            row["feet"]["left"]["original_endpoint_statuses"].pop("ankle")
        elif kind == "later_geometry":
            reference["frames"][8]["feet"]["left"] = deepcopy(row["feet"]["left"])
        elif kind == "shape_later_frame":
            reference["fixed_shapes"]["left"]["source_frames"][-1] = 8
    with pytest.raises(ValueError):
        api.load_calibrated_foot_reference(*_changed_reference(inputs, change))


def test_body_source_metadata_mismatch_is_unavailable_and_preserved(api, inputs):
    changed = _changed_reference(inputs, lambda r: r["coverage"].update(view_metadata_frames=[11, 11]))
    prepared = api.load_calibrated_foot_reference(*changed)
    assert not prepared.ready and prepared.reference_bytes == changed[1]
    assert prepared.describe()["unavailable_reasons"] == ["incompatible_body_camera_world_clock_binding"]
    assert prepared.describe()["coverage"]["view_metadata_frames"] == [11, 11]


@pytest.mark.parametrize("field", ["count", "clock"])
def test_boolean_clock_and_status_count_cannot_alias_numeric_values(api, inputs, field):
    def change(reference):
        if field == "clock":
            reference["frames"][0]["source_clocks"][0]["time_s"] = False
        else:
            reference["coverage"]["foot_status_counts"]["conditional_kinematic_reference"] = True
    with pytest.raises(ValueError):
        api.load_calibrated_foot_reference(*_changed_reference(inputs, change))


def test_oversized_knee_matrix_rejected_before_array_allocation(api, inputs, monkeypatch):
    prepared = api.load_calibrated_foot_reference(*inputs)
    frame = prepared.describe()["body_coordinate_frame"]
    large = [[0.] * 1000] * 1000
    monkeypatch.setattr(api.np, "asarray", _deny)
    with pytest.raises(ValueError, match="bounded 3×3"):
        prepared.initial_ankle_angles({"left": large, "right": large}, coordinate_frame=frame)


@pytest.mark.parametrize("kind", ["shape_length", "old_shape_status", "scaled_axis", "collinear_axes", "changed_endpoint",
                                 "wrong_frame", "attached", "sole", "missing_gaps"])
def test_endpoint_geometry_never_repaired_or_relabelled(api, inputs, kind):
    def change(reference):
        foot = reference["frames"][0]["feet"]["left"]
        if kind == "shape_length":
            reference["fixed_shapes"]["left"]["fixed_parameters"]["length"] = .6
        elif kind == "old_shape_status":
            reference["fixed_shapes"]["left"]["status"] = "inferred_fixed_triangle"
        elif kind == "scaled_axis":
            foot["reference"]["heel_to_index_axis"] = [.99, 0, 0]
        elif kind == "collinear_axes":
            foot["reference"]["axis_to_ankle_direction"] = [1, 0, 0]
        elif kind == "changed_endpoint":
            foot["endpoint_geometry"]["heel_relative"][0] += .001
        elif kind == "wrong_frame":
            foot["endpoint_geometry"]["coordinate_frame"]["id"] = "different-body"
        elif kind == "attached":
            foot["endpoint_geometry"]["physical_model_attached"] = True
        elif kind == "sole":
            foot["reference"]["sole_normal"] = [0, 0, 1]
        else:
            foot["source_geometry"].pop("signed_ground_clearance_m")
    with pytest.raises(ValueError):
        api.load_calibrated_foot_reference(*_changed_reference(inputs, change))


@pytest.mark.parametrize("raw", [b'[]', b'\xff', b'{"frames":[],"bad":NaN}', b'{"frames":[],"frames":[]}'])
def test_invalid_reference_json_rejected(api, inputs, raw):
    with pytest.raises(ValueError):
        api.load_calibrated_foot_reference(inputs[0], raw)


@pytest.mark.parametrize("name,key,maximum", [("body", "reference", 900), ("reference", "frames", 1800)])
def test_row_caps_precede_whole_canonical_serialization(api, monkeypatch, name, key, maximum):
    raw = _raw({key: [None] * (maximum + 1)})
    monkeypatch.setattr(api, "canonical_hash", _deny)
    monkeypatch.setattr(api, "_json_size", _deny)
    with pytest.raises(ValueError, match="row envelope"):
        api._parse(raw, name, float("inf"))


def test_byte_and_deadline_limits_fail_before_parse(api, inputs, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(api, "MAX_DOCUMENT_BYTES", 8)
        patch.setattr(api.json, "loads", _deny)
        with pytest.raises(ValueError, match="exact JSON"):
            api.load_calibrated_foot_reference(*inputs)
    with monkeypatch.context() as patch:
        patch.setattr(api, "MAX_SECONDS", 0.)
        patch.setattr(api.json, "loads", _deny)
        with pytest.raises(TimeoutError, match="deadline"):
            api.load_calibrated_foot_reference(*inputs)


def test_terminal_cache_encoding_cannot_cross_deadline(api, inputs, monkeypatch):
    actual = api.json.dumps
    clock = [0.]

    def late(value, *args, **kwargs):
        result = actual(value, *args, **kwargs)
        if isinstance(value, dict) and value.get("profile") == api.PROFILE:
            clock[0] = 6.
        return result

    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(api.json, "dumps", late)
    with pytest.raises(TimeoutError, match="deadline"):
        api.load_calibrated_foot_reference(*inputs)
