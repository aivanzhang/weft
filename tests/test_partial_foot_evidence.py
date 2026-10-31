"""Analytical partial-evidence checks; no source, native, inference or ray reconstruction."""

import builtins
from dataclasses import FrozenInstanceError
import importlib
import json
import os
import socket
import subprocess
import time

import pytest


def _deny(*_args, **_kwargs):
    raise AssertionError("Native/model/media/network/subprocess, geometry reconstruction or repeated preparation forbidden")


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if (name.split(".")[0] in {"mujoco", "cv2", "PIL", "gymnasium", "mediapipe", "torch", "tensorflow", "requests", "httpx"}
                or name in {"weft.human", "weft.mobile_human", "weft.foot_body", "weft.shared_world"}):
            _deny()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for owner, names in ((socket, ("socket", "create_connection")),
                         (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                         (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, _deny)
    for path, names in (("weft.multiview_pose", ("triangulate_pair", "_triangulate_rays")),
                        ("weft.multiview_root", ("make_two_view_root", "triangulate_pair")),
                        ("weft.camera_body", ("make_camera_body", "build_camera_reference")),
                        ("weft.camera_ground", ("apply_camera_ground", "audit_ground_clearance"))):
        for name in names:
            monkeypatch.setattr(importlib.import_module(path), name, _deny)


@pytest.fixture
def api():
    return importlib.import_module("weft.partial_foot_evidence")


def _raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()


def _inputs(*, change=None, ground_height=0., missing_camera=False, count=12, sampled=(0, 2, 4, 6, 8, 10), fps=10):
    producer = importlib.import_module("weft.multiview_foot_reference")
    fixture_tools = importlib.import_module("test_multiview_foot_reference")
    fixture = fixture_tools._fixture(producer, ground_height=ground_height, count=count, fps=fps, sampled=sampled)
    fixture["body"]["schema_version"] = 3
    for row in fixture["predictions"]["frames"]:
        for point in row["joints"]:
            point["observation_statuses"] = ["available", "available"] if point["status"] == "estimated" else ["pose_not_sampled"] * 2
    heel = fixture["predictions"]["frames"][sampled[0]]["joints"][30]
    heel.update(status="inconsistent_image_rays", world_point_m=None)
    if change:
        change(fixture)
    fixture_tools._pin(fixture, producer)
    arguments = fixture_tools._arguments(fixture)
    if missing_camera:
        arguments["camera_documents"][0] = None
    reference = producer.build_multiview_foot_reference(**arguments)
    return arguments["body_document"], _raw(reference), arguments["predictions_document"]


@pytest.fixture
def inputs():
    return _inputs()


def test_partial_ankle_toe_evidence_does_not_complete_original_start_heel(api, inputs):
    prepared = api.load_partial_foot_evidence(*inputs)
    metadata, row = prepared.describe(), prepared.frame(0)
    strict = importlib.import_module("weft.calibrated_foot_reference").load_calibrated_foot_reference(*inputs[:2])
    assert not strict.ready and strict.initial_geometry() is None
    assert metadata["profile"] == "partial-calibrated-foot-evidence-v1"
    assert metadata["strict_consumer"]["status"] == "unavailable"
    assert metadata["strict_consumer"]["ready"] is False
    assert metadata["original_start"]["frame"] == 0 and metadata["original_start"]["time_s"] == 0
    assert prepared.body_bytes is inputs[0] and prepared.reference_bytes is inputs[1] and prepared.triangulation_bytes is inputs[2]
    assert prepared.frame_count == metadata["coverage"]["source_slots"] == 12
    assert metadata["coverage"]["foot_slots"] == 24 and metadata["coverage"]["endpoint_slots"] == 72
    assert metadata["body_clock_binding_validated"] is True
    right = row["feet"]["right"]
    assert right["strict_reference_status"] == "unavailable"
    assert right["strict_reference_reason"] == "triangulated_endpoint_unavailable"
    for name in ("ankle", "foot_index"):
        point = right["endpoints"][name]
        assert point["original_status"] == "estimated" and point["geometry_status"] == "conditional_saved_estimate"
        assert point["confidence"] is None and point["confidence_status"] == "not_stored_in_saved_triangulation"
        assert point["stored_ray_diagnostics"]["observation_statuses"] == ["available", "available"]
    assert right["endpoints"]["ankle"]["source_world_point_m"] == [1.5, 2., .14]
    assert right["endpoints"]["ankle"]["body_point"] == [0.5, 0., .14]
    for side in ("left", "right"):
        heel = row["feet"][side]["endpoints"]["heel"]
        assert heel["geometry_status"] == "unsupported_by_partial_profile"
        assert heel["body_point"] is heel["source_world_point_m"] is heel["stored_ray_diagnostics"] is None
        assert row["feet"][side]["heel_orientation"] is None
    assert row["feet"]["left"]["endpoints"]["heel"]["original_status"] == "estimated"
    assert right["endpoints"]["heel"]["original_status"] == "inconsistent_image_rays"
    assert metadata["physical_admission"] is False
    assert all(metadata[name] == "unknown" for name in ("contact", "load", "grasp"))
    assert metadata["confidence"]["rule_is_not_a_measurement"] is True


def test_excluded_and_unsampled_slots_keep_status_without_reading_coordinates(api, monkeypatch):
    def change(fixture):
        for row in fixture["predictions"]["frames"]:
            if row["frame"] >= 8 or row["frame"] % 2:
                for index in (27, 28, 29, 30, 31, 32):
                    row["joints"][index]["world_point_m"] = "WITHHELD: not a coordinate"
                    row["joints"][index]["reprojection_px"] = "WITHHELD: not diagnostics"

    inputs = _inputs(change=change)
    endpoint = api._endpoint
    visited = []

    def guard(row, *args):
        assert row["frame"] < 8 and row["frame"] % 2 == 0
        visited.append(row["frame"])
        return endpoint(row, *args)

    monkeypatch.setattr(api, "_endpoint", guard)
    prepared = api.load_partial_foot_evidence(*inputs)
    assert visited and set(visited) == {0, 2, 4, 6}
    for index, expected in [(1, "pose_not_sampled_in_both_views"), (8, "excluded_later_source")]:
        row = prepared.frame(index)
        assert row["row_geometry_exclusion"] == expected
        assert row["feet"]["right"]["endpoints"]["ankle"]["geometry_status"] == expected
        assert row["feet"]["right"]["endpoints"]["ankle"]["source_world_point_m"] is None


def test_missing_binding_returns_all_slots_without_geometry(api):
    prepared = api.load_partial_foot_evidence(*_inputs(missing_camera=True))
    metadata = prepared.describe()
    assert metadata["status"] == "unavailable" and metadata["body_clock_binding_validated"] is False
    assert metadata["coverage"]["endpoint_slots"] == 72
    assert prepared.frame(0)["feet"]["right"]["endpoints"]["ankle"]["geometry_status"] == "body_world_binding_unavailable"


def test_original_source_gaps_remain_separate_from_partial_endpoint_series(api):
    inputs = _inputs(ground_height=.05)
    reference = json.loads(inputs[1])
    original_gaps = [gap for row in reference["frames"] for foot in row["feet"].values()
                     if foot["source_geometry"] for gap in foot["source_geometry"]["signed_ground_clearance_m"].values()]
    prepared = api.load_partial_foot_evidence(*inputs)
    assert prepared.reference_bytes is inputs[1]
    assert prepared.describe()["partial_endpoint_gap_summary"]["negative"] > 0
    assert any(value < 0 for value in original_gaps)
    assert len(original_gaps) != prepared.describe()["partial_endpoint_gap_summary"]["count"]
    assert prepared.frame(0)["feet"]["right"]["endpoints"]["foot_index"]["partial_endpoint_signed_ground_gap_m"] == pytest.approx(-.01)
    assert json.loads(prepared.reference_bytes) == reference


@pytest.mark.parametrize("which", [0, 1, 2])
def test_raw_input_changes_cannot_silently_rebind_saved_coordinates(api, inputs, which):
    changed = list(inputs)
    if which == 0:
        changed[which] += b" "
        prepared = api.load_partial_foot_evidence(*changed)
        assert not prepared.describe()["body_clock_binding_validated"]
        assert prepared.describe()["status"] == "unavailable"
    elif which == 1:
        value = json.loads(changed[which]); value["ground"]["height_m"] += .1
        changed[which] = _raw(value)
        prepared = api.load_partial_foot_evidence(*changed)
        assert prepared.describe()["status"] == "unavailable"
    else:
        changed[which] += b" "
        with pytest.raises(ValueError, match="triangulation bytes"):
            api.load_partial_foot_evidence(*changed)


@pytest.mark.parametrize("change", ["point", "observation_status", "reprojection", "angle", "ray_separation"])
def test_malformed_eligible_saved_endpoint_is_rejected_even_when_upstream_foot_was_missing(api, change):
    def alter(fixture):
        point = fixture["predictions"]["frames"][0]["joints"][28]
        if change == "point": point["world_point_m"] = [1., 2.]
        elif change == "observation_status": point["observation_statuses"] = ["available", "low_confidence"]
        elif change == "reprojection": point["reprojection_px"] = [-1., 0.]
        elif change == "angle": point["ray_angle_degrees"] = 0.
        elif change == "ray_separation": point["ray_separation_m"] = -1.

    # The complete-foot producer skipped this triangle at its missing heel; the partial boundary must check it.
    inputs = _inputs(change=alter)
    with pytest.raises(ValueError, match="estimated point|image-observation statuses|ray diagnostics"):
        api.load_partial_foot_evidence(*inputs)


@pytest.mark.parametrize("change", ["row_clock", "source_id"])
def test_changed_original_clock_or_source_identity_cannot_rebind(api, inputs, change):
    triangle = json.loads(inputs[2])
    if change == "row_clock": triangle["frames"][0]["time_s"] = .1
    else: triangle["frames"][0]["source_evidence_ids"][1] = "incorrect"
    with pytest.raises(ValueError, match="triangulation bytes"):
        api.load_partial_foot_evidence(inputs[0], inputs[1], _raw(triangle))


def test_preparation_uses_existing_strict_consumer_without_mutating_it(api, inputs, monkeypatch):
    load = api.load_calibrated_foot_reference
    calls = []

    def counted(*args):
        result = load(*args)
        calls.append(result.describe())
        return result

    monkeypatch.setattr(api, "load_calibrated_foot_reference", counted)
    prepared = api.load_partial_foot_evidence(*inputs)
    assert len(calls) == 1 and calls[0]["status"] == "unavailable"
    assert prepared.describe()["strict_consumer"]["unavailable_reasons"] == calls[0]["unavailable_reasons"]


def test_queries_only_parse_one_small_cache_and_never_prepare_full_evidence(api, inputs, monkeypatch):
    prepared = api.load_partial_foot_evidence(*inputs)
    monkeypatch.setattr(api, "load_calibrated_foot_reference", _deny)
    monkeypatch.setattr(api, "_documents", _deny)
    monkeypatch.setattr(api, "canonical_hash", _deny)
    loads = api.json.loads
    sizes = []

    def bounded(raw, *args, **kwargs):
        sizes.append(len(raw))
        assert len(raw) <= api.MAX_METADATA_BYTES
        return loads(raw, *args, **kwargs)

    monkeypatch.setattr(api.json, "loads", bounded)
    first = prepared.frame(0)
    first["feet"]["right"]["endpoints"]["ankle"]["body_point"][0] = 999
    assert prepared.frame(0)["feet"]["right"]["endpoints"]["ankle"]["body_point"][0] == .5
    assert prepared.describe()["profile"] == api.PROFILE
    cached = list(prepared.iter_frame_bytes())
    assert len(cached) == prepared.frame_count and cached[0] is prepared._frames[0]
    assert len(sizes) == 3
    with pytest.raises(FrozenInstanceError):
        prepared.reference_bytes = b"replacement"


@pytest.mark.parametrize("index", [-1, 12, True, .5, "0"])
def test_out_of_range_or_noninteger_frame_query_rejects(api, inputs, index):
    prepared = api.load_partial_foot_evidence(*inputs)
    with pytest.raises(ValueError, match="frame index"):
        prepared.frame(index)


@pytest.mark.parametrize("payload", [b"", "not-bytes", b"x" * ((8 << 20) + 1)], ids=["empty", "wrong-type", "oversized"])
def test_input_envelope_precedes_strict_preparation(api, inputs, monkeypatch, payload):
    monkeypatch.setattr(api, "load_calibrated_foot_reference", _deny)
    with pytest.raises(ValueError, match="bounded exact"):
        api.load_partial_foot_evidence(inputs[0], inputs[1], payload)


def test_final_cooperative_deadline_cannot_publish_a_late_prepared_cache(api, inputs, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    encode = api._encoded

    def late(value, limit):
        raw = encode(value, limit)
        if value.get("profile") == api.PROFILE:
            clock[0] = 6.
        return raw

    monkeypatch.setattr(api, "_encoded", late)
    with pytest.raises(TimeoutError):
        api.load_partial_foot_evidence(*inputs)


def test_cache_bound_never_drops_rows_to_fit(api, inputs, monkeypatch):
    monkeypatch.setattr(api, "MAX_OUTPUT_BYTES", api.MAX_METADATA_BYTES + 1)
    with pytest.raises(ValueError, match="cache exceeds"):
        api.load_partial_foot_evidence(*inputs)


def test_large_original_clock_uses_bounded_row_queries(api, monkeypatch, record_testsuite_property):
    inputs = _inputs(count=1002, fps=30, sampled=tuple(range(0, 1002, 2)))
    started = time.perf_counter()
    prepared = api.load_partial_foot_evidence(*inputs)
    prepare_seconds = time.perf_counter() - started
    assert prepared.frame_count == 1002
    coverage = prepared.describe()["coverage"]
    assert coverage["foot_slots"] == 2004 and coverage["endpoint_slots"] == 6012
    monkeypatch.setattr(api, "load_calibrated_foot_reference", _deny)
    monkeypatch.setattr(api, "_documents", _deny)
    monkeypatch.setattr(api, "_points", _deny)
    loads, parsed = api.json.loads, []

    def bounded(raw, *args, **kwargs):
        assert len(raw) <= api.MAX_ROW_BYTES
        parsed.append(len(raw))
        return loads(raw, *args, **kwargs)

    monkeypatch.setattr(api.json, "loads", bounded)
    started = time.perf_counter()
    for index in range(20):
        assert prepared.frame(index * 50)["frame"] == index * 50
    query_seconds = time.perf_counter() - started
    assert len(parsed) == 20
    record_testsuite_property("large_input_bytes", sum(map(len, inputs)))
    record_testsuite_property("large_source_slots", prepared.frame_count)
    record_testsuite_property("large_prepare_seconds", prepare_seconds)
    record_testsuite_property("twenty_row_queries_seconds", query_seconds)
    record_testsuite_property("largest_query_bytes", max(parsed))
    record_testsuite_property("retained_cache_bytes", sum(map(len, prepared._frames)) + len(prepared._metadata))
