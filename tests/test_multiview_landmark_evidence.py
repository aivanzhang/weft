"""Analytical source binding and additional-view geometry; no original video work."""

import builtins
from copy import deepcopy
from hashlib import sha256
import importlib
import json
import os
import socket
import subprocess
import time

import numpy as np
import pytest


def denied(*args, **kwargs):
    raise AssertionError("Native/model/media/network/process work is forbidden")


@pytest.fixture(autouse=True)
def local_only(monkeypatch):
    original = builtins.__import__
    blocked = ("mujoco", "gymnasium", "cv2", "torch", "mediapipe", "tensorflow",
               "requests", "httpx", "weft.human", "weft.foot_body", "weft.shared_world")

    def imported(name, *args, **kwargs):
        if any(name == item or name.startswith(item + ".") for item in blocked):
            denied()
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", imported)
    for owner, names in ((socket, ("socket", "create_connection")),
                         (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                         (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
        for name in names:
            if hasattr(owner, name):
                monkeypatch.setattr(owner, name, denied)


@pytest.fixture
def api():
    return importlib.import_module("weft.multiview_landmark_evidence")


def encode(value):
    return json.dumps(value, allow_nan=False, separators=(",", ":")).encode()


def canonical(value):
    return sha256(json.dumps(value, allow_nan=False, sort_keys=True).encode()).hexdigest()


def fixtures(count=12, sampled=None, views=3, fps=30):
    sampled = tuple(range(0, count, 2)) if sampled is None else sampled
    evidence, cameras = [], []
    translations = ([0., 0., 5.], [-1., 0., 5.], [0., -1., 5.], [1., 1., 5.])
    for view in range(views):
        source = str(view + 1) * 64
        camera = {"schema_version": 1, "source_sha256": source, "width": 640, "height": 480,
                  "fps": fps, "frame_count": count, "world_axes": "right_handed_z_up", "units": "m",
                  "provenance": "Authored analytical camera, not source measurements", "quality": "authored",
                  "K": [[640., 0., 320.], [0., 640., 240.], [0., 0., 1.]], "radial_k1": 0.,
                  "frames": [{"frame": index, "time_s": index / fps, "R": np.eye(3).tolist(),
                              "t": translations[view]} for index in range(count)]}
        rows = []
        for index in sampled:
            point = np.array([.2 + index * .0001, .1, .15])
            q = point + translations[view]
            pixel = np.array(camera["K"]) @ (q / q[2])
            landmarks = [{"x": float(pixel[0] / 640), "y": float(pixel[1] / 480), "z": 123.,
                          "visibility": .9, "presence": .95} for _ in range(33)]
            rows.append({"frame": index, "id": source + ":frame:" + str(index),
                         "time_s": index / fps, "decoder_timestamp_s": index / fps, "pose_count": 1,
                         "status": "low_confidence", "identity_status": "authored_single_model_hypothesis",
                         "landmarks": landmarks, "world_landmarks": "forbidden monocular depth"})
        evidence.append({"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": source,
                         "width": 640, "height": 480, "fps": fps, "frame_count": count, "decoded_frames": count,
                         "model": {"sha256": str(view + 4) * 64, "name": "Analytical model " + str(view),
                                   "running_mode": "IMAGE"}, "frames": rows})
        cameras.append(camera)
    return evidence, cameras


def mapping(camera, primary):
    frame = lambda value: {"id": "camera-world:" + canonical(value), "length_unit": "m", "axes": "x_right_y_depth_z_up"}
    claim = {"status": "assumed", "method": "Declared common coordinate system and original frame alignment",
             "uncertainty": "Analytical fixture, not a measured calibration", "evidence_refs": []}
    return {"schema_version": 1, "profile": "world-alignment-v1", "from_frame": frame(camera), "to_frame": frame(primary),
            "spatial": {"scale": 1., "rotation_wxyz": [1., 0., 0., 0.], "translation": [0., 0., 0.]},
            "clock": {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.},
            "claims": {"spatial": deepcopy(claim), "clock": deepcopy(claim)}}


def inputs(evidence, cameras):
    return [{"name": "view-" + str(index), "evidence_bytes": encode(pose), "camera_bytes": encode(camera),
             "clock_provenance": "Explicit aligned original frames; not independently measured"}
            for index, (pose, camera) in enumerate(zip(evidence, cameras, strict=True))]


def prepare(api, evidence, cameras, *, indices=(27, 28, 29, 30, 31, 32), alignments="default"):
    if alignments == "default":
        alignments = [encode(mapping(camera, cameras[0])) for camera in cameras[1:]]
    return api.prepare_multiview_landmark_evidence(inputs(evidence, cameras), landmark_indices=indices, alignments=alignments)


@pytest.mark.parametrize("view_count", [2, 3, 4])
def test_exact_geometry_uses_every_supplied_view_and_all_slots(api, view_count):
    evidence, cameras = fixtures(views=view_count)
    result = prepare(api, evidence, cameras)
    metadata = result.describe()
    assert metadata["profile"] == "calibrated-multiview-landmark-evidence-v1"
    assert metadata["world_clock_binding_available"] is True
    assert metadata["geometry_installed"] is False and metadata["physical_validation"] == "unverified"
    assert metadata["coverage"] == {
        "original_source_frames": 12, "construction_source_frames": 7, "excluded_later_source_frames": 5,
        "landmark_slots": 72, "view_landmark_slots": 72 * view_count,
        "status_counts": {"conditional_estimate": 24, "insufficient_qualified_views": 18, "excluded_later_source": 30},
        "observation_status_counts": {"available": 24 * view_count, "pose_not_sampled": 18 * view_count,
                                      "excluded_later_source": 30 * view_count},
        "solver_calls": 24}
    for index in (0, 2, 4, 6):
        for point in result.frame(index)["landmarks"]:
            assert point["status"] == "conditional_estimate"
            assert np.allclose(point["world_point_m"], [.2 + index * .0001, .1, .15], atol=1e-10)
            assert point["solver"]["view_count"] == view_count and len(point["qualified_view_names"]) == view_count
            assert len(point["solver"]["reprojection_px"]) == view_count
            assert point["solver"]["unit_pixel_sensitivity_m"] > 0
            assert point["contact_load_grasp"] == "unknown"
    assert len(list(result.iter_frame_bytes())) == 12


def test_conflicting_third_ray_is_never_discarded(api):
    evidence, cameras = fixtures()
    evidence[2]["frames"][0]["landmarks"][30]["y"] += .2
    result = prepare(api, evidence, cameras, indices=(30,))
    point = result.frame(0)["landmarks"][0]
    assert point["status"] in ("inconsistent_image_rays", "optimization_incomplete")
    assert point["world_point_m"] is None and point["solver"]["point_proposal_m"] is not None
    assert point["solver"]["view_count"] == 3 and point["qualified_view_names"] == ["view-0", "view-1", "view-2"]
    assert all(row["status"] == "available" for row in point["observations"])


def test_missing_first_point_is_not_replaced_by_later_pose(api):
    evidence, cameras = fixtures()
    for source in evidence[:2]:
        source["frames"][0]["landmarks"][30]["presence"] = .1
    result = prepare(api, evidence, cameras, indices=(30,))
    first = result.frame(0)["landmarks"][0]
    assert first["status"] == "insufficient_qualified_views" and first["world_point_m"] is None
    assert [row["model_presence"] for row in first["observations"]] == [.1, .1, .95]
    assert result.frame(2)["landmarks"][0]["status"] == "conditional_estimate"
    assert result.frame(0)["frame"] == 0 and result.frame_count == 12


@pytest.mark.parametrize("fault", ["missing", "unknown", "translation", "clock", "camera_pin"])
def test_unavailable_alignment_never_drops_a_view_or_invokes_solver(api, monkeypatch, fault):
    evidence, cameras = fixtures()
    mappings = [mapping(camera, cameras[0]) for camera in cameras[1:]]
    if fault == "unknown":
        mappings[-1]["spatial"] = None
        mappings[-1]["claims"]["spatial"]["status"] = "unknown"
    elif fault == "translation":
        mappings[-1]["spatial"]["translation"][0] = .01
    elif fault == "clock":
        mappings[-1]["clock"]["target_origin_s"] = .01
    elif fault == "camera_pin":
        mappings[-1]["from_frame"]["id"] = "camera-world:" + "f" * 64
    encoded = [encode(value) for value in mappings]
    if fault == "missing":
        encoded[-1] = None
    monkeypatch.setattr(api, "triangulate_pixels", denied)
    result = prepare(api, evidence, cameras, indices=(30,), alignments=encoded)
    assert result.describe()["world_clock_binding_available"] is False
    assert result.describe()["coverage"]["solver_calls"] == 0
    assert result.frame(0)["landmarks"][0]["status"] == "world_clock_binding_unavailable"
    assert len(result.frame(0)["landmarks"][0]["observations"]) == 3


def test_later_source_cannot_justify_measured_alignment(api):
    evidence, cameras = fixtures()
    alignment = mapping(cameras[1], cameras[0])
    alignment["claims"]["spatial"].update(status="inferred", evidence_refs=[
        {"document_sha256": canonical(evidence[0]), "evidence_id": evidence[0]["frames"][-1]["id"]}])
    with pytest.raises(ValueError, match="excluded evidence"):
        prepare(api, evidence, cameras, alignments=[encode(alignment), encode(mapping(cameras[2], cameras[0]))])


def test_mixed_model_exact_bytes_and_confidence_are_not_physical_truth(api):
    evidence, cameras = fixtures()
    supplied = inputs(evidence, cameras)
    result = api.prepare_multiview_landmark_evidence(supplied, landmark_indices=(30,),
        alignments=[encode(mapping(camera, cameras[0])) for camera in cameras[1:]])
    meta = result.describe()
    assert [row["model"] for row in meta["views"]] == [item["model"] for item in evidence]
    assert len({row["model"]["sha256"] for row in meta["views"]}) == 3
    for index, view in enumerate(supplied):
        assert result.view_inputs[index][1] is view["evidence_bytes"]
        assert result.view_inputs[index][2] is view["camera_bytes"]
        assert meta["views"][index]["evidence_pin"]["sha256"] == sha256(view["evidence_bytes"]).hexdigest()
    assert meta["uncertainty"]["image_confidence"] == "stored_model_visibility_presence_not_calibrated_accuracy"
    assert meta["uncertainty"]["comparison"].endswith("not_independent_accuracy_evidence")
    assert result.frame(0)["landmarks"][0]["observations"][0]["pose_status"] == "low_confidence"


def test_later_landmark_payload_and_monocular_depth_are_never_used(api, monkeypatch):
    evidence, cameras = fixtures()
    for source in evidence:
        for row in source["frames"]:
            if row["frame"] >= 7:
                row["landmarks"] = "Poison: no later image payload evaluation"
            row["world_landmarks"] = {"poison": "Never used at any frame"}
    original = api._observation
    seen = []

    def observed(pose, *args):
        if pose is not None:
            assert pose["frame"] < 7
            seen.append(pose["frame"])
        return original(pose, *args)

    monkeypatch.setattr(api, "_observation", observed)
    result = prepare(api, evidence, cameras, indices=(30,))
    assert set(seen) == {0, 2, 4, 6}
    for index in range(7, 12):
        point = result.frame(index)["landmarks"][0]
        assert point["status"] == "excluded_later_source" and point["solver"] is None
        assert all(row["pixel"] is row["model_presence"] is None for row in point["observations"])


def test_metadata_only_missing_and_occluded_slots_are_preserved(api):
    evidence, cameras = fixtures()
    evidence[2]["decoded_frames"] = 11
    evidence[1]["frames"][0]["pose_count"] = 0
    evidence[0]["frames"][0]["landmarks"][30]["visibility"] = .2
    result = prepare(api, evidence, cameras, indices=(30,))
    point = result.frame(0)["landmarks"][0]
    assert point["status"] == "insufficient_qualified_views"
    assert [row["status"] for row in point["observations"]] == ["low_confidence", "pose_unavailable_or_competing", "available"]
    assert result.frame(1)["landmarks"][0]["observations"][0]["status"] == "pose_not_sampled"
    assert result.frame(11)["partition"] == "excluded_later_source"
    assert result.frame(11)["landmarks"][0]["observations"][2]["status"] == "source_not_decoded"
    assert result.describe()["coverage"]["view_landmark_slots"] == 36


@pytest.mark.parametrize("fault", ["duplicate_source", "duplicate_name", "missing_model_sha", "clock_mismatch",
                                   "decoder_mismatch", "duplicate_index", "bad_index", "bad_alignment_count"])
def test_malformed_or_incompatible_source_envelopes_reject(api, fault):
    evidence, cameras = fixtures()
    kwargs = {"landmark_indices": (30,), "alignments": None}
    if fault == "duplicate_source":
        evidence[1]["source_sha256"] = evidence[0]["source_sha256"]
        cameras[1]["source_sha256"] = cameras[0]["source_sha256"]
        for row in evidence[1]["frames"]:
            row["id"] = evidence[0]["source_sha256"] + ":frame:" + str(row["frame"])
    elif fault == "missing_model_sha":
        evidence[0]["model"].pop("sha256")
    elif fault == "clock_mismatch":
        cameras[1]["fps"] = evidence[1]["fps"] = 20
        for row in cameras[1]["frames"]:
            row["time_s"] = row["frame"] / 20
        for row in evidence[1]["frames"]:
            row["time_s"] = row["decoder_timestamp_s"] = row["frame"] / 20
    elif fault == "decoder_mismatch":
        evidence[0]["frames"][0]["decoder_timestamp_s"] = .1
    elif fault == "duplicate_index":
        kwargs["landmark_indices"] = (30, 30)
    elif fault == "bad_index":
        kwargs["landmark_indices"] = (True,)
    elif fault == "bad_alignment_count":
        kwargs["alignments"] = [None]
    supplied = inputs(evidence, cameras)
    if fault == "duplicate_name":
        supplied[1]["name"] = supplied[0]["name"]
    with pytest.raises(ValueError):
        api.prepare_multiview_landmark_evidence(supplied, **kwargs)


def test_oversized_rows_reject_before_serialization_or_solver(api, monkeypatch):
    evidence, cameras = fixtures()
    evidence[0]["frames"] = [evidence[0]["frames"][0]] * 901
    monkeypatch.setattr(api, "_json_size", denied)
    monkeypatch.setattr(api, "triangulate_pixels", denied)
    with pytest.raises(ValueError, match="source row envelope"):
        prepare(api, evidence, cameras, alignments=None)


def test_raw_byte_bound_rejects_before_parsing(api, monkeypatch):
    evidence, cameras = fixtures()
    supplied = inputs(evidence, cameras)
    supplied[0]["evidence_bytes"] = b"x" * ((8 << 20) + 1)
    monkeypatch.setattr(api, "_decode", denied)
    with pytest.raises(ValueError, match="exact input bytes"):
        api.prepare_multiview_landmark_evidence(supplied, landmark_indices=(30,))


def test_work_bound_rejects_before_any_fit(api, monkeypatch):
    evidence, cameras = fixtures()
    monkeypatch.setattr(api, "MAX_FITS", 1)
    monkeypatch.setattr(api, "triangulate_pixels", denied)
    with pytest.raises(ValueError, match="fit count"):
        prepare(api, evidence, cameras)


def test_output_bound_never_returns_truncated_slots(api, monkeypatch):
    evidence, cameras = fixtures()
    monkeypatch.setattr(api, "MAX_OUTPUT_BYTES", 100)
    with pytest.raises(ValueError, match="cache exceeds"):
        prepare(api, evidence, cameras, indices=(30,))


def test_deadline_after_solver_cannot_publish_a_partial_result(api, monkeypatch):
    evidence, cameras = fixtures()
    original, check = api.triangulate_pixels, api._check
    finished = False

    def fitted(*args):
        nonlocal finished
        result = original(*args)
        finished = True
        return result

    def checked(deadline):
        if finished:
            raise TimeoutError("Synthetic final fit deadline")
        check(deadline)

    monkeypatch.setattr(api, "triangulate_pixels", fitted)
    monkeypatch.setattr(api, "_check", checked)
    with pytest.raises(TimeoutError, match="final fit"):
        prepare(api, evidence, cameras, indices=(30,))


def test_cached_queries_do_not_reparse_evidence_fit_or_hash(api, monkeypatch):
    evidence, cameras = fixtures()
    result = prepare(api, evidence, cameras, indices=(30,))
    first = result.frame(0)
    first["landmarks"][0]["world_point_m"][0] = 999
    meta = result.describe()
    meta["views"].clear()
    for name in ("_decode", "_view", "_observation", "triangulate_pixels", "canonical_hash", "_json_size"):
        monkeypatch.setattr(api, name, denied)
    assert result.frame(0)["landmarks"][0]["world_point_m"][0] != 999
    assert len(result.describe()["views"]) == 3
    assert len(list(result.iter_frame_bytes())) == 12
    with pytest.raises(ValueError):
        result.frame(True)


def test_large_source_retains_every_slot_with_bounded_queries(api, record_property):
    evidence, cameras = fixtures(count=1002, sampled=tuple(range(0, 1002, 10)), fps=30)
    begin = time.perf_counter()
    result = prepare(api, evidence, cameras, indices=(27, 28, 29, 30, 31, 32))
    prepare_seconds = time.perf_counter() - begin
    begin = time.perf_counter()
    for frame in range(20):
        result.frame(frame)
    query_seconds = time.perf_counter() - begin
    meta = result.describe()
    assert result.frame_count == 1002
    assert meta["coverage"]["landmark_slots"] == 6012 and meta["coverage"]["view_landmark_slots"] == 18036
    assert sum(meta["coverage"]["status_counts"].values()) == 6012
    assert sum(meta["coverage"]["observation_status_counts"].values()) == 18036
    assert meta["cached_row_bytes"] + len(result._metadata) <= 8 << 20
    assert meta["coverage"]["solver_calls"] == 366
    record_property("prepare_seconds", prepare_seconds)
    record_property("twenty_cached_queries_seconds", query_seconds)
    record_property("input_bytes", meta["input_bytes"])
    record_property("cached_bytes", meta["cached_row_bytes"] + len(result._metadata))


def rounded_camera_fixtures():
    evidence, cameras = fixtures(fps=30000 / 1001)
    angle = .37
    exact = np.array([[np.cos(angle), -np.sin(angle), 0.], [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    rounded = exact.astype(np.float32).astype(float).tolist()
    for source, camera in zip(evidence, cameras, strict=True):
        for row in camera["frames"]:
            row["R"] = deepcopy(rounded)
        for row in source["frames"]:
            if row["frame"] >= 7:
                row["landmarks"] = "Excluded landmark payload must remain unused"
                continue
            point = np.array([.2 + row["frame"] * .0001, .1, .15])
            q = exact @ point + camera["frames"][row["frame"]]["t"]
            pixel = np.asarray(camera["K"]) @ (q / q[2])
            for landmark in row["landmarks"]:
                landmark["x"], landmark["y"] = float(pixel[0] / 640), float(pixel[1] / 480)
    return evidence, cameras


def test_rotation_prior_default_stays_strict_without_silent_correction(api, monkeypatch):
    evidence, cameras = rounded_camera_fixtures()
    monkeypatch.setattr(api, "prepare_camera_rotation_prior", denied)
    with pytest.raises(ValueError, match="proper world-to-camera"):
        prepare(api, evidence, cameras, indices=(30,))


def test_rotation_prior_cached_per_used_frame_preserves_raw_cameras_and_clocks(api, monkeypatch, record_property):
    evidence, cameras = rounded_camera_fixtures()
    supplied = inputs(evidence, cameras)
    original_camera_bytes = [item["camera_bytes"] for item in supplied]
    mappings = [encode(mapping(camera, cameras[0])) for camera in cameras[1:]]
    prior_function, fit_function = api.prepare_camera_rotation_prior, api.triangulate_pixels
    calls, fits = [], []

    def prior(raw):
        calls.append(deepcopy(raw))
        return prior_function(raw)

    def fit(pixels, solver_cameras):
        for index, camera in enumerate(solver_cameras):
            source = cameras[index]
            assert np.array_equal(camera["matrix"], source["K"])
            assert np.array_equal(camera["translation"], source["frames"][0]["t"])
            assert camera["radial"] == source["radial_k1"]
            assert [camera["width"], camera["height"]] == [source["width"], source["height"]]
        fits.append(len(solver_cameras))
        return fit_function(pixels, solver_cameras)

    monkeypatch.setattr(api, "prepare_camera_rotation_prior", prior)
    monkeypatch.setattr(api, "triangulate_pixels", fit)
    begin = time.perf_counter()
    result = api.prepare_multiview_landmark_evidence(supplied, landmark_indices=(27, 28, 29, 30, 31, 32),
        alignments=mappings, rotation_prior="bounded-polar-rounding-prior-v1")
    elapsed = time.perf_counter() - begin
    assert len(calls) == 12 and len(fits) == 24 and set(fits) == {3}
    meta = result.describe()
    assert meta["numerical_rotation_prior"]["prepared_view_frames"] == 12
    assert meta["numerical_rotation_prior"]["assumption_status"] == "assumed"
    assert meta["coverage"]["landmark_slots"] == 72 and meta["coverage"]["view_landmark_slots"] == 216
    for index, raw in enumerate(original_camera_bytes):
        assert raw == encode(cameras[index]) and result.view_inputs[index][2] is raw
        assert meta["views"][index]["camera_pin"]["sha256"] == sha256(raw).hexdigest()
    for frame in range(12):
        row = result.frame(frame)
        assert row["time_s"] == frame / (30000 / 1001)
        expected = 3 if frame in (0, 2, 4, 6) else 0
        assert len(row["camera_rotation_priors"]) == expected
        for item in row["camera_rotation_priors"]:
            source_index = int(item["view_name"][-1])
            assert item["source_frame"] == frame
            assert item["prior"]["raw_rotation"] == cameras[source_index]["frames"][frame]["R"]
            assert item["camera_pin"]["sha256"] == meta["views"][source_index]["camera_pin"]["sha256"]
        for point in row["landmarks"]:
            if expected:
                assert point["status"] == "conditional_estimate"
                assert len(point["solver_camera_rotation_ids"]) == 3
                assert np.allclose(point["world_point_m"], [.2 + frame * .0001, .1, .15], atol=1e-7, rtol=0)
                assert all(obs["decoder_timestamp_s"] == frame / (30000 / 1001) for obs in point["observations"])
            else:
                assert point["solver_camera_rotation_ids"] == []
    record_property("rotation_prior_preparation_seconds", elapsed)
    record_property("cached_camera_priors", len(calls))
    record_property("all_ray_fits", len(fits))


def test_rotation_prior_does_not_touch_unbound_or_insufficient_ray_slots(api, monkeypatch):
    evidence, cameras = rounded_camera_fixtures()
    monkeypatch.setattr(api, "prepare_camera_rotation_prior", denied)
    monkeypatch.setattr(api, "triangulate_pixels", denied)
    result = api.prepare_multiview_landmark_evidence(inputs(evidence, cameras), landmark_indices=(30,),
        alignments=None, rotation_prior="bounded-polar-rounding-prior-v1")
    assert result.describe()["numerical_rotation_prior"]["prepared_view_frames"] == 0
    assert all(result.frame(frame)["camera_rotation_priors"] == [] for frame in range(12))
    for source in evidence[:2]:
        for row in source["frames"]:
            row["pose_count"] = 0
    result = api.prepare_multiview_landmark_evidence(inputs(evidence, cameras), landmark_indices=(30,),
        alignments=[encode(mapping(camera, cameras[0])) for camera in cameras[1:]],
        rotation_prior="bounded-polar-rounding-prior-v1")
    assert result.frame(0)["landmarks"][0]["status"] == "insufficient_qualified_views"
    assert result.describe()["numerical_rotation_prior"]["prepared_view_frames"] == 0


@pytest.mark.parametrize("choice", [True, False, "", "round", "bounded-polar-rounding-prior-v2"])
def test_rotation_prior_requires_exact_explicit_profile(api, monkeypatch, choice):
    evidence, cameras = fixtures()
    monkeypatch.setattr(api, "_decode", denied)
    with pytest.raises(ValueError, match="explicitly versioned"):
        api.prepare_multiview_landmark_evidence(inputs(evidence, cameras), landmark_indices=(30,), rotation_prior=choice)
