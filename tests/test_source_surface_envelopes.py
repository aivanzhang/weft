"""Analytical interval geometry and source custody; no media/model/native/source study."""

from copy import deepcopy
from fractions import Fraction
from hashlib import sha256
import importlib
import json
import time

import numpy as np
import pytest

from test_multiview_landmark_evidence import denied, local_only


@pytest.fixture
def api():
    return importlib.import_module("weft.source_surface_envelopes")


def fixture(count=4, channels=1):
    old = importlib.import_module("test_source_surface_anchors")
    data = old.fixture(count)
    data[2]["profile"] = "source-horizontal-surface-intervals-v1"
    data[2]["surfaces"][0].pop("height_m")
    data[2]["surfaces"][0]["height_interval_m"] = [.2, .6]
    for index in range(1, channels):
        channel = deepcopy(data[3]["channels"][0])
        channel["id"] = "anchor-"+str(index)
        data[3]["channels"].append(channel)
        for original in list(data[3]["annotations"][:count]):
            row = deepcopy(original)
            row["channel_id"] = channel["id"]
            row["id"] = row["id"].rsplit(":", 1)[0]+":"+channel["id"]
            data[3]["annotations"].append(row)
    return data


def encoded(data):
    old = importlib.import_module("test_source_surface_anchors")
    args = old.encoded(data)
    args["surface_interval_document"] = args.pop("surface_document")
    return args


def prepare(api, data=None, **changes):
    return api.prepare_source_surface_envelopes(**(encoded(fixture() if data is None else data) | changes))


def test_known_height_and_pixel_ranges_enclose_without_choosing_a_point(api):
    result = prepare(api)
    point = result.frame(0)["anchors"][0]
    bounds = point["world_envelope_m"]
    assert point["status"] == "conditional_surface_envelope"
    assert bounds["minimum"] == pytest.approx([.126, -.108, .2], abs=1e-15)
    assert bounds["maximum"] == pytest.approx([.198, -.056, .6], abs=1e-15)
    assert bounds["kind"] == "outer_axis_aligned_enclosure"
    assert "position_world_m" not in point and "midpoint" not in point
    meta = result.describe()
    assert meta["contact"] == meta["load"] == meta["grasp"] == "unknown"
    assert meta["body_or_floor_installed"] is meta["geometry_modified"] is False
    assert meta["physical_validation"] == "unverified"
    assert meta["counts"]["conditional_envelopes"] == meta["counts"]["retained_anchor_slots"] == 4
    assert meta["calibration_world_alignment"]["claims"]["spatial"]["status"] == "assumed"


def test_outward_rounding_encloses_exact_fraction_not_just_sampled_float(api):
    data = fixture()
    data[2]["surfaces"][0]["height_interval_m"] = [0, 0]
    for row in data[3]["annotations"]:
        row["uncertainty_px"] = 0
    bounds = prepare(api, data).frame(0)["anchors"][0]["world_envelope_m"]
    assert Fraction(bounds["minimum"][0]) <= Fraction(1, 5) <= Fraction(bounds["maximum"][0])
    assert bounds["minimum"][0] < bounds["maximum"][0]
    assert Fraction(bounds["minimum"][1]) <= Fraction(-1, 10) <= Fraction(bounds["maximum"][1])


def test_zero_width_height_matches_existing_point_method_without_its_midpoint(api):
    old = importlib.import_module("test_source_surface_anchors")
    original = old.fixture()
    expected = old.anchors.reconstruct_source_surface_anchors(**old.encoded(original))["rows"][0]["anchors"][0]
    data = fixture()
    data[2]["surfaces"][0]["height_interval_m"] = [0, 0]
    actual = prepare(api, data).frame(0)["anchors"][0]["world_envelope_m"]
    for edge in ("minimum", "maximum"):
        assert actual[edge] == pytest.approx(expected["sampled_pixel_box_world_range_m"][edge], abs=1e-14)


def test_wider_authored_uncertainty_cannot_shrink_the_geometry_enclosure(api):
    narrow = prepare(api).frame(0)["anchors"][0]["world_envelope_m"]
    data = fixture()
    data[2]["surfaces"][0]["height_interval_m"] = [0, .8]
    for row in data[3]["annotations"]:
        row["uncertainty_px"] = 3
    wide = prepare(api, data).frame(0)["anchors"][0]["world_envelope_m"]
    assert all(a <= b for a, b in zip(wide["minimum"], narrow["minimum"], strict=True))
    assert all(a >= b for a, b in zip(wide["maximum"], narrow["maximum"], strict=True))


def test_oblique_translated_camera_bounds_all_interior_pixel_height_combinations(api):
    data = fixture()
    rotation = np.array([[1., 0., 0.], [0., -.8, -.6], [0., .6, -.8]])
    translation = np.array([-1., 3., 4.])
    for row in data[1]["frames"]:
        row.update(R=rotation.tolist(), t=translation.tolist())
    data[2]["surfaces"][0].update(kind="static_support", bounds_xy_m=[-10., -10., 10., 10.])
    point = prepare(api, data).frame(0)["anchors"][0]
    assert point["status"] == "conditional_surface_envelope"
    box = point["world_envelope_m"]
    origin = np.linalg.solve(rotation, -translation)
    for x in np.linspace(59, 61, 7):
        for y in np.linspace(44, 46, 5):
            direction = np.linalg.solve(rotation, np.linalg.solve(np.asarray(data[1]["K"]), [x, y, 1.]))
            for height in np.linspace(.2, .6, 5):
                value = origin+(height-origin[2])/direction[2]*direction
                assert np.all(value >= np.asarray(box["minimum"])-1e-14)
                assert np.all(value <= np.asarray(box["maximum"])+1e-14)


def test_raw_near_rotation_is_inverted_without_hidden_normalization(api, monkeypatch):
    data = fixture()
    scale = 1.000001
    for row in data[1]["frames"]:
        row["R"] = (np.diag([1., -1., -1.]) * scale).tolist()
    data[2]["surfaces"][0]["height_interval_m"] = [0, 0]
    for row in data[3]["annotations"]:
        row["uncertainty_px"] = 0
    args = encoded(data)
    raw = args["camera_document"]
    monkeypatch.setattr(np.linalg, "svd", denied)
    monkeypatch.setattr(np.linalg, "eigh", denied)
    result = api.prepare_source_surface_envelopes(**args)
    box = result.frame(0)["anchors"][0]["world_envelope_m"]
    exact = Fraction(1, 5) / Fraction(scale)
    assert Fraction(box["minimum"][0]) <= exact <= Fraction(box["maximum"][0])
    assert result.additional_inputs[0] is raw
    assert result.describe()["bindings"]["camera_sha256"] == sha256(raw).hexdigest()


@pytest.mark.parametrize("case,reason", [("horizon", "ray_interval_crosses_surface_horizon"),
    ("behind", "height_interval_behind_camera_or_beyond_depth_bound"),
    ("radial", "radial_distortion_has_no_certified_interval_bound"),
    ("support", "envelope_exceeds_declared_surface_domain"), ("image", "pixel_interval_outside_source_image"),
    ("unknown", "unknown_surface_height_interval"), ("measured_camera", "missing_supplied_camera_measurement_receipt")])
def test_uncertified_geometry_abstains_without_clipping_or_fabricated_height(api, case, reason):
    data = fixture()
    if case == "horizon":
        for row in data[1]["frames"]:
            row.update(R=[[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]], t=[0., 2., 0.])
        for row in data[3]["annotations"]:
            row["pixel"] = [50., 40.]
    elif case == "behind":
        data[2]["surfaces"][0]["height_interval_m"] = [2., 3.]
    elif case == "radial":
        data[1]["radial_k1"] = -.1
    elif case == "support":
        data[2]["surfaces"][0].update(kind="static_support", bounds_xy_m=[0., 0., .1, .1])
    elif case == "image":
        data[3]["annotations"][0]["pixel"] = [0., 40.]
    elif case == "unknown":
        data[2]["surfaces"][0].update(height_interval_m=None, quality="unknown")
    else:
        data[1]["quality"] = "measured"
    result = prepare(api, data)
    point = result.frame(0)["anchors"][0]
    assert point["status"] == "unavailable" and point["reason"] == reason and point["world_envelope_m"] is None
    assert result.frame_count == 4 and result.describe()["counts"]["retained_anchor_slots"] == 4


@pytest.mark.parametrize("missing", ["camera_document", "surface_interval_document", "annotation_document"])
def test_missing_inputs_complete_as_unavailable_without_dropping_frames(api, missing):
    result = prepare(api, **{missing: None})
    meta = result.describe()
    assert result.frame_count == 4 and meta["status"] == "unavailable" and len(meta["missing_inputs"]) == 1
    assert meta["counts"]["retained_anchor_slots"] == (0 if missing == "annotation_document" else 4)


def test_excluded_occluded_unsampled_pixels_are_not_geometry_inputs(api, monkeypatch):
    data = fixture()
    data[3]["annotations"][1]["status"] = "occluded"
    data[0]["observations"][2]["partition"] = "audit"
    data[0]["observations"][3].update(partition="unsampled", visibility="not_sampled")
    for row in data[3]["annotations"][1:]:
        row.update(pixel="excluded payload", uncertainty_px="excluded payload")
    original = api._enclose
    seen = []

    def enclosed(annotation, *args):
        seen.append(annotation["frame"])
        return original(annotation, *args)

    monkeypatch.setattr(api, "_enclose", enclosed)
    result = prepare(api, data)
    assert seen == [0]
    assert [result.frame(i)["anchors"][0]["reason"] for i in range(4)] == [
        None, "annotation_occluded", "excluded_evidence_partition", "source_not_sampled"]


def test_source_native_clock_is_retained_not_retimed_to_camera(api):
    data = fixture()
    doc = data[0]
    doc["schema_version"] = 3
    doc["source"].pop("fps")
    doc["source"]["frame_times_s"] = [0., .034, .068, .102]
    doc["time"] = {"unit": "seconds", "origin": "submitted_video_start", "basis": "decoder_presentation_timestamps", "status": "observed"}
    for rows in (doc["evidence"], doc["observations"], data[3]["annotations"]):
        for row in rows:
            row["time_s"] = doc["source"]["frame_times_s"][row["frame"]]
    result = prepare(api, data)
    assert result.frame(0)["anchors"][0]["status"] == "conditional_surface_envelope"
    for index in range(1, 4):
        point = result.frame(index)["anchors"][0]
        assert point["reason"] == "camera_and_source_clocks_differ"
        assert point["source_time_s"] == data[0]["source"]["frame_times_s"][index]


@pytest.mark.parametrize("fault", ["source", "calibration_pin", "annotation_clock", "measured_interval", "reversed_interval", "moving_support", "blank_provenance"])
def test_foreign_or_unsupported_input_claims_reject(api, fault):
    data = fixture()
    if fault == "annotation_clock":
        data[3]["annotations"][0]["time_s"] = .1
    elif fault == "measured_interval":
        data[2]["surfaces"][0]["quality"] = "measured"
    elif fault == "reversed_interval":
        data[2]["surfaces"][0]["height_interval_m"] = [.6, .2]
    elif fault == "moving_support":
        data[2]["surfaces"][0]["kind"] = "handheld"
    elif fault == "blank_provenance":
        data[2]["surfaces"][0]["provenance"] = ""
    args = encoded(data)
    if fault == "source":
        args["source_sha256"] = "b"*64
    elif fault == "calibration_pin":
        surfaces = json.loads(args["surface_interval_document"])
        surfaces["calibration_sha256"] = "b"*64
        args["surface_interval_document"] = json.dumps(surfaces).encode()
    with pytest.raises(ValueError):
        api.prepare_source_surface_envelopes(**args)


def test_complete_metadata_only_and_missing_annotation_slots_are_retained(api):
    data = fixture(count=1100)
    data[0]["source"]["metadata_frames"] = 1101
    data[3]["annotations"] = data[3]["annotations"][:900]
    for row in data[3]["annotations"]:
        row["status"] = "unknown"
    result = prepare(api, data, camera_document=None)
    assert result.frame_count == 1101
    assert result.describe()["counts"]["retained_anchor_slots"] == 1101
    assert result.frame(1000)["anchors"][0]["reason"] == "missing_annotation"
    assert result.frame(1100)["anchors"][0]["reason"] == "metadata_only_not_decoded"


def test_extra_byte_cap_is_before_registry_loading(api, monkeypatch):
    monkeypatch.setattr(api, "_sources", denied)
    with pytest.raises(ValueError, match="4 MiB"):
        api.prepare_source_surface_envelopes({}, source_sha256="a"*64, camera_document=b" "*((4 << 20)+1))


def test_complete_frame_cap_precedes_camera_geometry(api, monkeypatch):
    data = fixture()
    data[0]["source"]["metadata_frames"] = 1801
    monkeypatch.setattr(api, "validate_camera", denied)
    with pytest.raises(ValueError, match="source metadata_frames|1800"):
        prepare(api, data)


def test_cache_overflow_or_expired_deadline_cannot_publish_a_partial_result(api, monkeypatch):
    monkeypatch.setattr(api, "MAX_CACHE_BYTES", 100)
    with pytest.raises(ValueError, match="no slots were dropped"):
        prepare(api)
    monkeypatch.setattr(api, "MAX_CACHE_BYTES", 8 << 20)
    original = api._enclose
    finished = False

    def envelope(*args):
        nonlocal finished
        result = original(*args)
        finished = True
        return result

    check = api._check

    def deadline(value):
        if finished:
            raise TimeoutError("Analytical terminal deadline")
        check(value)

    monkeypatch.setattr(api, "_enclose", envelope)
    monkeypatch.setattr(api, "_check", deadline)
    with pytest.raises(TimeoutError, match="terminal deadline"):
        prepare(api)


def test_cached_queries_preserve_raw_inputs_and_never_recompute(api, monkeypatch):
    args = encoded(fixture(channels=3))
    result = api.prepare_source_surface_envelopes(**args)
    assert result.describe()["counts"]["prepared_camera_frames"] == 4
    assert result.describe()["counts"]["conditional_envelopes"] == 12
    assert result.observation_inputs[0][1] is next(iter(args["observation_documents"].values()))
    assert result.additional_inputs[0] is args["camera_document"]
    first = result.frame(0)
    first["anchors"][0]["world_envelope_m"]["minimum"][0] = 999
    for name in ("_sources", "_document", "_inverse", "_camera_frame", "_enclose", "_intervals", "validate_camera"):
        monkeypatch.setattr(api, name, denied)
    assert result.frame(0)["anchors"][0]["world_envelope_m"]["minimum"][0] != 999
    assert len(list(result.iter_frame_bytes())) == 4
    with pytest.raises(ValueError):
        result.frame(True)


def test_representative_exact_interval_work_and_cached_queries_are_bounded(api, record_property):
    args = encoded(fixture(count=600, channels=2))
    begin = time.perf_counter()
    result = api.prepare_source_surface_envelopes(**args)
    seconds = time.perf_counter()-begin
    begin = time.perf_counter()
    for index in range(20):
        result.frame(index*29)
    queries = time.perf_counter()-begin
    meta = result.describe()
    assert meta["counts"]["conditional_envelopes"] == meta["counts"]["retained_anchor_slots"] == 1200
    assert meta["counts"]["prepared_camera_frames"] == 600
    assert result.frame_count == 600
    assert meta["cached_row_bytes"]+len(result._metadata) <= 8 << 20
    record_property("prepare_seconds_1200_envelopes", seconds)
    record_property("twenty_cached_queries_seconds", queries)
    record_property("cached_bytes", meta["cached_row_bytes"]+len(result._metadata))
    record_property("exact_input_bytes", sum(map(len, args["observation_documents"].values())) + sum(len(args[name]) for name in ("camera_document", "surface_interval_document", "annotation_document")))
