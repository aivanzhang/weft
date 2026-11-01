"""Source collection compiles into existing contracts; no media or native work."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
import time

import pytest

from test_multiview_landmark_evidence import denied, local_only


@pytest.fixture
def api():
    return importlib.import_module("weft.exercise_surface_inputs")


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def fixture(api, count=4, *, changes=None, with_body=True):
    old = importlib.import_module("test_source_surface_envelopes")
    data = old.fixture(count)
    source = data[0]["source"]["sha256"]
    for record, observed in zip(data[0]["evidence"], data[0]["observations"], strict=True):
        record["id"] = f"{source}:frame:{record['frame']}"
        observed["evidence_ids"] = [record["id"]]
    if changes is not None:
        changes(data)
    inputs = old.encoded(data)
    annotation = json.loads(inputs["annotation_document"])
    point = importlib.import_module("weft.source_surface_interval_construction")
    body = {"schema_version": 3, "profile": "exercise-v3", "source_sha256": source, "dt": 1/30,
        "reference": [{"time_s": i/30, "joints": {name: 0. for name in point.NAMES}} for i in (0, 2)],
        "reference_evidence": [{"evidence_id": f"{source}:frame:{i}", "time_s": i/30} for i in (0, 2)],
        "root_reference": [{"time_s": i/30, "translation": [0., 0., 0.]} for i in (0, 2)]}
    kwargs = {"observation_documents": inputs["observation_documents"], "source_sha256": source,
        "channels": annotation["channels"], "camera_document": inputs["camera_document"],
        "body_document": raw(body) if with_body else None,
        "human_entity_ref": annotation["channels"][0]["entity_ref"] if with_body else None}
    prepared = api.prepare_exercise_surface_inputs(**kwargs)
    request = json.loads(prepared.request_document())
    request.update(weight=2., surface_intervals=json.loads(inputs["surface_interval_document"]), annotations=annotation)
    if with_body:
        claim = {"status": "assumed", "method": "Explicit analytical fixture coordinates",
                 "uncertainty": "Authored alignment, not measured calibration or body anatomy", "evidence_refs": []}
        binding = request["body_binding"]
        binding["body_to_world"] = {"schema_version": 1, "profile": "world-alignment-v1",
            "from_frame": prepared.describe()["body_coordinate_frame"], "to_frame": request["surface_intervals"]["world_frame"],
            "spatial": {"scale": 1., "rotation_wxyz": [1., 0., 0., 0.], "translation": [0., 0., 0.]},
            "clock": {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.},
            "claims": {"spatial": deepcopy(claim), "clock": deepcopy(claim)}}
        binding["channels"][0].update(body_point="left_ankle", status="assumed",
            method="Explicit analytical ankle correspondence", uncertainty="Authored endpoint, not recovered contact")
    return prepared, request, kwargs


def test_draft_keeps_every_unknown_and_never_runs_geometry(api, monkeypatch):
    prepared, _, _ = fixture(api)
    draft = json.loads(prepared.request_document())
    assert draft["weight"] is None and draft["surface_intervals"]["world_frame"] is None
    assert draft["surface_intervals"]["surfaces"][0]["kind"] is None
    assert draft["body_binding"]["body_to_world"] is None
    assert draft["body_binding"]["channels"][0]["body_point"] is None
    assert draft["annotations"]["annotations"] == []
    frame = prepared.frame(0); annotation = frame["channels"][0]["annotation_template"]
    assert annotation["status"] == "unknown" and annotation["pixel"] is annotation["uncertainty_px"] is None
    monkeypatch.setattr(api, "prepare_source_surface_envelopes", denied)
    monkeypatch.setattr(api, "prepare_surface_interval_objective", denied)
    result = prepared.qualify(prepared.request_document())
    meta = result.describe()
    assert meta["status"] == "incomplete_request" and meta["config_member"] is None
    assert any("height_interval_m" in action for action in meta["required_explicit_inputs"])
    assert all(result.frame(i)["envelope"] is None for i in range(4))
    assert "surface-config.json" not in result.files()


def test_collection_compiles_existing_surface_config_and_real_source_joins(api):
    prepared, request, inputs = fixture(api)
    request_raw = raw(request); result = prepared.qualify(request_raw); meta = result.describe()
    assert meta["status"] == "conditional_inputs_available"
    assert meta["conditional_envelopes"] == 4 and meta["conditional_body_channel_terms"] == 2
    assert meta["source_counts"]["channel_slots"] == 4
    assert meta["contact"] == meta["load"] == meta["grasp"] == "unknown"
    assert meta["body_geometry_evaluations"] == 0 and not meta["fit_or_simulator_admission"]
    frame = result.frame(0)
    assert frame["body_join"]["channels"][0]["status"] == "conditional_interval_ready"
    assert frame["image_admission"]["status"] == "construction_image_source_ready"
    assert result.frame(1)["body_join"]["channels"][0]["reason"] == "original_body_reference_unavailable"
    files = result.files(); config = api.parse_surface_fit_config(files["surface-config.json"])
    snapshot = {"inputs/" + name: value for name, value in files.items() if name.startswith("surface-")}
    assert api.validate_surface_fit_snapshot(snapshot, api._descriptor(config, files["surface-config.json"])) == config
    assert files["collection-request.json"] == request_raw and files["original-human.json"] == inputs["body_document"]
    assert files["surface-camera.json"] == inputs["camera_document"]
    manifest = json.loads(files["collection-manifest.json"])
    assert all(sha256(files[name]).hexdigest() == pin for name, pin in manifest["files"].items())
    binding = json.loads(files["surface-body-binding.json"])
    assert binding["envelope_bindings"] == meta["envelope_metadata"]["bindings"]
    assert request["body_binding"]["envelope_bindings"] is None


def test_excluded_unsampled_occluded_and_metadata_tail_are_never_templates(api):
    def change(data):
        observations, camera, _, annotations = data
        observations["observations"][1]["partition"] = "audit"
        observations["observations"][2].update(partition="unsampled", visibility="not_sampled", evidence_ids=[])
        observations["observations"][3]["visibility"] = "not_detected"
        observations["source"]["metadata_frames"] += 1
        camera["frame_count"] += 1
        camera["frames"].append(dict(camera["frames"][-1], frame=4, time_s=4/30))
        annotations["annotations"][1]["pixel"] = "excluded_payload_not_geometry"
    prepared, request, _ = fixture(api, changes=change)
    expected = [None, "excluded_evidence_partition", "source_not_sampled", "source_entity_unavailable", "metadata_only_not_decoded"]
    assert [prepared.frame(i)["channels"][0]["reason"] for i in range(5)] == expected
    assert all(prepared.frame(i)["channels"][0]["annotation_template"] is None for i in range(1, 5))
    result = prepared.qualify(raw(request))
    assert result.describe()["source_counts"]["retained_source_frames"] == 5
    assert [result.frame(i)["envelope"]["anchors"][0]["reason"] for i in range(5)] == expected
    assert result.frame(2)["body_join"]["body_reference"]["reference_index"] == 1
    assert result.frame(2)["image_admission"]["status"] == "unavailable"
    assert len(result.files()["collection-frames.jsonl"].splitlines()) == 5


@pytest.mark.parametrize("field,reason", [
    ("height", "unknown_surface_height_interval"),
    ("mapping", "missing_or_unknown_body_world_alignment"),
    ("correspondence", "missing_or_unknown_endpoint_correspondence"),
    ("annotations", "missing_annotation"),
])
def test_unknowns_remain_valid_unavailable_not_zero_targets(api, field, reason):
    prepared, request, _ = fixture(api)
    if field == "height": request["surface_intervals"]["surfaces"][0].update(height_interval_m=None, quality="unknown")
    if field == "mapping": request["body_binding"]["body_to_world"] = None
    if field == "correspondence": request["body_binding"]["channels"][0].update(body_point=None, status="unknown")
    if field == "annotations": request["annotations"]["annotations"] = []
    result = prepared.qualify(raw(request)); meta = result.describe()
    assert meta["status"] == "qualified_but_unavailable" and meta["config_schema_qualified"]
    assert meta["conditional_body_channel_terms"] == 0 and meta["required_explicit_inputs"]
    assert result.frame(0)["body_join"]["channels"][0]["reason"] == reason
    assert result.frame(0)["image_admission"]["status"] == "construction_image_source_ready"


def test_static_support_requires_extents_and_explicit_classification(api):
    prepared, request, _ = fixture(api)
    surface = request["surface_intervals"]["surfaces"][0]; surface["kind"] = "static_support"
    incomplete = prepared.qualify(raw(request))
    assert incomplete.describe()["status"] == "incomplete_request"
    assert any("bounds_xy_m" in value for value in incomplete.describe()["missing_fields"])
    assert "surface-config.json" not in incomplete.files()
    surface["bounds_xy_m"] = [0., -.5, .5, .5]
    surface["provenance"] = "Explicit authored stationary equipment-top hypothesis; contact remains unknown"
    complete = prepared.qualify(raw(request))
    assert complete.describe()["conditional_body_channel_terms"] == 2
    assert complete.describe()["contact"] == "unknown"


@pytest.mark.parametrize("kind", ["moving_equipment", "handheld", "unknown"])
def test_no_automatic_static_equipment_upgrade(api, kind):
    prepared, request, _ = fixture(api)
    request["surface_intervals"]["surfaces"][0]["kind"] = kind
    with pytest.raises(ValueError, match="moving/handheld"):
        prepared.qualify(raw(request))


@pytest.mark.parametrize("change,match", [
    (lambda r:r["input_bindings"].update(source_sha256="b"*64), "original inputs"),
    (lambda r:r["surface_intervals"].update(calibration_sha256="a"*64), "calibration bytes"),
    (lambda r:r["annotations"]["channels"][0]["entity_ref"].update(object_id="other"), "channel/entity"),
    (lambda r:r["annotations"]["annotations"][0].update(time_s=.1), "identity/clock"),
    (lambda r:r["annotations"]["annotations"][0].update(observation_id="other"), "identity/clock"),
    (lambda r:r["body_binding"].update(envelope_bindings={}), "different envelope"),
    (lambda r:r.update(weight=0), "weight"),
])
def test_wrong_identity_clock_or_explicit_pins_fail_closed(api, change, match):
    prepared, request, _ = fixture(api); change(request)
    with pytest.raises(ValueError, match=match):
        prepared.qualify(raw(request))


def test_stale_original_body_pin_is_unavailable_not_repaired(api):
    prepared, request, _ = fixture(api)
    request["body_binding"]["body_sha256"] = "a"*64
    result = prepared.qualify(raw(request))
    assert result.frame(0)["body_join"]["channels"][0]["reason"] == "binding_body_or_source_identity_differs"
    assert result.describe()["conditional_body_channel_terms"] == 0
    assert json.loads(result.files()["surface-body-binding.json"])["body_sha256"] == "a"*64


def test_no_body_can_qualify_source_envelopes_without_body_admission(api):
    prepared, request, _ = fixture(api, with_body=False)
    result = prepared.qualify(raw(request))
    assert result.describe()["conditional_envelopes"] == 4
    assert result.describe()["conditional_body_channel_terms"] == 0
    assert result.frame(0)["body_join"] is result.frame(0)["image_admission"] is None
    assert json.loads(result.files()["surface-config.json"])["body_binding"] is None


def test_missing_camera_is_actionable_and_never_replaced_by_identity(api):
    _, _, inputs = fixture(api)
    inputs["camera_document"] = None
    prepared = api.prepare_exercise_surface_inputs(**inputs)
    result = prepared.qualify(prepared.request_document())
    assert any("camera_document" in value for value in result.describe()["missing_fields"])
    assert "surface-camera.json" not in result.files()
    assert result.frame(0)["envelope"] is None


def test_cached_requests_and_detached_queries_do_not_reprepare(api, monkeypatch):
    prepared, request, _ = fixture(api); request_raw = raw(request)
    result = prepared.qualify(request_raw)
    for name in ("_sources", "_document", "prepare_source_surface_envelopes", "prepare_surface_interval_objective"):
        monkeypatch.setattr(api, name, denied)
    assert prepared.qualify(request_raw) is result
    detached = result.frame(0); detached["source"]["channels"].clear()
    assert len(result.frame(0)["source"]["channels"]) == 1
    copied = result.files(); copied.clear()
    assert "surface-config.json" in result.files()


def test_existing_input_and_request_byte_caps_precede_processing(api, monkeypatch):
    prepared, _, inputs = fixture(api)
    monkeypatch.setattr(api, "_sources", denied)
    bad = dict(inputs, observation_documents={"x": b" "*((8<<20)+1)})
    with pytest.raises(ValueError, match="bounded exact"):
        api.prepare_exercise_surface_inputs(**bad)
    with pytest.raises(ValueError, match="request exceeds"):
        prepared.qualify(b" "*(api.MAX_REQUEST_BYTES+1))


def test_timeout_does_not_return_a_truncated_qualification(api, monkeypatch):
    prepared, request, _ = fixture(api)
    monkeypatch.setattr(api, "MAX_SECONDS", 0.)
    with pytest.raises(TimeoutError, match="cooperative deadline"):
        prepared.qualify(raw(request))
    assert prepared._last is None


@pytest.mark.parametrize("change", [
    lambda r:r["surface_intervals"].update(surfaces=[None]),
    lambda r:r["body_binding"].update(channels=[{}]*13),
    lambda r:r["body_binding"].update(channels=[None]),
])
def test_malformed_small_schema_fails_before_geometry(api, monkeypatch, change):
    prepared, request, _ = fixture(api); change(request)
    monkeypatch.setattr(api, "prepare_source_surface_envelopes", denied)
    with pytest.raises(ValueError, match="surface identity|body-channel"):
        prepared.qualify(raw(request))


def test_original_decoder_clock_is_retained_and_not_retimed_for_camera(api):
    def change(data):
        observation, _, _, annotations = data
        times = [.01 + i/30 for i in range(4)]
        observation["schema_version"] = 3
        observation["source"].pop("fps")
        observation["source"]["frame_times_s"] = times
        observation["time"] = {"unit":"seconds","origin":"submitted_video_start",
            "basis":"decoder_presentation_timestamps","status":"observed"}
        for rows in (observation["observations"], observation["evidence"], annotations["annotations"]):
            for row, stamp in zip(rows,times,strict=True): row["time_s"] = stamp
    prepared, request, _ = fixture(api, changes=change)
    result = prepared.qualify(raw(request))
    clock = next(iter(result.describe()["source_clocks"].values()))
    assert clock["basis"] == "decoder_presentation_timestamps"
    assert result.frame(0)["source"]["channels"][0]["source_time_s"] == .01
    assert result.frame(0)["envelope"]["anchors"][0]["reason"] == "camera_and_source_clocks_differ"
    assert result.frame(0)["image_admission"]["reason"] == "body_evidence_source_clock_differs"
    assert result.describe()["conditional_body_channel_terms"] == 0


def test_positive_radial_coefficients_pass_through_without_recalibration(api):
    def change(data): data[1]["radial_k1"] = .00047528394168603385
    prepared, request, inputs = fixture(api, changes=change)
    result = prepared.qualify(raw(request))
    assert result.files()["surface-camera.json"] == inputs["camera_document"]
    assert result.describe()["envelope_metadata"]["radial_inverse"]["radial_k1"] == .00047528394168603385
    assert result.describe()["conditional_body_channel_terms"] == 2


def test_representative_complete_collection_and_cached_query_cost(api, record_property):
    started = time.monotonic()
    prepared, request, _ = fixture(api, count=1200)
    prepared_seconds = time.monotonic() - started
    request["annotations"]["annotations"] = [request["annotations"]["annotations"][i] for i in (0, 2)]
    started = time.monotonic(); result = prepared.qualify(raw(request)); qualified_seconds = time.monotonic()-started
    started = time.monotonic()
    for i in range(1000):
        assert result.frame(i)["source"]["source_frame"] == i
    query_seconds = time.monotonic()-started
    meta = result.describe(); files = result.files()
    assert meta["source_counts"]["channel_slots"] == 1200
    assert meta["conditional_body_channel_terms"] == 2
    assert len(files["collection-frames.jsonl"].splitlines()) == 1200
    assert meta["cached_row_bytes"] <= api.MAX_CACHE_BYTES and len(files) <= 20
    measurement = {"frames":1200,"channels":1,"body_references":2,"authored_annotations":2,
        "prepare_seconds":prepared_seconds,"qualify_seconds":qualified_seconds,"cached_1000_queries_seconds":query_seconds,
        "worksheet_bytes":meta["source_worksheet_cache_bytes"],"qualification_row_bytes":meta["cached_row_bytes"],
        "output_files":len(files),"output_bytes":sum(map(len,files.values())),"native_calls":0,"body_geometry_evaluations":0}
    record_property("bounded_collection_measurement",json.dumps(measurement,sort_keys=True))
    print(json.dumps(measurement,sort_keys=True))
