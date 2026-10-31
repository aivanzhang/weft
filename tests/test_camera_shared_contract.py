"""Analytical admission only; fixture bodies are not physics-valid human models."""

import builtins
from copy import deepcopy
import importlib
import json
import os
import socket
import subprocess

import pytest


def _deny(*_args, **_kwargs):
    raise AssertionError("Native physics, reconstruction, media, network and subprocesses are forbidden")


@pytest.fixture(scope="module", autouse=True)
def local_only():
    with pytest.MonkeyPatch.context() as patch:
        original = builtins.__import__

        def guarded(name, *args, **kwargs):
            if (name.split(".")[0] in {"mujoco", "cv2", "PIL", "mediapipe", "torch", "tensorflow", "requests", "httpx"}
                    or name in {"weft.human", "weft.mobile_human", "weft.foot_body", "weft.shared_world"}):
                _deny()
            return original(name, *args, **kwargs)

        patch.setattr(builtins, "__import__", guarded)
        for owner, names in ((socket, ("socket", "create_connection")),
                             (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
                             (os, ("system", "popen", "fork", "posix_spawn", "posix_spawnp"))):
            for name in names:
                if hasattr(owner, name):
                    patch.setattr(owner, name, _deny)
        yield


@pytest.fixture(scope="module")
def api(local_only):
    module = importlib.import_module("weft.camera_shared_contract")
    with pytest.MonkeyPatch.context() as patch:
        for path, names in (("weft.multiview_pose", ("triangulate_pair", "_triangulate_rays")),
                            ("weft.camera_body", ("make_camera_body", "build_camera_reference")),
                            ("weft.multiview_root", ("make_two_view_root", "triangulate_pair")),
                            ("weft.camera_ground", ("apply_camera_ground", "audit_ground_clearance"))):
            for name in names:
                patch.setattr(importlib.import_module(path), name, _deny)
        yield module


def _fixture(*, sampled=(0, 2, 4, 6, 8, 10), fps=30):
    producer = importlib.import_module("weft.multiview_foot_reference")
    tools = importlib.import_module("test_multiview_foot_reference")
    value = tools._fixture(producer, fps=fps, sampled=sampled)
    value["body"]["schema_version"] = 3
    tools._pin(value, producer)
    return value


def _pin(value):
    tools = importlib.import_module("test_multiview_foot_reference")
    return tools._pin(value, importlib.import_module("weft.multiview_foot_reference"))


def _scene_arguments(api, body):
    root = body["camera_reference"]["root_refinement"]
    return {"dt": 1.0 / root["fps"], "coordinate_frame": {
        "axes": "x_right_y_depth_z_up", "calibration_status": api.CALIBRATION_STATUS,
        "units": {"length": api.LENGTH_UNIT, "mass": api.MASS_UNIT, "time": "second",
                  "orientation": "quaternion_wxyz", "angular_velocity": "world_radians_per_second"},
        "scale_bridge": {"body_height_to_world_scale": 1.0, "origin_world": [0.0, 0.0, 0.0],
                         "method": "Preserve supplied camera basis", "uncertainty_note": "No measured anatomy or mass"},
        "clock_binding": {"control_dt_s": 1.0 / root["fps"], "physics_substeps": 8, "source_fps": root["fps"],
                          "source_time_offset_s": body["reference"][0]["time_s"],
                          "alignment_method": "Retain every original source timestamp"}},
        "source_binding": {"mode": "source_qualified", "source_sha256": body["source_sha256"],
                           "total_source_frames": root["source_frames"]}}


def _raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def test_exact_pins_detached_envelope_and_complete_original_clock(api):
    body = _fixture(sampled=(1, 3, 5, 7, 9, 11))["body"]
    arguments = _scene_arguments(api, body)
    before = _raw([body, arguments])
    coupling = api.build_camera_coupling(body)
    result = api.validate_camera_coupling(body, coupling, **arguments)
    assert result == coupling and result is not coupling
    assert coupling["profile"] == "camera-shared-coupling-v1"
    assert coupling["body_sha256"] == api.canonical_hash(body)
    assert coupling["camera_binding_sha256"] == api.canonical_hash(body["camera_reference"])
    assert coupling["ground_sha256"] == api.canonical_hash(body["camera_reference"]["ground"])
    assert coupling["root_reference_sha256"] == api.canonical_hash(body["root_reference"])
    assert coupling["shape_sha256"] == api.shape_hash(body)
    source_clock = {"schema_version": 1, "profile": "camera-shared-source-clock-v1",
                    "source_sha256": body["source_sha256"], "source_fps": 30, "total_source_frames": 12,
                    "control_dt_s": 1 / 30, "rows": [
                        {"frame": frame, "time_s": frame / 30, "evidence_id": f"{body['source_sha256']}:frame:{frame}"}
                        for frame in (1, 3, 5, 7, 9, 11)]}
    assert coupling["source_clock_sha256"] == api.canonical_hash(source_clock)
    assert coupling["clock"] == {"control_dt_s": 1 / 30, "source_fps": 30, "source_time_offset_s": 1 / 30,
                                  "first_source_frame": 1, "last_source_frame": 11, "total_source_frames": 12,
                                  "retained_poses": 6, "horizon_steps": 10}
    assert coupling["runtime_bounds"] == {"root_abs_max": 50.1, "segment_contact_abs_max": 64.0}
    assert coupling["assumptions"]["support"] == "unscored"
    assert coupling["calibration_status"] == "estimated_prior"
    assert _raw([body, arguments]) == before
    result["world_to_local"]["rotation"][0][0] = 99
    assert coupling["world_to_local"]["rotation"][0][0] == 1.0


def test_world_translation_preserves_local_points_clocks_and_shape(api):
    fixture = _fixture()
    original = api.build_camera_coupling(fixture["body"])
    shift = [17.0, -8.0, 4.0]
    body = fixture["body"]
    binding = body["camera_reference"]
    binding["origin_world_m"] = [v + delta for v, delta in zip(binding["origin_world_m"], shift, strict=True)]
    for row in binding["root_refinement"]["roots"]:
        row["root_world_m"] = [v + delta for v, delta in zip(row["root_world_m"], shift, strict=True)]
        row["hip_world_m"] = [[v + delta for v, delta in zip(point, shift, strict=True)] for point in row["hip_world_m"]]
    fixture["ground"]["height_m"] += shift[2]
    _pin(fixture)
    changed = api.build_camera_coupling(body)
    assert changed["body_sha256"] != original["body_sha256"]
    assert changed["shape_sha256"] == original["shape_sha256"]
    assert changed["root_reference_sha256"] == original["root_reference_sha256"]
    assert changed["source_clock_sha256"] == original["source_clock_sha256"]
    point = [1.3, 2.7, .2]
    before = [v + offset for v, offset in zip(point, original["world_to_local"]["translation"], strict=True)]
    after = [v + delta + offset for v, delta, offset in zip(point, shift, changed["world_to_local"]["translation"], strict=True)]
    assert before == pytest.approx(after)
    assert changed["world_to_local"]["rotation"] == [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    assert changed["shared_placement"] == {"scale": 1., "origin": [0., 0., 0.]}


def test_nontrivial_body_rotation_and_root_motion_are_not_normalized(api):
    fixture = _fixture()
    body = fixture["body"]
    body["body_frame"]["rotation"] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    root = body["camera_reference"]["root_refinement"]["roots"][-1]
    root["root_world_m"] = [13., 1., 1.5]
    root["hip_world_m"] = [[12.5, 1., 1.5], [13.5, 1., 1.5]]
    body["root_reference"][-1]["translation"] = [12., -1., .5]
    _pin(fixture)
    before = _raw(body)
    coupling = api.build_camera_coupling(body)
    assert api.validate_camera_coupling(body, coupling, **_scene_arguments(api, body)) == coupling
    assert _raw(body) == before
    assert coupling["world_to_local"]["translation"] == [-1., -2., 0.]


def test_fractional_source_clock_does_not_rewrite_authored_body_dt(api):
    body = _fixture(fps=30000 / 1001)["body"]
    original = _raw(body)
    coupling = api.build_camera_coupling(body)
    arguments = _scene_arguments(api, body)
    assert body["dt"] == 1 / 30
    assert coupling["clock"]["control_dt_s"] == 1 / (30000 / 1001)
    assert coupling["clock"]["control_dt_s"] != body["dt"]
    assert api.validate_camera_coupling(body, coupling, **arguments) == coupling
    assert coupling["body_sha256"] == api.canonical_hash(body)
    assert coupling["shape_sha256"] == api.shape_hash(body)
    assert _raw(body) == original
    arguments["dt"] = 1 / 30
    arguments["coordinate_frame"]["clock_binding"]["control_dt_s"] = 1 / 30
    with pytest.raises(ValueError, match="source frame period"):
        api.validate_camera_coupling(body, coupling, **arguments)
    assert _raw(body) == original


@pytest.mark.parametrize("method", ["camera-root-body-v1", "two-view-body-v1", "two-view-body-refinement-v1", None])
def test_other_camera_methods_reject_before_hashing(api, monkeypatch, method):
    body = _fixture()["body"]
    body["camera_reference"]["method"] = method
    monkeypatch.setattr(api, "canonical_hash", _deny)
    monkeypatch.setattr(api, "validate_camera_binding", _deny)
    with pytest.raises(ValueError, match="grounded two-view"):
        api.build_camera_coupling(body)


@pytest.mark.parametrize("change", ["ground_missing", "body_dt", "angular_clock", "root_clock", "ground_height", "source_id", "reflection", "oversized_root"])
def test_changed_body_contract_is_not_repaired(api, change):
    fixture = _fixture()
    body = fixture["body"]
    if change == "ground_missing":
        body["camera_reference"].pop("ground")
    elif change == "body_dt":
        body["dt"] = 1 / 240
        _pin(fixture)
    elif change == "angular_clock":
        body["reference"][2]["time_s"] += .001
        _pin(fixture)
    elif change == "root_clock":
        body["root_reference"][2]["time_s"] += .001
        _pin(fixture)
    elif change == "ground_height":
        fixture["ground"]["height_m"] += .1
        _pin(fixture)
    elif change == "source_id":
        body["reference_evidence"][2]["evidence_id"] = "c" * 64 + ":frame:4"
        _pin(fixture)
    elif change == "reflection":
        body["body_frame"]["rotation"][0][0] = -1.
        _pin(fixture)
    else:
        body["root_reference"][-1]["translation"] = [50.01, 0., 0.]
        row = body["camera_reference"]["root_refinement"]["roots"][-1]
        row["root_world_m"] = [51.01, 2., 1.]
        row["hip_world_m"] = [row["root_world_m"][:], row["root_world_m"][:]]
        _pin(fixture)
    before = _raw(body)
    with pytest.raises(ValueError):
        api.build_camera_coupling(body)
    assert _raw(body) == before


@pytest.mark.parametrize("change", ["scale", "origin", "calibration", "units", "dt", "fps", "offset", "source", "frames", "mode"])
def test_shared_placement_clock_and_source_must_be_exact(api, change):
    body = _fixture()["body"]
    coupling = api.build_camera_coupling(body)
    arguments = _scene_arguments(api, body)
    frame = arguments["coordinate_frame"]
    if change == "scale":
        frame["scale_bridge"]["body_height_to_world_scale"] = 1.01
    elif change == "origin":
        frame["scale_bridge"]["origin_world"][0] = .1
    elif change == "calibration":
        frame["calibration_status"] = "calibrated"
    elif change == "units":
        frame["units"]["length"] = "measured_metres"
    elif change == "dt":
        arguments["dt"] = 1 / 60
    elif change == "fps":
        frame["clock_binding"]["source_fps"] = 60
    elif change == "offset":
        frame["clock_binding"]["source_time_offset_s"] = .1
    elif change == "source":
        arguments["source_binding"]["source_sha256"] = "c" * 64
    elif change == "frames":
        arguments["source_binding"]["total_source_frames"] -= 1
    else:
        arguments["source_binding"]["mode"] = "analytic_fixture"
    with pytest.raises(ValueError):
        api.validate_camera_coupling(body, coupling, **arguments)


@pytest.mark.parametrize("change", ["pin", "transform", "assumption", "bounds", "boolean_scale", "extra"])
def test_envelope_relabeling_and_weakening_reject(api, change):
    body = _fixture()["body"]
    coupling = api.build_camera_coupling(body)
    if change == "pin":
        coupling["body_sha256"] = "f" * 64
    elif change == "transform":
        coupling["world_to_local"]["translation"][0] = 0.
    elif change == "assumption":
        coupling["assumptions"]["anatomy"] = "measured"
    elif change == "bounds":
        coupling["runtime_bounds"]["root_abs_max"] = 1000.
    elif change == "boolean_scale":
        coupling["world_to_local"]["scale"] = True
    else:
        coupling["learned_policy_available"] = True
    with pytest.raises(ValueError, match="exact source"):
        api.validate_camera_coupling(body, coupling, **_scene_arguments(api, body))


@pytest.mark.parametrize("change", ["poses", "frames", "bytes", "cycle", "nonfinite", "depth"])
def test_malformed_or_excessive_input_rejects_before_hash_and_binding(api, monkeypatch, change):
    body = _fixture()["body"]
    if change == "poses":
        body["reference"] = body["reference"][:1] * 901
    elif change == "frames":
        body["camera_reference"]["root_refinement"]["source_frames"] = 1801
    elif change == "bytes":
        body["oversized"] = "x" * (api.MAX_BODY_BYTES + 1)
    elif change == "cycle":
        body["cycle"] = body
    elif change == "nonfinite":
        body["bad_number"] = float("inf")
    else:
        nested = body
        for _ in range(70):
            nested["nested"] = {}
            nested = nested["nested"]
    monkeypatch.setattr(api, "canonical_hash", _deny)
    monkeypatch.setattr(api, "validate_camera_binding", _deny)
    with pytest.raises(ValueError):
        api.build_camera_coupling(body)


def test_maximum_retained_pose_count_keeps_every_clock_row(api):
    fixture = _fixture()
    body = fixture["body"]
    binding = body["camera_reference"]
    root = binding["root_refinement"]
    root["source_frames"] = 1800
    templates = {name: deepcopy(body[name][0]) for name in ("reference", "reference_evidence", "root_reference")}
    source_template = deepcopy(root["roots"][0])
    for name in templates:
        body[name] = []
    root["roots"], binding["source_frame_ids"] = [], []
    for frame in range(0, 1800, 2):
        evidence_id = f"{body['source_sha256']}:frame:{frame}"
        for name, template in templates.items():
            row = deepcopy(template)
            row["time_s"] = frame / 30
            if name == "reference_evidence":
                row["evidence_id"] = evidence_id
            body[name].append(row)
        source = deepcopy(source_template)
        source.update(frame=frame, time_s=frame / 30,
                      source_evidence_ids=[f"{view['source_sha256']}:frame:{frame}" for view in root["views"]])
        root["roots"].append(source)
        binding["source_frame_ids"].append(evidence_id)
    _pin(fixture)
    coupling = api.build_camera_coupling(body)
    assert coupling["clock"]["retained_poses"] == 900
    assert coupling["clock"]["total_source_frames"] == 1800
    assert coupling["clock"]["horizon_steps"] == 1798
    assert api.validate_camera_coupling(body, coupling, **_scene_arguments(api, body)) == coupling


def test_deadline_is_cooperative_and_checked_before_hashing(api, monkeypatch):
    body = _fixture()["body"]
    ticks = iter((0., 6.))
    monkeypatch.setattr(api.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(api, "canonical_hash", _deny)
    with pytest.raises(ValueError, match="cooperative 5-second"):
        api.build_camera_coupling(body)
