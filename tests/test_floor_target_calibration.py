from copy import deepcopy
from hashlib import sha256
import builtins
import json
import math
import socket
import subprocess

import numpy as np
import pytest

from weft import floor_target_calibration as core
from weft.io import canonical_hash
from test_source_surface_anchors import fixture as observations_fixture


SOURCE = "a" * 64
R_CT = np.diag([1., -1., -1.])
T_CT = np.array([0., 0., 2.])


def raw(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def authored_fixture(*, mode="planar_target", units="m", radial=0., skew=0., count=4, decoded=3, fps=30., rotation=None, translation=None):
    """Simulated measurement receipts; no source video or physical truth."""
    observations, camera, _, _ = observations_fixture(decoded)
    rotation = R_CT if rotation is None else np.asarray(rotation)
    translation = T_CT if translation is None else np.asarray(translation)
    observations["source"].update(metadata_frames=count, fps=fps)
    camera.update(frame_count=count, fps=fps)
    camera["frames"] = [{"frame": f, "time_s": f/fps, "R": rotation.tolist(), "t": translation.tolist()} for f in range(count)]
    camera["K"][0][1] = skew; camera["radial_k1"] = radial
    evidence_ids = []
    for index, item in enumerate(observations["evidence"]):
        item.update(kind="target_measurement", status="observed", time_s=index/fps)
        observations["observations"][index]["time_s"] = index/fps
        evidence_ids.append(item["id"])
    if decoded > 2:
        observations["observations"][2]["partition"] = "audit"
    obs_raw = raw(observations); obs_sha = sha256(obs_raw).hexdigest()
    refs = [{"document_sha256": obs_sha, "evidence_id": evidence_ids[0]}]
    world = {"id": "authored-world", "length_unit": "m", "axes": "x_right_y_away_z_up"}
    def record(identifier, kind, value, uncertainty, evidence_refs=None):
        return {"id": identifier, "kind": kind, "quality": "measured", "method": "Authored analytical measurement simulation",
            "provenance": "Synthetic test receipt; no independent real measurement", "evidence_refs": refs if evidence_refs is None else evidence_refs,
            "uncertainty": {"interpretation": "declared_absolute_bound", **uncertainty}, "value": value}
    camera_raw = raw(camera) if mode == "supplied" else None
    camera_value = {"source_sha256": SOURCE, "units": "m", "width": 100, "height": 80, "fps": fps, "metadata_frames": count,
        "decoded_frames": decoded, "K": camera["K"], "distortion": {"model": "pinhole_k1", "k1": radial},
        "world_frame_id": world["id"], "camera_sha256": None if camera_raw is None else sha256(camera_raw).hexdigest()}
    rows = [record("camera", "camera_calibration", camera_value, {"intrinsics_px": .1, "radial_k1": .001,
        "translation_m": .001 if mode == "supplied" else None, "rotation_rad": .001 if mode == "supplied" else None})]
    factor = 1000 if units == "mm" else 1
    points = [[-.4, -.3], [.4, -.3], [.4, .3], [-.4, .3]]
    if mode == "planar_target":
        rows.append(record("target", "planar_target", {"id": "board", "frame_id": "target", "units": units,
            "entity_ref": {"document_sha256": obs_sha, "object_id": "person"},
            "corners": [{"id": str(i), "xy": [v*factor for v in p]} for i, p in enumerate(points)]},
            {"point_position": .001*factor, "scale_relative": .001}))
        claim = {"status": "observed", "method": "Authored measured-relation simulation", "uncertainty": "Test-only .001m/.001rad declaration", "evidence_refs": refs}
        relation = {"schema_version": 1, "profile": "world-alignment-v1", "from_frame": {"id": "target", "length_unit": "m", "axes": "right_handed_xyz"},
            "to_frame": world, "spatial": {"scale": 1., "rotation_wxyz": [1., 0., 0., 0.], "translation": [0., 0., 0.]},
            "clock": {"rate": 1., "source_origin_s": 0., "target_origin_s": 0.}, "claims": {"spatial": claim, "clock": claim}}
        rows.append(record("relation", "target_to_world", relation, {"translation_m": .001, "rotation_rad": .001}))
        for frame in range(decoded):
            image_points = []
            for index, point in enumerate(points):
                xyz = rotation @ np.array([*point, 0.]) + translation
                uv = xyz[:2]/xyz[2]; uv *= 1+radial*float(uv@uv)
                pixel = np.asarray(camera["K"]) @ np.array([*uv, 1.])
                image_points.append({"corner_id": str(index), "status": "observed", "pixel": pixel[:2].tolist(), "uncertainty_px": .2, "reason": None})
            value = {"frame": frame, "nominal_time_s": frame/fps, "source_time_s": frame/fps,
                "observation_ref": {"document_sha256": obs_sha, "object_id": "person", "observation_id": observations["observations"][frame]["id"]},
                "partition": observations["observations"][frame]["partition"], "status": "observed", "points": image_points}
            rows.append(record(f"corners-{frame}", "target_corners", value, {"pixel_error_note": "Authored .2px declaration"},
                [{"document_sha256": obs_sha, "evidence_id": evidence_ids[frame]}]))
    bundle = {"schema_version": 1, "profile": core.MEASUREMENT_PROFILE, "source_sha256": SOURCE, "records": rows}
    floor = record("floor", "floor_plane", {"world_frame_id": world["id"], "normal_world": [0., 0., 1.], "height_m": 0.,
        "origin_convention": "Independently declared synthetic world floor origin; not inferred from board", "target_to_world_ref": None},
        {"height_m": .001, "normal_rad": .001})
    floor_bundle = {"schema_version": 1, "profile": core.MEASUREMENT_PROFILE, "source_sha256": SOURCE, "records": [floor]}
    ground = {"schema_version": 1, "source_sha256": SOURCE, "calibration_sha256": canonical_hash(camera), "world_axes": "right_handed_z_up",
        "units": "m", "height_m": 0., "quality": "authored", "provenance": "Authored supplied nominal floor"}
    request = {"schema_version": 1, "profile": core.REQUEST_PROFILE, "source_sha256": SOURCE, "mode": mode,
        "observation_sha256s": [obs_sha], "measurement_sha256s": [], "supplied_camera_sha256": None if camera_raw is None else sha256(camera_raw).hexdigest(),
        "supplied_ground_sha256": None, "world_frame": world, "anchor_frame": 0 if mode == "planar_target" else None,
        "check_frames": [1] if mode == "planar_target" and decoded > 1 else [], "evaluation_frames": [2] if mode == "planar_target" and decoded > 2 else [],
        "fixed_interval": {"first_frame": 0, "last_frame": count-1, "camera_static": True, "target_static": True,
            "method": "Explicit authored static fixture declaration", "uncertainty": "Not verified physical stability", "evidence_refs": refs} if mode == "planar_target" else None,
        "measurement_refs": {"camera": None, "target": None, "target_to_world": None, "floor": None}}
    return {"observations": observations, "bundle": bundle, "floor_bundle": floor_bundle, "request": request,
            "camera_raw": camera_raw, "ground_raw": raw(ground) if mode == "supplied" else None}


def encoded(data):
    data = deepcopy(data); request = data["request"]
    obs_raw = raw(data["observations"]); obs_sha = sha256(obs_raw).hexdigest()
    # Fixture changes preserve original observation identity unless explicitly tested.
    request["observation_sha256s"] = [obs_sha]
    bundle_raw = raw(data["bundle"]); bundle_sha = sha256(bundle_raw).hexdigest()
    refs = {"camera": {"document_sha256": bundle_sha, "record_id": "camera"}, "target": None, "target_to_world": None}
    if request["mode"] == "planar_target":
        refs.update(target={"document_sha256": bundle_sha, "record_id": "target"}, target_to_world={"document_sha256": bundle_sha, "record_id": "relation"})
    data["floor_bundle"]["records"][0]["value"]["target_to_world_ref"] = refs["target_to_world"]
    floor_raw = raw(data["floor_bundle"]); floor_sha = sha256(floor_raw).hexdigest()
    refs["floor"] = {"document_sha256": floor_sha, "record_id": "floor"}
    request.update(measurement_refs=refs, measurement_sha256s=[bundle_sha, floor_sha],
        supplied_camera_sha256=None if data["camera_raw"] is None else sha256(data["camera_raw"]).hexdigest(),
        supplied_ground_sha256=None if data["ground_raw"] is None else sha256(data["ground_raw"]).hexdigest())
    return {"observation_documents": {obs_sha: obs_raw}, "measurement_documents": {bundle_sha: bundle_raw, floor_sha: floor_raw},
        "source_sha256": SOURCE, "request_document": raw(request), "supplied_camera_document": data["camera_raw"], "supplied_ground_document": data["ground_raw"]}


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"cv2", "mujoco", "mediapipe", "tensorflow", "torch"}:
            raise AssertionError("native/model import forbidden in pure test")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    def forbidden(*args, **kwargs):
        raise AssertionError("network/process work forbidden")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(core, "_solve_ippe", lambda *_: [{"R_ct": R_CT.tolist(), "t_ct": T_CT.tolist(), "solver_error": None}])


def run(data=None, **changes):
    return core.prepare_floor_target_calibration(**(encoded(authored_fixture() if data is None else data) | changes))


def record(data, name):
    return next(r for r in data["bundle"]["records"] if r["id"] == name)


def test_known_nominal_geometry_preserves_floor_slots_and_hash_domains():
    result = run(); report = result.describe(); files = result.files()
    assert report["status"] == "qualified_nominal" and report["work"]["solver_calls"] == 1
    assert report["counts"]["source_slots"] == 4 and report["counts"]["prepared_construction_frames"] == 2
    assert report["frames"][2]["reason"] == "excluded_source_partition" and report["frames"][2]["points"] == []
    assert report["frames"][3]["source_status"] == "metadata_only_not_decoded"
    camera = json.loads(files["camera.json"]); ground = json.loads(files["ground.json"])
    assert camera["frames"][0]["R"] == R_CT.tolist() and camera["frames"][0]["t"] == T_CT.tolist()
    assert ground["height_m"] == 0 and ground["calibration_sha256"] == canonical_hash(camera)
    assert ground["calibration_sha256"] != sha256(files["camera.json"]).hexdigest()
    assert camera["quality"] == ground["quality"] == "reconstructed"
    assert report["uncertainty"]["propagation"] == "not_propagated" and report["claims"]["contact"] == "unknown"
    assert not report["claims"]["physical_validation"] and not report["claims"]["body_or_floor_installed"]
    assert set(files) == {"calibration.json", "camera.json", "ground.json"}


@pytest.mark.parametrize("units", ["m", "mm"])
def test_target_world_composition_and_original_units(units):
    data = authored_fixture(units=units)
    relation = record(data, "relation")["value"]
    relation["spatial"].update(rotation_wxyz=[math.cos(.2), 0., 0., math.sin(.2)], translation=[.4, .2, .7])
    data["floor_bundle"]["records"][0]["value"]["height_m"] = -.1
    result = run(data); report = result.describe(); camera = json.loads(result.files()["camera.json"])
    rwt = core._matrix(relation["spatial"]["rotation_wxyz"])
    rcw = R_CT @ rwt.T
    assert np.asarray(camera["frames"][0]["R"]) == pytest.approx(rcw)
    assert camera["frames"][0]["t"] == pytest.approx(T_CT - rcw @ [.4, .2, .7])
    assert json.loads(result.files()["ground.json"])["height_m"] == -.1
    assert next(r for r in report["measurement_records"] if r["id"] == "target")["target_units"] == units


@pytest.mark.parametrize("radial,skew", [(.1, 7.), (-.05, 3.)])
def test_existing_full_k_radial_inverse_and_forward_are_used(radial, skew, monkeypatch):
    captured = []
    def solve(points, rays):
        captured.append(rays.copy())
        return [{"R_ct": R_CT.tolist(), "t_ct": T_CT.tolist(), "solver_error": None}]
    monkeypatch.setattr(core, "_solve_ippe", solve)
    report = run(authored_fixture(radial=radial, skew=skew)).describe()
    assert report["status"] == "qualified_nominal" and len(captured) == 1
    assert captured[0] == pytest.approx(np.asarray([[-.2, .15], [.2, .15], [.2, -.15], [-.2, -.15]]), abs=1e-10)
    assert report["alternatives"][0]["checks"]["maximum_px"] < 1e-10


@pytest.mark.parametrize("case,reason", [
    ("height", "missing_floor_height_or_origin_convention"), ("tilted", "nonhorizontal_floor_unsupported"),
    ("units", "unknown_target_units"), ("uncertainty", "unknown_required_measurement_uncertainty"),
    ("floor_quality", "measurement_quality_not_declared_measured"), ("fixed", "missing_fixed_camera_target_declaration"),
    ("anchor", "missing_anchor_frame"), ("distortion", "unsupported_distortion_model"),
    ("clock", "required_construction_frame_source_clock_or_partition_mismatch"),
    ("collinear", "degenerate_or_ill_conditioned_target"), ("missing", "required_construction_frame_corner_frame_missing")])
def test_missing_unsupported_or_degenerate_support_abstains_before_solve(case, reason, monkeypatch):
    data = authored_fixture()
    if case == "height": data["floor_bundle"]["records"][0]["value"]["height_m"] = None
    elif case == "tilted": data["floor_bundle"]["records"][0]["value"]["normal_world"] = [0., .1, .99]
    elif case == "units": record(data, "target")["value"]["units"] = "unknown"
    elif case == "uncertainty": record(data, "target")["uncertainty"]["point_position"] = None
    elif case == "floor_quality": data["floor_bundle"]["records"][0]["quality"] = "authored"
    elif case == "fixed": data["request"]["fixed_interval"] = None
    elif case == "anchor": data["request"]["anchor_frame"] = None
    elif case == "distortion": record(data, "camera")["value"]["distortion"]["k2"] = .01
    elif case == "clock": record(data, "corners-0")["value"]["source_time_s"] = .001
    elif case == "collinear":
        for i, c in enumerate(record(data, "target")["value"]["corners"]): c["xy"] = [i*.1, 0.]
    elif case == "missing": record(data, "corners-0")["value"]["status"] = "missing"
    monkeypatch.setattr(core, "_solve_ippe", lambda *_: pytest.fail("solver entered before qualification"))
    report = run(data).describe()
    assert report["status"] == "unavailable" and report["reason"] == reason
    assert report["work"]["solver_calls"] == 0 and report["counts"]["source_slots"] == 4
    assert report["artifact_bindings"] is None


def test_excluded_and_later_payloads_are_never_prepared(monkeypatch):
    data = authored_fixture()
    record(data, "corners-2")["value"]["points"] = {"coordinate_sentinel": [928374, 12]}
    calls = []; original = core._corner_payload
    def checked(record, *args):
        calls.append(record["value"]["frame"])
        return original(record, *args)
    monkeypatch.setattr(core, "_corner_payload", checked)
    report = run(data).describe()
    assert calls == [0, 1] and "coordinate_sentinel" not in json.dumps(report)
    assert report["claims"]["evaluation_used"] is False


@pytest.mark.parametrize("delta,expected", [(0., "qualified_nominal"), (1e-11, "qualified_nominal"), (.001, "ambiguous")])
def test_all_planar_alternatives_retained_without_lowest_residual_selection(delta, expected, monkeypatch):
    monkeypatch.setattr(core, "_solve_ippe", lambda *_: [
        {"R_ct": R_CT.tolist(), "t_ct": T_CT.tolist(), "solver_error": None},
        {"R_ct": R_CT.tolist(), "t_ct": (T_CT + [delta, 0, 0]).tolist(), "solver_error": None}])
    result = run(); report = result.describe()
    assert report["status"] == expected and len(report["alternatives"]) == 2
    assert report["work"]["point_reprojections"] == 16
    assert report["alternatives"][1]["duplicate_of"] == (0 if delta <= 1e-9 else None)
    if expected == "ambiguous": assert set(result.files()) == {"calibration.json"}


@pytest.mark.parametrize("case", ["reflection", "nonfinite", "depth", "residual", "none"])
def test_failed_solver_alternatives_are_kept_without_geometry(case, monkeypatch):
    pose = {"R_ct": R_CT.tolist(), "t_ct": T_CT.tolist(), "solver_error": None}
    if case == "reflection": pose["R_ct"] = np.diag([1., 1., -1.]).tolist()
    elif case == "nonfinite": pose["t_ct"] = [float("nan"), 0., 2.]
    elif case == "depth": pose["t_ct"] = [0., 0., -.1]
    elif case == "residual": pose["t_ct"] = [.1, 0., 2.]
    monkeypatch.setattr(core, "_solve_ippe", lambda *_: [] if case == "none" else [pose])
    result = run(); report = result.describe()
    assert report["status"] == "rejected" and report["selected_index"] is None
    assert report["counts"]["alternatives"] == (0 if case == "none" else 1)
    assert set(result.files()) == {"calibration.json"} and b"NaN" not in result.files()["calibration.json"]


def test_supplied_exact_bytes_quality_and_distinct_hashes_with_zero_solver(monkeypatch):
    data = authored_fixture(mode="supplied")
    monkeypatch.setattr(core, "_solve_ippe", lambda *_: pytest.fail("supplied mode invoked solver"))
    result = run(data); report = result.describe()
    assert report["status"] == "qualified_nominal" and report["selected_index"] is None
    assert result.files()["camera.json"] == data["camera_raw"] and result.files()["ground.json"] == data["ground_raw"]
    assert report["artifact_bindings"]["qualities"] == {"camera": "authored", "ground": "authored"}
    assert report["work"]["solver_calls"] == 0 and report["measurement_records"][0]["target_units"] is None


@pytest.mark.parametrize("case", ["raw_pin", "height", "uncertainty", "quality"])
def test_schema_valid_supplied_prior_does_not_replace_measurement_support(case):
    data = authored_fixture(mode="supplied")
    if case == "raw_pin": record(data, "camera")["value"]["camera_sha256"] = "b"*64
    elif case == "height": data["floor_bundle"]["records"][0]["value"]["height_m"] = .1
    elif case == "uncertainty": record(data, "camera")["uncertainty"]["translation_m"] = None
    elif case == "quality": record(data, "camera")["quality"] = "authored"
    report = run(data).describe()
    assert report["status"] == "unavailable" and report["artifact_bindings"] is None


def test_inputs_nonmutated_and_cached_results_isolated():
    inputs = encoded(authored_fixture()); original = deepcopy(inputs)
    result = core.prepare_floor_target_calibration(**inputs)
    assert inputs == original
    result.describe()["frames"].clear(); result.files().clear()
    assert len(result.describe()["frames"]) == 4 and len(result.files()) == 3


@pytest.mark.parametrize("case", ["hash", "source", "count", "duplicate_pin", "byte_bound"])
def test_malformed_custody_and_bounds_fail_before_solver(case, monkeypatch):
    data = authored_fixture(); inputs = encoded(data)
    if case == "hash":
        key = next(iter(inputs["measurement_documents"])); inputs["measurement_documents"][key] += b" "
    elif case == "source": inputs["source_sha256"] = "b"*64
    elif case == "count":
        record(data, "camera")["value"]["decoded_frames"] = True; inputs = encoded(data)
    elif case == "duplicate_pin":
        request = json.loads(inputs["request_document"]); request["measurement_sha256s"] *= 2; inputs["request_document"] = raw(request)
    elif case == "byte_bound": inputs["request_document"] = b" " * ((64 << 10)+1)
    monkeypatch.setattr(core, "_solve_ippe", lambda *_: pytest.fail("solver entered malformed input"))
    if case == "count":
        assert core.prepare_floor_target_calibration(**inputs).describe()["status"] == "unavailable"
    else:
        with pytest.raises(ValueError): core.prepare_floor_target_calibration(**inputs)


def test_deadline_after_nonpreemptible_solver_never_returns_success(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(core.time, "monotonic", lambda: clock[0])
    def slow(*_):
        clock[0] = 6.
        return [{"R_ct": R_CT.tolist(), "t_ct": T_CT.tolist(), "solver_error": None}]
    monkeypatch.setattr(core, "_solve_ippe", slow)
    with pytest.raises(TimeoutError): run()


def test_mapping_iteration_and_request_pin_order_do_not_select_geometry():
    inputs = encoded(authored_fixture()); a = core.prepare_floor_target_calibration(**inputs).files()
    inputs["measurement_documents"] = dict(reversed(list(inputs["measurement_documents"].items())))
    assert core.prepare_floor_target_calibration(**inputs).files() == a


def test_complete_1800_slots_are_bounded_without_per_frame_solves():
    data = authored_fixture(mode="supplied", count=1800, decoded=3)
    result = run(data); report = result.describe()
    assert report["counts"]["source_slots"] == 1800 and report["work"]["solver_calls"] == 0
    assert sum(map(len, result.files().values())) < 4 << 20


def test_native_clock_without_nominal_fps_is_retained_not_retimed():
    data = authored_fixture(mode="supplied", count=3, decoded=3)
    observations = data["observations"]
    observations["schema_version"] = 3
    observations["source"].pop("fps")
    observations["source"]["frame_times_s"] = [0., .034, .067]
    observations["time"] = {"unit": "seconds", "origin": "submitted_video_start", "basis": "decoder_presentation_timestamps", "status": "observed"}
    for rows in (observations["observations"], observations["evidence"]):
        for row in rows: row["time_s"] = observations["source"]["frame_times_s"][row["frame"]]
    report = run(data).describe()
    assert report["status"] == "unavailable" and report["reason"] == "missing_source_nominal_clock"
    assert report["source"]["native_frame_times_s"] == [0., .034, .067]
    assert [row["source_time_s"] for row in report["frames"]] == [0., .034, .067]
    assert all(row["nominal_time_s"] is None for row in report["frames"])


def test_unknown_pixel_uncertainty_retains_named_statuses_without_solving(monkeypatch):
    data = authored_fixture(); record(data, "corners-0")["value"]["points"][0]["uncertainty_px"] = None
    monkeypatch.setattr(core, "_solve_ippe", lambda *_: pytest.fail("unknown measurement entered solve"))
    report = run(data).describe()
    assert report["status"] == "unavailable" and report["reason"] == "unknown_pixel_uncertainty"
    assert len(report["frames"][0]["points"]) == 4 and report["work"]["solver_calls"] == 0


def test_fractional_clock_oblique_stub_and_preserved_analytical_payload(monkeypatch, tmp_path):
    angle = .23
    rotation = np.asarray([[1., 0., 0.], [0., math.cos(angle), -math.sin(angle)], [0., math.sin(angle), math.cos(angle)]]) @ R_CT
    inputs = encoded(authored_fixture(fps=30000/1001, radial=.08, skew=4., rotation=rotation))
    monkeypatch.setattr(core, "_solve_ippe", lambda *_: [{"R_ct": rotation.tolist(), "t_ct": T_CT.tolist(), "solver_error": None}])
    result = core.prepare_floor_target_calibration(**inputs); report = result.describe()
    assert report["status"] == "qualified_nominal"
    camera = json.loads(result.files()["camera.json"])
    assert camera["fps"] == 30000/1001 and camera["frames"][1]["time_s"] == 1001/30000
    assert report["alternatives"][0]["checks"]["maximum_px"] < 1e-10
    retained = {"request.json": inputs["request_document"], **result.files()}
    for kind in ("observation", "measurement"):
        retained.update({f"{kind}-{digest}.json": value for digest, value in inputs[kind+"_documents"].items()})
    for name, value in retained.items(): (tmp_path/name).write_bytes(value)
    (tmp_path/"fixture.json").write_bytes(raw({"profile": "floor-target-authored-stub-fixture-v1", "physical_measurements": False,
        "actual_ippe_calls": 0, "stub_calls": 1, "files": {name: sha256(value).hexdigest() for name, value in retained.items()}}))
    print("PRESERVED_ANALYTICAL_PAYLOAD=" + str(tmp_path))


def test_collapsed_image_rays_abstain_before_ippe(monkeypatch):
    data = authored_fixture()
    for point in record(data, "corners-0")["value"]["points"]:
        point["pixel"] = [50., 40.]
    calls = []
    def solve(*args):
        calls.append(args)
        return [{"R_ct": R_CT.tolist(), "t_ct": T_CT.tolist(), "solver_error": None}]
    monkeypatch.setattr(core, "_solve_ippe", solve)
    report = run(data).describe()
    assert report["status"] == "unavailable" and report["reason"] == "degenerate_or_ill_conditioned_image_rays"
    assert calls == [] and report["work"]["solver_calls"] == 0
