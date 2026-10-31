from copy import deepcopy
from hashlib import sha256
import builtins
import json

import pytest

from weft.exercise_support_evidence import estimate_exercise_support, CONTRACT
from weft.ground_orientation import estimate_ground_orientation
from weft.observations import adapt_tracks


SOURCE = "a" * 64


def pose(count=30, fps=20, stride=1):
    rows = []
    for index in range(count):
        frame = index * stride
        image = [{"x": .5, "y": .5, "z": 0., "presence": .95, "visibility": .95} for _ in range(33)]
        world = deepcopy(image)
        for joint, xy in {11: (.4, .2), 12: (.6, .2), 23: (.45, .5), 24: (.55, .5),
                          15: (.3, .55), 16: (.7, .55), 29: (.4, .8), 30: (.6, .8),
                          31: (.38, .81), 32: (.62, .81)}.items():
            image[joint].update(x=xy[0], y=xy[1])
        for joint, xyz in {29: (-.2, 0., -.1), 30: (.2, 0., -.1),
                           31: (-.2, 0., .1), 32: (.2, 0., .1)}.items():
            world[joint].update(x=xyz[0], y=xyz[1], z=xyz[2])
        rows.append({"id": f"{SOURCE}:frame:{frame}", "frame": frame, "time_s": frame / fps,
                     "decoder_timestamp_s": frame / fps + .014, "status": "tracked", "pose_count": 1,
                     "landmarks": image, "world_landmarks": world})
    return {"schema_version": 1, "profile": "exercise-pose-v1", "source_sha256": SOURCE,
            "width": 640, "height": 480, "fps": fps, "frame_count": count * stride,
            "decoded_frames": count * stride, "inference_stride": stride, "frames": rows,
            "model": {"name": "Authored analytical image and foot-plane fixture"}}


def prior(source=SOURCE):
    return {"mode": "fixed_source_camera", "status": "authored_unverified", "source_sha256": source,
            "rationale": "Explicit analytical camera assumption, not measured calibration"}


def equipment(data, *, native=False):
    rows = []
    for row in data["frames"]:
        hand = row["landmarks"][15]
        rows.append({"id": f"{SOURCE}:frame:{row['frame']}:object:weight", "frame": row["frame"],
                     "time_s": row["time_s"], "object_id": "weight", "x_px": hand["x"] * data["width"],
                     "y_px": hand["y"] * data["height"], "width_px": 30., "height_px": 30.,
                     "status": "measured_image_projection"})
    source = {"profile": "planar-rigid-v1", "source_sha256": SOURCE,
              **{key: data[key] for key in ("width", "height", "fps", "frame_count", "decoded_frames")},
              "generation": rows, "development_validation": [],
              "objects": [{"id": "weight", "label": "Authored image weight fixture", "role": "passive"}],
              "visibility": [{"frame": row["frame"], "object_id": "weight", "status": "measured"} for row in rows]}
    result = adapt_tracks(source, profile="planar", tracks={"observations": rows})
    if native:
        result["schema_version"] = 3
        result["source"].pop("fps")
        result["source"]["frame_times_s"] = [i / data["fps"] + .014 for i in range(data["decoded_frames"])]
        result["time"] = {"unit": "seconds", "origin": "submitted_video_start", "basis": "decoder_presentation_timestamps", "status": "observed"}
        for row in result["observations"]:
            row["time_s"] += .014
        for row in result["evidence"]:
            row["time_s"] += .014
            if "time_s" in row["payload"]:
                row["payload"]["time_s"] += .014
    return result


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def test_stationary_points_keep_every_slot_and_never_claim_contact():
    data = pose(stride=2)
    data["frame_count"] += 1
    before = deepcopy(data)
    result = estimate_exercise_support(data, camera_prior=prior())
    assert data == before
    assert result["status"] == "candidates_only"
    assert result["counts"]["retained_ground_point_slots"] == 120
    assert sum(result["counts"]["ground_status_counts"].values()) == 18 * 4
    assert len(result["coverage"]["unsampled_decoded_frame_indices"]) == 30
    assert result["coverage"]["container_only_frame_indices"] == [60]
    assert all(point["status"] == "excluded_later_partition" for row in result["rows"][18:] for point in row["ground_points"])
    assert result["rows"][0]["decoder_timestamp_s"] == .014
    assert result["contact_state"] == "unknown" and result["ground_height"] is result["support_force"] is None
    assert result["promoted"] is False
    assert all(point["verified_contact"] is False for row in result["rows"] for point in row["ground_points"])


def test_later_landmarks_do_not_select_ground_candidates():
    data = pose()
    before = estimate_exercise_support(data, camera_prior=prior())
    for row in data["frames"][18:]:
        row["landmarks"], row["world_landmarks"] = "not read", "not read"
    after = estimate_exercise_support(data, camera_prior=prior())
    assert after["rows"] == before["rows"]
    assert after["pose_evidence_canonical_sha256"] != before["pose_evidence_canonical_sha256"]
    assert after["counts"] == before["counts"]


def test_missing_point_is_not_interpolated_across_temporal_window():
    data = pose()
    data["frames"][5]["landmarks"][29]["visibility"] = .1
    result = estimate_exercise_support(data, camera_prior=prior())
    affected = result["rows"][5]["ground_points"][0]
    assert affected["pixel"] is None and affected["confidence"] == .1
    assert affected["status"] == "unavailable" and affected["window_missing_points"] == 1
    assert result["rows"][4]["ground_points"][0]["status"] == "unavailable"
    assert result["rows"][12]["ground_points"][0]["status"] == "stationary_image_candidate"


def test_time_gap_cannot_be_bridged_as_stationarity():
    data = pose(count=20)
    del data["frames"][5:8]
    result = estimate_exercise_support(data, camera_prior=prior())
    assert "temporal_gap" in result["rows"][4]["window"]["reasons"]
    assert result["rows"][4]["ground_points"][0]["status"] == "unavailable"


def test_missing_torso_scale_is_unknown_without_image_width_fallback():
    data = pose()
    for row in data["frames"]:
        row["landmarks"][11]["visibility"] = .1
    result = estimate_exercise_support(data, camera_prior=prior())
    assert result["status"] == "unavailable_image_scale"
    assert result["image_scale"]["median_torso_span_px"] is None
    assert set(result["counts"]["ground_status_counts"]) == {"unavailable"}


def test_identity_raw_targets_stay_distinct_from_processed_points():
    data = pose()
    for i, row in enumerate(data["frames"]):
        row["identity_raw"] = deepcopy(row)
        row["landmarks"][29]["y"] += .15 * (i % 2)
    result = estimate_exercise_support(data, camera_prior=prior())
    assert result["rows"][4]["landmark_origin"] == "identity_raw"
    assert result["rows"][4]["ground_points"][0]["status"] == "stationary_image_candidate"
    data["frames"][0]["identity_raw"]["decoder_timestamp_s"] = .123
    with pytest.raises(ValueError, match="raw pose identity/clock"):
        estimate_exercise_support(data, camera_prior=prior())


@pytest.mark.parametrize("change", [lambda d: d.update(status="measured"), lambda d: d.update(mode="moving_camera"),
                                    lambda d: d.update(source_sha256="b" * 64)])
def test_fixed_camera_prior_is_explicit_source_bound_and_unverified(change):
    camera = prior(); change(camera)
    with pytest.raises(ValueError, match="fixed-camera prior"):
        estimate_exercise_support(pose(), camera_prior=camera)


def test_nominal_native_clocks_and_decode_metadata_are_not_conflated():
    data = pose()
    data["frames"][1]["decoder_timestamp_s"] = None
    data["frame_count"] = 28
    result = estimate_exercise_support(data, camera_prior=prior())
    assert result["rows"][1]["decoder_timestamp_s"] is None
    assert result["coverage"]["decoded_beyond_container_metadata_indices"] == [28, 29]
    data["frames"][1]["time_s"] += .014
    with pytest.raises(ValueError, match="nominal/native"):
        estimate_exercise_support(data, camera_prior=prior())


def test_moving_hand_and_equipment_couple_only_in_image_coordinates():
    data = pose()
    for i, row in enumerate(data["frames"]):
        row["landmarks"][15]["x"] += .005 * i
    document = raw(equipment(data))
    result = estimate_exercise_support(data, camera_prior=prior(), equipment_observations=document, equipment_ids=["weight"])
    pair = result["rows"][5]["equipment_pairs"][0]
    assert pair["status"] == "projected_co_motion_candidate"
    assert pair["grasp"] == "unknown" and pair["verified_contact"] is False
    assert pair["object_observation_id"] == f"{SOURCE}:observation:5:weight"
    assert result["equipment_observations_sha256"] == sha256(document).hexdigest()
    assert result["counts"]["retained_equipment_pair_slots"] == 60
    assert result["rows"][5]["equipment_pairs"][1]["status"] == "image_separation"


@pytest.mark.parametrize("mode", ["occluded", "withheld", "missing_native_time"])
def test_equipment_missing_or_heldout_evidence_never_becomes_coupling(mode):
    data = pose()
    document = equipment(data, native=mode == "missing_native_time")
    if mode == "occluded": document["observations"][5]["visibility"] = "partial"
    elif mode == "withheld": document["observations"][5]["partition"] = "audit"
    else: data["frames"][5]["decoder_timestamp_s"] = None
    result = estimate_exercise_support(data, camera_prior=prior(), equipment_observations=raw(document), equipment_ids=["weight"])
    pair = result["rows"][5]["equipment_pairs"][0]
    assert pair["status"] == "unavailable" and pair["image_box_px"] is None
    assert pair["equipment_slot_reason"] in {"equipment_occluded_or_unavailable", "equipment_partition_excluded", "equipment_clock_unavailable_or_different"}
    assert result["rows"][4]["equipment_pairs"][0]["status"] == "unavailable"


def test_equipment_requires_exact_source_and_explicit_selection():
    data = pose()
    with pytest.raises(ValueError, match="select bounded"):
        estimate_exercise_support(data, camera_prior=prior(), equipment_observations=raw(equipment(data)), equipment_ids=["unknown"])
    with pytest.raises(ValueError, match="observation document"):
        estimate_exercise_support(data, camera_prior=prior(), equipment_ids=["weight"])


def test_support_conditioned_ground_rejects_lifted_points_not_physical_failures():
    data = pose()
    for i, row in enumerate(data["frames"][8:18], 8):
        for index in (30, 32):
            row["landmarks"][index]["y"] -= .1 + .08 * (i % 2)
            row["world_landmarks"][index]["y"] = -(.4 + .6 * (i % 2))
    before = deepcopy(data)
    baseline = estimate_ground_orientation(data)
    selected = estimate_ground_orientation(data, support_prior=prior())
    assert data == before
    assert baseline["status"] == "unavailable"
    assert selected["status"] == "available"
    assert selected["normal"] == pytest.approx([0, 0, 1], abs=1e-12)
    assert selected["floor_offset"] is None
    assert selected["usable_frames"] + selected["rejected_frames"] == 18
    assert selected["support_selection"]["contact_state"] == "unknown"
    assert selected["method"] == "stationary_candidate_foot_covariance_v1"


def test_one_stationary_foot_cannot_identify_a_plane():
    data = pose()
    for i, row in enumerate(data["frames"]):
        for index in (30, 32):
            row["landmarks"][index]["y"] -= .1 * (i % 2)
    result = estimate_ground_orientation(data, support_prior=prior())
    assert result["status"] == "unavailable" and result["normal"] is None
    assert result["usable_frames"] == 0
    assert all(row["stationary_candidate_indices"] == [29, 31] for row in result["audit"])


def test_stationary_elevated_feet_do_not_establish_true_ground():
    data = pose()
    for row in data["frames"]:
        for index in (29, 30, 31, 32):
            row["world_landmarks"][index]["y"] = -.7
    result = estimate_ground_orientation(data, support_prior=prior())
    assert result["status"] == "available" and result["floor_offset"] is None
    assert result["support_selection"]["ground_height"] is None
    assert result["support_selection"]["physical_validation"] == "unverified"


def test_maximum_pose_envelope_stays_bounded_and_has_no_native_work(monkeypatch):
    data = pose(900, 30, 2)
    original = builtins.__import__
    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"mujoco", "mediapipe", "cv2", "torch", "PIL"}:
            raise AssertionError("Unexpected native/media/inference dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guard)
    result = estimate_exercise_support(data, camera_prior=prior())
    assert len(result["rows"]) == 900 and len(raw(result)) <= CONTRACT["maximum_output_bytes"]
    assert result["counts"]["retained_ground_point_slots"] == 3600


def _forbid_json_serialization(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("Rejected input reached whole-object JSON serialization")
    monkeypatch.setattr(json, "dumps", fail)


def test_oversized_pose_count_rejected_before_serialization(monkeypatch):
    data = pose(3)
    data["frames"] = [data["frames"][0]] * 901
    _forbid_json_serialization(monkeypatch)
    with pytest.raises(ValueError, match="bounded nonempty pose rows"):
        estimate_exercise_support(data, camera_prior=prior())


@pytest.mark.parametrize("escaped", [False, True])
def test_oversized_metadata_rejected_before_serialization(monkeypatch, escaped):
    data = pose(3)
    budget = CONTRACT["maximum_input_bytes"]
    data["notes"] = "🧶" * (budget // 12 + 1) if escaped else "x" * (budget + 1)
    _forbid_json_serialization(monkeypatch)
    with pytest.raises(ValueError, match="byte bound"):
        estimate_exercise_support(data, camera_prior=prior())


@pytest.mark.parametrize("kind", ["nodes", "depth", "cycle", "nonfinite", "non_string_key"])
def test_invalid_json_structure_rejected_before_serialization(monkeypatch, kind):
    data = pose(3)
    if kind == "nodes":
        data["notes"] = [None] * (CONTRACT["maximum_json_nodes"] + 1)
    elif kind == "depth":
        nested = []
        for _ in range(CONTRACT["maximum_json_depth"] + 1):
            nested = [nested]
        data["notes"] = nested
    elif kind == "cycle":
        data["notes"] = data
    elif kind == "nonfinite":
        data["notes"] = {"value": float("nan")}
    else:
        data["notes"] = {1: "non-string key"}
    _forbid_json_serialization(monkeypatch)
    with pytest.raises(ValueError):
        estimate_exercise_support(data, camera_prior=prior())


def test_canonical_pose_hash_preserved_with_unicode_escapes_and_numbers():
    data = pose(9)
    data["notes"] = {"🧶": ["é\n\r\t\b\f\x00\"\\", "\ud800", True, False, None,
                           -0.0, 1.7976931348623157e308, 10 ** 64]}
    expected = sha256(raw(data)).hexdigest()
    result = estimate_exercise_support(data, camera_prior=prior())
    assert result["pose_evidence_canonical_sha256"] == expected


def test_expired_input_deadline_precedes_serialization(monkeypatch):
    from weft.exercise_support_evidence import _prepare
    data = pose(3)
    _forbid_json_serialization(monkeypatch)
    with pytest.raises(TimeoutError):
        _prepare(data, prior(), deadline=0)


def _equivalent_equipment(tmp_path):
    from weft.source_equivalence import load_source_equivalence
    from test_source_equivalence import fixture, seal

    bundle = fixture(tmp_path / "mapping", derivative_offset=1)
    # Scale the recorded-producer fixture; no real media is decoded or claimed.
    for name in ("report.json", "pre-run.json"):
        path = bundle / name
        value = json.loads(path.read_bytes()); value["dimensions"] = [128, 128]
        path.write_bytes(raw(value))
    for side in ("source", "derivative"):
        path = bundle / (side + "-frames.sha256")
        path.write_text(path.read_text().replace("2x2", "128x128").replace(", 6, ", ", 24576, "))
        path = bundle / (side + "-frames.log")
        path.write_text(path.read_text().replace("2x2", "128x128"))
    seal(bundle)
    equivalence = load_source_equivalence(bundle)
    data = pose(3, 30); data.update(width=128, height=128)
    document = equipment(data, native=True)
    data = json.loads(raw(data).replace(SOURCE.encode(), equivalence.original_sha256.encode()))
    document = json.loads(raw(document).replace(SOURCE.encode(), equivalence.derivative_sha256.encode()))
    for row, mapped in zip(data["frames"], equivalence.frames, strict=True):
        row["decoder_timestamp_s"] = mapped.source_decoder_pts / 1000
    document["source"]["frame_times_s"] = [row.derivative_decoder_pts / 1000 for row in equivalence.frames]
    for row in document["observations"] + document["evidence"]:
        row["time_s"] = document["source"]["frame_times_s"][row["frame"]]
        if "time_s" in row.get("payload", {}):
            row["payload"]["time_s"] = row["time_s"]
    return data, document, equivalence


def test_verified_equipment_mapping_preserves_both_sources_clocks_and_ids(tmp_path):
    data, document, proof = _equivalent_equipment(tmp_path)
    original, document_bytes = deepcopy(data), raw(document)
    result = estimate_exercise_support(data, camera_prior=prior(data["source_sha256"]),
        equipment_observations=document_bytes, equipment_ids=["weight"], equipment_source_equivalence=proof)
    assert data == original and raw(document) == document_bytes
    assert result["source_sha256"] == proof.original_sha256
    assert result["equipment_source_sha256"] == proof.derivative_sha256
    assert result["equipment_observations_sha256"] == sha256(document_bytes).hexdigest()
    assert result["equipment_source_equivalence"]["manifest_sha256"] == proof.manifest_sha256
    mapping = result["rows"][0]["equipment_source_mapping"]
    assert mapping["original_decoder_time_s"] == 0 and mapping["derivative_decoder_time_s"] == .002
    assert mapping["original_native_pts"] == mapping["derivative_native_pts"] == 3
    pair = result["rows"][0]["equipment_pairs"][0]
    assert pair["object_observation_id"] == document["observations"][0]["id"]
    assert pair["object_evidence_ids"] == document["observations"][0]["evidence_ids"]
    assert pair["object_time_s"] == .002 and pair["equipment_slot_reason"] is None
    assert pair["status"] == "unavailable" and "insufficient_temporal_context" in pair["reasons"]
    assert pair["grasp"] == "unknown" and pair["verified_contact"] is False
    assert result["counts"]["retained_equipment_pair_slots"] == 6
    assert result["rows"][-1]["equipment_source_mapping"]["original_frame"] == 2
    assert all(p["status"] == "excluded_later_partition" for p in result["rows"][-1]["equipment_pairs"])


@pytest.mark.parametrize("stamp", [None, .002])
def test_mapped_equipment_missing_or_wrong_pose_decoder_clock_abstains(tmp_path, stamp):
    data, document, proof = _equivalent_equipment(tmp_path)
    data["frames"][0]["decoder_timestamp_s"] = stamp
    result = estimate_exercise_support(data, camera_prior=prior(data["source_sha256"]),
        equipment_observations=raw(document), equipment_ids=["weight"], equipment_source_equivalence=proof)
    pair = result["rows"][0]["equipment_pairs"][0]
    assert pair["equipment_slot_reason"] == "pose_decoder_clock_unavailable_or_different_from_equivalence"
    assert pair["status"] == "unavailable" and pair["image_box_px"] is None


@pytest.mark.parametrize("kind", ["missing_proof", "unverified_proof", "wrong_source", "coverage", "equipment_clock"])
def test_equipment_mapping_rejects_missing_or_conflicting_proof(tmp_path, kind):
    data, document, proof = _equivalent_equipment(tmp_path)
    if kind == "missing_proof":
        proof = None
    elif kind == "unverified_proof":
        proof = proof.inspection()
    elif kind == "wrong_source":
        data = json.loads(raw(data).replace(data["source_sha256"].encode(), b"b" * 64))
    elif kind == "coverage":
        data.update(decoded_frames=4, frame_count=4)
    else:
        document["source"]["frame_times_s"] = [stamp + .001 for stamp in document["source"]["frame_times_s"]]
        for row in document["observations"] + document["evidence"]:
            row["time_s"] += .001
            if "time_s" in row.get("payload", {}):
                row["payload"]["time_s"] = row["time_s"]
    with pytest.raises(ValueError, match="equivalence|Equivalence"):
        estimate_exercise_support(data, camera_prior=prior(data["source_sha256"]),
            equipment_observations=raw(document), equipment_ids=["weight"], equipment_source_equivalence=proof)


def test_unused_equipment_mapping_or_mapping_without_document_rejects(tmp_path):
    _, _, proof = _equivalent_equipment(tmp_path)
    data = pose(3)
    with pytest.raises(ValueError, match="unused equivalence"):
        estimate_exercise_support(data, camera_prior=prior(), equipment_observations=raw(equipment(data)),
            equipment_ids=["weight"], equipment_source_equivalence=proof)
    with pytest.raises(ValueError, match="exact equipment observation document"):
        estimate_exercise_support(data, camera_prior=prior(), equipment_source_equivalence=proof)
