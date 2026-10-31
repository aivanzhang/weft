from copy import deepcopy
import hashlib
from io import BytesIO
import json

import cv2
import numpy as np
from PIL import Image
import pytest

from weft import pose_pixel_support as support


def raw(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def png(image):
    stream = BytesIO()
    Image.fromarray(image).save(stream, format="PNG")
    return stream.getvalue()


def fixture(frames=(0, 2), shifts=None):
    rng = np.random.default_rng(217)
    first = cv2.GaussianBlur(rng.integers(15, 240, (192, 256), np.uint8), (0, 0), .5)
    points = [[45.5 + 35 * (j % 4), 45.5 + 40 * (j // 4)] for j in range(12)]
    shifts = shifts or [(8 * i, -3 * i) for i in range(len(frames))]
    rows, images = [], {}
    for frame, (dx, dy) in zip(frames, shifts, strict=True):
        image = cv2.warpAffine(first, np.float32([[1, 0, dx], [0, 1, dy]]), (256, 192))
        images[frame] = png(image)
        landmarks = [{"x": .5, "y": .5, "visibility": 1., "presence": 1.} for _ in range(33)]
        for (index, _), (x, y) in zip(support.JOINTS, points, strict=True):
            landmarks[index].update(x=(x + dx) / 256, y=(y + dy) / 192)
        rows.append({"id": f"{'1' * 64}:frame:{frame}", "frame": frame,
                     "time_s": frame / 30, "decoder_timestamp_s": 1.002 + frame / 30,
                     "pose_count": 1, "status": "tracked", "landmarks": landmarks})
    evidence = {"source_sha256": "1" * 64, "width": 256, "height": 192,
                "decoded_frames": frames[-1] + 1, "pose_identity": {"mode": "raw"}, "frames": rows}
    return evidence, images


def inputs(evidence, images):
    document = raw(evidence)
    entries = []
    for row in evidence["frames"]:
        if row["frame"] in images:
            entries.append({key: row[key] for key in ("frame", "id", "time_s", "decoder_timestamp_s")}
                           | {"image_sha256": sha(images[row["frame"]])})
    bindings = {"schema_version": 1, "profile": support.BINDING_PROFILE,
                "source_sha256": evidence["source_sha256"], "source_manifest_sha256": "2" * 64,
                "evidence_sha256": sha(document), "frames": entries}
    return document, images, raw(bindings)


@pytest.mark.parametrize("shift", [(8, -3), (14, 0)])
def test_true_motion_including_rapid_motion_is_not_flagged(shift):
    evidence, images = fixture(shifts=[(0, 0), shift])
    original = deepcopy(evidence), dict(images)
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert result["status"] == "complete" and result["processed_slots"] == 12
    assert result["status_counts"] == {"not_contradicted": 12}
    assert all(row["patch_displacement_px"] == list(shift) for row in result["rows"])
    assert max(row["model_patch_residual_px"] for row in result["rows"]) < 1e-12
    assert (evidence, images) == original


def test_wrong_model_jump_is_flagged_without_correcting_observations():
    evidence, images = fixture()
    for index, _ in support.JOINTS:
        evidence["frames"][1]["landmarks"][index]["x"] += 12 / 256
    document, images, bindings = inputs(evidence, images)
    result = support.diagnose_pose_pixel_support(document, images, bindings)
    assert result["status_counts"] == {"contradicted": 12}
    assert all(row["model_patch_residual_px"] == pytest.approx(12) for row in result["rows"])
    assert result["evidence_sha256"] == sha(document)
    assert all("corrected_landmarks" not in row for row in result["rows"])
    assert json.loads(document) == evidence


def test_fractional_source_seeds_keep_the_original_pixel_convention():
    evidence, images = fixture()
    for row in evidence["frames"]:
        for index, _ in support.JOINTS:
            row["landmarks"][index]["x"] += .37 / 256
            row["landmarks"][index]["y"] -= .23 / 192
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert result["status_counts"] == {"not_contradicted": 12}
    assert all(row["patch_displacement_px"] == [8, -3] for row in result["rows"])


def test_residual_gray_band_abstains():
    evidence, images = fixture()
    for index, _ in support.JOINTS:
        evidence["frames"][1]["landmarks"][index]["x"] += 3 / 256
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert result["status_counts"] == {"unavailable": 12}
    assert all(row["reason"] == "residual_uncertain" for row in result["rows"])


@pytest.mark.parametrize("kind", ["blank", "repeated", "occluded", "new_image"])
def test_texture_ambiguity_occlusion_and_edit_like_change_abstain(kind):
    evidence, images = fixture(shifts=[(0, 0), (0, 0)])
    if kind == "blank":
        images = {n: png(np.full((192, 256), 100, np.uint8)) for n in images}
    elif kind == "repeated":
        tile = np.random.default_rng(9).integers(0, 255, (8, 8), np.uint8)
        images = {n: png(np.tile(tile, (24, 32))) for n in images}
    elif kind == "occluded":
        images[2] = png(np.full((192, 256), 100, np.uint8))
    else:
        images[2] = png(np.random.default_rng(88).integers(0, 255, (192, 256), np.uint8))
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert result["status_counts"] == {"unavailable": 12}
    reasons = {row["reason"] for row in result["rows"]}
    assert reasons == {dict(blank="weak_texture", repeated="ambiguous_patch_match",
                            occluded="occluded_or_texture_unavailable", new_image="occluded_or_appearance_changed")[kind]}


def test_search_boundary_does_not_certify_large_motion():
    evidence, images = fixture(shifts=[(0, 0), (16, 0)])
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert result["status_counts"] == {"unavailable": 12}
    assert all(row["reason"] == "search_boundary" for row in result["rows"])


def test_every_original_pair_joint_slot_includes_weak_border_and_competing():
    evidence, images = fixture(frames=(0, 2, 4, 6), shifts=[(0, 0)] * 4)
    evidence["frames"][0]["landmarks"][11]["x"] = 2.5 / 256
    evidence["frames"][0]["landmarks"][12]["visibility"] = .1
    evidence["frames"][2]["pose_count"] = 2
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert result["pair_count"] == 3 and result["slot_count"] == result["processed_slots"] == 36
    assert result["sampled_image_frames"] == [0, 2, 4, 6]
    assert result["source_pose_slots"] == 4 and result["source_decoded_frames"] == 7
    assert result["rows"][0]["reason"] == "image_border_support"
    assert result["rows"][1]["reason"] == "weak_original_landmark"
    assert all(row["reason"] == "missing_or_competing_pose" for row in result["rows"][12:])
    assert [(row["frames"], row["joint_index"]) for row in result["rows"]] == [
        ([a, b], j) for a, b in [(0, 2), (2, 4), (4, 6)] for j, _ in support.JOINTS]


def test_reverse_ambiguous_patch_abstains_with_forward_match_retained():
    evidence, images = fixture(shifts=[(0, 0), (0, 0)])
    first = np.array(Image.open(BytesIO(images[0])))
    second = np.random.default_rng(5).integers(0, 255, first.shape, np.uint8)
    first[40:51, 52:63] = first[40:51, 40:51]
    second[40:51, 43:54] = first[40:51, 40:51]
    images = {0: png(first), 2: png(second)}
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    row = result["rows"][0]
    assert row["forward"]["reason"] is None and row["forward"]["offset_px"] == [3, 0]
    assert row["reason"] == "reverse_ambiguous_patch_match" and row["status"] == "unavailable"


def test_nonreturning_match_cannot_qualify(monkeypatch):
    evidence, images = fixture()
    calls = 0
    def mismatch(*args):
        nonlocal calls
        calls += 1
        return {"reason": None, "offset_px": [3, 0] if calls % 2 else [0, 0]}
    monkeypatch.setattr(support, "_match", mismatch)
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert all(row["reason"] == "roundtrip_failed" for row in result["rows"])
    assert all(row["roundtrip_error_px"] == 3 for row in result["rows"])


def test_native_clock_gap_is_unknown_not_nominally_retimed():
    evidence, images = fixture()
    evidence["frames"][1]["decoder_timestamp_s"] += 1
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert all(row["reason"] == "native_clock_gap" for row in result["rows"])
    assert result["rows"][0]["time_s"] == [0, 2 / 30]
    assert result["rows"][0]["native_time_s"] == [1.002, 2.002 + 2 / 30]
    assert result["work"]["png_decodes"] == 0


def test_unknown_native_stamp_keeps_all_slots_unavailable():
    evidence, images = fixture()
    evidence["frames"][0]["decoder_timestamp_s"] = None
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert result["slot_count"] == result["processed_slots"] == 12
    assert all(row["reason"] == "unknown_native_clock" for row in result["rows"])
    assert result["work"]["png_decodes"] == 0


def test_missing_native_field_fails_clearly_before_matching(monkeypatch):
    evidence, images = fixture()
    document, images, bindings_raw = inputs(evidence, images)
    del evidence["frames"][0]["decoder_timestamp_s"]
    document = raw(evidence)
    bindings = json.loads(bindings_raw)
    bindings["evidence_sha256"] = sha(document)
    monkeypatch.setattr(support, "_match", lambda *a: pytest.fail("Missing clock must precede matching"))
    with pytest.raises(ValueError, match="Native timestamp field is required"):
        support.diagnose_pose_pixel_support(document, images, raw(bindings))


def test_malformed_landmark_cannot_expand_report_beyond_its_bound():
    evidence, images = fixture()
    for index, _ in support.JOINTS:
        evidence["frames"][0]["landmarks"][index]["x"] = "x" * 10_000
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert all(row["reason"] == "malformed_original_landmark" for row in result["rows"])
    assert len(raw(result)) < support.CONTRACT["maximum_report_bytes"]
    assert result["work"]["png_decodes"] == 0


@pytest.mark.parametrize("mutation", ["image", "pose", "clock", "source", "duplicate_frame", "manifest"])
def test_changed_source_bindings_fail_before_matching(monkeypatch, mutation):
    evidence, images = fixture()
    document, images, binding_raw = inputs(evidence, images)
    binding = json.loads(binding_raw)
    if mutation == "image":
        images[0] += b"changed"
    elif mutation == "pose":
        document += b" "
    elif mutation == "clock":
        binding["frames"][1]["decoder_timestamp_s"] += .001
    elif mutation == "source":
        binding["source_sha256"] = "3" * 64
    elif mutation == "manifest":
        binding["source_manifest_sha256"] = None
    else:
        binding["frames"][1] = deepcopy(binding["frames"][0])
    monkeypatch.setattr(support, "_match", lambda *a: pytest.fail("Custody failure must precede matching"))
    with pytest.raises(ValueError):
        support.diagnose_pose_pixel_support(document, images, raw(binding))


@pytest.mark.parametrize("malformed", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}', b'[]'])
def test_strict_original_json_is_required(malformed):
    evidence, images = fixture()
    _, images, bindings = inputs(evidence, images)
    with pytest.raises(ValueError):
        support.diagnose_pose_pixel_support(malformed, images, bindings)


def test_refined_input_and_oversized_request_are_not_partial_successes():
    evidence, images = fixture()
    evidence["frames"][0]["image_refinement"] = {"method": "another"}
    with pytest.raises(ValueError, match="refinement"):
        support.diagnose_pose_pixel_support(*inputs(evidence, images))
    evidence, images = fixture(frames=(0, 2, 4, 6, 8), shifts=[(0, 0)] * 5)
    with pytest.raises(ValueError, match="two to four"):
        support.diagnose_pose_pixel_support(*inputs(evidence, images))


def test_deadline_preserves_successful_prefix_and_every_unprocessed_slot(monkeypatch):
    evidence, images = fixture()
    actual = support._compare
    calls = 0
    def stop(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("injected bounded interruption")
        return actual(*args)
    monkeypatch.setattr(support, "_compare", stop)
    result = support.diagnose_pose_pixel_support(*inputs(evidence, images))
    assert result["status"] == "partial_deadline" and result["processed_slots"] == 2
    assert len(result["rows"]) == 12
    assert all(row["status"] == "not_contradicted" for row in result["rows"][:2])
    assert all(row["reason"] == "deadline_unprocessed" for row in result["rows"][2:])
    assert result["status_counts"] == {"not_contradicted": 2, "unavailable": 10}


def test_final_deadline_cannot_publish_complete_result(monkeypatch):
    evidence, images = fixture()
    actual = support._bytes
    exhausted = False
    def encode(value):
        nonlocal exhausted
        result = actual(value)
        if value.get("profile") == support.PROFILE:
            exhausted = True
        return result
    monkeypatch.setattr(support, "_bytes", encode)
    monkeypatch.setattr(support.time, "monotonic", lambda: 100. if exhausted else 0.)
    with pytest.raises(TimeoutError):
        support.diagnose_pose_pixel_support(*inputs(evidence, images))


def test_output_is_deterministic_and_bounded_without_source_work():
    evidence, images = fixture(frames=(0, 2, 4, 6))
    args = inputs(evidence, images)
    first, second = (support.diagnose_pose_pixel_support(*args) for _ in range(2))
    first.pop("seconds"); second.pop("seconds")
    assert first == second
    assert len(raw(first)) < support.CONTRACT["maximum_report_bytes"]
    assert first["slot_count"] == 36 and first["work"] == {
        "model_calls": 0, "video_decodes": 0, "physics_steps": 0, "png_decodes": 4, "max_patch_matches": 72}
    assert first["source_provenance_status"] == "caller_bound_not_independently_verified"
