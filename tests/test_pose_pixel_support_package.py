from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import pytest

from weft import pose_pixel_support_package as package
from test_pose_pixel_support import fixture as image_fixture


def raw(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else raw(value))


def seal(root, profile, status):
    files = {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in root.rglob("*")
             if p.is_file() and p.name not in {"manifest.json", "failure.json"}}
    put(root, "manifest.json", {"schema_version": 1, "profile": profile, "status": status, "files": files})


def inputs(root, *, legacy=False, video=b"\x1aE\xdf\xa3analytical fixture; no decoding is performed"):
    pose, pixels = root / "pose", root / "pixels"
    evidence, images = image_fixture(frames=(0, 2, 4, 6))
    source = sha(video)
    evidence.update(profile="exercise-pose-v1", schema_version=1, source_sha256=source,
                    model={"name": "Analytical image fixture", "network_inference": False},
                    fps=30., frame_count=12, decoded_frames=12, quality={"errors": []})
    for row in evidence["frames"]:
        row["id"] = f"{source}:frame:{row['frame']}"
    put(pose, "evidence.json", evidence)
    put(pose, "pose-landmarks.json", {"frames": evidence["frames"], "source_sha256": source})
    put(pose, "model.json", evidence["model"])
    put(pose, "identity-audit.json", {"mode": "raw", "source_sha256": source,
        "raw_artifact": "pose-landmarks.json", "raw_artifact_sha256": sha((pose / "pose-landmarks.json").read_bytes()),
        "repaired_evidence_ids": []})
    put(pose, "source-clock.json", {"source_sha256": source, "decoded_frames": 12,
        "samples": [{key: row[key] for key in ("frame", "time_s", "decoder_timestamp_s")} for row in evidence["frames"]]})
    put(pose, "source.webm", video)
    put(pose, "body-source.json", {"profile": "exercise-body-source-v1", "status": "construction_only",
        "simulation_ready": False, "source_sha256": source, "source_artifact": "source.webm",
        "pose_options": {"pose_identity": "raw"}, "artifacts": {
            name: sha((pose / name).read_bytes()) for name in package.POSE_FILES if name != "body-source.json"}})
    seal(pose, "exercise-body-source-v1", "construction_only")
    profile = package.LEGACY_EXTRACTION_PROFILE if legacy else package.EXTRACTION_PROFILE
    rows = []
    for row in evidence["frames"]:
        image = images[row["frame"]]
        filename = f"frame-{row['frame']:04d}.png"
        put(pixels, filename, image)
        rgb = np.array(Image.open(BytesIO(image)).convert("RGB"), dtype=np.uint8)
        rows.append({"frame": row["frame"], "file": filename, "sha256": sha(image),
            "decoded_rgb_sha256": sha(rgb.tobytes()), "source_sha256": source,
            "width": evidence["width"], "height": evidence["height"],
            "nominal_frame_over_fps_s" if legacy else "time_s": row["time_s"],
            "decoder_timestamp_s": row["decoder_timestamp_s"]})
    put(pixels, "report.json", {"schema_version": 1, "profile": profile, "status": "pixels_extracted",
        "source_sha256": source, "output_resampling": False, "automatic_repair": False,
        "api_calls": 0, "inference_calls": 0, "physics_steps": 0, "rows": rows})
    seal(pixels, profile, "pixels_extracted")
    return pose, pixels


def load(root, pose):
    return package.load_pose_pixel_support(root, evidence_document=(pose / "evidence.json").read_bytes(),
        source_manifest_raw=(pose / "manifest.json").read_bytes(), deadline=time.monotonic() + 5)


@pytest.fixture(scope="module")
def complete(tmp_path_factory):
    root = tmp_path_factory.mktemp("pixel-companion")
    pose, pixels = inputs(root)
    out = root / "companion"
    receipt = package.prepare_pose_pixel_support(pose, pixels, out)
    return pose, pixels, out, receipt


def test_complete_companion_preserves_exact_consumed_bytes_and_saved_only_readback(complete, monkeypatch):
    pose, pixels, out, receipt = complete
    assert receipt["status"] == "complete" and receipt["slot_count"] == 36
    assert receipt["status_counts"] == {"not_contradicted": 36}
    assert receipt["bytes"] <= package.MAX_BYTES and receipt["files"] <= package.MAX_FILES
    for name in package.POSE_FILES:
        assert (out / "inputs" / name).read_bytes() == (pose / name).read_bytes()
    assert not (out / "inputs/source.webm").exists()
    assert not (out / "inputs/pose-landmarks.json").exists()
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(sha((out / name).read_bytes()) == digest for name, digest in manifest["files"].items())
    monkeypatch.setattr(package.diagnostic, "diagnose_pose_pixel_support", lambda *a, **k: pytest.fail("No diagnostic rerun"))
    monkeypatch.setattr(package.diagnostic.Image, "open", lambda *a, **k: pytest.fail("No PNG decoding in saved readback"))
    saved = load(out, pose)
    try:
        assert saved.summary["slot_count"] == 36
        assert saved.summary["package_provenance"]["extraction_adapter"] == "native_schema"
        assert "not independently verified" in saved.summary["package_provenance"]["scope"]
        saved.recheck()
    finally:
        saved.close()


def test_legacy_extraction_has_explicit_adapter_without_source_specific_frames(tmp_path):
    pose, pixels = inputs(tmp_path, legacy=True)
    out = tmp_path / "new"
    package.prepare_pose_pixel_support(pose, pixels, out)
    saved = load(out, pose)
    try:
        assert saved.summary["package_provenance"]["extraction_adapter"] == "legacy_receipt_compatibility"
        assert saved.summary["sampled_image_frames"] == [0, 2, 4, 6]
    finally:
        saved.close()


def clone(complete, tmp_path):
    pose, pixels, existing, receipt = complete
    out = tmp_path / "changed"
    for p in existing.rglob("*"):
        if p.is_file():
            put(out, p.relative_to(existing).as_posix(), p.read_bytes())
    return pose, out


@pytest.mark.parametrize("mutation", ["residual", "roundtrip", "offset", "confidence", "rgb", "implementation", "counts", "slots"])
def test_resealed_internally_inconsistent_result_is_rejected(complete, tmp_path, mutation):
    pose, out = clone(complete, tmp_path)
    result = json.loads((out / "report.json").read_bytes())
    row = result["rows"][0]
    if mutation == "residual":
        row.update(status="contradicted", reason="local_patch_motion_disagrees", model_patch_residual_px=12.)
        result["status_counts"] = {"contradicted": 1, "not_contradicted": 35}
    elif mutation == "roundtrip":
        row["roundtrip_error_px"] = .5
    elif mutation == "offset":
        row["forward"]["offset_px"][0] = 7
    elif mutation == "confidence":
        row["original_landmarks"][0]["visibility"] = .1
    elif mutation == "rgb":
        result["decoded_rgb_sha256"].pop("0")
        result["work"]["png_decodes"] = 3
    elif mutation == "implementation":
        doc = json.loads((out / "implementation.json").read_bytes());doc["files"] = {}
        put(out, "implementation.json", doc)
        result["package_provenance"]["implementation_sha256"] = sha((out / "implementation.json").read_bytes())
    elif mutation == "counts":
        result["status_counts"] = {"not_contradicted": 35}
    else:
        result["rows"].pop()
    put(out, "report.json", result)
    seal(out, package.PROFILE, "complete")
    with pytest.raises(ValueError):
        load(out, pose)


@pytest.mark.parametrize("mutation", ["clock", "source", "resampled", "model", "failure", "png_hash", "schema", "boolean_clock"])
def test_source_preflight_rejects_before_new_output(tmp_path, mutation):
    pose, pixels = inputs(tmp_path)
    extraction = json.loads((pixels / "report.json").read_bytes())
    if mutation == "clock":
        extraction["rows"][0]["decoder_timestamp_s"] += .001
    elif mutation == "source":
        extraction["source_sha256"] = "3" * 64
    elif mutation == "resampled":
        extraction["output_resampling"] = True
    elif mutation == "png_hash":
        extraction["rows"][0]["sha256"] = "4" * 64
    elif mutation == "schema":
        extraction["schema_version"] = 2
    elif mutation == "boolean_clock":
        extraction["rows"][0]["time_s"] = False
    elif mutation == "model":
        put(pose, "model.json", {"name": "changed"})
        seal(pose, "exercise-body-source-v1", "construction_only")
    else:
        put(pose, "failure.json", {"status": "failed"})
    put(pixels, "report.json", extraction)
    seal(pixels, package.EXTRACTION_PROFILE, "pixels_extracted")
    out = tmp_path / "new"
    with pytest.raises(ValueError):
        package.prepare_pose_pixel_support(pose, pixels, out)
    assert not out.exists()


def test_output_reservation_precedes_creation(tmp_path, monkeypatch):
    pose, pixels = inputs(tmp_path)
    monkeypatch.setattr(package, "RESERVE", package.MAX_BYTES)
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="reservation"):
        package.prepare_pose_pixel_support(pose, pixels, out)
    assert not out.exists()


def test_algorithm_failure_preserves_inputs_and_authoritative_failure(tmp_path, monkeypatch):
    pose, pixels = inputs(tmp_path)
    def fail(*args, **kwargs):
        raise TimeoutError("injected time limit")
    monkeypatch.setattr(package.diagnostic, "diagnose_pose_pixel_support", fail)
    out = tmp_path / "new"
    with pytest.raises(TimeoutError):
        package.prepare_pose_pixel_support(pose, pixels, out)
    assert (out / "inputs/evidence.json").read_bytes() == (pose / "evidence.json").read_bytes()
    assert json.loads((out / "failure.json").read_bytes())["status"] == "failed"
    assert not (out / "manifest.json").exists()
    assert sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) <= package.MAX_BYTES


def test_late_input_mutation_rejects_with_failure_prefix(tmp_path, monkeypatch):
    pose, pixels = inputs(tmp_path)
    actual = package.diagnostic.diagnose_pose_pixel_support
    def mutate(*args, **kwargs):
        result = actual(*args, **kwargs)
        put(pixels, "failure.json", {"status": "failed"})
        return result
    monkeypatch.setattr(package.diagnostic, "diagnose_pose_pixel_support", mutate)
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="failure marker"):
        package.prepare_pose_pixel_support(pose, pixels, out)
    assert (out / "failure.json").exists() and not (out / "manifest.json").exists()


def test_partial_deadline_keeps_all_slots_and_is_not_complete(tmp_path, monkeypatch):
    pose, pixels = inputs(tmp_path)
    actual, calls = package.diagnostic._compare, 0
    def stop(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise TimeoutError("injected bounded interruption")
        return actual(*args)
    monkeypatch.setattr(package.diagnostic, "_compare", stop)
    out = tmp_path / "new"
    receipt = package.prepare_pose_pixel_support(pose, pixels, out)
    assert receipt["status"] == "partial_deadline" and receipt["slot_count"] == 36
    saved = load(out, pose)
    try:
        assert saved.summary["processed_slots"] == 2
        assert saved.summary["status_counts"] == {"not_contradicted": 2, "unavailable": 34}
    finally:
        saved.close()


@pytest.mark.parametrize("extra", ["failure.json", "unlisted.txt", "empty"])
def test_terminal_or_unlisted_companion_members_block_loading(complete, tmp_path, extra):
    pose, out = clone(complete, tmp_path)
    if extra == "empty":
        (out / extra).mkdir()
    else:
        put(out, extra, b"unlisted")
    with pytest.raises(ValueError):
        load(out, pose)


def test_recheck_detects_new_unlisted_companion_member(complete, tmp_path):
    pose, out = clone(complete, tmp_path)
    saved = load(out, pose)
    try:
        put(out, "late.txt", b"unlisted")
        with pytest.raises(ValueError, match="inventory changed"):
            saved.recheck()
    finally:
        saved.close()


def test_source_confidence_cannot_be_replaced_by_recorded_match_confidence(complete):
    pose, _, out, _ = complete
    evidence = json.loads((pose / "evidence.json").read_bytes())
    row = json.loads((out / "report.json").read_bytes())["rows"][0]
    first, second = deepcopy(evidence["frames"][:2])
    first["landmarks"][11]["visibility"] = .1
    with pytest.raises(ValueError, match="original confidence"):
        package._classification_check(row, first, second, evidence)


def test_oversized_saved_report_is_not_admitted(complete, tmp_path):
    pose, out = clone(complete, tmp_path)
    report = json.loads((out / "report.json").read_bytes())
    report["extra"] = "x" * package.diagnostic.CONTRACT["maximum_report_bytes"]
    put(out, "report.json", report)
    seal(out, package.PROFILE, "complete")
    with pytest.raises(ValueError, match="metadata bound"):
        load(out, pose)
