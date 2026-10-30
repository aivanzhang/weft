"""One source-qualified image proposal stays separate from admission and motion clearance."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import pytest

from weft import silhouette_pipeline as module
from weft.io import canonical_hash
from weft.shared_exercise_task import build_shared_exercise_task
from weft.shared_world import validate_shared_world
from weft.shared_world_packaging import read_package_file, write_package_files
from weft.source_equivalence import load_source_equivalence
from test_silhouette_geometry import fixture, raw, repin, SOURCE, TARGET, OTHERS
from test_source_equivalence import fixture as mapping_fixture


def seal(root, files):
    root.mkdir()
    write_package_files(root, files)
    write_package_files(root, {"manifest.json": raw({"files": {
        name: hashlib.sha256(data).hexdigest() for name, data in files.items()}})})


def prepared_inputs(tmp_path, *, reference_evidence=True, derivative_source=False, **fixture_options):
    scene, kwargs = fixture(**fixture_options)
    if derivative_source:
        equivalence = load_source_equivalence(mapping_fixture(tmp_path / "mapping"))
        scene["human_spec"]["source_sha256"] = equivalence.original_sha256
        seg, masks = json.loads(kwargs["segmentation_evidence"]), json.loads(kwargs["mask_document"])
        seg["source"].update(submitted_video_sha256=equivalence.derivative_sha256,
                            original_source_sha256=equivalence.original_sha256, width_px=2, height_px=2)
        for row in seg["observations"]:
            pair = equivalence.frames[row["frame"]]
            row["id"] = row["id"].replace(SOURCE, equivalence.derivative_sha256)
            row["original_source"] = {"sha256": equivalence.original_sha256, "frame": pair.source_frame,
                                      "time_s": pair.source_native_pts / 1000}
            if row["geometry"]:
                for key in ("box_xyxy_halfopen", "box_xywh", "center_px"):
                    row["geometry"][key] = [value / 320 for value in row["geometry"][key]]
        camera = scene["objects"][1]["provenance"]["camera_hypothesis"]
        camera["pixel_origin"] = [value / 320 for value in camera["pixel_origin"]]
        camera["focal_px"] /= 320
        camera["original_scale_at_anchor_model_units_per_pixel"] *= 320
        scene, kwargs = repin(scene, kwargs, seg, masks, source_equivalences=[equivalence])
    human = scene["human_spec"]
    human["reference"], human["root_reference"] = human["reference"][:2], human["root_reference"][:2]
    observations = json.loads(next(iter(kwargs["source_documents"].values())))["observations"]
    human["reference_evidence"] = []
    for frame, stamp in ((1, .033), (2, .067)):
        human["reference"][frame - 1]["time_s"] = human["root_reference"][frame - 1]["time_s"] = stamp
        row = next(row for row in observations if row["frame"] == frame and row["object_id"] == OTHERS[0])
        human["reference_evidence"].append({"time_s": stamp, "evidence_id": row["evidence_ids"][0]})
    if not reference_evidence:
        human.pop("reference_evidence")
    human["mobility"]["floor_fit"]["reference_time_s"] = .033
    scene["task"]["horizon_steps"] = 1
    scene = validate_shared_world(scene)
    task = build_shared_exercise_task(scene, root_assistance=True)
    source = tmp_path / "source"
    files = {"scene.json": raw(scene), "exercise-task.json": raw(task),
             **{scene["source_binding"]["evidence_documents"][sha]: data
                for sha, data in kwargs["source_documents"].items()}}
    for equivalence in kwargs.get("source_equivalences", ()):
        files.update({f"source-mappings/{equivalence.manifest_sha256}/{name}": equivalence.raw_bytes(name)
                      for name in equivalence.file_names})
    seal(source, files)
    masks_root = tmp_path / "masks"
    mask_files = {"masks/original-evidence.json": kwargs["segmentation_evidence"],
                  "masks/mask-evidence.json": kwargs["mask_document"],
                  "evidence/segmentation-evidence.json": kwargs["segmentation_evidence"],
                  **{"masks/" + name: data for name, data in kwargs["raster_documents"].items()}}
    seal(masks_root, mask_files)
    return scene, task, source, masks_root


def run(tmp_path, source, masks, **changes):
    return module.prepare_silhouette_world(source, masks, tmp_path / "out",
        **({"object_ids": ["marker_box"], "source_frame": 1, "conditional_exterior": True} | changes))


def document(path):
    return json.loads(path.read_bytes())


def assert_sealed(directory, *, complete=True):
    manifest = document(directory / "manifest.json")
    for name, expected in manifest["files"].items():
        read_package_file(directory, name, max_bytes=64 * 1024 * 1024, expected_sha256=expected)
    if complete:
        assert set(manifest["files"]) == {
            str(path.relative_to(directory)) for path in directory.rglob("*")
            if path.is_file() and path != directory / "manifest.json"}
    return manifest


def test_source_qualified_pipeline_preserves_all_stages_and_uses_one_context(tmp_path, monkeypatch):
    scene, task, source, masks = prepared_inputs(tmp_path)
    original = {str(path): path.read_bytes() for root in (source, masks) for path in root.rglob("*") if path.is_file()}
    calls = []
    prepare, propose = module.prepare_silhouette_evidence, module.propose_silhouette_geometry
    def tracked_prepare(*args, **kwargs):
        calls.append("prepare")
        return prepare(*args, **kwargs)
    def tracked_propose(context, object_id, **kwargs):
        calls.append((object_id, deepcopy(kwargs)))
        return propose(context, object_id, **kwargs)
    monkeypatch.setattr(module, "prepare_silhouette_evidence", tracked_prepare)
    monkeypatch.setattr(module, "propose_silhouette_geometry", tracked_propose)
    monkeypatch.setattr(mujoco, "mj_step", lambda *args, **kwargs: pytest.fail("The pipeline cannot simulate a rollout"))
    result = run(tmp_path, source, masks)
    out = tmp_path / "out"
    assert result["status"] == "completed", result
    assert result["outcome"] == "clearance_comparison_complete"
    assert result["candidate_directory"] == str(out / "geometry")
    assert result["report_path"] == str(out / "report.json")
    assert result["inspection_path"] == str(out / "shape-comparison.html")
    assert result["candidate_manifest_sha256"] == hashlib.sha256((out / "geometry/manifest.json").read_bytes()).hexdigest()
    assert calls[0] == "prepare" and len(calls) == 2
    assert calls[1][1]["occluder_ids"] == sorted(OTHERS)
    assert calls[1][1]["max_parts"] == 8 and calls[1][1]["conditional_exterior"] is True
    proposal = document(out / "objects/marker_box/result.json")
    assert proposal["status"] == "conditional_proposal"
    assert proposal["report"]["candidate"]["scores"]["conditional_clear_iou"] > proposal["report"]["baseline"]["scores"]["conditional_clear_iou"]
    assert len(proposal["report"]["all_source_slots"]) == 9
    inspection = document(out / "objects/marker_box/inspection-input.json")
    png_raw = (out / inspection["selected_png_path"]).read_bytes()
    assert hashlib.sha256(png_raw).hexdigest() == inspection["selected_png_sha256"]
    assert len(inspection["baseline_vertices_px"]) == 8
    html = (out / "shape-comparison.html").read_text()
    assert "Original box projection" in html and "Comparison computation complete" in html
    assert inspection["selected_png_sha256"] in html and "data:image/png;base64," in html
    assert "No camera/depth recovery" in html and "not a physical-validity verdict" in html
    clearance = document(out / "clearance-comparison.json")
    assert clearance["status"] == "complete", clearance["reasons"]
    assert clearance["source"]["status"] == "verified_source_binding"
    slots = [row for row in clearance["samples"] if row["kind"] == "source_slot"]
    assert [row["source_frame"] for row in slots] == [0, 1, 2]
    assert [row["reference_index"] for row in slots] == [None, 0, 1]
    assert slots[0]["status"] == "unavailable" and clearance["physics_steps"] == 0
    candidate = document(out / "geometry/scene.json")
    assert candidate["human_spec"] == scene["human_spec"] and candidate["task"] == scene["task"]
    next_task = document(out / "geometry/exercise-task.json")
    assert next_task | {"scene_sha256": task["scene_sha256"]} == task
    assert next_task["scene_sha256"] == canonical_hash(candidate)
    assert all(Path(path).read_bytes() == data for path, data in original.items())
    assert_sealed(out)


def test_derivative_source_pipeline_verifies_bundle_and_preserves_both_native_clocks(tmp_path, monkeypatch):
    scene, _, source, masks = prepared_inputs(tmp_path, derivative_source=True)
    alias = scene["source_binding"]["source_equivalences"][0]
    original = {name: (source / name).read_bytes() for name in document(source / "manifest.json")["files"]}
    monkeypatch.setattr(mujoco, "mj_step", lambda *args, **kwargs: pytest.fail("No physics rollout"))
    result = run(tmp_path, source, masks)
    assert result["status"] == "completed" and result["outcome"] == "clearance_comparison_complete", result
    out = tmp_path / "out"
    proposal = document(out / "objects/marker_box/result.json")["report"]
    assert proposal["input_hashes"]["source_mapping_manifests"] == [alias["bundle_manifest_sha256"]]
    assert proposal["submitted_time_s"] == .033
    assert proposal["original_source"] == {"sha256": alias["original_source_sha256"], "frame": 1, "time_s": .036}
    assert alias["original_source_sha256"] != alias["derivative_source_sha256"]
    copied = load_source_equivalence(out / "geometry-input" / alias["bundle_path"])
    assert copied.source_clock.native_minus_decoder_pts == copied.derivative_clock.native_minus_decoder_pts == 3
    assert copied.frames[1].source_native_pts == copied.frames[1].derivative_native_pts == 36
    assert copied.frames[1].derivative_decoder_pts == 33
    assert all((out / "geometry-input" / name).read_bytes() == data for name, data in original.items())
    assert all((source / name).read_bytes() == data for name, data in original.items())
    comparison = document(out / "clearance-comparison.json")
    assert comparison["source"]["canonical_source_sha256"] == alias["original_source_sha256"]
    assert comparison["source"]["document_source_sha256"] == alias["derivative_source_sha256"]
    slots = [row for row in comparison["samples"] if row["kind"] == "source_slot"]
    assert [row["source_time_s"] for row in slots] == [0., .033, .067]
    assert [row["reference_index"] for row in slots] == [None, 0, 1]
    html = (out / "shape-comparison.html").read_text()
    assert "0.036 s" in html and "0.033 s" in html
    assert_sealed(out)


def test_no_improvement_keeps_complete_candidate_without_admission(tmp_path, monkeypatch):
    _, _, source, masks = prepared_inputs(tmp_path, occlusion=True)
    monkeypatch.setattr(module, "prepare_shared_world_geometry", lambda *a, **k: pytest.fail("No qualified replacement"))
    result = run(tmp_path, source, masks)
    assert result["status"] == "completed" and result["outcome"] == "no_qualified_improvement", result
    assert result["candidate_directory"] is None and result["candidate_manifest_sha256"] is None
    saved = document(tmp_path / "out/objects/marker_box/result.json")
    assert saved["geometry_config"] is None and saved["candidate_geometry"]["parts"]
    assert saved["report"]["uncertain_rows"]
    assert not (tmp_path / "out/clearance-comparison.json").exists()
    assert "Not attached to this report" in (tmp_path / "out/shape-comparison.html").read_text()
    assert_sealed(tmp_path / "out")


@pytest.mark.parametrize("change", [{"conditional_exterior": False}, {"object_ids": ["marker_box", "marker_box"]},
                                   {"max_seconds": 61}, {"source_frame": True}])
def test_invalid_work_options_do_not_create_outputs(tmp_path, change):
    with pytest.raises(ValueError):
        run(tmp_path, tmp_path / "source", tmp_path / "masks", **change)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("damage", ["png", "manifest_status", "source_status"])
def test_bad_sealed_inputs_fail_before_any_fit_and_leave_receipt(tmp_path, monkeypatch, damage):
    _, _, source, masks = prepared_inputs(tmp_path)
    if damage == "png":
        next((masks / "masks/masks").glob("*.png")).write_bytes(b"changed")
    elif damage == "manifest_status":
        value = document(masks / "manifest.json")
        value["status"] = "running"
        (masks / "manifest.json").write_bytes(raw(value))
    else:
        (source / "report.json").write_bytes(raw({"status": "failed"}))
        value = document(source / "manifest.json")
        value["files"]["report.json"] = hashlib.sha256((source / "report.json").read_bytes()).hexdigest()
        (source / "manifest.json").write_bytes(raw(value))
    monkeypatch.setattr(module, "prepare_silhouette_evidence", lambda *a, **k: pytest.fail("Bad custody cannot reach fitting"))
    result = run(tmp_path, source, masks)
    assert result["status"] == "failed" and result["candidate_directory"] is None
    manifest = assert_sealed(tmp_path / "out")
    assert manifest["custody"] == "failure_receipt_only" and set(manifest["files"]) == {"failure.json"}


def test_one_failed_object_prevents_subset_admission_and_preserves_first_proposal(tmp_path, monkeypatch):
    _, _, source, masks = prepared_inputs(tmp_path)
    monkeypatch.setattr(module, "prepare_shared_world_geometry", lambda *a, **k: pytest.fail("Do not select a successful subset"))
    result = run(tmp_path, source, masks, object_ids=["marker_box", "ball"])
    assert result["status"] == "partial" and result["outcome"] == "proposal_failures", result
    out = tmp_path / "out"
    assert document(out / "objects/marker_box/result.json")["status"] == "conditional_proposal"
    assert document(out / "objects/ball/failure.json")["status"] == "failed"
    assert not (out / "geometry-config.json").exists()
    assert_sealed(out)


def test_full_stage_reservation_precedes_fitting_and_preserves_bounded_inputs(tmp_path, monkeypatch):
    _, _, source, masks = prepared_inputs(tmp_path)
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 4 * 1024 * 1024)
    monkeypatch.setattr(module, "prepare_silhouette_evidence", lambda *a, **k: pytest.fail("Reserve first"))
    result = run(tmp_path, source, masks)
    assert result["status"] == "failed" and "reserve" in result["error"]["message"], result
    out = tmp_path / "out"
    assert (out / "geometry-input/scene.json").exists()
    assert sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) < module.MAX_OUTPUT_BYTES
    assert_sealed(out, complete=False)


def test_original_input_changed_during_fit_cannot_publish_a_ready_pipeline(tmp_path, monkeypatch):
    _, _, source, masks = prepared_inputs(tmp_path)
    propose = module.propose_silhouette_geometry
    def changed(*args, **kwargs):
        result = propose(*args, **kwargs)
        (source / "scene.json").write_bytes(b"changed")
        return result
    monkeypatch.setattr(module, "propose_silhouette_geometry", changed)
    result = run(tmp_path, source, masks)
    assert result["status"] == "failed" and result["candidate_directory"] is None, result
    out = tmp_path / "out"
    assert (out / "geometry/scene.json").exists() and (out / "clearance-comparison.json").exists()
    assert not (out / "report.json").exists()
    assert assert_sealed(out, complete=False)["status"] == "failed"


def test_explicit_task_cannot_change_existing_sealed_task(tmp_path, monkeypatch):
    scene, _, source, masks = prepared_inputs(tmp_path)
    other = build_shared_exercise_task(scene, root_assistance=False)
    extra = tmp_path / "other-task"
    seal(extra, {"exercise-task.json": raw(other)})
    monkeypatch.setattr(module, "prepare_silhouette_evidence", lambda *a, **k: pytest.fail("Freeze original task before fitting"))
    result = run(tmp_path, source, masks, task_path=extra / "exercise-task.json")
    assert result["status"] == "failed" and "task definition" in result["error"]["message"]
    assert_sealed(tmp_path / "out")


def test_equivalent_explicit_task_keeps_both_original_byte_identities(tmp_path):
    _, task, source, masks = prepared_inputs(tmp_path, occlusion=True)
    supplied = json.dumps(task, indent=3).encode()
    original = (source / "exercise-task.json").read_bytes()
    assert supplied != original
    extra = tmp_path / "other-task"
    seal(extra, {"exercise-task.json": supplied})
    result = run(tmp_path, source, masks, task_path=extra / "exercise-task.json")
    assert result["status"] == "completed", result
    out = tmp_path / "out"
    assert (out / "geometry-input/exercise-task.json").read_bytes() == original
    assert (out / "inputs/explicit-exercise-task.json").read_bytes() == supplied
    settings = document(out / "settings.json")
    assert settings["task_file_sha256"] == hashlib.sha256(original).hexdigest()
    assert settings["explicit_task_file_sha256"] == hashlib.sha256(supplied).hexdigest()
    assert_sealed(out)


def test_output_cannot_be_nested_in_explicit_task_package(tmp_path):
    _, task, source, masks = prepared_inputs(tmp_path)
    extra = tmp_path / "other-task"
    seal(extra, {"exercise-task.json": raw(task)})
    with pytest.raises(ValueError, match="outside every immutable input"):
        module.prepare_silhouette_world(source, masks, extra / "out", task_path=extra / "exercise-task.json",
            object_ids=["marker_box"], source_frame=1, conditional_exterior=True)
    assert not (extra / "out").exists()


def test_deadline_after_proposal_preserves_result_without_admission_or_retry(tmp_path, monkeypatch):
    _, _, source, masks = prepared_inputs(tmp_path)
    remaining = module._remaining
    def expired(deadline, ceiling=30):
        if (tmp_path / "out/objects/marker_box/result.json").exists() and ceiling == 30:
            # Failure receipts have their own small preservation grace period.
            if not (tmp_path / "out/geometry-config.json").exists():
                return remaining(deadline, ceiling)
            if deadline > started[0] + 10:
                raise TimeoutError("Injected study deadline after the one proposal")
        return remaining(deadline, ceiling)
    started = [module.time.monotonic()]
    monkeypatch.setattr(module, "_remaining", expired)
    monkeypatch.setattr(module, "prepare_shared_world_geometry", lambda *a, **k: pytest.fail("No admission after deadline"))
    result = run(tmp_path, source, masks)
    assert result["status"] == "failed" and result["error"]["type"] == "TimeoutError", result
    out = tmp_path / "out"
    assert document(out / "objects/marker_box/result.json")["status"] == "conditional_proposal"
    assert not (out / "geometry").exists()
    assert_sealed(out, complete=False)


def test_late_manifest_failure_preserves_old_receipts_without_active_completed_seal(tmp_path, monkeypatch):
    _, _, source, masks = prepared_inputs(tmp_path, occlusion=True)
    write = module._write
    def failed(out, files):
        write(out, files)
        if set(files) == {"manifest.json"}:
            raise OSError("Injected late publication failure")
    monkeypatch.setattr(module, "_write", failed)
    result = run(tmp_path, source, masks)
    assert result["status"] == "failed" and result["candidate_directory"] is None
    out = tmp_path / "out"
    assert document(out / "manifest.publication-failed.json")["status"] == "completed"
    assert document(out / "report.publication-failed.json")["outcome"] == "no_qualified_improvement"
    assert not (out / "report.json").exists()
    manifest = assert_sealed(out, complete=False)
    assert manifest["status"] == "failed" and "manifest.publication-failed.json" in manifest["files"]


def test_unsupported_source_reference_keeps_admission_and_overlap_result_distinct(tmp_path):
    _, _, source, masks = prepared_inputs(tmp_path, reference_evidence=False)
    result = run(tmp_path, source, masks)
    assert result["status"] == "partial" and result["outcome"] == "clearance_comparison_unsupported_input", result
    out = tmp_path / "out"
    assert result["candidate_directory"] == str(out / "geometry")
    comparison = document(out / "clearance-comparison.json")
    assert comparison["status"] == "unsupported_input" and comparison["summary"]["scheduled"] == 0
    assert comparison["reasons"] and comparison["physics_steps"] == 0
    assert "Comparison input unsupported" in (out / "shape-comparison.html").read_text()
    assert_sealed(out)


def test_incompatible_source_layout_rejects_before_reading_hidden_payload(tmp_path, monkeypatch):
    _, _, source, masks = prepared_inputs(tmp_path)
    manifest = document(source / "manifest.json")
    manifest["files"][".unrelated"] = "a" * 64
    (source / "manifest.json").write_bytes(raw(manifest))
    read = module.read_package_file
    def guarded(root, name, **kwargs):
        assert name != ".unrelated", "Do not read hidden input payloads"
        return read(root, name, **kwargs)
    monkeypatch.setattr(module, "read_package_file", guarded)
    result = run(tmp_path, source, masks)
    assert result["status"] == "failed" and "hidden" in result["error"]["message"]
    assert_sealed(tmp_path / "out")


def test_manifest_metadata_is_bounded_before_publication(tmp_path, monkeypatch):
    write_package_files(tmp_path, {"first.json": b"{}", "second.json": b"{}"})
    monkeypatch.setattr(module, "METADATA_RESERVE", 128)
    with pytest.raises(ValueError, match="metadata reservation"):
        module._seal(tmp_path, "completed", module.time.monotonic() + 10)
    assert not (tmp_path / "manifest.json").exists()
    assert (tmp_path / "first.json").read_bytes() == b"{}"


def test_public_cli_bad_input_exits_one_and_prints_preserved_failure(tmp_path, monkeypatch, capsys):
    from weft.cli import main

    _, _, source, masks = prepared_inputs(tmp_path)
    next((masks / "masks/masks").glob("*.png")).write_bytes(b"changed before invocation")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["weft", "shared-world-shape", "--run", str(source),
        "--masks-run", str(masks), "--object-id", "marker_box", "--source-frame", "1",
        "--conditional-exterior", "--out", str(out)])
    monkeypatch.setattr(module, "prepare_silhouette_evidence", lambda *a, **k: pytest.fail("Reject altered bytes before fit"))
    with pytest.raises(SystemExit) as exited:
        main()
    assert exited.value.code == 1
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "failed" and result["report_path"] == str(out / "failure.json")
    assert result["report_path"] in captured.err
    saved = document(out / "failure.json")
    assert saved["status"] == "failed" and saved["stage"] == "inputs" and saved["objects"] == []
    assert_sealed(out)
