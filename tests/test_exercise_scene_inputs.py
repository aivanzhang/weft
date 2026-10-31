"""Raw scene admission rejects source mistakes before any pose or physics work."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json
import time

import pytest

from test_source_equivalence import fixture as mapping_fixture
from weft import exercise_scene_inputs as module
from weft.equipment_proposal import CONFIG_PROFILE
from weft.observations import COORDINATES, PROFILE


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.write_bytes((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def seal(root, **extra):
    write(root / "manifest.json", {"schema_version": 1, **extra, "files": {
        path.relative_to(root).as_posix(): sha(path.read_bytes()) for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"}})


def observations(source_sha, *, count=63, times=None, dimensions=(640, 480)):
    claim = {"value": None, "status": "unknown", "evidence_ids": []}
    native = times is not None
    times = times if native else [frame / 30 for frame in range(count)]
    source = {"sha256": source_sha, "width_px": dimensions[0], "height_px": dimensions[1],
              "metadata_frames": count, "decoded_frames": count}
    source.update({"frame_times_s": times} if native else {"fps": 30.})
    document = {"schema_version": 3 if native else 1, "profile": PROFILE, "source": source,
        "time": {"unit": "seconds", "origin": "submitted_video_start" if native else "video_start",
                 "basis": "decoder_presentation_timestamps" if native else "frame_index_divided_by_fps",
                 "status": "observed" if native else "assumed"},
        "image": {"width_px": dimensions[0], "height_px": dimensions[1]}, "coordinates": deepcopy(COORDINATES),
        "camera": {"motion": deepcopy(claim), "calibration": deepcopy(claim), "screen": {"status": "not_checked"}},
        "objects": [{"id": "foam-pad", "label": "Analytical foam pad",
                     "properties": {"kind": deepcopy(claim), "role": deepcopy(claim)},
                     "unknowns": ["physical shape", "scale", "mass"]}],
        "evidence": [], "observations": [], "relationships": [],
        "provenance": {"scope": "Analytical fixture, not measured physical video"}, "limits": []}
    for frame in range(count):
        identifier = f"{source_sha}:frame:{frame}:object:foam-pad"
        detected = frame != 0
        document["evidence"].append({"id": identifier, "frame": frame, "time_s": times[frame],
            "object_id": "foam-pad", "kind": "model_image_box", "status": "inferred" if detected else "unknown",
            "payload": {"id": identifier, "frame": frame, "time_s": times[frame]}})
        document["observations"].append({"id": f"{source_sha}:observation:{frame}:foam-pad", "frame": frame,
            "time_s": times[frame], "object_id": "foam-pad", "visibility": "detected" if detected else "not_detected",
            "partition": "construction", "evidence_ids": [identifier], "measurements": {
                "box": {"value": [.1, .1, 1., 1.], "coordinate_system": "image", "unit": "pixels", "status": "inferred"}}
                if detected else {}})
    return document


def inputs(tmp_path, monkeypatch, *, native=False):
    video = tmp_path / "source.webm"
    video.write_bytes(b"Analytical raw source; not a decodable or physically measured video.\n")
    model = tmp_path / "pose.task"
    model.write_bytes(b"Analytical model identity fixture")
    provenance = {"name": "Test-only model", "model_version": "fixture", "sha256": sha(model.read_bytes()),
                  "bytes": model.stat().st_size, "network_inference": False}
    monkeypatch.setattr(module.exercise_vision, "model_provenance", lambda path: deepcopy(provenance))
    evidence = tmp_path / "equipment"
    evidence.mkdir()
    document = observations(sha(video.read_bytes()), times=[frame / 30 + .1 for frame in range(63)] if native else None)
    write(evidence / "scene-observations.json", document)
    write(evidence / "report.json", {"status": "evidence_only", "physics_ready": False})
    seal(evidence)
    config = tmp_path / "config.json"
    write(config, {"schema_version": 1, "profile": CONFIG_PROFILE, "source_sha256": sha(video.read_bytes()),
        "camera": {"mode": "fixed_pelvis_perspective_prior", "distance_model_units": 2.},
        "geometry_prior": "floor_box_depth_equals_width", "equipment": [{"object_id": "foam-pad", "source_frame": 2}],
        "target_object_id": "foam-pad"})

    def forbidden(*_args, **_kwargs):
        pytest.fail("Input preflight cannot decode, infer, create physics or prepare a body")

    monkeypatch.setattr(module.exercise_vision.cv2, "VideoCapture", forbidden)
    monkeypatch.setattr(module.exercise_vision, "_detector", forbidden)
    monkeypatch.setattr(module.exercise_body_source, "prepare_exercise_body_source", forbidden)
    return video, evidence, config, model


def preflight(paths, **kwargs):
    video, evidence, config, model = paths
    return module.preflight_exercise_scene_inputs(video, evidence, config, pose_model=model, **kwargs)


def edit_equipment(paths, mutate):
    path = paths[1] / "scene-observations.json"
    document = json.loads(path.read_bytes())
    mutate(document)
    write(path, document)
    seal(paths[1])


@pytest.mark.parametrize("native", [False, True])
def test_read_only_preflight_keeps_exact_source_frames_clocks_and_default_pose_options(tmp_path, monkeypatch, native):
    paths = inputs(tmp_path, monkeypatch, native=native)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    result = preflight(paths)
    receipt = result.receipt()
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert receipt["status"] == "input_preflight_passed"
    assert receipt["pose_options"] == {"pose_mode": "video", "pose_identity": "raw"}
    assert receipt["source_note"] == module.SOURCE_NOTE
    assert receipt["selected_equipment"][0]["source_frame"] == 2
    assert receipt["selected_equipment"][0]["time_s"] == 2 / 30 + (.1 if native else 0)
    assert receipt["equipment_source_slots"] == 63
    assert receipt["work"] == {"video_decodes": 0, "inference_calls": 0, "physics_steps": 0, "api_calls": 0}
    assert receipt["equipment_document_sha256"] == sha(before[paths[1] / "scene-observations.json"])
    assert set(receipt["implementation_sha256"]) == set(module.IMPLEMENTATION_FILES)
    receipt["selected_equipment"].clear()
    assert len(result.receipt()["selected_equipment"]) == 1
    with pytest.raises(FrozenInstanceError):
        result.pose_identity = "temporal"
    with pytest.raises(TypeError, match="preflight"):
        module.RawExerciseInputs()


@pytest.mark.parametrize("kind", ["source", "equipment_source", "missing_id", "missing_frame", "audit", "generation",
                                 "development_validation", "not_detected", "authored_box", "roi_box", "zero_box"])
def test_invalid_or_withheld_equipment_never_reaches_inference(tmp_path, monkeypatch, kind):
    paths = inputs(tmp_path, monkeypatch)
    if kind == "source":
        paths[0].write_bytes(b"Different source")
    elif kind in {"missing_id", "missing_frame"}:
        config = json.loads(paths[2].read_bytes())
        if kind == "missing_id":
            config["equipment"][0]["object_id"] = config["target_object_id"] = "unobserved-weight"
        else:
            config["equipment"][0]["source_frame"] = 100
        write(paths[2], config)
    else:
        def change(document):
            row = document["observations"][2]
            if kind == "equipment_source":
                old = document["source"]["sha256"]
                replacement = "b" * 64
                updated = json.loads(json.dumps(document).replace(old, replacement))
                document.clear(); document.update(updated)
            elif kind in {"audit", "generation", "development_validation"}:
                row["partition"] = kind
            elif kind == "not_detected":
                row["visibility"] = "not_detected"; row["measurements"] = {}
            elif kind == "authored_box":
                row["measurements"]["box"]["status"] = "assumed"
            elif kind == "roi_box":
                document["evidence"][2]["payload"]["geometry_status"] = "authored_initial_ROI_translated_by_appearance_match"
            else:
                row["measurements"]["box"]["value"][2] = 0
        edit_equipment(paths, change)
    with pytest.raises(ValueError):
        preflight(paths)


@pytest.mark.parametrize("kind", ["manifest_failed", "report_failed", "failure_marker", "unsealed", "changed_sealed",
                                 "conflicting_documents", "partial_interval", "too_many_frames"])
def test_failed_incomplete_or_unsealed_equipment_is_not_admitted(tmp_path, monkeypatch, kind):
    paths = inputs(tmp_path, monkeypatch)
    root = paths[1]
    if kind == "manifest_failed":
        seal(root, status="failed")
    elif kind == "report_failed":
        write(root / "report.json", {"status": "failed"}); seal(root)
    elif kind == "failure_marker":
        write(root / "failure.json", {"status": "failed"})
    elif kind == "unsealed":
        (root / "manifest.json").unlink()
    elif kind == "changed_sealed":
        (root / "scene-observations.json").write_bytes(b"{}")
    elif kind == "conflicting_documents":
        (root / "evidence").mkdir()
        write(root / "evidence/scene-observations.json", {})
        seal(root)
    elif kind == "partial_interval":
        edit_equipment(paths, lambda doc: doc["source"].update(source_complete=False))
    else:
        write(root / "scene-observations.json", observations(sha(paths[0].read_bytes()), count=1801))
        seal(root)
    with pytest.raises((ValueError, OSError)):
        preflight(paths)


@pytest.mark.parametrize("field,value", [("pose_mode", "automatic"), ("pose_identity", "auto"),
    ("source_note", "x" * 4001), ("max_seconds", 31), ("max_seconds", True), ("deadline", float("nan"))])
def test_invalid_explicit_options_fail_without_work(tmp_path, monkeypatch, field, value):
    paths = inputs(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        preflight(paths, **{field: value})


@pytest.mark.parametrize("kind", ["video_bytes", "aggregate_bytes", "file_count", "manifest_count", "receipt_bytes", "deadline"])
def test_preflight_resource_bounds(tmp_path, monkeypatch, kind):
    paths = inputs(tmp_path, monkeypatch)
    if kind == "video_bytes":
        monkeypatch.setattr(module, "MAX_FILE_BYTES", paths[0].stat().st_size - 1)
    elif kind == "aggregate_bytes":
        monkeypatch.setattr(module, "MAX_INPUT_BYTES", paths[2].stat().st_size + 5)
    elif kind == "file_count":
        monkeypatch.setattr(module, "MAX_FILES", 2)
    elif kind == "manifest_count":
        monkeypatch.setattr(module, "MAX_MANIFEST_FILES", 1)
    elif kind == "receipt_bytes":
        monkeypatch.setattr(module, "MAX_RECEIPT_BYTES", 10)
    with pytest.raises((ValueError, TimeoutError)):
        preflight(paths, **({"deadline": time.monotonic() - 1} if kind == "deadline" else {}))


def test_missing_official_model_and_symlink_input_reject(tmp_path, monkeypatch):
    original_provenance = module.exercise_vision.model_provenance
    paths = inputs(tmp_path, monkeypatch)
    monkeypatch.setattr(module.exercise_vision, "model_provenance", original_provenance)
    with pytest.raises(ValueError, match="pinned official"):
        preflight(paths)
    linked = tmp_path / "linked.webm"
    linked.symlink_to(paths[0])
    with pytest.raises(ValueError, match="symlinks"):
        preflight((linked, *paths[1:]))


def test_large_outer_manifest_does_not_expand_consumed_file_scope(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    manifest = json.loads((paths[1] / "manifest.json").read_bytes())
    manifest["files"].update({f"unconsumed/mask-{index}.png": "b" * 64 for index in range(1209)})
    write(paths[1] / "manifest.json", manifest)
    result = preflight(paths)
    receipt = result.receipt()
    assert len(manifest["files"]) == 1211
    assert receipt["input_files"] < module.MAX_FILES
    assert all("unconsumed" not in row["path"] for row in receipt["input_pins"])
    assert "Only consumed artifacts" in receipt["scope"]
    assert receipt["limits"]["maximum_manifest_declarations"] == 100_000
    assert receipt["limits"]["maximum_equipment_mapping_files"] == 256


def test_public_recheck_validates_pins_late_failure_markers_and_shared_deadline(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    result = preflight(paths)
    assert module.recheck_exercise_scene_inputs(result, deadline=time.monotonic() + 5) is None
    with pytest.raises(TimeoutError):
        module.recheck_exercise_scene_inputs(result, deadline=time.monotonic() - 1)
    with pytest.raises(ValueError, match="finite"):
        module.recheck_exercise_scene_inputs(result, deadline=float("nan"))
    with pytest.raises(ValueError, match="verified raw"):
        module.recheck_exercise_scene_inputs({}, deadline=time.monotonic() + 5)
    write(paths[1] / "failure.json", {"status": "failed"})
    with pytest.raises(ValueError, match="Failed"):
        module.recheck_exercise_scene_inputs(result, deadline=time.monotonic() + 5)


def test_one_existing_body_preparation_receives_exact_explicit_options_and_remaining_budget(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    ready = preflight(paths, pose_mode="image", pose_identity="temporal", source_note="Test source rights remain unknown")
    calls, updates = [], []

    def body(video, out, **options):
        calls.append((video, out, options))
        out.mkdir()
        (out / "completed-body.txt").write_text("Saved body stage fixture")
        options["progress"]({"stage": "body", "status": "ready"})
        return out

    monkeypatch.setattr(module.exercise_body_source, "prepare_exercise_body_source", body)
    target = tmp_path / "body_source"
    assert module.prepare_exercise_scene_body(ready, target, deadline=time.monotonic() + 20,
                                              progress=updates.append) == target
    assert len(calls) == 1 and calls[0][:2] == (paths[0], target)
    options = calls[0][2]
    assert options["pose_model"] == paths[3]
    assert options["pose_mode"] == "image" and options["pose_identity"] == "temporal"
    assert options["source_note"] == "Test source rights remain unknown"
    assert 1 <= options["max_seconds"] <= 20 and updates == [{"stage": "body", "status": "ready"}]
    assert not (target / "scene.json").exists() and not (target / "lab").exists()


@pytest.mark.parametrize("changed", ["video", "config", "model", "manifest", "equipment", "failure_marker", "implementation"])
def test_changes_after_preflight_fail_before_body(tmp_path, monkeypatch, changed):
    paths = inputs(tmp_path, monkeypatch)
    ready = preflight(paths)
    path = {"video": paths[0], "config": paths[2], "model": paths[3], "manifest": paths[1] / "manifest.json",
            "equipment": paths[1] / "scene-observations.json", "failure_marker": paths[1] / "failure.json"}.get(changed)
    if path is not None:
        path.write_bytes(b"changed input")
    else:
        original = module.read_package_file

        def changed_implementation(directory, name, **options):
            if name == "exercise_scene_inputs.py":
                raise ValueError("Implementation changed")
            return original(directory, name, **options)
        monkeypatch.setattr(module, "read_package_file", changed_implementation)
    target = tmp_path / "body_source"
    with pytest.raises(ValueError):
        module.prepare_exercise_scene_body(ready, target)
    assert not target.exists()


@pytest.mark.parametrize("failure_marker", [False, True])
def test_post_body_input_change_keeps_completed_body_but_cannot_admit_scene(tmp_path, monkeypatch, failure_marker):
    paths = inputs(tmp_path, monkeypatch)
    ready = preflight(paths)

    def body(_video, out, **_options):
        out.mkdir()
        (out / "completed-body.txt").write_text("Completed body remains inspectable")
        (paths[1] / "failure.json" if failure_marker else paths[2]).write_bytes(b"changed during body")
        return out

    monkeypatch.setattr(module.exercise_body_source, "prepare_exercise_body_source", body)
    target = tmp_path / "body_source"
    with pytest.raises(ValueError):
        module.prepare_exercise_scene_body(ready, target)
    assert (target / "completed-body.txt").exists() and not (target / "lab").exists()


def mapped_inputs(tmp_path, monkeypatch):
    paths = inputs(tmp_path, monkeypatch)
    bundle = tmp_path / "mapping"
    mapping_fixture(bundle, derivative_offset=0)
    item = module.load_source_equivalence(bundle)
    paths[0].write_bytes((bundle / "source.webm").read_bytes())
    write(paths[1] / "scene-observations.json", observations(item.derivative_sha256, count=3,
        times=[float(row.derivative_decoder_pts) / 1000 for row in item.frames], dimensions=item.dimensions))
    seal(paths[1])
    config = json.loads(paths[2].read_bytes()); config["source_sha256"] = item.original_sha256
    write(paths[2], config)
    return paths, bundle, item


def test_verified_derivative_keeps_exact_original_frames_and_distinct_decoder_clock(tmp_path, monkeypatch):
    paths, bundle, item = mapped_inputs(tmp_path, monkeypatch)
    ready = preflight(paths, source_mappings=[bundle])
    receipt = ready.receipt()
    assert receipt["source_sha256"] == item.original_sha256
    assert receipt["equipment_source_sha256"] == item.derivative_sha256
    assert receipt["selected_equipment"][0]["source_frame"] == 2
    assert receipt["selected_equipment"][0]["time_s"] == .070
    assert receipt["source_mappings"][0]["source_clock"] != receipt["source_mappings"][0]["derivative_clock"]
    pinned = {row["path"] for row in receipt["input_pins"]}
    assert all(str(bundle / name) in pinned for name in item.file_names)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "wrong_original", "clock", "bytes", "late_failure", "proof_bound"])
def test_mapping_mismatch_or_mutation_rejects_without_inference(tmp_path, monkeypatch, kind):
    paths, bundle, item = mapped_inputs(tmp_path, monkeypatch)
    mappings = [bundle]
    if kind == "missing":
        mappings = []
    elif kind == "duplicate":
        mappings.append(bundle)
    elif kind == "wrong_original":
        paths[0].write_bytes(b"another original")
        config = json.loads(paths[2].read_bytes()); config["source_sha256"] = sha(paths[0].read_bytes())
        write(paths[2], config)
    elif kind == "clock":
        def shift(doc):
            doc["source"]["frame_times_s"] = [value + .1 for value in doc["source"]["frame_times_s"]]
            for row in (*doc["evidence"], *doc["observations"]):
                row["time_s"] += .1
                if "payload" in row:
                    row["payload"]["time_s"] += .1
        edit_equipment(paths, shift)
    elif kind == "bytes":
        (bundle / "lossless.mp4").write_bytes(b"changed derivative")
    elif kind == "proof_bound":
        monkeypatch.setattr(module, "MAX_EVIDENCE_BYTES", (paths[1] / "scene-observations.json").stat().st_size + 1)
    elif kind == "late_failure":
        ready = preflight(paths, source_mappings=mappings)
        write(bundle / "failure.json", {"status": "failed"})
        with pytest.raises(ValueError):
            module.prepare_exercise_scene_body(ready, tmp_path / "body_source")
        return
    with pytest.raises(ValueError):
        preflight(paths, source_mappings=mappings)
