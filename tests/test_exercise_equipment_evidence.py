"""Ordinary saved-run composition with authored arrays; no media/model/native jobs."""

from copy import deepcopy
from hashlib import sha256
import importlib
import json
from pathlib import Path
import time

import pytest

from test_multiview_landmark_evidence import denied, local_only
from test_equipment_hand_snapshot import authored_files


def encode(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sealed(root, files, *, ordinary=False, status="construction_only", extra_pins=None):
    root.mkdir()
    for name, raw in files.items():
        target = root / name; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(raw)
    pins = {name: sha256(raw).hexdigest() for name, raw in files.items()} | (extra_pins or {})
    metadata = ({"schema_version": 1, "profile": "exercise-body-source-v1", "status": status,
                 "simulation_ready": False} if ordinary else {"schema": 1})
    value = {**metadata, "files": pins}
    (root / "manifest.json").write_bytes(encode(value))
    return value


def parent_files(*, ordinary=True, failed=False, model=True, pose_change=None, mask_change=None):
    previous = authored_files()
    pose = json.loads(previous["inputs/context.json"])
    pose["inference_stride"] = 2
    if pose_change:
        pose_change(pose)
    raw_pose = previous["inputs/raw.json"]
    values = {"evidence.json": encode(pose), "pose-landmarks.json": raw_pose,
              "identity-audit.json": previous["inputs/audit.json"]}
    if ordinary:
        values["source.json"] = encode({"sha256": pose["source_sha256"], "artifact": "source.webm",
            "rights": "Authored bytes; no source video or authenticated model run"})
        values["run-config.json"] = encode({"profile": "exercise", "pose_identity": pose["pose_identity"]["mode"],
            "pose_mode": "video", "body_model": "mobile"})
        if model:
            values["model.json"] = encode(pose["model"])
        if failed:
            values["failure.json"] = encode({"profile": "exercise-body-source-v1", "status": "failed",
                "error": {"type": "UnsupportedVideo", "message": "Authored body quality failure retained"}})
    equipment = {"scene-observations.json": previous["inputs/equipment.json"],
                 "report.json": encode({"status": "evidence_only"})}
    masks = json.loads(previous["inputs/masks.json"])
    if mask_change:
        mask_change(masks)
    return values, equipment, retained_masks(encode(masks))


def retained_masks(mask_raw):
    masks = json.loads(mask_raw)
    mask_files = {"mask-evidence.json": mask_raw, "report.json": encode({"status": "decoded_proposals"})}
    records = []
    for row in masks["observations"]:
        if row["mask_status"] != "decoded":
            continue
        raster = b"\1\1\1\1\0\1\1\1\1"
        png = b"Authored opaque PNG member; never decoded by this test"
        path = row["raster_path"]
        mask_files[path] = png
        mask_files["binary/" + path.removesuffix(".png") + ".bin"] = raster
        records.append({"path": path, "png_sha256": sha256(png).hexdigest(), "png_bytes": len(png),
            "binary_sha256": sha256(raster).hexdigest(), "binary_bytes": len(raster),
            "dimensions_hw": [3, 3], "mask_evidence_id": row["id"]})
    mask_files["binary-retention.json"] = encode({"schema_version": 1,
        "profile": "decoded-segmentation-binary-retention-v1", "mask_document_sha256": sha256(mask_raw).hexdigest(),
        "records": records})
    return mask_files


def parents(tmp_path, *, ordinary=True, failed=False, model=True, pose_change=None, mask_change=None):
    files = parent_files(ordinary=ordinary, failed=failed, model=model, pose_change=pose_change, mask_change=mask_change)
    paths = tuple(tmp_path / name for name in ("pose", "equipment", "masks"))
    source = json.loads(files[0]["evidence.json"])["source_sha256"]
    sealed(paths[0], files[0], ordinary=ordinary, status="failed" if failed else "construction_only",
           extra_pins={"source.webm": source} if ordinary else None)
    equipment_manifest = sealed(paths[1], files[1])
    files[2]["report.json"] = encode({"status": "decoded_proposals", "source_verification": {
        "manifest_sha256": sha256((paths[1] / "manifest.json").read_bytes()).hexdigest(), "files": equipment_manifest["files"]}})
    sealed(paths[2], files[2])
    return paths, files


@pytest.fixture
def api(monkeypatch):
    module = importlib.import_module("weft.exercise_equipment_evidence")
    # Runtime capture is exercised separately; repeated cases test composition, not code archiving.
    monkeypatch.setattr(module, "_runtime_archive", lambda deadline: (b"authored runtime receipt", {}))
    return module


def run(api, paths, out, **kwargs):
    options = {"human_id": "person-0", "equipment_ids": ["sam-2"], "minimum_confidence": .5,
        "uncertainty_px": .25, "uncertainty_method": "Authored test pixel interval",
        "uncertainty_note": "Not calibrated or derived from source confidence"}
    return api.prepare_exercise_equipment_evidence(*paths, out, **(options | kwargs))


def assert_sealed(root, status="completed"):
    value = json.loads((root / "manifest.json").read_bytes())
    assert value["status"] == status
    assert all(sha256((root / name).read_bytes()).hexdigest() == pin for name, pin in value["files"].items())
    return value


@pytest.mark.parametrize("ordinary,failed,model", [(False, False, False), (True, False, True), (True, True, False)])
def test_actual_composition_loads_existing_hand_route_and_retains_parent(api, tmp_path, monkeypatch, ordinary, failed, model):
    from weft.equipment_support_snapshot import load_equipment_support_snapshot
    from weft.equipment_hand_snapshot import prepare_equipment_hand_snapshot
    paths, files = parents(tmp_path, ordinary=ordinary, failed=failed, model=model)
    reads = []
    original = api.read_package_file
    def read(root, name, **kwargs):
        assert not name.endswith((".mp4", ".webm"))
        if name.endswith((".png", ".bin")):
            frame = int(Path(name).stem.split("-")[0])
            assert frame in range(0, 14, 2)
        reads.append((str(root), name))
        return original(root, name, **kwargs)
    monkeypatch.setattr(api, "read_package_file", read)
    out = tmp_path / "composition"
    result = run(api, paths, out)
    assert result["sealed"] and result["status"] == "completed" and result["output_path"] == str(out)
    assert result["manifest_sha256"] == sha256((out / "manifest.json").read_bytes()).hexdigest()
    assert result["parent_pose"]["failure_retained"] is failed
    assert result["raster_loads"] == 7 and result["binary_bytes"] == 63
    assert result["counts"]["source_frames"] == 24 and result["counts"]["pose_slots"] == 12
    assert result["construction_frames"] == list(range(0, 14, 2))
    assert result["input_bytes"] == sum((Path(root) / name).stat().st_size for root, name in reads)
    assert len(reads) == len(set(reads))
    assert all(result[key] == 0 for key in ("source_decodes", "raster_decodes", "pose_model_calls", "native_calls", "api_calls"))
    assert result["contact"] == result["grasp"] == result["load"] == "unknown" and not result["physics_ready"]
    assert (out / "inputs/source_manifest.json").read_bytes() == (paths[0] / "manifest.json").read_bytes()
    if failed:
        assert (out / "inputs/pose-parent-failure.json").read_bytes() == files[0]["failure.json"]
        assert "model.json" not in json.loads((out / "inputs/source_manifest.json").read_bytes())["files"]
    assert_sealed(out)
    snapshot = load_equipment_support_snapshot(out, deadline=time.monotonic() + 25)
    hand = prepare_equipment_hand_snapshot(snapshot, deadline=time.monotonic() + 25)
    assert hand.evidence.describe()["counts"]["source_frames"] == 24
    assert hand.evidence.frame(2)["hands"][0]["status"] == "conditional_pointset_image_intersection"
    assert all(p["pixel"] is None for h in hand.evidence.frame(22)["hands"] for p in h["points"])


def test_unavailable_selected_and_excluded_arrays_never_trigger_raster_reads(api, tmp_path, monkeypatch):
    def change(pose):
        for row in pose["frames"]:
            if row["frame"] >= 14:
                row["landmarks"] = {"coordinate_sentinel": "excluded model payload"}
            if row["frame"] == 0:
                row["landmarks"] = []; row["status"] = "not_detected"; row["pose_count"] = 0
    paths, _ = parents(tmp_path, pose_change=change)
    original = api.read_package_file
    def read(root, name, **kwargs):
        if name.endswith((".png", ".bin")):
            assert int(Path(name).stem.split("-")[0]) in range(2, 14, 2)
        return original(root, name, **kwargs)
    monkeypatch.setattr(api, "read_package_file", read)
    result = run(api, paths, tmp_path / "out")
    assert result["raster_loads"] == 6
    human = json.loads((tmp_path / "out/human-observations.json").read_bytes())
    assert len(human["observations"]) == 23
    assert {row["frame"] for row in human["evidence"]} == set(range(0, 14, 2))
    assert "coordinate_sentinel" not in (tmp_path / "out/human-observations.json").read_text()
    assert all(row["measurements"] == {} for row in human["observations"] if row["frame"] >= 14)


def test_low_confidence_wrists_do_not_remove_masks_needed_by_qualified_fingers(api, tmp_path):
    from weft.equipment_support_snapshot import load_equipment_support_snapshot
    from weft.equipment_hand_snapshot import prepare_equipment_hand_snapshot
    paths, _ = parents(tmp_path)
    root = paths[0]; manifest = json.loads((root / "manifest.json").read_bytes())
    context, raw = (json.loads((root / name).read_bytes()) for name in ("evidence.json", "pose-landmarks.json"))
    for document in (context, raw):
        for row in document["frames"]:
            for index in (15, 16): row["landmarks"][index]["visibility"] = .1
    raw_bytes = encode(raw); pin = sha256(raw_bytes).hexdigest()
    context["pose_identity"]["raw_artifact_sha256"] = pin
    audit = json.loads((root / "identity-audit.json").read_bytes()); audit["raw_artifact_sha256"] = pin
    for name, value in {"evidence.json": encode(context), "pose-landmarks.json": raw_bytes, "identity-audit.json": encode(audit)}.items():
        (root / name).write_bytes(value); manifest["files"][name] = sha256(value).hexdigest()
    (root / "manifest.json").write_bytes(encode(manifest))
    out = tmp_path / "out"; result = run(api, paths, out)
    assert result["raster_loads"] == 0 and result["retained_binary_loads"] == 7
    assert len(json.loads((out / "raster-reads.json").read_bytes())) == 7
    snapshot = load_equipment_support_snapshot(out, deadline=time.monotonic() + 25)
    hand = prepare_equipment_hand_snapshot(snapshot, deadline=time.monotonic() + 25)
    group = hand.evidence.frame(2)["hands"][0]
    assert group["status"] == "unavailable" and group["contact"] == "unknown"
    assert group["points"][0]["reason"] == "below_authored_confidence_threshold"
    assert all(row["status"] == "conditional_image_intersection" for row in group["points"][1:])


@pytest.mark.parametrize("size", [168516, 256 << 10, (256 << 10) + 1])
def test_equipment_report_size_matches_ordinary_export_without_raising_aggregate_cap(api, tmp_path, size):
    paths, _ = parents(tmp_path)
    equip = paths[1]; mask = paths[2]
    original = (equip / "report.json").read_bytes()
    report = original.rstrip() + b" " * (size - len(original.rstrip()) - 1) + b"\n"
    assert len(report) == size
    (equip / "report.json").write_bytes(report)
    seal = json.loads((equip / "manifest.json").read_bytes())
    seal["files"]["report.json"] = sha256(report).hexdigest()
    (equip / "manifest.json").write_bytes(encode(seal))
    mask_report = json.loads((mask / "report.json").read_bytes())
    mask_report["source_verification"] = {"manifest_sha256": sha256((equip / "manifest.json").read_bytes()).hexdigest(),
                                           "files": seal["files"]}
    mask_report_raw = encode(mask_report); (mask / "report.json").write_bytes(mask_report_raw)
    mask_seal = json.loads((mask / "manifest.json").read_bytes())
    mask_seal["files"]["report.json"] = sha256(mask_report_raw).hexdigest()
    (mask / "manifest.json").write_bytes(encode(mask_seal))
    if size > 256 << 10:
        with pytest.raises(ValueError, match="bound"):
            run(api, paths, tmp_path / "out")
        assert_sealed(tmp_path / "out", "failed")
    else:
        result = run(api, paths, tmp_path / "out")
        assert (tmp_path / "out/inputs/equipment-report.json").read_bytes() == report
        assert result["bounds"]["input_bytes"] == 32 << 20
        assert result["retained_binary_loads"] == 7 and result["contact"] == "unknown"


@pytest.mark.parametrize("fault", ["schedule", "boolean_stride", "decoded", "unknown_parent", "body_status",
    "false_ready", "missing_failure", "contradictory_failure", "source_pin", "model", "options", "manifest_pin"])
def test_pose_parent_admission_fails_before_raster_and_preserves_failed_package(api, tmp_path, monkeypatch, fault):
    paths, _ = parents(tmp_path, failed=fault == "missing_failure")
    root = paths[0]
    manifest = json.loads((root / "manifest.json").read_bytes())
    def replace(name, change):
        value = json.loads((root / name).read_bytes()); change(value); raw = encode(value)
        (root / name).write_bytes(raw); manifest["files"][name] = sha256(raw).hexdigest()
    if fault == "schedule": replace("evidence.json", lambda value: value["frames"].pop())
    if fault == "boolean_stride": replace("evidence.json", lambda value: value.update(inference_stride=True))
    if fault == "decoded": replace("evidence.json", lambda value: value.update(frame_count=30))
    if fault == "unknown_parent": manifest["profile"] = "unknown-source-v1"
    if fault == "body_status": manifest["status"] = "completed"
    if fault == "false_ready": manifest["simulation_ready"] = True
    if fault == "missing_failure": del manifest["files"]["failure.json"]
    if fault == "contradictory_failure": manifest["files"]["failure.json"] = "f" * 64
    if fault == "source_pin": replace("source.json", lambda value: value.update(sha256="f" * 64))
    if fault == "model": replace("model.json", lambda value: value.update(sha256="f" * 64))
    if fault == "options": replace("run-config.json", lambda value: value.update(pose_identity="raw"))
    (root / "manifest.json").write_bytes(encode(manifest))
    original = api.read_package_file
    def read(root, name, **kwargs):
        assert not name.endswith((".png", ".bin"))
        return original(root, name, **kwargs)
    monkeypatch.setattr(api, "read_package_file", read)
    with pytest.raises(ValueError):
        run(api, paths, tmp_path / "out", **({"expected_pose_manifest_sha256": "e" * 64} if fault == "manifest_pin" else {}))
    assert_sealed(tmp_path / "out", "failed")
    assert json.loads((tmp_path / "out/failure.json").read_bytes())["sealed"] is False


@pytest.mark.parametrize("fault", ["missing_inventory", "mask_pin", "dimensions", "png_pin", "binary_pin", "duplicate", "unknown_id",
                                   "missing_binary", "binary_values", "failed_mask", "failed_equipment", "missing_mapping", "equipment_ancestor"])
def test_mask_and_source_custody_fail_closed_without_decoder_fallback(api, tmp_path, fault):
    paths, _ = parents(tmp_path)
    root = paths[2]; manifest = json.loads((root / "manifest.json").read_bytes())
    inventory = json.loads((root / "binary-retention.json").read_bytes())
    row = inventory["records"][0]
    if fault == "missing_inventory": del manifest["files"]["binary-retention.json"]
    if fault == "mask_pin": inventory["mask_document_sha256"] = "e" * 64
    if fault == "dimensions": row["dimensions_hw"] = [1, 9]
    if fault == "png_pin": row["png_sha256"] = "e" * 64
    if fault == "binary_pin": row["binary_sha256"] = "e" * 64
    if fault == "duplicate": inventory["records"].append(row)
    if fault == "unknown_id": row["mask_evidence_id"] = "unknown"
    if fault in {"missing_binary", "binary_values"}:
        path = root / ("binary/" + row["path"].removesuffix(".png") + ".bin")
        if fault == "missing_binary": path.rename(root / "retained-missing.bin")
        else: path.write_bytes(b"\2" * 9)
    if fault == "failed_mask":
        raw = encode({"status": "failed"}); (root / "report.json").write_bytes(raw)
        manifest["files"]["report.json"] = sha256(raw).hexdigest()
    if fault == "equipment_ancestor":
        value = json.loads((root / "report.json").read_bytes()); value["source_verification"]["manifest_sha256"] = "e" * 64
        raw = encode(value); (root / "report.json").write_bytes(raw); manifest["files"]["report.json"] = sha256(raw).hexdigest()
    if fault in {"failed_equipment", "missing_mapping"}:
        equip_root = paths[1]; seal = json.loads((equip_root / "manifest.json").read_bytes())
        name = "report.json" if fault == "failed_equipment" else "scene-observations.json"
        value = json.loads((equip_root / name).read_bytes())
        if fault == "failed_equipment": value["status"] = "failed"
        else: value["source"]["sha256"] = "e" * 64
        raw = encode(value); (equip_root / name).write_bytes(raw); seal["files"][name] = sha256(raw).hexdigest()
        (equip_root / "manifest.json").write_bytes(encode(seal))
    if fault != "missing_inventory":
        raw = encode(inventory); (root / "binary-retention.json").write_bytes(raw)
        manifest["files"]["binary-retention.json"] = sha256(raw).hexdigest()
    (root / "manifest.json").write_bytes(encode(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        run(api, paths, tmp_path / "out")
    assert_sealed(tmp_path / "out", "failed")


@pytest.mark.parametrize("cap", ["MAX_INPUT", "MAX_OUTPUT", "MAX_FILES", "MAX_BINARY", "MAX_LOADS"])
def test_explicit_byte_file_and_load_caps_preserve_failure_without_dropping_rows(api, tmp_path, monkeypatch, cap):
    paths, _ = parents(tmp_path)
    monkeypatch.setattr(api, cap, {"MAX_INPUT": 128, "MAX_OUTPUT": 300000, "MAX_FILES": 12,
                                  "MAX_BINARY": 8, "MAX_LOADS": 0}[cap])
    with pytest.raises(ValueError): run(api, paths, tmp_path / "out")
    assert not (tmp_path / "out/manifest.json").exists() or json.loads((tmp_path / "out/manifest.json").read_bytes())["status"] == "failed"


def test_output_path_no_overwrite_or_symlink_and_deadline_stops_before_new_stage(api, tmp_path):
    paths, _ = parents(tmp_path)
    with pytest.raises(ValueError): run(api, paths, paths[0])
    linked = tmp_path / "linked"; linked.symlink_to(paths[0], target_is_directory=True)
    with pytest.raises(ValueError): run(api, (linked, *paths[1:]), tmp_path / "not-created")
    with pytest.raises(ValueError, match="remaining time"): run(api, paths, tmp_path / "deadline", max_seconds=1.)
    assert_sealed(tmp_path / "deadline", "failed")


def test_final_verified_seal_is_not_relabelled_when_write_finishes_late(api, tmp_path, monkeypatch):
    paths, _ = parents(tmp_path)
    actual, clock = api.write_package_files, [time.monotonic()]
    monkeypatch.setattr(api.time, "monotonic", lambda: clock[0])
    def write(root, files):
        actual(root, files)
        if "manifest.json" in files and json.loads(files["manifest.json"])["status"] == "completed":
            clock[0] += 31
    monkeypatch.setattr(api, "write_package_files", write)
    result = run(api, paths, tmp_path / "out")
    assert result["status"] == "completed"
    assert_sealed(tmp_path / "out")
    assert not (tmp_path / "out/attempted-manifest.json").exists()
    assert not (tmp_path / "out/failure.json").exists()
    assert json.loads((tmp_path / "out/result.json").read_bytes())["status"] == "prepared_pending_terminal_seal"


@pytest.mark.parametrize("partial,occupied", [(False, False), (False, True), (True, False)])
def test_caller_failure_preserves_published_seal_and_partial_attempts(api, tmp_path, monkeypatch, partial, occupied):
    paths, _ = parents(tmp_path)
    actual = api.write_package_files
    sibling = tmp_path / "out.caller-failure.json"
    if occupied: sibling.write_bytes(b"Existing caller evidence must not be overwritten")
    def write(root, files):
        if "manifest.json" in files and json.loads(files["manifest.json"])["status"] == "completed":
            if partial:
                (root / "manifest.json").write_bytes(b'{"partial":')
            else:
                actual(root, files)
            raise OSError("Authored caller failure after terminal write")
        actual(root, files)
    monkeypatch.setattr(api, "write_package_files", write)
    with pytest.raises(OSError, match="Authored caller failure"):
        run(api, paths, tmp_path / "out")
    if partial:
        assert_sealed(tmp_path / "out", "failed")
        assert (tmp_path / "out/attempted-manifest.json").read_bytes() == b'{"partial":'
        assert not sibling.exists()
    else:
        seal = assert_sealed(tmp_path / "out")
        assert set(seal["files"]) | {"manifest.json"} == {
            path.relative_to(tmp_path / "out").as_posix() for path in (tmp_path / "out").rglob("*") if path.is_file()}
        if occupied:
            assert sibling.read_bytes() == b"Existing caller evidence must not be overwritten"
        else:
            assert json.loads(sibling.read_bytes())["artifact_status"] == "completed"
        assert not (tmp_path / "out/failure.json").exists()


@pytest.mark.parametrize("status", ["construction_only", "failed"])
def test_known_parent_precursor_keeps_exact_original_manifest_and_explicit_failure(api, tmp_path, status):
    from weft.pose_precursor_custody import prepare_pose_precursor_custody
    paths, files = parents(tmp_path, failed=status == "failed")
    manifest = (paths[0] / "manifest.json").read_bytes()
    custody = prepare_pose_precursor_custody(context_pose_document=files[0]["evidence.json"],
        raw_pose_document=files[0]["pose-landmarks.json"], identity_audit_document=files[0]["identity-audit.json"],
        manifest_document=manifest)
    assert custody.describe()["parent_run"]["status"] == status
    assert dict(custody.input_documents)["manifest.json"] == manifest
    assert custody.describe()["contact"] == "unknown" and custody.describe()["identity_authenticated"] is False


def test_committed_legacy_precursor_metadata_and_rows_remain_byte_exact():
    from weft.pose_precursor_custody import prepare_pose_precursor_custody
    from test_pose_precursor_custody import documents
    value = prepare_pose_precursor_custody(**documents())
    assert sha256(value._metadata).hexdigest() == "fb04fd151f7f95bf468f231e8c55ddeb12606680916f9c4042dae114d503c2c6"
    assert sha256(b"\n".join(value._source_rows)).hexdigest() == "d8c2c97bf68320a041f6244d4111191afa27ea978d7eec0b8660d09ba3d3a97c"
    assert "parent_run" not in value.describe()


@pytest.mark.parametrize("change", [{"profile": "unknown"}, {"status": "completed"}, {"simulation_ready": True},
                                     {"schema_version": True}, {"schema_version": 2}])
def test_unknown_parent_manifest_is_not_a_new_legacy_alias(change):
    from weft.pose_precursor_custody import prepare_pose_precursor_custody
    from test_pose_precursor_custody import documents
    values = documents(); manifest = json.loads(values["manifest_document"])
    del manifest["schema"]
    manifest.update(schema_version=1, profile="exercise-body-source-v1", status="failed", simulation_ready=False)
    manifest.update(change); values["manifest_document"] = encode(manifest)
    with pytest.raises(ValueError, match="hash-only run manifest"):
        prepare_pose_precursor_custody(**values)


def test_loaded_runtime_is_a_single_bounded_exact_source_archive():
    import io
    import zipfile
    from weft.exercise_equipment_evidence import _runtime_archive
    raw, pins = _runtime_archive(time.monotonic() + 5)
    assert len(raw) <= 9 << 20 and 1 <= len(pins) <= 128
    assert "exercise_equipment_evidence.py" in pins
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert set(archive.namelist()) == set(pins)
        assert all(sha256(archive.read(name)).hexdigest() == pin for name, pin in pins.items())


@pytest.mark.parametrize("mutate", [False, True])
def test_recorded_derivative_proof_resolves_declared_count_discrepancy_without_retiming(api, tmp_path, mutate):
    from weft.source_equivalence import load_source_equivalence
    from weft.equipment_support_snapshot import load_equipment_support_snapshot
    from test_source_equivalence import fixture as proof_fixture
    from test_equipment_silhouette_evidence import fixture as silhouette_fixture
    proof_root = proof_fixture(tmp_path / "proof")
    proof = load_source_equivalence(proof_root)
    files, _, _ = parent_files(ordinary=False)
    pose, raw = json.loads(files["evidence.json"]), json.loads(files["pose-landmarks.json"])
    pose.update(source_sha256=proof.original_sha256, width=2, height=2, frame_count=5, decoded_frames=3, inference_stride=1)
    clocks = [0., .033, .067]
    source_rows = []
    for index in range(3):
        row = deepcopy(raw["frames"][0]); row.update(frame=index, id=f"{proof.original_sha256}:frame:{index}",
            time_s=index / 30, decoder_timestamp_s=clocks[index])
        for point in row["landmarks"]: point.update(x=.4, y=.4)
        source_rows.append(row)
    raw["frames"] = deepcopy(source_rows); pose["frames"] = source_rows
    raw_bytes = encode(raw); raw_pin = sha256(raw_bytes).hexdigest()
    pose["pose_identity"]["raw_artifact_sha256"] = raw_pin
    audit = json.loads(files["identity-audit.json"])
    audit.update(source_sha256=proof.original_sha256, raw_artifact_sha256=raw_pin, unavailable_frames=[])
    files = {"evidence.json": encode(pose), "pose-landmarks.json": raw_bytes, "identity-audit.json": encode(audit)}
    _, args, _, _ = silhouette_fixture(importlib.import_module("weft.equipment_silhouette_evidence"),
        supplied_pose=pose, pose_change=lambda value: value.update(deepcopy(pose)), proof=proof, box=(0, 0, 2, 2))
    equip = {"scene-observations.json": args["equipment_document"], "report.json": encode({"status": "evidence_only"})}
    mask = retained_masks(args["mask_document"])
    paths = tuple(tmp_path / name for name in ("pose", "equipment", "masks"))
    sealed(paths[0], files)
    equipment_manifest = sealed(paths[1], equip)
    mask["report.json"] = encode({"status": "decoded_proposals", "source_verification": {
        "manifest_sha256": sha256((paths[1] / "manifest.json").read_bytes()).hexdigest(), "files": equipment_manifest["files"]}})
    sealed(paths[2], mask)
    out = tmp_path / "out"
    def progress(stage, detail):
        if mutate and stage == "seal":
            (proof_root / "report.json").write_bytes(proof.raw_bytes("report.json") + b" ")
    if mutate:
        with pytest.raises(ValueError, match="changed before publication"):
            run(api, paths, out, source_equivalence_run=proof_root, progress=progress)
        assert_sealed(out, "failed")
        assert (out / "source-equivalence/report.json").read_bytes() == proof.raw_bytes("report.json")
        return
    result = run(api, paths, out, source_equivalence_run=proof_root, progress=progress)
    assert result["proof_manifest_repeat_bytes"] == len(proof.raw_bytes("manifest.json"))
    assert result["input_read_operations"] == result["input_files"] + 1
    assert result["parent_pose"]["decode_completeness"] == "recorded_whole_video_source_equivalence_not_independent_decode"
    assert result["construction_frames"] == [0] and result["counts"]["source_frames"] == 5
    assert result["counts"]["decoded_frames"] == 3 and result["counts"]["pose_slots"] == 3
    assert result["retained_binary_loads"] == 1
    snapshot = load_equipment_support_snapshot(out, deadline=time.monotonic() + 5)
    assert snapshot.silhouette.source_equivalence.manifest_sha256 == proof.manifest_sha256
    assert snapshot.silhouette.frame(2)["pose_decoder_timestamp_s"] == .067
    assert snapshot.silhouette.frame(3)["source_status"] == "metadata_only_not_decoded"
    assert all((out / "source-equivalence" / name).read_bytes() == proof.raw_bytes(name) for name in proof.file_names)
