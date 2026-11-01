"""Authored saved-array custody and cached-mask integration; no real media or inference."""

from functools import lru_cache
from hashlib import sha256
import importlib
import json
import time

import pytest

from weft import equipment_hand_snapshot as hand_snapshot
from weft.equipment_support_snapshot import load_equipment_support_snapshot
from test_equipment_silhouette_evidence import fixture as wrist_fixture
from test_equipment_support_snapshot import encoded, seal_files
from test_pose_precursor_custody import documents


@lru_cache(maxsize=1)
def authored_files():
    def points(value):
        for row in value["frames"]:
            row["decoder_timestamp_s"] = row["time_s"]
            for index in (15, 16):
                row["landmarks"][index].update(x=.35, y=33/80)
            for index in (17, 18, 19, 20, 21, 22):
                row["landmarks"][index].update(x=.33, y=31/80)
    inputs = documents(count=24, context_change=points, precursor_change=points)
    context = json.loads(inputs["context_pose_document"])
    selected = [row["frame"] for row in context["frames"][:7]]
    def selection(value):
        for row in value["observations"]:
            if row["frame"] not in selected:
                row["partition"] = "unassigned"
    api = importlib.import_module("weft.equipment_silhouette_evidence")
    initial, args, calls, rasters = wrist_fixture(api, supplied_pose=context,
        pose_change=lambda value: value.update(context), observation_change=selection)
    custody = importlib.import_module("weft.pose_precursor_custody").prepare_pose_precursor_custody(**inputs)
    channels = json.loads(initial.annotation_document())["channels"]
    annotations = importlib.import_module("weft.pose_surface_annotations").prepare_pose_surface_annotations(
        dict(initial.input_documents), pose_document=inputs["context_pose_document"],
        source_sha256=context["source_sha256"], channels=channels,
        landmark_bindings=[{"channel_id": row["id"], "landmark_name": row["point_role"], "uncertainty_px": .25}
                           for row in channels], minimum_confidence=.5,
        uncertainty_method="Authored quarter-pixel analytical fixture",
        uncertainty_note="Not real source ancestry or calibrated error", precursor_custody=custody)
    prepared = api.prepare_equipment_silhouette_evidence(annotations, construction_frames=selected, **args)
    values = {
        "inputs/context.json": annotations.pose_document, "inputs/equipment.json": prepared.equipment_document,
        "inputs/masks.json": prepared.mask_document, "human-observations.json": annotations.input_documents[0][1],
        "source-annotations.json": annotations.annotation_document(), "annotation-metadata.json": encoded(annotations.describe()),
        "annotation-frames.jsonl": b"\n".join(annotations._rows) + b"\n",
        "silhouette/metadata.json": encoded(prepared.describe()),
        "silhouette/frames.jsonl": b"\n".join(prepared._rows) + b"\n",
        "inputs/raw.json": inputs["raw_pose_document"], "inputs/audit.json": inputs["identity_audit_document"],
        "inputs/source_manifest.json": inputs["manifest_document"],
    }
    records = []
    for request in calls:
        name = request["path"]; raster = rasters[name]
        values["binary/" + name.removesuffix(".png") + ".bin"] = raster
        records.append({"path": name, "png_sha256": "c" * 64, "png_bytes": 10,
            "binary_sha256": sha256(raster).hexdigest(), "binary_bytes": len(raster),
            "dimensions_hw": request["dimensions"], "mask_evidence_id": request["mask_evidence_id"]})
    values["raster-reads.json"] = encoded(records)
    values["inputs/mask_manifest.json"] = encoded({"files": {
        "mask-evidence.json": sha256(prepared.mask_document).hexdigest(),
        **{row["path"]: row["png_sha256"] for row in records}}})
    return values


def seal(source, values):
    manifest = seal_files(source, values)
    # These authored labels are never represented as decoded PNG test evidence.
    manifest["files"].update({"pngs/" + row["path"]: row["png_sha256"]
                              for row in json.loads(values["raster-reads.json"])})
    (source / "manifest.json").write_bytes(encoded(manifest))
    return manifest


@pytest.fixture
def study(tmp_path):
    values = dict(authored_files()); source = tmp_path / "source"
    seal(source, values)
    return source, values


def prepare(source):
    deadline = time.monotonic() + 25
    snapshot = load_equipment_support_snapshot(source, deadline=deadline)
    return hand_snapshot.prepare_equipment_hand_snapshot(snapshot, deadline=deadline)


def test_cached_masks_keep_exact_precursors_wrist_result_and_complete_slots(study, monkeypatch):
    source, values = study
    read = hand_snapshot.read_package_file
    names = []
    def bounded(root, name, **kwargs):
        assert not name.endswith((".png", ".mp4"))
        names.append(name)
        return read(root, name, **kwargs)
    monkeypatch.setattr(hand_snapshot, "read_package_file", bounded)
    result = prepare(source); meta = result.evidence.describe()
    assert meta["counts"]["source_frames"] == 24 and meta["counts"]["pose_slots"] == 12
    assert meta["construction_frames"] == list(range(0, 14, 2))
    assert result.snapshot.read_work["cached_binary_reads"] == 7
    assert result.snapshot.read_work["cached_binary_bytes"] == 63
    assert result.snapshot.read_work["point_relations"] == 56
    assert result.snapshot.source_files["silhouette/frames.jsonl"] == values["silhouette/frames.jsonl"]
    assert len(names) == len(set(names)) == 12
    group = result.evidence.frame(2)["hands"][0]
    assert group["points"][0]["status"] == "conditional_image_exclusion"
    assert group["status"] == "conditional_pointset_image_intersection" and group["contact"] == "unknown"
    assert all(point["pixel"] is None for group in result.evidence.frame(22)["hands"] for point in group["points"])
    assert all(binding["uncertainty_px"] == .25 for binding in json.loads(result.artifacts["hand/preparation.json"])["landmark_bindings"])
    assert result.snapshot.read_work["selected_bytes"] == sum(map(len, result.snapshot.source_files.values()))
    result.snapshot.recheck(time.monotonic() + 1)


@pytest.mark.parametrize("fault", ["raw_ancestor", "mask_ancestor", "png_ancestor", "binary_descriptor", "duplicate", "missing", "dimensions", "identity"])
def test_resealed_mismatched_ancestry_or_request_fails_closed(study, fault):
    source, values = study
    records = json.loads(values["raster-reads.json"])
    if fault == "raw_ancestor": values["inputs/raw.json"] += b" "
    if fault == "mask_ancestor":
        document = json.loads(values["inputs/mask_manifest.json"])
        document["files"]["mask-evidence.json"] = "f" * 64
        values["inputs/mask_manifest.json"] = encoded(document)
    if fault == "png_ancestor": records[0]["png_sha256"] = "f" * 64
    if fault == "binary_descriptor": records[0]["binary_sha256"] = "f" * 64
    if fault == "duplicate": records.append(records[0])
    if fault == "missing": records.pop(0)
    if fault == "dimensions": records[0]["dimensions_hw"] = [2, 3]
    if fault == "identity": records[0]["mask_evidence_id"] = "different-original-mask"
    values["raster-reads.json"] = encoded(records)
    seal(source, values)
    with pytest.raises(ValueError): prepare(source)


def test_binary_bytes_and_publication_identity_are_checked(study):
    source, values = study
    result = prepare(source)
    name = next(name for name in values if name.endswith(".bin"))
    (source / name).write_bytes(b"\2" * 9)
    with pytest.raises(ValueError, match="changed before publication"):
        result.snapshot.recheck(time.monotonic() + 1)
    with pytest.raises(ValueError): prepare(source)


def test_missing_cached_mask_cannot_be_replaced_by_new_decode(study):
    source, values = study
    name = next(name for name in values if name.endswith(".bin"))
    (source / name).rename(source / "retained-test-mask.bin")
    with pytest.raises(FileNotFoundError): prepare(source)


def test_preparation_deadline_reserves_next_stage_without_new_qualification(study, monkeypatch):
    source, _ = study
    snapshot = load_equipment_support_snapshot(source, deadline=time.monotonic() + 5)
    with pytest.raises(ValueError, match="remaining time"):
        hand_snapshot.prepare_equipment_hand_snapshot(snapshot, deadline=time.monotonic() + 4)


def test_opt_in_workflow_uses_real_hand_selector_and_report(study, tmp_path, monkeypatch):
    from weft import equipment_support_workflow as workflow
    from pathlib import Path

    source, values = study
    brand = (Path(workflow.__file__).parent / "assets/brand.css").read_bytes()
    monkeypatch.setattr(workflow, "_runtime_sources", lambda deadline: {"assets/brand.css": brand})
    out = tmp_path / "comparison"
    result = workflow.prepare_equipment_support(source, out, fixed_camera=True, hand_points=True, max_seconds=25.)
    assert result["evidence_mode"] == "named_hand_points" and result["cached_binary_reads"] == 7
    assert result["raster_decodes"] == result["pose_model_calls"] == result["native_calls"] == result["api_calls"] == 0
    support = json.loads((out / "support-evidence.json").read_bytes())
    assert support["equipment_hand_constraint"]["profile"] == "equipment-hand-window-constraint-v1"
    assert "equipment_silhouette_constraint" not in support
    assert (out / "source/silhouette/frames.jsonl").read_bytes() == values["silhouette/frames.jsonl"]
    manifest = json.loads((out / "manifest.json").read_bytes())
    assert all(sha256((out / name).read_bytes()).hexdigest() == pin for name, pin in manifest["files"].items())
