"""Restore complete saved diagnostics without repeating the raster/model producer."""

from copy import deepcopy
from functools import lru_cache
from hashlib import sha256
import importlib
import json
import time

import pytest

from weft import equipment_support_snapshot as snapshot
from test_equipment_silhouette_evidence import fixture as silhouette_fixture
from test_exercise_support_evidence import pose


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()


@lru_cache(maxsize=1)
def analytical_files():
    api = importlib.import_module("weft.equipment_silhouette_evidence")
    data = pose(count=12, fps=20)
    selected = [row["frame"] for row in data["frames"][:7]]
    def pose_change(value):
        for row in value["frames"]:
            row["decoder_timestamp_s"] = row["time_s"]
    def observation_change(value):
        for row in value["observations"]:
            if row["frame"] not in selected:
                row["partition"] = "unassigned"
    annotations, args, _, _ = silhouette_fixture(api, supplied_pose=data, pose_change=pose_change,
        observation_change=observation_change, raster=b"\1" * 9)
    prepared = api.prepare_equipment_silhouette_evidence(annotations, construction_frames=selected, **args)
    return {
        "inputs/context.json": annotations.pose_document,
        "inputs/equipment.json": prepared.equipment_document,
        "inputs/masks.json": prepared.mask_document,
        "human-observations.json": annotations.input_documents[0][1],
        "source-annotations.json": annotations.annotation_document(),
        "annotation-metadata.json": encoded(annotations.describe()),
        "annotation-frames.jsonl": b"".join(encoded(annotations.frame(i)) for i in range(12)),
        "silhouette/metadata.json": encoded(prepared.describe()),
        "silhouette/frames.jsonl": b"".join(encoded(prepared.frame(i)) for i in range(12)),
    }


def seal_files(source, values):
    source.mkdir(exist_ok=True)
    for name, raw in values.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    seal = {"schema_version": 1, "profile": "pose-equipment-silhouette-composition-v1", "status": "completed",
            "files": {name: sha256(raw).hexdigest() for name, raw in values.items()}}
    (source / "manifest.json").write_bytes(encoded(seal))
    return seal


@pytest.fixture
def study(tmp_path):
    source = tmp_path / "source"
    values = dict(analytical_files())
    seal_files(source, values)
    return source, values


def test_restore_exact_pins_caches_and_complete_slots_without_producer(study, monkeypatch):
    source, values = study
    api = importlib.import_module("weft.equipment_silhouette_evidence")
    def denied(*args, **kwargs):
        pytest.fail("Saved restoration cannot prepare new image relations")
    monkeypatch.setattr(api, "prepare_equipment_silhouette_evidence", denied)
    monkeypatch.setattr(api, "_runs", denied)
    monkeypatch.setattr(api, "_relation", denied)
    result = snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() + 5)
    assert result.equipment_ids == ("sam-2",)
    assert result.silhouette.source_equivalence is None
    assert result.silhouette.pose_annotations.precursor_custody is None
    assert len(result.silhouette._rows) == 12
    assert result.silhouette._metadata + b"\n" == values["silhouette/metadata.json"]
    assert b"\n".join(result.silhouette._rows) + b"\n" == values["silhouette/frames.jsonl"]
    assert result.source_files == {"manifest.json": (source / "manifest.json").read_bytes(), **values}
    assert result.read_work["selected_files"] == 10 and result.read_work["proof_files"] == 0
    assert result.read_work["selected_bytes"] == sum(map(len, result.source_files.values()))
    assert "not freshly recomputed" in result.read_work["scope"]
    result.recheck(time.monotonic() + 1)


@pytest.mark.parametrize("fault", ["raw_pin", "canonical_pose", "clock", "selection", "source_mapping",
    "annotation_id", "annotation_counts", "silhouette_counts", "excluded_pixel", "missing_slot", "duplicated_pair"])
def test_resealed_but_incompatible_saved_join_rejected(study, fault):
    source, values = study
    meta = json.loads(values["silhouette/metadata.json"])
    annotation_meta = json.loads(values["annotation-metadata.json"])
    rows = [json.loads(raw) for raw in values["silhouette/frames.jsonl"].splitlines()]
    annotations = [json.loads(raw) for raw in values["annotation-frames.jsonl"].splitlines()]
    if fault == "raw_pin": meta["mask_document_sha256"] = "f" * 64
    if fault == "canonical_pose":
        meta["pose_provenance"]["pose_canonical_sha256"] = "f" * 64
        annotation_meta["provenance"] = deepcopy(meta["pose_provenance"])
    if fault == "clock": meta["source_clocks"] = {}
    if fault == "selection": meta["construction_frames"] = [0]
    if fault == "source_mapping": rows[1]["equipment_source_frame"] = 9
    if fault == "annotation_id": annotations[0]["pose_evidence_id"] = "wrong"
    if fault == "annotation_counts": annotation_meta["counts"]["copied_estimates"] += 1
    if fault == "silhouette_counts": meta["counts"]["pair_slots"] += 1
    if fault == "excluded_pixel": annotations[-1]["channels"][0]["pixel"] = [1., 2.]
    if fault == "missing_slot": rows.pop()
    if fault == "duplicated_pair": rows[0]["pairs"][1] = deepcopy(rows[0]["pairs"][0])
    values.update({"silhouette/metadata.json": encoded(meta), "annotation-metadata.json": encoded(annotation_meta),
                   "silhouette/frames.jsonl": b"".join(map(encoded, rows)),
                   "annotation-frames.jsonl": b"".join(map(encoded, annotations))})
    seal_files(source, values)
    with pytest.raises(ValueError):
        snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() + 5)


@pytest.mark.parametrize("fault", ["wrong_seal", "unsealed", "changed_member", "missing_member", "wrong_profile", "duplicate_key",
                                  "missing_source_id", "invalid_source_id"])
def test_bad_archive_rejected_before_restoration(study, monkeypatch, fault):
    source, _ = study
    def denied(*args): pytest.fail("Invalid archive reached source restoration")
    monkeypatch.setattr(snapshot, "_restore", denied)
    seal = json.loads((source / "manifest.json").read_bytes())
    options = {}
    if fault == "wrong_seal": options["expected_manifest_sha256"] = "f" * 64
    if fault == "unsealed": seal["status"] = "running"
    if fault == "changed_member": (source / "inputs/context.json").write_bytes(b"{}")
    if fault == "missing_member": del seal["files"]["source-annotations.json"]
    if fault == "wrong_profile": seal["profile"] = "unrelated"
    if fault in {"missing_source_id", "invalid_source_id"}:
        meta = json.loads((source / "silhouette/metadata.json").read_bytes())
        if fault == "missing_source_id": del meta["source_sha256"]
        else: meta["source_sha256"] = 1
        metadata_raw = encoded(meta)
        (source / "silhouette/metadata.json").write_bytes(metadata_raw)
        seal["files"]["silhouette/metadata.json"] = sha256(metadata_raw).hexdigest()
    raw = encoded(seal)
    if fault == "duplicate_key": raw = b'{"schema_version":1,' + raw[1:]
    (source / "manifest.json").write_bytes(raw)
    with pytest.raises(ValueError):
        snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() + 5, **options)


def test_expired_budget_and_changed_input_are_explicit(study):
    source, _ = study
    with pytest.raises(TimeoutError):
        snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() - 1)
    result = snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() + 5)
    (source / "inputs/context.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="changed before publication"):
        result.recheck(time.monotonic() + 1)


def test_symlink_and_input_byte_bound_reject_without_restoration(study, tmp_path, monkeypatch):
    source, _ = study
    link = tmp_path / "alias"
    link.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        snapshot.load_equipment_support_snapshot(link, deadline=time.monotonic() + 5)
    monkeypatch.setattr(snapshot, "MAX_INPUT", 100)
    with pytest.raises(ValueError):
        snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() + 5)
