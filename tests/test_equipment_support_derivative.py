"""Recorded derivative-proof restoration without decoding or fabricated proof objects."""

from hashlib import sha256
import importlib
import json
import time

import pytest

from weft import equipment_support_snapshot as snapshot
from test_equipment_silhouette_evidence import fixture as silhouette_fixture
from test_equipment_support_snapshot import encoded, seal_files
from test_exercise_support_evidence import _equivalent_equipment


@pytest.fixture
def derivative(tmp_path):
    data, _, proof = _equivalent_equipment(tmp_path)
    selected = [data["frames"][0]["frame"]]
    def restrict(value):
        for row in value["observations"]:
            if row["frame"] not in selected: row["partition"] = "unassigned"
    api = importlib.import_module("weft.equipment_silhouette_evidence")
    annotations, args, _, _ = silhouette_fixture(api, supplied_pose=data, proof=proof,
        observation_change=restrict, raster=b"\1" * 9)
    prepared = api.prepare_equipment_silhouette_evidence(annotations, construction_frames=selected, **args)
    values = {
        "inputs/context.json": annotations.pose_document,
        "inputs/equipment.json": prepared.equipment_document,
        "inputs/masks.json": prepared.mask_document,
        "human-observations.json": annotations.input_documents[0][1],
        "source-annotations.json": annotations.annotation_document(),
        "annotation-metadata.json": encoded(annotations.describe()),
        "annotation-frames.jsonl": b"".join(encoded(annotations.frame(i)) for i in range(3)),
        "silhouette/metadata.json": encoded(prepared.describe()),
        "silhouette/frames.jsonl": b"".join(encoded(prepared.frame(i)) for i in range(3)),
        **{"source-equivalence/" + name: proof.raw_bytes(name) for name in proof.file_names},
    }
    source = tmp_path / "source"
    seal_files(source, values)
    return source, values, proof


def test_actual_recorded_proof_loader_runs_once_and_preserves_both_clocks(derivative, monkeypatch):
    source, values, proof = derivative
    api = importlib.import_module("weft.source_equivalence")
    original = api.load_source_equivalence
    calls = []
    def load(*args, **kwargs):
        calls.append((args, kwargs)); return original(*args, **kwargs)
    monkeypatch.setattr(api, "load_source_equivalence", load)
    restored = snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() + 5)
    assert len(calls) == 1
    assert type(restored.silhouette.source_equivalence) is api.SourceEquivalence
    assert restored.silhouette.source_equivalence.inspection() == proof.inspection()
    assert restored.read_work["proof_files"] == len(proof.file_names)
    assert restored.read_work["selected_bytes"] == sum(map(len, restored.source_files.values()))
    assert set(restored.source_identities) == set(restored.source_files)
    assert restored.silhouette.frame(0)["source_mapping"]["original_decoder_time_s"] != restored.silhouette.frame(0)["source_mapping"]["derivative_decoder_time_s"]
    assert all(restored.source_files[name] == raw for name, raw in values.items())


def test_outer_proof_pin_and_saved_clock_summary_must_both_match(derivative):
    source, values, _ = derivative
    meta = json.loads(values["silhouette/metadata.json"])
    meta["source_equivalence"]["source_clock"]["native_minus_decoder_pts"] += 1
    values["silhouette/metadata.json"] = encoded(meta)
    seal_files(source, values)
    with pytest.raises(ValueError):
        snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() + 5)


def test_separate_outer_seal_cannot_replace_a_verified_proof_pin(derivative):
    source, _, _ = derivative
    seal = json.loads((source / "manifest.json").read_bytes())
    seal["files"]["source-equivalence/report.json"] = "f" * 64
    (source / "manifest.json").write_bytes(encoded(seal))
    with pytest.raises(ValueError, match="differs from the study seal"):
        snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() + 5)


def test_late_proof_mutation_is_included_in_publication_recheck(derivative):
    source, _, _ = derivative
    restored = snapshot.load_equipment_support_snapshot(source, deadline=time.monotonic() + 5)
    (source / "source-equivalence/report.json").write_bytes(b"changed after snapshot")
    with pytest.raises(ValueError, match="changed before publication"):
        restored.recheck(time.monotonic() + 1)
