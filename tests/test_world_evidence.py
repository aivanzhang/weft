from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
import json

import pytest

from weft.observations import COORDINATES, validate_observations
from weft import world_evidence
from weft.world_evidence import load_world_evidence


SOURCE = "a" * 64
PARTITIONS = ("construction", "generation", "development_validation", "audit", "unassigned", "unsampled")


def document(*, version=1, first=0, partitions=PARTITIONS):
    count = len(partitions)
    times = [0., .031, .072, .109, .143, .179][:count]
    source = {"sha256": SOURCE, "width_px": 100, "height_px": 80, "fps": 30,
              "metadata_frames": first + count, "decoded_frames": count}
    time = {"unit": "seconds", "origin": "video_start", "basis": "frame_index_divided_by_fps", "status": "assumed"}
    if version == 2:
        source.update(decode_interval=[first, first + count - 1], source_complete=False, prefix_advanced_frames=first)
    if version == 3:
        source.pop("fps")
        source["frame_times_s"] = times
        time = {"unit": "seconds", "origin": "submitted_video_start", "basis": "decoder_presentation_timestamps", "status": "observed"}
    unknown = {"value": None, "status": "unknown", "evidence_ids": []}
    result = {"schema_version": version, "profile": "scene-observations-v1", "source": source, "time": time,
              "image": {"width_px": 100, "height_px": 80}, "coordinates": deepcopy(COORDINATES),
              "camera": {"motion": deepcopy(unknown), "calibration": deepcopy(unknown), "screen": {"hidden": "SCREEN_SECRET"}},
              "objects": [{"id": obj, "label": "LABEL_SECRET", "properties": {"kind": deepcopy(unknown), "role": deepcopy(unknown)},
                           "unknowns": ["UNKNOWN_TEXT_SECRET"]} for obj in ("person", "ball")],
              "evidence": [], "observations": [], "relationships": [],
              "provenance": {"scene": {"hidden": "PROVENANCE_SECRET"}}, "limits": ["LIMIT_SECRET"]}
    for offset, partition in enumerate(partitions):
        frame, stamp = first + offset, times[offset] if version == 3 else (first + offset) / 30
        for obj in ("person", "ball"):
            evidence_id = f"{SOURCE}:frame:{frame}:object:{obj}"
            result["evidence"].append({"id": evidence_id, "frame": frame, "time_s": stamp,
                                       "object_id": obj, "kind": "centroid", "status": "inferred",
                                       "payload": {"secret": f"{partition}_PAYLOAD_SECRET"}})
            missing = partition == "unsampled"
            result["observations"].append({"id": f"{SOURCE}:observation:{frame}:{obj}", "frame": frame,
                "time_s": stamp, "object_id": obj, "visibility": "not_sampled" if missing else "detected",
                "partition": partition, "evidence_ids": [evidence_id],
                "measurements": {} if missing else {"center": {"value": [offset + 1, 2], "coordinate_system": "image", "unit": "pixels", "status": "inferred"}}})
    return result


def inputs(doc):
    raw = json.dumps(doc, indent=1).encode()
    digest = hashlib.sha256(raw).hexdigest()
    return digest, raw, {digest: raw}


def export(doc, **kwargs):
    digest, raw, sources = inputs(doc)
    bundle = load_world_evidence(sources)
    return bundle, json.loads(bundle.construction_bytes(**kwargs)), digest, raw


def test_default_export_preserves_every_slot_without_leaking_excluded_payloads():
    doc = document()
    bundle, result, digest, raw = export(doc)
    source = result["sources"][0]
    assert bundle.source_bytes(digest) == raw
    assert result["physical_validation"] == "unverified"
    assert len(source["observations"]) == 12
    assert [row["admission"] for row in source["observations"]] == ["included"] * 2 + ["withheld"] * 10
    for original, row in zip(doc["observations"], source["observations"]):
        assert all(row[key] == original[key] for key in ("id", "frame", "time_s", "object_id", "partition"))
        assert row["entity_ref"] == {"document_sha256": digest, "object_id": original["object_id"]}
        if row["admission"] == "withheld":
            assert row["measurements"] == {} and row["evidence_refs"] == []
            assert row["visibility"] is None
        else:
            assert row["visibility"] == original["visibility"]
    assert len(source["evidence"]) == 2
    assert source["observations"][0]["measurements"] == doc["observations"][0]["measurements"]
    assert "SECRET" not in bundle.construction_bytes().decode()
    assert all("payload" not in item and item["raw_payload_omitted"] for item in source["evidence"])


def test_explicit_partition_access_never_admits_audit():
    bundle, result, _, _ = export(document(), allowed_partitions=("generation", "development_validation"))
    rows = result["sources"][0]["observations"]
    assert [row["partition"] for row in rows if row["admission"] == "included"] == ["generation"] * 2 + ["development_validation"] * 2
    assert all(row["admission"] == "withheld" for row in rows if row["partition"] == "audit")
    assert all(row["admission"] == "withheld" for row in json.loads(bundle.construction_bytes(allowed_partitions=()))["sources"][0]["observations"])
    for invalid in (("audit",), ("unassigned",), ("unsampled",), ("construction", "construction"), "construction", (None,)):
        with pytest.raises(ValueError, match="partitions|exported"):
            bundle.construction_bytes(allowed_partitions=invalid)


def test_claim_and_relationship_filtering_requires_all_cited_evidence():
    doc = document()
    first, second, audit = (doc["evidence"][index]["id"] for index in (0, 1, 6))
    doc["objects"][0]["properties"].update(
        safe={"value": "person", "status": "inferred", "evidence_ids": [first]},
        mixed={"value": "MIXED_SECRET", "status": "inferred", "evidence_ids": [first, audit]},
        prior={"value": "PRIOR_SECRET", "status": "assumed", "evidence_ids": []},
        blob={"value": {"hidden": "COMPOUND_SECRET"}, "status": "inferred", "evidence_ids": [first]})
    doc["camera"]["calibration"] = {"value": "CAMERA_SECRET", "status": "inferred", "evidence_ids": [audit]}
    for label, refs in (("allowed", [first, second]), ("excluded", [first, audit])):
        doc["relationships"].append({"id": label, "type": "visual_proximity_candidate", "objects": ["person", "ball"],
                                    "status": "inferred", "evidence_ids": refs, "interval_s": [0., 0.],
                                    "attributes": {"hidden": "RELATIONSHIP_SECRET"}})
    bundle, result, digest, _ = export(doc)
    source = result["sources"][0]
    properties = source["objects"][0]["properties"]
    assert properties["safe"]["value"] == "person"
    assert properties["safe"]["evidence_refs"] == [{"document_sha256": digest, "evidence_id": first}]
    assert all(properties[key]["value"] is None and properties[key]["admission"] == "withheld" for key in ("mixed", "prior", "blob"))
    assert properties["kind"]["status"] == "unknown"
    assert source["camera"]["calibration"]["admission"] == "withheld"
    assert [row["id"] for row in source["relationships"]] == ["allowed"]
    assert source["relationships_filtered"] is True
    assert "withheld_relationship_count" not in source
    assert "SECRET" not in bundle.construction_bytes().decode()


def test_relationship_interval_cannot_expose_excluded_frames_despite_admitted_references():
    doc = document()
    refs = [row["id"] for row in doc["evidence"][:2]]
    for name, interval in (("opening", [0., 0.]), ("later", [0., 1 / 30]), ("unbounded", None)):
        doc["relationships"].append({"id": name, "type": "visual_proximity_candidate", "objects": ["person", "ball"],
                                    "status": "inferred", "evidence_ids": refs, "interval_s": interval, "attributes": {}})
    _, result, _, _ = export(doc)
    assert [row["id"] for row in result["sources"][0]["relationships"]] == ["opening"]
    _, expanded, _, _ = export(doc, allowed_partitions=("construction", "generation"))
    assert [row["id"] for row in expanded["sources"][0]["relationships"]] == ["opening", "later"]


def test_audit_visibility_and_hidden_relationship_count_remain_inspection_only():
    doc = document()
    for row in doc["observations"]:
        if row["partition"] == "audit":
            row.update(visibility="unknown", measurements={})
    audit = doc["evidence"][6]["id"]
    for index in range(7):
        doc["relationships"].append({"id": f"hidden-{index}", "type": "visual_proximity_candidate",
                                    "objects": ["person", "ball"], "status": "inferred", "evidence_ids": [audit],
                                    "interval_s": [.1, .1], "attributes": {}})
    bundle, result, digest, _ = export(doc)
    source = result["sources"][0]
    assert [row["visibility"] for row in source["observations"] if row["partition"] == "audit"] == [None, None]
    assert source["relationships"] == [] and source["relationships_filtered"] is True
    assert "withheld_relationship_count" not in source
    assert "hidden-" not in bundle.construction_bytes().decode()
    assert bundle.summary()["sources"][0]["visibility_counts"]["unknown"] == 2
    original = json.loads(bundle.source_bytes(digest))
    assert original["observations"][6]["visibility"] == "unknown"
    assert len(original["relationships"]) == 7


@pytest.mark.parametrize("version,first", [(1, 0), (2, 10000), (3, 0)])
def test_complete_native_clock_and_absolute_interval_coverage(version, first):
    doc = document(version=version, first=first)
    bundle, result, _, _ = export(doc)
    summary = bundle.summary()["sources"][0]
    assert summary["decoded_frame_interval"] == [first, first + 5]
    assert summary["slots"] == 12
    assert result["sources"][0]["source"] == doc["source"]
    assert result["sources"][0]["time"] == doc["time"]
    assert result["sources"][0]["observations"][4]["time_s"] == doc["observations"][4]["time_s"]
    doc["observations"].pop(4)
    validate_observations(doc)
    with pytest.raises(ValueError, match="Incomplete decoded"):
        load_world_evidence(inputs(doc)[2])


def test_same_local_entity_ids_and_video_hash_do_not_merge_documents():
    left, right = document(), document()
    right["observations"][0]["measurements"]["center"]["value"][0] = 77
    first, _, sources = inputs(left)
    second, raw, _ = inputs(right)
    sources[second] = raw
    bundle = load_world_evidence(sources)
    assert set(bundle.entity_keys) == {(digest, obj) for digest in (first, second) for obj in ("person", "ball")}
    assert bundle.summary()["total_slots"] == 24
    result = json.loads(bundle.construction_bytes())
    assert {row["document_sha256"] for row in result["sources"]} == {first, second}
    assert [source["source"]["sha256"] for source in result["sources"]] == [SOURCE, SOURCE]


def test_registry_immutability_and_summary_mutation_cannot_change_exports():
    doc = document()
    bundle, _, digest, raw = export(doc)
    original = bundle.construction_bytes()
    doc["source"]["sha256"] = "b" * 64
    snapshot = bundle.summary()
    snapshot["sources"][0]["source"]["sha256"] = "c" * 64
    with pytest.raises(FrozenInstanceError):
        bundle._sources = ()
    with pytest.raises(TypeError):
        bundle._sources[0].document["source"]["sha256"] = "d" * 64
    assert bundle.summary()["sources"][0]["source"]["sha256"] == SOURCE
    assert bundle.source_bytes(digest) is raw
    assert bundle.construction_bytes() == original
    with pytest.raises(KeyError):
        bundle.source_bytes("f" * 64)


@pytest.mark.parametrize("raw", [
    b'{"x": 1, "x": 2}', b'{"x": {"y": 1, "y": 2}}', b'{"x": NaN}',
    b'{"x": Infinity}', b'{"x": -Infinity}', b'{"x": 1e999}', b'\xff', b'{', b'[]', b'null',
])
def test_duplicate_keys_nonfinite_and_malformed_json_rejected(raw):
    with pytest.raises(ValueError):
        load_world_evidence({hashlib.sha256(raw).hexdigest(): raw})


def test_input_digest_type_size_count_and_tree_bounds(monkeypatch):
    digest, raw, source = inputs(document())
    for invalid in ({}, [], {"x": raw}, {"b" * 64: raw}, {digest: bytearray(raw)}):
        with pytest.raises(ValueError):
            load_world_evidence(invalid)
    with pytest.raises(ValueError, match="1–8"):
        load_world_evidence({f"{index:064x}": b"{}" for index in range(9)})
    monkeypatch.setattr(world_evidence, "MAX_INPUT_BYTES", len(raw) - 1)
    with pytest.raises(ValueError, match="64 MiB"):
        load_world_evidence(source)
    monkeypatch.setattr(world_evidence, "MAX_INPUT_BYTES", 64 * 1024 * 1024)
    nested = b'[' * 66 + b'0' + b']' * 66
    with pytest.raises(ValueError, match="node/depth"):
        load_world_evidence({hashlib.sha256(nested).hexdigest(): nested})
    monkeypatch.setattr(world_evidence, "MAX_JSON_NODES", 5)
    with pytest.raises(ValueError, match="node/depth"):
        load_world_evidence(source)


def test_missing_observations_remain_missing_when_partition_is_admitted():
    doc = document(partitions=("construction", "construction"))
    for row, visibility in zip(doc["observations"], ("unknown", "not_detected", "not_sampled", "partial")):
        row["visibility"] = visibility
        if visibility != "partial":
            row["measurements"] = {}
    _, result, _, _ = export(doc)
    rows = result["sources"][0]["observations"]
    assert [row["visibility"] for row in rows] == ["unknown", "not_detected", "not_sampled", "partial"]
    assert all(row["admission"] == "included" for row in rows)
    assert all(not row["measurements"] for row in rows[:3])


def test_source_clock_and_foreign_evidence_are_still_validated():
    for change in (lambda d: d["observations"][0].update(time_s=.01),
                   lambda d: d["observations"][0].update(evidence_ids=["foreign"])):
        doc = document(version=3)
        change(doc)
        with pytest.raises(ValueError):
            load_world_evidence(inputs(doc)[2])
