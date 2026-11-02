"""Native-clock interchange against sealed recorded source-equivalence fixtures."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json

import pytest

from test_observations import exercise
from test_source_equivalence import fixture as equivalence_fixture
from weft.io import canonical_hash
from weft.observations import adapt_exercise
from weft.shared_world_evidence import build_shared_world_binding, verify_shared_world_sources
from weft.shared_world_reconstruction import _construction_index
from weft.source_clock import DETECTOR_CLOCK, PROFILE, derive_evidence
from weft.source_equivalence import load_source_equivalence


def raw(document):
    return (json.dumps(document, sort_keys=True) + "\n").encode()


def native_document(equivalence, *, derivative=False, native_shift=0, decoder_shift=0, tick_scale=1):
    source = equivalence.derivative_sha256 if derivative else equivalence.original_sha256
    evidence = exercise()
    evidence.update(schema_version=1, source_sha256=source, width=2, height=2,
        frame_count=99, decoded_frames=3,
        container_frame_audit={"encoded_video_frames": 3, "decoded_frames": 3, "matches": True},
        reference=[{"time_s": 0, "evidence_id": f"{source}:frame:0"}])
    rows = []
    for pair in equivalence.frames:
        pts = pair.derivative_native_pts if derivative else pair.source_native_pts
        decoder = pair.derivative_decoder_pts if derivative else pair.source_decoder_pts
        rows.append({"frame": pair.source_frame, "pts": (pts + native_shift) * tick_scale,
            "decoder_timestamp_s": float((decoder + decoder_shift) * Fraction(equivalence.time_base))})
    for sample in evidence["frames"]:
        sample["id"] = f"{source}:frame:{sample['frame']}"
        sample["decoder_timestamp_s"] = rows[sample["frame"]]["decoder_timestamp_s"]
    tick = Fraction(equivalence.time_base) / tick_scale
    binding = {"schema_version": 1, "profile": PROFILE, "source_sha256": source,
        "evidence_sha256": hashlib.sha256(raw(evidence)).hexdigest(),
        "evidence_canonical_sha256": canonical_hash(evidence), "metadata_frames": 99, "decoded_frames": 3,
        "nominal_fps": 30, "time_base": str(tick), "native_origin_pts": rows[0]["pts"],
        "decoder_minus_native_s": rows[0]["decoder_timestamp_s"] - float(rows[0]["pts"] * tick),
        "native_to_reconstruction_offset_s": -float(rows[0]["pts"] * tick),
        "capture_clock_status": "unknown", "detector_clock_status": DETECTOR_CLOCK, "rows": rows}
    return adapt_exercise(derive_evidence(evidence, binding), construction_frames=[0, 2])


def coupled(equivalence, *, original=None, derivative=None):
    documents, references = {}, {}
    for entity, document in (("human", original or native_document(equivalence)),
                             ("equipment", derivative or native_document(equivalence, derivative=True))):
        encoded = raw(document)
        sha = hashlib.sha256(encoded).hexdigest()
        documents[sha] = encoded
        references[entity] = (sha, "person-0")
    binding, construction = build_shared_world_binding(documents,
        source_sha256=equivalence.original_sha256, world_ids=list(references), entity_refs=references,
        source_equivalences=[equivalence])
    return binding, construction, documents


@pytest.fixture
def equivalence(tmp_path):
    return load_source_equivalence(equivalence_fixture(tmp_path / "mapping", derivative_offset=1))


def test_native_elapsed_slots_preserve_different_decoder_origins_and_metadata_counts(equivalence):
    binding, construction, _ = coupled(equivalence)
    sources = json.loads(construction)["sources"]
    by_media = {source["source"]["sha256"]: source for source in sources}
    original = by_media[equivalence.original_sha256]
    derivative = by_media[equivalence.derivative_sha256]
    assert binding["total_source_frames"] == 3
    assert len(binding["source_equivalences"]) == 1
    for source in sources:
        assert source["observation_schema_version"] == 4
        assert source["source"]["metadata_frames"] == 99
        assert source["source"]["frame_times_s"] == [0., .033, .067]
        assert [row["time_s"] for row in source["observations"]] == [0., .033, .067]
        assert source["observations"][1]["admission"] == "withheld"
        assert not source["observations"][1]["measurements"]
    assert original["source"]["native_clock"]["rows"][0]["decoder_timestamp_s"] == 0
    assert derivative["source"]["native_clock"]["rows"][0]["decoder_timestamp_s"] == .002
    assert original["source"]["native_clock_summary"] != derivative["source"]["native_clock_summary"]
    indexed, _ = _construction_index(construction, binding)
    assert len(indexed) == 6
    assert all(row["time_s"] == .067 for key, row in indexed.items() if key[2] == 2)


def test_scaled_native_tick_units_compare_exact_physical_time(equivalence):
    binding, construction, _ = coupled(equivalence, derivative=native_document(equivalence, derivative=True, tick_scale=2))
    assert binding["total_source_frames"] == 3
    assert len(_construction_index(construction, binding)[0]) == 6


@pytest.mark.parametrize("side", ["original", "derivative"])
@pytest.mark.parametrize("clock", ["native_shift", "decoder_shift"])
def test_internally_valid_but_foreign_clock_offsets_reject(equivalence, side, clock):
    document = native_document(equivalence, derivative=side == "derivative", **{clock: 1})
    with pytest.raises(ValueError, match="clock conflicts with verified source mapping"):
        coupled(equivalence, **{side: document})


def test_native_source_packaging_reverifies_exact_documents_and_equivalence(tmp_path, equivalence):
    binding, construction, documents = coupled(equivalence)
    root = tmp_path / "packaged"
    for sha, relative in binding["evidence_documents"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(documents[sha])
    for name in equivalence.file_names:
        path = root / "source-mappings" / equivalence.manifest_sha256 / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(equivalence.raw_bytes(name))
    scene = {"source_binding": binding, "objects": [{"id": "equipment"}]}
    report, files = verify_shared_world_sources(scene, root, include_files=True)
    assert report["source_evidence_verified"]
    assert report["construction_sha256"] == hashlib.sha256(construction).hexdigest()
    assert report["source_frames"] == 3
    assert all(row["time"]["basis"] == "native_pts_elapsed_time" for row in report["source_clocks"])
    assert all(files[relative] == documents[sha] for sha, relative in binding["evidence_documents"].items())


@pytest.mark.parametrize("change", ["nominal_row_time", "missing_slot", "duplicate_slot", "evidence_time"])
def test_constructor_index_rejects_timeline_or_slot_corruption(equivalence, change):
    binding, construction, _ = coupled(equivalence)
    modified = json.loads(construction)
    source = modified["sources"][0]
    if change == "nominal_row_time": source["observations"][2]["time_s"] = 2 / 30
    elif change == "missing_slot": source["observations"].pop()
    elif change == "duplicate_slot": source["observations"].append(deepcopy(source["observations"][0]))
    else: source["evidence"][0]["time_s"] = .003
    with pytest.raises(ValueError):
        _construction_index(raw(modified), binding)
