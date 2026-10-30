from copy import deepcopy
import hashlib
import json

import pytest

from weft.io import digest, write_json
from weft.observations import adapt_exercise
from weft.pipeline import verify_artifacts
from weft.shared_world_evidence import build_shared_world_binding, verify_shared_world_sources
from weft.shared_world_lab import _analytic_mobile_human_spec, adapt_to_shared_world, prepare_shared_world_lab
from test_world_evidence import SOURCE, document, inputs


def bound(doc, *, refs=None):
    sha, raw, docs = inputs(doc)
    ids = ["human", "ball", "table"]
    refs = refs or {"human": (sha, "person"), "ball": (sha, "ball"), "table": None}
    binding, construction = build_shared_world_binding(docs, source_sha256=SOURCE, world_ids=ids, entity_refs=refs)
    scene = {"source_binding": binding, "objects": [{"id": "ball"}, {"id": "table"}]}
    return scene, docs, construction


def save_sources(root, scene, docs):
    for sha, relative in scene["source_binding"]["evidence_documents"].items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(docs[sha])


def test_binding_retains_complete_slots_and_admits_only_construction(tmp_path):
    scene, docs, construction = bound(document())
    rows = scene["source_binding"]["entity_bindings"]
    assert scene["source_binding"]["total_source_frames"] == 6
    assert [(r["observed_slots"], r["missing_slots"], r["pre_init_slots"]) for r in rows] == [(1, 5, 0), (1, 5, 0), (0, 6, 0)]
    assert rows[2]["observation_document_sha256"] is None and rows[2]["observation_entity_id"] is None
    assert b"SECRET" not in construction
    save_sources(tmp_path, scene, docs)
    checked = verify_shared_world_sources(scene, tmp_path)
    assert checked["source_evidence_verified"] is True
    assert checked["construction_sha256"] == hashlib.sha256(construction).hexdigest()
    assert checked["source_clocks"][0]["time"]["status"] == "assumed"


@pytest.mark.parametrize("field,value", [("observed_slots", 2), ("missing_slots", 4), ("observation_entity_id", "invented"),
                                        ("observation_document_sha256", "b" * 64), ("provenance_mode", "inferred_prior")])
def test_verifier_rejects_fabricated_binding_even_if_other_files_are_sealed(tmp_path, field, value):
    scene, docs, _ = bound(document())
    save_sources(tmp_path, scene, docs)
    scene["source_binding"]["entity_bindings"][0][field] = value
    with pytest.raises(ValueError):
        verify_shared_world_sources(scene, tmp_path)


def test_verifier_rejects_changed_exact_bytes_and_escaping_paths(tmp_path):
    scene, docs, _ = bound(document())
    save_sources(tmp_path, scene, docs)
    sha, relative = next(iter(scene["source_binding"]["evidence_documents"].items()))
    (tmp_path / relative).write_bytes(docs[sha] + b" ")
    with pytest.raises(ValueError, match="pinned hash"):
        verify_shared_world_sources(scene, tmp_path)
    scene["source_binding"]["evidence_documents"][sha] = "../outside.json"
    with pytest.raises(ValueError, match="safe relative"):
        verify_shared_world_sources(scene, tmp_path)


def test_repeated_bare_ids_are_qualified_by_document_and_other_sources_reject():
    first = document(partitions=("construction",) * 6)
    second = deepcopy(first)
    second["provenance"]["view"] = "second_model_same_video"
    a, _, docs_a = inputs(first)
    b, _, docs_b = inputs(second)
    binding, _ = build_shared_world_binding(docs_a | docs_b, source_sha256=SOURCE,
        world_ids=["human", "ball"], entity_refs={"human": (a, "person"), "ball": (b, "person")})
    assert len(binding["evidence_documents"]) == 2
    with pytest.raises(ValueError, match="reused"):
        build_shared_world_binding(docs_a, source_sha256=SOURCE, world_ids=["human", "ball"],
                                   entity_refs={"human": (a, "person"), "ball": (a, "person")})
    with pytest.raises(ValueError, match="same original video"):
        build_shared_world_binding(docs_a, source_sha256="b" * 64, world_ids=["human"], entity_refs={"human": (a, "person")})


def test_partial_intervals_and_unreferenced_documents_do_not_claim_complete_source():
    partial = document(version=2, first=10)
    sha, _, docs = inputs(partial)
    with pytest.raises(ValueError, match="complete original frame interval"):
        build_shared_world_binding(docs, source_sha256=SOURCE, world_ids=["human"], entity_refs={"human": (sha, "person")})
    first, second = document(), document(version=3)
    a, _, docs_a = inputs(first)
    _, _, docs_b = inputs(second)
    with pytest.raises(ValueError, match="unreferenced"):
        build_shared_world_binding(docs_a | docs_b, source_sha256=SOURCE, world_ids=["human"], entity_refs={"human": (a, "person")})


def test_native_and_assumed_clocks_remain_explicit_not_silently_aligned(tmp_path):
    scene, docs, _ = bound(document(version=3))
    save_sources(tmp_path, scene, docs)
    checked = verify_shared_world_sources(scene, tmp_path)
    assert checked["source_clocks"][0]["time"]["basis"] == "decoder_presentation_timestamps"
    assert "not implicitly aligned" in checked["scope"]


def test_adaptation_and_lab_pin_actual_evidence_not_pose_quality_counts(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    human = _analytic_mobile_human_spec()
    human["source_sha256"] = SOURCE
    evidence = {"profile": "exercise-pose-v1", "source_sha256": SOURCE, "width": 640, "height": 480,
                "fps": 30, "frame_count": 63, "decoded_frames": 61,
                "frames": [{"id": f"{SOURCE}:frame:{i}", "frame": i, "time_s": i / 30,
                            "status": "tracked", "landmarks": [{"x": .2, "y": .4, "z": .0}]} for i in range(0, 61, 2)]}
    for name, data in {"human.json": human, "scene.json": {"profile": "exercise-v3", "source_sha256": SOURCE},
                       "evidence.json": evidence, "pose-quality.json": {"usable_frames": 2, "unusable_frames": 0}}.items():
        write_json(source / name, data)
    write_json(source / "manifest.json", {"files": {p.name: digest(p) for p in source.iterdir()}})
    original = {p.name: p.read_bytes() for p in source.iterdir()}
    out = tmp_path / "adapted"
    adapt_to_shared_world(source, out)
    scene = json.loads((out / "scene.json").read_text())
    binding = scene["source_binding"]
    assert binding["total_source_frames"] == 61
    assert binding["entity_bindings"][0]["observed_slots"] == 31
    assert binding["entity_bindings"][0]["missing_slots"] == 30
    assert all(row["observation_entity_id"] is None for row in binding["entity_bindings"][1:])
    assert verify_shared_world_sources(scene, out)["source_evidence_verified"]
    lab = tmp_path / "lab"
    prepare_shared_world_lab(out, lab)
    assert verify_artifacts(lab / "run")["verified_files"] > 0
    assert verify_shared_world_sources(scene, lab / "run")["source_evidence_verified"]
    assert {p.name: p.read_bytes() for p in source.iterdir()} == original

    document_path = next(iter(binding["evidence_documents"].values()))
    changed = json.loads((out / document_path).read_text())
    changed["observations"][0]["visibility"] = "partial"
    write_json(out / document_path, changed)
    with pytest.raises(ValueError, match="integrity"):
        prepare_shared_world_lab(out, tmp_path / "tampered_lab")
    assert not (tmp_path / "tampered_lab").exists()
