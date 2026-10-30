import time

import pytest

from weft.source_equivalence import load_source_equivalence
from weft.shared_world_evidence import verify_shared_world_sources
from test_source_equivalence import bundle
from test_shared_world_evidence import bound, save_sources
from test_world_evidence import document


def test_bundle_remaining_budgets_include_its_manifest(bundle):
    checked = load_source_equivalence(bundle)
    for kwargs in ({"max_input_bytes": checked.total_input_bytes - 1},
                   {"max_files": len(checked.file_names) - 1},
                   {"deadline": time.monotonic() - 1}):
        with pytest.raises(ValueError, match="bound"):
            load_source_equivalence(bundle, **kwargs)
    assert load_source_equivalence(bundle, max_input_bytes=checked.total_input_bytes,
                                   max_files=len(checked.file_names)).manifest_sha256 == checked.manifest_sha256


@pytest.mark.parametrize("kwargs", [{"max_input_bytes": True}, {"max_input_bytes": 0},
                                    {"max_files": 1}, {"max_files": 2.5},
                                    {"deadline": float("nan")}, {"deadline": float("inf")}])
def test_invalid_budgets_never_disable_bounds(bundle, kwargs):
    with pytest.raises(ValueError):
        load_source_equivalence(bundle, **kwargs)


@pytest.mark.parametrize("name", [None, ".", "./evidence.json", "a//b.json", "a\\b.json", "a\nb.json",
                                 "/outside.json", "../outside.json", "source-mappings/document.json"])
def test_source_verifier_rejects_noncanonical_document_paths(tmp_path, name):
    scene, docs, _ = bound(document())
    save_sources(tmp_path, scene, docs)
    sha = next(iter(docs))
    scene["source_binding"]["evidence_documents"][sha] = name
    with pytest.raises(ValueError, match="safe relative"):
        verify_shared_world_sources(scene, tmp_path)


def test_source_verifier_rejects_symlinked_document_directory(tmp_path):
    scene, docs, _ = bound(document())
    original = tmp_path / "original"
    save_sources(original, scene, docs)
    alias = tmp_path / "alias"
    alias.mkdir()
    (alias / "evidence").symlink_to(original / "evidence", target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe"):
        verify_shared_world_sources(scene, alias)


def test_source_snapshot_is_immutable_and_keeps_exact_bytes(tmp_path):
    scene, docs, _ = bound(document())
    save_sources(tmp_path, scene, docs)
    report, files = verify_shared_world_sources(scene, tmp_path, include_files=True)
    sha, name = next(iter(scene["source_binding"]["evidence_documents"].items()))
    assert report["source_evidence_verified"] and files[name] == docs[sha]
    with pytest.raises(TypeError):
        files[name] = b"replacement"
    (tmp_path / name).write_bytes(b"changed after snapshot")
    assert files[name] == docs[sha]


@pytest.mark.parametrize("text", ["x" * 500_001, "🧵" * 100_000])
def test_scene_provenance_cannot_exceed_compact_metadata_budget(text):
    from weft.shared_world import validate_shared_world
    from weft.shared_world_lab import authored_shared_world_scene

    scene = authored_shared_world_scene()
    scene["objects"][0]["provenance"]["oversized_annotation"] = text
    with pytest.raises(ValueError, match="500 KB"):
        validate_shared_world(scene)
