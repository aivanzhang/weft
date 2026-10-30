import hashlib
import json

import pytest

from weft.io import digest, write_json
from weft.shared_world_packaging import collect_shared_source_files
from weft.shared_world_pipeline import construct_source_equipment_run
from weft.shared_world_reconstruction import MAX_PROPOSAL_BYTES
from test_shared_world_reconstruction import fixture, raw


def seal(directory):
    write_json(directory / "manifest.json", {"files": {path.relative_to(directory).as_posix(): digest(path)
        for path in directory.rglob("*") if path.is_file() and path != directory / "manifest.json"}})


def inputs(root):
    scene, proposal, kwargs, documents = fixture(include_documents=True)
    base, geometry, evidence = [root / name for name in ("base", "geometry", "evidence")]
    for directory in (base, geometry, evidence):
        directory.mkdir()
    for sha, name in scene["source_binding"]["evidence_documents"].items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(documents[sha])
    (base / "human.json").write_bytes(kwargs["human_document"])
    write_json(base / "scene.json", scene)
    write_json(base / "source.json", {"sha256": "f" * 64, "decoded_frames": 1, "attribution": "Preserve without promoting metadata identity"})
    (geometry / "geometry-proposal.json").write_bytes(raw(proposal))
    (evidence / "evidence").mkdir()
    (evidence / "evidence/scene-observations.json").write_bytes(next(iter(documents.values())))
    for directory in (base, geometry, evidence):
        seal(directory)
    return base, geometry, evidence, scene


def test_pipeline_keeps_sources_and_generates_canonical_display_metadata(tmp_path):
    base, geometry, evidence, original = inputs(tmp_path)
    before = {path: path.read_bytes() for directory in (base, geometry, evidence) for path in directory.rglob("*") if path.is_file()}
    out = tmp_path / "constructed"
    result = construct_source_equipment_run(base, geometry, evidence, out, target_object_id="equipment-a")
    scene = json.loads((out / "scene.json").read_text())
    checked = collect_shared_source_files(scene, out)
    assert checked.report["source_evidence_verified"] and result["source_documents"] == 1
    assert scene["human_spec"] == original["human_spec"] and scene["gravity"] == original["gravity"]
    metadata = json.loads((out / "source.json").read_text())
    assert metadata["sha256"] == original["source_binding"]["source_sha256"]
    assert metadata["decoded_frames"] == 63
    assert (out / "inputs/base-source-metadata.json").read_bytes() == (base / "source.json").read_bytes()
    assert all(path.read_bytes() == value for path, value in before.items())


@pytest.mark.parametrize("invalid", [b'{"schema_version":1,"schema_version":1}', b'[]', b'{"equipment":null}',
                                     b'{"padding":"' + b'x' * MAX_PROPOSAL_BYTES + b'"}'])
def test_pipeline_rejects_bad_or_oversized_proposals_before_output(tmp_path, invalid):
    base, geometry, evidence, _ = inputs(tmp_path)
    (geometry / "geometry-proposal.json").write_bytes(invalid)
    seal(geometry)
    out = tmp_path / "rejected"
    with pytest.raises(ValueError):
        construct_source_equipment_run(base, geometry, evidence, out, target_object_id="equipment-a")
    assert not out.exists()


def test_complete_human_reference_document_has_its_own_larger_bound(tmp_path):
    base, geometry, evidence, _ = inputs(tmp_path)
    human = (base / "human.json").read_bytes() + b" " * MAX_PROPOSAL_BYTES
    (base / "human.json").write_bytes(human)
    proposal = json.loads((geometry / "geometry-proposal.json").read_text())
    proposal["body_binding"]["human_json_sha256"] = hashlib.sha256(human).hexdigest()
    write_json(geometry / "geometry-proposal.json", proposal)
    seal(base)
    seal(geometry)
    out = tmp_path / "larger-body"
    construct_source_equipment_run(base, geometry, evidence, out, target_object_id="equipment-a")
    assert (out / "inputs/human.json").read_bytes() == human
