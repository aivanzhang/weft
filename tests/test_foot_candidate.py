"""Source ownership and complete candidate preparation from sealed local inputs."""

import json

import pytest

from weft.foot_candidate import build_candidate
from weft.human import build_human
from weft.io import digest, write_json
from weft.pipeline import verify_artifacts
from test_foot_reference import fixture as foot_fixture
from test_mobile_integration import evidence


@pytest.fixture
def run(tmp_path):
    folder = tmp_path / "run"
    spec = build_human(evidence(), folder, body_model="mobile")
    _, raw, metadata = foot_fixture()
    # The fixture tests provenance plumbing, not video decoding or physical accuracy.
    (folder / "source.mp4").write_bytes(b"explicit synthetic source fixture")
    source = digest(folder / "source.mp4")
    text = json.dumps(spec).replace("a" * 64, source)
    write_json(folder / "human.json", json.loads(text))
    raw = json.loads(json.dumps(raw).replace("a" * 64, source))
    write_json(folder / "pose-landmarks.json", raw)
    write_json(folder / "evidence.json", metadata | {"source_sha256": source})
    write_json(folder / "source.json", {"artifact": "source.mp4", "sha256": source})
    write_json(folder / "manifest.json", {"files": {p.name: digest(p) for p in folder.iterdir() if p.is_file()}})
    return folder


@pytest.mark.parametrize("refine,profile", [(False, "kinematic-foot-reference-v1"), (True, "image-directed-foot-reference-v1")])
def test_build_keeps_source_unchanged_and_seals_complete_candidate(run, tmp_path, refine, profile):
    before = {p.name: digest(p) for p in run.iterdir() if p.is_file()}
    output = tmp_path / "candidate"
    result = build_candidate(run, output, image_directions=refine)
    assert result["interactive_start_available"] is True and result["accepted"] is False
    assert result["paid_requests"] == 0
    assert json.loads((output / "foot-reference.json").read_text())["profile"] == profile
    assert json.loads((output / "human.json").read_text()) == json.loads((run / "human.json").read_text())
    assert (output / "model.xml").is_file()
    assert (output / "implementation-source/foot_candidate.py").is_file()
    assert verify_artifacts(output) == result["verification"]
    assert before == {p.name: digest(p) for p in run.iterdir() if p.is_file()}


def test_rejects_input_mutation_and_output_inside_source(run, tmp_path):
    with pytest.raises(ValueError, match="outside"):
        build_candidate(run, run / "candidate")
    (run / "human.json").write_text("{}")
    with pytest.raises(ValueError):
        build_candidate(run, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()


def test_rejects_unsealed_replacement_input(run, tmp_path):
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["files"].pop("pose-landmarks.json")
    write_json(run / "manifest.json", manifest)
    with pytest.raises(ValueError, match="sealed"):
        build_candidate(run, tmp_path / "candidate")


def test_does_not_overwrite_existing_output(run, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep").write_text("saved evidence")
    with pytest.raises(ValueError, match="new candidate"):
        build_candidate(run, output)
    assert (output / "keep").read_text() == "saved evidence"


@pytest.mark.parametrize("filename", ["pose-landmarks.json", "manifest.json"])
def test_source_changes_after_verification_cannot_be_published(run, tmp_path, monkeypatch, filename):
    from weft import foot_candidate
    original = foot_candidate.verify_artifacts
    def change_after_verification(path):
        result = original(path)
        (run / filename).write_text("{}")
        return result
    monkeypatch.setattr(foot_candidate, "verify_artifacts", change_after_verification)
    with pytest.raises(ValueError, match="changed after verification"):
        build_candidate(run, tmp_path / "candidate")
    assert not (tmp_path / "candidate").exists()
