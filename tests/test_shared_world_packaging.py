"""Source packages remain self-contained and cannot overwrite unrelated artifacts."""

import hashlib
import json

import pytest

from weft.io import digest, write_json
from weft.pipeline import verify_artifacts
from weft.shared_world_evidence import build_shared_world_binding, verify_shared_world_sources
from weft.shared_world_lab import authored_shared_world_scene, prepare_shared_world_lab
from weft.shared_world_learning import train_shared_world_policy
from weft.shared_world_packaging import (
    collect_shared_source_files, copy_package_file, read_package_file, read_package_manifest, write_package_files,
)
from weft.source_equivalence import load_source_equivalence
from test_source_equivalence import fixture as mapping_fixture
from test_world_evidence import SOURCE, document


def seal(root):
    write_json(root / "manifest.json", {"files": {
        p.relative_to(root).as_posix(): digest(p) for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"}})


def candidate(root, *, equivalent=False, observation_path=None):
    root.mkdir()
    equivalences = []
    source_sha = SOURCE
    doc = document()
    if equivalent:
        item = load_source_equivalence(mapping_fixture(root.parent / (root.name + "-mapping")))
        equivalences = [item]
        source_sha = item.original_sha256
        doc = json.loads(json.dumps(document(version=3, partitions=("construction",) * 3))
                         .replace(SOURCE, item.derivative_sha256))
        doc["source"].update(width_px=2, height_px=2, frame_times_s=[0, .033, .067])
        doc["image"] = {"width_px": 2, "height_px": 2}
        for row in doc["observations"] + doc["evidence"]:
            row["time_s"] = doc["source"]["frame_times_s"][row["frame"]]
        for row in doc["observations"]:
            row["measurements"]["center"]["value"] = [1, 1]
    raw = json.dumps(doc, sort_keys=True).encode()
    sha = hashlib.sha256(raw).hexdigest()
    scene = authored_shared_world_scene()
    scene["human_spec"]["source_sha256"] = source_sha
    ids = ["human", *(obj["id"] for obj in scene["objects"])]
    refs = {name: None for name in ids}
    refs.update(human=(sha, "person"), ball=(sha, "ball"))
    scene["source_binding"], _ = build_shared_world_binding(
        {sha: raw}, source_sha256=source_sha, world_ids=ids, entity_refs=refs, source_equivalences=equivalences)
    if observation_path is not None:
        scene["source_binding"]["evidence_documents"][sha] = observation_path
    name = scene["source_binding"]["evidence_documents"][sha]
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    for item in equivalences:
        for filename in item.file_names:
            target = root / f"source-mappings/{item.manifest_sha256}/{filename}"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.raw_bytes(filename))
    write_json(root / "scene.json", scene)
    seal(root)
    return scene, name


def test_snapshot_preserves_exact_observations_and_complete_bundle_after_input_changes(tmp_path):
    source = tmp_path / "source"
    scene, doc = candidate(source, equivalent=True)
    checked = verify_artifacts(source)
    pinned = collect_shared_source_files(scene, source, expected_manifest_sha256=checked["manifest_sha256"])
    bundle = scene["source_binding"]["source_equivalences"][0]["bundle_path"]
    assert f"{bundle}/manifest.json" in pinned.files
    assert f"{bundle}/source.webm" in pinned.files and f"{bundle}/lossless.mp4" in pinned.files
    with pytest.raises(TypeError):
        pinned.files[doc] = b"changed"
    (source / doc).write_bytes(b"changed after snapshot")
    (source / bundle / "source.webm").write_bytes(b"changed media after snapshot")
    destination = tmp_path / "copied"
    destination.mkdir()
    pinned.copy_to(destination)
    assert all((destination / name).read_bytes() == raw for name, raw in pinned.files.items())
    assert verify_shared_world_sources(scene, destination)["source_evidence_verified"]


@pytest.mark.parametrize("missing", ["observation", "manifest.json", "lossless.mp4", "frame-mapping-and-pixels.json"])
def test_outer_manifest_must_cover_every_pinned_file(tmp_path, missing):
    source = tmp_path / "source"
    scene, doc = candidate(source, equivalent=True)
    name = doc if missing == "observation" else scene["source_binding"]["source_equivalences"][0]["bundle_path"] + "/" + missing
    manifest = json.loads((source / "manifest.json").read_bytes())
    del manifest["files"][name]
    write_json(source / "manifest.json", manifest)
    verify_artifacts(source)  # A partial outer manifest alone cannot establish source completeness.
    with pytest.raises(ValueError, match="Outer manifest must cover"):
        collect_shared_source_files(scene, source)


@pytest.mark.parametrize("operation", ["lab", "training"])
def test_incomplete_source_manifest_rejects_before_output_or_training(tmp_path, operation):
    source = tmp_path / "source"
    _, doc = candidate(source)
    manifest = json.loads((source / "manifest.json").read_bytes())
    del manifest["files"][doc]
    write_json(source / "manifest.json", manifest)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="Outer manifest must cover"):
        (prepare_shared_world_lab if operation == "lab" else train_shared_world_policy)(source, out)
    assert not out.exists()


@pytest.mark.parametrize("path", ["policy.json", "implementation/evidence.json", "lab-provenance.json"])
def test_generated_names_cannot_replace_pinned_evidence(tmp_path, path):
    source = tmp_path / "source"
    scene, _ = candidate(source, observation_path=path)
    pinned = collect_shared_source_files(scene, source)
    with pytest.raises(ValueError, match="collides"):
        pinned.validate_layout(reserved_paths={"policy.json", "implementation", "lab-provenance.json"})
    with pytest.raises(ValueError, match="collides"):
        prepare_shared_world_lab(source, tmp_path / "lab")
    assert not (tmp_path / "lab").exists()


@pytest.mark.parametrize("unsafe", ["../escape.json", "/absolute.json", "a/../b.json", "a//b.json", "a\\b.json", "a:b.json"])
def test_writer_rejects_noncanonical_paths_before_writing(tmp_path, unsafe):
    with pytest.raises(ValueError, match="safe relative"):
        write_package_files(tmp_path, {"first.json": b"first", unsafe: b"unsafe"})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("collision", ["existing_file", "file_symlink", "parent_symlink", "parent_file"])
def test_copy_rejects_existing_or_linked_destinations_without_overwrite(tmp_path, collision):
    source = tmp_path / "source"
    scene, name = candidate(source)
    pinned = collect_shared_source_files(scene, source)
    out, outside = tmp_path / "out", tmp_path / "outside"
    out.mkdir()
    outside.mkdir()
    sentinel = outside / "sentinel.json"
    sentinel.write_bytes(b"keep")
    target = out / name
    if collision == "existing_file":
        target.parent.mkdir()
        target.write_bytes(b"keep")
    elif collision == "file_symlink":
        target.parent.mkdir()
        target.symlink_to(sentinel)
    elif collision == "parent_symlink":
        target.parent.symlink_to(outside, target_is_directory=True)
    else:
        target.parent.write_bytes(b"keep")
    with pytest.raises(ValueError):
        pinned.copy_to(out)
    assert sentinel.read_bytes() == b"keep"
    if collision == "existing_file":
        assert target.read_bytes() == b"keep"
    if collision == "parent_file":
        assert target.parent.read_bytes() == b"keep"


def test_writer_rejects_hierarchy_collision_before_creating_files(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        write_package_files(tmp_path, {"a.json": b"a", "a.json/b.json": b"b"})
    assert list(tmp_path.iterdir()) == []


def test_source_symlink_and_manifest_change_reject(tmp_path):
    source = tmp_path / "source"
    scene, name = candidate(source)
    manifest_sha = digest(source / "manifest.json")
    with pytest.raises(ValueError, match="changed after candidate"):
        collect_shared_source_files(scene, source, expected_manifest_sha256="f" * 64)
    assert digest(source / "manifest.json") == manifest_sha
    original = (source / name).read_bytes()
    outside = tmp_path / "observations.json"
    outside.write_bytes(original)
    (source / name).unlink()
    (source / name).symlink_to(outside)
    with pytest.raises(ValueError, match="unsafe"):
        collect_shared_source_files(scene, source)


def test_generic_copy_checks_exact_input_hash_and_refuses_links(tmp_path):
    source, out = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    out.mkdir()
    (source / "data.json").write_bytes(b"actual")
    with pytest.raises(ValueError, match="changed while copying"):
        copy_package_file(source, out, "data.json", hashlib.sha256(b"different").hexdigest())
    assert (source / "data.json").read_bytes() == b"actual"
    (source / "link.json").symlink_to(source / "data.json")
    with pytest.raises(ValueError, match="unsafe"):
        copy_package_file(source, out, "link.json", digest(source / "data.json"))
    assert not (out / "link.json").exists()


def test_analytic_snapshot_has_no_source_claim_or_required_manifest():
    pinned = collect_shared_source_files(authored_shared_world_scene(), None)
    assert not pinned.files and pinned.manifest_sha256 is None
    assert pinned.report["source_evidence_verified"] is False


def test_bounded_reader_pins_bytes_and_exposes_immutable_manifest(tmp_path):
    path = tmp_path / "geometry.json"
    path.write_bytes(b'{"data": 1}\n')
    seal(tmp_path)
    raw, files = read_package_manifest(tmp_path)
    assert raw == (tmp_path / "manifest.json").read_bytes()
    assert read_package_file(tmp_path, "geometry.json", max_bytes=100,
                             expected_sha256=files["geometry.json"]) == path.read_bytes()
    with pytest.raises(TypeError):
        files["geometry.json"] = "a" * 64
    with pytest.raises(ValueError, match="bounded regular"):
        read_package_file(tmp_path, "geometry.json", max_bytes=2)
    with pytest.raises(ValueError, match="pinned SHA256"):
        read_package_file(tmp_path, "geometry.json", max_bytes=100, expected_sha256="f" * 64)
    (tmp_path / "link.json").symlink_to(path)
    with pytest.raises(ValueError, match="unsafe"):
        read_package_file(tmp_path, "link.json", max_bytes=100)


@pytest.mark.parametrize("raw", [
    b'{"files":{"a.json":"bad"}}', b'{"files":{},"files":{}}',
    b'{"files":{"../a.json":"' + b"a" * 64 + b'"}}',
    b'{"files":{"a.json":"' + b"a" * 64 + b'"}, "value":NaN}',
])
def test_manifest_identity_reader_rejects_noncanonical_manifests(tmp_path, raw):
    (tmp_path / "manifest.json").write_bytes(raw)
    with pytest.raises(ValueError):
        read_package_manifest(tmp_path)


def test_destination_plan_rejects_generated_directory_collision_before_output(tmp_path):
    source = tmp_path / "source"
    candidate(source)
    (source / "human.json").mkdir()
    (source / "human.json/nested.json").write_bytes(b"{}")
    seal(source)
    with pytest.raises(ValueError, match="overlap"):
        prepare_shared_world_lab(source, tmp_path / "lab")
    assert not (tmp_path / "lab").exists()


def test_equivalence_policy_and_lab_are_standalone_and_preserve_original_inputs(tmp_path):
    source = tmp_path / "source"
    scene, _ = candidate(source, equivalent=True)
    original = {p.relative_to(source).as_posix(): digest(p) for p in source.rglob("*") if p.is_file()}
    policy = tmp_path / "policy"
    train_shared_world_policy(source, policy, seed=42, generations=1, population=4, eval_episodes=2, max_seconds=30)
    trained = json.loads((policy / "scene.json").read_bytes())
    policy_sources = collect_shared_source_files(trained, policy)
    assert policy_sources.report["source_evidence_verified"]
    assert "implementation/source_equivalence.py" in policy_sources.manifest_files
    bundle_manifest = scene["source_binding"]["source_equivalences"][0]["bundle_path"] + "/manifest.json"
    assert bundle_manifest in policy_sources.manifest_files
    lab = tmp_path / "lab"
    prepare_shared_world_lab(source, lab, policy_run=policy)
    verify_artifacts(lab / "run")
    packaged = json.loads((lab / "run/scene.json").read_bytes())
    lab_sources = collect_shared_source_files(packaged, lab / "run")
    assert lab_sources.files == policy_sources.files
    assert bundle_manifest in lab_sources.manifest_files
    assert "lab-implementation/source_equivalence.py" in lab_sources.manifest_files
    assert "policy-implementation/source_equivalence.py" in lab_sources.manifest_files
    assert original == {p.relative_to(source).as_posix(): digest(p) for p in source.rglob("*") if p.is_file()}
    assert verify_shared_world_sources(packaged, lab / "run")["source_evidence_verified"]
    # A complete source candidate must not hide an incomplete optional policy package.
    manifest = json.loads((policy / "manifest.json").read_bytes())
    del manifest["files"][bundle_manifest]
    write_json(policy / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Outer manifest must cover"):
        prepare_shared_world_lab(source, tmp_path / "invalid-policy-lab", policy_run=policy)
    assert not (tmp_path / "invalid-policy-lab").exists()
