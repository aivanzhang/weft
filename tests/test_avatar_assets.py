"""Original custody and sealed render bytes remain separate from simulation evidence."""

from copy import deepcopy
import hashlib
import json

import pytest

from weft import avatar_assets as module
from test_avatar_gltf import fixture_model, mapping, packaged


def metadata():
    return {"asset_id": "analytic-two-bone", "display_name": "Analytic skin fixture",
            "rights": {"holder": "Test fixture", "license": "Synthetic test data", "source": "Created for analytic verification",
                       "declaration": "Local test use; not a user-facing character asset"},
            "provenance": {"purpose": "Verify geometry and custody without external artwork", "prompt_history": []},
            "retargeting": mapping()}


def factory(tmp_path, *, glb=False):
    root = tmp_path / "source"
    root.mkdir()
    doc, binary, extras = fixture_model(texture=True, embedded=glb)
    entrypoint, files = packaged(doc, binary, extras, glb=glb)
    manifest = {"schema_version": 1, "profile": module.SOURCE_PROFILE, **metadata(), "entrypoint": entrypoint,
                "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in files.items()}}
    files["avatar.json"] = json.dumps(manifest, indent=2).encode()
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    return root, files


@pytest.mark.parametrize("glb", [False, True])
def test_import_preserves_originals_texture_and_mapping_then_revalidates(tmp_path, glb):
    source, originals = factory(tmp_path, glb=glb)
    before = {name: (source / name).stat().st_mtime_ns for name in originals}
    out = tmp_path / "imported"
    receipt = module.import_avatar_package(source, out)
    loaded = module.load_imported_avatar(out)
    assert loaded.manifest["rights"] == metadata()["rights"]
    assert loaded.manifest["provenance"] == metadata()["provenance"]
    assert loaded.render["retargeting"] == mapping()
    for name, raw in originals.items():
        assert (source / name).read_bytes() == raw == (out / "originals" / name).read_bytes()
        assert (source / name).stat().st_mtime_ns == before[name]
    assert receipt["render_sha256"] == hashlib.sha256((out / "render.json").read_bytes()).hexdigest()
    assert receipt["source_manifest_sha256"] == hashlib.sha256(originals["avatar.json"]).hexdigest()
    assert loaded.render["counts"]["skin_joints"] == 2
    assert len(loaded.texture_files) == 1
    assert not (out / "sessions").exists()


def test_direct_model_import_builds_hashes_without_user_editing_manifests(tmp_path):
    source, originals = factory(tmp_path)
    out = tmp_path / "direct"
    result = module.import_avatar_model(source / "model.gltf", metadata(), out)
    loaded = module.load_imported_avatar(out)
    assert result["asset_id"] == metadata()["asset_id"]
    assert loaded.originals["model.gltf"] == originals["model.gltf"]
    assert loaded.originals["model.bin"] == originals["model.bin"]
    assert loaded.originals["skin.png"] == originals["skin.png"]


def test_declared_license_original_is_preserved_without_interpreting_it(tmp_path):
    source, _ = factory(tmp_path)
    license_raw = b"Synthetic fixture rights declaration. No commands are executed.\n"
    (source / "LICENSE").write_bytes(license_raw)
    manifest = json.loads((source / "avatar.json").read_bytes())
    manifest["files"]["LICENSE"] = hashlib.sha256(license_raw).hexdigest()
    (source / "avatar.json").write_text(json.dumps(manifest))
    out = tmp_path / "imported"
    module.import_avatar_package(source, out)
    assert module.load_imported_avatar(out).originals["LICENSE"] == license_raw


def test_declared_hash_cannot_be_replaced_by_current_file_content(tmp_path):
    source, _ = factory(tmp_path)
    (source / "model.bin").write_bytes(b"changed")
    with pytest.raises(module.AvatarAssetError, match="source hash differs") as error:
        module.load_avatar_package(source)
    assert error.value.code == "avatar_hash_mismatch"


def test_self_resealed_render_cannot_override_original_geometry(tmp_path):
    source, _ = factory(tmp_path)
    out = tmp_path / "imported"
    module.import_avatar_package(source, out)
    render = json.loads((out / "render.json").read_bytes())
    render["primitives"][0]["positions"][0] = 999
    raw = json.dumps(render).encode()
    (out / "render.json").write_bytes(raw)
    manifest = json.loads((out / "manifest.json").read_bytes())
    manifest["render_sha256"] = manifest["files"]["render.json"] = hashlib.sha256(raw).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(module.AvatarAssetError, match="original model"):
        module.load_imported_avatar(out)


@pytest.mark.parametrize("target", ["source", "resource", "resource_parent"])
def test_symlinked_sources_never_escape_package(tmp_path, target):
    source, _ = factory(tmp_path)
    if target == "source":
        linked = tmp_path / "linked"
        linked.symlink_to(source, target_is_directory=True)
        source = linked
    elif target == "resource":
        original = source / "model.bin"
        other = tmp_path / "outside.bin"
        other.write_bytes(original.read_bytes())
        original.unlink()
        original.symlink_to(other)
    else:
        manifest = json.loads((source / "avatar.json").read_bytes())
        old = manifest["files"].pop("model.bin")
        manifest["files"]["nested/model.bin"] = old
        (source / "nested").symlink_to(source, target_is_directory=True)
        (source / "avatar.json").write_text(json.dumps(manifest))
    with pytest.raises((module.AvatarAssetError, OSError)):
        module.load_avatar_package(source)


@pytest.mark.parametrize("case", ["missing_rights", "missing_mapping", "traversal", "code_file", "bad_sha", "unknown_field"])
def test_incomplete_or_unsafe_manifest_fails_before_import(tmp_path, case):
    source, _ = factory(tmp_path)
    manifest = json.loads((source / "avatar.json").read_bytes())
    if case == "missing_rights": manifest["rights"].pop("declaration")
    elif case == "missing_mapping": manifest["retargeting"] = {}
    elif case == "traversal": manifest["files"]["../outside.bin"] = "a" * 64
    elif case == "code_file": manifest["files"]["setup.py"] = "a" * 64
    elif case == "bad_sha": manifest["files"]["model.bin"] = "not-a-hash"
    elif case == "unknown_field": manifest["automatic_anatomy_guess"] = True
    (source / "avatar.json").write_text(json.dumps(manifest))
    with pytest.raises(module.AvatarAssetError):
        module.import_avatar_package(source, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()


def test_import_refuses_overwrite_and_changed_normalizer(tmp_path, monkeypatch):
    source, _ = factory(tmp_path)
    out = tmp_path / "imported"
    module.import_avatar_package(source, out)
    seal = (out / "manifest.json").read_bytes()
    with pytest.raises(module.AvatarAssetError, match="never overwritten"):
        module.import_avatar_package(source, out)
    assert (out / "manifest.json").read_bytes() == seal
    monkeypatch.setattr(module, "_normalizer_hashes", lambda: {"avatar_assets.py": "0" * 64})
    with pytest.raises(module.AvatarAssetError, match="normalizer changed"):
        module.load_imported_avatar(out)


def test_explicit_byte_file_and_time_bounds(tmp_path, monkeypatch):
    source, _ = factory(tmp_path)
    for seconds in (0, -1, 11, float("nan"), True):
        with pytest.raises(module.AvatarAssetError, match="deadline"):
            module.load_avatar_package(source, max_seconds=seconds)
    monkeypatch.setattr(module, "MAX_INPUT_BYTES", 100)
    with pytest.raises(module.AvatarAssetError, match="bounded regular|input bound"):
        module.load_avatar_package(source)
    monkeypatch.setattr(module, "MAX_INPUT_BYTES", 32 * 1024 * 1024)
    monkeypatch.setattr(module, "MAX_OUTPUT_BYTES", 100)
    with pytest.raises(module.AvatarAssetError, match="output bound"):
        module.import_avatar_package(source, tmp_path / "imported")
    assert not (tmp_path / "imported").exists()


def test_partial_output_without_terminal_manifest_is_not_admitted(tmp_path):
    out = tmp_path / "partial"
    out.mkdir()
    (out / "render.json").write_text("{}")
    with pytest.raises(module.AvatarAssetError, match="missing"):
        module.load_imported_avatar(out)
