"""Seal original avatar assets and their bounded, render-only glTF normalization."""

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
from types import MappingProxyType

from weft.avatar_gltf import (
    AvatarAssetError, _load_document, deadline_check, fail, json_document, normalize_avatar_gltf, safe_relative,
)


SOURCE_PROFILE = "weft-avatar-asset-v1"
IMPORT_PROFILE = "weft-avatar-import-v1"
MAX_FILES, MAX_INPUT_BYTES, MAX_OUTPUT_BYTES = 32, 32 * 1024 * 1024, 64 * 1024 * 1024
SOURCE_KEYS = {"schema_version", "profile", "asset_id", "display_name", "entrypoint", "files", "rights", "provenance", "retargeting"}
RESOURCE_SUFFIXES = frozenset((".glb", ".gltf", ".bin", ".png", ".jpg", ".jpeg", ".json", ".txt", ".md"))
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW


@dataclass(frozen=True)
class AvatarPackage:
    source_manifest_sha256: str
    manifest: dict
    originals: object
    render: dict
    render_json: bytes
    texture_files: object

    @property
    def render_sha256(self):
        return _sha(self.render_json)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _seconds(max_seconds):
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 10:
        fail("Avatar processing must have a positive deadline of at most 10 seconds")
    return time.monotonic() + max_seconds


def _directory(path, *, create=False):
    path = Path(path).absolute()
    if ".." in path.parts:
        fail("Avatar directories cannot use parent traversal")
    fd = os.open(path.anchor, DIR_FLAGS)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, mode=0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            next_fd = os.open(part, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(directory, name, maximum):
    safe_relative(name)
    try:
        fd = _directory(Path(directory) / PurePosixPath(name).parent)
    except OSError as error:
        raise AvatarAssetError("Avatar resource parent is missing or unsafe: " + name) from error
    try:
        child = os.open(PurePosixPath(name).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        with os.fdopen(child, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= maximum:
                fail("Avatar resource is not a bounded regular file: " + name)
            raw = stream.read(maximum + 1)
            if len(raw) != info.st_size or len(raw) > maximum:
                fail("Avatar resource changed or exceeded its byte bound: " + name)
            return raw
    except OSError as error:
        raise AvatarAssetError("Avatar resource is missing, unsafe or unreadable: " + name) from error
    finally:
        os.close(fd)


def _text(value, limit, label):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or any(ord(c) < 32 for c in value):
        fail(label + " must be a nonempty bounded text declaration")
    return value


def _manifest(raw):
    manifest = json_document(raw, maximum=128 * 1024)
    if (set(manifest) != SOURCE_KEYS or type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1 or manifest.get("profile") != SOURCE_PROFILE):
        fail("Avatar requires an exact weft-avatar-asset-v1 manifest")
    if not isinstance(manifest["asset_id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", manifest["asset_id"]):
        fail("Avatar asset_id must be a stable lowercase identifier")
    _text(manifest["display_name"], 128, "Avatar display name")
    entrypoint = safe_relative(manifest["entrypoint"])
    if not entrypoint.endswith((".gltf", ".glb")):
        fail("Avatar entrypoint must be .gltf or .glb", unsupported=True)
    files = manifest["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES or entrypoint not in files:
        fail("Avatar files must include its model within the file-count bound")
    for name, digest in files.items():
        safe_relative(name)
        if (name == "avatar.json" or (PurePosixPath(name).suffix.lower() not in RESOURCE_SUFFIXES
                                    and PurePosixPath(name).name not in ("LICENSE", "NOTICE"))
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)):
            fail("Avatar resource extension, name or SHA256 is invalid")
    rights = manifest["rights"]
    if not isinstance(rights, dict) or set(rights) != {"holder", "license", "source", "declaration"}:
        fail("Avatar rights require holder, license, source and an explicit usage declaration")
    for key, value in rights.items():
        _text(value, 4096, "Avatar rights " + key)
    if not isinstance(manifest["provenance"], dict) or len(_bytes(manifest["provenance"])) > 32 * 1024:
        fail("Avatar provenance must be a bounded object preserving source/generation history")
    return manifest


def _validate(raw, originals, deadline):
    manifest = _manifest(raw)
    if set(originals) != set(manifest["files"]) | {"avatar.json"} or originals["avatar.json"] != raw:
        fail("Avatar original resource set differs from its exact source manifest")
    if sum(map(len, originals.values())) > MAX_INPUT_BYTES:
        fail("Avatar originals exceed the 32MiB input bound")
    for name, expected in manifest["files"].items():
        deadline_check(deadline)
        if not isinstance(originals[name], bytes) or _sha(originals[name]) != expected:
            raise AvatarAssetError("Avatar source hash differs: " + name, code="avatar_hash_mismatch")
    render, render_json, textures = normalize_avatar_gltf(manifest["entrypoint"], originals, manifest["retargeting"], deadline=deadline)
    deadline_check(deadline)
    return AvatarPackage(_sha(raw), manifest, MappingProxyType(dict(originals)), render, render_json, MappingProxyType(textures))


def load_avatar_package(directory, *, max_seconds=5):
    """Validate a factory package without changing its originals or executing its content."""
    deadline = _seconds(max_seconds)
    raw = _read(directory, "avatar.json", 128 * 1024)
    manifest = _manifest(raw)
    originals, total = {"avatar.json": raw}, len(raw)
    for name in manifest["files"]:
        deadline_check(deadline)
        content = _read(directory, name, max(1, MAX_INPUT_BYTES - total))
        originals[name] = content
        total += len(content)
        if total > MAX_INPUT_BYTES:
            fail("Avatar originals exceed the 32MiB input bound")
    return _validate(raw, originals, deadline)


def _normalizer_hashes():
    root = Path(__file__).parent
    return {name: _sha(_read(root, name, 512 * 1024)) for name in ("avatar_assets.py", "avatar_gltf.py")}


def _seal(package, out, deadline):
    files = {"originals/" + name: raw for name, raw in package.originals.items()}
    files.update(package.texture_files)
    files["render.json"] = package.render_json
    receipt = {"schema_version": 1, "profile": IMPORT_PROFILE, "status": "ready",
               "asset_id": package.manifest["asset_id"], "source_manifest_sha256": package.source_manifest_sha256,
               "render_sha256": package.render_sha256, "normalizer_sha256": _normalizer_hashes(),
               "files": {name: _sha(raw) for name, raw in files.items()},
               "scope": "Validated visual avatar and declared rights/provenance, not independent rights verification, anatomy, metric calibration or physics."}
    files["manifest.json"] = _bytes(receipt)
    if len(files) > 64 or sum(map(len, files.values())) > MAX_OUTPUT_BYTES:
        fail("Avatar import exceeds its 64-file/64MiB output bound")
    out = Path(out).absolute()
    parent = _directory(out.parent, create=True)
    try:
        os.mkdir(out.name, mode=0o700, dir_fd=parent)
    except FileExistsError as error:
        raise AvatarAssetError("Avatar output already exists; originals and imports are never overwritten") from error
    finally:
        os.close(parent)
    # The terminal seal is last; interrupted output cannot masquerade as a ready package.
    for name, raw in files.items():
        deadline_check(deadline)
        directory = _directory(out / PurePosixPath(name).parent, create=True)
        try:
            child = os.open(PurePosixPath(name).name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                            mode=0o600, dir_fd=directory)
            with os.fdopen(child, "wb") as stream:
                stream.write(raw)
        finally:
            os.close(directory)
    return {"directory": str(out), "manifest_sha256": _sha(files["manifest.json"]),
            "source_manifest_sha256": package.source_manifest_sha256, "render_sha256": package.render_sha256,
            "asset_id": package.manifest["asset_id"], "files": len(files), "bytes": sum(map(len, files.values()))}


def import_avatar_package(directory, out, *, max_seconds=10):
    """Import a factory package into a new sealed directory; never rewrite an existing run."""
    deadline = _seconds(max_seconds)
    package = load_avatar_package(directory, max_seconds=max(.000001, deadline - time.monotonic()))
    return _seal(package, out, deadline)


def import_avatar_model(model, metadata, out, *, max_seconds=10):
    """Hash a local model/resources for callers supplying explicit rights and retarget metadata."""
    deadline = _seconds(max_seconds)
    if not isinstance(metadata, dict) or set(metadata) != {"asset_id", "display_name", "rights", "provenance", "retargeting"}:
        fail("Model import requires explicit identity, rights, provenance and retarget metadata")
    model = Path(model).absolute()
    entrypoint = safe_relative(model.name)
    raw = _read(model.parent, entrypoint, MAX_INPUT_BYTES)
    originals = {entrypoint: raw}
    document, _ = _load_document(entrypoint, originals)
    total = len(raw)
    for group in ("buffers", "images"):
        values = document.get(group, [])
        if not isinstance(values, list) or len(values) > 16:
            fail("Model resource list exceeds its bound")
        for value in values:
            if not isinstance(value, dict):
                fail("Model resources must be objects")
            if "uri" in value:
                name = safe_relative(value["uri"])
                if name not in originals:
                    if len(originals) >= MAX_FILES:
                        fail("Model resources exceed their file-count bound")
                    deadline_check(deadline)
                    originals[name] = _read(model.parent, name, max(1, MAX_INPUT_BYTES - total))
                    total += len(originals[name])
    manifest = {"schema_version": 1, "profile": SOURCE_PROFILE, **metadata, "entrypoint": entrypoint,
                "files": {name: _sha(content) for name, content in originals.items()}}
    originals["avatar.json"] = _bytes(manifest)
    package = _validate(originals["avatar.json"], originals, deadline)
    return _seal(package, out, deadline)


def load_imported_avatar(directory, *, max_seconds=5):
    """Revalidate originals and exact derived bytes before an imported package is selected."""
    deadline = _seconds(max_seconds)
    raw = _read(directory, "manifest.json", 128 * 1024)
    manifest = json_document(raw, maximum=128 * 1024)
    keys = {"schema_version", "profile", "status", "asset_id", "source_manifest_sha256", "render_sha256", "normalizer_sha256", "files", "scope"}
    if (set(manifest) != keys or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("profile") != IMPORT_PROFILE or manifest.get("status") != "ready"
            or not isinstance(manifest.get("files"), dict) or not 1 <= len(manifest["files"]) <= 63):
        fail("Imported avatar requires its complete ready seal")
    files, total = {}, len(raw)
    for name, expected in manifest["files"].items():
        deadline_check(deadline)
        safe_relative(name)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            fail("Imported avatar hashes must be exact SHA256 values")
        content = _read(directory, name, max(1, MAX_OUTPUT_BYTES - total))
        if _sha(content) != expected:
            raise AvatarAssetError("Imported avatar hash differs: " + name, code="avatar_hash_mismatch")
        files[name] = content
        total += len(content)
        if total > MAX_OUTPUT_BYTES:
            fail("Imported avatar exceeds its byte bound")
    originals = {name[len("originals/"):]: content for name, content in files.items() if name.startswith("originals/")}
    if "avatar.json" not in originals:
        fail("Imported avatar is missing its original source manifest")
    package = _validate(originals["avatar.json"], originals, deadline)
    expected_files = {"originals/" + name for name in originals} | {"render.json"} | set(package.texture_files)
    if (set(files) != expected_files or files.get("render.json") != package.render_json
            or any(files[name] != content for name, content in package.texture_files.items())
            or manifest["source_manifest_sha256"] != package.source_manifest_sha256
            or manifest["render_sha256"] != package.render_sha256 or manifest["asset_id"] != package.manifest["asset_id"]):
        fail("Imported avatar differs from its original model and explicit mapping")
    if manifest["normalizer_sha256"] != _normalizer_hashes():
        fail("Avatar normalizer changed; create a new import rather than rewriting this package")
    deadline_check(deadline)
    return package
