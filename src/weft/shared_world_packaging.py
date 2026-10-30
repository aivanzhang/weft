"""Preserve complete, hash-pinned source snapshots across standalone packages."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType

from weft.shared_world_evidence import verify_shared_world_sources


MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _name(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 1024 or "\\" in value or ":" in value
            or any(ord(char) < 32 for char in value) or PurePosixPath(value).is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError("Package files require safe relative paths")
    return value


def _layout(names, reserved_paths=()):
    names = {_name(name) for name in names}
    reserved = {_name(name) for name in reserved_paths}
    for name in names:
        parents = {parent.as_posix() for parent in PurePosixPath(name).parents}
        if parents & names:
            raise ValueError("Package file and directory paths overlap")
        if name in reserved or parents & reserved or any(item.startswith(name + "/") for item in reserved):
            raise ValueError(f"Pinned source path collides with a generated package path: {name}")


def validate_package_layout(names):
    """Reject file/directory collisions across a complete destination plan."""
    _layout(names)


def _root_fd(directory):
    try:
        return os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("Package directory is missing or unsafe") from error


def _parent_fd(root_fd, name, *, create=False):
    fd = os.dup(root_fd)
    try:
        for part in PurePosixPath(name).parts[:-1]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _check_absent(root_fd, names):
    for name in names:
        try:
            parent = _parent_fd(root_fd, name)
        except FileNotFoundError:
            continue
        try:
            try:
                os.stat(PurePosixPath(name).name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise ValueError(f"Package destination already contains {name}")
        finally:
            os.close(parent)


def write_package_files(directory, files):
    """Write exact bytes without following links or replacing existing artifacts."""
    _layout(files)
    root_fd = _root_fd(directory)
    try:
        _check_absent(root_fd, files)
        for name, raw in files.items():
            parent = _parent_fd(root_fd, name, create=True)
            try:
                fd = os.open(PurePosixPath(name).name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
                with os.fdopen(fd, "w+b") as stream:
                    stream.write(raw)
                    stream.flush()
                    stream.seek(0)
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                    if actual != hashlib.sha256(raw).hexdigest():
                        raise ValueError("Copied package bytes differ from the pinned snapshot")
            finally:
                os.close(parent)
    except OSError as error:
        raise ValueError("Package destination is missing, occupied or unsafe") from error
    finally:
        os.close(root_fd)


def copy_package_file(source, destination, name, expected_sha256, *, target_name=None):
    """Stream a sealed non-source artifact through no-follow, exclusive file handles."""
    name, target_name = _name(name), _name(target_name or name)
    source_fd, destination_fd = _root_fd(source), None
    try:
        destination_fd = _root_fd(destination)
        source_parent = _parent_fd(source_fd, name)
        try:
            fd = os.open(PurePosixPath(name).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=source_parent)
        finally:
            os.close(source_parent)
        with os.fdopen(fd, "rb") as incoming:
            before = os.fstat(incoming.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Package inputs must be regular files")
            parent = _parent_fd(destination_fd, target_name, create=True)
            try:
                fd = os.open(PurePosixPath(target_name).name,
                             os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            finally:
                os.close(parent)
            digest = hashlib.sha256()
            remaining = before.st_size
            with os.fdopen(fd, "wb") as outgoing:
                while block := incoming.read(min(1 << 20, remaining + 1)):
                    remaining -= len(block)
                    if remaining < 0:
                        raise ValueError(f"Sealed package input grew while copying: {name}")
                    outgoing.write(block)
                    digest.update(block)
            after = os.fstat(incoming.fileno())
            if (digest.hexdigest() != expected_sha256 or
                    (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise ValueError(f"Sealed package input changed while copying: {name}")
    except OSError as error:
        raise ValueError("Package copy path is missing, occupied or unsafe") from error
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Package manifest contains duplicate keys")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"Package manifest contains non-finite JSON: {value}")


def read_package_file(directory, name, *, max_bytes, expected_sha256=None):
    """Read a bounded regular artifact through no-follow handles, optionally pinning its hash."""
    name = _name(name)
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_FILE_BYTES:
        raise ValueError("Package reads require a positive bound of at most 64 MiB")
    root_fd = _root_fd(directory)
    try:
        parent = _parent_fd(root_fd, name)
        try:
            fd = os.open(PurePosixPath(name).name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        finally:
            os.close(parent)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
                raise ValueError("Package input is not a bounded regular file")
            raw = stream.read(max_bytes + 1)
            after = os.fstat(stream.fileno())
            if (len(raw) > max_bytes or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                    (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
                raise ValueError("Package input changed during bounded read")
    except OSError as error:
        raise ValueError("Package input is missing or unsafe") from error
    finally:
        os.close(root_fd)
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Package input differs from its pinned SHA256")
    return raw


def read_package_manifest(directory):
    """Return exact manifest bytes and an immutable, path-validated file hash map."""
    raw = read_package_file(directory, "manifest.json", max_bytes=MAX_MANIFEST_BYTES)
    try:
        data = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Package manifest must be bounded strict JSON") from error
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, dict) or not files or len(files) > 100_000:
        raise ValueError("Package manifest requires a bounded nonempty file map")
    _layout(files, {"manifest.json"})
    if any(not isinstance(value, str) or not _DIGEST.fullmatch(value) for value in files.values()):
        raise ValueError("Package manifest requires exact SHA256 file identities")
    return raw, MappingProxyType(files)


@dataclass(frozen=True)
class PinnedSourceFiles:
    report: dict
    files: object
    manifest_bytes: bytes | None
    manifest_files: object
    manifest_sha256: str | None

    def validate_layout(self, *, reserved_paths=()):
        _layout(self.files, reserved_paths)

    def copy_to(self, destination, *, reserved_paths=()):
        self.validate_layout(reserved_paths=reserved_paths)
        write_package_files(destination, self.files)


def collect_shared_source_files(scene, source_dir, *, expected_manifest_sha256=None):
    """Require outer-manifest coverage for every exact source byte before any output."""
    raw, manifest_files = None, {}
    if source_dir is not None:
        raw, manifest_files = read_package_manifest(Path(source_dir))
        if expected_manifest_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
            raise ValueError("Package manifest changed after candidate verification")
    elif scene["source_binding"]["mode"] != "analytic_fixture":
        raise ValueError("Pinned sources require a sealed source directory")
    report, files = verify_shared_world_sources(scene, source_dir, include_files=True)
    for name, data in files.items():
        if manifest_files.get(name) != hashlib.sha256(data).hexdigest():
            raise ValueError(f"Outer manifest must cover every pinned source file exactly: {name}")
    _layout(files)
    return PinnedSourceFiles(report, files, raw, MappingProxyType(manifest_files),
                             hashlib.sha256(raw).hexdigest() if raw is not None else None)
