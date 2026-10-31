"""Copy a verified media subset without decoding, regenerating, or deleting assets."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from avatar_assets.cli import verify_pack
from avatar_assets.safety import new_directory, preserve_failure, require, sha256, write_json


@dataclass(frozen=True)
class PortableLimits:
    manifest_bytes: int = 1024 * 1024
    source_entries: int = 4096
    source_bytes: int = 512 * 1024 * 1024
    file_bytes: int = 64 * 1024 * 1024
    output_bytes: int = 64 * 1024 * 1024
    retained_entries: int = 256
    tree_entries: int = 8192
    path_depth: int = 12
    wall_seconds: float = 60


def _relative(value, limits):
    require(isinstance(value, str) and 0 < len(value) <= 512, "Invalid artifact path")
    parts = value.split("/")
    require(len(parts) <= limits.path_depth and all(part not in ("", ".", "..") for part in parts)
            and not any(char in value for char in ("\\", ":", "\x00")), "Unsafe artifact path")
    require(all(len(part) <= 128 for part in parts), "Artifact path component is too long")
    return parts


@contextmanager
def _source_file(root_fd, relative, limits):
    opened = []
    handle = None
    try:
        parts = _relative(relative, limits)
        current = root_fd
        for component in parts[:-1]:
            current = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
            opened.append(current)
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=current)
        handle = os.fdopen(fd, "rb")
        info = os.fstat(handle.fileno())
        require(stat.S_ISREG(info.st_mode), f"Artifact is not a regular file: {relative}")
        yield handle, info
    except OSError as error:
        raise ValueError(f"Unsafe or unreadable artifact: {relative}") from error
    finally:
        if handle is not None:
            handle.close()
        for fd in reversed(opened):
            os.close(fd)


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate manifest JSON key")
            value[key] = item
        return value

    def bad_constant(value):
        raise ValueError(f"Nonfinite manifest JSON constant: {value}")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad_constant)
    except (UnicodeError, RecursionError) as error:
        raise ValueError("Invalid or excessively nested manifest JSON") from error


def _tree_inventory(root_fd, expected, limits, check):
    found = set()
    count = 0

    def walk(directory, prefix="", depth=0):
        nonlocal count
        require(depth <= limits.path_depth, "Source directory nesting limit exceeded")
        with os.scandir(directory) as entries:
            for entry in entries:
                check()
                count += 1
                require(count <= limits.tree_entries, "Source directory entry limit exceeded")
                relative = prefix + entry.name
                _relative(relative, limits)
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISDIR(mode):
                    child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                    try:
                        walk(child, relative + "/", depth + 1)
                    finally:
                        os.close(child)
                else:
                    require(stat.S_ISREG(mode), f"Source contains a symlink or special file: {relative}")
                    if relative != "manifest.json":
                        found.add(relative)

    walk(root_fd)
    require(found == expected, "Unrecorded or missing source artifacts")


def _omitted(path):
    if path.startswith("frames/"):
        return "Native rendered frame; unchanged in the full local source pack"
    if path == "render.log":
        return "Verbose native-render log; unchanged in the full local source pack"
    return None


def _transfer(root_fd, item, limits, check, destination=None):
    digest = hashlib.sha256()
    total = 0
    with _source_file(root_fd, item["path"], limits) as (source, before):
        require(before.st_size == item["bytes"], f"Artifact size changed: {item['path']}")
        output = None
        try:
            if destination is not None:
                destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                output = destination.open("xb")
                os.chmod(destination, 0o600)
            while True:
                check()
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                require(total <= item["bytes"] and total <= limits.file_bytes, "Artifact grew beyond its byte bound")
                digest.update(chunk)
                if output is not None:
                    output.write(chunk)
            after = os.fstat(source.fileno())
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                    f"Artifact changed while copying: {item['path']}")
            require(total == item["bytes"] and digest.hexdigest() == item["sha256"],
                    f"Artifact hash changed: {item['path']}")
        finally:
            if output is not None:
                output.close()


def package_portable_media(source, output, limits=PortableLimits()):
    """Return a deterministic distribution manifest; keep the complete source untouched."""
    hard = PortableLimits()
    require(isinstance(limits, PortableLimits) and all((type(value) in (int, float) if key == "wall_seconds" else type(value) is int)
            and 0 < value <= getattr(hard, key)
            for key, value in asdict(limits).items()), "Limits must be positive and cannot exceed hard caps")
    start = time.monotonic()

    def check():
        require(time.monotonic() - start <= limits.wall_seconds, "Portable packaging runtime limit exceeded")

    root = Path(source)
    require(not root.is_symlink() and root.is_dir(), "Source must be a regular local directory")
    root = root.resolve()
    target = Path(output)
    require(not target.exists() and not target.is_symlink(), "Output already exists; use a new folder")
    require(not target.resolve().is_relative_to(root), "Output cannot be inside the source pack")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    out = None
    try:
        with _source_file(root_fd, "manifest.json", limits) as (handle, info):
            require(info.st_size <= limits.manifest_bytes, "Source manifest byte limit exceeded")
            raw = handle.read(limits.manifest_bytes + 1)
            require(len(raw) == info.st_size, "Source manifest changed while reading")
        original = _json(raw)
        require(isinstance(original, dict) and original.get("schema_version") == 1
                and original.get("kind") == "rendered_2d_avatar" and original.get("status") != "failed",
                "Expected a successful rendered_2d_avatar pack")
        require("distribution" not in original, "Source is already a derived distribution")
        require(isinstance(original.get("parent_3d_manifest_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", original["parent_3d_manifest_sha256"]), "Missing parent 3D manifest hash")
        inventory = original.get("artifacts")
        require(isinstance(inventory, list) and 0 < len(inventory) <= limits.source_entries, "Source artifact count limit exceeded")
        seen, folded = set(), set()
        kept, omitted = [], []
        total_bytes = 0
        for entry in inventory:
            check()
            require(isinstance(entry, dict) and set(entry) == {"path", "bytes", "sha256"}, "Invalid artifact inventory entry")
            path = entry["path"]
            _relative(path, limits)
            require(path.casefold() not in ("manifest.json", "source-manifest.json") and path not in seen
                    and path.casefold() not in folded, "Duplicate, reserved, or case-colliding artifact path")
            seen.add(path)
            folded.add(path.casefold())
            require(type(entry["bytes"]) is int and 0 <= entry["bytes"] <= limits.file_bytes, "Artifact byte limit exceeded")
            require(isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), "Invalid artifact hash")
            total_bytes += entry["bytes"]
            require(total_bytes <= limits.source_bytes, "Source aggregate byte limit exceeded")
            reason = _omitted(path)
            if reason:
                omitted.append({**entry, "reason": reason})
            else:
                kept.append(deepcopy(entry))
        kept.sort(key=lambda entry: entry["path"])
        omitted.sort(key=lambda entry: entry["path"])
        require(len(kept) + 1 <= limits.retained_entries, "Portable artifact count limit exceeded")
        source_manifest = {"path": "source-manifest.json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        retained_bytes = sum(entry["bytes"] for entry in kept) + len(raw)
        require(retained_bytes <= limits.output_bytes, "Portable aggregate byte limit exceeded")
        kept_names = {entry["path"] for entry in kept}
        require("portrait.webp" in kept_names, "Portable pack lacks its portrait")
        require(isinstance(original.get("clips"), list) and original["clips"], "Portable pack lacks animation records")
        for clip in original["clips"]:
            require(isinstance(clip, dict) and clip.get("path") in kept_names and clip.get("fallback") in kept_names,
                    "Clip or static fallback is missing or excluded")
        _tree_inventory(root_fd, seen, limits, check)
        # Verify even omitted data before creating the derived pack.
        for entry in sorted(inventory, key=lambda item: item["path"]):
            _transfer(root_fd, entry, limits, check)
        result = deepcopy(original)
        result["artifacts"] = sorted(kept + [source_manifest], key=lambda entry: entry["path"])
        result["distribution"] = {
            "schema_version": 1,
            "kind": "portable_byte_preserving_subset",
            "source_pack_name": root.name,
            "source_manifest": source_manifest,
            "source_inventory_verified": len(inventory),
            "source_artifact_bytes": total_bytes,
            "retained_original_artifact_bytes": retained_bytes - len(raw),
            "omitted_artifacts": omitted,
            "omission_policy": ["frames/**", "render.log"],
            "native_frames_included": False,
            "source_preservation": "Full source pack and its original manifest remain unchanged; omitted files are not deleted.",
            "regeneration": "No media decoding, rendering, re-encoding, or generation; every retained original file is byte-identical.",
            "parent_3d_binding": "Original parent_3d_manifest_sha256 retained without alteration.",
            "historical_records": "Original render/packaging jobs and absolute paths describe source creation, not new jobs or portable dependency paths.",
            "implementation_sha256": sha256(__file__),
            "limits": asdict(limits),
            "provider_calls": 0,
            "provider_cost_usd": 0,
        }
        # Include the new manifest itself in the output-byte bound, without making it self-referential.
        manifest_bytes = (json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()
        require(len(manifest_bytes) <= limits.manifest_bytes and retained_bytes + len(manifest_bytes) <= limits.output_bytes,
                "Portable manifest or total output byte limit exceeded")
        out = new_directory(target)
        for entry in kept:
            _transfer(root_fd, entry, limits, check, out / entry["path"])
        with (out / "source-manifest.json").open("xb") as handle:
            handle.write(raw)
        with _source_file(root_fd, "manifest.json", limits) as (handle, info):
            require(info.st_size == len(raw) and handle.read(len(raw) + 1) == raw, "Source manifest changed during packaging")
        check()
        write_json(out / "manifest.json", result)
        verify_pack(out)
        check()
        return result
    except Exception as error:
        if out is not None:
            preserve_failure(out, error)
        raise
    finally:
        os.close(root_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", metavar="SOURCE")
    parser.add_argument("--out", required=True, metavar="DEST")
    args = parser.parse_args()
    start = time.monotonic()
    try:
        result = package_portable_media(args.source, args.out)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Portable packaging failed: {error}\n")
    print(json.dumps({"output": args.out, "artifacts": len(result["artifacts"]),
                      "artifact_bytes": sum(entry["bytes"] for entry in result["artifacts"]),
                      "manifest_sha256": sha256(Path(args.out) / "manifest.json"),
                      "omitted_artifacts": len(result["distribution"]["omitted_artifacts"]),
                      "omitted_bytes": sum(entry["bytes"] for entry in result["distribution"]["omitted_artifacts"]),
                      "elapsed_seconds": round(time.monotonic()-start, 3)}))


if __name__ == "__main__":
    main()
