"""Shared hard limits and append-only output handling."""

import hashlib
import json
import os
import re
import time
from pathlib import Path

LIMITS = {
    "input_bytes": 64 * 1024 * 1024,
    "archive_bytes": 128 * 1024 * 1024,
    "archive_entries": 32,
    "archive_expanded_bytes": 256 * 1024 * 1024,
    "compression_ratio": 200,
    "pixels": 16_777_216,
    "video_pixels": 1920 * 1080,
    "frames": 600,
    "seconds": 20,
    "fps": 60,
    "states": 12,
    "output_bytes": 512 * 1024 * 1024,
    "wall_seconds": 180,
    "rss_bytes": 1536 * 1024 * 1024,
    "json_bytes": 1024 * 1024,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def slug(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", value),
            "IDs must be lowercase safe slugs, at most 64 characters")
    return value


def local_file(value, max_bytes=None):
    path = Path(value)
    require(not path.is_symlink() and path.is_file(), f"Not a regular local file: {path}")
    require(path.stat().st_size <= (max_bytes or LIMITS["input_bytes"]), f"Input too large: {path}")
    return path.resolve()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    path = local_file(path, LIMITS["json_bytes"])
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, f"Duplicate JSON key: {k}")
            result[k] = v
        return result
    def bad_constant(value):
        raise ValueError(f"Nonfinite JSON constant: {value}")
    return json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=bad_constant)


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def new_directory(path):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), f"Output already exists; use a new folder: {path}")
    path.mkdir(parents=True, mode=0o700)
    return path.resolve()


def contained(root, relative):
    require(isinstance(relative, str), "Manifest path must be a string")
    candidate = Path(root) / relative
    require(not Path(relative).is_absolute() and not candidate.is_symlink(), "Unsafe manifest path")
    require(candidate.resolve().is_relative_to(Path(root).resolve()), "Path escapes pack")
    return candidate


def artifacts(root, enforce_budget=True):
    result = []
    size = 0
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), "Symlinks are not pack artifacts")
        if path.is_file() and path != Path(root) / "manifest.json":
            size += path.stat().st_size
            require(not enforce_budget or size <= LIMITS["output_bytes"], "Output byte budget exceeded")
            result.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                           "sha256": sha256(path)})
    return result


def preserve_failure(output, error):
    root = Path(output)
    if root.is_dir() and not (root / "manifest.json").exists():
        write_json(root / "manifest.json", {"schema_version": 1, "status": "failed",
                   "error": str(error), "rejected_attempts": [{"reason": str(error), "preserved": True}],
                   "artifacts": artifacts(root, enforce_budget=False)})


class Deadline:
    def __init__(self):
        self.start = time.monotonic()

    def check(self):
        require(time.monotonic() - self.start < LIMITS["wall_seconds"], "Runtime budget exceeded")

    def seconds(self):
        return round(time.monotonic() - self.start, 3)
