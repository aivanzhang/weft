"""Retain an exact sealed body beside a preflight without claiming new inference."""

from hashlib import sha256
import math
import re
import time

from weft.exercise_body_source import _json, _path
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files

PROFILE = "perception-sealed-body-reuse-v1"
SCOPE = "Exact copied construction evidence; no new pose inference or physical qualification"
MAX_BYTES = 128 << 20


def _check(deadline):
    if type(deadline) not in (int, float) or not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError("Sealed body reuse exceeded its deadline")


def validate_reuse_receipt(receipt, *, source_sha256, body_manifest_sha256):
    expected = {"schema_version": 1, "profile": PROFILE, "origin": "sealed_body_copy",
                "source_sha256": source_sha256, "body_manifest_sha256": body_manifest_sha256,
                "inference_calls": 0, "api_calls": 0, "scope": SCOPE}
    if (receipt != expected or type(receipt.get("schema_version")) is not int
            or any(type(receipt.get(k)) is not int for k in ("inference_calls", "api_calls"))
            or any(not isinstance(v, str) or not re.fullmatch("[a-f0-9]{64}", v)
                   for v in (source_sha256, body_manifest_sha256))):
        raise ValueError("Body reuse must retain exact source custody and zero-inference accounting")
    return receipt


def copy_sealed_body(body_source, target, *, source_sha256, deadline):
    """Copy bounded verified bytes; ordinary seed construction still admits the full body."""
    _check(deadline)
    source, target = _path(body_source), _path(target)
    if target.exists() or target == source or target.is_relative_to(source) or source.is_relative_to(target):
        raise ValueError("Body reuse requires a new output outside its source")
    manifest, files = read_package_manifest(source)
    marker = _json(manifest)
    if (marker.get("profile") != "exercise-body-source-v1" or marker.get("status") != "construction_only"
            or not 1 <= len(files) <= 511 or "body-source.json" not in files
            or (source / "failure.json").exists() or (source / "failure.json").is_symlink()):
        raise ValueError("Body reuse requires a complete construction-only package without a failure")
    body = _json(read_package_file(source, "body-source.json", max_bytes=1 << 20,
                                   expected_sha256=files["body-source.json"]))
    if body.get("source_sha256") != source_sha256:
        raise ValueError("Copied body must match the original uploaded video")
    receipt = {"schema_version": 1, "profile": PROFILE, "origin": "sealed_body_copy",
               "source_sha256": source_sha256, "body_manifest_sha256": sha256(manifest).hexdigest(),
               "inference_calls": 0, "api_calls": 0, "scope": SCOPE}
    validate_reuse_receipt(receipt, source_sha256=source_sha256,
                           body_manifest_sha256=receipt["body_manifest_sha256"])
    _check(deadline)
    target.mkdir(mode=0o700)
    total = len(manifest)
    for name, pin in files.items():
        _check(deadline)
        raw = read_package_file(source, name, max_bytes=min(64 << 20, max(1, MAX_BYTES-total)),
                                expected_sha256=pin)
        total += len(raw)
        if total > MAX_BYTES:
            raise ValueError("Copied body exceeds its original 128 MiB bound")
        write_package_files(target, {name: raw})
    if (read_package_file(source, "manifest.json", max_bytes=1 << 20) != manifest
            or (source / "failure.json").exists() or (source / "failure.json").is_symlink()):
        raise ValueError("Original body changed while copying its sealed evidence")
    _check(deadline)
    write_package_files(target, {"manifest.json": manifest})
    return receipt
