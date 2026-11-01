"""Inspect sealed image-evidence snapshots without repeating reconstruction."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from weft.io import local_time
from weft.shared_world_packaging import read_package_file, write_package_files


PROFILE = "equipment-silhouette-inspection-v1"
MEMBERS = {"silhouette/metadata.json": 256 << 10, "silhouette/frames.jsonl": (8 << 20) + 1800}
MAX_MANIFEST = 1 << 20
MAX_INPUT = (9 << 20) + 1801
MAX_OUTPUT, MAX_FILES = 20 << 20, 48
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REPORT_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}\.(?:html|css|json|jsonl)\Z")
_RESERVED = {"manifest.json", "result.json", "source-manifest.json"}
_SCOPE = "Selected saved metadata and frame bytes only; original source media, masks, geometry and physical contact are not revalidated"


def _require(value, message):
    if not value:
        raise ValueError("Equipment evidence inspection: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Equipment evidence inspection exceeded its cooperative deadline")


def _path(value):
    path = Path(value).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts
             and not any(part.is_symlink() for part in (path, *path.parents)), "paths cannot traverse symlinks")
    return path


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "duplicate manifest JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=pairs)
        json.dumps(value, allow_nan=False)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Equipment evidence inspection requires strict finite JSON") from error
    _require(type(value) is dict, "manifest must be a JSON object")
    return value


def _encoded(value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _require(len(raw) <= MAX_MANIFEST, "inspection metadata exceeds one MiB")
    return raw


def _identity(path):
    stat = _path(path).stat(follow_symlinks=False)
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def prepare_equipment_silhouette_inspection(run, out, *, expected_manifest_sha256=None,
                                           max_seconds=10., progress=None):
    """Create a new offline report from two pinned saved members; never replay a study."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 30,
             "time bound must be within thirty seconds")
    _require(expected_manifest_sha256 is None or type(expected_manifest_sha256) is str
             and _SHA.fullmatch(expected_manifest_sha256), "invalid expected manifest SHA256")
    start = time.monotonic()
    deadline = start + max_seconds
    source, target = _path(run), _path(out)
    _require(source.is_dir() and target.parent.is_dir() and not target.exists(),
             "existing source and parent plus a new output directory are required")
    _require(source != target and source not in target.parents and target not in source.parents,
             "output must be outside the original study")
    selected, identities, read_bytes = {}, {}, 0

    def emit(stage, detail):
        if progress is not None:
            progress(stage, detail)
        _check(deadline)

    def read(name, maximum, expected=None):
        nonlocal read_bytes
        _check(deadline)
        before = _identity(source / name)
        raw = read_package_file(source, name, max_bytes=min(maximum, MAX_INPUT - read_bytes),
                                expected_sha256=expected)
        after = _identity(source / name)
        _require(before == after, "selected input changed while reading")
        identities[name] = after
        read_bytes += len(raw)
        _check(deadline)
        return raw

    emit("inputs", "Read the saved seal and two image-evidence members; no source or mask processing")
    try:
        manifest_raw = read("manifest.json", MAX_MANIFEST, expected_manifest_sha256)
    except FileNotFoundError as error:
        raise ValueError("Equipment evidence study has no completed manifest. Inspect its failure.json or process receipt; a failed or interrupted study cannot produce this report.") from error
    manifest = _json(manifest_raw)
    _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
             and manifest.get("status") == "completed" and type(manifest.get("profile")) is str
             and 0 < len(manifest["profile"]) <= 120, "a completed versioned study seal is required")
    pins = manifest.get("files")
    _require(type(pins) is dict and 2 <= len(pins) <= 384,
             "study seal needs a bounded file map")
    for name in MEMBERS:
        _require(type(pins.get(name)) is str and _SHA.fullmatch(pins[name]),
                 "study seal omits an exact image-evidence member")
        selected[name] = read(name, MEMBERS[name], pins[name])
    _require(sum(map(len, selected.values())) <= (8 << 20) + 1801,
             "saved evidence exceeds eight MiB plus JSONL separators")

    emit("report", "Build frame-by-frame inspection; overlap is not proof of contact")
    from weft.equipment_silhouette_report import build_equipment_silhouette_report
    report = build_equipment_silhouette_report(selected["silhouette/metadata.json"],
        selected["silhouette/frames.jsonl"], max_seconds=min(5., deadline - time.monotonic()))
    _require(type(report) is dict and 1 <= len(report) <= 32 and "index.html" in report
             and all(type(name) is str and _REPORT_NAME.fullmatch(name) and name not in _RESERVED
                     and type(raw) is bytes for name, raw in report.items()), "report has unsupported output members")
    _require(sum(map(len, report.values())) <= 16 << 20, "report exceeds sixteen MiB")
    _require(report.get("equipment-metadata.json") == selected["silhouette/metadata.json"]
             and report.get("equipment-frames.jsonl") == selected["silhouette/frames.jsonl"],
             "report must retain both exact saved inputs")
    files = {**report, "source-manifest.json": manifest_raw}
    runtime = {}
    package = Path(__file__).parent
    for name in ("equipment_silhouette_inspection.py", "equipment_silhouette_report.py", "shared_world_packaging.py", "io.py",
                 "assets/brand.css"):
        _check(deadline)
        raw = read_package_file(package, name, max_bytes=(32 if name == "assets/brand.css" else 512) << 10)
        if name == "assets/brand.css":
            _require(report.get("brand.css") == raw, "rendered brand stylesheet differs from the archived asset")
        runtime[name] = sha256(raw).hexdigest()
        files["generator/" + name] = raw
    emit("seal", "Preserve a separate report; the original study is unchanged")
    for name, expected in identities.items():
        _require(_identity(source / name) == expected, "selected source changed before report publication")
        _check(deadline)
    result = {"schema_version": 1, "profile": PROFILE, "status": "completed", "sealed": True,
        "report_path": str(target / "index.html"), "source_manifest_sha256": sha256(manifest_raw).hexdigest(),
        "source_profile": manifest["profile"], "selected_members": {name: pins[name] for name in MEMBERS},
        "verification_scope": _SCOPE, "generator_sha256": runtime,
        "read_work": {"selected_files": len(identities), "selected_bytes": read_bytes},
        "api_calls": 0, "source_decodes": 0, "raster_decodes": 0, "native_calls": 0,
        "time_limit_kind": "cooperative_not_hard_process_limit", "prepared_seconds": time.monotonic() - start}
    files["result.json"] = _encoded(result)
    seal = {"schema_version": 1, "profile": PROFILE, "status": "completed", "created_at_local": local_time(),
            "verification_scope": _SCOPE, "source_manifest_sha256": result["source_manifest_sha256"],
            "files": {name: sha256(raw).hexdigest() for name, raw in sorted(files.items())}}
    files["manifest.json"] = _encoded(seal)
    _require(len(files) <= MAX_FILES and sum(map(len, files.values())) <= MAX_OUTPUT,
             "complete inspection exceeds its output bounds")
    _check(deadline)
    target.mkdir(mode=0o700)
    # The terminal seal is last; interrupted writes remain unsealed and are never reused.
    write_package_files(target, {name: raw for name, raw in files.items() if name != "manifest.json"})
    _check(deadline)
    write_package_files(target, {"manifest.json": files["manifest.json"]})
    return result
