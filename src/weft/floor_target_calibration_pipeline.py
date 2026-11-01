"""Saved measurements to nominal calibration, with one isolated numerical call."""

from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

from weft.floor_target_calibration_worker import RUNTIME_FILES, run_floor_target_worker
from weft.io import local_time
from weft.shared_world_packaging import read_package_file, validate_package_layout, write_package_files
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


PROFILE = "floor-target-calibration-package-v1"
MAX_INPUT_BYTES, MAX_INPUT_FILES = 4 << 20, 19
MAX_RUNTIME_BYTES, MAX_PACKAGE_BYTES, MAX_FILES = 4 << 20, 32 << 20, 96
MAX_WORKER_BYTES, MAX_WORKER_FILES = 6 << 20, 16
MAX_METADATA_BYTES, FINAL_RESERVE = 256 << 10, 512 << 10
MAX_SECONDS = 30.
RUNTIME_ROOT = Path(__file__).parent
EXTRA_RUNTIME_FILES = ("floor_target_calibration_pipeline.py", "floor_target_calibration_report.py", "cli.py",
                       "shared_world_packaging.py", "shared_world_evidence.py", "assets/brand.css")
CALIBRATION_STATUSES = {"qualified_nominal", "unavailable", "rejected", "ambiguous"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
SCOPE = ("Source-bound nominal camera and floor only. Measurement claims are not authenticated, uncertainty is not "
         "propagated, and no body, ground placement, collision, contact, grasp or load is installed.")


def _require(value, message):
    if not value:
        raise ValueError("Floor calibration: " + message)


def _check(deadline, reserve=0.):
    if time.monotonic() + reserve >= deadline:
        raise TimeoutError("Floor calibration workflow deadline exceeded")


def _path(value):
    path = Path(value).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts and
             not any(p.is_symlink() for p in (path, *path.parents)), "paths must not traverse symlinks or parent components")
    return path


def _identity(path):
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode), "input must be a regular file")
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _encoded(value, limit=MAX_METADATA_BYTES):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    _require(len(raw) <= limit, "serialized metadata exceeds its bound")
    return raw


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    _check_tree(value)
    _require(type(value) is dict, "JSON object required")
    return value


def _read(path, maximum, identities, deadline):
    _check(deadline)
    path = _path(path)
    before = _identity(path)
    _require(0 < before[2] <= maximum, "input exceeds its remaining byte bound")
    raw = read_package_file(path.parent, path.name, max_bytes=maximum)
    _require(_identity(path) == before, "input changed while reading")
    identities[path] = before
    _check(deadline)
    return raw


def _recheck(identities, deadline):
    for path, identity in identities.items():
        _check(deadline)
        _require(_identity(_path(path)) == identity, "selected input or implementation changed before publication")


def _write(out, files, pins, sizes, deadline):
    _check(deadline)
    validate_package_layout([*pins, *files])
    _require(not set(files) & set(pins), "output files cannot be replaced")
    _require(len(pins) + len(files) <= MAX_FILES - 2 and
             sum(sizes.values()) + sum(map(len, files.values())) <= MAX_PACKAGE_BYTES - FINAL_RESERVE,
             "complete output exceeds its file/byte bound")
    for name, raw in files.items():
        _require(type(raw) is bytes, "immutable output bytes required")
        _check(deadline)
        write_package_files(out, {name: raw})
        pins[name], sizes[name] = sha256(raw).hexdigest(), len(raw)


def _runtime(identities, deadline):
    names = set(RUNTIME_FILES) | {"runtime/weft/" + name for name in EXTRA_RUNTIME_FILES}
    _require(len(names) <= 40 and all(name.startswith("runtime/weft/") for name in names), "runtime closure is not bounded")
    validate_package_layout(names)
    files, total = {}, 0
    for name in sorted(names):
        raw = _read(RUNTIME_ROOT / name.removeprefix("runtime/weft/"), MAX_RUNTIME_BYTES - total, identities, deadline)
        files[name] = raw
        total += len(raw)
    versions = {"python": sys.version.split()[0], "numpy": version("numpy"), "opencv": version("opencv-contrib-python")}
    return files, versions


def _adopt_worker(out, result, pins, sizes, deadline):
    _require(type(result) is dict and result.get("status") == "completed", "isolated worker did not complete; its evidence is preserved")
    members = result.get("output_files")
    _require(type(members) is dict and 1 <= len(members) <= MAX_WORKER_FILES and
             all(type(name) is str and name.startswith("worker/") and type(pin) is str and _SHA.fullmatch(pin)
                 for name, pin in members.items()), "worker output inventory is invalid")
    validate_package_layout([*pins, *members])
    _require(not set(members) & set(pins), "worker overwrote parent-owned files")
    values, total = {}, 0
    for name, expected in sorted(members.items()):
        _check(deadline)
        raw = read_package_file(_path(out), name, max_bytes=MAX_WORKER_BYTES - total, expected_sha256=expected)
        total += len(raw)
        values[name] = raw
        pins[name], sizes[name] = expected, len(raw)
    _require(len(pins) <= MAX_FILES - 2 and sum(sizes.values()) <= MAX_PACKAGE_BYTES - FINAL_RESERVE,
             "worker evidence exceeds package bounds")
    _require("worker/calibration.json" in values, "worker omitted its calibration result")
    metadata = _json(values["worker/calibration.json"])
    status = metadata.get("status")
    _require(metadata.get("profile") == "floor-target-calibration-result-v1" and status in CALIBRATION_STATUSES
             and result.get("calibration_status") == status, "worker and calibration status disagree")
    aliases = {"calibration.json": values["worker/calibration.json"]}
    for name in ("camera.json", "ground.json"):
        _require(("worker/" + name in values) == (status == "qualified_nominal"), "nominal output pair disagrees with qualification")
        if status == "qualified_nominal":
            aliases[name] = values["worker/" + name]
    return aliases, metadata


def _inventory(out, deadline):
    pending, pins, total, visits = [_path(out)], {}, 0, 0
    while pending:
        _check(deadline)
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                visits += 1
                _require(visits <= MAX_FILES * 3, "output directory count exceeds its bound")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                    continue
                _require(stat.S_ISREG(info.st_mode) and len(pins) < MAX_FILES - 1
                         and total + info.st_size <= MAX_PACKAGE_BYTES - FINAL_RESERVE,
                         "output contains unsafe or oversized files")
                name = Path(entry.path).relative_to(out).as_posix()
                raw = read_package_file(out, name, max_bytes=max(1, info.st_size))
                _check(deadline)
                pins[name], total = sha256(raw).hexdigest(), total + len(raw)
    return pins, total


def prepare_floor_target_calibration_package(observations, out, *, source_sha256, request, measurements=(),
                                             camera=None, ground=None, max_seconds=30., progress=None):
    """Retain selected inputs and run one bounded worker; never install calibration."""
    _require(type(observations) in (list, tuple) and 1 <= len(observations) <= 8, "one to eight observation paths required")
    _require(type(measurements) in (list, tuple) and len(measurements) <= 8, "at most eight measurement paths allowed")
    _require(type(source_sha256) is str and _SHA.fullmatch(source_sha256), "exact original source SHA256 required")
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 10 <= max_seconds <= MAX_SECONDS,
             "workflow seconds must be between ten and thirty")
    _require(progress is None or callable(progress), "progress must be callable")
    target = _path(out)
    paths = [("observation", _path(p)) for p in observations] + [("measurement", _path(p)) for p in measurements]
    paths += [("request", _path(request))]
    paths += [(role, _path(p)) for role, p in (("camera", camera), ("ground", ground)) if p is not None]
    _require(len(paths) <= MAX_INPUT_FILES and len({p for _, p in paths}) == len(paths), "input paths must be distinct and bounded")
    _require(all(p != target and target not in p.parents for _, p in paths), "inputs cannot be inside the new output")
    _require(not target.exists() and target.parent.is_dir(), "output must be new with an existing parent directory")
    target.mkdir(mode=0o700)
    started = time.monotonic()
    deadline, phase = started + max_seconds, "inputs"
    identities, pins, sizes, inputs, records = {}, {}, {}, {}, []
    terminal = None

    def emit(stage, detail):
        _check(deadline)
        if progress:
            progress(stage, detail)

    try:
        emit(phase, "Preserve exact source observations and supplied measurements")
        for role, path in paths:
            remaining = MAX_INPUT_BYTES - sum(map(len, inputs.values()))
            raw = _read(path, min(remaining, 64 << 10) if role == "request" else remaining, identities, deadline)
            digest = sha256(raw).hexdigest()
            name = f"inputs/{role}s/{digest}.json" if role in ("observation", "measurement") else f"inputs/{role}.json"
            _require(name not in inputs, "duplicate document bytes within an input role")
            inputs[name] = raw
            records.append({"role": role, "path": name, "sha256": digest, "bytes": len(raw)})
        index = _encoded({"schema_version": 1, "profile": "floor-target-calibration-input-files-v1", "files": records})
        _write(target, inputs | {"input-files.json": index}, pins, sizes, deadline)
        phase = "runtime"
        runtime, versions = _runtime(identities, deadline)
        _write(target, runtime, pins, sizes, deadline)
        plan = {"schema_version": 1, "profile": "floor-target-calibration-worker-plan-v1", "status": "frozen",
                "source_sha256": source_sha256, "input_files": {name: pins[name] for name in inputs},
                "runtime_files": {name: pins[name] for name in RUNTIME_FILES}, "runtime_versions": versions,
                "settings": {"max_seconds": 5.}}
        _write(target, {"worker-plan.json": _encoded(plan)}, pins, sizes, deadline)
        phase = "calibration"
        emit(phase, "Check measurements and retain every nominal camera alternative")
        _check(deadline, reserve=5.)
        worker = run_floor_target_worker(target, expected_plan_sha256=pins["worker-plan.json"],
                                        max_seconds=min(10., deadline-time.monotonic()-5.))
        aliases, metadata = _adopt_worker(target, worker, pins, sizes, deadline)
        _write(target, aliases, pins, sizes, deadline)
        phase = "report"
        emit(phase, "Explain source coverage, alternatives and remaining uncertainty")
        from weft.floor_target_calibration_report import build_floor_target_calibration_report
        _check(deadline, reserve=2.)
        report = build_floor_target_calibration_report(aliases | {"inputs/request.json": inputs["inputs/request.json"],
            "input-files.json": index}, max_seconds=min(5., deadline-time.monotonic()-2.))
        _require(type(report) is dict and "index.html" in report and "brand.css" in report and
                 len(report) <= 16 and all(type(raw) is bytes for raw in report.values()) and
                 sum(map(len, report.values())) <= 4 << 20, "offline report exceeded its output contract")
        validate_package_layout(report)
        _require(report["brand.css"] == runtime["runtime/weft/assets/brand.css"], "report stylesheet differs from the retained implementation")
        _write(target, {"inspection/" + name: raw for name, raw in report.items()}, pins, sizes, deadline)
        status = metadata["status"]
        result = {"schema_version": 1, "profile": PROFILE, "status": status, "sealed": True,
                  "output_path": str(target), "report_path": str(target/"inspection/index.html"),
                  "calibration_path": str(target/"calibration.json"),
                  "camera_path": str(target/"camera.json") if status == "qualified_nominal" else None,
                  "ground_path": str(target/"ground.json") if status == "qualified_nominal" else None,
                  "source_sha256": source_sha256, "counts": metadata.get("counts"), "scope": SCOPE,
                  "input_bytes": sum(map(len, inputs.values())), "input_files": len(inputs),
                  "runtime_versions": versions, "worker_receipt_path": str(target/"worker/worker.json"),
                  "created_at_local": local_time(), "seconds": time.monotonic()-started}
        _write(target, {"result.json": _encoded(result | {"sealed": False, "publication": "prepared_pending_terminal_seal"})}, pins, sizes, deadline)
        phase = "seal"
        emit(phase, "Save one immutable package; do not apply it to a body or floor")
        _recheck(identities, deadline)
        actual, total = _inventory(target, deadline)
        _require(actual == pins, "saved inputs, worker evidence or report changed before publication")
        terminal = _encoded({"schema_version": 1, "profile": PROFILE, "status": status, "files": pins})
        _require(len(pins)+1 <= MAX_FILES and total+len(terminal) <= MAX_PACKAGE_BYTES, "terminal seal exceeds package bounds")
        _check(deadline)
        write_package_files(target, {"manifest.json": terminal})
        result["manifest_sha256"] = sha256(terminal).hexdigest()
        return result
    except Exception as error:
        failure = {"schema_version": 1, "profile": PROFILE, "status": "failed", "sealed": False,
                   "output_path": str(target), "phase": phase, "error": {"type": type(error).__name__, "message": str(error)[:1000]},
                   "scope": SCOPE, "seconds": time.monotonic()-started, "input_files": records}
        marker = target/"manifest.json"
        if terminal is not None and marker.is_file() and read_package_file(target, "manifest.json", max_bytes=MAX_METADATA_BYTES) == terminal:
            sibling = failure | {"status": "caller_failed_after_terminal_publication", "artifact_unchanged": True,
                                 "artifact_manifest_sha256": sha256(terminal).hexdigest()}
            write_package_files(target.parent, {target.name+".caller-failure.json": _encoded(sibling)})
            raise
        write_package_files(target, {"failure.json": _encoded(failure)})
        return failure
