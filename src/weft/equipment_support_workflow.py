"""Compare saved equipment candidates in the existing source-only selector."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import sys
import time

from weft.equipment_support_snapshot import SCOPE, _path, load_equipment_support_snapshot
from weft.io import local_time
from weft.shared_world_packaging import read_package_file, write_package_files


PROFILE = "equipment-support-workflow-v1"
MAX_OUTPUT, MAX_FILES = 64 << 20, 384
_REPORT_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}\.(?:html|css|json)\Z")


def _require(value, message):
    if not value:
        raise ValueError("Equipment support workflow: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Equipment support workflow exceeded its cooperative deadline")


def _encode(value, limit=1 << 20):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    _require(len(raw) <= limit, "saved result exceeds its byte bound")
    return raw


def _runtime_sources(deadline):
    package = Path(__file__).parent
    modules = [(name, module) for name, module in tuple(sys.modules.items())
               if name == "weft" or name.startswith("weft.")]
    _require(len(modules) <= 128, "loaded local runtime exceeds 128 modules")
    files, total = {}, 0
    for module_name, module in sorted(modules):
        _check(deadline)
        relative = module_name[5:].replace(".", "/") if module_name != "weft" else ""
        name = relative + ".py" if relative and (package / (relative + ".py")).is_file() else (relative + "/" if relative else "") + "__init__.py"
        _require(getattr(module, "__file__", None) is not None
                 and _path(module.__file__) == _path(package / name), "loaded Weft module is outside the recorded package")
        raw = read_package_file(package, name, max_bytes=512 << 10)
        total += len(raw)
        _require(total <= 8 << 20, "loaded runtime source exceeds eight MiB")
        files[name] = raw
    for name, limit in (("__init__.py", 32 << 10), ("cli.py", 512 << 10), ("assets/brand.css", 32 << 10)):
        _check(deadline)
        files[name] = read_package_file(package, name, max_bytes=limit)
    return files


def prepare_equipment_support(run, out, *, fixed_camera=False, expected_manifest_sha256=None,
                              max_seconds=20., progress=None):
    """Reuse sealed image results; require an explicit camera assumption and new output."""
    _require(fixed_camera is True, "explicit --fixed-camera assumption required; camera calibration is not inferred")
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 15 <= max_seconds <= 30,
             "cooperative time bound must be between fifteen and thirty seconds")
    start = time.monotonic()
    deadline = start + max_seconds
    source, target = _path(run), _path(out)
    _require(source.is_dir() and target.parent.is_dir() and not target.exists(),
             "existing source and parent plus a new output directory required")
    _require(source != target and source not in target.parents and target not in source.parents,
             "comparison must be outside the original study")
    def emit(stage, detail):
        if progress is not None:
            progress(stage, detail)
        _check(deadline)
    emit("inputs", "Restore sealed diagnostics and verify source bindings; no video or mask processing")
    snapshot = load_equipment_support_snapshot(source, expected_manifest_sha256=expected_manifest_sha256, deadline=deadline)
    prior = {"mode": "fixed_source_camera", "status": "authored_unverified",
             "source_sha256": snapshot.pose["source_sha256"],
             "rationale": "Explicit --fixed-camera user assumption; not measured camera calibration"}
    from weft.exercise_support_evidence import CONTRACT, estimate_exercise_support
    emit("selection", "Keep the original windows; image contradictions restrict candidates, not physics")
    # Reserve the estimator's full existing deadline rather than silently exceeding this workflow's limit.
    _require(deadline - time.monotonic() >= CONTRACT["maximum_seconds"],
             "not enough remaining time for the bounded selector; no selection was started")
    support = estimate_exercise_support(snapshot.pose, camera_prior=prior,
        equipment_observations=snapshot.silhouette.equipment_document, equipment_ids=snapshot.equipment_ids,
        equipment_source_equivalence=snapshot.silhouette.source_equivalence, equipment_silhouette=snapshot.silhouette)
    support_raw = _encode(support, (8 << 20) + 1)
    emit("report", "Explain every original pair, decision, missing input and unchanged physical uncertainty")
    from weft.equipment_support_report import build_equipment_support_report
    report = build_equipment_support_report(support_raw, max_seconds=min(5., deadline - time.monotonic()))
    _require(type(report) is dict and 1 <= len(report) <= 32 and "index.html" in report
             and all(type(name) is str and (_REPORT_NAME.fullmatch(name) or name == "window-index.js")
                     and name not in {"manifest.json", "result.json"}
                     and type(raw) is bytes for name, raw in report.items()), "report has unsupported members")
    _require(report.get("support-evidence.json") == support_raw
             and sum(map(len, report.values())) <= 16 << 20, "report must retain the exact bounded support result")
    runtime = _runtime_sources(deadline)
    _require(report.get("brand.css") == runtime["assets/brand.css"], "report stylesheet differs from archived bytes")
    files = {**report, **{"source/" + name: raw for name, raw in snapshot.source_files.items()},
             **{"generator/" + name: raw for name, raw in runtime.items()}}
    result = {"schema_version": 1, "profile": PROFILE, "status": "completed", "sealed": True,
        "report_path": str(target / "index.html"), "source_manifest_sha256": snapshot.source_manifest_sha256,
        "support_evidence_sha256": sha256(support_raw).hexdigest(), "read_work": snapshot.read_work,
        "camera_prior": prior, "verification_scope": SCOPE,
        "source_member_sha256s": {name: sha256(raw).hexdigest() for name, raw in snapshot.source_files.items()},
        "generator_sha256s": {name: sha256(raw).hexdigest() for name, raw in runtime.items()},
        "generator_scope": "All loaded local Weft module sources, CLI wrapper and exact stylesheet; unexecuted optional imports are not a standalone runtime claim",
        "python_version": sys.version, "source_decodes": 0, "raster_decodes": 0, "pose_model_calls": 0,
        "native_calls": 0, "api_calls": 0, "contact": "unknown", "support": "unknown", "grasp": "unknown", "load": "unknown",
        "physical_validation": "unverified", "body_or_floor_installed": False,
        "time_limit_kind": "cooperative_not_hard_process_limit", "prepared_seconds": time.monotonic() - start}
    files["result.json"] = _encode(result)
    manifest = {"schema_version": 1, "profile": PROFILE, "status": "completed", "created_at_local": local_time(),
                "source_manifest_sha256": snapshot.source_manifest_sha256,
                "files": {name: sha256(raw).hexdigest() for name, raw in sorted(files.items())}}
    files["manifest.json"] = _encode(manifest)
    _require(len(files) <= MAX_FILES and sum(map(len, files.values())) <= MAX_OUTPUT,
             "complete comparison exceeds file/byte bounds")
    emit("seal", "Write a separate comparison; leave the original completed study unchanged")
    snapshot.recheck(deadline)
    _check(deadline)
    target.mkdir(mode=0o700)
    # The completed seal is last; interruptions leave an unsealed directory, never a completed result.
    write_package_files(target, {name: raw for name, raw in files.items() if name != "manifest.json"})
    _check(deadline)
    write_package_files(target, {"manifest.json": files["manifest.json"]})
    return result
