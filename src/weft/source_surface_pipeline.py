"""Saved-input surface-anchor artifacts; cooperative trusted CLI work, not a sandbox."""

from collections import Counter
from hashlib import sha256
from html import escape
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time

from weft.io import local_time
from weft.shared_world_packaging import read_package_file, write_package_files
from weft.world_evidence import MAX_DOCUMENTS, MAX_INPUT_BYTES


PROFILE = "source-surface-anchor-workflow-v1"
MAX_EXTRA_BYTES, MAX_ANCHOR_BYTES = 4 << 20, 8 << 20
MAX_RUNTIME_BYTES, MAX_PACKAGE_BYTES, MAX_FILES = 4 << 20, 80 << 20, 40
MAX_FILE_BYTES = 64 << 20
MAX_METADATA_BYTES, FINAL_RESERVE = 256 << 10, 512 << 10
MAX_SECONDS = 30.
RUNTIME_ROOT = Path(__file__).parent
RUNTIME_FILES = ("__init__.py", "source_surface_pipeline.py", "source_surface_anchors.py", "camera_pose.py",
                 "camera_ground.py", "world_alignment.py", "world_evidence.py", "observations.py", "io.py",
                 "shared_world_packaging.py", "shared_world_evidence.py")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SCOPE = "Conditional surface coordinates only; no body/floor installation or verified contact, load, grasp or physical accuracy."


def _require(value, message):
    if not value:
        raise ValueError("Surface-anchor workflow: " + message)


def _check(deadline):
    now = time.monotonic()
    if now >= deadline:
        raise TimeoutError("Surface-anchor workflow exceeded its cooperative deadline")
    return now


def _path(value):
    path = Path(value).absolute()
    _require(len(path.parts) <= 64 and ".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
             "paths must not contain traversal or symlinks")
    return path


def _encoded(value, limit=MAX_METADATA_BYTES):
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    _require(len(raw) <= limit, "serialized output exceeds its byte bound")
    return raw


def _runtime(deadline):
    files, size = {}, 0
    for name in RUNTIME_FILES:
        _check(deadline)
        root = _path(RUNTIME_ROOT)
        raw = read_package_file(root, name, max_bytes=min(2 << 20, MAX_RUNTIME_BYTES - size))
        size += len(raw)
        files["runtime/weft/" + name] = raw
    return files, {"python": sys.version.split()[0], "numpy": version("numpy")}


def _write(out, files, pins, sizes, *, terminal=False):
    _path(out)
    reserve = 0 if terminal else FINAL_RESERVE
    _require(not set(files) & set(pins), "an output member cannot be replaced")
    _require(len(pins) + len(files) <= MAX_FILES - (0 if terminal else 2)
             and sum(sizes.values()) + sum(map(len, files.values())) <= MAX_PACKAGE_BYTES - reserve, "package byte/file bound exceeded")
    write_package_files(out, files)
    for name, raw in files.items():
        pins[name], sizes[name] = sha256(raw).hexdigest(), len(raw)


def _inventory(out, deadline):
    _path(out)
    pending, files, total, visits = [out], {}, 0, 0
    while pending:
        _check(deadline)
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                _check(deadline)
                visits += 1
                _require(visits <= MAX_FILES * 3, "output directory bound exceeded")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                    continue
                _require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE_BYTES, "output is not a bounded regular file")
                name = Path(entry.path).relative_to(out).as_posix()
                _require(len(files) < MAX_FILES - 1 and total + info.st_size <= MAX_PACKAGE_BYTES, "output package bound exceeded")
                raw = read_package_file(_path(out), name, max_bytes=max(1, info.st_size))
                _check(deadline)
                files[name] = sha256(raw).hexdigest()
                total += len(raw)
    return files, total


def _seal(out, status, deadline, expected=None, *, written=None):
    files, total = _inventory(out, deadline)
    _check(deadline)
    if expected is not None:
        _require(files == expected, "archived inputs, runtime or output changed before terminal seal")
    raw = _encoded({"schema_version": 1, "profile": PROFILE, "status": status, "files": files})
    _require(total + len(raw) <= MAX_PACKAGE_BYTES, "manifest exceeds reserved package bytes")
    target = _path(out)
    _check(deadline)
    write_package_files(target, {"manifest.json": raw})
    if written is not None:
        written["manifest.json"] = (sha256(raw).hexdigest(), len(raw))
    _check(deadline)
    return sha256(raw).hexdigest()


def _preserve_terminal_attempts(out, written, pins, sizes):
    """Remove authoritative names only after preserving this invocation's exact bytes."""
    root = _path(out)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for name in ("manifest.json", "result.json"):
            if name not in written:
                continue
            expected, size = written[name]
            read_package_file(root, name, max_bytes=max(1, size), expected_sha256=expected)
            before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            attempt = "attempted-" + name
            # Exclusive hard links preserve evidence without replacing another artifact.
            os.link(name, attempt, src_dir_fd=descriptor, dst_dir_fd=descriptor, follow_symlinks=False)
            archived = os.stat(attempt, dir_fd=descriptor, follow_symlinks=False)
            current = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            _require(stat.S_ISREG(current.st_mode) and
                     (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                     (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns) ==
                     (archived.st_dev, archived.st_ino, archived.st_size, archived.st_mtime_ns),
                     "terminal attempt changed while preserving its bytes")
            os.unlink(name, dir_fd=descriptor)
            pins.pop(name, None)
            sizes.pop(name, None)
            pins[attempt], sizes[attempt] = expected, size
    finally:
        os.close(descriptor)


def _call_api(documents, source_sha256, optional):
    from weft.source_surface_anchors import reconstruct_source_surface_anchors
    return reconstruct_source_surface_anchors(documents, source_sha256=source_sha256,
        camera_document=optional.get("camera"), surface_document=optional.get("surfaces"), annotation_document=optional.get("annotations"))


def _validate_result(result, source_sha256, documents, optional):
    _require(type(result) is dict and result.get("profile") == "source-surface-anchors-v1"
             and result.get("source_sha256") == source_sha256
             and result.get("status") in ("conditional_anchors_available", "unavailable"), "API result identity/status differs")
    _require(all(result.get(key) is False for key in ("geometry_modified", "body_or_floor_installed", "physical_validation"))
             and all(result.get(key) == "unknown" for key in ("contact", "load", "grasp")), "API result cannot admit geometry or assert physical contact")
    expected = {"observation_document_sha256s": sorted(documents), "camera_sha256": None,
                "surfaces_sha256": None, "annotations_sha256": None}
    for role, raw in optional.items():
        expected[{"camera": "camera_sha256", "surfaces": "surfaces_sha256", "annotations": "annotations_sha256"}[role]] = sha256(raw).hexdigest()
    _require(result.get("bindings") == expected, "API result binds different exact inputs")
    counts, rows, channels = result.get("counts"), result.get("rows"), result.get("channels")
    _require(type(counts) is dict and type(rows) is list and type(channels) is list and len(channels) <= 12
             and 1 <= len(rows) <= 1808 and counts.get("retained_source_frames") == len(rows)
             and counts.get("declared_channels") == len(channels), "API source/channel coverage differs")
    _require(type(counts.get("decoded_frames")) is int and 1 <= counts["decoded_frames"] <= 1800
             and type(counts.get("metadata_frames")) is int and 1 <= counts["metadata_frames"] <= 1808
             and len(rows) == max(counts["decoded_frames"], counts["metadata_frames"]), "API original source denominators differ")
    _require(all(type(row) is dict and type(row.get("source_frame")) is int and row["source_frame"] == index and type(row.get("anchors")) is list
                 and len(row["anchors"]) == len(channels) for index, row in enumerate(rows))
             and counts.get("retained_anchor_slots") == len(rows) * len(channels), "API output dropped original frame/anchor slots")
    channel_ids = [channel.get("id") for channel in channels if type(channel) is dict]
    _require(len(channel_ids) == len(channels) and all(type(item) is str for item in channel_ids)
             and len(set(channel_ids)) == len(channel_ids), "API channel identities differ")
    actual = Counter({"conditional_surface_anchor": 0})
    for row in rows:
        _require(all(type(anchor) is dict for anchor in row["anchors"])
                 and [anchor.get("channel_id") for anchor in row["anchors"]] == channel_ids, "API source-slot channel order differs")
        for anchor in row["anchors"]:
            _require(anchor.get("status") in ("conditional_surface_anchor", "unavailable"), "API anchor status differs")
            if anchor["status"] == "conditional_surface_anchor":
                point = anchor.get("position_world_m")
                _require(anchor.get("reason") is None and type(point) is list and len(point) == 3
                         and all(type(v) in (int, float) and math.isfinite(v) for v in point), "API conditional coordinates are not finite")
            else:
                _require(type(anchor.get("reason")) is str and anchor["reason"] and anchor.get("position_world_m") is None,
                         "API unavailable anchor must retain a reason and null geometry")
            actual[anchor.get("reason") or anchor["status"]] += 1
    _require(counts.get("conditional_anchors") == actual["conditional_surface_anchor"]
             and counts.get("slot_status_counts") == dict(actual)
             and (result["status"] == "conditional_anchors_available") == (actual["conditional_surface_anchor"] > 0),
             "API summary counts differ from retained source slots")


def _render_report(report, result):
    e = lambda value: escape(str(value), quote=True)
    inputs = "".join(f'<tr><td>{e(row["role"])}</td><td><a href="{e(row["artifact"])}">{e(row["artifact"])}</a></td>'
                     f'<td>{row["bytes"]:,}</td><td><code>{e(row["sha256"])}</code></td></tr>' for row in report["inputs"])
    counts = result["counts"]
    reasons = "".join(f'<tr><td>{e(name)}</td><td>{e(count)}</td></tr>' for name, count in sorted(counts["slot_status_counts"].items()))
    preview = []
    # Chronological preview, never a success-selected subset; the complete result stays in JSON.
    for row in result["rows"]:
        for anchor in row["anchors"]:
            if len(preview) >= 40:
                break
            position = anchor.get("position_world_m")
            xyz = ", ".join(f"{v:.5g}" for v in position) if position is not None else "Unavailable"
            preview.append(f'<tr><td>{row["source_frame"]}</td><td>{e(anchor["channel_id"])}</td><td>{e(anchor["status"])}</td>'
                           f'<td>{e(anchor.get("reason") or "Conditional on supplied inputs")}</td><td>{e(xyz)}</td></tr>')
        if len(preview) >= 40:
            break
    missing = ", ".join(result.get("missing_inputs", [])) or "None"
    title = "Conditional coordinates available" if result["status"] == "conditional_anchors_available" else "Completed: no anchors available"
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>Weft surface anchors</title><style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#214f50;background:#f5eadc}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-size:.9rem}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid #bbb;padding:.5rem;overflow-wrap:anywhere}}code{{font-size:.75rem}}a{{color:#214f50}}section{{margin:2rem 0}}.scroll{{overflow:auto}}</style>
<h1>Weft · Surface anchors</h1><p><strong>{e(title)}</strong></p><p>{e(_SCOPE)}</p>
<section><h2>1. Saved source evidence</h2><p>Declared source identity: <code>{e(report["source_sha256"])}</code>. Copied from supplied observations; no source media was decoded or verified here, and no model was run.</p>
<div class="scroll"><table><thead><tr><th>Input</th><th>Exact saved bytes</th><th>Bytes</th><th>SHA-256</th></tr></thead><tbody>{inputs}</tbody></table></div></section>
<section><h2>2. Additional inputs</h2><p>Missing inputs: {e(missing)}. Authored calibration, surface heights and point annotations remain authored assumptions. Metres alone do not calibrate a human body.</p></section>
<section><h2>3. Reconstruct conditional points</h2><p>Retained {counts["retained_source_frames"]:,} source frames and {counts["retained_anchor_slots"]:,} anchor slots. Conditional anchors: {counts["conditional_anchors"]:,}. Missing, occluded, excluded and unsampled slots are preserved.</p>
<table><thead><tr><th>Outcome</th><th>Slots</th></tr></thead><tbody>{reasons}</tbody></table>
<p>First {len(preview)} source-ordered anchor slots; this preview is not the complete result.</p><div class="scroll"><table><thead><tr><th>Frame</th><th>Point</th><th>Status</th><th>Reason</th><th>World XYZ (m)</th></tr></thead><tbody>{''.join(preview)}</tbody></table></div></section>
<section><h2>4. Inspect and continue</h2><p><a href="anchors.json">Complete anchors and coordinate mappings</a> · <a href="report.json">Workflow report</a> · <a href="plan.json">Frozen plan</a> · <a href="manifest.json">Terminal integrity seal</a></p>
<p>No body, floor, collision constraint or support-load task was installed. Existing construction needs an explicit body-to-world scale/origin mapping before these conditional targets can be used. Contact, load and grasp remain unknown.</p>
<p>Execution uses cooperative time checks, not hard process or memory isolation. A failed terminal manifest or failure receipt invalidates any earlier partial report.</p></section></html>'''.encode()
    _require(len(html) <= MAX_METADATA_BYTES, "offline report exceeds its byte bound")
    return html


def prepare_source_surface_anchors(observations, out, *, source_sha256, camera=None, surfaces=None,
                                   annotations=None, max_seconds=30., progress=None):
    """Seal saved-input results in a new directory; all time bounds are cooperative."""
    _require(type(observations) in (list, tuple) and 1 <= len(observations) <= MAX_DOCUMENTS, "supply one to eight observation JSON paths")
    _require(type(source_sha256) is str and _SHA.fullmatch(source_sha256), "exact original source SHA256 required")
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 1 <= max_seconds <= MAX_SECONDS,
             "cooperative workflow seconds must be between one and thirty")
    _require(progress is None or callable(progress), "progress callback must be callable")
    out = _path(out)
    paths = [("observation", _path(value)) for value in observations]
    paths += [(role, _path(value)) for role, value in (("camera", camera), ("surfaces", surfaces), ("annotations", annotations)) if value is not None]
    _require(len({path for _, path in paths}) == len(paths) and all(out != path and out not in path.parents for _, path in paths), "duplicate paths or inputs inside the output directory")
    _require(not out.exists() and out.parent.is_dir(), "output must be new and its parent directory must already exist")
    out.mkdir(mode=0o700, exist_ok=False)
    started = time.monotonic()
    deadline, stage, calls = started + max_seconds, "inputs", 0
    pins, sizes, records, documents, optional, runtime, versions = {}, {}, [], {}, {}, {}, {}
    terminal_writes = {}
    try:
        observed_bytes, extra_bytes = 0, 0
        for role, path in paths:
            _check(deadline)
            remaining = MAX_INPUT_BYTES - observed_bytes if role == "observation" else MAX_EXTRA_BYTES - extra_bytes
            _require(remaining > 0, "input aggregate byte bound exceeded")
            raw = read_package_file(_path(path).parent, path.name, max_bytes=remaining)
            digest = sha256(raw).hexdigest()
            if role == "observation":
                _require(digest not in documents, "duplicate observation document bytes")
                artifact = f"inputs/observations/{len(documents):02d}-{digest}.json"
                documents[digest] = raw
                observed_bytes += len(raw)
            else:
                artifact = f"inputs/{role}.json"
                optional[role] = raw
                extra_bytes += len(raw)
            _write(out, {artifact: raw}, pins, sizes)
            records.append({"role": role, "source_path": str(path), "artifact": artifact, "bytes": len(raw), "sha256": digest})
        stage = "runtime"
        runtime, versions = _runtime(deadline)
        _write(out, runtime, pins, sizes)
        _require(sum(sizes.values()) + MAX_ANCHOR_BYTES + FINAL_RESERVE <= MAX_PACKAGE_BYTES, "inputs/runtime leave insufficient bounded result space")
        plan = {"schema_version": 1, "profile": PROFILE, "status": "frozen", "created_at": local_time(),
                "source_sha256": source_sha256, "inputs": records, "input_files": {r["artifact"]: r["sha256"] for r in records},
                "runtime_files": {name: sha256(raw).hexdigest() for name, raw in runtime.items()}, "runtime_versions": versions,
                "bounds": {"maximum_observation_documents": MAX_DOCUMENTS, "maximum_observation_bytes": MAX_INPUT_BYTES,
                           "maximum_additional_bytes": MAX_EXTRA_BYTES, "maximum_anchor_bytes": MAX_ANCHOR_BYTES,
                           "maximum_package_bytes": MAX_PACKAGE_BYTES, "maximum_files": MAX_FILES,
                           "maximum_workflow_seconds": max_seconds, "api_cooperative_seconds": 10.,
                           "deadline_scope": "Cooperative trusted-code/I-O checks, not hard process or memory isolation"},
                "scope": _SCOPE, "automatic_installation": False}
        _write(out, {"plan.json": _encoded(plan)}, pins, sizes)
        if progress:
            progress("inputs", {"out": str(out), "observation_documents": len(documents), "input_bytes": observed_bytes + extra_bytes})
        _check(deadline)
        stage = "reconstruction"
        call_started = time.monotonic()
        calls += 1
        result = _call_api(documents, source_sha256, optional)
        api_seconds = time.monotonic() - call_started
        _check(deadline)
        _validate_result(result, source_sha256, documents, optional)
        _write(out, {"anchors.json": _encoded(result, MAX_ANCHOR_BYTES)}, pins, sizes)
        if progress:
            progress("anchors", {"diagnostic_status": result["status"], "counts": result["counts"], "simulation_ready": False})
        stage = "report"
        report = {"schema_version": 1, "profile": PROFILE, "diagnostic_status": result["status"], "source_sha256": source_sha256,
                  "inputs": records, "counts": result["counts"], "missing_inputs": result.get("missing_inputs", []),
                  "runtime_versions": versions, "api_seconds": api_seconds, "source_surface_api_calls": calls,
                  "source_and_runtime_recheck": "Required before terminal seal", "scope": _SCOPE,
                  "simulation_ready": False, "body_or_floor_installed": False, "physical_validation": False,
                  "contact": "unknown", "load": "unknown", "grasp": "unknown", "deadline_scope": plan["bounds"]["deadline_scope"]}
        html = _render_report(report, result)
        if progress:
            progress("report", {"report_path": str(out / "index.html"), "simulation_ready": False})
        stage = "integrity"
        for row in records:
            _check(deadline)
            path = _path(row["source_path"])
            read_package_file(path.parent, path.name, max_bytes=max(1, row["bytes"]), expected_sha256=row["sha256"])
        for name, raw in runtime.items():
            _check(deadline)
            read_package_file(_path(RUNTIME_ROOT), Path(name).name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
        report["source_and_runtime_recheck"] = "Exact selected original bytes unchanged"
        _write(out, {"report.json": _encoded(report), "index.html": html}, pins, sizes)
        summary = {"schema_version": 1, "profile": PROFILE, "status": "completed", "created_at": local_time(),
                   "out": str(out), "diagnostic_status": result["status"], "counts": result["counts"],
                   "report_path": str(out / "index.html"), "anchors_path": str(out / "anchors.json"),
                   "source_surface_api_calls": calls, "elapsed_before_seal_seconds": time.monotonic() - started,
                   "simulation_ready": False, "body_or_floor_installed": False, "physical_validation": False,
                   "contact": "unknown", "load": "unknown", "grasp": "unknown", "scope": _SCOPE}
        _write(out, {"result.json": _encoded(summary)}, pins, sizes)
        terminal_writes["result.json"] = (pins["result.json"], sizes["result.json"])
        stage = "seal"
        manifest_sha = _seal(out, "completed", deadline, pins, written=terminal_writes)
        finished = _check(deadline)
        return {**summary, "manifest_sha256": manifest_sha, "sealed": True, "elapsed_seconds": finished - started}
    except (Exception, KeyboardInterrupt) as error:
        failure = {"schema_version": 1, "profile": PROFILE, "status": "failed", "created_at": local_time(),
                   "out": str(out), "source_sha256": source_sha256, "error": {"stage": stage, "type": type(error).__name__, "message": str(error)[:2000]},
                   "source_surface_api_calls_started": calls, "elapsed_seconds": time.monotonic() - started,
                   "simulation_ready": False, "body_or_floor_installed": False, "physical_validation": False,
                   "report_path": None, "scope": _SCOPE}
        try:
            _preserve_terminal_attempts(out, terminal_writes, pins, sizes)
            _write(out, {"failure.json": _encoded(failure)}, pins, sizes, terminal=True)
            # Failure bytes survive even when the original deadline no longer allows sealing.
            manifest_sha = _seal(out, "failed", deadline)
        except Exception as sealing_error:
            return {**failure, "sealed": False, "manifest_sha256": None,
                    "sealing_error": {"type": type(sealing_error).__name__, "message": str(sealing_error)[:1000]}}
        return {**failure, "sealed": True, "manifest_sha256": manifest_sha}
