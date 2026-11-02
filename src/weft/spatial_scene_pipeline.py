"""Seal one authored spatial hypothesis; runtime admission remains a separate step."""

from hashlib import sha256
from importlib.metadata import version
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import time

from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, validate_package_layout, write_package_files
from weft.spatial_scene_inputs import (
    MAX_CONFIG_BYTES, MAX_FILE_BYTES, MAX_FILES as MAX_SOURCE_FILES, _json, _path, prepare_spatial_scene_inputs,
)


PROFILE = "spatial-scene-construction-v1"
MAX_INPUT_BYTES = 64 << 20
MAX_OUTPUT_BYTES = 64 << 20
MAX_OUTPUT_FILES = MAX_SOURCE_FILES + 64
MAX_METADATA_BYTES = 1 << 20
MAX_RUNTIME_BYTES = 4 << 20
RUNTIME_ROOT = Path(__file__).parent
RUNTIME_FILES = (
    "__init__.py", "spatial_scene_pipeline.py", "spatial_scene_inputs.py", "scene.py", "spatial_rigid.py",
    "brand.py", "assets/palette.json", "io.py", "shared_world_packaging.py", "shared_world_evidence.py", "world_evidence.py",
    "source_equivalence.py", "observations.py",
)
_MEDIA_SUFFIXES = {".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv"}
_PREVIEWS = ("tracks.gif", "tracks-contact-sheet.png")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_SCOPE = ("One authored 3D construction hypothesis, not recovered depth, equivalent planar physics, "
          "a trained policy, runtime admission or physical validation.")


def _require(condition, message):
    if not condition:
        raise ValueError("Spatial scene package: " + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Spatial scene packaging exceeded its cooperative time bound")


def _bytes(value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _require(len(raw) <= MAX_METADATA_BYTES, "metadata exceeds 1 MiB")
    return raw


def _runtime(deadline):
    files, total = {}, 0
    for name in RUNTIME_FILES:
        _check(deadline)
        raw = read_package_file(RUNTIME_ROOT, name, max_bytes=min(2 << 20, MAX_RUNTIME_BYTES - total))
        total += len(raw)
        files[name] = raw
    versions = {"python": sys.version.split()[0],
                **{name: version(name) for name in ("numpy", "mujoco", "gymnasium", "pymunk", "Pillow")}}
    return files, versions


def _source_documents(files, prefixes, source_sha256, deadline):
    aliases, rows, documents = {}, [], {}
    for leaf in ("evidence.json", "observations.json"):
        _check(deadline)
        paths = [prefix + leaf for prefix in prefixes if prefix + leaf in files]
        row = {"artifact": leaf, "status": "unavailable", "source_artifacts": paths,
               "archive_artifacts": ["inputs/planar/" + path for path in paths],
               "scope": "Original source image/2D evidence, not observations of the authored 3D scene; "
                        "source IDs, coordinates, clocks, partitions and unknowns remain unchanged"}
        rows.append(row)
        if len({sha256(files[path]).hexdigest() for path in paths}) != 1:
            row["reason"] = "Missing or ambiguous saved source document"
            continue
        try:
            document = _json(files[paths[0]], MAX_FILE_BYTES)
            if leaf == "observations.json":
                from weft.observations import validate_observations
                validate_observations(document)
                _require(document["provenance"].get("adapter") in {"planar", "appearance"},
                         "observations are not a known source image/2D adapter")
                pin = document["source"]["sha256"]
                row.update(representation=document["profile"], time=document["time"])
            else:
                _require(document.get("profile") in {"planar-rigid-v1", "planar-rigid-v2"}
                         and type(document.get("schema_version")) is int and document["schema_version"] in {1, 2}
                         and all(isinstance(document.get(key), list) for key in ("generation", "development_validation")),
                         "evidence is not a known planar source document")
                pin = document.get("source_sha256")
                row["representation"] = document["profile"]
            _require(type(pin) is str and _SHA.fullmatch(pin), "source document lacks an exact source hash")
            _require(source_sha256 is None or pin == source_sha256, "source document differs from bound media identity")
            row.update(source_sha256=pin, sha256=sha256(files[paths[0]]).hexdigest())
            documents[leaf] = row
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            row["reason"] = str(error)[:300]
        _check(deadline)
    if len({row["source_sha256"] for row in documents.values()}) > 1:
        for row in documents.values():
            row["reason"] = "Saved evidence and observation source identities disagree"
        documents.clear()
    for leaf, row in documents.items():
        row["status"] = "available"
        aliases[leaf] = files[row["source_artifacts"][0]]
    return aliases, rows


def _media_aliases(inputs, deadline):
    files = inputs.source_files
    parent = PurePosixPath(inputs.scene_artifact).parent
    prefixes = ("",) if parent == PurePosixPath(".") else ("", parent.as_posix() + "/")
    candidates, aliases = [], {}
    for prefix in prefixes:
        name = prefix + "source.json"
        if name not in files:
            continue
        document = _json(files[name], MAX_CONFIG_BYTES)
        artifact = document.get("artifact")
        legacy = "artifact" not in document
        if legacy:
            _require(set(document) == {"sha256", "original_name", "rights", "scope"}
                     and document["scope"] == "User-provided local input",
                     "source.json without artifact is not the known planar-lab format")
            original = document["original_name"]
            _require(type(original) is str and 1 <= len(original) <= 255
                     and PurePosixPath(original).name == original and "\\" not in original,
                     "legacy original_name must be a bounded basename")
            suffix = PurePosixPath(original).suffix.lower()
            _require(suffix in _MEDIA_SUFFIXES, "unsupported original video suffix")
            artifact = "source" + suffix
        _require(type(artifact) is str and PurePosixPath(artifact).name == artifact
                 and PurePosixPath(artifact).suffix.lower() in _MEDIA_SUFFIXES,
                 "source video must have an explicit local basename and supported suffix")
        validate_package_layout([artifact])
        pin = document.get("sha256")
        media = prefix + artifact
        _require(type(pin) is str and _SHA.fullmatch(pin) and media in files
                 and sha256(files[media]).hexdigest() == pin, "declared source media is absent or differs from its exact hash")
        published_artifact = "source" + PurePosixPath(artifact).suffix.lower()
        changed = legacy or artifact != published_artifact
        normalized = _bytes({**document, "artifact": published_artifact}) if changed else files[name]
        candidates.append({"metadata_path": name, "media_path": media, "artifact": published_artifact,
            "sha256": pin, "metadata_bytes": normalized,
            "metadata_derivative": {"field": "artifact", "original": document.get("artifact"),
                "published": published_artifact,
                "rule": "Known planar-lab source+suffix" if legacy else "Explicit bound media normalized for standard lab views"} if changed else None})
    if candidates:
        signatures = {(item["artifact"], item["sha256"], item["metadata_bytes"]) for item in candidates}
        _require(len(signatures) == 1, "root and selected-child source media are ambiguous")
        item = candidates[0]
        _require(item["artifact"] not in {"scene.json", "candidate.json", "report.json", "manifest.json"},
                 "source media collides with package metadata")
        aliases = {"source.json": item["metadata_bytes"], item["artifact"]: files[item["media_path"]]}
        source = {"status": "available", "artifact": item["artifact"], "sha256": item["sha256"],
            "source_metadata_artifacts": [row["metadata_path"] for row in candidates],
            "source_media_artifacts": [row["media_path"] for row in candidates],
            "metadata_derivative": item["metadata_derivative"],
            "original_metadata_sha256": sha256(files[item["metadata_path"]]).hexdigest(),
            "published_metadata_sha256": sha256(item["metadata_bytes"]).hexdigest(),
            "scope": "Exact saved media identity; no decoding or source-content validation"}
    else:
        source = {"status": "unavailable", "reason": "No declared standard source.json binds one video; no filename search or parent traversal"}
    previews = []
    for leaf in _PREVIEWS:
        paths = [prefix + leaf for prefix in prefixes if prefix + leaf in files]
        hashes = {sha256(files[path]).hexdigest() for path in paths}
        if len(hashes) == 1:
            aliases[leaf] = files[paths[0]]
            previews.append({"artifact": leaf, "status": "available", "source_artifacts": paths, "sha256": next(iter(hashes))})
        else:
            previews.append({"artifact": leaf, "status": "unavailable", "reason": "Missing or ambiguous saved preview"})
    source_aliases, documents = _source_documents(files, prefixes, source.get("sha256"), deadline)
    aliases.update(source_aliases)
    return aliases, {"source": source, "previews": previews, "documents": documents}


def _recheck(inputs, config_path, runtime, deadline):
    read_bytes = inputs.receipt()["input_bytes_read_including_rechecks"]
    source = {"manifest.json": inputs.source_manifest_bytes, **inputs.source_files}
    for name, raw in source.items():
        _check(deadline); _path(inputs.source_directory)
        read_package_file(inputs.source_directory, name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
        read_bytes += len(raw)
        _require(read_bytes <= MAX_INPUT_BYTES, "source readback exceeds its aggregate byte allowance")
    _check(deadline); _path(config_path)
    read_package_file(config_path.parent, config_path.name, max_bytes=len(inputs.config_bytes),
                      expected_sha256=sha256(inputs.config_bytes).hexdigest())
    read_bytes += len(inputs.config_bytes)
    _require(read_bytes <= MAX_INPUT_BYTES, "configuration readback exceeds the aggregate byte allowance")
    parent = PurePosixPath(inputs.scene_artifact).parent
    prefixes = ("",) if parent == PurePosixPath(".") else ("", parent.as_posix() + "/")
    for prefix in prefixes:
        for leaf in ("failure.json", "report.json", "candidate.json", "manifest.json"):
            name, path = prefix + leaf, inputs.source_directory / (prefix + leaf)
            _require(not (path.exists() or path.is_symlink()) or name in source and leaf != "failure.json",
                     "source acquired conflicting completion metadata")
    for name, raw in runtime.items():
        _check(deadline)
        read_package_file(RUNTIME_ROOT, name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
    return {"source_and_config_bytes": read_bytes, "runtime_bytes": 2 * sum(map(len, runtime.values())),
            "scope": "Selected source/config/runtime reads, not global filesystem I/O telemetry"}


def _inventory(out, deadline, expected=None):
    hashes, total, visits, pending = {}, 0, 0, [out]
    while pending:
        _check(deadline)
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                _check(deadline)
                visits += 1
                _require(visits <= MAX_OUTPUT_FILES * 64 and not entry.is_symlink(), "output directory/symlink bound exceeded")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISDIR(info.st_mode):
                    pending.append(Path(entry.path))
                    continue
                name = Path(entry.path).relative_to(out).as_posix()
                _require(name != "manifest.json" and stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE_BYTES
                         and len(hashes) < MAX_OUTPUT_FILES - 1, "output file bound exceeded")
                _require(expected is None or name in expected, "unexpected output member before sealing")
                total += info.st_size
                _require(total <= MAX_OUTPUT_BYTES, "output byte bound exceeded")
                raw = read_package_file(out, name, max_bytes=max(1, info.st_size),
                                        expected_sha256=None if expected is None else expected[name])
                hashes[name] = sha256(raw).hexdigest()
    _require(expected is None or set(hashes) == set(expected), "expected output member is missing")
    return hashes, total


def _seal(out, status, deadline, expected=None):
    hashes, total = _inventory(out, deadline, expected)
    raw = _bytes({"schema_version": 1, "profile": PROFILE, "status": status,
                  "physical_validation": "unverified", "simulation_ready": False, "files": hashes})
    _require(total + len(raw) <= MAX_OUTPUT_BYTES, "manifest exceeds reserved output bytes")
    write_package_files(out, {"manifest.json": raw})
    return {"manifest_sha256": sha256(raw).hexdigest(), "output_files": len(hashes) + 1, "output_bytes": total + len(raw)}


def prepare_spatial_scene(planar_run, config_path, out, *, scene_artifact="scene.json",
                          expected_manifest_sha256=None, max_seconds=20):
    """Create a new sealed construction package; never launch, fit, train or simulate."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 20,
             "cooperative time bound must be positive and at most twenty seconds")
    source, config_path, out = map(_path, (planar_run, config_path, out))
    _require(out != source and not out.is_relative_to(source) and not source.is_relative_to(out)
             and out != config_path and not config_path.is_relative_to(out), "output must be outside immutable inputs")
    if out.exists():
        raise FileExistsError("Spatial construction requires a new output directory")
    started = time.monotonic()
    deadline = started + max_seconds
    work_deadline = deadline - min(2., max_seconds * .1)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "profile": "spatial-rigid-v1", "construction_profile": PROFILE,
        "status": "failed", "stage": "preparing", "created_at": local_time(), "simulation_ready": False,
        "physical_validation": "unverified", "physics_ready": False, "learned_policy_available": False,
        "promoted": False, "scope": _SCOPE, "scene_artifact": scene_artifact, "next_step": "lab",
        "source_directory": str(source), "config_path": str(config_path),
        "expected_source_manifest_sha256": expected_manifest_sha256,
        "limits": {"maximum_seconds": max_seconds, "maximum_input_bytes_including_rechecks": MAX_INPUT_BYTES,
            "maximum_output_bytes": MAX_OUTPUT_BYTES, "maximum_output_files": MAX_OUTPUT_FILES,
            "maximum_runtime_snapshot_bytes": MAX_RUNTIME_BYTES, "hard_rss_cap": False}}
    try:
        _check(work_deadline)
        config_raw = read_package_file(config_path.parent, config_path.name, max_bytes=MAX_CONFIG_BYTES)
        report["config_file_sha256"] = sha256(config_raw).hexdigest()
        runtime, versions = _runtime(work_deadline)
        report["stage"] = "validating_source_and_constructing"
        _check(work_deadline)
        inputs = prepare_spatial_scene_inputs(source, config_raw, scene_artifact=scene_artifact,
            expected_manifest_sha256=expected_manifest_sha256, max_seconds=min(10., work_deadline - time.monotonic()),
            max_input_bytes=(2 * MAX_INPUT_BYTES - len(config_raw)) // 3)
        scene = _json(inputs.scene_bytes, MAX_METADATA_BYTES)
        construction = _json(inputs.report_bytes, MAX_METADATA_BYTES)
        candidate = {"schema_version": 1, "profile": PROFILE, "status": "constructed_hypothesis", "scene": scene,
            "source_manifest_sha256": inputs.receipt()["source_manifest_sha256"], "source_scene_artifact": scene_artifact,
            "source_scene_sha256": construction["source_scene_sha256"], "config_sha256": construction["config_sha256"],
            "physical_validation": "unverified", "physics_ready": False, "simulation_ready": False, "scope": _SCOPE}
        report.update(stage="selecting_saved_presentation", source_manifest_sha256=inputs.receipt()["source_manifest_sha256"],
                      scene_sha256=canonical_hash(scene))
        aliases, presentation = _media_aliases(inputs, work_deadline)
        files = {"inputs/planar/manifest.json": inputs.source_manifest_bytes,
            **{"inputs/planar/" + name: raw for name, raw in inputs.source_files.items()},
            "inputs/original-scene.json": inputs.original_scene_bytes, "inputs/authored-config.json": inputs.config_bytes,
            "inputs/input-receipt.json": _bytes(inputs.receipt()), "construction-report.json": inputs.report_bytes,
            "scene.json": inputs.scene_bytes, "candidate.json": _bytes(candidate),
            **{"runtime/weft/" + name: raw for name, raw in runtime.items()}, **aliases}
        validate_package_layout([*files, "report.json", "manifest.json", "failure.json"])
        _require(all(len(PurePosixPath(name).parts) <= 64 for name in files)
                 and len(files) + 3 <= MAX_OUTPUT_FILES
                 and sum(map(len, files.values())) + 2 * MAX_METADATA_BYTES <= MAX_OUTPUT_BYTES,
                 "complete output cannot fit its reserved file/byte bounds")
        report.update(stage="copying", scene_sha256=canonical_hash(scene), candidate_sha256=canonical_hash(candidate),
            source_manifest_sha256=inputs.receipt()["source_manifest_sha256"],
            original_scene_file_sha256=sha256(inputs.original_scene_bytes).hexdigest(),
            config_file_sha256=sha256(config_raw).hexdigest(), presentation=presentation,
            runtime_sha256={name: sha256(raw).hexdigest() for name, raw in runtime.items()}, runtime_versions=versions,
            counts=construction["counts"], source_archive_members=len(inputs.source_files),
            source_archive_bytes=len(inputs.source_manifest_bytes) + sum(map(len, inputs.source_files.values())))
        for name, raw in files.items():
            _check(work_deadline)
            write_package_files(out, {name: raw})
        report["stage"] = "input_readback"
        report["readback"] = _recheck(inputs, config_path, runtime, work_deadline)
        _check(work_deadline)
        report.update(status="constructed_hypothesis", stage="constructed", seconds=time.monotonic() - started)
        report_raw = _bytes(report)
        write_package_files(out, {"report.json": report_raw})
        expected = {name: sha256(raw).hexdigest() for name, raw in {**files, "report.json": report_raw}.items()}
        sealed = _seal(out, report["status"], deadline, expected)
        return {**report, **sealed, "directory": str(out), "report": str(out / "report.json")}
    except Exception as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)[:2000]},
                      seconds=time.monotonic() - started, next_step="inspect_saved_failure")
        # A late failure cannot overwrite a saved report or turn an unfinished package into an admitted lab.
        try:
            if not (out / "report.json").exists():
                write_package_files(out, {"report.json": _bytes(report)})
                sealed = _seal(out, "failed", deadline)
            else:
                write_package_files(out, {"failure.json": _bytes(report)})
                sealed = {}
        except Exception as final_error:
            sealed = {}
            report["finalization_error"] = {"type": type(final_error).__name__, "message": str(final_error)[:500]}
            try:
                write_package_files(out, {"failure.json": _bytes({"status": "failed", "error": report["error"],
                    "finalization_error": report["finalization_error"]})})
            except Exception:
                pass
        return {**report, **sealed, "directory": str(out), "report": str(out / "report.json")}
