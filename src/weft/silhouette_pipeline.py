"""One bounded conditional image proposal, initial admission and independent clearance study.

Outputs preserve exact baseline custody in geometry-input/, selected masks in inputs/masks/,
complete proposals in objects/<ID>/result.json and independently admitted geometry in geometry/.
The returned candidate_directory is an initial-admission result, not a motion/task pass.
"""

from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import re
import time

from weft.io import canonical_hash, local_time
from weft.shared_exercise_task import validate_shared_exercise_task
from weft.shared_world import validate_shared_world
from weft.shared_world_clearance import clearance_protocol, compare_shared_world_clearance
from weft.shared_world_geometry import (
    CONFIG_PROFILE, IMPLEMENTATION_FILES, MAX_OUTPUT_BYTES as GEOMETRY_BYTES,
    MAX_OUTPUT_FILES as GEOMETRY_FILES, MAX_INPUT_FILES as GEOMETRY_INPUT_FILES,
    _path, _preserve_failed_file, _terminal_input, prepare_shared_world_geometry,
    validate_geometry_hypothesis_config,
)
from weft.shared_world_packaging import (
    collect_shared_source_files, read_package_file, read_package_manifest, write_package_files,
)
from weft.silhouette_geometry import CONTRACT, _projected_vertices, prepare_silhouette_evidence, propose_silhouette_geometry
from weft.silhouette_report import MAX_HTML_BYTES, render_silhouette_report
from weft.source_equivalence import load_source_equivalence
from weft.world_evidence import _check_tree, _nonfinite, _unique_object


PROFILE = "conditional-silhouette-world-study-v1"
MAX_INPUT_BYTES = 128 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024
MAX_INPUT_FILES = 2048
MAX_OUTPUT_BYTES = 384 * 1024 * 1024
MAX_OUTPUT_FILES = 2048
MAX_OBJECTS = 4
MAX_PARTS = 8
MAX_CLEARANCE_BYTES = 64 * 1024 * 1024
METADATA_RESERVE = 2 * 1024 * 1024
FAILURE_GRACE_SECONDS = 5
_SOURCE_STATUSES = {"ready", "trained", "evaluated"}
_MASK_STATUSES = {"ready", "completed", "verified", "decoded_proposals", "partial_decode", "evidence_only"}
_RUNTIME = tuple(dict.fromkeys((*IMPLEMENTATION_FILES, "silhouette_pipeline.py", "silhouette_geometry.py", "silhouette_report.py", "cli.py",
    "shared_world_clearance.py", "shared_world_reconstruction.py", "arm_reference_alignment.py",
    "segmentation_observations.py", "segmentation_evidence.py", "segmentation_masks.py", "primitive_geometry.py")))
SCOPE = ("Conditional construction-image evidence only. Image fit, initial admission and full-reference kinematic "
         "overlap are separate results; none establishes recovered depth, grasp, task success or physical accuracy.")


def _bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    result = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
    _check_tree(result)
    if not isinstance(result, dict):
        raise ValueError("Pipeline documents must be JSON objects")
    return result


def _remaining(deadline, ceiling=30):
    remaining = deadline - time.monotonic()
    if remaining < 1:
        raise TimeoutError("Silhouette pipeline has insufficient remaining cooperative time")
    return min(ceiling, remaining)


def _manifest(root, statuses):
    raw, files = read_package_manifest(root)
    doc = _json(raw)
    if len(raw) > 2 * 1024 * 1024 or not 1 <= len(files) <= MAX_INPUT_FILES:
        raise ValueError("Pipeline input manifest exceeds its file/byte bounds")
    if "status" in doc and (not isinstance(doc["status"], str) or doc["status"] not in statuses):
        raise ValueError("Pipeline input must have a supported terminal status or historical absent status")
    return raw, files


def _read(root, names, name, cap=16 * 1024 * 1024):
    if name not in names:
        raise ValueError("Required input is absent from its outer manifest: " + name)
    return read_package_file(root, name, max_bytes=cap, expected_sha256=names[name])


def _verify_input(root, hashes, deadline, *, max_bytes, keep):
    retained, sizes, total = {}, {}, 0
    for name, expected in hashes.items():
        _remaining(deadline)
        raw = read_package_file(root, name, max_bytes=min(64 * 1024 * 1024, max(1, max_bytes - total)),
                                expected_sha256=expected)
        total += len(raw)
        if total > max_bytes:
            raise ValueError("Pipeline input aggregate byte bound exceeded")
        sizes[name] = len(raw)
        if keep is None or name in keep:
            retained[name] = raw
    return retained, sizes, total


def _inventory(out):
    names, total = [], 0
    for index, path in enumerate(out.rglob("*")):
        if index >= MAX_OUTPUT_FILES * 8:
            raise ValueError("Pipeline output directory count exceeds its bound")
        if path.is_symlink():
            raise ValueError("Pipeline output cannot contain symbolic links")
        if path.is_file():
            names.append(path.relative_to(out).as_posix())
            total += path.stat().st_size
            if len(names) > MAX_OUTPUT_FILES or total > MAX_OUTPUT_BYTES:
                raise ValueError("Pipeline output exceeds its file/byte bounds; partial files retained")
    return names, total


def _reserve(out, added_bytes, added_files):
    names, total = _inventory(out)
    if total + added_bytes > MAX_OUTPUT_BYTES or len(names) + added_files > MAX_OUTPUT_FILES:
        raise ValueError("Next stage cannot reserve its complete output budget")


def _write(out, files):
    _reserve(out, sum(map(len, files.values())), len(files))
    write_package_files(out, files)


def _seal(out, status, deadline):
    names, total = _inventory(out)
    hashes = {}
    for name in sorted(names):
        _remaining(deadline)
        hashes[name] = _sha(read_package_file(out, name, max_bytes=min(64 * 1024 * 1024, MAX_OUTPUT_BYTES)))
    raw = _bytes({"schema_version": 1, "status": status, "files": hashes,
                  "scope": "Complete immutable pipeline outputs; no dynamic task success claimed."})
    if len(raw) > METADATA_RESERVE:
        raise ValueError("Pipeline manifest exceeds its metadata reservation")
    _write(out, {"manifest.json": raw})
    return {"manifest_sha256": _sha(raw), "files": len(hashes), "bytes_before_manifest": total}


def _failure(out, report, error):
    deadline = time.monotonic() + FAILURE_GRACE_SECONDS
    report.update(status="failed", error={"type": type(error).__name__, "message": str(error)[:2000]})
    preserved = {}
    for name in ("report.json", "manifest.json"):
        _remaining(deadline)
        if (out / name).exists():
            raw = read_package_file(out, name, max_bytes=METADATA_RESERVE)
            target = name.removesuffix(".json") + ".publication-failed.json"
            _preserve_failed_file(out, name, target)
            preserved[target] = _sha(raw)
    report["preserved_publication_files"] = preserved
    raw = _bytes(report)
    if len(raw) > METADATA_RESERVE:
        raw = _bytes({"profile": PROFILE, "status": "failed", "error": report["error"],
                      "scope": "Failure receipt only; complete earlier stage outputs remain untouched."})
    # Prior completed-looking receipts remain inspectable, never active after publication failure.
    write_package_files(out, {"failure.json": raw})
    manifest = _bytes({"schema_version": 1, "status": "failed", "custody": "failure_receipt_only",
                       "unsealed_outputs_preserved": True, "files": {**preserved, "failure.json": _sha(raw)}})
    _remaining(deadline)
    write_package_files(out, {"manifest.json": manifest})
    return {"status": "failed", "outcome": "preserved_failure", "directory": str(out),
            "manifest_sha256": _sha(manifest), "manifest_file": "manifest.json", "error": report["error"],
            "report_path": str(out / "failure.json"), "candidate_directory": None,
            "candidate_manifest_sha256": None, "inspection_path": None}


def prepare_silhouette_world(run, masks_run, out, *, task_path=None, object_ids, source_frame,
                             conditional_exterior=False, max_seconds=60):
    """Prepare one image-only candidate, then report admission and unchanged-reference overlaps separately."""
    if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 60
            or type(conditional_exterior) is not bool or not conditional_exterior
            or type(source_frame) is not int or source_frame < 0
            or not isinstance(object_ids, (tuple, list)) or not 1 <= len(object_ids) <= MAX_OBJECTS
            or any(not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name) for name in object_ids)
            or len(set(object_ids)) != len(object_ids)):
        raise ValueError("Require explicit conditional-exterior, unique rigid IDs, one source frame and 1–60 seconds")
    source, masks_root, out = map(_path, (run, masks_run, out))
    task_path = _path(task_path) if task_path is not None else None
    inputs = [source, masks_root] + ([task_path.parent] if task_path is not None else [])
    if out.exists() or any(out.is_relative_to(path) or path.is_relative_to(out) for path in inputs):
        raise ValueError("Pipeline output must be new and outside every immutable input")
    out.mkdir(parents=True)
    started, deadline = time.monotonic(), time.monotonic() + max_seconds
    report = {"profile": PROFILE, "schema_version": 1, "status": "preparing", "scope": SCOPE,
              "source_fit": "conditional_only", "physical_validation": "unverified", "objects": [],
              "api_calls": 0, "inference_calls": 0, "video_decodes": 0, "provider_cost_usd": 0,
              "workers": 1, "retries": 0, "task_rollouts": 0, "stage": "inputs"}
    try:
        source_manifest, source_files = _manifest(source, _SOURCE_STATUSES)
        mask_manifest, mask_files = _manifest(masks_root, _MASK_STATUSES)
        if len(source_files) + len(mask_files) + 2 + (2 if task_path is not None else 0) > MAX_INPUT_FILES:
            raise ValueError("Combined input packages exceed the file-count bound")
        if len(source_files) + int(task_path is not None and "exercise-task.json" not in source_files) > GEOMETRY_INPUT_FILES:
            raise ValueError("Base package cannot fit the subsequent geometry admission file bound")
        for name in source_files:
            mapping_log = re.fullmatch(r"(?:(?:source-run|learning-run|policy-evaluation-run)/)*source-mappings/[0-9a-f]{64}/[^/]+\.log", name)
            if (any(part.startswith(".") or part == "__pycache__" for part in Path(name).parts)
                    or name.endswith(".pyc") or name.endswith(".log") and not mapping_log):
                raise ValueError("Base package cannot contain hidden files, caches or unrelated logs")
        source_data, source_sizes, source_size = _verify_input(source, source_files, deadline,
            max_bytes=MAX_SOURCE_BYTES - len(source_manifest), keep=None)
        for name in ("status.json", "report.json", "reconstruction.json", "evaluation.json", "training.json"):
            if name in source_data:
                _terminal_input(source_data[name], name)
        scene = validate_shared_world(_json(source_data["scene.json"]))
        proof = collect_shared_source_files(scene, source, expected_manifest_sha256=_sha(source_manifest))
        _remaining(deadline)
        segmentation_raw = _read(masks_root, mask_files, "masks/original-evidence.json")
        mask_raw = _read(masks_root, mask_files, "masks/mask-evidence.json")
        seg, masks = _json(segmentation_raw), _json(mask_raw)
        if any(not isinstance(doc.get("observations"), list)
               or any(not isinstance(row, dict) for row in doc["observations"]) for doc in (seg, masks)):
            raise ValueError("Segmentation and mask observations must be lists of records")
        selected = [row for row in masks["observations"] if row.get("frame") == source_frame]
        raster_names = [row["raster_path"] for row in selected if row.get("mask_status") == "decoded"]
        wanted = {"masks/original-evidence.json", "masks/mask-evidence.json", *("masks/" + name for name in raster_names)}
        if "evidence/segmentation-evidence.json" in mask_files:
            wanted.add("evidence/segmentation-evidence.json")
        mask_data, mask_sizes, mask_size = _verify_input(masks_root, mask_files, deadline,
            max_bytes=MAX_INPUT_BYTES - source_size - len(source_manifest) - len(mask_manifest), keep=wanted)
        if not wanted <= set(mask_data):
            raise ValueError("Every selected decoded mask must be sealed by the outer mask manifest")
        if (mask_data["masks/original-evidence.json"] != segmentation_raw or mask_data["masks/mask-evidence.json"] != mask_raw
                or "evidence/segmentation-evidence.json" in mask_data
                and canonical_hash(_json(mask_data["evidence/segmentation-evidence.json"])) != canonical_hash(seg)):
            raise ValueError("Mask inputs changed or duplicate segmentation evidence conflicts")
        task_raw, task_input_manifest, explicit_task_raw = source_data.get("exercise-task.json"), None, None
        if task_path is not None:
            task_input_manifest, task_files = _manifest(task_path.parent, _SOURCE_STATUSES)
            explicit_task_raw = _read(task_path.parent, task_files, task_path.name, 64 * 1024)
            if task_raw is not None and _json(task_raw) != _json(explicit_task_raw):
                raise ValueError("This pipeline cannot replace an existing sealed task definition")
            if task_raw is None:
                task_raw = explicit_task_raw
        verified_bytes = source_size + mask_size + len(source_manifest) + len(mask_manifest)
        if explicit_task_raw is not None:
            verified_bytes += len(explicit_task_raw) + len(task_input_manifest)
        if verified_bytes > MAX_INPUT_BYTES:
            raise ValueError("Pipeline input aggregate byte bound exceeded including manifests and explicit task")
        task = validate_shared_exercise_task(_json(task_raw), scene) if task_raw is not None else None
        document_bytes = {sha: proof.files[name] for sha, name in scene["source_binding"]["evidence_documents"].items()}
        equivalences = []
        for item in scene["source_binding"].get("source_equivalences", []):
            _remaining(deadline)
            equivalences.append(load_source_equivalence(source / item["bundle_path"], deadline=deadline,
                max_input_bytes=MAX_SOURCE_BYTES))
        package = Path(__file__).parent
        runtime = {name: read_package_file(package, name, max_bytes=2 * 1024 * 1024) for name in _RUNTIME}
        runtime.update({"project/" + name: read_package_file(package.parent.parent, name, max_bytes=1024 * 1024)
                        for name in ("pyproject.toml", "requirements.lock", "requirements-lab.lock",
                                     "scripts/studies/check_silhouette_shared_world.py")})
        if sum(map(len, runtime.values())) > 16 * 1024 * 1024:
            raise ValueError("Pipeline implementation snapshot exceeds 16 MiB")
        selections = []
        bindings = {row["world_entity_id"]: row for row in scene["source_binding"]["entity_bindings"]}
        for object_id in object_ids:
            if object_id == "human" or object_id not in {obj["id"] for obj in scene["objects"]}:
                raise ValueError("Only existing rigid objects may be proposed")
            binding = bindings[object_id]
            target_id = binding["observation_entity_id"]
            occluders = sorted(row["object_id"] for row in seg["observations"]
                if row["frame"] == source_frame and row["proposal_status"] == "returned" and row["object_id"] != target_id)
            selections.append({"world_object_id": object_id, "document_sha256": binding["observation_document_sha256"],
                               "observation_entity_id": target_id, "source_frame": source_frame, "occluder_ids": occluders})
        settings = {"profile": PROFILE, "created_at": local_time(), "run": str(source), "masks_run": str(masks_root),
                    "task_path": str(task_path) if task_path else None, "source_manifest_sha256": _sha(source_manifest),
                    "mask_manifest_sha256": _sha(mask_manifest), "scene_sha256": canonical_hash(scene),
                    "task_file_sha256": _sha(task_raw) if task_raw is not None else None,
                    "explicit_task_file_sha256": _sha(explicit_task_raw) if explicit_task_raw is not None else None,
                    "selections": selections, "conditional_exterior": True, "max_parts": MAX_PARTS,
                    "max_seconds": max_seconds, "workers": 1, "api_budget_usd": 0,
                    "deadline_scope": "Cooperative stage/readback/publication checks; bounded native and source-verification calls are not preemptible.",
                    "failure_receipt_grace_seconds": FAILURE_GRACE_SECONDS,
                    "selection_rule": "One proposal per requested object; combine every qualified replacement once. Any proposal error prevents admission. No contact/reward-aware fitting or subset retries.",
                    "bounds": {"input_bytes": MAX_INPUT_BYTES, "input_files": MAX_INPUT_FILES,
                               "output_bytes": MAX_OUTPUT_BYTES, "output_files": MAX_OUTPUT_FILES},
                    "silhouette_contract_sha256": canonical_hash(CONTRACT),
                    "clearance_protocol_sha256": canonical_hash(clearance_protocol()),
                    "runtime_sha256": {name: _sha(raw) for name, raw in runtime.items()},
                    "versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "mujoco", "Pillow")}}}
        report["settings_sha256"] = canonical_hash(settings)
        preserved = {"inputs/base-manifest.json": source_manifest, "inputs/masks-manifest.json": mask_manifest,
                     "settings.json": _bytes(settings), "clearance-protocol.json": _bytes(clearance_protocol()),
                     "silhouette-contract.json": _bytes(CONTRACT), "source-verification.json": _bytes(proof.report),
                     **{"implementation/" + name: raw for name, raw in runtime.items()},
                     **{"inputs/masks/" + name: raw for name, raw in mask_data.items()}}
        if task_input_manifest is not None:
            preserved["inputs/task-manifest.json"] = task_input_manifest
            preserved["inputs/explicit-exercise-task.json"] = explicit_task_raw
        geometry_inputs = dict(source_data)
        if task_raw is not None:
            geometry_inputs["exercise-task.json"] = task_raw
        geometry_inputs["manifest.json"] = _bytes({"schema_version": 1, "status": "ready",
            "source_manifest_sha256": _sha(source_manifest), "scope": "Exact base files plus the explicitly frozen task.",
            "files": {name: _sha(raw) for name, raw in geometry_inputs.items()}})
        preserved.update({"geometry-input/" + name: raw for name, raw in geometry_inputs.items()})
        _write(out, preserved)
        _reserve(out, GEOMETRY_BYTES + MAX_CLEARANCE_BYTES + MAX_OBJECTS * 4 * 1024 * 1024 + MAX_HTML_BYTES + METADATA_RESERVE,
                 GEOMETRY_FILES + MAX_OBJECTS * 3 + 6)
        report["stage"] = "prepared_context"
        _remaining(deadline)
        context = prepare_silhouette_evidence(scene, source_documents=document_bytes,
            segmentation_evidence=segmentation_raw, mask_document=mask_raw,
            raster_documents={name: mask_data["masks/" + name] for name in raster_names}, source_equivalences=equivalences)
        _remaining(deadline)
        replacements, render_items, proposal_errors = [], [], False
        report["stage"] = "conditional_proposals"
        for selection in selections:
            name = selection["world_object_id"]
            try:
                result = propose_silhouette_geometry(context, name, source_frame=source_frame,
                    occluder_ids=selection["occluder_ids"], conditional_exterior=True, max_parts=MAX_PARTS,
                    max_seconds=_remaining(deadline, 15))
                _write(out, {f"objects/{name}/result.json": _bytes(result)})
                row = {"object_id": name, "status": result["status"], "report": f"objects/{name}/result.json",
                       "before_conditional_mask_fit": result["report"]["baseline"]["scores"],
                       "after_conditional_mask_fit": result["report"]["candidate"]["scores"]}
                if result["status"] == "conditional_proposal":
                    config = validate_geometry_hypothesis_config(result["geometry_config"])
                    if (config["base_scene_sha256"] != canonical_hash(scene) or len(config["replacements"]) != 1
                            or config["replacements"][0]["object_id"] != name):
                        raise ValueError("Proposal config is not bound to this exact single-object baseline")
                    replacements.extend(deepcopy(config["replacements"]))
                elif result["status"] != "no_qualified_improvement" or result["geometry_config"] is not None:
                    raise ValueError("Proposal status/config conflict")
                mask_row = next(row for row in selected if row["object_id"] == selection["observation_entity_id"])
                obj = next(obj for obj in scene["objects"] if obj["id"] == name)
                bridge = scene["coordinate_frame"]["scale_bridge"]
                scale = bridge["body_height_to_world_scale"]
                center = [(p - origin) / scale for p, origin in zip(obj["position"], bridge["origin_world"])]
                half = [size / (2 * scale) for size in obj["size"]]
                vertices = _projected_vertices([p - h for p, h in zip(center, half)],
                    [p + h for p, h in zip(center, half)], result["report"]["camera_hypothesis"])
                png_bytes = mask_data["masks/" + mask_row["raster_path"]]
                inspection = {"object_id": name, "selected_mask_row": mask_row,
                    "baseline_vertices_px": vertices, "proposal_path": f"objects/{name}/result.json",
                    "selected_png_path": "inputs/masks/masks/" + mask_row["raster_path"], "selected_png_sha256": _sha(png_bytes)}
                _write(out, {f"objects/{name}/inspection-input.json": _bytes(inspection)})
                render_items.append({"object_id": name, "result": result, "selected_mask_row": mask_row,
                                     "selected_png_bytes": png_bytes, "baseline_vertices_px": vertices})
                report["objects"].append(row)
            except (ValueError, RuntimeError, OSError, TimeoutError) as error:
                proposal_errors = True
                row = {"object_id": name, "status": "failed", "error": {"type": type(error).__name__, "message": str(error)[:2000]}}
                report["objects"].append(row)
                _write(out, {f"objects/{name}/failure.json": _bytes(row)})
        report["outcome"] = "proposal_failures" if proposal_errors else "no_qualified_improvement"
        report["status"] = "partial" if proposal_errors else "completed"
        if replacements and not proposal_errors:
            config = {"schema_version": 1, "profile": CONFIG_PROFILE, "base_scene_sha256": canonical_hash(scene),
                      "assumptions": ["Conditional exterior is an explicit image prior, not measured empty space.",
                          "All image-qualified replacements are combined once; no contact or reward feedback selects the fit."],
                      "replacements": replacements}
            _write(out, {"geometry-config.json": _bytes(config)})
            report["stage"] = "initial_admission"
            report["geometry_admission"] = prepare_shared_world_geometry(out / "geometry-input", out / "geometry-config.json",
                out / "geometry", max_seconds=_remaining(deadline))
            if report["geometry_admission"]["status"] == "ready":
                candidate = validate_shared_world(_json(read_package_file(out / "geometry", "scene.json", max_bytes=16 * 1024 * 1024)))
                candidate_task = _json(read_package_file(out / "geometry", "exercise-task.json", max_bytes=64 * 1024)) if task else None
                report["stage"] = "clearance_comparison"
                clearance = compare_shared_world_clearance(scene, candidate, source_run=out / "geometry-input",
                    exercise_tasks=(task, candidate_task) if task else None, max_seconds=_remaining(deadline))
                _write(out, {"clearance-comparison.json": _bytes(clearance)})
                report.update(outcome="clearance_comparison_" + clearance["status"],
                              status="completed" if clearance["status"] == "complete" else "partial",
                              clearance={"status": clearance["status"], "summary": clearance["summary"],
                                         "comparison_summary": clearance.get("comparison_summary", []),
                                         "scope": "Overlap changes are not image fitting feedback or task success."})
            else:
                report.update(outcome="initial_admission_" + report["geometry_admission"]["status"], status="partial")
        report["stage"] = "static_inspection"
        report["inspection"] = {"status": "unavailable", "reason": "No complete object proposal was returned."}
        if render_items:
            _remaining(deadline)
            html = render_silhouette_report(render_items, clearance_summary=report.get("clearance"))
            _remaining(deadline)
            _write(out, {"shape-comparison.html": html})
            report["inspection"] = {"status": "ready", "path": "shape-comparison.html", "bytes": len(html),
                                    "objects": [item["object_id"] for item in render_items], "scope": SCOPE}
        report["stage"] = "input_readback"
        for root, manifest, hashes, sizes in ((source, source_manifest, source_files, source_sizes),
                                             (masks_root, mask_manifest, mask_files, mask_sizes)):
            _readback_manifest = read_package_file(root, "manifest.json", max_bytes=2 * 1024 * 1024, expected_sha256=_sha(manifest))
            for name, expected in hashes.items():
                _remaining(deadline)
                read_package_file(root, name, max_bytes=max(1, sizes[name]), expected_sha256=expected)
        if task_path is not None:
            read_package_file(task_path.parent, task_path.name, max_bytes=64 * 1024, expected_sha256=_sha(explicit_task_raw))
            read_package_file(task_path.parent, "manifest.json", max_bytes=2 * 1024 * 1024, expected_sha256=_sha(task_input_manifest))
        for name, raw in runtime.items():
            _remaining(deadline)
            root = package.parent.parent if name.startswith("project/") else package
            relative = name.removeprefix("project/") if name.startswith("project/") else name
            read_package_file(root, relative, max_bytes=max(1, len(raw)), expected_sha256=_sha(raw))
        candidate_ready = report.get("geometry_admission", {}).get("status") == "ready"
        candidate_manifest = read_package_file(out / "geometry", "manifest.json", max_bytes=2 * 1024 * 1024) if candidate_ready else None
        report.update(stage="finished", original_inputs_unchanged=True, seconds_before_seal=time.monotonic() - started,
                      input_bytes_verified=verified_bytes,
                      input_files_verified=len(source_files) + len(mask_files) + 2 + (2 if task_path is not None else 0),
                      candidate_directory="geometry" if candidate_ready else None,
                      candidate_manifest_sha256=_sha(candidate_manifest) if candidate_manifest is not None else None)
        _write(out, {"report.json": _bytes(report)})
        integrity = _seal(out, report["status"], deadline)
        return {"status": report["status"], "outcome": report["outcome"], "directory": str(out),
                "report_path": str(out / "report.json"),
                "candidate_directory": str(out / "geometry") if candidate_ready else None,
                "candidate_manifest_sha256": report["candidate_manifest_sha256"],
                "inspection_path": str(out / "shape-comparison.html") if render_items else None,
                "seconds": time.monotonic() - started, **integrity}
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, TimeoutError, OverflowError, RecursionError) as error:
        report["seconds_before_failure"] = time.monotonic() - started
        return _failure(out, report, error)
