"""Source-bound camera/root proposals through the existing owned body worker path."""

from hashlib import sha256
from importlib.metadata import version
import math
from pathlib import Path
import sys
import time

from weft.io import local_time
from weft.shared_world_packaging import read_package_file, write_package_files
from weft.whole_body_reference_pipeline import (
    MAX_BYTES, MAX_FILES, MAX_MEDIA_BYTES, MAX_RECOVERY_BYTES,
    _bytes, _check, _inventory, _json, _path, _run_worker, _seal,
)


PROFILE = "body-camera-root-workflow-v1"
STUDY_PROFILE = "body-camera-root-study-v1"
OUTPUT_NAMES = frozenset({"geometry.json", "camera-root-input.json", "target-provenance.json",
                          "source-slots.json", "proposal.json", "geometry-check.json"})
WORKFLOW_MODULES = ("cli", "body_camera_root_pipeline", "body_camera_root_report",
                    "whole_body_reference_pipeline", "whole_body_reference_inputs", "shared_world_packaging")


def _runtime():
    from weft.body_camera_root_worker import RUNTIME_MODULES

    source = Path(__file__).parent
    files = {f"runtime/weft/{name}.py": read_package_file(source, name + ".py", max_bytes=2 << 20)
             for name in sorted(RUNTIME_MODULES)}
    files.update({f"workflow/{name}.py": read_package_file(source, name + ".py", max_bytes=2 << 20)
                  for name in WORKFLOW_MODULES})
    versions = {"python": sys.version.split()[0],
                **{name: version(name) for name in ("numpy", "mujoco", "gymnasium")}}
    return files, versions


def _authored_camera(snapshot, focal_px):
    if type(focal_px) not in (int, float) or not math.isfinite(focal_px) or not 16 <= focal_px <= 65536:
        raise ValueError("Explicit authored focal length must be 16–65536 pixels")
    import numpy as np
    from weft.body_frame import CAMERA_TO_SIM
    from weft.whole_body_reference_refinement import _prepare

    files = snapshot["files"]
    human, evidence = _json(files["inputs/human.json"]), _json(files["inputs/evidence-derived.json"])
    _, _, roots, _, root = _prepare(human, evidence)
    first = root["reference"][0]
    if first.get("status") != "observed_image_estimate" or first.get("pelvis_image_px") is None:
        raise ValueError("Authored camera initialization requires the first observed pelvis; provide an explicit camera instead")
    width, height = evidence["width"], evidence["height"]
    rotation = CAMERA_TO_SIM.T
    intrinsics = np.array([[focal_px, 0., width / 2], [0., focal_px, height / 2], [0., 0., 1.]])
    distance = focal_px * root["scale"]["model_units_per_pixel"]
    # This selects an explicit camera gauge, not measured depth or camera calibration.
    pelvis = roots[0] + np.array([0., 0., human["mobility"]["base_height"]])
    ray = np.linalg.solve(intrinsics, [*first["pelvis_image_px"], 1.])
    center = pelvis - rotation.T @ (distance * ray)
    return _bytes({"schema_version": 1, "profile": "fixed-body-camera-hypothesis-v1",
        "bindings": {"source_sha256": human["source_sha256"], "body_sha256": sha256(files["inputs/human.json"]).hexdigest(),
                     "evidence_sha256": sha256(files["inputs/evidence-derived.json"]).hexdigest()},
        "image": {key: evidence[key] for key in ("width", "height", "fps", "frame_count")},
        "coordinate_frame": "original_body_model_units", "scale_model_units_per_unit": 1.,
        "partition": "exposed_development", "selection": "explicit_before_proposal_no_search",
        "camera": {"rotation": rotation.tolist(), "center": center.tolist(), "K": intrinsics.tolist(), "radial_k1": 0.,
                   "quality": "authored", "provenance":
                       f"Authored pinhole gauge v1: caller focal={focal_px:g}px, centered principal point, zero distortion, "
                       "fixed CAMERA_TO_SIM inverse, distance=focal*original model-units-per-pixel; "
                       "center anchors the original first model pelvis to its selected image estimate. "
                       "Frozen before residuals, no search or measured calibration; source scale remains an estimate."}})


def _worker_outputs(study, worker, plan_sha, versions):
    if (worker.get("status") != "completed" or worker.get("plan_sha256") != plan_sha
            or worker.get("runtime_versions") != versions):
        raise ValueError("Camera/root completion differs from the frozen plan/runtime")
    pins = worker.get("output_files")
    if not isinstance(pins, dict) or set(pins) != OUTPUT_NAMES:
        raise ValueError("Camera/root worker did not pin all required outputs")
    outputs, total = {}, 0
    for name in sorted(OUTPUT_NAMES):
        raw = read_package_file(study, name, max_bytes=16 << 20 if name == "proposal.json" else 8 << 20,
                                expected_sha256=pins[name])
        total += len(raw)
        if total > 16 << 20:
            raise ValueError("Combined camera/root outputs exceed 16 MiB")
        outputs[name] = raw
    proposal = _json(outputs["proposal.json"])
    report = proposal.get("report", {})
    if (proposal.get("status") not in {"proposed", "unchanged", "partial", "unavailable"}
            or report.get("admitted") is not False or report.get("promoted") is not False
            or report.get("physical_validation") != "unverified"
            or report.get("input_sha256") != sha256(outputs["camera-root-input.json"]).hexdigest()):
        raise ValueError("Camera/root proposal cannot admit a body or alter its input identity")
    return outputs, proposal


def prepare_camera_root_proposal(source, camera, out, *, expected_manifest_sha256=None,
                                 fit_seconds=10., max_seconds=60., camera_focal_px=None, progress=None):
    """Save one explicit camera hypothesis, diagnostic proposal and report; never install it."""
    from weft.body_camera_root_report import render_camera_root_report
    from weft.whole_body_reference_inputs import prepare_reference_inputs, recheck_reference_inputs

    if (type(fit_seconds) not in (int, float) or not math.isfinite(fit_seconds) or not .001 <= fit_seconds <= 10
            or type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 20 <= max_seconds <= 60):
        raise ValueError("Camera fitting time must be 0.001–10 seconds; workflow time 20–60 seconds")
    if (camera is None) == (camera_focal_px is None):
        raise ValueError("Supply exactly one explicit camera file or authored focal length")
    source, out = _path(source), _path(out)
    camera = _path(camera) if camera is not None else None
    if out.exists() or out.is_relative_to(source) or source.is_relative_to(out) or (camera is not None and camera.is_relative_to(out)):
        raise ValueError("Camera/root output must be new and outside its source/camera")
    started = time.monotonic()
    deadline = started + max_seconds
    camera_raw = read_package_file(camera.parent, camera.name, max_bytes=128 << 10) if camera is not None else None
    if camera_raw is not None and _json(camera_raw).get("profile") != "fixed-body-camera-hypothesis-v1":
        raise ValueError("An explicit fixed-body-camera-hypothesis-v1 camera is required")
    if progress:
        progress("Verify saved video observations", {"status": "running"})
    snapshot = prepare_reference_inputs(source, expected_manifest_sha256=expected_manifest_sha256,
                                        deadline=min(deadline, started + 10))
    if camera_raw is None:
        camera_raw = _authored_camera(snapshot, camera_focal_px)
    runtime, versions = _runtime()
    initial = {**snapshot["files"], **runtime,
               "inputs/input-receipt.json": _bytes(snapshot["identity"]), "inputs/camera-config.json": camera_raw}
    media_name, output_limit = None, MAX_BYTES
    if snapshot["identity"].get("recovery") is not None:
        media = snapshot["identity"].get("source_artifact")
        if media not in {"source" + suffix for suffix in (".mp4", ".webm", ".mov", ".m4v", ".avi", ".mkv")}:
            raise ValueError("Recovery requires its exact original source container")
        media_name, output_limit = "inputs/" + media, MAX_RECOVERY_BYTES
        if (snapshot["identity"]["recovery"].get("media", {}).get("artifact") != media_name
                or media_name not in initial or len(initial[media_name]) > MAX_MEDIA_BYTES):
            raise ValueError("Recovery media differs from its verified source or byte bound")
    metadata = {name: raw for name, raw in initial.items() if name.startswith("inputs/") and name != media_name}
    if len(metadata) > 64 or any(len(raw) > 8 << 20 for raw in metadata.values()) or sum(map(len, metadata.values())) > 32 << 20:
        raise ValueError("Camera/root metadata exceeds its bounded snapshot")
    if len(initial) + 14 > MAX_FILES or sum(map(len, initial.values())) + (26 << 20) > output_limit:
        raise ValueError("Camera/root input leaves no reserved output capacity")
    _check(deadline)
    out.mkdir(parents=True, mode=0o700)
    study = out / "proposal"
    study.mkdir(mode=0o700)
    (study / "temporary").mkdir(mode=0o700)
    result = {"schema_version": 1, "profile": PROFILE, "status": "failed", "created_at": local_time(),
              "directory": str(out), "source": snapshot["identity"], "proposal_status": None,
              "simulation_ready": False, "admitted": False, "promoted": False,
              "physical_validation": "unverified", "external_api_usd": 0,
              "report_path": None, "worker": None, "error": None}
    stage = "snapshot"
    try:
        write_package_files(study, initial)
        plan = {"schema_version": 1, "profile": "body-camera-root-plan-v1", "status": "frozen",
                "input_files": {name: sha256(raw).hexdigest() for name, raw in metadata.items()},
                "runtime_files": {name: sha256(raw).hexdigest() for name, raw in runtime.items() if name.startswith("runtime/")},
                "orchestration_files": {name: sha256(raw).hexdigest() for name, raw in runtime.items() if name.startswith("workflow/")},
                "runtime_versions": versions, "settings": {"maximum_seconds": fit_seconds},
                "source": snapshot["identity"],
                "bounds": {"worker_wall_seconds": 30, "workflow_seconds": max_seconds,
                           "output_bytes": output_limit, "output_files": MAX_FILES,
                           "copied_input_bytes": sum(map(len, snapshot["files"].values())),
                           "source_media_bytes": len(initial[media_name]) if media_name else 0,
                           "memory_scope": "Bounded inputs, calls and outputs; observed RSS, not a hard RSS cap",
                           "execution_scope": "Trusted reviewed code in an owned restricted child, not a generated-code OS sandbox"},
                "physical_validation": "unverified", "promoted": False, "external_api_usd": 0}
        plan_raw = _bytes(plan)
        plan_sha = sha256(plan_raw).hexdigest()
        write_package_files(study, {"plan.json": plan_raw})
        stage = "geometry_and_proposal"
        if progress:
            progress("Compare the fixed body and camera hypothesis", {"status": "running", "physics_actions": 0})
        remaining = min(30., deadline - time.monotonic() - 10.)
        if remaining <= 0:
            raise TimeoutError("No time remains for a bounded camera/root proposal")
        process = _run_worker(study, plan_sha, remaining, script_name="body_camera_root_worker.py")
        result["process"] = process
        supervision_raw = _bytes(process)
        write_package_files(study, {"supervision.json": supervision_raw})
        if (study / "worker.json").exists():
            worker_raw = read_package_file(study, "worker.json", max_bytes=64 << 10)
            result["worker"] = _json(worker_raw)
        if process["error"] is not None or process["returncode"] != 0 or not isinstance(result["worker"], dict):
            raise RuntimeError("Camera/root worker failed; its logs and receipt are preserved")
        outputs, proposal = _worker_outputs(study, result["worker"], plan_sha, versions)
        stage = "inspection"
        html = render_camera_root_report(outputs["camera-root-input.json"], outputs["proposal.json"],
            outputs["geometry-check.json"], outputs["target-provenance.json"],
            artifact_links={name: name for name in sorted(OUTPUT_NAMES | {"worker.json", "manifest.json"})})
        if not isinstance(html, bytes) or len(html) > 8 << 20:
            raise ValueError("Camera/root report exceeds 8 MiB")
        write_package_files(study, {"index.html": html})
        stage = "integrity"
        recheck_reference_inputs(snapshot, deadline=deadline)
        if camera is not None:
            read_package_file(camera.parent, camera.name, max_bytes=128 << 10, expected_sha256=sha256(camera_raw).hexdigest())
        for name, raw in initial.items():
            _check(deadline)
            read_package_file(study, name, max_bytes=max(1, len(raw)), expected_sha256=sha256(raw).hexdigest())
        for name, raw in runtime.items():
            source_name = name.rsplit("/", 1)[-1]
            read_package_file(Path(__file__).parent, source_name, max_bytes=2 << 20, expected_sha256=sha256(raw).hexdigest())
        read_package_file(study, "plan.json", max_bytes=len(plan_raw), expected_sha256=plan_sha)
        expected = {name: sha256(raw).hexdigest() for name, raw in {
            **initial, **outputs, "plan.json": plan_raw, "worker.json": worker_raw,
            "supervision.json": supervision_raw, "index.html": html}.items()}
        result["proposal_manifest_sha256"] = _seal(study, STUDY_PROFILE, "completed", deadline,
                                                  max_bytes=output_limit, media_name=media_name, expected_files=expected)
        result.update(status="completed", proposal_status=proposal["status"], report_path=str(study / "index.html"))
        if progress:
            progress("Save the comparison and retained failures", {"status": "completed", "proposal_status": proposal["status"],
                                                                  "admitted": False})
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = {"stage": stage, "type": type(error).__name__, "message": str(error)[:2000]}
        if not (study / "manifest.json").exists():
            try:
                result["proposal_manifest_sha256"] = _seal(study, STUDY_PROFILE, "failed", time.monotonic() + 5,
                                                          max_bytes=output_limit, media_name=media_name)
            except Exception as seal_error:
                result["proposal_seal_error"] = {"type": type(seal_error).__name__, "message": str(seal_error)[:1000]}
    try:
        files, total = _inventory(out, time.monotonic() + 5, max_bytes=output_limit,
                                  media_name="proposal/" + media_name if media_name else None)
        result["seal_scope"] = "complete_immutable_outputs"
        if len(files) + 2 > MAX_FILES or total + (256 << 10) > output_limit:
            raise ValueError("No space reserved for terminal camera/root seal")
    except Exception as error:
        files = {}
        result.update(status="failed", seal_scope="failure_receipt_only",
                      seal_error={"type": type(error).__name__, "message": str(error)[:1000]})
    result["seconds"] = time.monotonic() - started
    result_raw = _bytes(result)
    write_package_files(out, {"result.json": result_raw})
    files["result.json"] = sha256(result_raw).hexdigest()
    manifest = _bytes({"schema_version": 1, "profile": PROFILE, "status": result["status"],
                       "seal_scope": result["seal_scope"], "files": files})
    write_package_files(out, {"manifest.json": manifest})
    result["manifest_sha256"] = sha256(manifest).hexdigest()
    return result
