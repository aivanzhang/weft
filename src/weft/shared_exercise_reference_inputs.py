"""Selected saved reference-controller inputs; no physics or full world admission."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from weft import whole_body_reference_report as saved_report
from weft.shared_world_packaging import read_package_file, validate_package_layout


PROFILE = "shared-exercise-reference-inputs-v1"
MAX_INPUT_BYTES = 32 * 1024 * 1024
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 128 * 1024
MAX_MEMBERS = 1024
PROPOSAL_GATES = frozenset({
    "complete_bounded_solve", "first_reference_frozen", "fixed_nonreference_fields",
    "source_times_and_count", "angle_correction", "correction_rate", "joint_limits",
    "pointwise_floor_nonregression", "floor_feasibility", "endpoint_depth_change",
    "source_endpoint_image_nonregression",
})


@dataclass(frozen=True, slots=True)
class ReferenceInputs:
    """Exact authenticated local snapshots, not an admitted scene or controller."""

    proposal_directory: Path
    source_directory: Path
    proposal_manifest_bytes: bytes
    scene_manifest_bytes: bytes
    original_human_bytes: bytes
    candidate_human_bytes: bytes
    report_bytes: bytes
    evidence_bytes: bytes
    scene_bytes: bytes
    task_bytes: bytes
    _receipt_json: str

    def receipt(self):
        return json.loads(self._receipt_json)

    @property
    def identities(self):
        return self.receipt()["identities"]


def _require(condition, message):
    if not condition:
        raise ValueError("Saved reference inputs: " + message)


def _path(value):
    path = Path(value).absolute()
    _require(".." not in path.parts and len(path.parts) <= 64
             and not any(part.is_symlink() for part in (path, *path.parents)), "unsafe input path")
    return path


def _pin(value):
    _require(value is None or isinstance(value, str) and saved_report._SHA.fullmatch(value), "expected exact SHA256")


def _manifest(raw):
    document = saved_report._json(raw)
    files = document.get("files")
    _require(type(document.get("schema_version")) is int and document["schema_version"] == 1
             and isinstance(files, dict) and 1 <= len(files) <= MAX_MEMBERS, "bounded versioned manifest required")
    _require("manifest.json" not in files and "failure.json" not in files, "failed or self-referential package")
    for name, identity in files.items():
        _require(isinstance(name, str) and len(name) <= 512
                 and isinstance(identity, str) and saved_report._SHA.fullmatch(identity), "invalid manifest member or hash")
    validate_package_layout([*files, "manifest.json"])
    return document, files


def _legacy_scene_completion(source, files, scene, read):
    # This constructor sealed completion records, not a status field on its manifest.
    names = ("reconstruction.json", "source-binding-verification.json", "reference-feasibility.json")
    _require(set(names) <= set(files) and not {"status.json", "result.json", "report.json", "construction-seed.json"} & set(files),
             "unsupported legacy scene completion metadata")
    documents = {name: saved_report._json(read(source, name, files[name])) for name in names}
    reconstruction, verification, feasibility = (documents[name] for name in names)
    canonical = saved_report._canonical
    _require(type(reconstruction.get("schema_version")) is int and reconstruction["schema_version"] == 1
             and reconstruction.get("profile") == scene["profile"] == "shared-world-v2"
             and reconstruction.get("mode") == "source_equipment_initial_geometry"
             and "status" not in reconstruction and reconstruction.get("physical_validation") == "unverified"
             and reconstruction.get("scene_sha256") == canonical(scene)
             and reconstruction.get("preserved_human_spec_sha256") == canonical(scene["human_spec"]),
             "legacy construction completion differs from original scene/body")
    interval = reconstruction.get("reference_interval")
    scene_task = scene["task"]
    duration = scene["human_spec"]["reference"][-1]["time_s"] - scene["human_spec"]["reference"][0]["time_s"]
    _require(isinstance(interval, dict) and interval.get("mode") == "complete_reference"
             and interval.get("selected") == {key: scene_task[key] for key in ("start_offset_s", "horizon_steps")}
             and interval.get("first_reference_time_s") == scene["human_spec"]["reference"][0]["time_s"]
             and interval.get("last_reference_time_s") == scene["human_spec"]["reference"][-1]["time_s"]
             and interval.get("dt") == scene["dt"] == 1 / 30
             and interval.get("substeps") == scene["substeps"]
             and scene_task.get("start_offset_s") == 0
             and type(scene_task.get("horizon_steps")) is int and 1 <= scene_task["horizon_steps"] <= 1800
             and 0 < duration <= 60
             and math.isclose(scene_task["horizon_steps"] * scene["dt"], duration, rel_tol=0, abs_tol=1e-9),
             "legacy construction did not preserve the complete original interval")
    bindings = {"base_manifest_sha256": "inputs/base-manifest.json",
                "proposal_manifest_sha256": "inputs/proposal-manifest.json",
                "evidence_manifest_sha256": "inputs/evidence-manifest.json", "proposal_sha256": "geometry-proposal.json"}
    _require(all(name in files and reconstruction.get(key) == files[name] for key, name in bindings.items()),
             "legacy construction input identities differ from its seal")
    binding = scene["source_binding"]
    _require(reconstruction.get("source_binding_verification") == verification
             and verification.get("source_evidence_verified") is True
             and verification.get("mode") == binding.get("mode") == "source_qualified"
             and verification.get("source_sha256") == binding["source_sha256"]
             and type(verification.get("source_frames")) is int and verification["source_frames"] > 0
             and verification["source_frames"] == binding.get("total_source_frames")
             and "construction-evidence.json" in files
             and verification.get("construction_sha256") == files["construction-evidence.json"],
             "legacy sealed source-completion records disagree")
    _require(type(feasibility.get("schema_version")) is int and feasibility["schema_version"] == 1
             and feasibility.get("profile") == "shared-world-feasibility-v1"
             and feasibility.get("scene_sha256") == canonical(scene)
             and feasibility.get("physical_validation") == "unverified"
             and feasibility.get("status") in {"feasible", "infeasible"} and isinstance(feasibility.get("summary"), dict)
             and reconstruction.get("reference_feasibility") == {key: feasibility[key] for key in ("status", "summary")},
             "legacy saved feasibility audit is incomplete or has different scene identity")
    return {"route": "legacy_source_equipment_constructor", "selected_members": {name: files[name] for name in names},
            "reference_feasibility_status": feasibility["status"],
            "scope": "Sealed constructor completion, not current source admission, geometry feasibility or task success."}


def load_reference_inputs(proposal, scene_run, task_path, *, expected_proposal_manifest_sha256=None,
                          expected_scene_manifest_sha256=None, expected_task_sha256=None,
                          max_seconds=10, max_input_bytes=MAX_INPUT_BYTES):
    """Freeze exact selected bytes; callers still admit sources, runtime and provider.

    Without an expected seal, the current bounded local manifest is frozen and
    rechecked, not externally authenticated. The task is required and never built.
    """
    try:
        return _load(proposal, scene_run, task_path, expected_proposal_manifest_sha256,
                     expected_scene_manifest_sha256, expected_task_sha256, max_seconds, max_input_bytes)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError, RecursionError) as error:
        raise ValueError("Saved reference inputs: malformed metadata contract") from error


def _load(proposal, scene_run, task_path, proposal_pin, scene_pin, task_pin, max_seconds, max_input_bytes):
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 10,
             "time bound must be positive and at most ten seconds")
    _require(type(max_input_bytes) is int and 1024 <= max_input_bytes <= MAX_INPUT_BYTES,
             "aggregate read bound must be 1 KiB–32 MiB, including rechecks")
    for pin in (proposal_pin, scene_pin, task_pin):
        _pin(pin)
    proposal, source, task_path = _path(proposal), _path(scene_run), _path(task_path)
    started = time.monotonic()
    deadline, snapshots, selected_bytes = started + max_seconds, {}, 0

    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Saved reference inputs exceeded their cooperative time bound")

    def not_failed(directory):
        check()
        _path(directory)
        failure = directory / "failure.json"
        _require(not failure.exists() and not failure.is_symlink(), "package has a failure marker")

    def read(directory, name, expected=None, maximum=MAX_FILE_BYTES):
        nonlocal selected_bytes
        check()
        key = (directory, name)
        if key not in snapshots:
            # Reserve the same bytes for an exact final recheck; this is an aggregate read cap.
            remaining = max_input_bytes // 2 - selected_bytes
            _require(remaining > 0, "selected metadata byte bound exhausted")
            value = read_package_file(directory, name, max_bytes=min(maximum, remaining), expected_sha256=expected)
            snapshots[key] = value
            selected_bytes += len(value)
        value = snapshots[key]
        _require(0 < len(value) <= maximum, "selected file exceeds its byte bound")
        _require(expected is None or sha256(value).hexdigest() == expected, "selected identity differs")
        return value

    not_failed(proposal); not_failed(source)
    proposal_manifest_raw = read(proposal, "manifest.json", proposal_pin, MAX_MANIFEST_BYTES)
    proposal_manifest, proposal_files = _manifest(proposal_manifest_raw)
    _require(proposal_manifest.get("profile") == "whole-body-reference-candidate-study-v1"
             and proposal_manifest.get("status") == "completed", "completed proposal study required")
    _require(set(saved_report.MEMBERS) <= set(proposal_files), "proposal lacks required sealed inputs")
    raw = {name: read(proposal, name, proposal_files[name]) for name in saved_report.MEMBERS}
    documents = {name: saved_report._json(value) for name, value in raw.items()}
    model = saved_report._model(documents, {name: proposal_files[name] for name in raw}, check)
    report = documents["report.json"]
    _require(report.get("status") == "unadmitted_construction_proposal" and report.get("promoted") is False
             and report.get("geometry_admission") == "unadmitted_pending_independent_current_scene_checks"
             and report.get("failures") == [] and report.get("budget_reason") is None
             and set(report["gates"]) == PROPOSAL_GATES and all(value is True for value in report["gates"].values()),
             "complete unpromoted proposal with all eleven saved gates passing required")
    _require(all(report["contract"].get(key) == value for key, value in {
        "sample_hz": 30, "maximum_grid_samples": 1801, "numerical_tolerance": 1e-8,
        "maximum_endpoint_depth_change_model_units": .08}.items()), "saved geometry contract differs")
    times = [row["time_s"] for row in documents["inputs/human.json"]["reference"]]
    # Match the fitter's saved schedule exactly; this checks coverage, not geometry.
    grid_count = math.floor((times[-1] - times[0]) * 30 + 1e-9) + 1
    _require(grid_count <= 1801, "saved control-grid bound exceeded")
    schedule = [("source_pose", float(stamp), index) for index, stamp in enumerate(times)]
    schedule += [("control_grid", min(float(times[-1]), float(times[0] + index / 30)), None)
                 for index in range(grid_count)]
    _require([(row["kind"], row["time_s"], row.get("source_reference_index"))
              for row in report["geometry_checks"]] == schedule,
             "passing proposal lacks the complete original source and 30-Hz control-grid schedule")
    _require(model["source_references_without_geometry"] == 0 and model["grid"]
             and all(row.get("maximum_depth_change_model_units") is not None
                     and all(row.get(side + "_max_penetration_model_units") is not None
                         and isinstance(row.get(side + "_floor_support_gaps"), list)
                         and all(gap is not None for gap in row[side + "_floor_support_gaps"])
                         for side in ("before", "after")) for row in report["geometry_checks"]),
             "passing proposal lacks complete saved support checks")
    _require(all(point.get("weight") is not None and (point["weight"] == 0
                 or point.get("error_before_px") is not None and point.get("error_after_px") is not None)
                 for reference in report["references"] for point in reference["points"]),
             "passing image gate requires finite positive-weight endpoint errors")
    _require(all(row.get("no_worse_floor") is True and row["maximum_depth_change_model_units"] <= .08 + 1e-8
                 and all(after >= min(0., before) - 1e-8 for before, after in
                         zip(row["before_floor_support_gaps"], row["after_floor_support_gaps"], strict=True))
                 for row in report["geometry_checks"]), "saved geometry contradicts passing floor/depth gates")

    scene_manifest_raw = read(source, "manifest.json", scene_pin, MAX_MANIFEST_BYTES)
    scene_manifest, scene_files = _manifest(scene_manifest_raw)
    legacy_scene = set(scene_manifest) == {"schema_version", "files"}
    _require((scene_manifest.get("status") == "ready" or legacy_scene) and "scene.json" in scene_files,
             "ready sealed original scene required")
    scene_raw = read(source, "scene.json", scene_files["scene.json"])
    scene = saved_report._json(scene_raw)
    original = documents["inputs/human.json"]
    canonical = saved_report._canonical
    _require(type(scene.get("schema_version")) is int and scene["schema_version"] == 1
             and scene.get("profile") in {"shared-world-v2", "shared-world-v3"}
             and scene.get("physical_validation") == "unverified" and scene.get("controlled_id") == "human"
             and isinstance(scene.get("human_spec"), dict) and canonical(scene["human_spec"]) == canonical(original),
             "original full-reference scene/body identity differs")
    _require(isinstance(scene.get("source_binding"), dict)
             and scene["source_binding"].get("source_sha256") == model["source_sha256"], "original scene source differs")
    scene_completion = (_legacy_scene_completion(source, scene_files, scene, read) if legacy_scene else
                        {"route": "manifest_status_ready", "status": "ready"})
    if "human.json" in scene_files:
        _require(read(source, "human.json", scene_files["human.json"]) == raw["inputs/human.json"],
                 "sealed original human bytes differ from proposal input")
    task_raw = read(task_path.parent, task_path.name, task_pin, 64 * 1024)
    task = saved_report._json(task_raw)
    if "exercise-task.json" in scene_files:
        _require(read(source, "exercise-task.json", scene_files["exercise-task.json"], 64 * 1024) == task_raw,
                 "supplied task differs from the sealed original task")
    task_profile = "shared-exercise-task-v2" if scene["profile"] == "shared-world-v3" else "shared-exercise-task-v1"
    _require(type(task.get("schema_version")) is int and task["schema_version"] == 1
             and task.get("profile") == task_profile and task.get("scene_sha256") == canonical(scene),
             "supplied exact task is not bound to the original scene")

    for (directory, name), value in snapshots.items():
        check(); _path(directory)
        read_package_file(directory, name, max_bytes=max(1, len(value)), expected_sha256=sha256(value).hexdigest())
    not_failed(proposal); not_failed(source)
    check()
    identities = {"proposal_manifest_sha256": sha256(proposal_manifest_raw).hexdigest(),
        "scene_manifest_sha256": sha256(scene_manifest_raw).hexdigest(), "source_sha256": model["source_sha256"],
        "original_human_sha256": sha256(raw["inputs/human.json"]).hexdigest(),
        "original_human_canonical_sha256": canonical(original),
        "candidate_human_sha256": sha256(raw["candidate-human.json"]).hexdigest(),
        "candidate_canonical_sha256": model["candidate_canonical_sha256"],
        "proposal_report_sha256": sha256(raw["report.json"]).hexdigest(),
        "input_evidence_sha256": sha256(raw["inputs/evidence-derived.json"]).hexdigest(),
        "scene_file_sha256": sha256(scene_raw).hexdigest(), "scene_sha256": canonical(scene),
        "task_file_sha256": sha256(task_raw).hexdigest(), "task_sha256": canonical(task)}
    receipt = {"schema_version": 1, "profile": PROFILE, "status": "selected_metadata_verified",
        "identities": identities, "scene_completion": scene_completion,
        "external_pins_supplied": {"proposal_manifest": proposal_pin is not None,
            "scene_manifest": scene_pin is not None, "task": task_pin is not None},
        "selected_files": [{"directory": str(directory), "name": name, "bytes": len(value),
                            "sha256": sha256(value).hexdigest()} for (directory, name), value in snapshots.items()],
        "input_bytes_read": 2 * selected_bytes, "maximum_input_bytes": max_input_bytes,
        "maximum_seconds": max_seconds, "seconds": time.monotonic() - started,
        "saved_gates": dict(report["gates"]), "promoted": False, "physical_validation": "unverified",
        "preparation": {"source_directory": str(source), "expected_manifest_sha256": identities["scene_manifest_sha256"],
            "required_next_checks": ["Full independent world-source admission", "Original scene/task validation",
                "Exact current runtime and trusted velocity helper verification", "Numerical reference-controller validation"]},
        "scope": "Selected metadata only; unselected members, media and runtime bytes were not read. Local seals without supplied pins are frozen identities, not external authentication. No source selection, geometry, physics, task rewrite or promotion."}
    return ReferenceInputs(proposal, source, proposal_manifest_raw, scene_manifest_raw, raw["inputs/human.json"],
        raw["candidate-human.json"], raw["report.json"], raw["inputs/evidence-derived.json"], scene_raw, task_raw,
        json.dumps(receipt, sort_keys=True, allow_nan=False))
