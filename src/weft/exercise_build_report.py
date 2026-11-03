"""Offline inspection of saved exercise builds, including rejected construction.

Only selected manifest members are verified and copied. A report is neither
whole-package admission nor a playable world, and never runs source code.
"""

import hashlib
from collections import Counter
from html import escape
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time

from weft import evidence_scorecard, reference_equipment_report
from weft.exercise_coverage import summarize_coverage
from weft.exercise_vision import review_competing_gaps
from weft.io import canonical_hash, local_time
from weft.lab_inspection import _open as _open_member, _pairs, _nonfinite, _float


PROFILE = "shared-exercise-build-report-v1"
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_COPIED_FILES = 128
MAX_MANIFEST_FILES = 5632
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 24 * 1024 * 1024
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_VALUES = 250000
MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_DIAGNOSTIC_BYTES = 128 * 1024
MAX_DIAGNOSTIC_SPANS = 64
MAX_SECONDS = 30
_RESERVE = 5 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}")
_VIDEO = re.compile(r"source\.(webm|mp4|m4v|mov|mkv|avi)")
_PREFIXES = ("body_source", "recovered_poses", "pose_proposals/selected", "pose_proposals/whole_image",
             "body_seed", "candidate", "lab/run", "learning_workflow/lab/run")
_BODY_DOCS = {
    "body-source.json": "Body construction receipt",
    "source.json": "Original video attribution and identity",
    "source-clock.json": "Preserved source clocks",
    "evidence.json": "Original pose samples and missing samples",
    "observations.json": "Canonical body observations",
    "pose-landmarks.json": "Model-reported body landmarks",
    "human.json": "Authored body model and retained references",
    "human.xml": "Body model definition — not a playable world",
    "pose-quality.json": "Pose visibility and quality checks",
    "root-motion.json": "Image-derived body movement and gaps",
    "body-frame.json": "Body coordinate-frame assumptions",
    "floor-assumption.json": "Authored first-pose floor placement",
    "identity-audit.json": "Person-selection audit",
    "model.json": "Pose model identity",
    "run-config.json": "Saved pose options — not evidence that training ran",
    "joint-angles.json": "Derived joint angles",
    "spatial-pose.json": "Derived spatial pose",
}
_VISUALS = {"background.jpg": "Saved source-frame preview",
            "tracks-contact-sheet.png": "Saved pose contact sheet",
            "tracks.gif": "Saved pose annotations — not a simulated body"}
_OTHER_DOCS = {
    "result.json": ("Build", "Exact terminal result"),
    "settings.json": ("Build", "Requested options — execution is listed separately"),
    "stages.json": ("Build", "Recorded stage history"),
    "geometry_source/resume.json": ("Equipment", "Saved conditional geometry reuse and retained study claims"),
    "geometry_source/study-manifest.json": ("Equipment", "Original silhouette study seal — selected metadata is copied"),
    "geometry_source/candidate-manifest.json": ("Equipment", "Selected compound package seal"),
    "geometry_source/baseline-manifest.json": ("Equipment", "Original box package seal"),
    "geometry_source/geometry-input/scene.json": ("Equipment", "Original box scene — retained alternative"),
    "geometry_source/geometry-input/exercise-task.json": ("Equipment", "Original box task and gates"),
    "geometry_source/geometry/scene.json": ("Equipment", "Selected saved compound scene — not executed by this report"),
    "geometry_source/geometry/exercise-task.json": ("Equipment", "Exact selected task — only its scene identity changed"),
    "geometry_source/geometry/previous-geometry.json": ("Equipment", "Original geometry and provenance retained by the producer"),
    "geometry_source/geometry/geometry-hypothesis.json": ("Equipment", "Saved admission claims — not rerun here"),
    "geometry_source/geometry-config.json": ("Equipment", "Exact saved compound configuration"),
    "geometry_source/settings.json": ("Equipment", "Original conditional image settings"),
    "geometry_source/report.json": ("Equipment", "Original source-fit and clearance claims — not newly measured"),
    "input-preflight.json": ("Build", "Original input checks and identities"),
    "person-inventory.json": ("World", "User-declared person/ground scope — absence is not detected"),
    "support-priors.json": ("World", "Exact supplied static-support assumptions — not recovered contact"),
    "support-config.json": ("World", "Resolved support assumptions pinned to the original body manifest"),
    "pose_proposals/source.json": ("Original", "Pinned video attribution — original may not be copied"),
    "pose_proposals/report.json": ("Body", "Source-only crop preparation result"),
    "pose_proposals/crop-plan.json": ("Body", "Proposed image crops and eligibility checks"),
    "pose_proposals/decisions.json": ("Body", "Accepted and rejected pose replacements"),
    "pose_proposals/decode-audit.json": ("Body", "Independent decode and original-clock checks"),
    "recovered_poses/pose-recovery.json": ("Body", "Verified pose recovery — not a body or playable world"),
    "body_seed/construction-seed.json": ("Body", "Construction-only seed receipt — not a lab"),
    "body_seed/source-binding-verification.json": ("Body", "Body/source binding checks"),
    "proposal/report.json": ("Equipment", "Equipment proposal result and limitations"),
    "proposal/geometry-candidate.rejected.json": ("Equipment", "Rejected geometry candidate"),
    "proposal/geometry-proposal.json": ("Equipment", "Authored geometry proposal"),
    "proposal/input-config.json": ("Equipment", "Explicit camera, shape and material assumptions"),
    "proposal/frame-selection.json": ("Equipment", "Saved equipment-frame selections and complete observations"),
    "proposal/initial-geometry-admission.json": ("Equipment", "Saved initial-geometry check"),
    "proposal/frame-migration.json": ("Equipment", "Coordinate-frame translation checks"),
    "proposal/source-binding.json": ("Equipment", "Body/equipment source-slot bindings"),
    "proposal/source-equivalences.json": ("Equipment", "Original/derivative source mapping identities"),
    "candidate/reconstruction.json": ("World", "Saved shared-world construction result"),
    "candidate/inventory.json": ("World", "Source-bound person-only declaration"),
    "candidate/reference-feasibility.json": ("World", "Reference geometry audit — not task success"),
    "candidate/source-binding-verification.json": ("World", "Verified body/source identities and complete slots"),
    "candidate/scene.json": ("World", "Constructed scene definition — not executed here"),
    "candidate/exercise-task.json": ("World", "Frozen exercise task and gates"),
    "candidate/inputs/body-source-manifest.json": ("Body", "Exact body-source manifest used for support construction"),
    "learning_workflow/result.json": ("Learning", "Saved learning workflow outcome"),
    "learning_workflow/plan.json": ("Learning", "Frozen training/evaluation plan"),
    "learning_workflow/selected/selection.json": ("Learning", "Training-only policy selection"),
    "learning_workflow/evaluation/evaluation.json": ("Learning", "Independent evaluation result"),
}
_POSE_ERRORS = {
    "competing_poses": "Multiple people or competing pose hypotheses were detected; inspect the source and raw poses",
    "visibility": "Insufficient full-body visibility: need at least 8 usable poses and 65% usable sampled frames",
    "motion": "Insufficient measurable exercise motion; a repeated still image is not a dynamic exercise video",
    "duration": "Usable pose sequence must span at least 0.4 seconds",
}
_BODY_POSE_REJECTION = "Rejected/unsupported pose evidence cannot become a construction body"


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Build inspection reached its cooperative wall-time bound")


def _text(value, limit=2000):
    if not isinstance(value, str) or len(value) > limit or any(ord(c) < 32 and c not in "\n\r\t" for c in value):
        raise ValueError("Build inspection text is invalid or oversized")
    return value


def _name(value):
    _text(value, 512)
    if (not value or "\\" in value or ":" in value or value.startswith("/")
            or any(c in value for c in "\n\r\t") or any(p in ("", ".", "..") for p in value.split("/"))):
        raise ValueError("Build inspection requires safe relative artifact paths")
    return value


def _absolute(value):
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Build inspection paths must not contain '..'")
    return path.absolute()


def _directory(path, *, create=False):
    fd = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read(fd, name, limit, deadline):
    _check(deadline)
    with os.fdopen(_open_member(fd, _name(name)), "rb") as stream:
        before = os.fstat(stream.fileno())
        if before.st_size > limit:
            raise ValueError(f"Inspection input exceeds its byte bound: {name}")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (len(raw) != before.st_size or len(raw) > limit
            or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
               (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
        raise ValueError(f"Inspection input changed while reading: {name}")
    _check(deadline)
    return raw


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_nonfinite, parse_float=_float)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("Invalid build inspection JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Build inspection documents must be JSON objects")
    stack, count = [(iter((value,)), 0)], 0
    while stack:
        iterator, depth = stack[-1]
        try:
            item = next(iterator)
        except StopIteration:
            stack.pop()
            continue
        count += 1
        if depth > MAX_JSON_DEPTH or count > MAX_JSON_VALUES:
            raise ValueError("Invalid build inspection JSON: nesting/value bound exceeded")
        if isinstance(item, (dict, list)):
            stack.append((iter(item.values() if isinstance(item, dict) else item), depth + 1))
    return value


class _Source:
    def __init__(self, path, deadline):
        self.path, self.deadline = path, deadline
        self.fd = _directory(path)
        self.identity = os.fstat(self.fd)
        self.raw, self.documents, self.consumed, self.pins = {}, {}, 0, {}
        try:
            self.manifest_raw = _read(self.fd, "manifest.json", MAX_MANIFEST_BYTES, deadline)
            self.manifest = _json(self.manifest_raw)
            self.files = self.manifest.get("files")
            if (self.manifest.get("schema_version") != 1 or type(self.manifest.get("schema_version")) is not int
                    or self.manifest.get("seal_scope") not in ("complete_immutable_outputs", "failure_receipt_only")
                    or not isinstance(self.files, dict) or not 1 <= len(self.files) <= MAX_MANIFEST_FILES):
                raise ValueError("Expected a bounded sealed exercise-build manifest")
            for name, digest in self.files.items():
                _name(name)
                if name == "manifest.json" or not isinstance(digest, str) or not _SHA.fullmatch(digest):
                    raise ValueError("Invalid exercise-build manifest identity")
                if any(parent.as_posix() in self.files for parent in PurePosixPath(name).parents):
                    raise ValueError("Exercise-build file/directory paths overlap")
            self.result = self.document("result.json", required=True)
            if (self.result.get("schema_version") != 1 or type(self.result.get("schema_version")) is not int
                    or self.result.get("seal_scope") != self.manifest["seal_scope"]
                    or self.result.get("lab_status") not in ("ready", "not_ready")):
                raise ValueError("Expected a matching terminal exercise-build receipt")
            _text(self.result.get("status"), 128)
            self.receipt_only = self.manifest["seal_scope"] == "failure_receipt_only"
            if self.receipt_only:
                if self.manifest.get("status") != "failed" or set(self.files) != {"result.json"}:
                    raise ValueError("A failure-only seal may cover only the terminal receipt")
            elif self.manifest.get("status") != self.result["status"]:
                raise ValueError("Build manifest and terminal status differ")
        except BaseException:
            os.close(self.fd)
            raise

    def read(self, name, limit=MAX_FILE_BYTES):
        if name not in self.files:
            raise ValueError(f"Inspection requires a manifest-listed artifact: {name}")
        raw = _read(self.fd, name, limit, self.deadline)
        if _sha(raw) != self.files[name]:
            raise ValueError(f"Inspection artifact hash differs from its source manifest: {name}")
        self.pins[name] = self.files[name]
        return raw

    def document(self, name, *, required=False):
        if name not in self.files:
            if required:
                raise ValueError(f"Missing sealed inspection metadata: {name}")
            return {}
        if name not in self.documents:
            raw = self.read(name, min(MAX_JSON_BYTES, MAX_METADATA_BYTES - self.consumed))
            self.consumed += len(raw)
            self.raw[name], self.documents[name] = raw, _json(raw)
        return self.documents[name]

    def recheck(self):
        fd = _directory(self.path)
        try:
            identity = os.fstat(fd)
            if (identity.st_dev, identity.st_ino) != (self.identity.st_dev, self.identity.st_ino):
                raise ValueError("Inspection source directory changed")
        finally:
            os.close(fd)
        if _read(self.fd, "manifest.json", MAX_MANIFEST_BYTES, self.deadline) != self.manifest_raw:
            raise ValueError("Inspection source manifest changed")
        for name in tuple(self.pins):
            self.read(name)


def _history(result, settings, receipt_only):
    rows = result.get("stages")
    if not isinstance(rows, list) or len(rows) > 2048:
        raise ValueError("Missing or oversized saved build stage history")
    history, previous = [], -1.
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid saved build stage")
        name, status_, seconds = _text(row.get("stage"), 128), _text(row.get("status"), 128), row.get("elapsed_seconds")
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or not previous <= seconds <= 86400:
            raise ValueError("Invalid or reversed saved stage clock")
        previous = seconds
        history.append({"stage": name, "status": status_, "elapsed_seconds": seconds})
    latest = {row["stage"]: row for row in rows}
    person_only = settings.get("scene_scope") == "human_only"
    static_supports = settings.get("scene_scope") == "human_static_supports" or "support_priors" in latest
    training = bool(settings.get("training")) or any(name.startswith("learning_") for name in latest)
    recovery = settings.get("pose_recovery") == "selective-crop" or any(
        name in latest for name in ("source_pose_proposals", "source_pose_recovery", "source_body_source"))
    source_steps = [
        ("source_pose_proposals", "Inspect original poses and propose image crops", True),
        ("source_pose_recovery", "Verify accepted pose replacements", True),
        ("source_body_source", "Construct the body from verified poses", True),
    ] if recovery else [("body_source", "Estimate body from video", bool(settings.get("video")) or "body_source" in latest)]
    construction_steps = [("support_priors", "Resolve authored static-support assumptions", True)] if static_supports else [
        ("body_seed", "Prepare construction-only body", not person_only and (bool(settings.get("human_run") or settings.get("video")) or "body_seed" in latest)),
        ("equipment_priors", "Propose equipment geometry", not person_only),
    ]
    definitions = source_steps + construction_steps + [
        ("shared_world", "Construct the shared world", True),
        ("learning_training", "Train a controller", training),
        ("learning_training_selection", "Select using training evidence", training),
        ("learning_independent_evaluation", "Test on independent starts", training),
        ("learning_interactive_lab" if training else "interactive_lab", "Prepare the interactive lab", True),
    ]
    if settings.get("scene_scope") == "saved_conditional_geometry" or "geometry_source" in latest:
        definitions = [("geometry_source", "Reuse saved conditional equipment geometry", True),
                       ("interactive_lab", "Prepare the interactive lab", True)]
    stages = []
    for name, label, requested in definitions:
        row = latest.get(name)
        state = ("completed" if row["status"] in {"ready", "proposal_ready", "trained", "selected", "evaluated"}
                 else "stopped") if row else ("unknown" if receipt_only else "not_run" if requested else "not_requested")
        if (static_supports and name == "learning_training_selection" and row
                and row["status"] == "not_applicable" and row.get("performed") is False):
            state = "not_applicable"
        stages.append({"stage": name, "label": label, "state": state,
                       "saved_status": row["status"] if row else None,
                       "elapsed_seconds": row["elapsed_seconds"] if row else None})
    return stages, history


def _reasons(result):
    rows = result.get("reasons", [])
    if not isinstance(rows, list) or len(rows) > 32:
        raise ValueError("Invalid or oversized saved build reasons")
    reasons = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid saved build reason")
        reasons.append({key: _text(row[key]) for key in ("stage", "code", "error_type", "message") if key in row})
    for key in ("error", "sealing_error"):
        if key in result:
            row = result[key]
            if not isinstance(row, dict):
                raise ValueError("Invalid saved build failure")
            reasons.append({"stage": key, **{name: _text(row[name]) for name in ("type", "message") if name in row}})
    return reasons


def _selection(source):
    chosen = {"result.json": ("Build", "Exact terminal result")}
    if source.receipt_only:
        return chosen, None, None
    chosen.update({name: value for name, value in _OTHER_DOCS.items() if name in source.files})
    if source.document("settings.json").get("scene_scope") in {"human_only", "human_static_supports", "saved_conditional_geometry"}:
        chosen.update({name: ("World", "Saved final lab scene — not executed here")
                       for name in ("lab/run/scene.json", "learning_workflow/lab/run/scene.json") if name in source.files})
    if source.document("settings.json").get("scene_scope") == "saved_conditional_geometry":
        reports = [name for name in source.files if re.fullmatch(r"geometry_source/objects/[a-z][a-z0-9_-]{0,63}/result\.json", name)]
        if len(reports) > 4:
            raise ValueError("Saved geometry report exceeds its object bound")
        chosen.update({name: ("Equipment", "Exact retained per-object image-fit claim — not rerun") for name in reports})
    # A failed body writer can contain metadata but no poses; retain earlier verified source evidence.
    body_prefix = next((prefix for prefix in _PREFIXES if prefix + "/evidence.json" in source.files), None)
    if body_prefix is None:
        body_prefix = next((prefix for prefix in _PREFIXES if any(prefix + "/" + name in source.files for name in _BODY_DOCS)), None)
    if body_prefix:
        chosen.update({body_prefix + "/" + name: ("Body", label) for name, label in _BODY_DOCS.items()
                       if body_prefix + "/" + name in source.files})
        chosen.update({body_prefix + "/" + name: ("Body", label) for name, label in _VISUALS.items()
                       if body_prefix + "/" + name in source.files})
        chosen.update(_diagnostic_paths(source, body_prefix))
    video = None
    for prefix in _PREFIXES:
        candidates = [name for name in source.files if name.startswith(prefix + "/") and _VIDEO.fullmatch(name[len(prefix) + 1:])]
        if candidates:
            if len(candidates) != 1:
                raise ValueError("Ambiguous original video in saved build")
            video = candidates[0]
            chosen[video] = ("Original", "Preserved original video")
            if prefix + "/source.json" in source.files:
                chosen[prefix + "/source.json"] = ("Original", "Original video attribution and identity")
            break
    for name in source.files:
        if re.fullmatch(r"proposal/inputs/observations/[0-9a-f]{64}\.json", name):
            chosen[name] = ("Equipment", "Canonical source observations — exact source identity in JSON")
    if len(chosen) > MAX_COPIED_FILES:
        raise ValueError("Selected inspection artifacts exceed the file bound")
    return chosen, body_prefix, video


def _reference_equipment_paths(source):
    if source.receipt_only:
        return {}
    prefixes = [prefix for prefix in ("learning_workflow/lab/run", "lab/run")
                if prefix + "/human-reference.json" in source.files]
    if not prefixes:
        return {}
    names = {"scene.json": "Exact lab scene for reference overlap inspection",
             "human-reference.json": "Complete saved kinematic poses and collision records — not physical contact",
             "exercise-lab.json": "Lab identity binding for saved reference inspection"}
    for name in names:
        if any(prefix + "/" + name not in source.files for prefix in prefixes):
            raise ValueError("Saved reference inspection requires the matching lab scene and contract")
        if len({source.files[prefix + "/" + name] for prefix in prefixes}) != 1:
            raise ValueError("Saved lab reference inputs disagree; no automatic preference")
    return {prefixes[0] + "/" + name: label for name, label in names.items()}


def _reference_equipment(source, source_sha256=None):
    paths = _reference_equipment_paths(source)
    if not paths:
        return {"status": "unavailable", "physical_validation": "unverified",
                "reason": "No sealed complete lab reference poses were retained; equipment clearance during the reference is unknown."}
    needed = 0
    for name in paths:
        if name in source.documents:
            continue
        fd = _open_member(source.fd, name)
        try:
            size = os.fstat(fd).st_size
        finally:
            os.close(fd)
        if size > MAX_JSON_BYTES:
            return {"status": "unavailable", "physical_validation": "unverified",
                    "reason": "Complete reference inputs exceed the bounded metadata reader; no poses were sampled or clipped."}
        needed += size
    if needed > MAX_METADATA_BYTES - source.consumed:
        return {"status": "unavailable", "physical_validation": "unverified",
                "reason": "Complete reference inputs exceed the remaining report metadata budget; no partial overlap count was inferred."}
    scene_name, reference_name, contract_name = paths
    scene = source.document(scene_name, required=True)
    contract = source.document(contract_name, required=True)
    scene_sha = canonical_hash(scene)
    if (contract.get("profile") != "shared-world" or contract.get("schema_version") != 1
            or type(contract["schema_version"]) is not int or contract.get("scene_sha256") != scene_sha
            or source.result.get("scene_sha256", scene_sha) != scene_sha):
        raise ValueError("Saved reference scene and lab/build identities differ")
    value = reference_equipment_report.summarize_reference_equipment(scene,
        source.document(reference_name, required=True), check=lambda: _check(source.deadline))
    if (source_sha256 is not None and value["status"] != "unavailable"
            and value["source_sha256"] != source_sha256):
        raise ValueError("Saved reference and inspected video identities differ")
    value.update(scene_artifact=scene_name, reference_artifact=reference_name, contract_artifact=contract_name,
                 selected_members_sha256={name: source.files[name] for name in paths},
                 implementation_sha256=_sha(Path(reference_equipment_report.__file__).read_bytes()))
    return value


def _diagnostic_paths(source, body_prefix):
    paths = {}
    evidence_sha = source.files.get(body_prefix + "/evidence.json")
    for prefix in _PREFIXES:
        # A sibling quality file describes this selection only when its evidence bytes match.
        name = prefix + "/pose-quality.json"
        if evidence_sha and source.files.get(prefix + "/evidence.json") == evidence_sha and name in source.files:
            paths[name] = ("Body", "Saved quality checks for these exact pose samples")
    for prefix in dict.fromkeys((body_prefix, "body_source")):
        name = prefix + "/failure.json"
        if name in source.files:
            paths[name] = ("Body", "Exact failed body-preparation receipt")
    return paths


def _number(value, name, low=0., high=1e6):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Invalid saved pose diagnostic value: " + name)
    return value


def _affected_samples(frames, statuses):
    groups, group = [], []
    for row in frames:
        if row["status"] in statuses:
            group.append(row)
        elif group:
            groups.append(group)
            group = []
    if group:
        groups.append(group)
    spans = []
    for rows in groups[:MAX_DIAGNOSTIC_SPANS]:
        native = [row.get("decoder_timestamp_s") for row in rows]
        available = all(value is not None for value in native)
        monotonic = available and all(right >= left for left, right in zip(native, native[1:]))
        spans.append({"first_frame": rows[0]["frame"], "last_frame": rows[-1]["frame"],
            "sampled_frames": [row["frame"] for row in rows],
            "nominal_start_s": rows[0]["time_s"], "nominal_end_s": rows[-1]["time_s"],
            "native_start_s": native[0] if monotonic else None,
            "native_end_s": native[-1] if monotonic else None,
            "native_status": "saved_decoder_timestamps" if monotonic else "unavailable_or_nonmonotonic"})
    return {"sample_count": sum(len(rows) for rows in groups),
            "original_frames": [row["frame"] for rows in groups for row in rows],
            "sample_runs": spans, "total_sample_runs": len(groups),
            "unlisted_sample_runs": max(0, len(groups) - len(spans)),
            "scope": "Consecutive affected samples, not continuous failing video intervals. Unsampled frames remain unknown; frame numbers start at zero.",
            "clock_scope": "Nominal time is frame index / saved fps. Native values are retained decoder timestamps, not calibrated camera time or synchronized playback."}


def _pose_diagnostics(source, body_prefix, coverage, stages):
    result = {"profile": "exercise-build-diagnostics-v1", "status": "not_available", "checks": [],
              "scope": "Saved source-pose checks only; not a new gate evaluation, verified person count or physical diagnosis.",
              "next_steps": []}
    if source.receipt_only or not body_prefix or coverage.get("status") != "available":
        result["message"] = "No sealed pose-quality details are available. Inspect the exact terminal reason; no cause or measurements are inferred."
        return result
    evidence_name = body_prefix + "/evidence.json"
    evidence = source.document(evidence_name)
    quality, frames = evidence.get("quality", {}), evidence["frames"]
    errors = coverage["quality_errors"]
    if len(set(errors)) != len(errors):
        raise ValueError("Saved pose-quality errors are duplicated")
    if "passes" in quality and (type(quality["passes"]) is not bool or quality["passes"] != (not errors)):
        raise ValueError("Saved pose-quality pass flag and errors disagree")
    paths = _diagnostic_paths(source, body_prefix)
    failures = []
    for name in paths:
        doc = source.document(name)
        if name.endswith("/pose-quality.json"):
            if doc != quality:
                raise ValueError("Saved pose-quality sidecar differs from its exact pose evidence")
        else:
            error = doc.get("error")
            if (doc.get("profile") != "exercise-body-source-v1" or doc.get("schema_version") != 1
                    or type(doc.get("schema_version")) is not int or doc.get("status") != "failed"
                    or doc.get("simulation_ready") is not False or not isinstance(error, dict)):
                raise ValueError("Invalid saved body failure receipt")
            failures.append({"source_name": name, "type": _text(error.get("type"), 128),
                             "message": _text(error.get("message"))})
    counts = dict(Counter(row["status"] for row in frames))
    tracked = [row for row in frames if row["status"] in {"tracked", "tracked_partial"}]
    expected = {"sampled_frames": len(frames), "tracked_frames": len(tracked), "status_counts": counts,
                "tracked_fraction": len(tracked) / len(frames)}
    for key, value in expected.items():
        if key not in quality:
            continue
        actual = quality[key]
        valid = ((isinstance(actual, dict) and all(type(v) is int and v >= 0 for v in actual.values()))
                 if key == "status_counts" else type(actual) in (int, float) if key == "tracked_fraction" else type(actual) is int)
        if not valid or actual != value:
            raise ValueError("Saved pose-quality counts differ from original samples: " + key)
    stopped = next((row["stage"] for row in stages if row["stage"] in {"source_body_source", "body_source"}
                    and row["state"] == "stopped"), None)
    terminal = source.result.get("error", {})
    if stopped and failures and terminal and any(row["type"] != terminal.get("type") or row["message"] != terminal.get("message") for row in failures):
        raise ValueError("Saved body failure and terminal reason disagree")
    result.update(status="saved_quality_rejected" if errors else "no_recorded_pose_quality_failure",
                  evidence={"source_name": evidence_name, "sha256": source.files[evidence_name],
                            "source_sha256": coverage["source_sha256"]},
                  supporting_documents=[{"source_name": name, "sha256": source.files[name]} for name in paths],
                  saved_errors=errors, sampled_frames=len(frames), usable_samples=len(tracked),
                  status_counts=counts, saved_failures=failures, stopped_body_stage=stopped,
                  explains_body_stop=bool(errors and stopped and terminal.get("message") in
                      (_BODY_POSE_REJECTION, "; ".join(errors))),
                  message="The saved pose-quality checks rejected this evidence." if errors else
                          "No pose-quality failure is recorded in these samples; this does not establish body or lab readiness.")
    if not errors:
        return result
    if source.result["lab_status"] == "ready":
        raise ValueError("Rejected pose evidence conflicts with the saved ready lab")

    def check(identifier, label, observed, required, failed, message, affected=None):
        if (_POSE_ERRORS[identifier] in errors) != failed:
            raise ValueError("Saved pose-quality error contradicts its measurements: " + identifier)
        row = {"id": identifier, "label": label, "status": "failed" if failed else "passed",
               "observed": observed, "required": required, "message": message}
        if affected is not None:
            row["affected_samples"] = affected
        result["checks"].append(row)

    competing = counts.get("multiple_people", 0)
    competing_failed = _POSE_ERRORS["competing_poses"] in errors
    review = review_competing_gaps(frames) if competing else None
    if competing_failed and not competing:
        raise ValueError("Saved pose-quality error contradicts its measurements: competing_poses")
    # The current bounded transient envelope can resolve gaps an older saved gate rejected;
    # a saved pass with unresolved recomputed gaps is still a contradiction.
    if competing and not competing_failed and not review["resolved"]:
        raise ValueError("Saved pose-quality error contradicts its measurements: competing_poses")
    check("competing_poses", "Unresolved pose estimates",
          {"samples": competing, "bounded_transient_review": None if review is None else
           {"resolved": review["resolved"], "gaps": review["gaps"]}},
          {"maximum_unresolved_samples": 0},
          competing_failed, f"{competing} sampled frames contain competing pose estimates; unresolved gaps reject the clip. This does not prove that multiple people are present.",
          _affected_samples(frames, {"multiple_people"}) if competing else None)
    if all(key in quality for key in ("minimum_tracked_frames", "minimum_tracked_fraction")):
        minimum = _number(quality["minimum_tracked_frames"], "minimum tracked frames", 1, 900)
        fraction = _number(quality["minimum_tracked_fraction"], "minimum tracked fraction", 0, 1)
        if type(minimum) is not int:
            raise ValueError("Saved minimum pose count must be an integer")
        failed = len(tracked) < minimum or len(tracked) / len(frames) < fraction
        check("visibility", "Usable pose coverage", {"usable_samples": len(tracked), "sampled_frames": len(frames),
              "usable_fraction": len(tracked) / len(frames)}, {"minimum_samples": minimum, "minimum_fraction": fraction}, failed,
              f"{len(tracked)} of {len(frames)} samples ({len(tracked) / len(frames):.1%}) were usable; saved minimums are {minimum} samples and {fraction:.0%}.",
              _affected_samples(frames, set(counts) - {"tracked", "tracked_partial"}) if failed else None)
    if "motion_threshold_radians" in quality and "joint_range_radians" in quality:
        ranges = quality["joint_range_radians"]
        if not isinstance(ranges, dict) or not 1 <= len(ranges) <= 32:
            raise ValueError("Invalid saved joint-range diagnostic")
        largest = max(_number(value, "joint range", 0, 100) for value in ranges.values())
        minimum = _number(quality["motion_threshold_radians"], "minimum joint range", 0, 100)
        check("motion", "Recorded movement", {"maximum_joint_range_radians": largest}, {"minimum_radians": minimum}, largest < minimum,
              f"Largest saved joint-angle range: {largest:.3g} radians; saved minimum: {minimum:.3g}. These are model estimates, not measured anatomy.")
    if _POSE_ERRORS["duration"] in errors:
        span = tracked[-1]["time_s"] - tracked[0]["time_s"] if tracked else None
        if span is None:
            raise ValueError("Saved duration failure has no usable sample endpoints")
        check("duration", "Usable motion duration", {"nominal_seconds": span, "first_frame": tracked[0]["frame"],
              "last_frame": tracked[-1]["frame"]}, {"minimum_seconds": .4}, span < .4,
              f"Usable samples span {span:.3g} nominal seconds; the saved error requires at least 0.4 seconds. Native timing is not substituted.")
    explained = {_POSE_ERRORS[row["id"]] for row in result["checks"] if row["status"] == "failed"}
    result["unquantified_saved_errors"] = [error for error in errors if error not in explained]
    result["next_steps"] = ["Inspect the listed original frames and saved pose annotations alongside the exact recorded errors."]
    if competing:
        result["next_steps"].append("Competing estimates can come from one person; this report cannot decide which estimate is correct. Keep this evidence for identity-resolution work.")
    if competing or _POSE_ERRORS["visibility"] in errors:
        result["next_steps"].append("If the estimates are wrong or limbs are hidden, use a clearer recording with the complete person visible, a fixed camera, and less overlap or motion blur. Keep the full intended exercise interval.")
    if any(_POSE_ERRORS[key] in errors for key in ("motion", "duration")):
        result["next_steps"].append("Provide a continuous recording of the actual movement, including the full intended exercise interval; a still image or very short usable sequence cannot supply motion evidence.")
    result["next_steps"].append("Keep this rejected build and its evidence. Do not lower acceptance thresholds, discard failing frames, or treat a crop proposal as resolved identity.")
    if len(_bytes(result)) > MAX_DIAGNOSTIC_BYTES:
        raise ValueError("Saved pose diagnostics exceed their summary bound")
    return result


def _small_fields(value, keys):
    if not isinstance(value, dict):
        raise ValueError("Expected saved inspection metadata object")
    result = {key: value[key] for key in keys if key in value}
    if len(_bytes(result)) > 128 * 1024:
        raise ValueError("Saved inspection summary fields exceed their bound")
    return result


def _candidate_summary(value, artifact):
    if not isinstance(value, dict):
        raise ValueError("Expected saved inspection metadata object")
    equipment = value.get("equipment", [])
    if not isinstance(equipment, list) or len(equipment) > 128 or any(not isinstance(row, dict) for row in equipment):
        raise ValueError("Invalid or oversized saved equipment list")
    rows = []
    for index, row in enumerate(equipment):
        if "frame_selection_receipt" not in row:
            rows.append(row)
            continue
        receipt = row["frame_selection_receipt"]
        if not isinstance(receipt, dict) or "frame_selection_summary" in row:
            raise ValueError("Invalid or conflicting saved frame-selection metadata")
        # Complete observations stay in the linked artifact, not the bounded display summary.
        summary = {key: receipt[key] for key in ("profile", "method", "status", "selected") if key in receipt}
        summary.update(receipt_sha256=canonical_hash(receipt), receipt_hash_kind="canonical_json_sha256",
                       source_artifact=artifact, equipment_index=index,
                       scope="Display summary only; the complete receipt remains in the linked proposal. Not revalidated here.")
        rows.append({**{key: item for key, item in row.items() if key != "frame_selection_receipt"},
                     "frame_selection_summary": summary})
    selected = {key: value[key] for key in ("status", "camera_hypothesis", "equipment") if key in value}
    if "equipment" in selected:
        selected["equipment"] = rows
    return _small_fields(selected, ("status", "camera_hypothesis", "equipment"))


def _reference_feasibility(source, source_sha256=None):
    name = "candidate/reference-feasibility.json"
    unavailable = {"status": "unavailable", "physical_validation": "unverified",
                   "reason": "No sealed reference-geometry audit is available; motion feasibility is unknown."}
    if source.receipt_only or name not in source.files:
        return unavailable
    audit = source.document(name, required=True)
    scene = source.document("candidate/scene.json", required=True)

    def require(condition, message):
        if not condition:
            raise ValueError("Invalid saved reference feasibility: " + message)

    def number(value, label, *, integer=False, high=1e6):
        require(type(value) in ((int,) if integer else (int, float))
                and 0 <= value <= high and math.isfinite(value), label)
        return value

    def bounded_rows(value, label, maximum):
        require(isinstance(value, list) and len(value) <= maximum
                and all(isinstance(row, dict) for row in value), label)
        return value

    require(type(audit.get("schema_version")) is int and audit["schema_version"] == 1
            and audit.get("profile") == "shared-world-feasibility-v1"
            and audit.get("physical_validation") == "unverified", "unsupported audit contract")
    scene_sha = canonical_hash(scene)
    require(scene.get("profile") in ("shared-world-v1", "shared-world-v2", "shared-world-v3", "shared-world-v4")
            and audit.get("scene_sha256") == scene_sha
            and source.result.get("scene_sha256", scene_sha) == scene_sha, "scene identity differs")
    human, binding, task = scene.get("human_spec"), scene.get("source_binding"), scene.get("task")
    require(all(isinstance(value, dict) for value in (human, binding, task)), "scene metadata missing")
    bound_sha = binding.get("source_sha256")
    if binding.get("mode") == "analytic_fixture":
        require(bound_sha is None and human.get("source_sha256") is None and source_sha256 is None,
                "analytic audit contradicts source evidence")
    else:
        require(binding.get("mode") == "source_qualified" and isinstance(bound_sha, str)
                and _SHA.fullmatch(bound_sha) and human.get("source_sha256") in (None, bound_sha)
                and source_sha256 in (None, bound_sha), "source identities differ")
    limits = task.get("human_limits")
    require(isinstance(limits, dict), "saved task limits missing")
    limit = number(limits.get("max_floor_penetration_model_units"), "floor limit")
    # Audit v1 used this protected gate; importing its producer would load physics.
    require(limit == 0.025, "audit v1 floor limit differs from its protected gate")
    references = bounded_rows(human.get("reference"), "source reference bound", 900)
    require(len(references) >= 2, "source reference interval missing")
    times = [number(row.get("time_s"), "reference time", high=3600) for row in references]
    require(all(left < right for left, right in zip(times, times[1:]))
            and 0 < times[-1] - times[0] <= 60, "source reference order or duration")
    grid_count = math.floor((times[-1] - times[0]) * 30 + 1e-9) + 1
    require(grid_count <= 1801, "control-grid bound")
    # The audit covers the full reference, independently of any shorter task window.
    schedule = [("source_pose", stamp, index) for index, stamp in enumerate(times)]
    schedule.extend(("control_grid", min(times[-1], times[0] + index / 30), None)
                    for index in range(grid_count))
    rows = bounded_rows(audit.get("checks"), "check row bound", 2701)
    require(len(rows) == len(schedule), "incomplete source/control-grid checks")
    initial = bounded_rows(audit.get("initial_intersections"), "initial intersection bound", 4096)
    counts = {kind: {"checks": 0, "floor_failures": 0, "failed_preflight": 0}
              for kind in ("source_pose", "control_grid")}
    peak, static_poses, examples = 0., 0, []
    for row, (kind, stamp, index) in zip(rows, schedule, strict=True):
        _check(source.deadline)
        actual_index = row.get("source_reference_index")
        require(row.get("kind") == kind
                and (actual_index is None if index is None else type(actual_index) is int and actual_index == index)
                and "source_reference_index" in row
                and math.isclose(number(row.get("time_s"), "check time", high=3600), stamp,
                                 rel_tol=0, abs_tol=1e-9), "source/control-grid schedule differs")
        penetration = number(row.get("max_penetration_model_units"), "penetration")
        contacts = bounded_rows(row.get("floor_contacts"), "floor contact bound", 4096)
        conflicts = bounded_rows(row.get("static_conflicts"), "static conflict bound", 4096)
        require(number(row.get("floor_contact_count"), "floor contact count", integer=True) == len(contacts)
                and number(row.get("static_conflict_count"), "static conflict count", integer=True) == len(conflicts),
                "contact counts differ")
        require(penetration == max((number(contact.get("penetration_model_units"), "contact penetration")
                                   for contact in contacts), default=0.), "contact peak differs")
        floor_pass = penetration <= limit
        coupled_pass = floor_pass and not conflicts
        require(row.get("passes_floor_limit") is floor_pass
                and row.get("passes_coupled_preflight") is coupled_pass, "saved pass flags contradict measurements")
        counts[kind]["checks"] += 1
        counts[kind]["floor_failures"] += not floor_pass
        counts[kind]["failed_preflight"] += not coupled_pass
        peak = max(peak, penetration)
        static_poses += bool(conflicts)
        if not coupled_pass and len(examples) < 8:
            examples.append({"kind": kind, "time_s": row["time_s"], "source_reference_index": index,
                "max_penetration_model_units": penetration, "passes_floor_limit": floor_pass,
                "static_conflict_count": len(conflicts)})
    expected = {"source_pose_checks": counts["source_pose"]["checks"],
        "control_grid_checks": counts["control_grid"]["checks"], "checks": len(rows),
        "failed_source_poses": counts["source_pose"]["failed_preflight"],
        "failed_control_grid_poses": counts["control_grid"]["failed_preflight"],
        "static_conflict_poses": static_poses, "initial_cross_entity_intersections": len(initial), "physics_steps": 0}
    saved = audit.get("summary")
    require(isinstance(saved, dict), "count summary missing")
    require(all(number(saved.get(key), key, integer=True) == value for key, value in expected.items())
            and number(saved.get("max_penetration_model_units"), "summary penetration") == peak,
            "summary contradicts check rows")
    failed = sum(row["failed_preflight"] for row in counts.values())
    status = "infeasible" if failed or initial else "feasible"
    require(audit.get("status") == status, "status contradicts check rows or initial intersections")
    return {"status": status, "physical_validation": "unverified",
        "scope": "Saved zero-physics geometry preflight only; not dynamic stability, balance, task success or physical accuracy.",
        "audit_artifact": name, "audit_sha256": source.files[name], "scene_sha256": scene_sha,
        "source_sha256": bound_sha, "source_binding_mode": binding["mode"],
        "floor_limit_model_units": limit, "max_penetration_model_units": peak, **counts,
        "static_conflict_poses": static_poses, "initial_cross_entity_intersections": len(initial),
        "physics_steps": 0, "failure_examples": examples, "failure_examples_omitted": failed - len(examples)}


def _static_support_summary(source, settings, source_sha256, body_prefix):
    declared = source.document("support-priors.json", required=True)
    fields = {"schema_version", "profile", "source_sha256", "scale", "objects", "support_requirements", "root_assistance"}
    if (set(declared) != fields or type(declared["schema_version"]) is not int or declared["schema_version"] != 1
            or declared["profile"] != "shared-static-support-priors-v1"
            or not isinstance(declared["source_sha256"], str) or not _SHA.fullmatch(declared["source_sha256"])
            or source_sha256 not in (None, declared["source_sha256"])
            or settings.get("support_priors_sha256") != source.files["support-priors.json"]
            or type(declared["root_assistance"]) is not bool
            or settings.get("root_assistance") is not declared["root_assistance"]
            or settings.get("body_scale") != declared["scale"] or settings.get("full_reference") is not True):
        raise ValueError("Static-support declarations differ from saved source/settings identities")
    _number(declared["scale"], "static-support scale", .5, 2.)
    objects, requirements = declared["objects"], declared["support_requirements"]
    if (not isinstance(objects, list) or not 1 <= len(objects) <= 12
            or not isinstance(requirements, list) or not 1 <= len(requirements) <= 24
            or any(not isinstance(row, dict) for row in requirements)):
        raise ValueError("Invalid or oversized saved static-support assumptions")
    assumptions, identifiers = [], set()
    for obj in objects:
        if (not isinstance(obj, dict) or not isinstance(obj.get("id"), str) or obj["id"] in identifiers
                or obj.get("role") != "static" or not isinstance(obj.get("provenance"), dict)):
            raise ValueError("Saved support objects must be distinct authored static shapes")
        _text(obj["id"], 64)
        identifiers.add(obj["id"])
        provenance = obj["provenance"]
        if (provenance.get("mode") != "authored_addition" or provenance.get("calibration_status") != "estimated_prior"
                or provenance.get("physical_validation") != "unverified" or provenance.get("source_contact_verified") is not False):
            raise ValueError("Saved static supports cannot claim recovered or verified contact")
        assumptions.extend(_strings(provenance.get("assumptions", [])))
    if (any(not isinstance(row.get("object_id"), str) or row["object_id"] not in identifiers for row in requirements)
            or settings.get("evidence_run") is not None or settings.get("config") is not None
            or any(name in source.files for name in ("person-inventory.json", "candidate/inventory.json", "proposal/report.json"))):
        raise ValueError("Static-support scope contradicts equipment or person-only declarations")
    resolved = source.document("support-config.json")
    body_manifest_status = "not_resolved"
    if "support-config.json" in source.files:
        expected = {**declared, "profile": "shared-static-support-construction-v1",
                    "body_source_manifest_sha256": resolved.get("body_source_manifest_sha256")}
        pin = expected["body_source_manifest_sha256"]
        if resolved != expected or not isinstance(pin, str) or not _SHA.fullmatch(pin):
            raise ValueError("Resolved support config differs from the declared priors or body identity")
        body_manifest_status = "pin_only_no_sealed_body_manifest_copy"
        manifest_name = "candidate/inputs/body-source-manifest.json"
        if manifest_name in source.files:
            if source.files[manifest_name] != pin:
                raise ValueError("Resolved support body-manifest identity differs from its sealed copy")
            body_manifest_status = "matches_copied_body_manifest"
    scenes = [name for name in ("candidate/scene.json", "lab/run/scene.json", "learning_workflow/lab/run/scene.json")
              if name in source.files]
    for name in scenes:
        scene = source.document(name, required=True)
        task, binding = scene.get("task"), scene.get("source_binding")
        if (not resolved or scene.get("profile") != "shared-world-v4" or scene.get("objects") != objects
                or scene.get("controlled_id") != "human" or scene.get("target", "missing") is not None or "inventory" in scene
                or not isinstance(task, dict) or task.get("type") != "human_support_v1" or task.get("object_goal", "missing") is not None
                or not isinstance(binding, dict) or binding.get("mode") != "source_qualified"
                or binding.get("source_sha256") != declared["source_sha256"]):
            raise ValueError("Saved scene contradicts the authored static-support scope")
        human_name = body_prefix + "/human.json" if body_prefix else None
        if "human_provenance" in scene:
            raise ValueError("Compact human provenance is unsupported for static-support scenes")
        if human_name in source.files and scene.get("human_spec") != source.document(human_name):
            raise ValueError("Saved static-support scene differs from its preserved body")
    task_name = "candidate/exercise-task.json"
    if task_name in source.files:
        task = source.document(task_name)
        if (task.get("profile") != "shared-support-task-v1" or task.get("support_requirements") != requirements
                or task.get("control") != {"root_assistance": "explicit_3_axis" if declared["root_assistance"] else "disabled", "object_forces": "forbidden"}):
            raise ValueError("Saved static-support task differs from its authored requirements")
    summary = {"status": "authored_static_supports" if scenes else "declared_static_supports",
        "declared_priors": _small_fields(declared, tuple(fields)), "declared_priors_sha256": source.files["support-priors.json"],
        "source_identity": "matches_preserved_source" if source_sha256 else "declared_pin_only",
        "resolved_config": _small_fields(resolved, tuple(fields | {"body_source_manifest_sha256"})),
        "resolved_config_sha256": source.files.get("support-config.json"), "body_manifest_identity": body_manifest_status,
        "saved_scene_consistency": "metadata_only_checked" if scenes else "no_constructed_scene_available",
        "scope": "Authored static supports and load requirements; not recovered equipment, ground/contact truth or verified equipment absence."}
    return summary, assumptions


def _learning_summary(source, settings):
    workflow = source.document("learning_workflow/result.json")
    summary = _small_fields(workflow, ("profile", "status", "training", "selection", "evaluation", "scene_sha256", "task_sha256"))
    if settings.get("scene_scope") != "human_static_supports" or not workflow:
        return summary
    if workflow.get("profile") != "shared-support-learning-workflow-v1":
        raise ValueError("Saved learning workflow differs from the static-support task")
    training, selection = workflow.get("training"), workflow.get("selection")
    if training is not None:
        if (not isinstance(training, dict) or not isinstance(training.get("selected_feedback_kind"), str)
                or training["selected_feedback_kind"] not in {"authored_unchanged", "searched_joint_residuals"}
                or not isinstance(training.get("policy_sha256"), str) or not _SHA.fullmatch(training["policy_sha256"])):
            raise ValueError("Saved support training is missing its selected feedback identity")
        summary["training_feedback"] = ("Training selected unchanged programmed feedback, not new learned corrections."
            if training["selected_feedback_kind"] == "authored_unchanged" else
            "Training selected small phase-dependent joint corrections on top of programmed feedback; permitted root assistance remains programmed, not learned.")
    if selection is not None:
        if (not isinstance(selection, dict) or selection.get("selection_status") != "not_applicable"
                or selection.get("performed") is not False or selection.get("evaluation_used") is not False
                or not training or selection.get("policy_sha256") != training.get("policy_sha256")):
            raise ValueError("Saved support workflow cannot imply a secondary feasibility selection")
        summary["selection_scope"] = "Secondary training selection: not applicable and not performed. The optimizer's frozen training policy goes directly to independent changed-start evaluation; no feasibility-selection pass is claimed."
    return summary


def _summary(source, body_prefix, video):
    settings = {} if source.receipt_only else source.document("settings.json")
    stages, history = _history(source.result, settings, source.receipt_only)
    if not source.receipt_only and "stages.json" in source.files:
        if source.document("stages.json").get("stages") != source.result["stages"]:
            raise ValueError("Saved build stage histories differ")
    result = {"schema_version": 1, "profile": PROFILE, "status": "report_generated", "created_at": local_time(),
        "source_manifest_sha256": _sha(source.manifest_raw), "source_seal_scope": source.manifest["seal_scope"],
        "build_status": source.result["status"], "saved_lab_status": source.result["lab_status"],
        "runtime_admission": "not_checked", "task_success": "not_assessed_by_report", "physical_validation": "unverified",
        "inspection_scope": "Selected manifest members only; this report does not independently verify the full source package.",
        "source_mutation_check": "Manifest and every consumed/copied member rehashed before report sealing.",
        "unsealed_stage_outputs": "not_read_or_exposed" if source.receipt_only else "not_read",
        "stages": stages, "stage_history": history, "reasons": _reasons(source.result),
        "source": {}, "coverage": {"status": "not_available"}, "body": {}, "equipment": {}, "learning": {}, "assumptions": [],
        "media_sync_verified": False, "video_artifact": video,
        "work": {"video_decodes": 0, "inference_calls": 0, "physics_steps": 0, "training_episodes": 0, "api_calls": 0}}
    if source.receipt_only:
        result["reference_equipment"] = _reference_equipment(source)
        result["reference_feasibility"] = _reference_feasibility(source)
        result["diagnostics"] = _pose_diagnostics(source, body_prefix, result["coverage"], stages)
        result["assumptions"] = ["Only the terminal failure receipt is sealed. Other preserved stage outputs are unverified and are not exposed."]
        return result
    for name in ("scene_sha256", "task_sha256"):
        value = source.result.get(name)
        if value is not None:
            if not isinstance(value, str) or not _SHA.fullmatch(value):
                raise ValueError("Invalid sealed build identity: " + name)
            result[name] = value
    if video:
        doc_name = str(PurePosixPath(video).parent / "source.json")
        doc = source.document(doc_name, required=True)
        if doc.get("sha256") != source.files[video] or doc.get("artifact") not in (None, PurePosixPath(video).name):
            raise ValueError("Original video identity differs from saved provenance")
        result["source"] = _small_fields(doc, ("sha256", "title", "author", "license", "license_url", "source_page", "url", "changes", "derived_overlays"))
    elif "pose_proposals/source.json" in source.files:
        doc = source.document("pose_proposals/source.json")
        if not isinstance(doc.get("sha256"), str) or not _SHA.fullmatch(doc["sha256"]):
            raise ValueError("Invalid pinned video identity in pose proposal provenance")
        result["source"] = _small_fields(doc, ("sha256", "title", "author", "license", "license_url", "source_page", "url", "changes", "derived_overlays"))
    if body_prefix:
        evidence_name, root_name = body_prefix + "/evidence.json", body_prefix + "/root-motion.json"
        if evidence_name in source.files:
            evidence = source.document(evidence_name)
            if evidence.get("profile") == "exercise-pose-v1":
                root = source.document(root_name) if root_name in source.files else None
                if root is not None and root.get("profile") != "image-pelvis-root-reference-v1":
                    root = None
                result["coverage"] = summarize_coverage(evidence, root)
                if result["source"] and result["coverage"]["source_sha256"] != result["source"]["sha256"]:
                    raise ValueError("Pose coverage and original video identities differ")
        marker = source.document(body_prefix + "/body-source.json")
        result["body"] = _small_fields(marker, ("profile", "status", "simulation_ready", "limits", "construction"))
        if marker.get("source_sha256") is not None and result["source"] and marker["source_sha256"] != result["source"]["sha256"]:
            raise ValueError("Body construction and original video identities differ")
        result["assumptions"].extend(_strings(marker.get("limits", [])))
    result["diagnostics"] = _pose_diagnostics(source, body_prefix, result["coverage"], stages)
    proposal = source.document("proposal/report.json")
    result["equipment"] = _small_fields(proposal, ("status", "promoted", "initial_geometry_only", "full_reference_clearance_checked",
                                                  "physical_validation", "resolved_config", "source_binding"))
    result["assumptions"].extend(_strings(proposal.get("limitations", [])))
    geometry_name = next((name for name in ("proposal/geometry-candidate.rejected.json", "proposal/geometry-proposal.json") if name in source.files), None)
    if geometry_name:
        geometry = source.document(geometry_name)
        result["equipment"]["candidate"] = _candidate_summary(geometry, geometry_name)
        result["assumptions"].extend(_strings(geometry.get("assumptions", [])))
    result["assumptions"] = list(dict.fromkeys(result["assumptions"]))
    if settings.get("scene_scope") == "human_only":
        from weft.person_inventory import validate_person_inventory

        if proposal or geometry_name or settings.get("evidence_run") is not None or settings.get("config") is not None:
            raise ValueError("Person-only build contains contradictory equipment inputs or proposals")
        inventory = source.document("person-inventory.json", required=True)
        validate_person_inventory(inventory, source_sha256=result["source"].get("sha256", inventory.get("source_sha256")))
        if settings.get("inventory") != inventory:
            raise ValueError("Person-only inventory differs from saved build settings")
        if "candidate/inventory.json" in source.files and source.document("candidate/inventory.json") != inventory:
            raise ValueError("Person-only candidate inventory differs from the declared build scope")
        scenes = [name for name in ("candidate/scene.json", "lab/run/scene.json", "learning_workflow/lab/run/scene.json")
                  if name in source.files]
        for name in scenes:
            scene = source.document(name, required=True)
            task, binding = scene.get("task"), scene.get("source_binding")
            if (scene.get("profile") != "shared-world-v3" or scene.get("objects") != []
                    or scene.get("controlled_id") != "human" or scene.get("target", "missing") is not None
                    or scene.get("inventory") != inventory or not isinstance(task, dict)
                    or task.get("type") != "human_motion_v1" or task.get("object_goal", "missing") is not None
                    or not isinstance(binding, dict) or binding.get("source_sha256") != inventory["source_sha256"]):
                raise ValueError("Saved scene contradicts the person-only inventory or source identity")
        result["equipment"] = {"status": "not_applicable" if scenes else "declared_scope_only", "inventory": inventory,
            "saved_scene_consistency": "metadata_only_checked" if scenes else "no_constructed_scene_available",
            "scope": "No task-relevant equipment or obstacles declared; absence is not detector-verified."}
        result["assumptions"].append("Person-only scope is user-declared. Ground, body scale and depth remain authored or monocular priors.")
    elif settings.get("scene_scope") == "human_static_supports":
        result["equipment"], support_assumptions = _static_support_summary(source, settings,
            result["source"].get("sha256", result["coverage"].get("source_sha256")), body_prefix)
        result["assumptions"].extend(support_assumptions)
        result["assumptions"].append("Support shapes, placement and load windows are authored simulator assumptions. The original body and ground are not corrected by this report; physical contact and equipment absence remain unverified.")
        result["assumptions"] = list(dict.fromkeys(result["assumptions"]))
    elif settings.get("scene_scope") == "saved_conditional_geometry":
        result["equipment"] = _saved_geometry_summary(source, settings)
        result["assumptions"].append(result["equipment"]["scope"])
    result["requested_options"] = _small_fields(settings, ("scene_scope", "pose_recovery", "body_scale", "body_scale_scope", "root_assistance", "full_reference", "training"))
    result["learning"] = _learning_summary(source, settings)
    result["reference_feasibility"] = _reference_feasibility(source,
        result["source"].get("sha256", result["coverage"].get("source_sha256")))
    result["reference_equipment"] = _reference_equipment(source,
        result["source"].get("sha256", result["coverage"].get("source_sha256")))
    return result


def _saved_geometry_summary(source, settings):
    if "geometry_source/resume.json" not in source.files:
        # A progress callback or partial writer can fail before the snapshot's final receipt.
        if (source.result.get("status") != "failed" or source.result.get("lab_status") != "not_ready"
                or settings.get("training") is not False
                or any(row.get("stage") == "geometry_source" and row.get("status") == "ready"
                       for row in source.result.get("stages", []))
                or any(name.startswith(("lab/", "proposal/", "candidate/")) for name in source.files)):
            raise ValueError("Ready geometry build lacks its selected metadata snapshot")
        return {"status": "saved_geometry_snapshot_incomplete",
            "study_manifest_sha256": settings.get("geometry_manifest_sha256"),
            "scope": "Saved geometry reuse stopped before its selected metadata snapshot completed; no lab is ready. "
                     "Any retained source-fit or clearance values were not newly measured."}
    receipt = source.document("geometry_source/resume.json", required=True)
    if (receipt.get("profile") != "exercise-geometry-resume-inputs-v1"
            or receipt.get("status") != "selected_metadata_verified" or source.result.get("geometry_source") != receipt
            or settings.get("geometry_manifest_sha256") != receipt.get("study_manifest_sha256")
            or settings.get("training") is not False or "proposal/report.json" in source.files):
        raise ValueError("Saved geometry build receipt or scope differs")
    for name, key in (("study-manifest.json", "study_manifest_sha256"), ("candidate-manifest.json", "candidate_manifest_sha256")):
        if source.files.get("geometry_source/" + name) != receipt.get(key):
            raise ValueError("Saved geometry package identity differs")
    names = {"baseline_scene_sha256": "geometry-input/scene.json", "candidate_scene_sha256": "geometry/scene.json",
             "baseline_task_sha256": "geometry-input/exercise-task.json", "task_sha256": "geometry/exercise-task.json"}
    documents = {}
    for key, name in names.items():
        documents[key] = source.document("geometry_source/" + name, required=True)
        if canonical_hash(documents[key]) != receipt.get(key):
            raise ValueError("Retained geometry scene/task identity differs")
    if documents["task_sha256"] != {**documents["baseline_task_sha256"], "scene_sha256": receipt["candidate_scene_sha256"]}:
        raise ValueError("Retained geometry task changed more than its scene identity")
    if source.result["lab_status"] == "ready":
        for name in ("scene.json", "exercise-task.json"):
            if source.files.get("lab/run/" + name) != source.files.get("geometry_source/geometry/" + name):
                raise ValueError("Saved lab differs from the selected geometry scene/task bytes")
        if source.files.get("lab/run/source-run/manifest.json") != receipt.get("candidate_manifest_sha256"):
            raise ValueError("Saved lab source differs from the selected geometry package")
    return {**_small_fields(receipt, ("study_manifest_sha256", "candidate_manifest_sha256", "baseline_scene_sha256",
        "candidate_scene_sha256", "baseline_task_sha256", "task_sha256", "source_sha256", "selected_objects", "preserved",
        "retained_source_fit", "source_fit", "saved_clearance_status", "physical_validation", "scope")),
        "status": "saved_conditional_alternative_reused", "verification_scope": "Selected saved metadata only; no new source-fit or clearance measurement"}


def _strings(values):
    if not isinstance(values, list) or len(values) > 64:
        raise ValueError("Saved assumptions exceed their list bound")
    return [_text(value) for value in values]


def _media(raw, suffix):
    valid = {".png": raw.startswith(b"\x89PNG\r\n\x1a\n"), ".jpg": raw.startswith(b"\xff\xd8\xff"),
             ".gif": raw.startswith((b"GIF87a", b"GIF89a")), ".webm": raw.startswith(b"\x1aE\xdf\xa3"),
             ".mkv": raw.startswith(b"\x1aE\xdf\xa3"), ".avi": raw.startswith(b"RIFF") and raw[8:12] == b"AVI "}
    if suffix in (".mp4", ".mov", ".m4v"):
        valid[suffix] = raw[4:8] == b"ftyp"
    if suffix in valid and not valid[suffix]:
        raise ValueError("Saved inspection media has an unsupported file signature")


def _scorecard_html(scorecard, artifacts, manifest_sha):
    def text(value):
        return escape(str(value), quote=True)

    by_name = {row["source_name"]: row for row in artifacts}

    def evidence_link(pointer):
        name = pointer["source_name"]
        if name == "manifest.json" and pointer["sha256"] == manifest_sha and pointer["path"] is None:
            path = "source-manifest.json.txt"
        else:
            artifact = by_name.get(name)
            if artifact is None or any(pointer[key] != artifact[key] for key in ("path", "sha256")):
                raise ValueError("Scorecard evidence does not match an exact copied artifact")
            path = artifact["path"]
        return f'<a href="{text(path)}">{text(name)}</a>'

    def metric_value(metric):
        if metric["value"] is None:
            return "Unavailable"
        if metric["denominator"] is not None:
            value = f'{metric["numerator"]} / {metric["denominator"]}'
        else:
            value = format(metric["value"], ".8g")
        return value + " " + metric["unit"].replace("_", " ")

    status_labels = {"unavailable": "Not measured here", "evidence_available": "Evidence available — not a pass",
                     "failure_reported": "Failures recorded"}
    rows = []
    for row in scorecard["rows"]:
        available = [metric for metric in row["metrics"] if metric["value"] is not None]
        highlights = "<br>".join(text(metric["label"]) + ": <strong>" + text(metric_value(metric)) + "</strong>"
                                  for metric in available[:3]) or "No bound measurement is available."
        metrics = []
        for metric in row["metrics"]:
            gate = metric["recorded_gate"]
            gate_text = (" Recorded limit: " + json.dumps(gate, sort_keys=True, allow_nan=False)) if gate else ""
            links = " · ".join(evidence_link(pointer) for pointer in metric["evidence"])
            metrics.append('<tr><th scope="row">' + text(metric["label"]) + '</th><td>'
                + text(metric_value(metric)) + text(gate_text) + '</td><td>'
                + text(metric["evidence_kind"].replace("_", " ")) + '<br>'
                + text(metric["partition"].replace("_", " ")) + '</td><td>' + (links or "Unavailable") + '</td></tr>')
        limits = '<ul>' + ''.join('<li>' + text(item) + '</li>' for item in row["uncertainty_and_missing_measurements"]) + '</ul>'
        examples = ""
        if row["failure_examples"]:
            examples = '<h4>Saved failure examples</h4><pre>' + text(json.dumps(row["failure_examples"], indent=2,
                sort_keys=True, allow_nan=False)) + '</pre>'
        if row["failure_examples_omitted"]:
            examples += '<p>' + str(row["failure_examples_omitted"]) + ' additional examples omitted; counts above retain them.</p>'
        rows.append(f'<tr id="evidence-{text(row["id"])}"><th scope="row">{text(row["label"])}</th><td>'
            + '<strong>' + text(status_labels[row["status"]]) + '</strong><p>' + highlights + '</p>'
            + '<details><summary>Measurements, evidence and limits</summary>' + limits
            + '<table><thead><tr><th>Measurement</th><th>Value</th><th>Evidence kind / partition</th><th>Exact evidence</th>'
            + '</tr></thead><tbody>' + ''.join(metrics) + '</tbody></table>' + examples + '</details></td></tr>')
    return ('<section id="evidence"><h2>5. Evidence and remaining gaps</h2>'
        '<p>All 14 evaluation categories are shown below. Available evidence is not a passed category or a completion percentage. '
        'Unavailable means not measured here, not zero error. Simulator outcomes do not establish physical accuracy.</p>'
        '<p>Construction evidence describes how the world was made. Exposed development evidence has already been examined; '
        'it is not fresh final evaluation. Unknown means the saved summary does not establish that partition.</p>'
        '<table><thead><tr><th>Category</th><th>What this saved build establishes</th></tr></thead><tbody>'
        + ''.join(rows) + '</tbody></table><p>' + text(scorecard["identity_scope"]) + '</p></section>')


def _render(summary, artifacts, palette):
    def text(value):
        return escape(str(value), quote=True)

    def dump(value):
        return "<pre>" + text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)) + "</pre>"

    def link(row):
        return f'<a href="{row["path"]}">{text(row["label"])}</a>'

    by_name = {row["source_name"]: row for row in artifacts}
    rows = "".join(f'<tr><th scope="row">{text(row["label"])}</th><td>{text(row["state"].replace("_", " "))}</td>'
                   f'<td>{text(round(row["elapsed_seconds"], 3) if row["elapsed_seconds"] is not None else "—")}</td></tr>'
                   for row in summary["stages"])
    video = by_name.get(summary["video_artifact"])
    media = (f'<video controls preload="metadata" src="{video["path"]}"></video>' if video else
             "<p>No sealed original video was retained in the inspected portion of this build.</p>")
    previews = "".join(f'<figure><img loading="lazy" src="{row["path"]}" alt="{text(row["label"])}"><figcaption>{text(row["label"])}</figcaption></figure>'
                       for row in artifacts if PurePosixPath(row["source_name"]).name in _VISUALS)
    groups = []
    for group in ("Build", "Original", "Body", "Equipment", "World", "Learning"):
        items = [row for row in artifacts if row["group"] == group]
        if items:
            groups.append(f'<details><summary>{group} artifacts ({len(items)})</summary><ul>' + "".join(
                f'<li>{link(row)} <small>{text(row["source_name"])} · {row["bytes"]:,} bytes · SHA256 <code>{row["sha256"]}</code></small></li>' for row in items) + "</ul></details>")
    ready = summary["saved_lab_status"] == "ready"
    availability = ("The saved receipt reports a prepared lab. This offline report does not load or re-admit it; no simulation controls run here." if ready else
                    "No interactive lab was produced. The stopped stage must be resolved before a simulation can be made available.")
    receipt = ("<p class=notice>Only the failure receipt is sealed. Other stage outputs were not read or copied; their validity is unknown.</p>"
               if summary["source_seal_scope"] == "failure_receipt_only" else "")
    overlap = (summary["build_status"] == "rejected_initial_geometry" and any(
        "geometries" in row.get("message", "") and "intersect" in row.get("message", "") for row in summary["reasons"]))
    label = {"ready": "Build completed", "failed": "Build stopped before completion",
             "unsupported_input": "Stopped: input is not supported by this workflow",
             "rejected_initial_geometry": "Stopped: initial geometry was rejected"}.get(summary["build_status"], "Build stopped")
    if overlap:
        label = "Stopped: initial geometry overlaps"
        reasons = "<p>Two proposed collision shapes overlap at the starting pose. The saved geometry check rejected this candidate.</p>"
    elif summary["reasons"]:
        reasons = "<ul>" + "".join("<li>" + text(row.get("message", "A stage reported a failure; see its exact saved reason below.")) + "</li>" for row in summary["reasons"]) + "</ul>"
    else:
        reasons = "<p>No build-stage failure is recorded. This does not establish motion feasibility or task success.</p>"
    reasons += '<details><summary>Exact recorded reason and identifiers</summary>' + dump({"build_status": summary["build_status"], "reasons": summary["reasons"]}) + '</details>'
    feasibility = summary["reference_feasibility"]
    if feasibility["status"] == "unavailable":
        feasibility_html = ('<aside class="notice"><h3>Motion geometry audit unavailable</h3><p>'
                            + text(feasibility["reason"]) + '</p></aside>')
    else:
        floor_failed = any(feasibility[kind]["floor_failures"] for kind in ("source_pose", "control_grid"))
        if floor_failed:
            heading = "Estimated motion intersects the assumed ground"
        elif feasibility["status"] == "infeasible":
            heading = "Saved geometry overlaps remain"
        else:
            heading = "Saved floor/static and initial-placement checks passed"
        if summary["build_status"] == "ready" and feasibility["status"] == "infeasible":
            label = "Sandbox prepared; motion geometry checks failed"
        check_rows = ''.join('<tr><th scope="row">' + title + '</th><td>' + str(feasibility[kind]["checks"])
            + '</td><td>' + str(feasibility[kind]["floor_failures"]) + '</td><td>'
            + str(feasibility[kind]["failed_preflight"]) + '</td></tr>'
            for kind, title in (("source_pose", "Source-pose estimates"), ("control_grid", "30 Hz control grid")))
        audit_link = by_name[feasibility["audit_artifact"]]
        feasibility_html = (f'<aside class="notice"><h3>{heading}</h3>'
            '<p>This saved preflight checks estimated body geometry against the assumed ground and static objects. '
            'It covers the complete saved reference, even when the task uses a shorter interval. '
            'Dynamic equipment is checked only at initial placement, not throughout the reference.</p>'
            '<table><thead><tr><th>Check set</th><th>Poses checked</th><th>Over floor limit</th>'
            '<th>Failed any geometry check</th></tr></thead><tbody>' + check_rows + '</tbody></table>'
            f'<p>Maximum ground penetration: <strong>{feasibility["max_penetration_model_units"]:.8g} model units</strong>; '
            f'unchanged task limit: <strong>{feasibility["floor_limit_model_units"]:.8g} model units</strong>. '
            'Model units are simulation coordinates, not measured physical distances.</p>'
            f'<p>Poses with static-object conflicts: {feasibility["static_conflict_poses"]}. '
            f'Initial cross-entity overlaps: {feasibility["initial_cross_entity_intersections"]}.</p>'
            '<p>No physics steps were taken by this saved check. It does not establish dynamic stability, balance, '
            'task success, or physical accuracy. Ground placement and body geometry remain assumptions; '
            'these results are not an anatomical diagnosis. Lab availability is unchanged.</p><p>'
            + link(audit_link) + '</p><details><summary>Saved geometry counts, identities and failure examples</summary>'
            + dump(feasibility) + '</details></aside>')
    reference_equipment = summary.get("reference_equipment", {"status": "unavailable",
        "reason": "No saved reference-equipment inspection is available."})
    if reference_equipment["status"] == "unavailable":
        equipment_overlap_html = '<aside class="notice"><h3>Reference-equipment overlap unavailable</h3><p>' + text(reference_equipment["reason"]) + '</p></aside>'
    else:
        overlaps = reference_equipment["poses_with_equipment_overlap"]
        title = "Reference poses overlap the equipment" if overlaps else "No equipment overlap recorded at saved reference poses"
        if overlaps and ready:
            label = "Sandbox prepared; reference overlaps equipment"
        overlap_rows = ''.join('<tr><th scope="row">' + text(row["object_id"]) + '</th><td>'
            + str(row["poses_with_overlap"]) + '</td><td>' + format(row["max_penetration_model_units"], '.8g')
            + '</td><td>' + ("—" if row["first_overlap"] is None else
                text(row["first_overlap"]["source_frame"] if row["first_overlap"]["source_frame"] is not None else "No source frame")
                + " / " + format(row["first_overlap"]["source_time_s"], '.8g') + " s") + '</td></tr>'
            for row in reference_equipment["objects"])
        equipment_overlap_html = '<aside class="notice"><h3>' + title + '</h3><p><strong>' + str(overlaps) + ' of '
        equipment_overlap_html += str(reference_equipment["reference_poses"]) + ' saved reference poses</strong> overlap the fixed equipment before any action is applied. '
        equipment_overlap_html += 'Check body pose, camera assumptions and equipment geometry together before treating this as only a controller problem.</p>'
        equipment_overlap_html += '<table><thead><tr><th>Equipment</th><th>Overlapping poses</th><th>Largest penetration (model units)</th><th>First source frame / reference time</th></tr></thead><tbody>' + overlap_rows + '</tbody></table>'
        equipment_overlap_html += '<p>' + text(reference_equipment["scope"]) + '</p><p>' + text(reference_equipment["clock_scope"]) + '. Model units are not measured metres.</p><p>'
        equipment_overlap_html += ' · '.join(link(by_name[reference_equipment[key]]) for key in ("reference_artifact", "scene_artifact", "contract_artifact"))
        equipment_overlap_html += '</p><details><summary>Exact saved overlap counts, identities and examples</summary>' + dump(reference_equipment) + '</details></aside>'
    diagnostic = summary.get("diagnostics", {})
    diagnostic_html = ""
    if diagnostic.get("status") == "saved_quality_rejected":
        heading = "Why body construction stopped" if diagnostic["explains_body_stop"] else "Saved pose-quality failures"
        if diagnostic["explains_body_stop"]:
            label = "Stopped: source pose quality was rejected"
        checks = diagnostic["checks"]
        failed = [row for row in checks if row["status"] == "failed"]
        diagnostic_html = f'<h3>{heading}</h3><ul>' + ''.join(
            '<li><strong>' + text(row["label"]) + ':</strong> ' + text(row["message"]) + '</li>' for row in failed) + '</ul>'
        if diagnostic.get("unquantified_saved_errors"):
            diagnostic_html += '<p>Other saved failures (no supported numeric explanation is available):</p><ul>' + ''.join(
                '<li>' + text(value) + '</li>' for value in diagnostic["unquantified_saved_errors"]) + '</ul>'
        coverage_check = next((row for row in checks if row["id"] == "visibility" and row["status"] == "passed"), None)
        if coverage_check:
            diagnostic_html += '<p>Coverage was sufficient for this check: ' + text(coverage_check["message"]) + '</p>'
        spans_html = ''
        for row in failed:
            affected = row.get("affected_samples")
            if not affected:
                continue
            span_rows = ''
            for span in affected["sample_runs"]:
                frames = str(span["first_frame"]) if span["first_frame"] == span["last_frame"] else f'{span["first_frame"]}–{span["last_frame"]}'
                nominal = f'{span["nominal_start_s"]:.3f}–{span["nominal_end_s"]:.3f}'
                native = (f'{span["native_start_s"]:.3f}–{span["native_end_s"]:.3f}'
                          if span["native_start_s"] is not None else 'Unavailable; not replaced with nominal time')
                span_rows += f'<tr><td>{text(frames)}</td><td>{len(span["sampled_frames"])}</td><td>{text(nominal)}</td><td>{text(native)}</td></tr>'
            spans_html += '<h4>' + text(row["label"]) + '</h4><p>' + text(affected["scope"]) + '</p>'
            spans_html += '<table><thead><tr><th>Original frames</th><th>Affected samples</th><th>Nominal seconds</th><th>Saved decoder seconds</th></tr></thead><tbody>' + span_rows + '</tbody></table>'
            if affected["unlisted_sample_runs"]:
                spans_html += f'<p>{affected["unlisted_sample_runs"]} further sample runs are not tabulated. The exact copied evidence and report retain every affected frame number.</p>'
            spans_html += '<p>' + text(affected["clock_scope"]) + '</p>'
        evidence_link = by_name.get(diagnostic["evidence"]["source_name"])
        diagnostic_html += '<details><summary>Affected original frames and saved clocks</summary>' + (spans_html or '<p>This saved gate does not identify a local failing interval.</p>')
        if evidence_link:
            diagnostic_html += '<p>' + link(evidence_link) + '</p>'
        diagnostic_html += '</details><h4>Next steps</h4><ul>' + ''.join('<li>' + text(value) + '</li>' for value in diagnostic["next_steps"]) + '</ul>'
        diagnostic_html += '<details><summary>Diagnostic evidence and limits</summary>' + dump(diagnostic) + '</details>'
    pixel_html = ""
    pixel_support = summary.get("pixel_support")
    if pixel_support is not None:
        counts = pixel_support["status_counts"]
        labels = {"contradicted": "Motion disagrees", "not_contradicted": "No disagreement found", "unavailable": "Cannot check"}
        pixel_html = ('<aside class="notice"><h3>Compare estimated motion with images</h3>'
            '<p>These saved checks compare small image areas around predicted joints with estimated joint motion. '
            'A mismatch applies to the original image area; it does not prove that a pose is wrong. '
            'Agreement does not verify anatomy, hidden joints, or 3D motion.</p>'
            f'<p>Motion disagrees: {counts.get("contradicted", 0)}; '
            f'no disagreement found: {counts.get("not_contradicted", 0)}; '
            f'cannot check: {counts.get("unavailable", 0)}. '
            f'{pixel_support["processed_slots"]}/{pixel_support["slot_count"]} joint checks across selected image pairs completed.</p>')
        if pixel_support["status"] == "partial_deadline":
            pixel_html += '<p><strong>Partial check: the saved time limit left slots unprocessed.</strong></p>'
        slot_rows = ''.join('<tr><td>' + ' → '.join(f'<a href="#pixel-frame-{frame}">{frame}</a>' for frame in row["frames"]) + '</td><td>'
            + ' → '.join(f'<a href="#pixel-frame-{frame}">' + text('unknown' if stamp is None else stamp) + '</a>'
                         for frame, stamp in zip(row["frames"], row["native_time_s"], strict=True))
            + '</td><td>' + text(row["joint"].replace('_', ' ')) + '</td><td>'
            + labels[row["status"]] + '</td><td>' + text(row["reason"].replace('_', ' ')) + '</td></tr>'
            for row in pixel_support["rows"])
        pixel_html += ('<details><summary>All checked source pairs and joint slots</summary><table><thead><tr>'
            '<th>Source frames</th><th>Native times (s)</th><th>Joint label</th><th>Patch check</th><th>Reason</th>'
            '</tr></thead><tbody>' + slot_rows + '</tbody></table></details>')
        for row in artifacts:
            if row["source_name"].startswith("pixel_support/images/"):
                frame = int(PurePosixPath(row["source_name"]).stem.split('-')[-1])
                pixel_html += (f'<figure id="pixel-frame-{frame}"><img loading="lazy" src="{row["path"]}" alt="{text(row["label"])}">'
                               f'<figcaption>{text(row["label"])}</figcaption></figure>')
        pixel_html += ('<p>Only the supplied image pairs were checked. Other frames remain unqualified. '
            'The saved build status, original poses, confidence and quality gates are unchanged. '
            'This report reads the saved check; it does not repeat image matching.</p>'
            '<details><summary>Pixel-check provenance and limits</summary>' + dump(pixel_support) + '</details></aside>')
    source = (dump(summary["source"]) if summary["source"] else "<p>Original source attribution is not available in the inspected sealed files.</p>")
    coverage = summary["coverage"]
    if coverage.get("status") == "available":
        coverage_html = (f'<p><strong>{coverage["usable_samples"]}</strong> usable pose samples; '
            f'<strong>{coverage["unusable_samples"]}</strong> unusable sampled frames; '
            f'<strong>{coverage["unsampled_frames"]}</strong> frames not sampled. '
            f'{coverage["decoded_frames"]} decoded source slots; metadata declared {coverage["metadata_frames"]}.</p>'
            + '<details><summary>Exact coverage and saved gaps</summary>' + dump(coverage) + '</details>')
    else:
        coverage_html = "<p>Original pose coverage is not available in the inspected sealed files.</p>"
    assumptions = "".join("<li>" + text(value) + "</li>" for value in summary["assumptions"])
    equipment_rows = summary["equipment"].get("candidate", {}).get("equipment", [])
    if not isinstance(equipment_rows, list) or len(equipment_rows) > 128 or any(not isinstance(row, dict) for row in equipment_rows):
        raise ValueError("Invalid or oversized saved equipment list")
    equipment_table = ""
    for row in equipment_rows:
        geometry = row.get("geometry", {})
        if not isinstance(geometry, dict):
            raise ValueError("Invalid saved equipment geometry")
        equipment_table += (f'<tr><th scope="row">{text(row.get("object_id", "unknown"))}</th>'
            f'<td>{text(row.get("source_frame", "unknown"))}</td>'
            f'<td>{text(geometry.get("source_bounds_px", "not recorded"))}</td>'
            f'<td>{text(geometry.get("dimensions_model_units", "not recorded"))}</td></tr>')
    equipment_table = ('<table><thead><tr><th>Object</th><th>Saved source frame</th><th>Image box (pixels)</th>'
                       '<th>Proposed size (model units)</th></tr></thead><tbody>' + equipment_table + '</tbody></table>') if equipment_table else "<p>No equipment candidate is available in the inspected sealed files.</p>"
    if summary["equipment"].get("status") == "not_applicable":
        equipment_table = "<p>Person-only scene: one selected body and ground, without dummy equipment. Equipment and object-contact gates are not applicable, not passed. Equipment absence is user-declared, not detector-verified.</p>"
    elif summary["equipment"].get("status") == "declared_scope_only":
        equipment_table = "<p>Person-only scope was requested, but no constructed scene is available in the sealed files. Equipment absence is user-declared, not detector-verified.</p>"
    elif summary["equipment"].get("status") == "saved_conditional_alternative_reused":
        equipment_table = ("<p>The saved conditional compound alternative and original box scene are retained with the unchanged task. "
            "Image-fit and clearance values are saved study claims, not newly measured shape or contact.</p><p>"
            + " · ".join(link(by_name[name]) for name in ("geometry_source/geometry-input/scene.json",
                "geometry_source/geometry/scene.json", "geometry_source/geometry/exercise-task.json") if name in by_name) + "</p>")
    elif summary["equipment"].get("status") == "saved_geometry_snapshot_incomplete":
        equipment_table = "<p>Saved geometry reuse stopped before its metadata snapshot completed. No lab is ready; available partial artifacts and the terminal failure are retained below.</p>"
    elif summary["equipment"].get("status") in {"authored_static_supports", "declared_static_supports"}:
        supports = summary["equipment"]
        support_rows = "".join(f'<tr><th scope="row">{text(obj["id"])}</th><td>{text(obj.get("shape", "not recorded"))}</td>'
            f'<td>{text(obj.get("position", "not recorded"))}</td></tr>' for obj in supports["declared_priors"]["objects"])
        equipment_table = ('<p>Authored static supports: fixed collision shapes and declared segment/load windows, not recovered equipment or contact. '
            'The original body and source remain separate from these assumptions. This is not a person-only scene or an equipment-absence assertion.</p>'
            + ('<p>No constructed support scene is available in the sealed files. The saved priors are not a playable environment.</p>'
               if supports["status"] == "declared_static_supports" else '<p>Saved scene/support metadata agrees; runtime admission and physical validity were not checked by this report.</p>')
            + '<table><thead><tr><th>Authored support</th><th>Shape</th><th>Position (model units)</th></tr></thead><tbody>'
            + support_rows + '</tbody></table><p>'
            + ' · '.join(link(by_name[name]) for name in ("support-priors.json", "support-config.json") if name in by_name) + '</p>')
    learning_explanation = "".join("<p>" + text(summary["learning"][key]) + "</p>"
                                   for key in ("training_feedback", "selection_scope") if key in summary["learning"])
    evidence_html = _scorecard_html(summary["evidence_scorecard"], artifacts, summary["source_manifest_sha256"])
    css = palette + """
body{margin:0;background:var(--weft-canvas);color:var(--weft-ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:32px 20px}h1{font:700 2.5rem/1.1 Georgia,serif}h2{margin-top:0}section{background:var(--weft-surface);border:1px solid var(--weft-border);border-radius:14px;padding:24px;margin:22px 0}a{color:var(--weft-primary)}small{display:block;color:var(--weft-muted);overflow-wrap:anywhere}code,pre{overflow-wrap:anywhere;white-space:pre-wrap}pre{font-size:.82rem;max-height:420px;overflow:auto;padding:14px;background:var(--weft-canvas)}table{width:100%;border-collapse:collapse;text-align:left}td,th{padding:10px;border-bottom:1px solid var(--weft-border);vertical-align:top}video,img{max-width:100%;max-height:560px;border-radius:8px}video{width:100%}figure{margin:20px 0}figcaption{color:var(--weft-muted)}summary{cursor:pointer;font-weight:650;padding:10px 0}li{margin:10px 0}.notice{border-left:4px solid var(--weft-primary);padding:12px 16px}.tag{font-weight:700;color:var(--weft-primary)}nav{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:600px){main{padding:20px 12px}section{padding:16px}table{font-size:.85rem}td,th{padding:6px}h1{font-size:2rem}}"""
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; media-src 'self' file:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; object-src 'none'">
<title>Weft · Build inspection</title><style>{css}</style></head><body><main>
<header><p class="tag">weft · saved build inspection</p><h1>What happened in this build</h1><p><strong>{text(label)}</strong></p><p>{availability}</p>{receipt}{feasibility_html}{equipment_overlap_html}
<nav><a href="#stages">Stages</a><a href="#original">Original video</a><a href="#body">Body evidence</a><a href="#equipment">Equipment</a><a href="#evidence">Evidence and gaps</a><a href="#artifacts">Artifacts</a></nav></header>
<section id="stages"><h2>1. Completed, stopped, and not run</h2><table><thead><tr><th>Step</th><th>State</th><th>Elapsed seconds</th></tr></thead><tbody>{rows}</tbody></table>{diagnostic_html}<h3>Saved outcome</h3>{reasons}<p>Collision-shape identifiers are shown exactly as recorded; they are not a diagnosis of anatomical contact.</p><details><summary>Recorded stage events</summary>{dump(summary["stage_history"])}</details></section>
<section id="original"><h2>2. Original video</h2>{media}<p>Playback uses the copied original. This report does not decode, synchronize, or seek video to inferred pose or equipment times.</p><details><summary>Source identity, attribution and rights</summary>{source}</details></section>
<section id="body"><h2>3. Body evidence</h2>{coverage_html}{previews}{pixel_html}<p>Pose annotations show model-reported observations, not a simulated body, recovered anatomy, or measured muscle activation. Construction-only body seeds are not playable worlds.</p><details><summary>Saved body construction scope</summary>{dump(summary["body"])}</details></section>
<section id="equipment"><h2>4. Equipment and assumptions</h2><p>Boxes, scale, camera depth, mass, and support are hypotheses. A box matching image edges does not prove physical shape or contact. Rejected candidates remain rejected.</p>{equipment_table}<details><summary>Exact saved geometry, source bindings and clocks</summary>{dump(summary["equipment"])}</details><h3>Saved limitations</h3><ul>{assumptions}</ul><details><summary>Requested options — not proof of execution</summary>{dump(summary.get("requested_options", {}))}</details>{learning_explanation}<details><summary>Saved learning and evaluation outcomes — not rerun here</summary>{dump(summary["learning"])}</details></section>
{evidence_html}
<section id="artifacts"><h2>6. Exact copied artifacts</h2><p>Each link opens a local exact-byte copy verified against the supplied source manifest. Other declared files were not independently checked or copied. No external paths or saved scripts are followed.</p>{''.join(groups)}<p><a href="source-manifest.json.txt">Original manifest: all declared paths and hashes</a> · <a href="report.json.txt">Report summary and copy provenance</a></p><p>Source manifest SHA256: <code>{summary["source_manifest_sha256"]}</code></p></section>
<footer><p>Report generated: {text(summary["created_at"])} (America/Los_Angeles). Report generation does not establish task success or physical accuracy. No simulation, inference, training, API call, or server was started by this report.</p></footer>
</main></body></html>'''.encode()


def _write(fd, name, raw, written, deadline):
    _check(deadline)
    if sum(value[1] for value in written.values()) + len(raw) > MAX_OUTPUT_BYTES - 64 * 1024:
        raise ValueError("Build inspection exceeds its output byte bound")
    parent = os.dup(fd)
    try:
        parts = PurePosixPath(_name(name)).parts
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        handle = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(handle, "wb") as stream:
            stream.write(raw)
        written[name] = (_sha(raw), len(raw))
    finally:
        os.close(parent)


def _output_inventory(fd, expected, deadline):
    found = set()
    with os.scandir(fd) as entries:
        for entry in entries:
            _check(deadline)
            info = entry.stat(follow_symlinks=False)
            if entry.name == "artifacts" and stat.S_ISDIR(info.st_mode):
                child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    with os.scandir(child) as files:
                        for item in files:
                            _check(deadline)
                            name = "artifacts/" + item.name
                            if name not in expected or not item.is_file(follow_symlinks=False):
                                raise ValueError("Unexpected or unsafe report output")
                            found.add(name)
                finally:
                    os.close(child)
            elif entry.name in expected and stat.S_ISREG(info.st_mode):
                found.add(entry.name)
            else:
                raise ValueError("Unexpected or unsafe report output")
    if found != set(expected):
        raise ValueError("Report outputs changed before sealing")


def prepare_exercise_build_report(run, out, *, max_seconds=30, pixel_support=None):
    """Copy bounded sealed evidence into a new offline report; never modify the build."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError(f"Build inspection requires a 1–{MAX_SECONDS} second bound")
    source_path, target = _absolute(run), _absolute(out)
    if target.is_relative_to(source_path) or source_path.is_relative_to(target):
        raise ValueError("Inspection output must be new and outside its source")
    started = time.monotonic()
    deadline = started + max_seconds
    source = _Source(source_path, deadline)
    output_fd, companion = None, None
    try:
        chosen, body_prefix, video = _selection(source)
        summary = _summary(source, body_prefix, video)
        if "selected_members_sha256" in summary["reference_equipment"]:
            chosen.update({name: ("World", label) for name, label in _reference_equipment_paths(source).items()})
        if pixel_support is not None:
            from weft.pose_pixel_support_package import load_pose_pixel_support

            companion_path = _absolute(pixel_support)
            if target.is_relative_to(companion_path) or companion_path.is_relative_to(target):
                raise ValueError("Inspection output must be outside its pixel-support companion")
            if source.receipt_only or body_prefix is None:
                raise ValueError("Pixel support requires sealed original body-source evidence")
            evidence_name = body_prefix + "/evidence.json"
            evidence_raw = source.raw[evidence_name] if evidence_name in source.raw else source.read(evidence_name)
            companion = load_pose_pixel_support(companion_path, evidence_document=evidence_raw,
                source_manifest_raw=source.read(body_prefix + "/manifest.json", MAX_MANIFEST_BYTES), deadline=deadline)
            summary["pixel_support"] = companion.summary
        entities = summary["equipment"].get("source_binding", {}).get("entity_bindings", [])
        if not isinstance(entities, list) or len(entities) > 128 or any(not isinstance(row, dict) for row in entities):
            raise ValueError("Invalid or oversized saved entity bindings")
        for name in chosen:
            if name.startswith("proposal/inputs/observations/"):
                sha = PurePosixPath(name).stem
                matching = [row for row in entities if row.get("observation_document_sha256") == sha]
                if matching:
                    labels = ", ".join(_text(row.get("world_entity_id"), 128) for row in matching)
                    group = "Body" if all(row.get("entity_kind") == "articulated_human" for row in matching) else "Equipment"
                    chosen[name] = (group, "Canonical source observations for " + labels)
        total = len(source.manifest_raw)
        for name in chosen:
            fd = _open_member(source.fd, name)
            try:
                size = os.fstat(fd).st_size
            finally:
                os.close(fd)
            if size > MAX_FILE_BYTES:
                raise ValueError("Selected inspection artifact exceeds its byte bound")
            total += size
        companion_members = ({"manifest.json": companion.manifest_raw,
            **{name: companion.raw[name] for name in companion.copy_names}} if companion is not None else {})
        total += sum(map(len, companion_members.values()))
        if len(chosen) + len(companion_members) > MAX_COPIED_FILES:
            raise ValueError("Inspection and pixel-support copies exceed the combined file bound")
        if total > MAX_OUTPUT_BYTES - _RESERVE:
            raise ValueError("Selected inspection artifacts exceed the complete output reservation")
        # Brand resources are trusted application bytes, not content from the inspected build.
        palette_path = Path(__file__).parent / "assets/brand.css"
        with palette_path.open("rb") as stream:
            palette_raw = stream.read(16 * 1024 + 1)
        if len(palette_raw) > 16 * 1024 or b"</" in palette_raw:
            raise ValueError("Invalid or oversized report palette")
        palette = palette_raw.decode()
        _check(deadline)
        parent = _directory(target.parent, create=True)
        try:
            os.mkdir(target.name, 0o700, dir_fd=parent)
            output_fd = os.open(target.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
        os.mkdir("artifacts", 0o700, dir_fd=output_fd)
        written, artifacts = {}, []
        _write(output_fd, "source-manifest.json.txt", source.manifest_raw, written, deadline)
        for name, (group, label) in chosen.items():
            raw = source.raw[name] if name in source.raw else source.read(name)
            suffix = PurePosixPath(name).suffix
            _media(raw, suffix)
            extension = suffix if suffix in (".png", ".jpg", ".gif", ".webm", ".mp4", ".m4v", ".mov", ".mkv", ".avi") else suffix + ".txt"
            destination = "artifacts/" + source.files[name] + extension
            if destination not in written:
                _write(output_fd, destination, raw, written, deadline)
            artifacts.append({"source_name": name, "path": destination, "sha256": source.files[name],
                              "bytes": len(raw), "group": group, "label": label})
        copied_build_files = len(artifacts)
        for name, raw in companion_members.items():
            suffix = PurePosixPath(name).suffix
            _media(raw, suffix)
            extension = ".png" if suffix == ".png" else suffix + ".txt"
            destination = "artifacts/" + _sha(raw) + extension
            if destination not in written:
                _write(output_fd, destination, raw, written, deadline)
            label = ("Original saved source pixels: " + PurePosixPath(name).stem if suffix == ".png" else
                     "Saved pixel-support evidence: " + name)
            artifacts.append({"source_name": "pixel_support/" + name, "path": destination, "sha256": _sha(raw),
                              "bytes": len(raw), "group": "Body", "label": label})
        if companion is not None:
            summary["copied_companion_files"] = len(companion_members)
        summary.update(artifacts=artifacts, source_directory=str(source_path), source_declared_files=len(source.files),
                       copied_source_files=copied_build_files, uninspected_declared_files=len(set(source.files) - set(source.pins)),
                       implementation_sha256=_sha(Path(__file__).read_bytes()), palette_sha256=_sha(palette_raw))
        summary["evidence_scorecard"] = evidence_scorecard.build_evidence_scorecard(summary)
        summary["evidence_scorecard_implementation_sha256"] = _sha(Path(evidence_scorecard.__file__).read_bytes())
        report_raw, html = _bytes(summary), _render(summary, artifacts, palette)
        if len(report_raw) > MAX_REPORT_BYTES or len(html) > MAX_REPORT_BYTES:
            raise ValueError("Build inspection summary or HTML exceeds its byte bound")
        _write(output_fd, "report.json.txt", report_raw, written, deadline)
        _write(output_fd, "index.html", html, written, deadline)
        source.recheck()
        if companion is not None:
            companion.recheck()
        output_check = _directory(target)
        try:
            if (os.fstat(output_check).st_dev, os.fstat(output_check).st_ino) != (os.fstat(output_fd).st_dev, os.fstat(output_fd).st_ino):
                raise ValueError("Report output directory changed")
        finally:
            os.close(output_check)
        _output_inventory(output_fd, written, deadline)
        for name, (expected, size) in written.items():
            if _sha(_read(output_fd, name, size, deadline)) != expected:
                raise ValueError("Copied report output changed before sealing")
        manifest = _bytes({"schema_version": 1, "profile": PROFILE, "status": "report_generated",
                           "source_manifest_sha256": _sha(source.manifest_raw),
                           "files": {name: value[0] for name, value in sorted(written.items())}})
        if len(manifest) > 64 * 1024:
            raise ValueError("Build inspection manifest exceeds its metadata bound")
        _write(output_fd, "manifest.json", manifest, written, deadline)
        return {"profile": PROFILE, "status": "report_generated", "report_path": str(target / "index.html"),
                "directory": str(target), "manifest_sha256": _sha(manifest),
                "source_manifest_sha256": _sha(source.manifest_raw), "build_status": summary["build_status"],
                "saved_lab_status": summary["saved_lab_status"], "files": len(written),
                "bytes": sum(value[1] for value in written.values()), "seconds": time.monotonic() - started}
    except Exception as error:
        if output_fd is not None:
            try:
                raw = _bytes({"profile": PROFILE, "status": "failed", "error": type(error).__name__,
                              "message": str(error)[:2000], "source_manifest_sha256": _sha(source.manifest_raw)})
                handle = os.open("failure.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=output_fd)
                with os.fdopen(handle, "wb") as stream:
                    stream.write(raw)
            except OSError:
                pass
        raise
    finally:
        os.close(source.fd)
        if companion is not None:
            companion.close()
        if output_fd is not None:
            os.close(output_fd)
