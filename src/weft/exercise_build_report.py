"""Offline inspection of saved exercise builds, including rejected construction.

Only selected manifest members are verified and copied. A report is neither
whole-package admission nor a playable world, and never runs source code.
"""

import hashlib
from html import escape
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time

from weft.exercise_coverage import summarize_coverage
from weft.io import local_time
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
    "input-preflight.json": ("Build", "Original input checks and identities"),
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
    "proposal/initial-geometry-admission.json": ("Equipment", "Saved initial-geometry check"),
    "proposal/frame-migration.json": ("Equipment", "Coordinate-frame translation checks"),
    "proposal/source-binding.json": ("Equipment", "Body/equipment source-slot bindings"),
    "proposal/source-equivalences.json": ("Equipment", "Original/derivative source mapping identities"),
    "candidate/reconstruction.json": ("World", "Saved shared-world construction result"),
    "candidate/scene.json": ("World", "Constructed scene definition — not executed here"),
    "candidate/exercise-task.json": ("World", "Frozen exercise task and gates"),
    "learning_workflow/result.json": ("Learning", "Saved learning workflow outcome"),
    "learning_workflow/plan.json": ("Learning", "Frozen training/evaluation plan"),
    "learning_workflow/selected/selection.json": ("Learning", "Training-only policy selection"),
    "learning_workflow/evaluation/evaluation.json": ("Learning", "Independent evaluation result"),
}


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
    latest = {row["stage"]: row for row in history}
    training = bool(settings.get("training")) or any(name.startswith("learning_") for name in latest)
    recovery = settings.get("pose_recovery") == "selective-crop" or any(
        name in latest for name in ("source_pose_proposals", "source_pose_recovery", "source_body_source"))
    source_steps = [
        ("source_pose_proposals", "Inspect original poses and propose image crops", True),
        ("source_pose_recovery", "Verify accepted pose replacements", True),
        ("source_body_source", "Construct the body from verified poses", True),
    ] if recovery else [("body_source", "Estimate body from video", bool(settings.get("video")) or "body_source" in latest)]
    definitions = source_steps + [
        ("body_seed", "Prepare construction-only body", bool(settings.get("human_run") or settings.get("video")) or "body_seed" in latest),
        ("equipment_priors", "Propose equipment geometry", True),
        ("shared_world", "Construct the shared world", True),
        ("learning_training", "Train a controller", training),
        ("learning_training_selection", "Select using training evidence", training),
        ("learning_independent_evaluation", "Test on independent starts", training),
        ("learning_interactive_lab" if training else "interactive_lab", "Prepare the interactive lab", True),
    ]
    stages = []
    for name, label, requested in definitions:
        row = latest.get(name)
        state = ("completed" if row["status"] in {"ready", "proposal_ready", "trained", "selected", "evaluated"}
                 else "stopped") if row else ("unknown" if receipt_only else "not_run" if requested else "not_requested")
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
    # A failed body writer can contain metadata but no poses; retain earlier verified source evidence.
    body_prefix = next((prefix for prefix in _PREFIXES if prefix + "/evidence.json" in source.files), None)
    if body_prefix is None:
        body_prefix = next((prefix for prefix in _PREFIXES if any(prefix + "/" + name in source.files for name in _BODY_DOCS)), None)
    if body_prefix:
        chosen.update({body_prefix + "/" + name: ("Body", label) for name, label in _BODY_DOCS.items()
                       if body_prefix + "/" + name in source.files})
        chosen.update({body_prefix + "/" + name: ("Body", label) for name, label in _VISUALS.items()
                       if body_prefix + "/" + name in source.files})
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


def _small_fields(value, keys):
    if not isinstance(value, dict):
        raise ValueError("Expected saved inspection metadata object")
    result = {key: value[key] for key in keys if key in value}
    if len(_bytes(result)) > 128 * 1024:
        raise ValueError("Saved inspection summary fields exceed their bound")
    return result


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
        result["assumptions"] = ["Only the terminal failure receipt is sealed. Other preserved stage outputs are unverified and are not exposed."]
        return result
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
    proposal = source.document("proposal/report.json")
    result["equipment"] = _small_fields(proposal, ("status", "promoted", "initial_geometry_only", "full_reference_clearance_checked",
                                                  "physical_validation", "resolved_config", "source_binding"))
    result["assumptions"].extend(_strings(proposal.get("limitations", [])))
    geometry_name = next((name for name in ("proposal/geometry-candidate.rejected.json", "proposal/geometry-proposal.json") if name in source.files), None)
    if geometry_name:
        geometry = source.document(geometry_name)
        result["equipment"]["candidate"] = _small_fields(geometry, ("status", "camera_hypothesis", "equipment"))
        result["assumptions"].extend(_strings(geometry.get("assumptions", [])))
    result["assumptions"] = list(dict.fromkeys(result["assumptions"]))
    result["requested_options"] = _small_fields(settings, ("pose_recovery", "body_scale", "body_scale_scope", "root_assistance", "full_reference", "training"))
    result["learning"] = _small_fields(source.document("learning_workflow/result.json"),
        ("status", "training", "selection", "evaluation", "scene_sha256", "task_sha256"))
    return result


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
        reasons = "<p>No failure reason is recorded in this terminal receipt.</p>"
    reasons += '<details><summary>Exact recorded reason and identifiers</summary>' + dump({"build_status": summary["build_status"], "reasons": summary["reasons"]}) + '</details>'
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
    css = palette + """
body{margin:0;background:var(--weft-canvas);color:var(--weft-ink);font:16px/1.55 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:32px 20px}h1{font:700 2.5rem/1.1 Georgia,serif}h2{margin-top:0}section{background:var(--weft-surface);border:1px solid var(--weft-border);border-radius:14px;padding:24px;margin:22px 0}a{color:var(--weft-primary)}small{display:block;color:var(--weft-muted);overflow-wrap:anywhere}code,pre{overflow-wrap:anywhere;white-space:pre-wrap}pre{font-size:.82rem;max-height:420px;overflow:auto;padding:14px;background:var(--weft-canvas)}table{width:100%;border-collapse:collapse;text-align:left}td,th{padding:10px;border-bottom:1px solid var(--weft-border);vertical-align:top}video,img{max-width:100%;max-height:560px;border-radius:8px}video{width:100%}figure{margin:20px 0}figcaption{color:var(--weft-muted)}summary{cursor:pointer;font-weight:650;padding:10px 0}li{margin:10px 0}.notice{border-left:4px solid var(--weft-primary);padding:12px 16px}.tag{font-weight:700;color:var(--weft-primary)}nav{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:600px){main{padding:20px 12px}section{padding:16px}table{font-size:.85rem}td,th{padding:6px}h1{font-size:2rem}}"""
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; media-src 'self' file:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; object-src 'none'">
<title>Weft · Build inspection</title><style>{css}</style></head><body><main>
<header><p class="tag">weft · saved build inspection</p><h1>What happened in this build</h1><p><strong>{text(label)}</strong></p><p>{availability}</p>{receipt}
<nav><a href="#stages">Stages</a><a href="#original">Original video</a><a href="#body">Body evidence</a><a href="#equipment">Equipment</a><a href="#artifacts">Artifacts</a></nav></header>
<section id="stages"><h2>1. Completed, stopped, and not run</h2><table><thead><tr><th>Step</th><th>State</th><th>Elapsed seconds</th></tr></thead><tbody>{rows}</tbody></table><h3>Saved outcome</h3>{reasons}<p>Collision-shape identifiers are shown exactly as recorded; they are not a diagnosis of anatomical contact.</p><details><summary>Recorded stage events</summary>{dump(summary["stage_history"])}</details></section>
<section id="original"><h2>2. Original video</h2>{media}<p>Playback uses the copied original. This report does not decode, synchronize, or seek video to inferred pose or equipment times.</p><details><summary>Source identity, attribution and rights</summary>{source}</details></section>
<section id="body"><h2>3. Body evidence</h2>{coverage_html}{previews}<p>Pose annotations show model-reported observations, not a simulated body, recovered anatomy, or measured muscle activation. Construction-only body seeds are not playable worlds.</p><details><summary>Saved body construction scope</summary>{dump(summary["body"])}</details></section>
<section id="equipment"><h2>4. Equipment and assumptions</h2><p>Boxes, scale, camera depth, mass, and support are hypotheses. A box matching image edges does not prove physical shape or contact. Rejected candidates remain rejected.</p>{equipment_table}<details><summary>Exact saved geometry, source bindings and clocks</summary>{dump(summary["equipment"])}</details><h3>Saved limitations</h3><ul>{assumptions}</ul><details><summary>Requested options — not proof of execution</summary>{dump(summary.get("requested_options", {}))}</details><details><summary>Saved learning and evaluation outcomes — not rerun here</summary>{dump(summary["learning"])}</details></section>
<section id="artifacts"><h2>5. Exact copied artifacts</h2><p>Each link opens a local exact-byte copy verified against the supplied source manifest. Other declared files were not independently checked or copied. No external paths or saved scripts are followed.</p>{''.join(groups)}<p><a href="source-manifest.json.txt">Original manifest: all declared paths and hashes</a> · <a href="report.json.txt">Report summary and copy provenance</a></p><p>Source manifest SHA256: <code>{summary["source_manifest_sha256"]}</code></p></section>
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


def prepare_exercise_build_report(run, out, *, max_seconds=30):
    """Copy bounded sealed evidence into a new offline report; never modify the build."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= MAX_SECONDS:
        raise ValueError(f"Build inspection requires a 1–{MAX_SECONDS} second bound")
    source_path, target = _absolute(run), _absolute(out)
    if target.is_relative_to(source_path) or source_path.is_relative_to(target):
        raise ValueError("Inspection output must be new and outside its source")
    started = time.monotonic()
    deadline = started + max_seconds
    source = _Source(source_path, deadline)
    output_fd = None
    try:
        chosen, body_prefix, video = _selection(source)
        summary = _summary(source, body_prefix, video)
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
        summary.update(artifacts=artifacts, source_directory=str(source_path), source_declared_files=len(source.files),
                       copied_source_files=len(artifacts), uninspected_declared_files=len(set(source.files) - set(source.pins)),
                       implementation_sha256=_sha(Path(__file__).read_bytes()), palette_sha256=_sha(palette_raw))
        report_raw, html = _bytes(summary), _render(summary, artifacts, palette)
        if len(report_raw) > MAX_REPORT_BYTES or len(html) > MAX_REPORT_BYTES:
            raise ValueError("Build inspection summary or HTML exceeds its byte bound")
        _write(output_fd, "report.json.txt", report_raw, written, deadline)
        _write(output_fd, "index.html", html, written, deadline)
        source.recheck()
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
        if output_fd is not None:
            os.close(output_fd)
