"""Reuse a sealed conditional geometry result without fitting or interpreting masks."""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import stat
import time

from weft.io import canonical_hash
from weft.shared_world_geometry import _invariants, validate_geometry_hypothesis_config
from weft.shared_world_packaging import read_package_file, validate_package_layout


PROFILE = "exercise-geometry-resume-inputs-v1"
STUDY_PROFILE = "conditional-silhouette-world-study-v1"
MAX_FILES = 20
MAX_BYTES = 16 << 20
MAX_FILE_BYTES = 8 << 20
MAX_MANIFEST_BYTES = 1 << 20
MAX_RECEIPT_BYTES = 32 << 10
MAX_SECONDS = 5.
_SHA = re.compile(r"[0-9a-f]{64}")
_QUALIFICATION = {"conditional_exterior_opt_in", "original_foreground_preserved", "uncertain_envelope_preserved",
                  "within_original_envelope", "all_band_row_hulls_preserved", "strict_conditional_image_improvement"}
SCOPE = ("Reuse a saved conditional source-qualified geometry alternative, not fresh-video reconstruction. "
         "Source-fit and clearance values are retained study claims, not newly independently measured. "
         "Normal lab source verification and native admission remain required; physical accuracy is unverified.")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Saved geometry metadata exceeded its cooperative deadline")


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, "Duplicate saved geometry JSON key")
        result[key] = value
    return result


def _json(raw):
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError("Invalid saved geometry JSON") from error
    _require(isinstance(value, dict), "Saved geometry metadata must be a JSON object")
    stack, count = [(value, 0)], 0
    while stack:
        item, depth = stack.pop()
        count += 1
        _require(depth <= 64 and count <= 250000, "Saved geometry JSON exceeds its structure bound")
        if isinstance(item, float):
            _require(math.isfinite(item), "Saved geometry JSON contains a nonfinite value")
        elif isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
    return value


def _signature(path):
    info = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(info.st_mode), "Saved geometry inputs must be regular files")
    return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"))


class _Reader:
    def __init__(self, root, deadline):
        self.root, self.deadline = root, deadline
        self.raw, self.pins, self.total = {}, {}, 0

    def take(self, name, digest=None, maximum=MAX_FILE_BYTES):
        _check(self.deadline)
        if name not in self.raw:
            _require(len(self.raw) < MAX_FILES and self.total < MAX_BYTES, "Saved geometry metadata exceeds its file/byte bound")
            before = _signature(self.root / name)
            raw = read_package_file(self.root, name, max_bytes=min(maximum, MAX_BYTES - self.total), expected_sha256=digest)
            _require(_signature(self.root / name) == before, "Saved geometry metadata changed while reading")
            self.raw[name], self.pins[name] = raw, before
            self.total += len(raw)
        raw = self.raw[name]
        _require(len(raw) <= maximum, "Saved geometry metadata exceeds its per-file bound")
        _require(digest is None or sha256(raw).hexdigest() == digest, "Saved geometry metadata pins conflict")
        return raw

    def manifest(self, name, digest=None):
        raw = self.take(name, digest, MAX_MANIFEST_BYTES)
        document = _json(raw)
        files = document.get("files")
        _require(type(document.get("schema_version")) is int and document["schema_version"] == 1
                 and isinstance(files, dict) and 1 <= len(files) <= 2048, "Unsupported saved geometry manifest")
        validate_package_layout(files)
        _require("manifest.json" not in files and all(type(pin) is str and _SHA.fullmatch(pin) for pin in files.values()),
                 "Saved geometry manifest requires exact member hashes")
        return document, files


@dataclass(frozen=True)
class GeometryResumeInputs:
    study: Path
    candidate_run: Path
    members: tuple
    signatures: tuple
    summary_bytes: bytes

    def receipt(self):
        return json.loads(self.summary_bytes)

    def snapshot_files(self):
        # The selected snapshot is not a complete copy of any original package.
        aliases = {"manifest.json": "study-manifest.json", "geometry/manifest.json": "candidate-manifest.json",
                   "geometry-input/manifest.json": "baseline-manifest.json"}
        return {**{aliases.get(name, name): raw for name, raw in self.members}, "resume.json": self.summary_bytes}


def recheck_geometry_resume_inputs(prepared, *, deadline):
    _require(type(prepared) is GeometryResumeInputs, "Use prepared saved geometry inputs")
    for name, signature in prepared.signatures:
        _check(deadline)
        _require(_signature(prepared.study / name) == signature, "Saved geometry input changed after qualification")


def _read_geometry_resume_inputs(run, *, expected_manifest_sha256, deadline):
    _require(expected_manifest_sha256 is None or type(expected_manifest_sha256) is str
             and _SHA.fullmatch(expected_manifest_sha256), "Geometry manifest pin must be a SHA256")
    root = Path(run).absolute()
    _require(root.resolve() == root and root.is_dir(), "Use a real saved geometry study directory without symlinks")
    limit = min(deadline, time.monotonic() + MAX_SECONDS)
    reader = _Reader(root, limit)
    manifest, files = reader.manifest("manifest.json", expected_manifest_sha256)
    _require(manifest.get("status") == "completed", "Geometry resume requires a completed silhouette study, not a partial or failed run")

    def member(name):
        _require(name in files, "Saved geometry study lacks required metadata: " + name)
        return reader.take(name, files[name])

    report, settings = _json(member("report.json")), _json(member("settings.json"))
    _require(report.get("profile") == settings.get("profile") == STUDY_PROFILE
             and report.get("status") == "completed" and report.get("stage") == "finished"
             and report.get("original_inputs_unchanged") is True, "Unsupported or unfinished saved silhouette study")
    _require(report.get("settings_sha256") == canonical_hash(settings) and settings.get("conditional_exterior") is True,
             "Saved conditional settings identity differs")
    _require(report.get("candidate_directory") == "geometry" and report.get("geometry_admission", {}).get("status") == "ready"
             and report.get("clearance", {}).get("status") == "complete", "Saved study has no ready alternative with a complete recorded comparison")
    _require(report.get("physical_validation") == "unverified", "Saved geometry must not claim physical validation")
    member("geometry/manifest.json")
    candidate_manifest, candidate_files = reader.manifest("geometry/manifest.json", files["geometry/manifest.json"])
    _require(candidate_manifest.get("status") == "ready"
             and report.get("candidate_manifest_sha256") == sha256(reader.raw["geometry/manifest.json"]).hexdigest(),
             "Selected geometry manifest differs from its study")
    for name, digest in candidate_files.items():
        _require(files.get("geometry/" + name) == digest, "Study and selected geometry member identities differ")

    config_raw = member("geometry-config.json")
    _require(member("geometry/input-config.json") == config_raw, "Saved geometry configuration bytes differ")
    config = validate_geometry_hypothesis_config(_json(config_raw))
    baseline_raw, scene_raw = member("geometry-input/scene.json"), member("geometry/scene.json")
    baseline, scene = _json(baseline_raw), _json(scene_raw)
    base_sha, scene_sha = canonical_hash(baseline), canonical_hash(scene)
    _require(settings.get("scene_sha256") == config["base_scene_sha256"] == base_sha
             and baseline.get("source_binding", {}).get("mode") == "source_qualified", "Alternative is not bound to the exact source-qualified baseline")
    selected = [row["object_id"] for row in config["replacements"]]
    _require(set(scene) == set(baseline), "Alternative changed the world document fields")
    preserved = _invariants(baseline, scene, set(selected))
    _require(scene.get("learned_policy_available") is False, "A geometry alternative cannot inherit a prior learned policy")
    original_objects = {obj["id"]: obj for obj in baseline["objects"]}
    candidate_objects = {obj["id"]: obj for obj in scene["objects"]}
    for row in config["replacements"]:
        obj = candidate_objects[row["object_id"]]
        _require(obj.get("shape") == "compound" and obj.get("geometry") == row["geometry"], "Selected geometry differs from its configuration")

    base_task_raw, task_raw = member("geometry-input/exercise-task.json"), member("geometry/exercise-task.json")
    base_task, task = _json(base_task_raw), _json(task_raw)
    _require(base_task.get("scene_sha256") == base_sha and task == {**base_task, "scene_sha256": scene_sha},
             "Saved task may change only its scene identity; parameters, gates, source scope and clocks must remain exact")
    _require(settings.get("task_file_sha256") == sha256(base_task_raw).hexdigest(), "Original task differs from study settings")
    admission = _json(member("geometry/geometry-hypothesis.json"))
    _require(admission.get("profile") == "shared-world-geometry-hypothesis-v1" and admission.get("status") == "ready"
             and admission.get("initial_geometry_admitted") is True
             and admission.get("base_scene_sha256") == base_sha and admission.get("candidate_scene_sha256") == scene_sha
             and admission.get("input_config_sha256") == sha256(config_raw).hexdigest()
             and admission.get("selected_objects") == selected, "Saved admission identities disagree")
    rebound = admission.get("exercise_task", {})
    _require(rebound.get("status") == "metadata_only_scene_rebind" and rebound.get("changed_fields") == ["scene_sha256"]
             and rebound.get("parameters_source_scope_and_gates_unchanged") is True
             and rebound.get("previous_file_sha256") == sha256(base_task_raw).hexdigest()
             and rebound.get("new_file_sha256") == sha256(task_raw).hexdigest(), "Saved task-rebind receipt differs")
    member("geometry-input/manifest.json")
    _, base_files = reader.manifest("geometry-input/manifest.json", files["geometry-input/manifest.json"])
    _require(base_files.get("scene.json") == sha256(baseline_raw).hexdigest()
             and base_files.get("exercise-task.json") == sha256(base_task_raw).hexdigest()
             and admission.get("base_manifest_sha256") == sha256(reader.raw["geometry-input/manifest.json"]).hexdigest(),
             "Original box package identity differs")
    _require(settings.get("source_manifest_sha256") == sha256(member("inputs/base-manifest.json")).hexdigest()
             and settings.get("mask_manifest_sha256") == sha256(member("inputs/masks-manifest.json")).hexdigest(),
             "Recorded source or mask manifest differs")
    previous = _json(member("geometry/previous-geometry.json"))
    expected_previous = [{"object_id": row["object_id"],
        "previous_geometry": {key: deepcopy(original_objects[row["object_id"]][key]) for key in ("shape", "size", "radius", "geometry")
                              if key in original_objects[row["object_id"]]},
        "previous_provenance": original_objects[row["object_id"]]["provenance"], "authored_replacement": row["geometry"]}
        for row in config["replacements"]]
    _require(previous.get("objects") == expected_previous, "Original box alternative was not preserved exactly")

    selections, objects = settings.get("selections"), report.get("objects")
    _require(isinstance(selections, list) and isinstance(objects, list) and 1 <= len(selections) == len(objects) <= 4,
             "Saved geometry requires a bounded complete selection list")
    _require([row.get("world_object_id") for row in selections] == [row.get("object_id") for row in objects]
             and len({row["world_object_id"] for row in selections}) == len(selections), "Saved object selections differ")
    replacements, claims = [], []
    for selection, summary in zip(selections, objects):
        name = selection["world_object_id"]
        _require(name in original_objects and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", name), "Invalid selected equipment identity")
        path = "objects/" + name + "/result.json"
        _require(summary.get("report") == path, "Saved object report path differs")
        proposal = _json(member(path))
        detail = proposal.get("report", {})
        ref = original_objects[name]["provenance"].get("source_reference")
        _require(proposal.get("status") == summary.get("status") == detail.get("status")
                 and detail.get("source_reference") == ref and isinstance(ref, dict)
                 and ref.get("frame") == selection.get("source_frame")
                 and ref.get("document_sha256") == selection.get("document_sha256")
                 and ref.get("object_id") == selection.get("observation_entity_id")
                 and detail.get("camera_hypothesis") == original_objects[name]["provenance"].get("camera_hypothesis")
                 and detail.get("input_hashes", {}).get("scene_sha256") == base_sha,
                 "Saved mask/geometry report is not bound to the original object, camera and source frame")
        _require(summary.get("before_conditional_mask_fit") == detail.get("baseline", {}).get("scores")
                 and summary.get("after_conditional_mask_fit") == detail.get("candidate", {}).get("scores"), "Retained source-fit summaries differ")
        if proposal["status"] == "conditional_proposal":
            one = validate_geometry_hypothesis_config(proposal.get("geometry_config"))
            _require(one["base_scene_sha256"] == base_sha and len(one["replacements"]) == 1
                     and one["replacements"][0]["object_id"] == name
                     and one["replacements"][0]["geometry"] == proposal.get("candidate_geometry") == detail.get("candidate", {}).get("geometry")
                     and set(detail.get("qualification", {})) == _QUALIFICATION
                     and all(value is True for value in detail["qualification"].values()), "Saved conditional proposal is inconsistent")
            replacements.extend(one["replacements"])
        else:
            _require(proposal["status"] == "no_qualified_improvement" and proposal.get("geometry_config") is None,
                     "Failed or unsupported object proposals cannot be skipped")
        claims.append({"object_id": name, "saved_status": proposal["status"],
            "before_conditional_mask_fit": summary["before_conditional_mask_fit"],
            "after_conditional_mask_fit": summary["after_conditional_mask_fit"],
            "scope": "Retained study claims; not newly measured", "report_sha256": files[path]})
    _require(replacements == config["replacements"], "The selected compound omits or adds a qualified replacement")
    summary = {"schema_version": 1, "profile": PROFILE, "status": "selected_metadata_verified", "scope": SCOPE,
        "study_manifest_sha256": sha256(reader.raw["manifest.json"]).hexdigest(),
        "candidate_manifest_sha256": sha256(reader.raw["geometry/manifest.json"]).hexdigest(),
        "baseline_scene_sha256": base_sha, "candidate_scene_sha256": scene_sha,
        "baseline_task_sha256": canonical_hash(base_task), "task_sha256": canonical_hash(task),
        "source_sha256": baseline["source_binding"].get("source_sha256"), "selected_objects": selected,
        "preserved": preserved, "retained_source_fit": claims, "source_fit": "retained_conditional_study_claims",
        "saved_clearance_status": report["clearance"]["status"], "physical_validation": "unverified",
        "metadata_members": {name: sha256(raw).hexdigest() for name, raw in reader.raw.items()},
        "metadata_files_read": len(reader.raw), "metadata_bytes_read": reader.total,
        "work": {"fits": 0, "media_decodes": 0, "physics_steps": 0, "inference_calls": 0, "api_calls": 0}}
    encoded = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _require(len(encoded) <= MAX_RECEIPT_BYTES, "Saved geometry resume receipt exceeds its bound")
    prepared = GeometryResumeInputs(root, root / "geometry", tuple(reader.raw.items()), tuple(reader.pins.items()), encoded)
    recheck_geometry_resume_inputs(prepared, deadline=limit)
    return prepared


def read_geometry_resume_inputs(run, *, expected_manifest_sha256=None, deadline):
    """Check selected metadata custody; ordinary lab admission still verifies the full candidate."""
    try:
        return _read_geometry_resume_inputs(run, expected_manifest_sha256=expected_manifest_sha256, deadline=deadline)
    except (KeyError, IndexError, TypeError, AttributeError) as error:
        raise ValueError("Malformed or unsupported saved geometry metadata") from error
