"""Queue one local body-detail child without changing its failed parent."""

import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from uuid import uuid4

from weft.io import canonical_hash, local_time
from weft.shared_world_packaging import read_package_file, validate_package_layout, write_package_files

PROFILE = "portal-body-detail-child-v1"
RECIPE = "full-dense-region-v1"
REQUEST = "body-detail/request.json"
BODY = "perception/preflight/body_source/"
MAX_INPUT_BYTES = 16 << 20
MAX_SOURCE_BYTES = 37_000_000
PARENT_FILES = ("job.json", "status.json", "perception/status.json", "perception/failure.json",
    "perception/preflight/failure.json", BODY + "manifest.json", BODY + "failure.json")
INPUT_FILES = {"evidence.json": BODY + "evidence.json", "pose-landmarks.json": BODY + "pose-landmarks.json"}
FIELDS = {"schema_version", "profile", "recipe", "child_id", "parent_id", "request_id", "parent_files",
    "source_file", "source_sha256", "input_sha256", "model", "detail_identity", "api_calls_authorized",
    "local_pose_attempts_authorized"}
SCOPE = ("One local Full-model pass over every frame, using a fixed region containing every saved credible pose hypothesis. "
         "Person identity and missing coverage remain uncertain; original body, floor and clock checks still apply. "
         "No provider call is authorized. The failed parent is preserved.")


def _require(condition, message):
    if not condition:
        raise ValueError("Body detail: " + message)


def _check(deadline):
    _require(type(deadline) in (int, float) and math.isfinite(deadline), "finite worker deadline required")
    if time.monotonic() >= deadline:
        raise TimeoutError("Body-detail preparation exceeded its deadline")


def _json(raw):
    from weft.spatial_scene_inputs import _json as bounded_json
    return bounded_json(raw, 128 << 10)


def _bytes(value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _require(len(raw) <= 128 << 10, "metadata exceeds its bound")
    return raw


def _read(root, name, *, maximum=128 << 10, pin=None):
    return read_package_file(root, name, max_bytes=maximum, expected_sha256=pin)


def _directory(path):
    path = Path(path).absolute()
    _require(".." not in path.parts and path.is_dir()
             and not any(p.is_symlink() for p in (path, *path.parents)), "safe existing directory required")
    return path


def _size(path, maximum):
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "linked input")
    info = path.stat()
    _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= maximum, "input is missing or outside its byte bound")
    return info.st_size


def _model():
    from weft.portal_body_detail_recipe import MODEL_BYTES, MODEL_FILENAME, MODEL_SHA256
    return {"filename": MODEL_FILENAME, "sha256": MODEL_SHA256, "bytes": MODEL_BYTES}


def _parent(parent):
    """Read only small saved metadata and file sizes; semantic source admission belongs to the worker."""
    parent = _directory(parent)
    _require(re.fullmatch("[a-f0-9]{32}", parent.name), "exact parent project ID required")
    snapshots = {name: _read(parent, name) for name in PARENT_FILES}
    job, state, status, failure, preflight, manifest, body_failure = (_json(snapshots[n]) for n in PARENT_FILES)
    pin = job.get("source_sha256")
    _require(job.get("id") == parent.name and job.get("kind", "video") == "video"
             and job.get("mode") == "exercise" and job.get("source_file") == "source.mp4"
             and isinstance(pin, str) and re.fullmatch("[a-f0-9]{64}", pin)
             and "body_detail" not in job and "body_reuse" not in job,
             "original exercise upload required; recovery chains are unsupported")
    _require(state.get("status") in {"needs_input", "failed"} and state.get("route") == "exercise"
             and not state.get("lab") and not state.get("equipment_candidate_id")
             and status.get("status") == failure.get("status") == "failed"
             and preflight.get("stage") == "body" and type(preflight.get("api_calls")) is int
             and preflight["api_calls"] == 0, "failed local body preflight required")
    _require(not any((parent / name).exists() or (parent / name).is_symlink()
                     for name in ("lab", "equipment", "muse", "moving-equipment", "perception/request.json",
                                  "perception/preflight-complete.json", "cancel-request.json")),
             "active, paid, selected or completed work cannot be rebound")
    files = manifest.get("files")
    _require(manifest.get("profile") == "exercise-body-source-v1" and manifest.get("status") == "failed"
             and body_failure.get("status") == "failed" and type(files) is dict and len(files) <= 512,
             "sealed failed-body artifacts required")
    validate_package_layout([*files, "manifest.json"])
    _require(all(isinstance(v, str) and re.fullmatch("[a-f0-9]{64}", v) for v in files.values())
             and files.get("failure.json") == sha256(snapshots[BODY + "failure.json"]).hexdigest()
             and files.get("source.mp4") == pin
             and set(INPUT_FILES) <= files.keys(), "failed body input identities differ")
    _size(parent / "source.mp4", MAX_SOURCE_BYTES)
    total = sum(_size(parent / relative, MAX_INPUT_BYTES) for relative in INPUT_FILES.values())
    _require(total <= MAX_INPUT_BYTES, "original pose histories exceed 16 MiB")
    return job, snapshots, {name: files[name] for name in INPUT_FILES}


def body_detail_availability(parent):
    """Structural availability only; never parse poses or hash a model on the request thread."""
    try:
        job = _json(_read(parent, "job.json"))
        binding = job.get("body_detail")
        if isinstance(binding, dict) and binding.get("profile") == PROFILE and binding.get("recipe") == RECIPE:
            identity = binding.get("parent_id")
            if isinstance(identity, str) and re.fullmatch("[a-f0-9]{32}", identity) and job.get("parent_id") == identity:
                return {"available": False, "recipe": RECIPE, "parent_id": identity,
                    "reason": "Separate local attempt; the original failure is preserved. This child cannot start another retry."}
        job, _, _ = _parent(parent)
        return {"available": True, "recipe": RECIPE, "parent_id": job["id"],
            "reason": "Retained failed body evidence is available. Exact source, region and model eligibility are checked in the worker.",
            "scope": SCOPE}
    except (ValueError, OSError, TypeError, KeyError):
        return {"available": False, "recipe": RECIPE,
            "reason": "Requires an unresolved original exercise upload with a sealed failed local body preflight and bounded raw poses."}


def _identity(request):
    return canonical_hash({k: request[k] for k in ("parent_id", "parent_files", "source_sha256", "input_sha256", "model", "recipe")})


def _metadata(child):
    job_raw = _read(child, "job.json"); job = _json(job_raw)
    binding = job.get("body_detail")
    _require(type(binding) is dict and set(binding) == {"profile", "recipe", "parent_id", "request_id", "request_sha256"}
             and binding.get("profile") == PROFILE and binding.get("recipe") == RECIPE,
             "closed child body-detail binding required")
    request_raw = _read(child, REQUEST, pin=binding["request_sha256"]); request = _json(request_raw)
    _require(set(request) == FIELDS and request.get("schema_version") == 1 and type(request["schema_version"]) is int
             and request.get("profile") == PROFILE and request.get("recipe") == RECIPE
             and request.get("model") == _model() and request.get("detail_identity") == _identity(request)
             and job.get("detail_identity") == request["detail_identity"]
             and all(isinstance(request.get(k), str) and re.fullmatch("[a-f0-9]{32}", request[k])
                     for k in ("child_id", "parent_id", "request_id"))
             and job.get("id") == request["child_id"] and job.get("parent_id") == request["parent_id"]
             and binding["parent_id"] == request["parent_id"] and binding["request_id"] == request["request_id"]
             and job.get("kind") == "video" and job.get("mode") == "exercise" and job.get("ai") is False
             and job.get("source_file") == request.get("source_file") == "source.mp4"
             and job.get("source_sha256") == request.get("source_sha256")
             and type(request.get("api_calls_authorized")) is int and request["api_calls_authorized"] == 0
             and type(request.get("local_pose_attempts_authorized")) is int and request["local_pose_attempts_authorized"] == 1
             and type(request.get("parent_files")) is dict and set(request["parent_files"]) == set(PARENT_FILES)
             and type(request.get("input_sha256")) is dict and set(request["input_sha256"]) == set(INPUT_FILES),
             "child request or original source identity differs")
    pre = _json(_read(child, "perception/preflight-request.json"))
    _require(set(pre) == {"profile", "source_sha256", "requested_at", "api_calls_authorized", "body_detail"}
             and pre.get("profile") == "portal-exercise-perception-v1" and pre.get("body_detail") == binding
             and pre.get("source_sha256") == request["source_sha256"]
             and type(pre.get("api_calls_authorized")) is int and pre["api_calls_authorized"] == 0,
             "queued local preflight differs from child request")
    for name, pin in request["parent_files"].items():
        _read(child, "body-detail/parent/" + name, pin=pin)
    return job, request, request_raw


def create_body_detail_child(root, parent_id, payload, *, lease_fd, existing_jobs, stored_bytes,
                             reserved_bytes, pending_jobs, deadline):
    """Called under the live portal's lock/lease; queue only small immutable metadata."""
    from weft.portal import MAX_JOBS, MAX_JOB_BYTES, MAX_STORAGE
    from weft.portal_inventory import validate_intent

    _check(deadline)
    root = _directory(root)
    _require(type(parent_id) is str and re.fullmatch("[a-f0-9]{32}", parent_id)
             and type(payload) is dict and set(payload) == {"recipe", "request_id"}
             and payload["recipe"] == RECIPE and type(payload["request_id"]) is str
             and re.fullmatch("[a-f0-9]{32}", payload["request_id"]), "choose the fixed recipe and one request ID")
    _require(type(lease_fd) is int and not (root / "portal.lock").is_symlink(), "existing portal lease required")
    lock = os.fstat(lease_fd); path = (root / "portal.lock").stat()
    _require((lock.st_dev, lock.st_ino) == (path.st_dev, path.st_ino), "lease does not belong to this portal")
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent = root / parent_id
    parent_job, snapshots, inputs = _parent(parent)
    _require(parent_id in existing_jobs and existing_jobs[parent_id] == parent_job,
             "parent differs from the portal inventory")
    request = {"schema_version": 1, "profile": PROFILE, "recipe": RECIPE, "child_id": uuid4().hex,
        "parent_id": parent_id, "request_id": payload["request_id"],
        "parent_files": {n: sha256(v).hexdigest() for n, v in snapshots.items()}, "source_file": "source.mp4",
        "source_sha256": parent_job["source_sha256"], "input_sha256": inputs, "model": _model(),
        "api_calls_authorized": 0, "local_pose_attempts_authorized": 1}
    request["detail_identity"] = _identity(request)
    _require(isinstance(existing_jobs, dict) and len(existing_jobs) <= MAX_JOBS, "bounded portal inventory required")
    for identity, existing in existing_jobs.items():
        _check(deadline)
        if "body_detail" not in existing:
            _require(existing.get("request_id") != payload["request_id"], "request ID already belongs to another project")
            continue
        binding = existing["body_detail"]
        if binding.get("parent_id") == parent_id or binding.get("request_id") == payload["request_id"]:
            saved, prior, _ = _metadata(_directory(root / identity))
            _require(saved == existing and prior["detail_identity"] == request["detail_identity"],
                     "one attempt per parent and recipe; prior child is preserved")
            return saved
    _require(all(type(n) is int and n >= 0 for n in (stored_bytes, reserved_bytes, pending_jobs))
             and len(existing_jobs) < MAX_JOBS and pending_jobs < 4
             and stored_bytes + reserved_bytes + MAX_JOB_BYTES <= MAX_STORAGE,
             "portal cannot reserve a complete new child; existing evidence is preserved")
    identity = request["child_id"]; stage = root / (".body-detail-" + identity)
    stage.mkdir(mode=0o700)
    try:
        request_raw = _bytes(request)
        binding = {"profile": PROFILE, "recipe": RECIPE, "parent_id": parent_id,
                   "request_id": payload["request_id"], "request_sha256": sha256(request_raw).hexdigest()}
        child = {"id": identity, "name": parent_job.get("name", "Exercise.mp4"), "kind": "video", "mode": "exercise",
            "ai": False, "created_at": local_time(), "parent_id": parent_id, "source_file": "source.mp4",
            "source_sha256": request["source_sha256"], "body_detail": binding, "detail_identity": request["detail_identity"],
            "exercise_intent": validate_intent(parent_job.get("exercise_intent"))}
        state = _bytes({"status": "needs_input", "route": "exercise", "stage": "body_detail", "events": [], "scope": SCOPE})
        pre = {"profile": "portal-exercise-perception-v1", "source_sha256": request["source_sha256"],
               "requested_at": local_time(), "api_calls_authorized": 0, "body_detail": binding}
        write_package_files(stage, {"job.json": _bytes(child), "status.json": state, REQUEST: request_raw,
            "perception/parent-status.json": state, "perception/preflight-request.json": _bytes(pre),
            "perception/status.json": _bytes({"status": "queued_preflight", "stage": "body_detail", "message": SCOPE}),
            **{"body-detail/parent/" + n: raw for n, raw in snapshots.items()}})
        for name, pin in request["parent_files"].items():
            _check(deadline); _read(parent, name, pin=pin)
        _parent(parent)
        _check(deadline)
        os.rename(stage, root / identity)
        return child
    except Exception as error:
        write_package_files(stage, {"publication-failure.json": _bytes({"published": False,
            "error": str(error)[:1000], "partial_evidence_preserved": True})})
        raise


def prepare_queued_body_detail(child, *, project, deadline):
    """Run once in the existing keyless worker, after its exclusive start marker."""
    from weft.portal_body_detail_recipe import _encode, build_body_detail_recipe

    _check(deadline)
    child = _directory(child)
    job, request, request_raw = _metadata(child)
    _require(child.name == job["id"], "child directory identity differs")
    start = _json(_read(child, "perception/preflight-start.json"))
    _require(type(start.get("api_calls")) is int and start["api_calls"] == 0,
             "exclusive zero-provider preflight start marker required")
    _require(not any((child / n).exists() or (child / n).is_symlink()
                     for n in ("source.mp4", "body-detail/inputs", "body-detail/recipe.json", "perception/preflight",
                               "perception/request.json", "perception/quote.json", "perception/cancel-request.json", "cancel-request.json")),
             "started or partial body-detail work cannot be retried")
    parent = child.parent / request["parent_id"]
    parent_job, snapshots, inputs = _parent(parent)
    _require({n: sha256(v).hexdigest() for n, v in snapshots.items()} == request["parent_files"]
             and inputs == request["input_sha256"] and parent_job["source_sha256"] == request["source_sha256"],
             "failed parent changed after child queueing")
    _check(deadline)
    video = _read(parent, "source.mp4", maximum=MAX_SOURCE_BYTES, pin=request["source_sha256"])
    evidence = _read(parent, INPUT_FILES["evidence.json"], maximum=MAX_INPUT_BYTES, pin=inputs["evidence.json"])
    _require(len(evidence) < MAX_INPUT_BYTES, "raw history bound exhausted")
    raw = _read(parent, INPUT_FILES["pose-landmarks.json"], maximum=MAX_INPUT_BYTES-len(evidence), pin=inputs["pose-landmarks.json"])
    model = request["model"]; model_root = Path(project).absolute() / ".local/models"
    model_raw = _read(model_root, model["filename"], maximum=model["bytes"], pin=model["sha256"])
    _require(len(model_raw) == model["bytes"], "fixed Full model size differs")
    _check(deadline)
    parent_pins = {"project_id": request["parent_id"], "job_sha256": request["parent_files"]["job.json"],
        "status_sha256": request["parent_files"]["status.json"],
        "preflight_failure_sha256": request["parent_files"]["perception/failure.json"]}
    recipe = build_body_detail_recipe(evidence, raw, parent=parent_pins)
    _require(recipe.get("source_sha256") == request["source_sha256"], "recipe belongs to another source")
    _check(deadline)
    write_package_files(child, {"source.mp4": video, "body-detail/inputs/evidence.json": evidence,
        "body-detail/inputs/pose-landmarks.json": raw, "body-detail/recipe.json": _encode(recipe)})
    for name, pin in request["parent_files"].items():
        _check(deadline); _read(parent, name, pin=pin)
    _read(parent, "source.mp4", maximum=MAX_SOURCE_BYTES, pin=request["source_sha256"])
    for name, relative in INPUT_FILES.items():
        _check(deadline); _read(parent, relative, maximum=MAX_INPUT_BYTES, pin=inputs[name])
    _read(child, REQUEST, pin=sha256(request_raw).hexdigest())
    _check(deadline)
    return {"pose_model": model_root / model["filename"], "body_detail_recipe": recipe,
            "body_detail_inputs": (evidence, raw)}
