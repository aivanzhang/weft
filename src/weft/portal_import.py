"""Copy a sealed exercise lab into the stopped portal without rerunning reconstruction."""

import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import time
from uuid import uuid4

from weft.io import local_time, write_json
from weft.openai_access import PROJECT_ROOT
from weft.portal import MAX_JOB_BYTES, MAX_JOBS, MAX_STORAGE, TERMINAL, bounded_json, private_directory
from weft.portal_lab_runtime import open_portal_lab
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


SCOPE = "Imported simulation candidate, not automatic video reconstruction or verified physical accuracy. Task success is not implied by a playable world."


def _directory(value):
    path = Path(value).absolute()
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)) or not path.is_dir():
        raise ValueError("Import requires an existing directory without traversal or links")
    return path


def _storage(root, check):
    size = count = 0
    for directory, children, files in os.walk(root, followlinks=False):
        check()
        for name in children + files:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError("Portal storage contains a link")
            count += 1
            if count > 100_000:
                raise ValueError("Portal storage exceeds the import inventory bound")
        size += sum((Path(directory) / name).stat().st_size for name in files)
        if size > MAX_STORAGE:
            raise ValueError("Portal storage is full")
    return size


def import_lab(source, root, *, name=None, project=PROJECT_ROOT, max_seconds=60):
    """Hold the portal lease; publish only after ordinary lab admission succeeds."""
    from weft import portal_moving_equipment, portal_perception_jobs

    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 60:
        raise ValueError("Import deadline must be positive and at most 60 seconds")
    started = time.monotonic()

    def check():
        if time.monotonic() - started >= max_seconds:
            raise TimeoutError("Portal import exceeded its deadline")

    source = _directory(source)
    root = Path(root).absolute()
    if source == root or source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError("Import source and portal storage must be separate")
    config_raw = read_package_file(source, "lab-config.json", max_bytes=64 << 10)
    config = json.loads(config_raw)
    manifest_raw, files = read_package_manifest(source / "run")
    manifest_hash = sha256(manifest_raw).hexdigest()
    exploration = config.get("control_profile") == "shared_world_exploration"
    placement = config.get("control_profile") == "moving_equipment_task"
    if (len(manifest_raw) > 1 << 20 or len(files) > 6000
            or config.get("control_profile") not in {"shared_exercise", "shared_world_exploration", "moving_equipment_task"}
            or config.get("manifest_sha256") != manifest_hash
            or json.loads(manifest_raw).get("status") != "ready"):
        raise ValueError("Import requires an exact ready source-bound exercise or moving-task lab")
    scene = json.loads(read_package_file(source / "run", "scene.json", max_bytes=16 << 20,
                                       expected_sha256=files.get("scene.json")))
    binding = scene.get("source_binding", {})
    source_hash = binding.get("source_sha256")
    if binding.get("mode") != "source_qualified" or not isinstance(source_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", source_hash):
        raise ValueError("Portal import requires a source-bound world, not an unbound fixture")
    media = [n for n, h in files.items() if re.fullmatch(r"source\.(webm|mp4|mov|m4v|avi|mkv)", n) and h == source_hash]
    if len(media) != 1:
        raise ValueError("Import requires one sealed original source video at the lab run root")
    media = media[0]
    # A closed 256 MiB task child needs only one additional source copy and small import receipts.
    import_limit = min(MAX_JOB_BYTES, (256 << 20) + (64 << 20) + 1_000_000) if placement else MAX_JOB_BYTES
    name = name or "Imported world" + Path(media).suffix
    if (not isinstance(name, str) or not 1 <= len(name) <= 180 or Path(name).name != name
            or any(ord(c) < 32 for c in name) or Path(name).suffix.lower() != Path(media).suffix):
        raise ValueError("Project name must be a filename with the original video extension")
    key = sha256(manifest_raw + config_raw).hexdigest()
    root = private_directory(root)
    lease = os.open(root / "portal.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stage = None
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Stop the portal before importing; its active jobs and recordings must not be interrupted") from None
        jobs, pending_bytes = [], 0
        for folder in root.iterdir():
            check()
            if not re.fullmatch(r"[a-f0-9]{32}", folder.name):
                continue
            _directory(folder)
            job = bounded_json(folder / "job.json")
            jobs.append(job)
            status = bounded_json(folder / "status.json")
            if job.get("import_identity") == key and status.get("status") == "ready":
                return {"id": folder.name, "status": "already_imported", "directory": str(folder),
                        "manifest_sha256": manifest_hash, "revalidated": False, "api_calls": 0}
            moving_growth = portal_moving_equipment.reserved_growth(folder)
            active = status.get("status") not in TERMINAL
            for relative in ("equipment/status.json", "muse/status.json", "muse/checks/status.json", "muse/repairs/status.json"):
                if (folder / relative).is_file():
                    active |= bounded_json(folder / relative).get("status") in {"queued", "running", "cancelling"}
            active |= portal_perception_jobs.read_status(folder).get("status") in portal_perception_jobs.ACTIVE
            pending_bytes += moving_growth or (MAX_JOB_BYTES if active else 0)
        if len(jobs) >= MAX_JOBS or _storage(root, check) + pending_bytes + import_limit > MAX_STORAGE:
            raise ValueError("Portal capacity is full; existing projects and queued reservations are preserved")
        identity = uuid4().hex
        stage = root / (".import-" + identity)
        stage.mkdir(mode=0o700)
        target = stage / "lab"
        (target / "run").mkdir(mode=0o700, parents=True)
        total = len(config_raw) + len(manifest_raw)
        for filename, expected in files.items():
            check()
            raw = read_package_file(source / "run", filename, max_bytes=64 << 20, expected_sha256=expected)
            total += len(raw) * (2 if filename == media else 1)
            if total > import_limit - 1_000_000:
                raise ValueError("Imported lab exceeds the portal project byte limit")
            write_package_files(target / "run", {filename: raw})
            if filename == media:
                write_package_files(stage, {media: raw})
        write_package_files(target / "run", {"manifest.json": manifest_raw})
        write_package_files(target, {"lab-config.json": config_raw})
        # The ordinary loader requires terminal lifecycle metadata; this copy remains unpublished until admission.
        write_json(target / "status.json", {"status": "ready", "events": [{"stage": "import",
                   "message": "Copied sealed run; ordinary admission required before portal publication", "time": local_time()}]})
        check()
        if placement:
            from weft.moving_equipment_task_lab import TaskLab
            lab = TaskLab(target, max_seconds=max(.01, max_seconds - (time.monotonic() - started)))
        else:
            lab = open_portal_lab(target, project=project)
        try:
            summary = lab.summary()
            if summary.get("control_profile") != config["control_profile"] or summary.get("scene") != scene:
                raise ValueError("Imported lab admission changed the saved world")
            learned = summary.get("learned_policy_available") is True
        finally:
            lab.close()
        check()
        if (read_package_file(source, "lab-config.json", max_bytes=64 << 10) != config_raw
                or read_package_file(source / "run", "manifest.json", max_bytes=1 << 20) != manifest_raw):
            raise ValueError("Original lab seals changed during import")
        job = {"id": identity, "name": name, "created_at": local_time(), "ai": False, "mode": "exercise",
               "kind": "imported_lab", "source_file": media, "source_sha256": source_hash, "import_identity": key,
               "inventory_method": "saved_world", "learned_policy_available": learned,
               **({"control_profile": "shared_world_exploration", "task_success_available": False} if exploration else {}),
               **({"control_profile": "moving_equipment_task", "task_scope": "authored_virtual_force_placement"} if placement else {})}
        write_json(stage / "job.json", job)
        (stage / "analysis").mkdir(mode=0o700)
        receipt = {"profile": "portal-lab-import-v1", "source_manifest_sha256": manifest_hash,
                   "source_lab_config_sha256": sha256(config_raw).hexdigest(), "files_copied": len(files),
                   "bytes_copied": total, "original_sessions_imported": False, "learned_policy_available": learned,
                   "seconds": time.monotonic() - started, "api_calls": 0, "physics_steps": 0, "scope": SCOPE}
        write_json(stage / "analysis/import.json", receipt)
        write_json(stage / "analysis/plan.json", {"route": "exercise", "summary": SCOPE,
                   "warnings": ["Saved evidence and task are unchanged; recordings made here belong to this new project.",
                                "Muse review needs a verified body-evidence attachment; it is unavailable for this import."]})
        write_json(stage / "status.json", {"status": "ready", "stage": "ready", "route": "exercise", "lab": "lab",
                   "events": [{"stage": "import", "message": "Saved world verified and copied; no reconstruction or training rerun", "time": local_time()}],
                   "scope": SCOPE, "learned_policy_available": learned})
        check()
        os.rename(stage, root / identity)
        return {"id": identity, "status": "ready", "directory": str(root / identity), **receipt}
    except Exception as error:
        if stage is not None and stage.exists():
            if (stage / "lab/status.json").is_file():
                write_json(stage / "lab/status.json", {"status": "failed", "events": [], "error": "Import admission failed; partial evidence retained"})
            write_json(stage / "import-failure.json", {"type": type(error).__name__, "error": str(error)[:1000],
                        "partial_evidence_preserved": True, "published": False})
        raise
    finally:
        os.close(lease)


def attach_existing_import(root, project_id, body_source, *, project=PROJECT_ROOT, max_seconds=30):
    """Attach an explicit sealed source beside the imported run while the portal is stopped."""
    from weft.lab_worker import atomic_json
    from weft.portal_evidence_attachment import DIRECTORY, _body_files, _join, _receipt, _validate_body, validate_attachment

    if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= 30
            or not isinstance(project_id, str) or not re.fullmatch(r"[a-f0-9]{32}", project_id)):
        raise ValueError("Attachment needs a project ID and a positive deadline of at most 30 seconds")
    started = time.monotonic()
    deadline = started + max_seconds
    root, source = _directory(root), _directory(body_source)
    folder = _directory(root / project_id)
    if source == folder or source.is_relative_to(folder):
        raise ValueError("Attach an explicit external sealed body source, not nested imported files")
    lease = os.open(root / "portal.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stage = None
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Stop the portal before attaching evidence; active work must not be interrupted") from None
        job = bounded_json(folder / "job.json")
        if job.get("kind") != "imported_lab" or bounded_json(folder / "status.json").get("status") != "ready":
            raise ValueError("Attachment requires a ready imported exercise lab")
        if (folder / DIRECTORY).exists() or "review_attachment_sha256" in job:
            if "review_attachment_sha256" not in job:
                raise ValueError("Unregistered attachment preserved after interrupted publication; inspect it before recovery")
            validate_attachment(folder, project=project, max_seconds=max_seconds)
            return {"id": project_id, "status": "already_attached", "api_calls": 0}
        if (folder / "muse").exists():
            raise ValueError("Existing review history must not be rebound to different construction evidence")
        manifest, files = _body_files(source, deadline)
        human, observations, validation = _validate_body(source, files, project=project, deadline=deadline)
        world, document = _join(folder, files, human, observations, deadline)
        receipt = _receipt(manifest, human, world, document, validation)
        total = len(manifest) + sum(map(len, files.values()))
        def check():
            if time.monotonic() >= deadline:
                raise TimeoutError("Construction attachment exceeded its deadline")
        pending = sum(bounded_json(path / "status.json").get("status") not in TERMINAL
                      for path in root.iterdir() if re.fullmatch(r"[a-f0-9]{32}", path.name))
        if (_storage(root, check) + total + 65536 + pending * MAX_JOB_BYTES > MAX_STORAGE
                or _storage(folder, check) + total + 65536 > MAX_JOB_BYTES):
            raise ValueError("Construction attachment exceeds reserved portal storage")
        stage = folder / (".attachment-" + uuid4().hex)
        (stage / "body").mkdir(mode=0o700, parents=True)
        write_package_files(stage / "body", {**files, "manifest.json": manifest})
        receipt_raw = (json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        write_package_files(stage, {"receipt.json": receipt_raw})
        if read_package_file(source, "manifest.json", max_bytes=1 << 20) != manifest:
            raise ValueError("Original source seal changed during attachment")
        # Recheck the current world before publishing; no task or lab artifacts are edited.
        after, after_document = _join(folder, files, human, observations, deadline)
        if after != world or after_document != document:
            raise ValueError("Imported world changed during attachment")
        check()
        stage.rename(folder / DIRECTORY)
        job["review_attachment_sha256"] = sha256(receipt_raw).hexdigest()
        atomic_json(folder / "job.json", job)
        return {"id": project_id, "status": "attached", "receipt": str(folder / DIRECTORY / "receipt.json"),
                "body_manifest_sha256": receipt["body_manifest_sha256"], "scene_sha256": world["input_sha256"]["scene.json"],
                "copied_files": len(files) + 1, "copied_bytes": total, "api_calls": 0,
                "seconds": time.monotonic() - started, "repair_qualified": False}
    except Exception as error:
        if stage is not None and stage.is_dir():
            write_json(stage / "attachment-failure.json", {"published": False, "error_type": type(error).__name__, "api_calls": 0})
        raise
    finally:
        os.close(lease)


def register_prepared_review(root, project_id, *, expected_files):
    """Pin existing prepared bytes after explicit offline preparation; never decode or send them."""
    from weft.lab_worker import atomic_json
    from weft.portal_evidence_attachment import (DIRECTORY, PREPARATION_PROFILE, PREPARED_FILES,
        attachment_artifact, _json)
    from weft.io import canonical_hash

    if (not isinstance(project_id, str) or not re.fullmatch(r"[a-f0-9]{32}", project_id)
            or not isinstance(expected_files, dict) or set(expected_files) != set(PREPARED_FILES)
            or any(not isinstance(pin, str) or not re.fullmatch(r"[a-f0-9]{64}", pin) for pin in expected_files.values())):
        raise ValueError("Preparation registration requires exact pre-recorded hashes for its three files")
    root = _directory(root)
    folder = _directory(root / project_id)
    lease = os.open(root / "portal.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError("Stop the portal before registering prepared snapshots") from None
        job = bounded_json(folder / "job.json")
        attachment_artifact(folder, job, DIRECTORY + "/receipt.json")
        prepared = folder / DIRECTORY / "prepared"
        values = {name: _json(read_package_file(prepared, name, max_bytes=maximum, expected_sha256=expected_files[name]))
                  for name, maximum in PREPARED_FILES.items()}
        context, request, call = (values[n] for n in ("context.json", "request.json", "advisor-call.json"))
        from weft.portal_scene_context import construction_context

        world, world_binding = construction_context(folder)
        attached = _json(read_package_file(folder, DIRECTORY + "/receipt.json", max_bytes=32 << 10,
                                          expected_sha256=job["review_attachment_sha256"]))
        try:
            text = _json(request["messages"][1]["content"][0]["text"].encode())
        except (KeyError, IndexError, TypeError, AttributeError) as error:
            raise ValueError("Prepared review requires its exact context payload") from error
        if (text != context or call.get("status") != "prepared" or type(call.get("request_count")) is not int
                or call["request_count"] != 0 or call.get("request_sha256") != canonical_hash(request)
                or call.get("source_sha256") != job["source_sha256"] or context.get("source_sha256") != job["source_sha256"]
                or request.get("model") != call.get("model")
                or context.get("simulation_world") != world or attached["world"] != world_binding
                or call.get("simulation_world_sha256") != canonical_hash(context.get("simulation_world"))):
            raise ValueError("Prepared review request, context or source identity differs")
        manifest = {"schema_version": 1, "profile": PREPARATION_PROFILE,
            "attachment_receipt_sha256": job["review_attachment_sha256"], "source_sha256": job["source_sha256"],
            "files": dict(expected_files)}
        raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        if (prepared / "manifest.json").exists():
            if read_package_file(prepared, "manifest.json", max_bytes=8192) != raw:
                raise ValueError("A different prepared snapshot is already preserved")
        else:
            write_package_files(prepared, {"manifest.json": raw})
        job["review_preparation_sha256"] = sha256(raw).hexdigest()
        atomic_json(folder / "job.json", job)
        return {"id": project_id, "status": "prepared_snapshot_registered", "manifest_sha256": sha256(raw).hexdigest(),
                "files": expected_files, "api_calls": 0, "decodes": 0}
    finally:
        os.close(lease)
