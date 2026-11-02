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
    if (len(manifest_raw) > 1 << 20 or len(files) > 6000
            or config.get("control_profile") != "shared_exercise"
            or config.get("manifest_sha256") != manifest_hash
            or json.loads(manifest_raw).get("status") != "ready"):
        raise ValueError("Import requires an exact ready shared-exercise lab")
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
        jobs, pending = [], 0
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
            pending += status.get("status") not in TERMINAL
        if len(jobs) >= MAX_JOBS or _storage(root, check) + (pending + 1) * MAX_JOB_BYTES > MAX_STORAGE:
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
            if total > MAX_JOB_BYTES - 1_000_000:
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
        lab = open_portal_lab(target, project=project)
        try:
            summary = lab.summary()
            if summary.get("control_profile") != "shared_exercise" or summary.get("scene") != scene:
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
               "inventory_method": "saved_world", "learned_policy_available": learned}
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
