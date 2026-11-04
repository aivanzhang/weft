"""Local product portal: authenticated uploads, durable jobs and the existing sandbox."""

from collections import OrderedDict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4

from weft.io import local_time, write_json
from weft.lab import LabHandler, LabServer, PUBLIC_ASSETS
from weft.lab_worker import atomic_json
from weft.openai_access import MODEL, PROJECT_ROOT, OpenAIBudget, credential_status
from weft.portal_review import TERMINAL as REVIEW_TERMINAL, availability as muse_availability
from weft.portal_checks import TERMINAL as CHECK_TERMINAL
from weft.portal_repair_jobs import TERMINAL as REPAIR_TERMINAL, availability as repair_availability, read_small_report
from weft.portal_repair_jobs import choice_requested
from weft.portal_repair_jobs import availability as repair_choice_availability
from weft.portal_supports import body_path
from weft.portal_lab_runtime import open_portal_lab
from weft.portal_inventory import exercise_intent
from weft.portal_equipment_candidates import (TERMINAL as EQUIPMENT_TERMINAL, read_catalog,
    availability as equipment_availability, read_status as equipment_status, read_report as equipment_report,
    select_candidate)
from weft import portal_perception_jobs as perception_jobs
from weft import portal_moving_equipment as moving_jobs

MAX_UPLOAD = 100_000_000
MAX_JOBS = 50
MAX_STORAGE = 2_000_000_000
MAX_JOB_BYTES = 650_000_000
TERMINAL = {"ready", "preview", "needs_input", "failed", "cancelled", "interrupted", "complete"}
DEMO_BRAND_ASSETS = {"/demo-font.woff2": "website/fonts/InterVariable.woff2",
                     "/demo-wordmark.png": "assets/brand/source/weft-wordmark.png"}
ASSETS = {"/": "portal.html", "/portal.js": "portal.js", "/portal.css": "portal.css",
          "/demo": "demo_build.html", "/demo/runs": "demo.html", "/demo.js": "demo.js", "/demo.css": "demo.css",
          "/demo/build": "demo_build.html", "/demo_build.js": "demo_build.js",
          "/demo_focus.js": "demo_focus.js"}
PERCEPTION_ARTIFACTS = {"perception/" + name for name in (
    "preflight/metadata.json", "preflight/body-anchor.json", "preflight/result.json", "preflight/manifest.json",
    "preflight/body_source/human.json", "preflight/body_source/human.xml", "preflight/body_source/evidence.json",
    "preflight/body_source/pose-landmarks.json", "pose-diagnostic/report.json", "pose-diagnostic/summary.json",
    "pose-diagnostic/manifest.json", "pose-diagnostic-error.json",
    "preflight/body_source/native-clock.json", "preflight/body_source/tracks.gif", "preflight/body_source/tracks-contact-sheet.png",
    "quote.json", "request.json", "status.json", "failure.json", "preflight-complete.json", "prepared-source.json",
    "preflight-worker-receipt.json", "preparation-worker-receipt.json",
    "analysis/frames.json", "analysis/plan.json", "analysis/request.json", "analysis/response.json", "analysis/receipt.json",
    "bridge/result.json", "bridge/manifest.json", "bridge/plan.json", "bridge/body-anchor.json", "bridge/preflight-descriptor.json")}


def bounded_json(path, maximum=128_000):
    if path.stat().st_size > maximum:
        raise ValueError("Saved metadata exceeds its bound")
    return json.loads(path.read_bytes())


def private_directory(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Portal storage cannot contain symlinks")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o077:
        raise ValueError("Portal storage must be owner-only (mode 0700)")
    return path.resolve()


def equipment_recovery_summary(folder, config):
    """Read only small recovery receipts; source qualification belongs to the worker."""
    from weft.portal_perception_recovery import PROFILE, FIELDS, INPUTS, RECEIPT, SCOPE
    from weft.portal_equipment_candidates import PROFILE as EQUIPMENT_PROFILE
    from weft.shared_world_packaging import read_package_file
    from weft.exercise_body_source import _json

    if not any((folder/name).exists() or (folder/name).is_symlink()
               for name in (RECEIPT, 'equipment/recovery-start.json', 'equipment/recovery-inputs')):
        return None
    try:
        raw = read_package_file(folder, RECEIPT, max_bytes=64 << 10)
        receipt = _json(raw)
        if (set(receipt) != FIELDS or receipt.get('profile') != PROFILE or receipt.get('scope') != SCOPE
                or type(receipt.get('schema_version')) is not int or receipt['schema_version'] != 1
                or receipt.get('project_id') != config['id'] or receipt.get('source_sha256') != config['source_sha256']
                or any(type(receipt.get(k)) is not int or receipt[k] != 0 for k in ('api_calls', 'inference_calls'))
                or any(not isinstance(receipt.get(k), str) or not re.fullmatch('[a-f0-9]{64}', receipt[k])
                       for k in ('source_sha256', 'bridge_manifest_sha256', 'descriptor_sha256', 'request_sha256'))
                or not isinstance(receipt.get('inputs'), dict) or set(receipt['inputs']) != set(INPUTS)
                or any(not isinstance(pin, str) or not re.fullmatch('[a-f0-9]{64}', pin) for pin in receipt['inputs'].values())):
            raise ValueError('Recovery receipt differs from its closed source-bound contract')
        saved = {}
        for name, pin in receipt['inputs'].items():
            saved[name] = read_package_file(folder, 'equipment/recovery-inputs/'+name, max_bytes=64 << 10, expected_sha256=pin)
            # Explicit candidate selection changes the current root status, not the original receipt.
            if name != 'status.json':
                read_package_file(folder, name, max_bytes=64 << 10, expected_sha256=pin)
        old_status = _json(saved['status.json'])
        old_job = _json(saved['job.json'])
        if (old_status.get('status') != 'needs_input' or old_status.get('route') != 'exercise' or old_status.get('lab')
                or old_job.get('id') != config['id'] or old_job.get('source_sha256') != config['source_sha256']
                or _json(saved['perception/status.json']).get('status') != 'failed'
                or _json(saved['perception/status.json']) != _json(saved['perception/failure.json'])):
            raise ValueError('Recovery must retain the original unresolved project and failed preparation')
        request = _json(read_package_file(folder, 'equipment/request.json', max_bytes=16 << 10,
                                         expected_sha256=receipt['request_sha256']))
        if request != {'profile': EQUIPMENT_PROFILE, 'source_kind': 'perception',
                       'source_sha256': receipt['source_sha256'], 'descriptor_sha256': receipt['descriptor_sha256']}:
            raise ValueError('Recovery no longer names the original source-bound assembly request')
        read_package_file(folder, 'perception/prepared-source.json', max_bytes=16 << 10,
                          expected_sha256=receipt['descriptor_sha256'])
        start = folder/'equipment/recovery-start.json'
        started = start.exists() or start.is_symlink()
        if started:
            marker = _json(read_package_file(folder, 'equipment/recovery-start.json', max_bytes=8192))
            if (set(marker) != {'profile', 'started_at', 'api_calls', 'inference_calls'} or marker.get('profile') != PROFILE
                    or not isinstance(marker.get('started_at'), str) or not 1 <= len(marker['started_at']) <= 80
                    or any(type(marker.get(k)) is not int or marker[k] != 0 for k in ('api_calls', 'inference_calls'))):
                raise ValueError('Recovery start receipt differs from the recorded zero-provider operation')
        return {'status': 'recorded', 'started': started, 'receipt_sha256': hashlib.sha256(raw).hexdigest(),
                'artifact': RECEIPT, 'original_perception_status': 'failed', 'api_calls': 0, 'inference_calls': 0,
                'validation_scope': 'bounded_metadata_only'}
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return {'status': 'unavailable', 'reason': 'Recovery metadata is missing or changed; original failure and candidate records remain separate.'}


class Portal:
    def __init__(self, root, *, project=PROJECT_ROOT, start=True):
        self.root = private_directory(root)
        self.project = Path(project).resolve()
        self.budget = OpenAIBudget(project)
        # A stable per-root token keeps an open browser tab (and a judge's saved link)
        # working across restarts; it lives owner-only inside the already-private root.
        self.token = self._persistent_token()
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.stopping = threading.Event()
        self.labs = OrderedDict()
        self.active = None
        self.process = None
        self.lease = os.open(self.root / "portal.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(self.lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(self.lease)
            raise ValueError("Another portal owns this job directory") from None
        self.jobs = {}
        self.body_detail_children = {}
        self.supported_window_children = {}
        self.avatar_manager = None
        try:
            self.equipment_catalog = read_catalog(self.root)
            for folder in sorted(self.root.iterdir()):
                if not re.fullmatch(r"[a-f0-9]{32}", folder.name):
                    continue
                if folder.is_symlink() or not folder.is_dir() or len(self.jobs) >= MAX_JOBS:
                    raise ValueError("Portal inventory is unsafe or exceeds 50 jobs")
                config = bounded_json(folder / "job.json")
                self.jobs[folder.name] = config
                if "body_detail" in config and isinstance(config.get("parent_id"), str):
                    self.body_detail_children[config["parent_id"]] = folder.name
                if "supported_window" in config and isinstance(config.get("parent_id"), str):
                    self.supported_window_children[config["parent_id"]] = folder.name
                status = self._status(folder.name)
                if status["status"] not in TERMINAL:
                    atomic_json(folder / "status.json", {"status": "interrupted", "stage": "interrupted",
                        "events": status.get("events", []), "error": "Portal restarted before this job finished. Evidence and uncertain API charges are preserved; no automatic retry."})
                review = folder / "muse/status.json"
                if review.is_file() and bounded_json(review)["status"] not in REVIEW_TERMINAL:
                    atomic_json(review, {"status": "interrupted", "message": "Portal restarted during Muse review. No automatic retry; uncertain charges remain reserved."})
                checks = folder / "muse/checks/status.json"
                if checks.is_file() and bounded_json(checks)["status"] not in CHECK_TERMINAL:
                    atomic_json(checks, {"status": "interrupted", "message": "Portal restarted during local checks; partial traces preserved."})
                repairs = folder / "muse/repairs/status.json"
                if repairs.is_file() and bounded_json(repairs)["status"] not in REPAIR_TERMINAL:
                    atomic_json(repairs, {"status": "interrupted", "message": "Portal restarted during candidate testing; partial results preserved, no automatic retry.", "applied": False})
                equipment = equipment_status(folder)
                resume_recovery = False
                if equipment.get("status") == "queued" and (folder / "equipment/recovery.json").is_file():
                    from weft.portal_perception_recovery import validate_pending_recovery
                    resume_recovery = validate_pending_recovery(folder)
                if equipment and not resume_recovery and equipment.get("status") not in EQUIPMENT_TERMINAL:
                    atomic_json(folder / "equipment/status.json", {**equipment, "status": "interrupted",
                        "message": "Portal restarted during equipment construction; upload and partial candidates preserved. No automatic retry."})
                moving = moving_jobs.read_status(folder)
                if moving.get("status") in moving_jobs.ACTIVE:
                    atomic_json(folder / "moving-equipment/status.json", {**moving, "status": "interrupted",
                        "message": "Portal restarted during moving preparation; saved evidence and partial worlds retained. No automatic retry."})
                perception = perception_jobs.read_status(folder)
                resume_reuse = False
                if perception.get("status") == "queued_preflight" and "body_reuse" in config:
                    from weft.portal_body_reuse import validate_pending_reuse
                    resume_reuse = validate_pending_reuse(folder)
                if not resume_reuse and (perception.get("status") in perception_jobs.ACTIVE or perception.get("status") == "cancelling"):
                    atomic_json(folder / "perception/status.json", {**perception, "status": "interrupted",
                        "message": "Portal restarted during perception; source evidence and uncertain charges are preserved. No automatic retry."})
            from weft.portal_avatar_jobs import PortalAvatarManager
            self.avatar_manager = PortalAvatarManager(self.root / "avatars", self.project)
            from weft.portal_avatar_studio import AvatarStudioBridge
            self.avatar_studio = AvatarStudioBridge(self.root, self.project, self.prepare_avatars)
            from weft.portal_demo import DemoManager
            self.demo = DemoManager(self)
            self.thread = threading.Thread(target=self._work, name="weft-portal-queue", daemon=True) if start else None
            if self.thread:
                self.thread.start()
        except Exception:
            if self.avatar_manager is not None:
                self.avatar_manager.close()
            os.close(self.lease)
            raise

    def _folder(self, identity):
        if not re.fullmatch(r"[a-f0-9]{32}", identity) or identity not in self.jobs:
            raise ValueError("Unknown project")
        return self.root / identity

    def _status(self, identity):
        path = self._folder(identity) / "status.json"
        return bounded_json(path) if path.is_file() else {"status": "uploading", "stage": "upload"}

    def _stored_bytes(self):
        from weft.portal_import import _storage
        until = time.monotonic() + 2
        def bounded_inventory():
            if time.monotonic() >= until:
                raise ValueError("Portal storage inventory exceeded its bounded preparation window")
        return _storage(self.root, bounded_inventory)

    def _pending_jobs(self):
        return sum(self._status(i)["status"] not in TERMINAL
            or equipment_status(self._folder(i)).get("status") in {"queued", "running", "cancelling"}
            or moving_jobs.read_status(self._folder(i)).get("status") in moving_jobs.ACTIVE
            or perception_jobs.read_status(self._folder(i)).get("status") in perception_jobs.ACTIVE
            for i in self.jobs)

    def _reserved_growth(self):
        manager = getattr(self, "avatar_manager", None)
        total = manager.status().get("reserved_bytes", 0) if manager else 0
        for identity in self.jobs:
            folder = self._folder(identity)
            moving = moving_jobs.reserved_growth(folder)
            if moving:
                total += moving
            elif (self._status(identity)["status"] not in TERMINAL
                    or equipment_status(folder).get("status") in {"queued", "running", "cancelling"}
                    or perception_jobs.read_status(folder).get("status") in perception_jobs.ACTIVE):
                total += MAX_JOB_BYTES
        return total

    def _exploration(self, identity):
        return (self.jobs[identity].get("control_profile") == "shared_world_exploration"
                or moving_jobs.selected_profile(self._folder(identity), self._status(identity)) == "shared_world_exploration")

    def _placement(self, identity):
        return self.jobs[identity].get("control_profile") == "moving_equipment_task"

    def _spatial_import(self, identity):
        return (self.jobs[identity].get("kind") == "imported_lab"
                and self.jobs[identity].get("control_profile") == "spatial_rigid")

    def summary(self, identity):
        with self.lock:
            config = self.jobs[identity]
            folder = self._folder(identity)
            result = {k: config[k] for k in ("id", "name", "created_at", "ai", "mode", "source_sha256")}
            result["kind"] = config.get("kind", "video")
            result["exercise_intent"] = config.get("exercise_intent")
            if result["kind"] == "imported_lab":
                from weft.portal_evidence_attachment import attachment_summary

                attachment = ({"available": False, "reason": "Manual exploration has no qualified Muse review or repair contract"}
                    if config.get("control_profile") in {"shared_world_exploration", "moving_equipment_task"} else attachment_summary(folder, config))
                if config.get("control_profile") == "shared_world_exploration":
                    result["exploration"] = {"evaluation_status": "not_configured", "task_success_available": False,
                        "scope": "Manual conditional exploration; no exercise success or learned policy"}
                result["review_attachment"] = attachment
                result["review_unavailable_reason"] = None if attachment["available"] else attachment["reason"]
                result["learned_policy_available"] = config.get("learned_policy_available", False)
            if "inventory" in config:
                result["inventory"] = config["inventory"]
            result["parent_id"] = config.get("parent_id")
            result["source_available"] = bool(config.get("source_file") and (folder / config["source_file"]).is_file())
            status = self._status(identity)
            result.update(status)
            if result["kind"] == "task_job":
                from weft.portal_task_jobs import summary as task_summary
                result["task_job"] = task_summary(folder)
            from weft import portal_saved_body
            result["saved_body"] = (portal_saved_body.summary(folder) if result["kind"] == "saved_body"
                else portal_saved_body.availability(folder) if result["kind"] == "video"
                and status["status"] == "needs_input" and (folder / "perception/preflight-complete.json").is_file()
                else {"available": False})
            if result["kind"] == "saved_body":
                result["route"] = "exercise"
                result["review_scope"] = "conditional_body_authored_ground"
            result["perception"] = {**perception_jobs.summary(folder, config, status),
                "busy": self.active == identity and perception_jobs.read_status(folder).get("status") not in {"not_started", None}}
            result["body_detail"] = {"available": False}
            if "body_detail" in config or (result["perception"]["status"] == "failed"
                    and status.get("route") == "exercise" and status["status"] in {"needs_input", "failed"}):
                from weft.portal_body_detail import body_detail_availability
                result["body_detail"] = body_detail_availability(folder)
            detail_child = getattr(self, "body_detail_children", {}).get(identity)
            if detail_child:
                result["body_detail"].update(available=False, child_id=detail_child)
            result["supported_window"] = {"available": False}
            if "supported_window" in config or (result["perception"]["status"] == "failed"
                    and status.get("route") == "exercise" and status["status"] in {"needs_input", "failed"}):
                from weft.portal_supported_window import supported_window_availability
                result["supported_window"] = supported_window_availability(folder)
            window_child = getattr(self, "supported_window_children", {}).get(identity)
            if window_child:
                result["supported_window"].update(available=False, child_id=window_child)
            result["moving_equipment"] = moving_jobs.summary(folder)
            if self._exploration(identity):
                result["control_profile"] = "shared_world_exploration"
                result["review_unavailable_reason"] = "Manual exploration has no qualified Muse review, scoring or repair contract"
            if self._placement(identity):
                result["control_profile"] = "moving_equipment_task"
                result["review_unavailable_reason"] = "Authored placement tasks have no qualified Muse review or repair contract"
            result["task_tools"] = {"create_available": status["status"] == "ready" and self._exploration(identity),
                "learning_available": status["status"] == "ready" and self._placement(identity)}
            if self._spatial_import(identity):
                result["control_profile"] = "spatial_rigid"
                result["source_scope"] = config.get("source_scope")
                result["review_unavailable_reason"] = "Imported rigid worlds have no qualified Muse review or repair contract"
            result["equipment_availability"] = equipment_availability(config, status, self.equipment_catalog)
            equipment = equipment_status(folder)
            if equipment:
                result["equipment_candidates"] = {**equipment, "report": equipment_report(folder, equipment)}
                recovery = equipment_recovery_summary(folder, config)
                if recovery is not None:
                    result["equipment_candidates"]["recovery"] = recovery
            if status.get("equipment_candidate_id"):
                from weft.portal_perception_review import attachment_summary

                attachment = attachment_summary(folder)
                result["review_attachment"] = attachment
                result["review_unavailable_reason"] = None if attachment["available"] else attachment["reason"]
            inventory_path = folder / "analysis/scene-inventory.json"
            if inventory_path.is_file():
                inventory = bounded_json(inventory_path, maximum=8192)
                if inventory.get("source_sha256") == config["source_sha256"]:
                    result["scene_inventory"] = inventory
            if status["status"] == "preview":
                result["body_preview"] = {"body": "build/body_source/human.json", "model": "build/body_source/human.xml",
                    "evidence": "build/body_source/evidence.json", "clock": "build/body_source/native-clock.json",
                    "manifest": "build/body_source/manifest.json",
                    "image": next(("build/body_source/" + name for name in ("tracks.gif", "tracks-contact-sheet.png")
                                   if (folder / "build/body_source" / name).is_file()), None)}
            if status["status"] == "running":
                for name in ("lab", "planar"):
                    path = folder / name / "status.json"
                    if path.is_file():
                        child = bounded_json(path)
                        if child["status"] == "running":
                            result["stage"] = child.get("stage", status.get("stage"))
                            result["events"] = [*status.get("events", []), *child.get("events", [])][-60:]
            for name in ("plan", "receipt"):
                path = folder / "analysis" / (name + ".json")
                if path.is_file():
                    result[name] = bounded_json(path)
            if status["status"] == "failed":
                path = folder / "analysis/reconstruction-readiness-summary.json"
                try:
                    summary = bounded_json(path, maximum=8192)
                    if (summary.get("profile") == "portal-reconstruction-readiness-summary-v1"
                            and summary.get("source_sha256") == config["source_sha256"]):
                        result["reconstruction_readiness"] = summary
                except (OSError, ValueError, AttributeError):
                    pass
            result["sandbox"] = f"/labs/{identity}/" if status["status"] == "ready" else None
            result["support_experiment_available"] = (status["status"] == "ready" and status.get("route") == "exercise"
                and result["kind"] == "video" and not status.get("equipment_candidate_id") and not status.get("moving_candidate_id")
                and (body_path(folder, config) / "manifest.json").is_file())
            review = folder / "muse/status.json"
            if review.is_file():
                result["muse"] = bounded_json(review)
                result["muse"]["repair_availability"] = ({"available": False,
                    "reason": "Body approximations support review only; local repairs are not qualified"}
                    if result["kind"] == "saved_body" else repair_availability(folder))
                choice_supported = (result["kind"] in {"video", "saved_body", "support_experiment"}
                    and not status.get("equipment_candidate_id") and not status.get("moving_candidate_id"))
                result["muse"]["choice_availability"] = (repair_choice_availability(folder, model_choice=True)
                    if choice_supported else {"available": False, "reason": "This world has no qualified controller-choice adapter."})
                for field, name in (("receipt", "advisor-call"), ("proposal", "hypotheses")):
                    path = folder / "muse/advice" / (name + ".json")
                    if path.is_file():
                        try:
                            result["muse"][field] = bounded_json(path, maximum=140_000)
                        except json.JSONDecodeError:
                            if result["muse"]["status"] in REVIEW_TERMINAL:
                                raise
                checks = folder / "muse/checks/status.json"
                if checks.is_file():
                    result["muse"]["checks"] = bounded_json(checks)
                    report = folder / "muse/checks/report.json"
                    if result["muse"]["checks"]["status"] == "complete":
                        result["muse"]["checks"]["report"] = bounded_json(report, maximum=256_000)
                repairs = folder / "muse/repairs/status.json"
                if repairs.is_file():
                    result["muse"]["repairs"] = bounded_json(repairs)
                    if result["muse"]["repairs"].get("proposal_available"):
                        choice = bounded_json(folder / "muse/repairs/proposal/advice/hypotheses.json", maximum=140_000)
                        result["muse"]["repairs"]["choice_summary"] = choice.get("advice", {}).get("summary", "")
                    if result["muse"]["repairs"]["status"] == "complete":
                        result["muse"]["repairs"]["report"] = read_small_report(folder)
            return result

    def info(self):
        with self.lock:
            ids = list(self.jobs)
        from weft.portal_avatar_studio import list_library
        return {"provider": "openai", "model": MODEL, "credential": credential_status(self.project),
                "muse": muse_availability(self.project),
                "avatars": self.avatar_manager.status(),
                "avatar_library": list_library(self.project),
                "avatar_studio": self.avatar_studio.poll(),
                "budget": self.budget.status(), "local_only": True, "max_upload_bytes": MAX_UPLOAD,
                "jobs": [self.summary(identity) for identity in reversed(ids)],
                "limits": "60-second videos, 1,800 frames, one processing job at a time; failed reconstructions remain inspectable"}

    def upload(self, stream, size, *, name, mode, ai, request_id, exercise_inventory=None, exercise_output=None):
        if (type(size) is not int or not 1 <= size <= MAX_UPLOAD or type(ai) is not bool
                or mode not in ("auto", "exercise", "objects") or (mode == "auto" and not ai)):
            raise ValueError("Choose a supported scene type and a video up to 100 MB; auto mode needs AI")
        intent = exercise_intent(exercise_inventory, exercise_output)
        if mode == "objects" and (exercise_inventory is not None or exercise_output is not None):
            raise ValueError("Exercise inventory applies only to exercise or automatic scene selection")
        if (not re.fullmatch(r"[a-f0-9]{32}", request_id) or not isinstance(name, str)
                or not 1 <= len(name) <= 180 or Path(name).name != name or any(ord(c) < 32 for c in name)
                or Path(name).suffix.lower() not in {".mp4", ".mov", ".webm", ".m4v", ".mkv", ".avi"}):
            raise ValueError("Invalid video name or upload identifier")
        with self.lock:
            existing = next((j for j in self.jobs.values() if j.get("request_id") == request_id), None)
            if existing:
                raise ValueError("This upload identifier already exists; reload the project list instead of resubmitting")
            pending = self._pending_jobs()
            if len(self.jobs) >= MAX_JOBS or pending >= 4:
                raise ValueError("Portal job capacity reached; preserve existing projects before making room")
            if ai and (not credential_status(self.project)["configured"]
                       or self.budget.status()["remaining_usd"] < self.budget.status()["reservation_usd"]):
                raise ValueError("OpenAI key or remaining allowance is unavailable; no video uploaded")
            stored = self._stored_bytes()
            # Reserve expansion for every queued job so parallel uploads cannot overcommit disk.
            if stored + self._reserved_growth() + MAX_JOB_BYTES > MAX_STORAGE:
                raise ValueError("The portal's 2 GB local storage allowance is full")
            identity = uuid4().hex
            folder = self.root / identity
            folder.mkdir(mode=0o700)
            filename = "source" + Path(name).suffix.lower()
            config = {"id": identity, "request_id": request_id, "name": name, "source_file": filename,
                      "source_sha256": None, "created_at": local_time(), "ai": ai, "mode": mode, "episodes": 300,
                      **({"exercise_intent": intent} if mode != "objects" else {})}
            self.jobs[identity] = config
            write_json(folder / "job.json", config)
            atomic_json(folder / "status.json", {"status": "uploading", "stage": "upload", "events": []})
        try:
            sha, remaining = hashlib.sha256(), size
            with (folder / filename).open("xb") as target:
                while remaining:
                    chunk = stream.read(min(65536, remaining))
                    if not chunk:
                        raise ValueError("Video upload was interrupted")
                    target.write(chunk)
                    sha.update(chunk)
                    remaining -= len(chunk)
                target.flush()
                os.fsync(target.fileno())
            with self.lock:
                config["source_sha256"] = sha.hexdigest()
                write_json(folder / "job.json", config)
                cancelled = (folder / "cancel-request.json").exists()
                atomic_json(folder / "status.json", {"status": "cancelled" if cancelled else "queued", "stage": "queued", "events": [
                    {"stage": "upload", "message": "Original video saved; waiting for the processing slot", "time": local_time()}]})
            self.wake.set()
            return self.summary(identity)
        except Exception:
            atomic_json(folder / "status.json", {"status": "failed", "stage": "upload", "error": "Upload incomplete; partial bytes preserved", "events": []})
            raise

    def _work(self):
        while not self.stopping.is_set():
            with self.lock:
                identity = next((i for i in self.jobs if self._status(i)["status"] == "queued"), None)
                review = False
                checks = False
                repairs = False
                equipment = False
                perception = False
                moving = False
                if identity is None:
                    identity = next((i for i in self.jobs if self._review_status(i).get("status") == "queued"), None)
                    review = identity is not None
                if identity is None:
                    identity = next((i for i in self.jobs if self._check_status(i).get("status") == "queued"), None)
                    checks = identity is not None
                if identity is None:
                    identity = next((i for i in self.jobs if self._repair_status(i).get("status") == "queued"), None)
                    repairs = identity is not None
                if identity is None:
                    identity = next((i for i in self.jobs if equipment_status(self._folder(i)).get("status") == "queued"), None)
                    equipment = identity is not None
                if identity is None:
                    identity = next((i for i in self.jobs if perception_jobs.read_status(self._folder(i)).get("status") in {"queued_preflight", "queued"}), None)
                    perception = identity is not None
                if identity is None:
                    identity = next((i for i in self.jobs if moving_jobs.read_status(self._folder(i)).get("status") == "queued"), None)
                    moving = identity is not None
                if identity:
                    self.active = identity
                    target = self._folder(identity) / ("moving-equipment/status.json" if moving else "equipment/status.json" if equipment else "muse/repairs/status.json" if repairs else "muse/checks/status.json" if checks else "muse/status.json" if review else "status.json")
                    if not repairs and not perception and not moving and self.jobs[identity].get("kind") not in {"task_job", "saved_body"}:
                        atomic_json(target, {"status": "running", "stage": "starting", "events": []})
            if identity is None:
                self.wake.wait(1)
                self.wake.clear()
                continue
            self._run(identity, review=review, checks=checks, **({"repairs": True} if repairs else {}),
                **({"equipment": True} if equipment else {}), **({"perception": True} if perception else {}), **({"moving": True} if moving else {}))
            with self.lock:
                self.active, self.process = None, None

    def support_experiment(self, identity, raw):
        from weft.exercise_support_inputs import validate_support_priors
        from weft.shared_world_human import _json

        if not isinstance(raw, bytes) or not 1 <= len(raw) <= 128_000:
            raise ValueError("Support request must be bounded JSON")
        request = _json(raw)
        if (set(request) != {"request_id", "priors", "train"} or type(request["train"]) is not bool
                or not isinstance(request["request_id"], str) or not re.fullmatch(r"[a-f0-9]{32}", request["request_id"])):
            raise ValueError("Invalid support request fields")
        priors_raw = json.dumps(request["priors"], sort_keys=True, allow_nan=False).encode()
        priors = validate_support_priors(priors_raw)
        request_sha = hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()
        with self.lock:
            parent = self.jobs.get(identity)
            if parent is None or not self.summary(identity)["support_experiment_available"]:
                raise ValueError("Start a support experiment from a completed original exercise project")
            if priors["source_sha256"] != parent["source_sha256"]:
                raise ValueError("Support assumptions refer to a different source video")
            existing = next((j for j in self.jobs.values() if j.get("request_id") == request["request_id"]), None)
            if existing:
                if existing.get("parent_id") != identity or existing.get("support_request_sha256") != request_sha:
                    raise ValueError("This request identifier belongs to a different experiment")
                return self.summary(existing["id"])
            pending = self._pending_jobs()
            if len(self.jobs) >= MAX_JOBS or pending >= 4:
                raise ValueError("Portal job capacity reached")
            stored = self._stored_bytes()
            if stored + self._reserved_growth() + MAX_JOB_BYTES > MAX_STORAGE:
                raise ValueError("The portal's 2 GB local storage allowance is full")
            source = body_path(self._folder(identity), parent)
            manifest = source / "manifest.json"
            if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 1_000_000:
                raise ValueError("Parent body manifest is missing or unsafe")
            child_id = uuid4().hex
            folder = self.root / child_id
            folder.mkdir(mode=0o700)
            config = {"id": child_id, "request_id": request["request_id"], "name": parent["name"],
                "source_file": parent["source_file"], "source_sha256": parent["source_sha256"],
                "created_at": local_time(), "ai": False, "mode": "exercise", "kind": "support_experiment",
                "parent_id": identity, "parent_body_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "body_source_directory": "body_source", "support_request_sha256": request_sha,
                "support_priors_sha256": hashlib.sha256(priors_raw).hexdigest(), "train": request["train"]}
            write_json(folder / "job.json", config)
            (folder / "support-request.json").write_bytes(raw)
            (folder / "support-priors.json").write_bytes(priors_raw)
            atomic_json(folder / "status.json", {"status": "queued", "stage": "queued", "events": [
                {"stage": "upload", "message": "New authored-support experiment; the original project is unchanged", "time": local_time()}]})
            self.jobs[child_id] = config
            self.wake.set()
            return self.summary(child_id)

    def _review_status(self, identity):
        path = self._folder(identity) / "muse/status.json"
        return bounded_json(path) if path.is_file() else {}

    def _check_status(self, identity):
        path = self._folder(identity) / "muse/checks/status.json"
        return bounded_json(path) if path.is_file() else {}

    def _repair_status(self, identity):
        path = self._folder(identity) / "muse/repairs/status.json"
        return bounded_json(path) if path.is_file() else {}

    def moving_equipment(self, identity, action, payload):
        with self.lock:
            folder = self._folder(identity)
            if identity in self.labs:
                raise ValueError("An opened sandbox cannot change moving assumptions or selection")
            if action == "select":
                if not isinstance(payload, dict) or set(payload) != {"id"} or not isinstance(payload["id"], str):
                    raise ValueError("Choose one admitted candidate identifier")
                moving_jobs.select_candidate(folder, payload["id"])
            else:
                if action not in {"prepare", "build"}:
                    raise ValueError("Unknown moving action")
                request_path = folder / "moving-equipment" / ("request.json" if action == "prepare" else "build-request.json")
                if not request_path.exists():
                    if self.active == identity or self._pending_jobs() >= 4:
                        raise ValueError("Finish active work before queuing another moving phase")
                    from weft.portal_import import _storage
                    until = time.monotonic() + 2
                    def check():
                        if time.monotonic() > until:
                            raise ValueError("Moving storage inventory exceeded its bounded preparation window")
                    count = len(payload.get("choices", [])) if isinstance(payload, dict) and action == "build" else 0
                    if action == "build" and not 1 <= count <= 3:
                        raise ValueError("Choose one to three moving hypotheses")
                    growth = moving_jobs.RECEIPT_BYTES if action == "prepare" else moving_jobs.RECEIPT_BYTES + count * (256 << 20)
                    if _storage(folder, check) + growth > moving_jobs.MAX_JOB_BYTES or self._stored_bytes() + self._reserved_growth() + growth > MAX_STORAGE:
                        raise ValueError("Moving preparation exceeds its scoped job or total portal storage allowance")
                if action == "prepare":
                    moving_jobs.queue_prepare(folder, payload)
                else:
                    moving_jobs.queue_build(folder, payload)
                self.wake.set()
            return self.summary(identity)

    def prepare_equipment(self, identity, entry_id):
        from weft.io import canonical_hash
        from weft.portal_equipment_candidates import PROFILE
        from weft.shared_world_packaging import write_package_files

        with self.lock:
            root = self._folder(identity)
            if equipment_status(root):
                return self.summary(identity)
            catalog = read_catalog(self.root)
            available = equipment_availability(self.jobs[identity], self._status(identity), catalog)
            if not available["available"] or entry_id not in {e["id"] for e in available["matches"]}:
                raise ValueError("No registered prepared evidence matches this upload")
            if sum(equipment_status(self._folder(i)).get("status") in {"queued", "running", "cancelling"} for i in self.jobs) >= 4:
                raise ValueError("Equipment candidate queue is full")
            pending = self._pending_jobs()
            stored = self._stored_bytes()
            if stored + self._reserved_growth() + MAX_JOB_BYTES > MAX_STORAGE:
                raise ValueError("Equipment candidate storage reservation exceeds the portal allowance")
            entry = next(e for e in catalog["entries"] if e["id"] == entry_id)
            work = root / "equipment"
            work.mkdir(mode=0o700)
            request = {"profile": PROFILE, "source_sha256": self.jobs[identity]["source_sha256"],
                "entry": entry, "entry_sha256": canonical_hash(entry), "requested_at": local_time()}
            write_package_files(work, {"parent-status.json": (root / "status.json").read_bytes(),
                "request.json": (json.dumps(request, sort_keys=True, indent=2) + "\n").encode()})
            atomic_json(work / "status.json", {"status": "queued", "message": "Waiting to validate prepared evidence and build conditional equipment candidates. No AI call or training."})
            self.wake.set()
            return self.summary(identity)

    def choose_equipment(self, identity, candidate_id):
        with self.lock:
            if identity in self.labs:
                raise ValueError("A project with an opened sandbox cannot switch equipment candidates")
            select_candidate(self._folder(identity), candidate_id)
            return self.summary(identity)

    def preflight_perception(self, identity):
        with self.lock:
            folder = self._folder(identity)
            if perception_jobs.read_status(folder):
                perception_jobs.queue_preflight(folder)
                return self.summary(identity)
            if equipment_availability(self.jobs[identity], self._status(identity), self.equipment_catalog)["available"]:
                raise ValueError("Verified prepared equipment evidence is already available; use its existing candidate path")
            self._reserve_perception_slot()
            perception_jobs.queue_preflight(folder)
            self.wake.set()
            return self.summary(identity)

    def prepare_perception(self, identity, payload):
        with self.lock:
            folder = self._folder(identity)
            if (folder / "perception/request.json").is_file():
                perception_jobs.queue_prepare(folder, payload, project=self.project)
                return self.summary(identity)
            if self.active == identity:
                raise ValueError("The local worker is finishing; refresh before starting the separately consented preparation")
            self._reserve_perception_slot()
            perception_jobs.queue_prepare(folder, payload, project=self.project)
            self.wake.set()
            return self.summary(identity)

    def prepare_body_detail(self, identity, payload):
        from weft.portal_body_detail import create_body_detail_child

        with self.lock:
            self._folder(identity)
            if self.active == identity:
                raise ValueError("The original worker is finishing; refresh before starting a separate body attempt")
            config = create_body_detail_child(self.root, identity, payload, lease_fd=self.lease,
                existing_jobs=self.jobs, stored_bytes=self._stored_bytes(), reserved_bytes=self._reserved_growth(),
                pending_jobs=self._pending_jobs(), deadline=time.monotonic()+2)
            self.jobs[config["id"]] = config
            self.body_detail_children[identity] = config["id"]
            self.wake.set()
            return self.summary(config["id"])

    def prepare_supported_window(self, identity, payload):
        from weft.portal_supported_window import create_supported_window_child

        with self.lock:
            self._folder(identity)
            if self.active == identity:
                raise ValueError("The original worker is finishing; refresh before requesting the window attempt")
            config = create_supported_window_child(self.root, identity, payload, lease_fd=self.lease,
                existing_jobs=self.jobs, stored_bytes=self._stored_bytes(), reserved_bytes=self._reserved_growth(),
                pending_jobs=self._pending_jobs(), deadline=time.monotonic()+2)
            self.jobs[config["id"]] = config
            self.supported_window_children[identity] = config["id"]
            self.wake.set()
            return self.summary(config["id"])

    def task_options(self, identity):
        with self.lock:
            if self._status(identity)["status"] != "ready" or not (self._exploration(identity) or self._placement(identity)):
                raise ValueError("Task settings require a ready moving exploration or authored placement world")
            return self.lab(identity).task_options()

    def prepare_saved_body(self, identity, payload):
        from weft.portal_saved_body import queue_saved_body_child

        with self.lock:
            self._folder(identity)
            if self.active == identity:
                raise ValueError("The source worker is finishing; refresh before creating a separate sandbox")
            config = queue_saved_body_child(self.root, identity, payload, lease_fd=self.lease,
                existing_jobs=self.jobs, stored_bytes=self._stored_bytes(), reserved_bytes=self._reserved_growth(),
                pending_jobs=self._pending_jobs(), deadline=time.monotonic()+2)
            self.jobs[config["id"]] = config
            self.wake.set()
            return self.summary(config["id"])

    def task_action(self, identity, operation, payload):
        from weft.portal_task_jobs import create_task_child, create_learning_child

        if operation not in {"create", "train"}:
            raise ValueError("Unknown task action")
        with self.lock:
            self._folder(identity)
            if self.active == identity:
                raise ValueError("The parent worker is finishing; refresh before creating a separate task job")
            lab = self.labs.get(identity)
            if lab is not None and (getattr(lab, "loading", False) or lab.recorder is not None
                    or (lab.export_future is not None and not lab.export_future.done())):
                raise ValueError("Finish loading, recording or exporting the parent sandbox first")
            create = create_task_child if operation == "create" else create_learning_child
            config = create(self.root, identity, payload, lease_fd=self.lease, existing_jobs=self.jobs,
                stored_bytes=self._stored_bytes(), reserved_bytes=self._reserved_growth(),
                pending_jobs=self._pending_jobs(), deadline=time.monotonic()+2)
            self.jobs[config["id"]] = config
            self.wake.set()
            return self.summary(config["id"])

    def _reserve_perception_slot(self):
        pending = self._pending_jobs()
        if pending >= 4:
            raise ValueError("The shared processing queue is full")
        if self._stored_bytes() + self._reserved_growth() + MAX_JOB_BYTES > MAX_STORAGE:
            raise ValueError("Perception preparation exceeds the portal storage reservation")

    def repair(self, identity, *, model_choice=False):
        with self.lock:
            root = self._folder(identity)
            self._require_review_body(identity, action="repair_choice" if model_choice else "repairs")
            folder = root / "muse/repairs"
            if folder.exists():
                return self.summary(identity)
            available = repair_choice_availability(root, model_choice=True) if model_choice else repair_availability(root)
            if not available["available"]:
                raise ValueError(available["reason"])
            if model_choice and not muse_availability(self.project)["available"]:
                raise ValueError("Meta project key or remaining allowance is unavailable")
            if sum(self._repair_status(i).get("status") in {"queued", "running"} for i in self.jobs) >= 4:
                raise ValueError("Candidate comparison queue is full")
            folder.mkdir(mode=0o700)
            if model_choice:
                atomic_json(folder / "request.json", {"schema_version": 1, "profile": "muse-controller-choice-v1"})
            message = ("Waiting for one bounded Muse controller choice and fixed tests; no automatic application."
                       if model_choice else "Waiting to test bounded suggestions; no new AI request and no automatic application.")
            atomic_json(folder / "status.json", {"status": "queued", "message": message,
                "requested_at": local_time(), "applied": False})
            self.wake.set()
            return self.summary(identity)

    def check(self, identity):
        with self.lock:
            folder = self._folder(identity) / "muse/checks"
            self._require_review_body(identity, action="checks")
            if folder.exists():
                return self.summary(identity)
            if self._status(identity)["status"] != "ready" or self._review_status(identity).get("status") != "proposed":
                raise ValueError("Local checks require a completed simulation and Muse review")
            if sum(self._check_status(i).get("status") in {"queued", "running"} for i in self.jobs) >= 4:
                raise ValueError("Local check queue is full")
            folder.mkdir(mode=0o700)
            atomic_json(folder / "status.json", {"status": "queued", "message": "Waiting for the shared processing slot", "requested_at": local_time()})
            self.wake.set()
            return self.summary(identity)

    def review(self, identity):
        with self.lock:
            folder = self._folder(identity)
            self._require_review_body(identity)
            if (folder / "muse").exists():
                return self.summary(identity)
            if self._status(identity)["status"] != "ready":
                raise ValueError("Finish reconstruction before requesting Muse review")
            if not muse_availability(self.project)["available"]:
                raise ValueError("Meta project key or remaining allowance is unavailable")
            if sum(self._review_status(i).get("status") in {"queued", "running"} for i in self.jobs) >= 4:
                raise ValueError("Muse review queue is full")
            (folder / "muse").mkdir(mode=0o700)
            atomic_json(folder / "muse/status.json", {"status": "queued", "message": "Waiting for the shared processing slot", "requested_at": local_time()})
            self.wake.set()
            return self.summary(identity)

    def _require_review_body(self, identity, *, action="review"):
        if self.jobs[identity].get("kind") == "saved_body" and action not in {"review", "repair_choice"}:
            raise ValueError("Body approximations support Muse review only; local checks and repairs are not qualified")
        if self.jobs[identity].get("kind") == "task_job" and not self._placement(identity):
            raise ValueError("Task-learning reports have no qualified Muse review or repair contract")
        if self._spatial_import(identity):
            raise ValueError("Imported rigid worlds have no qualified Muse review or repair contract")
        if self._placement(identity):
            raise ValueError("Authored placement tasks have no qualified Muse review or repair contract")
        if self._exploration(identity):
            raise ValueError("Manual exploration has no qualified Muse review, scoring or repair contract")
        if self._status(identity).get("equipment_candidate_id"):
            from weft.portal_perception_review import attachment_summary

            attached = attachment_summary(self._folder(identity))
            if not attached["available"]:
                raise ValueError(attached["reason"])
            if action != "review":
                raise ValueError("Conditional equipment attachments enable Muse review only; local checks and repairs are not qualified")
        if self.jobs[identity].get("kind") == "imported_lab":
            from weft.portal_evidence_attachment import attachment_summary

            attached = attachment_summary(self._folder(identity), self.jobs[identity])
            if not attached["available"]:
                raise ValueError(attached["reason"])
            if action != "review":
                raise ValueError("Imported construction attachments enable Muse review only; local checks and repairs are not qualified")

    def _run(self, identity, *, review=False, checks=False, repairs=False, equipment=False, perception=False, moving=False):
        folder = self._folder(identity)
        work = folder / "moving-equipment" if moving else folder / "perception" if perception else folder / "equipment" if equipment else folder / "muse/repairs" if repairs else folder / "muse/checks" if checks else folder / "muse" if review else folder
        terminal = moving_jobs.TERMINAL if moving else perception_jobs.RESTING if perception else EQUIPMENT_TERMINAL if equipment else REPAIR_TERMINAL if repairs else CHECK_TERMINAL if checks else REVIEW_TERMINAL if review else TERMINAL
        paid_perception = perception and (work / "request.json").is_file()
        try:
            paid_repair = repairs and choice_requested(folder)
        except (OSError, ValueError) as error:
            atomic_json(work / "status.json", {"status": "failed", "message": str(error),
                "applied": False, "partial_evidence_preserved": True})
            return
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
               "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLCONFIGDIR": str(folder / "runtime/matplotlib")}
        support = self.jobs[identity].get("kind") == "support_experiment" and not (review or checks or repairs or equipment or perception or moving)
        tasks = self.jobs[identity].get("kind") == "task_job" and not (review or checks or repairs or equipment or perception or moving)
        saved_body = self.jobs[identity].get("kind") == "saved_body" and not (review or checks or repairs or equipment or perception or moving)
        if (paid_perception or not (review or checks or repairs or support or tasks or saved_body or equipment or perception or moving)) and os.environ.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        if review or paid_perception or paid_repair:
            for name in ("META_API_KEY", "MODEL_API_KEY"):
                if os.environ.get(name):
                    env[name] = os.environ[name]
        process = None
        reason = None
        output_thread = None
        output_limit = threading.Event()
        try:
            log_name = ("preparation-worker.log" if paid_perception else "preflight-worker.log") if perception else "worker.log"
            if moving:
                log_name = moving_jobs.read_status(folder)["phase"] + "-worker.log"
            with (work / log_name).open("xb") as log:
                mode = ["moving"] if moving else ["perception"] if perception else ["equipment"] if equipment else ["repairs"] if repairs else ["checks"] if checks else ["review"] if review else ["supports"] if support else ["tasks"] if tasks else ["saved-body"] if saved_body else []
                process = subprocess.Popen([sys.executable, "-P", "-B", "-m", "weft.portal_worker", str(folder), str(self.project), *mode],
                                           stdin=subprocess.DEVNULL, stdout=subprocess.PIPE if moving else log, stderr=subprocess.STDOUT, env=env)
                if moving:
                    def capture_output():
                        remaining = 8 << 20
                        while True:
                            block = process.stdout.read(65536)
                            if not block:
                                break
                            log.write(block[:remaining])
                            log.flush()
                            remaining -= min(remaining, len(block))
                            if remaining == 0:
                                output_limit.set()
                                break
                    output_thread = threading.Thread(target=capture_output, name="weft-moving-log", daemon=True)
                    output_thread.start()
                with self.lock:
                    self.process = process
                deadline = time.monotonic() + (195 if moving or tasks else 135 if saved_body else 450 if perception else 120 if equipment else 125 if paid_repair else 65 if repairs else 60 if checks else 90 if review else 420)
                next_inventory = 0
                while process.poll() is None:
                    if moving and output_limit.is_set():
                        reason = "Moving worker reached its 8 MiB output allowance"
                        break
                    if self.stopping.wait(.5):
                        reason = "Portal stopped; no automatic retry"
                        break
                    if (work / "cancel-request.json").exists():
                        reason = "Cancelled by user; partial evidence and uncertain charges preserved"
                        break
                    if time.monotonic() >= deadline:
                        reason = "Processing exceeded its wall-time limit"
                        break
                    if time.monotonic() >= next_inventory:
                        next_inventory = time.monotonic() + 2
                        files = list(folder.rglob("*"))
                        limit = moving_jobs.storage_limit(folder) if moving else MAX_JOB_BYTES
                        if len(files) > 6000 or sum(p.stat().st_size for p in files if p.is_file()) > limit:
                            reason = "Processing reached its local artifact storage limit"
                            break
                if reason and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                process.wait()
                if output_thread:
                    output_thread.join(timeout=2)
                    process.stdout.close()
                current = bounded_json(work / "status.json")
                if current["status"] not in terminal:
                    state = "cancelled" if (work / "cancel-request.json").exists() else "interrupted" if self.stopping.is_set() else "failed"
                    atomic_json(work / "status.json", {**current, "status": state,
                        "error": reason or f"Worker exited ({process.returncode}); inspect saved artifacts"})
        except Exception as error:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if output_thread:
                output_thread.join(timeout=2)
                if process is not None and process.stdout is not None:
                    process.stdout.close()
            current = bounded_json(work / "status.json")
            if current["status"] not in terminal:
                atomic_json(work / "status.json", {**current, "status": "failed", "error": str(error)[:500]})
        finally:
            if perception:
                with self.lock:
                    self._cancel_perception_handoff(identity)

    def _cancel_perception_handoff(self, identity):
        folder = self._folder(identity)
        equipment = equipment_status(folder)
        if (folder / "perception/cancel-request.json").exists() and equipment.get("status") == "queued":
            request = bounded_json(folder / "equipment/request.json")
            if request.get("source_kind") == "perception":
                atomic_json(folder / "equipment/cancel-request.json", {"requested_at": local_time()})
                atomic_json(folder / "equipment/status.json", {**equipment, "status": "cancelled",
                    "message": "Cancelled during perception handoff; prepared evidence is preserved."})

    def cancel(self, identity):
        with self.lock:
            perception = perception_jobs.read_status(self._folder(identity))
            if perception.get("status") in perception_jobs.ACTIVE or perception.get("status") == "cancelling":
                folder = self._folder(identity) / "perception"
                atomic_json(folder / "cancel-request.json", {"requested_at": local_time()})
                atomic_json(folder / "status.json", {**perception,
                    "status": "cancelling" if self.active == identity else "cancelled",
                    "message": "Perception cancelled; original source, local evidence and any uncertain provider charges are preserved."})
                self._cancel_perception_handoff(identity)
                return self.summary(identity)
            moving = moving_jobs.read_status(self._folder(identity))
            if moving.get("status") in moving_jobs.ACTIVE:
                work = self._folder(identity) / "moving-equipment"
                atomic_json(work / "cancel-request.json", {"requested_at": local_time()})
                atomic_json(work / "status.json", {**moving, "status": "cancelling" if self.active == identity else "cancelled", "message": "Moving work cancelled; original source and partial alternatives retained."})
                return self.summary(identity)
            equipment = equipment_status(self._folder(identity))
            if equipment and equipment.get("status") not in EQUIPMENT_TERMINAL:
                folder = self._folder(identity) / "equipment"
                atomic_json(folder / "cancel-request.json", {"requested_at": local_time()})
                atomic_json(folder / "status.json", {**equipment,
                    "status": "cancelling" if equipment.get("status") in {"running", "cancelling"} else "cancelled",
                    "message": "Equipment preparation cancelled; parent upload and partial candidates preserved."})
                return self.summary(identity)
            repairs = self._repair_status(identity)
            queued_repair_next = (repairs.get("status") == "queued"
                and self._check_status(identity).get("status") not in {"queued", "running", "cancelling"}
                and self._review_status(identity).get("status") in REVIEW_TERMINAL)
            if repairs.get("status") in {"running", "cancelling"} or queued_repair_next:
                folder = self._folder(identity) / "muse/repairs"
                atomic_json(folder / "cancel-request.json", {"requested_at": local_time()})
                atomic_json(folder / "status.json", {**repairs,
                    "status": "cancelling" if repairs["status"] in {"running", "cancelling"} else "cancelled",
                    "message": "Candidate testing cancelled; original simulation and partial results preserved.", "applied": False})
                return self.summary(identity)
            checks = self._check_status(identity)
            if checks and checks["status"] not in CHECK_TERMINAL:
                folder = self._folder(identity) / "muse/checks"
                atomic_json(folder / "cancel-request.json", {"requested_at": local_time()})
                atomic_json(folder / "status.json", {**checks, "status": "cancelling" if self.active == identity else "cancelled",
                    "message": "Local checks cancelled; original simulation and partial traces preserved."})
                return self.summary(identity)
            review = self._review_status(identity)
            if review and review["status"] not in REVIEW_TERMINAL:
                folder = self._folder(identity) / "muse"
                atomic_json(folder / "cancel-request.json", {"requested_at": local_time()})
                atomic_json(folder / "status.json", {**review, "status": "cancelling" if self.active == identity else "cancelled",
                    "message": "Review cancelled; uncertain charges stay reserved. The simulation is unchanged."})
                return self.summary(identity)
            status = self._status(identity)
            if status["status"] in TERMINAL:
                return self.summary(identity)
            atomic_json(self._folder(identity) / "cancel-request.json", {"requested_at": local_time()})
            state = "cancelling" if self.active == identity else "cancelled"
            atomic_json(self._folder(identity) / "status.json", {**status, "status": state, "stage": state,
                "error": "Cancelled; retained evidence and any uncertain API charges remain available"})
        return self.summary(identity)

    def prepare_avatars(self, raw=None):
        with self.lock:
            reserve = self.avatar_manager.reserve_for(raw)
            if self._stored_bytes() + self._reserved_growth() + reserve > MAX_STORAGE:
                raise ValueError("Avatar preparation would exceed portal storage capacity")
            return self.avatar_manager.submit(raw)

    def attach_avatars(self, identity):
        with self.lock:
            self._folder(identity)
            if identity not in self.labs or getattr(self.labs[identity], "loading", False):
                raise ValueError("Open this sandbox first, wait for its body to load, then return here to add characters")
            lab = self.lab(identity)
            return self.avatar_manager.attach_async(identity, lab)

    def lab(self, identity):
        with self.lock:
            if self._status(identity)["status"] != "ready":
                raise ValueError("Sandbox is not ready; inspect this project's progress and failure details")
            if identity not in self.labs:
                if len(self.labs) >= 3:
                    attachments = self.avatar_manager.status()["attachments"]
                    idle = next((key for key, lab in self.labs.items() if not getattr(lab, "loading", False) and lab.recorder is None
                                 and attachments.get(key, {}).get("status") != "running"
                                 and (lab.export_future is None or lab.export_future.done())), None)
                    if idle is None:
                        raise ValueError("Finish recording in another sandbox before opening a fourth")
                    evicted = self.labs.pop(idle)
                    (self.avatar_manager.attached(idle) or evicted).close()
                folder = self._folder(identity)
                lab_path = (folder / self._status(identity).get("lab", "lab")).resolve()
                if not lab_path.is_relative_to(folder):
                    raise ValueError("Sandbox path is outside this project")
                if self._placement(identity):
                    from weft.portal_task_options import LoadingTaskLearningLab
                    lab = LoadingTaskLearningLab(lab_path, project=self.project)
                elif self._spatial_import(identity):
                    from weft.portal_spatial_lab import LoadingSpatialPortalLab
                    lab = LoadingSpatialPortalLab(lab_path, project=self.project)
                elif self._exploration(identity):
                    from weft.portal_task_options import LoadingTaskAuthoringLab
                    lab = LoadingTaskAuthoringLab(lab_path, project=self.project)
                else:
                    lab = open_portal_lab(lab_path, project=self.project)
                lab.token = self.token
                self.labs[identity] = lab
            self.labs.move_to_end(identity)
            manager = getattr(self, "avatar_manager", None)
            return (manager.attached(identity) if manager else None) or self.labs[identity]

    def artifact(self, identity, name):
        folder = self._folder(identity)
        if name.startswith("saved-body/"):
            from weft.portal_saved_body import ARTIFACTS
            from weft.shared_world_packaging import read_package_file
            relative = name.removeprefix("saved-body/")
            if self.jobs[identity].get("kind") != "saved_body" or relative not in ARTIFACTS:
                raise ValueError("File is not a public saved-body artifact")
            read_package_file(folder / "saved-body", relative, max_bytes=16 << 20)
            return folder / name
        if name.startswith("task-job/"):
            from weft.portal_task_jobs import ARTIFACTS
            from weft.shared_world_packaging import read_package_file
            relative = name.removeprefix("task-job/")
            if self.jobs[identity].get("kind") != "task_job" or relative not in ARTIFACTS:
                raise ValueError("File is not a public task artifact")
            read_package_file(folder / "task-job", relative, max_bytes=(4 << 20) + 1)
            return folder / name
        if name.startswith("supported-window/"):
            if name != "supported-window/request.json":
                raise ValueError("File is not a public supported-window artifact")
        if name.startswith("body-detail/"):
            if name not in {"body-detail/request.json", "body-detail/recipe.json"}:
                raise ValueError("File is not a public body-detail artifact")
            from weft.shared_world_packaging import read_package_file
            read_package_file(folder, name, max_bytes=(2 << 20) if name.endswith("recipe.json") else (128 << 10))
            return folder / name
        if name == 'equipment/recovery.json':
            from weft.shared_world_packaging import read_package_file
            recovery = equipment_recovery_summary(folder, self.jobs[identity])
            if not recovery or recovery.get('status') != 'recorded':
                raise ValueError('Recovery receipt is unavailable or no longer matches this project')
            read_package_file(folder, name, max_bytes=64 << 10, expected_sha256=recovery['receipt_sha256'])
            return folder/name
        if name.startswith("moving-equipment/"):
            relative = name.removeprefix("moving-equipment/")
            if relative not in moving_jobs.ARTIFACTS:
                raise ValueError("Moving artifact is not in the approved receipt inventory")
            from weft.shared_world_packaging import read_package_file
            read_package_file(folder / "moving-equipment", relative, max_bytes=moving_jobs.MAX_FAMILY_BYTES)
            return folder / name
        if name.startswith("perception/"):
            if name not in PERCEPTION_ARTIFACTS and not re.fullmatch(r"perception/analysis/frame-\d{5}\.jpg", name):
                raise ValueError("File is not a public perception artifact")
            file = folder / name
            if any(p.is_symlink() for p in (file, *file.parents)) or not file.is_file():
                raise ValueError("Perception artifact is missing or unsafe")
            if file.stat().st_size > ((16 << 20) if name.startswith("perception/preflight/body_source/") or name == "perception/pose-diagnostic/report.json" else 300_000 if file.suffix == ".jpg" else 1 << 20):
                raise ValueError("Perception artifact exceeds its bounded download size")
            return file
        if name.startswith("review-evidence/"):
            from weft.portal_evidence_attachment import attachment_artifact

            return attachment_artifact(folder, self.jobs[identity], name)
        file = (folder / name).resolve()
        if not file.is_relative_to(folder) or not file.is_file():
            raise ValueError("Artifact is outside this project")
        relative = file.relative_to(folder).as_posix()
        if relative.startswith("muse/repairs/proposal/"):
            allowed = {"binding.json", *("advice/" + value for value in ("context.json", "hypotheses.json", "advisor-call.json", "request.json", "response.json"))}
            if (any(p.is_symlink() for p in ((folder / name), *(folder / name).parents))
                    or relative.removeprefix("muse/repairs/proposal/") not in allowed
                    or file.stat().st_size > (2 << 20)):
                raise ValueError("File is not a public controller-choice artifact")
            return file
        if not relative.startswith(("analysis/", "muse/advice/", "muse/checks/", "muse/repairs/run/", "equipment/run/", "build/", "lab/run/", "lab/sessions/", "planar/run/", "spatial/", "learning/")) and relative not in {self.jobs[identity].get("source_file"), "failure.json", "worker.log", "muse/binding.json", "support-priors.json", "body-reuse.json", "equipment/candidates.json", "equipment/source-validation.json", "equipment/selection.json"}:
            raise ValueError("File is not a public project artifact")
        return file

    def _persistent_token(self):
        saved = self.root / ".access-token"
        try:
            if saved.is_file() and not saved.is_symlink() and saved.stat().st_uid == os.getuid():
                value = saved.read_text().strip()
                if 32 <= len(value) <= 128:
                    return value
        except OSError:
            pass
        value = secrets.token_urlsafe(32)
        try:
            handle = os.open(saved, os.O_CREAT | os.O_WRONLY | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
            with os.fdopen(handle, "w") as stream:
                stream.write(value)
        except OSError:
            pass
        return value

    def close(self):
        self.stopping.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=10)
        if getattr(self, "avatar_studio", None) is not None:
            self.avatar_studio.close()
        if self.avatar_manager is not None:
            self.avatar_manager.close()
        for identity, lab in self.labs.items():
            (self.avatar_manager.attached(identity) or lab).close()
        os.close(self.lease)


class PortalServer(LabServer):
    def __init__(self, address, portal):
        self.portal = portal
        super().__init__(address, portal)
        self.RequestHandlerClass = PortalHandler


class PortalHandler(LabHandler):
    @property
    def allow_same_origin_frame(self):
        # Only portal-mounted labs can appear inside the run viewer.
        return getattr(self, "selected_lab", None) is not None

    @property
    def lab(self):
        return getattr(self, "selected_lab", self.server.portal)

    def route_lab(self):
        parsed = urlsplit(self.path)
        match = re.fullmatch(r"/labs/([a-f0-9]{32})(/.*)", parsed.path)
        if match:
            if not self.authorized(public=match[2] == "/"):
                return False
            self.selected_lab = self.server.portal.lab(match[1])
            self.path = match[2] + ("?" + parsed.query if parsed.query else "")
            return True
        return None

    def do_GET(self):
        try:
            route = self.route_lab()
            if route is not None:
                return super().do_GET() if route else None
            path = unquote(urlsplit(self.path).path)
            if path in PUBLIC_ASSETS and path != "/":
                return super().do_GET()
            if not self.authorized(public=path in ASSETS or path in DEMO_BRAND_ASSETS):
                return
            if path in DEMO_BRAND_ASSETS:
                return self.send_file(self.server.portal.project / DEMO_BRAND_ASSETS[path])
            if path in ASSETS:
                file = Path(__file__).parent / "assets" / ASSETS[path]
                if path in ("/", "/demo", "/demo/build", "/demo/runs"):
                    raw = file.read_bytes()
                    self.response_headers(200, "text/html; charset=utf-8", len(raw))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                return self.send_file(file)
            if path == "/api/portal":
                return self.reply(self.server.portal.info())
            if path == "/api/demo/state":
                query = parse_qs(urlsplit(self.path).query)
                run = (query.get("run") or [None])[0]
                demo = self.server.portal.demo
                return self.reply(demo.state(run) if run else demo.summary())
            if path == "/api/demo/steps":
                from weft.portal_demo import STEPS
                return self.reply({"steps": [dict(step) for step in STEPS]})
            if path == "/api/demo/featured":
                base = self.server.portal.demo.root
                names = []
                for saved in (base / "featured.json", base / "demo/featured.json"):
                    try:
                        if saved.is_file() and not saved.is_symlink():
                            names = [str(v)[:64] for v in json.loads(saved.read_text())][:24]
                            break
                    except (OSError, ValueError):
                        continue
                return self.reply({"featured": names})
            if path == "/api/demo/clips":
                return self.reply(self.server.portal.demo.clips())
            match = re.fullmatch(r"/api/demo/clips/([A-Za-z0-9-]{1,64})/video", path)
            if match:
                return self.send_file(self.server.portal.demo.clip_path(match[1]))
            if path == "/api/demo/gallery":
                return self.reply(self.server.portal.demo.gallery())
            match = re.fullmatch(r"/api/demo/gallery/([A-Za-z0-9-]{1,64})/video", path)
            if match:
                return self.send_file(self.server.portal.demo.gallery_video(match[1]))
            match = re.fullmatch(r"/api/demo/studio/([A-Za-z0-9_-]{1,64})", path)
            if match:
                return self.reply(self.server.portal.demo.studio(match[1]))
            match = re.fullmatch(r"/api/demo/artifact/([a-z_]{1,24})(?:/([A-Za-z0-9_-]{1,64}))?", path)
            if match:
                return self.send_file(self.server.portal.demo.artifact(match[1], match[2]))
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/task-options", path)
            if match:
                return self.reply(self.server.portal.task_options(match[1]))
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})(?:/files/(.+))?", path)
            if match:
                if match[2]:
                    return self.send_file(self.server.portal.artifact(match[1], match[2]))
                return self.reply(self.server.portal.summary(match[1]))
            return self.reply({"error": "Not found"}, 404)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            return self.reply({"error": str(error)}, 400)

    def do_POST(self):
        try:
            route = self.route_lab()
            if route is not None:
                return super().do_POST() if route else None
            if not self.authorized():
                return
            path = urlsplit(self.path).path
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Use a fixed-length upload")
            if path == "/api/demo/unlock":
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 512:
                    raise ValueError("Enter the access code from the submission.")
                self.connection.settimeout(15)
                from weft.exercise_body_source import _json as bounded_json
                body = bounded_json(self.rfile.read(length))
                return self.reply(self.server.portal.demo.unlock(body.get("code")))
            if path == "/api/demo/start":
                query = parse_qs(urlsplit(self.path).query)
                length = int(self.headers.get("Content-Length", "0"))
                if length > MAX_UPLOAD:
                    raise ValueError("Choose a video up to 100 MB")
                video = None
                if length:
                    self.connection.settimeout(120)
                    video = self.rfile.read(length)
                    if len(video) != length:
                        raise ValueError("The video upload was interrupted")
                return self.reply(self.server.portal.demo.start(
                    self.server.server_port, video=video,
                    name=(query.get("name") or [None])[0],
                    prompt=(query.get("prompt") or [None])[0],
                    clip=(query.get("clip") or [None])[0],
                    sample=not video and not query.get("clip")), 202)
            if path in {"/api/avatars/library", "/api/avatars/create"}:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 8192:
                    raise ValueError("Avatar request must be bounded JSON")
                self.connection.settimeout(30)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Avatar request upload was interrupted")
                from weft.exercise_body_source import _json as bounded_json
                body = bounded_json(raw)
                if not isinstance(body, dict):
                    raise ValueError("Avatar request must be a JSON object")
                if path == "/api/avatars/library":
                    from weft.portal_avatar_studio import read_library_zip
                    if set(body) != {"id"}:
                        raise ValueError("Library import takes exactly the character id")
                    payload = read_library_zip(self.server.portal.project, body["id"])
                    return self.reply(self.server.portal.prepare_avatars(payload), 202)
                if (not {"prompt", "request_id", "spend_approval"} <= set(body)
                        or set(body) - {"prompt", "style", "request_id", "spend_approval"}):
                    raise ValueError("Creation takes prompt, optional style, request_id and spend_approval")
                result = self.server.portal.avatar_studio.create(
                    body["prompt"], style=body.get("style", ""), request_id=body["request_id"],
                    spend_approval=body["spend_approval"])
                return self.reply(result, 202)
            avatar_attach = re.fullmatch(r"/api/projects/([a-f0-9]{32})/avatars/attach", path)
            if path in {"/api/avatars/builtins", "/api/avatars/import"} or avatar_attach:
                length = int(self.headers.get("Content-Length", "0"))
                maximum = 32 << 20 if path == "/api/avatars/import" else 2
                if not 1 <= length <= maximum:
                    raise ValueError("Avatar upload exceeds its bound")
                self.connection.settimeout(30)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Avatar upload was interrupted")
                if path != "/api/avatars/import" and raw != b"{}":
                    raise ValueError("This avatar operation takes an empty JSON object")
                result = (self.server.portal.attach_avatars(avatar_attach[1]) if avatar_attach else
                    self.server.portal.prepare_avatars(raw if path == "/api/avatars/import" else None))
                return self.reply(result, 202)
            if path == "/api/projects":
                query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
                if (not {"name", "mode", "ai", "request_id"} <= query.keys()
                        or query.keys() - {"name", "mode", "ai", "request_id", "exercise_inventory", "exercise_output"}
                        or any(len(v) != 1 for v in query.values()) or query["ai"][0] not in {"true", "false"}):
                    raise ValueError("Invalid upload options")
                length = int(self.headers.get("Content-Length", "0"))
                self.connection.settimeout(30)
                result = self.server.portal.upload(self.rfile, length, name=query["name"][0], mode=query["mode"][0],
                    ai=query["ai"][0] == "true", request_id=query["request_id"][0],
                    exercise_inventory=query.get("exercise_inventory", [None])[0], exercise_output=query.get("exercise_output", [None])[0])
                return self.reply(result, 202)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/cancel", path)
            if match:
                if int(self.headers.get("Content-Length", "0")) > 2:
                    raise ValueError("Cancellation takes no arguments")
                return self.reply(self.server.portal.cancel(match[1]))
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/tasks/(create|train)", path)
            if match:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 4096:
                    raise ValueError("Task requests require bounded JSON")
                self.connection.settimeout(10)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Task request upload was interrupted")
                from weft.spatial_scene_inputs import _json
                return self.reply(self.server.portal.task_action(match[1], match[2], _json(raw, 4096)), 202)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/moving-equipment/(prepare|build|select)", path)
            if match:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 16 << 10:
                    raise ValueError("Moving choices require bounded JSON")
                self.connection.settimeout(10)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Moving choice upload interrupted")
                from weft.shared_world_action_replay import _json
                return self.reply(self.server.portal.moving_equipment(match[1], match[2], _json(raw)), 200 if match[2] == "select" else 202)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/equipment/(prepare|select)", path)
            if match:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 256:
                    raise ValueError("Equipment choice requires a bounded candidate identifier")
                self.connection.settimeout(10)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Equipment choice request was interrupted")
                from weft.exercise_body_source import _json
                value = _json(raw)
                if set(value) != {"id"} or not isinstance(value["id"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value["id"]):
                    raise ValueError("Equipment choice accepts only a candidate identifier")
                method = self.server.portal.prepare_equipment if match[2] == "prepare" else self.server.portal.choose_equipment
                return self.reply(method(match[1], value["id"]), 202 if match[2] == "prepare" else 200)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/saved-body", path)
            if match:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 2048:
                    raise ValueError("Saved-body scope requires bounded JSON")
                self.connection.settimeout(10)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Saved-body request was interrupted")
                from weft.exercise_body_source import _json
                return self.reply(self.server.portal.prepare_saved_body(match[1], _json(raw)), 202)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/perception/(preflight|prepare|body-detail|supported-window)", path)
            if match:
                length = int(self.headers.get("Content-Length", "0"))
                if match[2] == "preflight":
                    if length != 0:
                        raise ValueError("Local preflight takes no arguments or paid consent")
                    return self.reply(self.server.portal.preflight_perception(match[1]), 202)
                if not 1 <= length <= 2048:
                    raise ValueError("Perception consent must be bounded JSON")
                self.connection.settimeout(10)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Perception consent upload was interrupted")
                from weft.exercise_body_source import _json
                if match[2] == "body-detail":
                    return self.reply(self.server.portal.prepare_body_detail(match[1], _json(raw)), 202)
                if match[2] == "supported-window":
                    return self.reply(self.server.portal.prepare_supported_window(match[1], _json(raw)), 202)
                return self.reply(self.server.portal.prepare_perception(match[1], _json(raw)), 202)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/muse", path)
            if match:
                if int(self.headers.get("Content-Length", "0")) != 0:
                    raise ValueError("Muse review takes no arguments")
                return self.reply(self.server.portal.review(match[1]), 202)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/muse/checks", path)
            if match:
                if int(self.headers.get("Content-Length", "0")) != 0:
                    raise ValueError("Local checks take no arguments")
                return self.reply(self.server.portal.check(match[1]), 202)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/muse/repairs(/choose)?", path)
            if match:
                if int(self.headers.get("Content-Length", "0")) != 0:
                    raise ValueError("Candidate testing takes no arguments")
                return self.reply(self.server.portal.repair(match[1], model_choice=True)
                                  if match[2] else self.server.portal.repair(match[1]), 202)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/supports", path)
            if match:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 128_000:
                    raise ValueError("Support request must be at most 128 KB")
                self.connection.settimeout(10)
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Support request was interrupted")
                return self.reply(self.server.portal.support_experiment(match[1], raw), 202)
            return self.reply({"error": "Not found"}, 404)
        except (ValueError, OSError, RuntimeError, KeyError) as error:
            return self.reply({"error": str(error)}, 400)


def launch_portal(args):
    if not 0 <= args.port <= 65535:
        raise ValueError("Port must be 0–65535")
    portal = Portal(args.root)
    server = None
    try:
        server = PortalServer(("127.0.0.1", args.port), portal)
        print(f"Weft portal: http://127.0.0.1:{server.server_port}/#token={portal.token}", flush=True)
        print("Upload videos in the browser. API keys stay server-side. Ctrl+C preserves projects and recordings.", flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        portal.close()
    return {"status": "stopped", "projects": str(portal.root)}
