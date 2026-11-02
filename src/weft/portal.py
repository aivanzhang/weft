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
from weft.portal_supports import body_path
from weft.portal_lab_runtime import open_portal_lab
from weft.portal_inventory import exercise_intent

MAX_UPLOAD = 100_000_000
MAX_JOBS = 50
MAX_STORAGE = 2_000_000_000
MAX_JOB_BYTES = 650_000_000
TERMINAL = {"ready", "preview", "needs_input", "failed", "cancelled", "interrupted"}
ASSETS = {"/": "portal.html", "/portal.js": "portal.js", "/portal.css": "portal.css"}


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


class Portal:
    def __init__(self, root, *, project=PROJECT_ROOT, start=True):
        self.root = private_directory(root)
        self.project = Path(project).resolve()
        self.budget = OpenAIBudget(project)
        self.token = secrets.token_urlsafe(32)
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
        try:
            for folder in sorted(self.root.iterdir()):
                if not re.fullmatch(r"[a-f0-9]{32}", folder.name):
                    continue
                if folder.is_symlink() or not folder.is_dir() or len(self.jobs) >= MAX_JOBS:
                    raise ValueError("Portal inventory is unsafe or exceeds 50 jobs")
                config = bounded_json(folder / "job.json")
                self.jobs[folder.name] = config
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
            self.thread = threading.Thread(target=self._work, name="weft-portal-queue", daemon=True) if start else None
            if self.thread:
                self.thread.start()
        except Exception:
            os.close(self.lease)
            raise

    def _folder(self, identity):
        if not re.fullmatch(r"[a-f0-9]{32}", identity) or identity not in self.jobs:
            raise ValueError("Unknown project")
        return self.root / identity

    def _status(self, identity):
        path = self._folder(identity) / "status.json"
        return bounded_json(path) if path.is_file() else {"status": "uploading", "stage": "upload"}

    def summary(self, identity):
        with self.lock:
            config = self.jobs[identity]
            folder = self._folder(identity)
            result = {k: config[k] for k in ("id", "name", "created_at", "ai", "mode", "source_sha256")}
            result["kind"] = config.get("kind", "video")
            result["exercise_intent"] = config.get("exercise_intent")
            if result["kind"] == "imported_lab":
                from weft.portal_evidence_attachment import attachment_summary

                attachment = attachment_summary(folder, config)
                result["review_attachment"] = attachment
                result["review_unavailable_reason"] = None if attachment["available"] else attachment["reason"]
                result["learned_policy_available"] = config.get("learned_policy_available", False)
            if "inventory" in config:
                result["inventory"] = config["inventory"]
            result["parent_id"] = config.get("parent_id")
            result["source_available"] = (folder / config["source_file"]).is_file()
            status = self._status(identity)
            result.update(status)
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
                and result["kind"] == "video" and (body_path(folder, config) / "manifest.json").is_file())
            review = folder / "muse/status.json"
            if review.is_file():
                result["muse"] = bounded_json(review)
                result["muse"]["repair_availability"] = repair_availability(folder)
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
                    if result["muse"]["repairs"]["status"] == "complete":
                        result["muse"]["repairs"]["report"] = read_small_report(folder)
            return result

    def info(self):
        with self.lock:
            ids = list(self.jobs)
        return {"provider": "openai", "model": MODEL, "credential": credential_status(self.project),
                "muse": muse_availability(self.project),
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
            pending = sum(self._status(i)["status"] not in TERMINAL for i in self.jobs)
            if len(self.jobs) >= MAX_JOBS or pending >= 4:
                raise ValueError("Portal job capacity reached; preserve existing projects before making room")
            if ai and (not credential_status(self.project)["configured"]
                       or self.budget.status()["remaining_usd"] < self.budget.status()["reservation_usd"]):
                raise ValueError("OpenAI key or remaining allowance is unavailable; no video uploaded")
            stored = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file() and not p.is_symlink())
            # Reserve expansion for every queued job so parallel uploads cannot overcommit disk.
            if stored + (pending + 1) * MAX_JOB_BYTES > MAX_STORAGE:
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
                if identity is None:
                    identity = next((i for i in self.jobs if self._review_status(i).get("status") == "queued"), None)
                    review = identity is not None
                if identity is None:
                    identity = next((i for i in self.jobs if self._check_status(i).get("status") == "queued"), None)
                    checks = identity is not None
                if identity is None:
                    identity = next((i for i in self.jobs if self._repair_status(i).get("status") == "queued"), None)
                    repairs = identity is not None
                if identity:
                    self.active = identity
                    target = self._folder(identity) / ("muse/repairs/status.json" if repairs else "muse/checks/status.json" if checks else "muse/status.json" if review else "status.json")
                    if not repairs:
                        atomic_json(target, {"status": "running", "stage": "starting", "events": []})
            if identity is None:
                self.wake.wait(1)
                self.wake.clear()
                continue
            self._run(identity, review=review, checks=checks, **({"repairs": True} if repairs else {}))
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
            pending = sum(self._status(i)["status"] not in TERMINAL for i in self.jobs)
            if len(self.jobs) >= MAX_JOBS or pending >= 4:
                raise ValueError("Portal job capacity reached")
            stored = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file() and not p.is_symlink())
            if stored + (pending + 1) * MAX_JOB_BYTES > MAX_STORAGE:
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

    def repair(self, identity):
        with self.lock:
            root = self._folder(identity)
            self._require_review_body(identity, action="repairs")
            folder = root / "muse/repairs"
            if folder.exists():
                return self.summary(identity)
            available = repair_availability(root)
            if not available["available"]:
                raise ValueError(available["reason"])
            if sum(self._repair_status(i).get("status") in {"queued", "running"} for i in self.jobs) >= 4:
                raise ValueError("Candidate comparison queue is full")
            folder.mkdir(mode=0o700)
            atomic_json(folder / "status.json", {"status": "queued", "message": "Waiting to test bounded suggestions; no new AI request and no automatic application.",
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
        if self.jobs[identity].get("kind") == "imported_lab":
            from weft.portal_evidence_attachment import attachment_summary

            attached = attachment_summary(self._folder(identity), self.jobs[identity])
            if not attached["available"]:
                raise ValueError(attached["reason"])
            if action != "review":
                raise ValueError("Imported construction attachments enable Muse review only; local checks and repairs are not qualified")

    def _run(self, identity, *, review=False, checks=False, repairs=False):
        folder = self._folder(identity)
        work = folder / "muse/repairs" if repairs else folder / "muse/checks" if checks else folder / "muse" if review else folder
        terminal = REPAIR_TERMINAL if repairs else CHECK_TERMINAL if checks else REVIEW_TERMINAL if review else TERMINAL
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
               "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLCONFIGDIR": str(folder / "runtime/matplotlib")}
        support = self.jobs[identity].get("kind") == "support_experiment" and not (review or checks or repairs)
        if not (review or checks or repairs or support) and os.environ.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        if review:
            for name in ("META_API_KEY", "MODEL_API_KEY"):
                if os.environ.get(name):
                    env[name] = os.environ[name]
        process = None
        reason = None
        try:
            with (work / "worker.log").open("xb") as log:
                mode = ["repairs"] if repairs else ["checks"] if checks else ["review"] if review else ["supports"] if support else []
                process = subprocess.Popen([sys.executable, "-P", "-B", "-m", "weft.portal_worker", str(folder), str(self.project), *mode],
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
                with self.lock:
                    self.process = process
                deadline = time.monotonic() + (65 if repairs else 60 if checks else 90 if review else 420)
                next_inventory = 0
                while process.poll() is None:
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
                        if len(files) > 6000 or sum(p.stat().st_size for p in files if p.is_file()) > MAX_JOB_BYTES:
                            reason = "Processing reached its local artifact storage limit"
                            break
                if reason and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                process.wait()
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
            current = bounded_json(work / "status.json")
            if current["status"] not in terminal:
                atomic_json(work / "status.json", {**current, "status": "failed", "error": str(error)[:500]})

    def cancel(self, identity):
        with self.lock:
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

    def lab(self, identity):
        with self.lock:
            if self._status(identity)["status"] != "ready":
                raise ValueError("Sandbox is not ready; inspect this project's progress and failure details")
            if identity not in self.labs:
                if len(self.labs) >= 3:
                    idle = next((key for key, lab in self.labs.items() if lab.recorder is None
                                 and (lab.export_future is None or lab.export_future.done())), None)
                    if idle is None:
                        raise ValueError("Finish recording in another sandbox before opening a fourth")
                    self.labs.pop(idle).close()
                folder = self._folder(identity)
                lab_path = (folder / self._status(identity).get("lab", "lab")).resolve()
                if not lab_path.is_relative_to(folder):
                    raise ValueError("Sandbox path is outside this project")
                lab = open_portal_lab(lab_path, project=self.project)
                lab.token = self.token
                self.labs[identity] = lab
            self.labs.move_to_end(identity)
            return self.labs[identity]

    def artifact(self, identity, name):
        folder = self._folder(identity)
        if name.startswith("review-evidence/"):
            from weft.portal_evidence_attachment import attachment_artifact

            return attachment_artifact(folder, self.jobs[identity], name)
        file = (folder / name).resolve()
        if not file.is_relative_to(folder) or not file.is_file():
            raise ValueError("Artifact is outside this project")
        relative = file.relative_to(folder).as_posix()
        if not relative.startswith(("analysis/", "muse/advice/", "muse/checks/", "muse/repairs/run/", "build/", "lab/run/", "lab/sessions/", "planar/run/", "spatial/", "learning/")) and relative not in {self.jobs[identity]["source_file"], "failure.json", "worker.log", "muse/binding.json", "support-priors.json", "body-reuse.json"}:
            raise ValueError("File is not a public project artifact")
        return file

    def close(self):
        self.stopping.set()
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=10)
        for lab in self.labs.values():
            lab.close()
        os.close(self.lease)


class PortalServer(LabServer):
    def __init__(self, address, portal):
        self.portal = portal
        super().__init__(address, portal)
        self.RequestHandlerClass = PortalHandler


class PortalHandler(LabHandler):
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
            if not self.authorized(public=path in ASSETS):
                return
            if path in ASSETS:
                file = Path(__file__).parent / "assets" / ASSETS[path]
                if path == "/":
                    raw = file.read_bytes()
                    self.response_headers(200, "text/html; charset=utf-8", len(raw))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                return self.send_file(file)
            if path == "/api/portal":
                return self.reply(self.server.portal.info())
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
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/muse/repairs", path)
            if match:
                if int(self.headers.get("Content-Length", "0")) != 0:
                    raise ValueError("Candidate testing takes no arguments")
                return self.reply(self.server.portal.repair(match[1]), 202)
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
