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
from weft.lab import Lab, LabHandler, LabServer, PUBLIC_ASSETS
from weft.lab_worker import atomic_json
from weft.openai_access import MODEL, PROJECT_ROOT, OpenAIBudget, credential_status

MAX_UPLOAD = 100_000_000
MAX_JOBS = 50
MAX_STORAGE = 2_000_000_000
MAX_JOB_BYTES = 650_000_000
TERMINAL = {"ready", "failed", "cancelled", "interrupted"}
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
            status = self._status(identity)
            result.update(status)
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
            result["sandbox"] = f"/labs/{identity}/" if status["status"] == "ready" else None
            return result

    def info(self):
        with self.lock:
            ids = list(self.jobs)
        return {"provider": "openai", "model": MODEL, "credential": credential_status(self.project),
                "budget": self.budget.status(), "local_only": True, "max_upload_bytes": MAX_UPLOAD,
                "jobs": [self.summary(identity) for identity in reversed(ids)],
                "limits": "60-second videos, 1,800 frames, one processing job at a time; failed reconstructions remain inspectable"}

    def upload(self, stream, size, *, name, mode, ai, request_id):
        if (type(size) is not int or not 1 <= size <= MAX_UPLOAD or type(ai) is not bool
                or mode not in ("auto", "exercise", "objects") or (mode == "auto" and not ai)):
            raise ValueError("Choose a supported scene type and a video up to 100 MB; auto mode needs AI")
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
                      "source_sha256": None, "created_at": local_time(), "ai": ai, "mode": mode, "episodes": 300}
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
                if identity:
                    self.active = identity
                    atomic_json(self._folder(identity) / "status.json", {"status": "running", "stage": "starting", "events": []})
            if identity is None:
                self.wake.wait(1)
                self.wake.clear()
                continue
            self._run(identity)
            with self.lock:
                self.active, self.process = None, None

    def _run(self, identity):
        folder = self._folder(identity)
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
               "PYTHONUNBUFFERED": "1", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLCONFIGDIR": str(folder / "runtime/matplotlib")}
        if os.environ.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = os.environ["OPENAI_API_KEY"]
        process = None
        reason = None
        try:
            with (folder / "worker.log").open("xb") as log:
                process = subprocess.Popen([sys.executable, "-P", "-B", "-m", "weft.portal_worker", str(folder), str(self.project)],
                                           stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
                with self.lock:
                    self.process = process
                deadline = time.monotonic() + 420
                next_inventory = 0
                while process.poll() is None:
                    if self.stopping.wait(.5):
                        reason = "Portal stopped; no automatic retry"
                        break
                    if (folder / "cancel-request.json").exists():
                        reason = "Cancelled by user; partial evidence and uncertain charges preserved"
                        break
                    if time.monotonic() >= deadline:
                        reason = "Processing exceeded the seven-minute wall-time limit"
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
                current = self._status(identity)
                if current["status"] not in TERMINAL:
                    state = "cancelled" if (folder / "cancel-request.json").exists() else "interrupted" if self.stopping.is_set() else "failed"
                    atomic_json(folder / "status.json", {**current, "status": state,
                        "error": reason or f"Worker exited ({process.returncode}); inspect saved artifacts"})
        except Exception as error:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            current = self._status(identity)
            if current["status"] not in TERMINAL:
                atomic_json(folder / "status.json", {**current, "status": "failed", "error": str(error)[:500]})

    def cancel(self, identity):
        with self.lock:
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
                lab = Lab(lab_path)
                lab.token = self.token
                self.labs[identity] = lab
            self.labs.move_to_end(identity)
            return self.labs[identity]

    def artifact(self, identity, name):
        folder = self._folder(identity)
        file = (folder / name).resolve()
        if not file.is_relative_to(folder) or not file.is_file():
            raise ValueError("Artifact is outside this project")
        relative = file.relative_to(folder).as_posix()
        if not relative.startswith(("analysis/", "build/", "lab/run/", "lab/sessions/", "planar/run/", "spatial/", "learning/")) and relative not in {self.jobs[identity]["source_file"], "failure.json", "worker.log"}:
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
                if set(query) != {"name", "mode", "ai", "request_id"} or any(len(v) != 1 for v in query.values()) or query["ai"][0] not in {"true", "false"}:
                    raise ValueError("Invalid upload options")
                length = int(self.headers.get("Content-Length", "0"))
                self.connection.settimeout(30)
                result = self.server.portal.upload(self.rfile, length, name=query["name"][0], mode=query["mode"][0],
                    ai=query["ai"][0] == "true", request_id=query["request_id"][0])
                return self.reply(result, 202)
            match = re.fullmatch(r"/api/projects/([a-f0-9]{32})/cancel", path)
            if match:
                if int(self.headers.get("Content-Length", "0")) > 2:
                    raise ValueError("Cancellation takes no arguments")
                return self.reply(self.server.portal.cancel(match[1]))
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
