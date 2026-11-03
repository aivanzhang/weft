"""Type a prompt in the portal and receive a chooser-ready character.

The portal never generates avatars itself: it manages the local Avatar Studio server as a
child process, relays one creation request per explicit user submission, and adopts the
finished model through the ordinary bounded delivery (automatic rig mapping, native
decimation, consumer import). Paid work stays behind the studio's own canonical-ledger
gate: a fresh generation starts only after the user's verbatim spending approval is
recorded as a targeted per-job grant, and the studio refuses anything beyond it.

Everything here is local; the studio's provider keys, budgets and failure evidence live
in its own package and are never copied into portal projects.
"""

import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path
from uuid import uuid4

from weft.io import local_time
from weft.lab_worker import atomic_json

STUDIO_RELATIVE = Path("tools/avatar-assets")
GENERATION_CREDITS = 35
MAX_CREATIONS = 8
MAX_MODEL_BYTES = 32 * 1024 * 1024
_PORT_LINE = re.compile(r"http://127\.0\.0\.1:(\d+)/")
_TERMINAL = {"ready", "failed", "unsupported", "needs_inspection", "adopt_failed", "cancelled"}


def list_library(project):
    """Enumerate the pre-generated character deliveries available for one-click import."""
    folder = Path(project) / ".local/avatar-library"
    rows = []
    if not folder.is_dir() or folder.is_symlink():
        return rows
    for archive in sorted(folder.glob("*.zip")):
        if archive.is_symlink() or not archive.is_file() or archive.stat().st_size > MAX_MODEL_BYTES:
            continue
        display = archive.stem
        sidecar = archive.with_suffix(".json")
        if sidecar.is_file() and not sidecar.is_symlink():
            try:
                meta = json.loads(sidecar.read_text())
                if isinstance(meta.get("display_name"), str):
                    display = meta["display_name"][:128]
            except ValueError:
                pass
        rows.append({"id": archive.stem, "display_name": display, "bytes": archive.stat().st_size})
        if len(rows) >= 16:
            break
    return rows


def read_library_zip(project, identifier):
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", identifier):
        raise ValueError("Choose a library character by its listed id")
    archive = Path(project) / ".local/avatar-library" / (identifier + ".zip")
    if archive.is_symlink() or not archive.is_file():
        raise ValueError("That library character is not available")
    if archive.stat().st_size > MAX_MODEL_BYTES:
        raise ValueError("Library character exceeds the 32 MiB import bound")
    return archive.read_bytes()


class AvatarStudioBridge:
    """Owns the studio child process and the portal-side creation registry."""

    def __init__(self, root, project, submit):
        self.root = Path(root) / "avatar-studio"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.project = Path(project).resolve()
        self.studio = self.project / STUDIO_RELATIVE
        self.python = self.studio / ".venv/bin/python"
        self.submit = submit
        self.lock = threading.RLock()
        self.process = None
        self.port = None
        self.token = None
        self._drain = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="weft-avatar-studio")
        self._registry = self._load()

    # -- registry -------------------------------------------------------------
    def _load(self):
        path = self.root / "creations.json"
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and len(data) <= MAX_CREATIONS * 4:
                return {key: value for key, value in data.items() if isinstance(value, dict)}
        except (OSError, ValueError):
            pass
        return {}

    def _save(self):
        atomic_json(self.root / "creations.json", self._registry)

    # -- studio process -------------------------------------------------------
    def available(self):
        return self.python.is_file() and (self.studio / "avatar_assets").is_dir()

    def ensure(self, timeout=30):
        with self.lock:
            if self.process is not None and self.process.poll() is None and self.port:
                return self.port
            if not self.available():
                raise ValueError("Avatar Studio is not installed beside this portal")
            log = (self.root / "studio.log").open("ab")
            process = subprocess.Popen(
                [str(self.python), "-m", "avatar_assets.cli", "studio", "--port", "0"],
                cwd=self.studio, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + timeout
            port = None
            while time.monotonic() < deadline:
                line = process.stdout.readline()
                if not line:
                    break
                log.write(line)
                log.flush()
                match = _PORT_LINE.search(line.decode("utf-8", "replace"))
                if match:
                    port = int(match.group(1))
                    break
            if port is None:
                process.terminate()
                log.close()
                raise RuntimeError("Avatar Studio did not report its address; see avatar-studio/studio.log")

            def drain(stream, sink):
                for row in iter(stream.readline, b""):
                    sink.write(row)
                    sink.flush()
                sink.close()

            self._drain = threading.Thread(target=drain, args=(process.stdout, log), daemon=True)
            self._drain.start()
            self.process, self.port = process, port
            session = json.loads(self._request("GET", "/api/studio/session"))
            self.token = session.get("csrf_token")
            return port

    def _request(self, method, path, payload=None, request_id=None, timeout=30):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {}
        body = b""
        if payload is not None or method != "GET":
            body = json.dumps(payload if payload is not None else {}).encode()
            headers["Content-Type"] = "application/json"
        if request_id is not None:
            headers["X-Avatar-Request-Id"] = request_id
        if method != "GET" and self.token:
            headers["X-Avatar-Studio-Token"] = self.token
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read(MAX_MODEL_BYTES + 1)
            if response.status >= 400:
                detail = raw.decode("utf-8", "replace")[:400]
                raise RuntimeError(f"Studio refused {path}: {detail}")
            return raw
        finally:
            connection.close()

    def _json(self, method, path, payload=None, request_id=None):
        return json.loads(self._request(method, path, payload, request_id))

    # -- creation lifecycle ---------------------------------------------------
    def create(self, prompt, *, style="", request_id, spend_approval):
        if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
            raise ValueError("Describe the character in 1-4000 characters")
        if not isinstance(style, str) or len(style) > 800:
            raise ValueError("Style must be at most 800 characters")
        if not isinstance(request_id, str) or not re.fullmatch(r"[a-f0-9]{32}", request_id):
            raise ValueError("A unique 32-hex request id is required")
        if not isinstance(spend_approval, str) or not 1 <= len(spend_approval) <= 4000:
            raise ValueError("Generation spends Meshy credits; include your explicit approval text")
        with self.lock:
            if request_id in self._registry:
                return self.summary()
            active = sum(1 for row in self._registry.values() if row.get("state") not in _TERMINAL)
            if active >= 2 or len(self._registry) >= MAX_CREATIONS:
                raise ValueError("Finish or clear existing character creations first")
        self.ensure()
        job = self._json("POST", "/api/studio/jobs",
                         {"prompt": prompt, "style": style, "preset": "preview"},
                         request_id=request_id)
        identifier = job["id"]
        self._json("POST", f"/api/studio/jobs/{identifier}/plan", {})
        with self.lock:
            self._registry[request_id] = {
                "studio_job": identifier, "prompt": prompt[:200], "state": "planning",
                "approval": spend_approval, "granted": False, "created_at": local_time(),
                "estimated_credits": GENERATION_CREDITS}
            self._save()
        return self.summary()

    AGENT_GRANT_MICRODOLLARS = 1_900_000

    def _grant_tool(self, *arguments):
        result = subprocess.run(
            [str(self.python), "-m", "avatar_assets.studio_grant", *arguments],
            cwd=self.studio, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout)[-400:])

    def _grant_openai(self, entry):
        self._grant_tool("openai", entry["studio_job"],
                         "--microdollars", str(self.AGENT_GRANT_MICRODOLLARS),
                         "--reply", entry["approval"])

    def _grant(self, entry):
        output = self.studio / "outputs" / f"studio-{entry['studio_job']}"
        self._grant_tool("meshy", str(output), "--credits", str(GENERATION_CREDITS),
                         "--reply", entry["approval"])

    def poll(self):
        """Advance every live creation one step and return the condensed summary."""
        with self.lock:
            live = [(key, dict(row)) for key, row in self._registry.items()
                    if row.get("state") not in _TERMINAL]
        if live and (self.process is None or self.process.poll() is not None):
            try:
                self.ensure()
            except (ValueError, RuntimeError) as error:
                with self.lock:
                    for key, _ in live:
                        self._registry[key].update(state="failed", error=str(error)[:300])
                    self._save()
                return self.summary()
        for key, entry in live:
            try:
                payload = self._json("GET", f"/api/studio/jobs/{entry['studio_job']}")
                self._advance(key, entry, payload)
            except (RuntimeError, OSError, ValueError, KeyError) as error:
                with self.lock:
                    self._registry[key]["last_error"] = str(error)[:300]
                    self._save()
        return self.summary()

    def _advance(self, key, entry, payload):
        status = payload.get("status")
        update = {"studio_status": status,
                  "stages": [{"stage": row.get("stage") or row.get("id"), "status": row.get("status")}
                             for row in (payload.get("stages") or [])[:12] if isinstance(row, dict)],
                  "elapsed_seconds": payload.get("elapsed_seconds")}
        if payload.get("error"):
            update["error"] = payload["error"].get("message")
        plan = payload.get("plan") or {}
        authorization = plan.get("authorization") or {}
        if status == "awaiting_authorization" and not payload.get("plan_id"):
            # Planning itself needs the bounded agent-dollar grant before it can run.
            if not entry.get("openai_granted"):
                try:
                    self._grant_openai(entry)
                    update["openai_granted"] = True
                    self._json("POST", f"/api/studio/jobs/{entry['studio_job']}/plan", {})
                    update["state"] = "planning"
                except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                    update["state"] = "failed"
                    update["error"] = "Agent spending authorization failed: " + str(error)[:300]
            else:
                update["state"] = "failed"
                update["error"] = (payload.get("error") or {}).get("message") or                     "Planning stayed unauthorized after the recorded agent grant"
        elif status in ("planned", "awaiting_authorization") and payload.get("plan_id"):
            if authorization.get("can_start"):
                self._json("POST", f"/api/studio/jobs/{entry['studio_job']}/start",
                           {"plan_id": payload["plan_id"]})
                update["state"] = "generating"
            elif not entry.get("granted"):
                try:
                    self._grant(entry)
                    update["granted"] = True
                    update["state"] = "authorized"
                except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
                    update["state"] = "failed"
                    update["error"] = ("Spending authorization failed: " + str(error)[:260]
                                       + " | studio: " + str(authorization.get("reason"))[:200])
            else:
                update["state"] = "failed"
                update["error"] = str(authorization.get("reason"))[:400]
        elif status in ("queued", "launching", "planning", "running", "cancel_requested"):
            update["state"] = "generating" if entry.get("granted") or status != "planning" else "planning"
        elif status == "complete":
            if not entry.get("adopting"):
                update["adopting"] = True
                update["state"] = "adopting"
                self._executor.submit(self._adopt, key, dict(entry), payload)
            else:
                update["state"] = entry.get("state", "adopting")
        elif status in ("failed", "unsupported", "needs_inspection", "cancelled"):
            update["state"] = status
            update.setdefault("error", (payload.get("error") or {}).get("message")
                              if isinstance(payload.get("error"), dict) else None)
        with self.lock:
            self._registry[key].update({k: v for k, v in update.items() if v is not None or k == "error"})
            self._save()

    def _adopt(self, key, entry, payload):
        try:
            model_url = (payload.get("result") or {}).get("model_url")
            if not isinstance(model_url, str) or not model_url.startswith("/api/studio/jobs/"):
                raise RuntimeError("Completed job did not expose its delivery model")
            raw = self._request("GET", model_url, timeout=60)
            if not 1 <= len(raw) <= MAX_MODEL_BYTES:
                raise RuntimeError("Delivery model exceeds the 32 MiB adoption bound")
            workdir = self.root / "adoptions" / entry["studio_job"]
            workdir.mkdir(parents=True, exist_ok=True)
            model = workdir / "model.glb"
            if not model.exists():
                model.write_bytes(raw)
            # The delivery refuses to write beside its immutable source folder.
            pack_dir = self.root / "adoptions" / f"{entry['studio_job']}-pack-{uuid4().hex[:8]}"
            asset_id = f"created-{entry['studio_job'][:12]}"
            result = subprocess.run(
                [str(self.python), "-m", "avatar_assets.weft_autopack", str(model), str(pack_dir),
                 "--asset-id", asset_id, "--name", ("Created: " + entry.get("prompt", "character"))[:120]],
                cwd=self.studio, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout)[-400:])
            archive = pack_dir / "delivery/weft-avatar.zip"
            self.submit(archive.read_bytes())
            with self.lock:
                self._registry[key].update(state="ready", asset_id=asset_id,
                                           ready_at=local_time(), error=None)
                self._save()
        except Exception as error:
            with self.lock:
                self._registry[key].update(state="adopt_failed", error=str(error)[:400])
                self._save()

    def summary(self):
        with self.lock:
            rows = sorted(self._registry.items(), key=lambda item: item[1].get("created_at", ""),
                          reverse=True)[:MAX_CREATIONS]
            creations = [{"request_id": key,
                          **{field: row.get(field) for field in
                             ("prompt", "state", "studio_status", "stages", "elapsed_seconds",
                              "error", "asset_id", "estimated_credits", "granted")}}
                         for key, row in rows]
        return {"available": self.available(),
                "studio_running": self.process is not None and self.process.poll() is None,
                "creations": creations,
                "scope": ("Local Avatar Studio generation with its own keys and canonical budgets; "
                          "a creation spends credits only after the recorded explicit approval.")}

    def close(self):
        with self.lock:
            process, self.process, self.port = self.process, None, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self._executor.shutdown(wait=False, cancel_futures=True)
