"""Guided judge demo: one bounded run of the real pipeline, with evidence per step.

Nothing here reimplements the pipeline. A worker thread drives the portal's own HTTP
API on 127.0.0.1 exactly as a person would — upload, local preflight, saved-body world,
lab, recorded rollout with a mid-take character swap, Meta Muse review — and records
what each step actually produced. Every step keeps its honest outcome: a refused clip
stays refused with its reason instead of silently falling back to the sample.

Each step packs a bounded zip of its own evidence, and a finished run joins the gallery
with its world still loadable, so a reviewer can drive it rather than trust the screen.
"""

import json
import re
import secrets
import shutil
import threading
import time
import zipfile
from http.client import HTTPConnection
from pathlib import Path
from urllib.parse import urlencode

from weft.io import local_time

ACCESS_FILE = ".local/demo-access-code"
# Clips whose whole chain was measured headlessly, offered so a judge can start from a
# known-admitting source instead of hunting for footage the reader will refuse.
CLIPS_DIR = ".local/demo-clips"
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_ARTIFACT_BYTES = 48 * 1024 * 1024
RUN_ID = re.compile(r"run-[a-f0-9]{1,32}")
STEP_ID = re.compile(r"[a-z_]{1,32}")
# Muse spends the shared Meta allowance; a demo must never be able to drain it.
MAX_MUSE_CALLS = 8
# A finished run leaves a ~150 MB portal project, and the portal reserves 650 MB of
# headroom per new job inside a 2 GB allowance. Keeping four runs leaves room for the
# next one; without the eviction below, a reviewer's fourth run fails on a storage wall
# the walkthrough itself built.
MAX_RUNS_KEPT = 4
# The portal allows 2 GB and reserves 650 MB of headroom per new job, with the avatar
# library and this walkthrough's own bundles taking roughly 250 MB more.
KEEP_BYTES = 800 * 1024 * 1024
PROJECT_DIR = re.compile(r"[0-9a-f]{32}")
# Live character generation spends credits; bound how often a demo may start one.
MAX_GENERATION_ATTEMPTS = 2
# A clip without a recorded creator and licence cannot be offered to a reviewer, however
# well it runs. Manifests state missing provenance in words rather than omitting it.
UNRECORDED = ("unverified", "unknown", "not recorded", "none", "n/a", "-")


def _attributable(clip):
    creator, licence = str(clip.get("creator", "")).strip(), str(clip.get("licence", "")).strip()
    if not creator or not licence:
        return False
    return not any(str(value).lower().startswith(UNRECORDED) for value in (creator, licence))


def _own(run_dir, project):
    """Record a project this run created so eviction can reclaim it, failure or not."""
    saved = run_dir / "projects.json"
    try:
        owned = json.loads(saved.read_text())
    except (OSError, ValueError):
        owned = []
    if project not in owned:
        owned.append(project)
        saved.write_text(json.dumps(owned) + "\n")


def _runs_locally(clip):
    """A clip is offered unless the manifest says this walkthrough's path refuses it.

    The hosted container and the local portal disagree on clips near the pose threshold,
    and this page runs the local one. An absent verdict means unknown, not broken.
    """
    verdict = (clip.get("paths") or {}).get("local_portal")
    return verdict is None or str(verdict).strip().lower() == "ready"

# Judge-facing copy: what runs, why it matters against the rubric, and the cost it
# removes. Estimates come from verified local runs, not aspiration.
STEPS = (
    {
        "id": "intake",
        "title": "Read the video as evidence",
        "explainer": "Local pose tracking turns your clip into joint trajectories with per-sample"
                     " confidence, competing hypotheses and explicit gaps — no API calls, nothing discarded.",
        "matters": "Technical depth: evidence is retained and inspectable, never asserted.",
        "impact": "Motion capture normally means a studio booking at roughly $500–$5,000 a day."
                  " This reads an ordinary phone clip in about half a minute.",
        "estimate_seconds": 20,
    },
    {
        "id": "world",
        "title": "Build an executable world",
        "explainer": "That evidence becomes a running MuJoCo simulation: a scaled articulated body,"
                     " joint limits, contacts and an authored ground plane, every assumption labelled.",
        "matters": "Impact: this is the artifact reinforcement-learning teams wait weeks for.",
        "impact": "Hand-authoring one task environment is engineer-weeks — on the order of $15k–$50k"
                  " in loaded cost per environment. This step takes about twenty seconds.",
        "estimate_seconds": 6,
    },
    {
        "id": "task",
        "title": "Freeze the task and its gates",
        "explainer": "A reinforcement-learning task is attached with success gates frozen before the run:"
                     " tracking-error thresholds the pipeline cannot edit to flatter its own result.",
        "matters": "Technical depth: immutable scoring, decided before the evidence arrives.",
        "impact": "Movable goalposts are why most simulation results never transfer. Freezing them"
                  " first is what makes the number worth reading.",
        "estimate_seconds": 5,
    },
    {
        "id": "rollout",
        "title": "Run an agent inside it",
        "explainer": "A reference controller drives hundreds of continuous control steps while the frozen"
                     " gates score tracking error live, and the session is recorded for exact replay.",
        "matters": "Demo: working software end to end, not a mock or a scripted animation.",
        "impact": "An environment you cannot run is a diagram. This is where the video becomes"
                  " something an agent can actually train against.",
        "estimate_seconds": 8,
    },
    {
        "id": "avatar",
        "title": "Swap the character, keep the physics",
        "explainer": "Mid-recording the appearance swaps to a different character. Simulation state,"
                     " controls and score are untouched — only what you see changes.",
        "matters": "Demo and creative range: appearance is decoupled from the physics it rides on.",
        "impact": "One captured behaviour can be re-skinned for any product, game or study without"
                  " re-capturing or re-simulating anything.",
        "estimate_seconds": 14,
    },
    {
        "id": "muse",
        "title": "Have Meta's model review the physics",
        "explainer": "Meta Muse reads the construction evidence and source frames, then proposes missing"
                     " constraints and alternative explanations for what the simulation got wrong.",
        "matters": "Use of Meta AI tooling: the model interprets real evidence and proposes a constrained"
                   " change that an independent scorer — not the model — accepts or rejects.",
        "impact": "Calibrating a wrong simulator is the expensive part of sim-to-real. This is the loop"
                  " that makes a wrong world improvable instead of disposable.",
        "estimate_seconds": 60,
    },
    {
        "id": "create",
        "title": "Generate a character from text",
        "explainer": "A generated character is imported and superimposed on the same recorded physics."
                     " Live text-to-character generation sits behind a recorded spending gate and is"
                     " reported honestly, including when its own quality gates refuse a result.",
        "matters": "Creative range under the same honesty: generated assets face gates too.",
        "impact": "Character art is normally days of modelling and rigging per asset.",
        "estimate_seconds": 6,
        "experimental": True,
    },
)

GALLERY_SEEDS = (
    {
        "id": "waiters-squat-panda",
        "title": "Overhead squat → playable world",
        "caption": "A licensed clip the pipeline had never seen, taken end to end in about fifteen"
                   " seconds: admitted locally, simulated, scored against frozen gates, recorded.",
        "metrics": "246 control steps · every gate passed · 0 API calls",
        "source": "verified run",
        "video": ".local/verify-video-01/portal/9eadcc9655864070b0039fa3feb93fbc/"
                 "lab/sessions/session-9f8a60b20993/avatar-video.mp4",
    },
    {
        "id": "waiters-squat-created",
        "title": "Same physics, generated character",
        "caption": "The identical recording re-skinned with a character built through the"
                   " text-to-avatar flow. The simulation state is unchanged.",
        "metrics": "246 control steps · swap at step 120 · physics unchanged",
        "source": "verified run",
        "video": ".local/verify-video-01/portal/9eadcc9655864070b0039fa3feb93fbc/"
                 "lab/sessions/session-1de315688ae1/avatar-video.mp4",
    },
    {
        "id": "sots-press-swap",
        "title": "Mid-take character swap",
        "caption": "456 continuous control steps in one unbroken recording, with the character"
                   " swapped at step 300 — the agent never notices.",
        "metrics": "456 control steps · swap at step 300 · every gate passed",
        "source": "verified run",
        "video": ".local/portal-unfamiliar-source-acceptance-02/portal/"
                 "776f9c6965794a21947da36e8e871d2d/lab/sessions/session-4b80a50a9c9f/avatar-video.mp4",
    },
)


def _zip_bounded(dest, entries, note):
    """Pack what fits under the artifact cap and record what was left out, by name."""
    included, omitted, total = [], [], 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arcname, path in entries:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if total + size > MAX_ARTIFACT_BYTES:
                omitted.append({"name": arcname, "bytes": size})
                continue
            archive.write(path, arcname=arcname)
            included.append({"name": arcname, "bytes": size})
            total += size
        archive.writestr("MANIFEST.json", json.dumps({
            "profile": "weft-demo-step-artifact-v1", "note": note, "packed_at": local_time(),
            "included": included, "omitted_for_size": omitted,
            "scope": "Evidence copied from this run. Simulations are conditional models with"
                     " disclosed assumptions, not verified copies of the physical world.",
        }, indent=2) + "\n")
    return dest.stat().st_size


def _collect(base, patterns):
    """Resolve glob patterns under one directory into (arcname, path) pairs."""
    base = Path(base)
    found = []
    for pattern in patterns:
        for path in sorted(base.glob(pattern)):
            if path.is_file() and not path.is_symlink():
                found.append((str(path.relative_to(base)), path))
    return found


class DemoManager:
    """Owns the access gate, the single active run, finished runs, and the gallery."""

    def __init__(self, root, project=None, token=None):
        # Accepts either the Portal itself or its (root, project, token) triple.
        self.portal = None
        if project is None and token is None and hasattr(root, "token"):
            # Kept so eviction can retire a job from the live registry rather than
            # deleting its directory underneath the portal.
            self.portal = root
            root, project, token = root.root, root.project, root.token
        self.root = Path(root)
        self.project = Path(project)
        self.token = token
        self.dir = self.root / "demo"
        self.dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.port = None
        self.run = None
        self.thread = None
        self._muse_calls = 0

    # -- access ---------------------------------------------------------------
    def code(self):
        """The judge code lives in an ignored local file, never in source control."""
        path = self.project / ACCESS_FILE
        try:
            value = path.read_text().strip()
            if value:
                return value
        except OSError:
            pass
        alphabet = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"
        value = "WEFT-JUDGE-" + "-".join(
            "".join(secrets.choice(alphabet) for _ in range(4)) for _ in range(2))
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(value + "\n")
        path.chmod(0o600)
        return value

    def unlock(self, value):
        if not isinstance(value, str) or not 1 <= len(value) <= 64:
            raise ValueError("Enter the access code from the submission")
        if not secrets.compare_digest(value.strip().upper().encode(), self.code().upper().encode()):
            raise ValueError("That access code is not recognized")
        return {"ok": True}

    # -- http to our own portal ----------------------------------------------
    def _request(self, method, path, payload=None, timeout=180):
        connection = HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        headers = {"Authorization": "Bearer " + self.token}
        if isinstance(payload, bytes):
            body = payload
        elif payload is None:
            body = b""
        else:
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        try:
            try:
                connection.request(method, path, body=body, headers=headers)
            except OSError as error:
                # The portal answers a rejected upload and closes while the body is still
                # going out. Read the reply it already sent, or the judge sees "Broken pipe".
                try:
                    response = connection.getresponse()
                    raw = response.read(64 << 20)
                except OSError:
                    raise RuntimeError(f"The local upload was interrupted ({error})") from error
                detail = raw[:400].decode("utf-8", "replace")
                try:
                    detail = json.loads(raw).get("error", detail)
                except ValueError:
                    pass
                raise RuntimeError(detail or f"The local upload was interrupted ({error})") from error
            response = connection.getresponse()
            raw = response.read(64 << 20)
            if response.status >= 400:
                detail = raw[:400].decode("utf-8", "replace")
                try:
                    detail = json.loads(raw).get("error", detail)
                except ValueError:
                    pass
                raise RuntimeError(detail)
            return json.loads(raw) if raw[:1] in (b"{", b"[") else raw
        finally:
            connection.close()

    # -- state ----------------------------------------------------------------
    def _blank_steps(self):
        """Full copy up front so the walkthrough reads as a guide before anything runs."""
        return [{**step, "status": "pending", "started": None, "elapsed_seconds": None,
                 "detail": "", "metrics": {}, "artifact": None, "error": None} for step in STEPS]

    def summary(self):
        with self.lock:
            if self.run is None:
                return {"status": "idle", "run": None, "steps": self._blank_steps(),
                        "elapsed_seconds": 0, "total_estimate_seconds": sum(
                            step["estimate_seconds"] for step in STEPS), "error": None, "source": None}
            return json.loads(json.dumps(self._public(self.run)))

    def _public(self, run):
        return {"status": run["status"], "run": run["id"], "steps": run["steps"],
                "elapsed_seconds": run["elapsed_seconds"],
                "total_estimate_seconds": run["total_estimate_seconds"],
                "error": run["error"], "source": run["source"], "project": run.get("project")}

    def _touch(self):
        self.run["elapsed_seconds"] = round(time.time() - self.run["created"], 1)

    def _begin(self, index, detail):
        with self.lock:
            self.run["steps"][index].update(status="running", started=time.time(), detail=detail)
            self._touch()

    def _update(self, index, **fields):
        with self.lock:
            step = self.run["steps"][index]
            step.update(fields)
            if step["started"]:
                step["elapsed_seconds"] = round(time.time() - step["started"], 1)
            self._touch()

    def _finish(self, index, detail, metrics=None, artifact=None, status="done", error=None):
        self._update(index, status=status, detail=detail, metrics=metrics or {},
                     artifact=artifact, error=error)

    # -- lifecycle ------------------------------------------------------------
    def sample_video(self):
        for candidate in (".local/verify-video-01/waiters-squat-window3.mp4",
                          "data/exercise/squat.webm", "data/exercise/single-leg-squat.webm"):
            path = self.project / candidate
            if path.is_file():
                return path
        raise ValueError("No bundled sample clip is available")

    def _retire(self, project):
        """Drop a job from the live registry before its directory goes.

        A job whose folder vanished reads as non-terminal forever, and four of those
        exhaust the portal's pending-job capacity — every later run is refused.
        Only a job this walkthrough finished is retired; an active one is left alone.
        """
        portal = self.portal
        if portal is None or not hasattr(portal, "jobs"):
            return True
        from weft.portal import TERMINAL as TERMINAL_JOB  # imported here: portal imports this module
        with portal.lock:
            if project not in portal.jobs:
                return True
            try:
                if portal._status(project)["status"] not in TERMINAL_JOB:
                    return False
            except (OSError, ValueError, KeyError):
                pass
            portal.jobs.pop(project, None)
        return True

    def _evict(self):
        """Drop runs past the keep limit and the portal projects they own.

        Only projects this walkthrough created are removed: each run records its project
        id, so a project the reviewer made in the lab is never a candidate.
        """
        owned = {}
        for entry in sorted(self.dir.glob("run-*/record.json")):
            try:
                owned.setdefault(entry.parent.name, []).append(
                    str(json.loads(entry.read_text()).get("project") or ""))
            except (OSError, ValueError):
                pass
        for entry in sorted(self.dir.glob("run-*/project.txt")):  # pre-projects.json runs
            owned.setdefault(entry.parent.name, []).append(entry.read_text().strip())
        for entry in sorted(self.dir.glob("run-*/projects.json")):
            try:
                owned.setdefault(entry.parent.name, []).extend(json.loads(entry.read_text()))
            except (OSError, ValueError):
                pass

        def project_bytes(run_id):
            total = 0
            for name in owned.get(run_id, []):
                target = self.root / name
                if PROJECT_DIR.fullmatch(name) and target.is_dir():
                    total += sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            return total

        # Runs vary from 30 MB to 240 MB, so a count alone cannot keep the portal under
        # its allowance. Keep the newest runs that fit the budget, and always at least one.
        kept, used = [], 0
        for row in self._index()[:MAX_RUNS_KEPT]:
            size = project_bytes(row["id"])
            if kept and used + size > KEEP_BYTES:
                break
            kept.append(row["id"])
            used += size

        for run_id in list(owned) + [d.name for d in self.dir.glob("run-*") if d.is_dir()]:
            if run_id in kept or (self.run or {}).get("id") == run_id:
                continue
            for name in owned.get(run_id, []):
                if not PROJECT_DIR.fullmatch(name):
                    continue
                target = (self.root / name).resolve()
                if target.is_dir() and target.parent == self.root.resolve() and self._retire(name):
                    shutil.rmtree(target, ignore_errors=True)
            shutil.rmtree(self.dir / run_id, ignore_errors=True)
        index = [row for row in self._index() if row["id"] in kept]
        (self.dir / "runs.json").write_text(json.dumps(index, indent=2) + "\n")
        featured = [v for v in self._featured() if v in kept]
        (self.dir / "featured.json").write_text(json.dumps(featured, indent=2) + "\n")

    def clips(self):
        """Verified starting clips, newest measurement first. Empty is fine: upload still works."""
        manifest = self.project / CLIPS_DIR / "manifest.json"
        try:
            data = json.loads(manifest.read_text())
        except (OSError, ValueError):
            return {"clips": [], "constraints": {}}
        out = []
        for clip in data.get("clips", []):
            path = manifest.parent / str(clip.get("file", ""))
            if clip.get("recommended") != "showcase" or not path.is_file():
                continue
            if not _attributable(clip) or not _runs_locally(clip):
                continue
            rl = clip.get("rl") or {}
            out.append({
                "id": str(clip.get("id", ""))[:64], "title": str(clip.get("title", ""))[:80],
                "creator": str(clip.get("creator", ""))[:120], "licence": str(clip.get("licence", ""))[:40],
                "seconds": clip.get("chain_seconds"), "why": str(clip.get("why", ""))[:200],
                "learned": rl.get("learned"), "random": rl.get("random"),
            })
        return {"clips": out, "constraints": data.get("constraints") or {}}

    def clip_path(self, clip_id):
        base = self.project / CLIPS_DIR
        try:
            data = json.loads((base / "manifest.json").read_text())
        except (OSError, ValueError):
            data = {}
        for clip in data.get("clips", []):
            if (clip.get("id") == clip_id and clip.get("recommended") == "showcase"
                    and _attributable(clip) and _runs_locally(clip)):
                # Resolve under the clips directory so a manifest cannot point outside it.
                path = (base / Path(str(clip.get("file", ""))).name).resolve()
                if path.is_file() and path.parent == base.resolve():
                    return path
        raise ValueError("That clip is not one of the verified starting clips")

    # Route handlers have been written against two shapes of this API while the page and
    # the orchestrator were built in parallel; accept both rather than break on either.
    def state(self, run_id=None):
        current = self.summary()
        if run_id and current.get("run") and current["run"] != run_id:
            record = self._record(run_id)
            if record:
                return {"status": record.get("status", "complete"), "run": record["id"],
                        "steps": record.get("steps", []), "elapsed_seconds": 0,
                        "total_estimate_seconds": sum(s["estimate_seconds"] for s in STEPS),
                        "error": None, "source": None, "project": record.get("project")}
        return current

    def start(self, server_port=None, *, video=None, name=None, prompt=None, sample=None, clip=None):
        # First argument is the server port in one caller and the video bytes in another.
        if isinstance(server_port, (bytes, bytearray)) or (server_port is None and sample is not None):
            video, server_port = (bytes(server_port) if server_port else None), self.port
        if sample:
            video = None
        if server_port is not None:
            self.port = server_port
        if self.port is None:
            raise ValueError("The demo is still starting; refresh in a moment")
        label = None
        if clip and not video:
            path = self.clip_path(clip)
            video, name = path.read_bytes(), path.name
            label = next((c["title"] for c in self.clips()["clips"] if c["id"] == clip), None)
        return self._start(video=video, name=name, prompt=prompt, label=label)

    def _start(self, *, video=None, name=None, prompt=None, label=None):
        server_port = self.port
        with self.lock:
            if self.run and self.run["status"] == "running":
                raise ValueError("A run is already in progress; watch it finish below")
            self.port = server_port
            if not video:
                path = self.sample_video()
                video, name, sample = path.read_bytes(), path.name, True
            else:
                sample = False
                name = name or "clip.mp4"
            if not isinstance(video, bytes) or not 1 <= len(video) <= MAX_VIDEO_BYTES:
                raise ValueError("Choose a video up to 100 MB")
            if Path(name).suffix.lower() not in {".mp4", ".mov", ".webm", ".m4v", ".mkv", ".avi"}:
                raise ValueError("Choose an MP4, MOV, WebM, M4V, MKV or AVI file")
            if prompt is not None and (not isinstance(prompt, str) or len(prompt) > 400):
                raise ValueError("Keep the character description under 400 characters")
            run_id = "run-" + secrets.token_hex(12)
            self.run = {"id": run_id, "status": "running", "created": time.time(), "elapsed_seconds": 0,
                        "total_estimate_seconds": sum(step["estimate_seconds"] for step in STEPS),
                        "steps": self._blank_steps(), "error": None, "project": None,
                        "prompt": (prompt or "").strip(),
                        "source": {"name": Path(name).name[:120], "bytes": len(video),
                                   "sample": sample, "label": label}}
            (self.dir / run_id).mkdir(mode=0o700, parents=True, exist_ok=True)
            self._evict()
            self.thread = threading.Thread(target=self._drive, args=(run_id, video, Path(name).name),
                                           name="weft-demo", daemon=True)
            self.thread.start()
            return self._public(self.run)

    def _poll(self, path, ready, *, limit=360, key=None):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            state = self._request("GET", path)
            value = key(state) if key else state.get("status")
            if value in ready:
                return state, value
            time.sleep(0.6)
        raise RuntimeError("The pipeline did not settle within its time budget")

    def _drive(self, run_id, video, name):
        run_dir = self.dir / run_id
        try:
            parent = self._step_intake(run_dir, video, name)
            child = self._step_world(run_dir, parent)
            with self.lock:
                self.run["project"] = child
            lab, catalog = self._step_task(run_dir, child)
            session, metrics, handoff = self._step_rollout(lab, catalog)
            self._step_avatar(run_dir, child, lab, session, metrics, handoff)
            self._step_muse(run_dir, child)
            self._step_create(run_dir, child, catalog)
            with self.lock:
                self.run["status"] = "complete"
                self._touch()
            self._archive(run_id)
        except Exception as error:  # a run records its own failure rather than vanishing
            with self.lock:
                if self.run and self.run["id"] == run_id:
                    self.run["status"] = "failed"
                    self.run["error"] = str(error)[:400]
                    for step in self.run["steps"]:
                        if step["status"] == "running":
                            step.update(status="failed", error=str(error)[:400])
                        elif step["status"] == "pending":
                            step.update(status="skipped", detail="Not reached")
                    self._touch()
            self._archive(run_id)

    # -- steps ----------------------------------------------------------------
    def _normalize(self, run_dir, video, name):
        """Judge clips arrive as .mov/.avi or oversized; perception needs a bounded MP4.

        The backend helper lands separately, so stay working without it."""
        try:
            from weft.portal_pipeline import prepare_demo_source
        except ImportError:
            return video, name, None
        folder = run_dir / "source"
        folder.mkdir(parents=True, exist_ok=True)
        original = folder / ("original" + (Path(name).suffix.lower() or ".mp4"))
        original.write_bytes(video)
        try:
            path, receipt = prepare_demo_source(original, folder / "prepared")
        except Exception:
            return video, name, None
        if receipt is None:
            return video, name, None
        path = Path(path)
        return path.read_bytes(), path.name, receipt

    def _step_intake(self, run_dir, video, name):
        self._begin(0, "Uploading and checking the clip locally")
        video, name, receipt = self._normalize(run_dir, video, name)
        if receipt:
            self._update(0, detail="Re-encoded once to the supported envelope; original untouched")
        query = urlencode({"name": name, "mode": "exercise", "ai": "false",
                           "request_id": secrets.token_hex(16),
                           "exercise_inventory": "unknown", "exercise_output": "simulation"})
        parent = self._request("POST", "/api/projects?" + query, video, timeout=300)["id"]
        _own(run_dir, parent)
        self._poll(f"/api/projects/{parent}", {"needs_input", "failed", "unsupported"})
        self._update(0, detail="Tracking the body, frame by frame")
        self._request("POST", f"/api/projects/{parent}/perception/preflight")
        state, status = self._poll(f"/api/projects/{parent}",
                                   {"awaiting_consent", "failed", "unsupported"},
                                   key=lambda s: (s.get("perception") or {}).get("status"))
        artifact = self._pack(run_dir, "intake", self.root / parent, [
            "perception/preflight/body_source/evidence.json",
            "perception/preflight/body_source/pose-landmarks.json",
            "perception/preflight/body_source/human.xml",
            "perception/preflight/body_source/tracks.gif",
            "perception/status.json", "perception/failure.json", "job.json",
        ], "Pose evidence, the estimated body model and the local check result.")
        if status != "awaiting_consent":
            perception = state.get("perception") or {}
            reason = perception.get("error") or "This clip was not admitted"
            proposal = (state.get("supported_window") or {}).get("proposal")
            if proposal:
                reason += (" · the pipeline proposed its own supported window "
                           f"({proposal['start_s']:.1f}s–{proposal['end_s']:.1f}s) as a retry")
            self._finish(0, "Refused — and the reasons are kept", {"outcome": "not admitted"},
                         artifact, status="failed", error=reason[:400])
            raise RuntimeError(reason)
        self._finish(0, "Admitted — evidence retained and inspectable", {
            "API calls": "0", "kept": "pose, confidence, gaps",
            **({"normalized": "re-encoded to a supported MP4"} if receipt else {})}, artifact)
        return parent

    def _settle(self, identity, limit=90):
        """The portal keeps one worker slot; a new child is refused until it is released."""
        deadline = time.monotonic() + limit
        state = self._request("GET", f"/api/projects/{identity}")
        while time.monotonic() < deadline:
            if not (state.get("perception") or {}).get("busy"):
                return state
            time.sleep(0.5)
            state = self._request("GET", f"/api/projects/{identity}")
        return state

    def _step_world(self, run_dir, parent):
        self._begin(1, "Assembling the simulated body and ground")
        available = (self._settle(parent).get("saved_body") or {})
        if not available.get("available"):
            raise RuntimeError(available.get("reason") or "This clip cannot produce a body world")
        request = {"request_id": secrets.token_hex(16),
                   "preflight_manifest_sha256": available["preflight_manifest_sha256"],
                   "modeling_scope": available["modeling_scope"]}
        child = None
        for attempt in range(12):
            try:
                child = self._request("POST", f"/api/projects/{parent}/saved-body", request)["id"]
                break
            except RuntimeError as error:
                if "worker is finishing" not in str(error) or attempt == 11:
                    raise
                time.sleep(2)
        _own(run_dir, child)
        state, status = self._poll(f"/api/projects/{child}",
                                   {"ready", "failed", "unsupported", "cancelled"})
        if status != "ready":
            raise RuntimeError(state.get("error") or "The world could not be built from this evidence")
        artifact = self._pack(run_dir, "world", self.root / child, [
            "status.json", "job.json", "lab/lab-config.json", "lab/status.json",
            "lab/run/*.json", "lab/run/*.xml", "lab/world/*.xml", "lab/world/*.json",
        ], "The executable world: body model, ground assumptions and build report.")
        report = (state.get("saved_body") or {}).get("report") or {}
        self._finish(1, "World built and loaded", {
            "physics": "MuJoCo", "scope": "body + authored ground",
            "validation": str(report.get("physical_validation", "unverified"))}, artifact)
        return child

    def _step_task(self, run_dir, child):
        self._begin(2, "Attaching the task and preparing characters")
        self._request("GET", f"/labs/{child}/")
        self._poll(f"/labs/{child}/api/status", {"ready", "failed", "cancelled"})
        if not ((self._request("GET", "/api/portal").get("avatars") or {}).get("builtins_ready")):
            self._update(2, detail="Preparing the built-in characters")
            self._request("POST", "/api/avatars/builtins", b"{}")
            self._avatars_idle()
        self._update(2, detail="Attaching characters to this world")
        self._request("POST", f"/api/projects/{child}/avatars/attach", b"{}")
        self._avatars_idle()
        catalog = {row["display_name"]: row["key"] for row in
                   ((self._request("GET", "/api/portal").get("avatars") or {}).get("choices") or [])}
        artifact = self._pack(run_dir, "task", self.root / child, [
            "lab/manifest.json", "lab/run/*.json",
        ], "The frozen task definition and its success gates.")
        self._finish(2, "Task attached with gates frozen before the run", {
            "gates": "frozen", "characters ready": str(len(catalog))}, artifact)
        return f"/labs/{child}/api/", catalog

    def _avatars_idle(self, limit=300):
        deadline = time.monotonic() + limit
        while time.monotonic() < deadline:
            state = (self._request("GET", "/api/portal").get("avatars") or {})
            if state.get("status") != "running":
                return state
            time.sleep(0.6)
        raise RuntimeError("Character preparation did not settle")

    def _appearance(self, lab, asset, zoom=1.6):
        return self._request("POST", lab + "appearance", {
            "asset": asset, "overlay": False, "show_muscles": False,
            "camera": {"yaw": 0.6, "zoom": zoom, "follow": True}})

    def _step_rollout(self, lab, catalog):
        self._begin(3, "Starting the recording")
        robot = next((key for name, key in catalog.items() if "robot" in name.lower()), None)
        other = next((key for name, key in catalog.items()
                      if key != robot and "panda" in name.lower()), None)
        if robot:
            self._appearance(lab, robot)
        session = self._request("POST", lab + "record/start", {})["directory"]
        steps, metrics, ended = 0, {}, False
        while steps < 2000:
            snapshot = self._request("POST", lab + "step",
                                     {"steps": 6, "controller": "reference"})["snapshot"]
            steps += 6
            metrics = snapshot.get("task_metrics") or metrics
            if steps % 30 == 0:
                self._update(3, detail=f"{steps} control steps · tracking error scored live")
            if snapshot.get("terminated") or snapshot.get("truncated"):
                ended = True
                break
            if other and steps >= 120 and metrics.get("observed_substeps", 0):
                break
        self._finish(3, f"{steps} continuous control steps executed", {
            "control steps": str(steps), "controller": "reference",
            "tracking error": f"{metrics.get('angular_rmse_rad', 0):.3f} rad"})
        return session, metrics, (steps, other, ended)

    def _step_avatar(self, run_dir, child, lab, session, metrics, handoff):
        steps, other, ended = handoff
        self._begin(4, "Swapping the character mid-recording")
        swap_at = 0
        if other and not ended:
            self._appearance(lab, other)
            swap_at = steps
        while steps < 2000 and not ended:
            snapshot = self._request("POST", lab + "step",
                                     {"steps": 6, "controller": "reference"})["snapshot"]
            steps += 6
            metrics = snapshot.get("task_metrics") or metrics
            if steps % 30 == 0:
                self._update(4, detail=f"{steps} steps · same physics, new character")
            if snapshot.get("terminated") or snapshot.get("truncated"):
                break
        self._update(4, detail="Exporting the recording and its exact replay")
        self._request("POST", lab + "record/stop", {})
        session_dir = self.root / child / "lab/sessions" / session
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if (session_dir / "avatar-video.mp4").is_file():
                break
            time.sleep(1)
        artifact = self._pack(run_dir, "avatar", session_dir, [
            "recording.json", "avatar-video.mp4", "video.mp4",
            "appearance.json", "appearance-manifest.json", "appearance-events.jsonl",
        ], "The recorded session: rendered video and the mid-take appearance swap it logged.")
        # The rollout's own journal is only complete once the recording closes, so it is
        # packed here and attached back to the step that produced it.
        self._update(3, artifact=self._pack(run_dir, "rollout", session_dir, [
            "events.jsonl", "manifest.json",
        ], "Every control step and the state it produced, scored live against the frozen gates."))
        success = bool(metrics.get("success"))
        failed = metrics.get("failed_gates") or []
        with self.lock:
            self.run["session"] = str(session_dir)
            self.run["controls"] = steps
            self.run["success"] = success
        # avc1 can silently resolve to mp4v where a build lacks libx264, and mp4v renders
        # as a black player. Say so rather than letting a reviewer think the run failed.
        from weft.browser_video import playable_in_browser
        video = session_dir / "avatar-video.mp4"
        metrics = {"swap": f"step {swap_at}" if swap_at else "single character",
                   "task success": "yes" if success else "no",
                   "failed gates": ", ".join(failed) if failed else "none"}
        if video.is_file() and not playable_in_browser(video):
            metrics["video"] = ("recorded, but this machine's encoder produced MPEG-4 Part 2,"
                                " which browsers cannot play — download it to watch")
        self._finish(4, "Character swapped mid-take; physics untouched", metrics, artifact)

    def _step_muse(self, run_dir, child):
        index = 5
        self._begin(index, "Sending construction evidence to Meta Muse")
        muse = self._request("GET", "/api/portal").get("muse") or {}
        if not muse.get("available") or self._muse_calls >= MAX_MUSE_CALLS:
            reason = ("The Meta allowance reserved for this demo is used up"
                      if self._muse_calls >= MAX_MUSE_CALLS
                      else str(muse.get("reason") or "Meta review is not configured on this machine"))
            self._finish(index, "Skipped — and said so plainly", {"outcome": "unavailable"},
                         status="skipped", error=reason[:300])
            return
        try:
            from weft.portal_review import TERMINAL as REVIEW_TERMINAL
            self._request("POST", f"/api/projects/{child}/muse")
            self._muse_calls += 1
            # A real Muse round trip on a full world runs past a minute; the earlier
            # 100-second budget expired while the review was still completing.
            state, status = self._poll(f"/api/projects/{child}", REVIEW_TERMINAL,
                                       key=lambda s: (s.get("muse") or {}).get("status"), limit=300)
        except (RuntimeError, OSError) as error:
            self._finish(index, "Review did not finish inside its demo budget",
                         {"outcome": "skipped"}, status="skipped", error=str(error)[:300])
            return
        artifact = self._pack(run_dir, "muse", self.root / child, ["muse/*.json", "muse/*/*.json"],
                              "What was sent to Meta Muse, what it proposed, and the API receipt.")
        if status != "proposed":
            # The reviewer records why it stopped; a bare "skipped" would hide a real limit.
            saved = {}
            try:
                saved = json.loads((self.root / child / "muse/status.json").read_text())
            except (OSError, ValueError):
                pass
            reason = " · ".join(str(saved[k]) for k in ("message", "error_type") if saved.get(k))
            self._finish(index, "Review did not complete", {"outcome": str(status)},
                         artifact, status="skipped",
                         error=(reason or f"The review ended as {status}")[:300])
            return
        # The judge-visible proof is the model's own output, which lives in the advice
        # file; the project summary only carries lifecycle status.
        advice, cost = {}, None
        try:
            saved = json.loads((self.root / child / "muse/advice/hypotheses.json").read_text())
            advice = saved.get("advice") or {}
            cost = json.loads((self.root / child / "muse/status.json").read_text()).get("cost_usd")
        except (OSError, ValueError):
            pass
        hypotheses = advice.get("hypotheses") or []
        questions = advice.get("unanswered_questions") or []
        metrics = {"model": "Meta Muse", "hypotheses": str(len(hypotheses)),
                   "open questions": str(len(questions)),
                   "applied": "no — proposals are tested, never auto-applied"}
        if cost:
            metrics["API cost"] = f"${cost:.4f}"
        summary = str(advice.get("summary") or "").strip()
        self._finish(index, summary[:240] or "Meta Muse reviewed the evidence and proposed tests",
                     metrics, artifact)

    def _step_create(self, run_dir, child, catalog):
        index = 6
        prompt = (self.run or {}).get("prompt") or ""
        self._begin(index, "Importing a generated character")
        library = self._request("GET", "/api/portal").get("avatar_library") or []
        # Older library packs ship unwelded geometry that reads as cracked in motion;
        # prefer a welded delivery when one is present.
        library = sorted(library, key=lambda row: 0 if "plush" in str(row.get("id", "")).lower()
                         or "welded" in str(row.get("id", "")).lower() else 1)
        imported = None
        for item in library:
            try:
                self._request("POST", "/api/avatars/library", {"id": item["id"]})
                self._avatars_idle()
                imported = item.get("display_name") or item.get("id")
                break
            except (RuntimeError, OSError, KeyError):
                continue
        studio = self._request("GET", "/api/portal").get("avatar_studio") or {}
        # With a prompt, actually ask the studio to generate. Paid work sits behind a
        # recorded spending gate, so a refusal here is the honest outcome, not an error.
        attempt = None
        # Each live generation can spend Meshy credits and agent dollars, so the
        # walkthrough offers it a bounded number of times, not once per judge run.
        budget = self.dir / "generation-attempts.json"
        try:
            used = int(json.loads(budget.read_text()).get("used", 0))
        except (OSError, ValueError, TypeError):
            used = 0
        if prompt and studio.get("available") and used >= MAX_GENERATION_ATTEMPTS:
            attempt = (f"not started · the walkthrough allows {MAX_GENERATION_ATTEMPTS} live"
                       " generations and they are used; the imported character still shows the step")
        elif prompt and studio.get("available"):
            budget.write_text(json.dumps({"used": used + 1}) + "\n")
            self._update(index, detail=f"Asking the studio to generate “{prompt[:80]}”")
            try:  # noqa: SIM105 - the refusal text is the judge-visible result
                self._request("POST", "/api/avatars/create", {
                    "prompt": prompt, "style": "soft matte toy look",
                    "request_id": secrets.token_hex(16),
                    "spend_approval": ("Judge walkthrough: approved up to 35 Meshy credits plus"
                                       " bounded agent calls for this one generation.")}, timeout=60)
                attempt = "requested"
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    rows = (self._request("GET", "/api/portal").get("avatar_studio") or {}).get("creations") or []
                    row = next((r for r in rows if (r.get("prompt") or "").startswith(prompt[:40])), None)
                    if row and row.get("state"):
                        attempt = row["state"]
                        if row.get("error"):
                            attempt += " · " + str(row["error"])[:160]
                        if row["state"] in {"failed", "unsupported", "ready", "adopt_failed"}:
                            break
                    time.sleep(2.5)
            except (RuntimeError, OSError) as error:
                attempt = "refused · " + str(error)[:180]
        detail = ("A generated character is in the chooser and superimposed on this run's physics."
                  " Live generation takes about ten minutes and its own quality gates can refuse a"
                  " result; whatever happened is reported above, not hidden.")
        metrics = {"superimposed": str(imported or "already in the chooser"),
                   "live generation": "experimental" if studio.get("available") else "not configured"}
        if prompt:
            metrics["your prompt"] = prompt[:60]
        if attempt:
            metrics["generation attempt"] = attempt[:120]
        artifact = self._pack(run_dir, "create", self.root / "avatars", [
            "catalog.json", "status.json", "*/delivery.json", "*/request.json",
        ], "The character chooser including the imported generated character, and the studio's"
           " recorded response to this request.")
        self._finish(index, "Generated character ready to superimpose", metrics, artifact)
        self._update(index, detail=detail)

    # -- artifacts, archive and gallery ---------------------------------------
    def _pack(self, run_dir, step_id, base, patterns, note):
        try:
            entries = _collect(base, patterns)
            if not entries:
                return None
            dest = run_dir / "artifacts" / f"{step_id}.zip"
            return {"name": f"weft-{step_id}.zip", "bytes": _zip_bounded(dest, entries, note)}
        except (OSError, ValueError, zipfile.BadZipFile):
            return None

    def artifact(self, step_id, run_id=None):
        # Callers pass (step, run) or (run, step); step ids are a closed set, so detect.
        known = {step["id"] for step in STEPS}
        if step_id not in known and run_id in known:
            step_id, run_id = run_id, step_id
        if not STEP_ID.fullmatch(step_id or ""):
            raise ValueError("Unknown artifact")
        if run_id is None:
            with self.lock:
                run_id = self.run["id"] if self.run else None
        if not run_id or not RUN_ID.fullmatch(run_id):
            raise ValueError("Unknown run")
        path = self.dir / run_id / "artifacts" / f"{step_id}.zip"
        if not path.is_file() or path.is_symlink():
            raise ValueError("That artifact is not available")
        return path

    def _index(self):
        try:
            saved = json.loads((self.dir / "runs.json").read_text())
        except (OSError, ValueError):
            return []
        return [row for row in saved if isinstance(row, dict)] if isinstance(saved, list) else []

    def _archive(self, run_id):
        """Keep a finished run inspectable: its steps, its world and its video."""
        with self.lock:
            run = json.loads(json.dumps(self.run)) if self.run and self.run["id"] == run_id else None
        if not run:
            return
        video = Path(run.get("session") or "") / "avatar-video.mp4"
        record = {
            "id": run_id, "created_at": local_time(), "status": run["status"],
            "title": "Your run · " + (run["source"].get("label")
                                      or ("sample clip" if run["source"]["sample"]
                                          else run["source"]["name"][:40])),
            "caption": ("Built in this session from " +
                        (f"“{run['source']['label']}”." if run["source"].get("label")
                         else "the bundled sample clip." if run["source"]["sample"]
                         else f"“{run['source']['name']}”.")),
            "metrics": " · ".join(filter(None, [
                f"{run.get('controls', 0)} control steps" if run.get("controls") else None,
                "task success" if run.get("success") else None,
                run["status"] if run["status"] != "complete" else None])) or "run recorded",
            "source": "your run", "kind": "studio", "project": run.get("project"),
            "video": str(video) if video.is_file() else None, "steps": run["steps"],
        }
        (self.dir / run_id / "record.json").write_text(json.dumps(record, indent=2) + "\n")
        index = [row for row in self._index() if row.get("id") != run_id]
        index.insert(0, {"id": run_id})
        (self.dir / "runs.json").write_text(json.dumps(index[:MAX_RUNS_KEPT], indent=2) + "\n")
        # The carousel shows curated work only; a run that finished with a video earns
        # its place there beside the seeded demos.
        if record["status"] == "complete" and record["video"]:
            featured = [run_id] + [v for v in self._featured() if v != run_id]
            (self.dir / "featured.json").write_text(
                json.dumps(featured[:MAX_RUNS_KEPT], indent=2) + "\n")

    def _featured(self):
        try:
            saved = json.loads((self.dir / "featured.json").read_text())
        except (OSError, ValueError):
            return []
        return [str(v)[:64] for v in saved] if isinstance(saved, list) else []

    def _record(self, run_id):
        path = self.dir / run_id / "record.json"
        try:
            value = json.loads(path.read_text())
            return value if isinstance(value, dict) else None
        except (OSError, ValueError):
            return None

    def studio(self, run_id):
        if not RUN_ID.fullmatch(run_id or ""):
            raise ValueError("Unknown run")
        record = self._record(run_id)
        if not record:
            raise ValueError("That run is no longer available")
        project = record.get("project")
        # Offer the live world only while its lab is still loadable in this portal.
        if project and not (self.root / project / "lab").is_dir():
            project = None
        return {"id": record["id"], "title": record.get("title", "Inside this run"),
                "caption": record.get("caption", ""), "steps": record.get("steps", []),
                "project": project}

    def gallery(self):
        demos = self._demos()
        return {"demos": demos, "items": demos}

    def _demos(self):
        demos = []
        for row in self._index():
            record = self._record(row.get("id", ""))
            if record and record.get("video") and Path(record["video"]).is_file():
                demos.append({"id": record["id"], "title": record["title"],
                              "caption": record["caption"], "metrics": record["metrics"],
                              "source": record.get("source", "your run"), "kind": "studio",
                              "video_url": f"/api/demo/gallery/{record['id']}/video"})
        for seed in GALLERY_SEEDS:
            if (self.project / seed["video"]).is_file():
                demos.append({"id": seed["id"], "title": seed["title"], "caption": seed["caption"],
                              "metrics": seed["metrics"], "source": seed["source"], "kind": "verified",
                              "video_url": f"/api/demo/gallery/{seed['id']}/video"})
        return demos

    def gallery_video(self, identity):
        if not re.fullmatch(r"[A-Za-z0-9-]{1,64}", identity or ""):
            raise ValueError("Unknown demo")
        for seed in GALLERY_SEEDS:
            if seed["id"] == identity:
                path = self.project / seed["video"]
                if path.is_file():
                    return path
        record = self._record(identity) if RUN_ID.fullmatch(identity) else None
        if record and record.get("video"):
            path = Path(record["video"])
            if path.is_file() and not path.is_symlink():
                return path
        raise ValueError("That demo is not available")
