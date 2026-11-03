"""One deduplicated local repair comparison; lifecycle metadata never rewrites its run."""

import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import time

from weft.io import local_time
from weft.lab_worker import atomic_json

PROFILE = "portal-controller-repair-job-v1"
TERMINAL = frozenset({"complete", "failed", "cancelled", "interrupted", "abstained"})
MAX_SECONDS = 60.
MAX_REPORT_BYTES = 128 * 1024
MAX_MANIFEST_BYTES = 64 * 1024


def _read(path, maximum):
    path = Path(path).absolute()
    if ".." in path.parts or any(item.is_symlink() for item in (path, *path.parents)) or not path.is_file():
        raise ValueError("Repair metadata must be an existing regular unlinked file")
    if path.stat().st_size > maximum:
        raise ValueError("Repair metadata exceeds its read bound")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("Repair metadata grew beyond its read bound")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("Repair metadata must be an object")
    return value, sha256(raw).hexdigest()


def _eligibility(root, *, model_choice=False):
    status, _ = _read(root / "status.json", 64 * 1024)
    if status.get("status") != "ready" or status.get("route") != "exercise":
        return False, "A completed exercise simulation is required."
    review, _ = _read(root / "muse/status.json", 64 * 1024)
    if review.get("status") != "proposed":
        return False, "A completed Muse review is required; no new AI request will be made."
    context, _ = _read(root / "muse/advice/context.json", 32 * 1024)
    proposal, _ = _read(root / "muse/advice/hypotheses.json", 140_000)
    if context.get("simulation_world", {}).get("scene_profile") not in {None, "shared-world-v1", "shared-world-v2", "shared-world-v3"}:
        return False, "This world needs a separately qualified controller adapter; its task will not be changed."
    if proposal.get("source") != "model_proposal" or proposal.get("accepted") is not False:
        return False, "An unchanged, unaccepted Muse proposal is required."
    if (root / "job.json").exists():
        job, _ = _read(root / "job.json", 64 << 10)
        if job.get("kind") == "saved_body" and not model_choice:
            return False, "Body approximations require an explicit new Muse controller-choice request."
    observations, objects = context.get("observations", []), context.get("objects", [])
    hypotheses = proposal.get("advice", {}).get("hypotheses", [])
    if (not isinstance(observations, list) or len(observations) > 12 or not isinstance(objects, list) or len(objects) > 14
            or not isinstance(hypotheses, list) or len(hypotheses) > 6):
        return False, "The saved review does not meet the bounded candidate contract."
    evidence_ids = {row.get("id") for row in observations if isinstance(row, dict) and isinstance(row.get("id"), str)}
    object_ids = {row.get("id") for row in objects if isinstance(row, dict) and isinstance(row.get("id"), str)}
    if model_choice:
        return bool(evidence_ids and object_ids & {"person", "world:human"}), "Ask Muse to choose bounded simulator tests or abstain; all source/task checks run again before the request."
    for row in hypotheses:
        if not isinstance(row, dict):
            continue
        refs, entities = row.get("evidence_ids"), row.get("object_ids")
        if (row.get("status") == "hypothesis" and row.get("concept") in {"control", "articulation"}
                and isinstance(refs, list) and refs and all(isinstance(ref, str) and ref in evidence_ids for ref in refs)
                and isinstance(entities, list) and all(isinstance(entity, str) and entity in object_ids for entity in entities)
                and set(entities) & {"person", "world:human"}):
            return True, "Compare up to three authored controller candidates locally; the original world and policy stay unchanged."
    return False, "No supported evidence-linked control or articulation hypothesis; physical measurements cannot be replaced by simulation."


def availability(root, *, model_choice=False):
    """Small metadata-only preview; the worker independently verifies full source custody."""
    root = Path(root).absolute()
    result = {"available": False, "reason": "A completed exercise simulation and Muse review are required.",
              "estimated_external_api_usd": 0, "maximum_seconds": MAX_SECONDS, "maximum_candidates": 3,
              "requires_new_ai_request": False}
    if model_choice:
        from weft.scene_advisor import ESTIMATE
        result.update(estimated_external_api_usd=ESTIMATE, requires_new_ai_request=True,
                      maximum_seconds=120, maximum_provider_calls=1)
    if (root / "muse/repairs").exists():
        return result | {"reason": "A comparison already exists; it will not be rerun or overwrite its evidence."}
    try:
        available, reason = _eligibility(root, model_choice=model_choice)
        return result | {"available": available, "reason": reason}
    except (OSError, ValueError, TypeError, AttributeError):
        return result


def _report(root, *, manifest_sha256=None):
    run = root / "muse/repairs/run"
    manifest, actual = _read(run / "manifest.json", MAX_MANIFEST_BYTES)
    report, report_sha = _read(run / "report.json", MAX_REPORT_BYTES)
    if (manifest.get("status") != "complete" or not isinstance(manifest.get("files"), dict)
            or manifest["files"].get("report.json") != report_sha
            or manifest_sha256 is not None and actual != manifest_sha256
            or report.get("profile") != "portal-controller-repair-comparison-v1" or report.get("status") != "complete"
            or report.get("applied") is not False or report.get("physical_validation") != "unverified"
            or type(report.get("accepted_in_simulation")) is not bool
            or not isinstance(report.get("candidates"), list) or len(report["candidates"]) > 3):
        raise ValueError("Repair report differs from its completed immutable manifest")
    return report, actual


def read_small_report(root):
    """Read only the pinned summary and manifest, never source video, poses or trial traces."""
    root = Path(root).absolute()
    status, _ = _read(root / "muse/repairs/status.json", 64 * 1024)
    if status.get("status") != "complete" or not isinstance(status.get("manifest_sha256"), str):
        raise ValueError("Repair comparison is not complete")
    return _report(root, manifest_sha256=status["manifest_sha256"])[0]


def _complete(report, manifest, *, recovered=False):
    return {"profile": PROFILE, "status": "complete", "message":
        "A candidate improved the fixed simulator score; nothing was applied." if report["accepted_in_simulation"] else
        "No candidate passed all acceptance checks; the original controller is unchanged.",
        "run": "muse/repairs/run", "report": "muse/repairs/run/report.json", "manifest_sha256": manifest,
        "seconds": report.get("seconds"), "selected_candidate": report.get("selected_candidate"),
        "accepted_in_simulation": report["accepted_in_simulation"], "candidates_tested": len(report["candidates"]),
        "control_steps": report.get("control_steps"), "applied": False, "physical_validation": "unverified",
        "lifecycle_recovered_from_sealed_run": recovered,
        "model_choice": "model_choice" in report, "proposal_available": "model_choice" in report,
        "external_api_usd": report.get("external_api_usd", 0)}


def choice_requested(root):
    """Only an explicit closed request can authorize the separate paid proposal phase."""
    path = Path(root) / "muse/repairs/request.json"
    if not path.exists():
        return False
    value, _ = _read(path, 4096)
    if value != {"schema_version": 1, "profile": "muse-controller-choice-v1"}:
        raise ValueError("Unknown controller-choice request")
    return True


def execute_repair_job(root, *, max_seconds=60, project=None):
    """Never retry a prior attempt, including an interrupted one with partial artifacts."""
    if type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 0 < max_seconds <= MAX_SECONDS:
        raise ValueError("Repair jobs require a positive runtime bound of at most 60 seconds")
    root = Path(root).absolute()
    work = root / "muse/repairs"
    if ".." in root.parts or any(path.is_symlink() for path in (work, *work.parents)) or not work.is_dir():
        raise ValueError("Repair worker requires its existing unlinked queued job directory")
    started = time.monotonic()
    fd = os.open(work / "worker.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0  # The owner alone can publish status or start physics.
        saved, _ = _read(work / "status.json", 64 * 1024)
        if saved.get("status") in TERMINAL:
            return 0 if saved["status"] == "complete" else 1
        if (work / "run").exists():
            try:
                report, manifest = _report(root)
                atomic_json(work / "status.json", _complete(report, manifest, recovered=True))
                return 0
            except (OSError, ValueError):
                atomic_json(work / "status.json", {"profile": PROFILE, "status": "interrupted", "applied": False,
                    "message": "A prior comparison left partial or unverified outputs; preserved without an automatic retry."})
                return 1
        if (work / "attempt.json").exists() or saved.get("status") != "queued":
            atomic_json(work / "status.json", {"profile": PROFILE, "status": "interrupted", "applied": False,
                "message": "A prior worker attempt cannot be resumed safely; existing evidence is preserved."})
            return 1
        choice = False
        try:
            choice = choice_requested(root)
            if (work / "cancel-request.json").exists():
                raise InterruptedError("Comparison cancelled before simulation; the original project is unchanged")
            allowed, reason = _eligibility(root, model_choice=choice)
            if not allowed:
                raise ValueError(reason)
            with (work / "attempt.json").open("x") as stream:
                json.dump({"profile": PROFILE, "started_at": local_time(), "pid": os.getpid(),
                           "maximum_comparison_seconds": float(max_seconds),
                           "maximum_provider_calls": 1 if choice else 0,
                           **({"external_api_usd": 0} if not choice else {})}, stream)
            atomic_json(work / "status.json", {"profile": PROFILE, "status": "running", "applied": False,
                "message": "Testing three fixed authored controller candidates against the unchanged task; no AI calls or policy changes.",
                "started_at": local_time(), "maximum_seconds": float(max_seconds)})
            from weft.portal_repairs import run_review_repairs, request_controller_review, run_controller_review_repairs
            if choice:
                if project is None:
                    raise ValueError("Controller-choice execution requires the canonical project budget")
                atomic_json(work / "status.json", {"profile": PROFILE, "status": "running", "applied": False,
                    "model_choice": True, "proposal_available": False,
                    "message": "Muse is selecting bounded authored simulator tests; no physical repair or automatic application."})
                request_controller_review(root, work / "proposal", project=project, execute=True)
                if (work / "cancel-request.json").exists():
                    raise InterruptedError("Cancelled after proposal; saved provider evidence retained, no simulation")
                atomic_json(work / "status.json", {"profile": PROFILE, "status": "running", "applied": False,
                    "model_choice": True, "proposal_available": True,
                    "message": "Testing Muse's explicit choices against the unchanged task and scorer."})
                run_controller_review_repairs(root, work / "proposal", work / "run", max_seconds=max_seconds)
            else:
                run_review_repairs(root, work / "run", max_seconds=max_seconds - (time.monotonic() - started))
            report, manifest = _report(root)
            atomic_json(work / "status.json", _complete(report, manifest))
            return 0
        except Exception as error:
            from weft.portal_repairs import MuseAbstained
            atomic_json(work / "status.json", {"profile": PROFILE,
                "status": "abstained" if isinstance(error, MuseAbstained) else "cancelled" if isinstance(error, InterruptedError) else "failed",
                "message": str(error)[:500], "error_type": type(error).__name__, "applied": False,
                "physical_validation": "unverified", "partial_evidence_preserved": True,
                "model_choice": choice,
                "proposal_available": choice and (work / "proposal/advice/hypotheses.json").is_file(),
                "seconds": time.monotonic() - started})
            return 1
