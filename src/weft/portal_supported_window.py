"""Propose and build the longest supported window of a refused exercise source.

A localized defect — an unresolved competing-pose moment or a tracking dropout longer
than the root-reference horizon — currently refuses a whole clip even when most of it
is cleanly tracked. This module generalizes the manual remedy: after a failed local
body preflight, the worker computes the longest window of the already-saved evidence
that satisfies the same frozen constraints, records the proposal with its reasons, and
encodes the windowed derivative beside the preserved failure. An explicit user request
then creates an ordinary child project on that derivative; nothing is re-scored, no
gate changes, and the original refusal stays untouched. Admission of the window is
body-evidence admission, not task success or physical accuracy.
"""

import fcntl
import json
import math
import os
import re
import stat
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from weft.io import local_time
from weft.shared_world_packaging import read_package_file, write_package_files

PROFILE = "portal-supported-window-child-v1"
PROPOSAL = "perception/supported-window/proposal.json"
DERIVATIVE = "perception/supported-window/source.mp4"
REQUEST = "supported-window/request.json"
BODY = "perception/preflight/body_source/"
PARENT_FILES = ("job.json", "status.json", "perception/status.json", "perception/failure.json",
                BODY + "manifest.json")
MAX_SOURCE_BYTES = 37_000_000
SCOPE = ("One ordinary local preflight over an explicitly recorded window of the original video. "
         "The window excludes the saved defect moments under the same frozen constraints; thresholds, "
         "scoring and the original refusal are unchanged. No provider call is authorized.")

# settling_seconds: a defect inside the leading span leaves the controller converging
# against bridged or weak references, which shows up as a start-up tracking peak.
WINDOW_POLICY = {
    "profile": "supported-window-proposal-v1",
    "settling_seconds": 2.0,
    "minimum_window_seconds": 5.0,
    "maximum_window_frames": 500,
    "edge_margin_seconds": 0.12,
}


def _require(condition, message):
    if not condition:
        raise ValueError("Supported window: " + message)


def _json(raw):
    from weft.spatial_scene_inputs import _json as bounded_json
    return bounded_json(raw, 24 << 20)


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


def propose_supported_window(evidence):
    """Pure proposal from saved body evidence; reuses the frozen gap and review verdicts."""
    from weft.exercise_vision import USABLE_STATUSES
    from weft.root_motion import ROOT_MOTION_CONTRACT

    policy = {**WINDOW_POLICY,
              "maximum_reference_gap_seconds": ROOT_MOTION_CONTRACT["maximum_reference_gap_seconds"],
              "minimum_usable_fraction": ROOT_MOTION_CONTRACT["minimum_observed_reference_fraction"]}
    frames = evidence.get("frames")
    fps = evidence.get("fps")
    if (not isinstance(frames, list) or not 12 <= len(frames) <= 4096
            or type(fps) not in (int, float) or not 1 <= fps <= 240):
        return {"proposal": None, "policy": policy, "reasons": ["evidence shape is unsupported"]}
    samples = []
    for row in frames:
        t = row.get("decoder_timestamp_s", row.get("time_s"))
        if type(t) not in (int, float) or not math.isfinite(t):
            return {"proposal": None, "policy": policy, "reasons": ["evidence timestamps are unsupported"]}
        samples.append((float(t), row.get("status"), row.get("id")))
    samples.sort()
    duration = samples[-1][0]
    excluded = []
    # Unresolved competing moments come from the recorded review verdict, never recomputed
    # with different rules; a review that resolved its gaps leaves nothing to exclude.
    review = (evidence.get("quality") or {}).get("competing_hypotheses") or {}
    unresolved_ids = {identifier for gap in review.get("gaps") or []
                      if isinstance(gap, dict) and gap.get("status") == "unresolved"
                      for identifier in gap.get("evidence_ids") or []}
    for t, status, identifier in samples:
        if status == "multiple_people" and (not review or identifier in unresolved_ids):
            excluded.append({"time_s": round(t, 3), "kind": "unresolved_competing_hypotheses"})
    # Dropouts: spans between usable samples beyond the root-reference interpolation horizon.
    usable_times = [t for t, status, _ in samples if status in USABLE_STATUSES]
    if len(usable_times) < 2:
        return {"proposal": None, "policy": policy, "reasons": ["too few usable samples"]}
    horizon = policy["maximum_reference_gap_seconds"]
    for left, right in zip(usable_times, usable_times[1:]):
        if right - left > horizon + 1e-9:
            excluded.append({"time_s": round((left + right) / 2, 3), "kind": "reference_gap",
                             "span": [round(left, 3), round(right, 3)]})
    margin = policy["edge_margin_seconds"]
    cuts = []
    for item in excluded:
        left, right = item.get("span", [item["time_s"], item["time_s"]])
        cuts.append((max(0.0, left - margin), min(duration, right + margin)))
    spans, cursor = [], 0.0
    for left, right in sorted(cuts):
        if left > cursor:
            spans.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < duration:
        spans.append((cursor, duration))
    defect_times = sorted(t for t, status, _ in samples if status not in USABLE_STATUSES)
    settling = policy["settling_seconds"]
    maximum = policy["maximum_window_frames"] / fps
    candidates = []
    for left, right in spans:
        if right - left < policy["minimum_window_seconds"]:
            continue
        clean_starts = [left] + [t + margin for t in defect_times if left <= t < right]
        start = next((s for s in sorted(clean_starts)
                      if s + policy["minimum_window_seconds"] <= right
                      and not any(s <= t < s + settling for t in defect_times)), None)
        settled = start is not None
        start = start if settled else left
        end = min(right, start + maximum)
        inside = [status in USABLE_STATUSES for t, status, _ in samples if start <= t <= end]
        coverage = sum(inside) / len(inside) if inside else 0.0
        if end - start >= policy["minimum_window_seconds"] and coverage >= policy["minimum_usable_fraction"]:
            candidates.append({"start_s": round(start, 2), "end_s": round(end, 2),
                               "seconds": round(end - start, 2), "settled_start": settled,
                               "usable_fraction": round(coverage, 4)})
    if not candidates:
        return {"proposal": None, "policy": policy, "excluded": excluded,
                "reasons": ["no span satisfies the frozen window constraints"]}
    best = sorted(candidates, key=lambda c: (c["settled_start"], c["seconds"]))[-1]
    return {"proposal": best, "policy": policy, "excluded": excluded, "candidates": candidates,
            "scope": ("Longest evidence-supported window under the recorded review verdicts and the "
                      "root-reference horizon; admission of the window is not guaranteed")}


def _encode_window(video, target, *, start_s, end_s):
    from imageio_ffmpeg import get_ffmpeg_exe
    # proposal.json is the only commit point; an orphan derivative from an interrupted
    # attempt is unreferenced and may be remade, so overwrite is safe here.
    command = [get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error", "-i", str(video),
               "-ss", f"{start_s:.2f}", "-t", f"{end_s - start_s:.2f}",
               "-map", "0:v:0", "-an", "-c:v", "libx264", "-crf", "18", "-preset", "fast",
               "-pix_fmt", "yuv420p", "-fps_mode", "passthrough",
               "-video_track_timescale", "1000000", "-threads", "1", str(target)]
    subprocess.run(command, check=True, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    info = target.stat()
    _require(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= MAX_SOURCE_BYTES,
             "windowed derivative is missing or outside the upload bound")
    return {"command_profile": "h264-crf18-native-timing-v1", "bytes": info.st_size,
            "sha256": sha256(target.read_bytes()).hexdigest()}


def prepare_supported_window_proposal(root, *, deadline):
    """Worker-side, after a sealed unpaid body failure; writes the proposal and derivative."""
    root = _directory(root)
    _require(time.monotonic() < deadline, "no time left for the optional proposal")
    work = root / "perception"
    _require((work / "failure.json").is_file() and not (work / "request.json").exists(),
             "an unpaid sealed preflight failure is required")
    _require(not (root / Path(PROPOSAL)).exists(), "a proposal already exists for this failure")
    # The evidence is this worker's own sealed output; bound its bytes, not its structure.
    evidence = json.loads(_read(root, BODY + "evidence.json", maximum=24 << 20))
    _require(isinstance(evidence, dict), "sealed body evidence must be a JSON object")
    result = propose_supported_window(evidence)
    job = _json(_read(root, "job.json", maximum=64 << 10))
    record = {"schema_version": 1, "profile": WINDOW_POLICY["profile"], "computed_at": local_time(),
              "source_sha256": job.get("source_sha256"), "scope": SCOPE, **result}
    folder = root / "perception/supported-window"
    folder.mkdir(mode=0o700, exist_ok=True)
    if result.get("proposal"):
        derived = _encode_window(root / "source.mp4", folder / "source.mp4",
                                 start_s=result["proposal"]["start_s"], end_s=result["proposal"]["end_s"])
        record["derivative"] = derived
    staged = folder / ".proposal.staged"
    staged.write_bytes(_bytes(record))
    os.rename(staged, folder / "proposal.json")
    return record


def supported_window_availability(parent):
    """Structural availability from the small saved proposal; never parses poses here."""
    try:
        parent = _directory(parent)
        record = _json(_read(parent, PROPOSAL, maximum=128 << 10))
        proposal = record.get("proposal")
        if not isinstance(proposal, dict):
            return {"available": False, "reasons": record.get("reasons"),
                    "reason": "No evidence-supported window satisfies the frozen constraints."}
        job = _json(_read(parent, "job.json", maximum=64 << 10))
        if "supported_window" in job or "body_detail" in job:
            return {"available": False, "reason": "Derived attempts cannot chain another derivation."}
        return {"available": True, "parent_id": parent.name, "proposal": proposal,
                "excluded": record.get("excluded", []), "scope": SCOPE,
                "reason": "A recorded window of the saved evidence satisfies the same frozen constraints."}
    except (ValueError, OSError, TypeError, KeyError):
        return {"available": False,
                "reason": "Requires a failed local body preflight with a recorded window proposal."}


def _parent(parent):
    parent = _directory(parent)
    _require(re.fullmatch("[a-f0-9]{32}", parent.name), "exact parent project ID required")
    snapshots = {name: _read(parent, name) for name in PARENT_FILES}
    job, state, status, failure, manifest = (_json(snapshots[n]) for n in PARENT_FILES)
    pin = job.get("source_sha256")
    _require(job.get("id") == parent.name and job.get("kind", "video") == "video"
             and job.get("mode") == "exercise" and job.get("source_file") == "source.mp4"
             and isinstance(pin, str) and re.fullmatch("[a-f0-9]{64}", pin)
             and "supported_window" not in job and "body_detail" not in job and "body_reuse" not in job,
             "original exercise upload required; derived chains are unsupported")
    _require(state.get("status") in {"needs_input", "failed"} and state.get("route") == "exercise"
             and not state.get("lab") and status.get("status") == "failed"
             and failure.get("status") == "failed", "failed local body preflight required")
    _require(not any((parent / name).exists() or (parent / name).is_symlink()
                     for name in ("lab", "equipment", "muse", "perception/request.json",
                                  "perception/preflight-complete.json", "cancel-request.json")),
             "active, paid, selected or completed work cannot be rebound")
    record = _json(_read(parent, PROPOSAL))
    derived = record.get("derivative")
    _require(isinstance(record.get("proposal"), dict) and isinstance(derived, dict)
             and record.get("source_sha256") == pin
             and isinstance(derived.get("sha256"), str) and re.fullmatch("[a-f0-9]{64}", derived["sha256"]),
             "recorded window proposal with its encoded derivative required")
    size = (parent / Path(DERIVATIVE)).stat().st_size
    _require(0 < size <= MAX_SOURCE_BYTES and size == derived.get("bytes"),
             "windowed derivative differs from its record")
    return job, snapshots, record


def create_supported_window_child(root, parent_id, payload, *, lease_fd, existing_jobs, stored_bytes,
                                  reserved_bytes, pending_jobs, deadline):
    """Called under the live portal's lock/lease; queues only small immutable metadata."""
    from weft.portal import MAX_JOBS, MAX_JOB_BYTES, MAX_STORAGE
    from weft.portal_inventory import validate_intent

    _require(time.monotonic() < deadline, "portal is busy; retry the window request")
    root = _directory(root)
    _require(type(parent_id) is str and re.fullmatch("[a-f0-9]{32}", parent_id)
             and type(payload) is dict and set(payload) == {"request_id"}
             and type(payload["request_id"]) is str and re.fullmatch("[a-f0-9]{32}", payload["request_id"]),
             "one request ID is required")
    _require(type(lease_fd) is int and not (root / "portal.lock").is_symlink(), "existing portal lease required")
    lock = os.fstat(lease_fd)
    path = (root / "portal.lock").stat()
    _require((lock.st_dev, lock.st_ino) == (path.st_dev, path.st_ino), "lease does not belong to this portal")
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    parent_job, snapshots, record = _parent(root / parent_id)
    _require(parent_id in existing_jobs and existing_jobs[parent_id] == parent_job,
             "parent differs from the portal inventory")
    request = {"schema_version": 1, "profile": PROFILE, "child_id": uuid4().hex, "parent_id": parent_id,
               "request_id": payload["request_id"], "window": record["proposal"],
               "parent_files": {n: sha256(v).hexdigest() for n, v in snapshots.items()},
               "parent_source_sha256": parent_job["source_sha256"], "source_file": "source.mp4",
               "source_sha256": record["derivative"]["sha256"], "derivative_bytes": record["derivative"]["bytes"],
               "api_calls_authorized": 0}
    for identity, existing in existing_jobs.items():
        binding = existing.get("supported_window")
        if not isinstance(binding, dict):
            _require(existing.get("request_id") != payload["request_id"],
                     "request ID already belongs to another project")
            continue
        if binding.get("parent_id") == parent_id or binding.get("request_id") == payload["request_id"]:
            _require(binding.get("parent_id") == parent_id
                     and existing.get("source_sha256") == request["source_sha256"],
                     "one window attempt per parent; the prior child is preserved")
            return existing
    _require(all(type(n) is int and n >= 0 for n in (stored_bytes, reserved_bytes, pending_jobs))
             and len(existing_jobs) < MAX_JOBS and pending_jobs < 4
             and stored_bytes + reserved_bytes + MAX_JOB_BYTES <= MAX_STORAGE,
             "portal cannot reserve a complete new child; existing evidence is preserved")
    identity = request["child_id"]
    stage = root / (".supported-window-" + identity)
    stage.mkdir(mode=0o700)
    try:
        request_raw = _bytes(request)
        binding = {"profile": PROFILE, "parent_id": parent_id, "request_id": payload["request_id"],
                   "window": record["proposal"], "request_sha256": sha256(request_raw).hexdigest()}
        child = {"id": identity, "name": parent_job.get("name", "Exercise.mp4"), "kind": "video",
                 "mode": "exercise", "ai": False, "created_at": local_time(), "parent_id": parent_id,
                 "source_file": "source.mp4", "source_sha256": request["source_sha256"],
                 "supported_window": binding,
                 "exercise_intent": validate_intent(parent_job.get("exercise_intent"))}
        state = _bytes({"status": "needs_input", "route": "exercise", "stage": "supported_window",
                        "events": [], "scope": SCOPE})
        pre = {"profile": "portal-exercise-perception-v1", "source_sha256": request["source_sha256"],
               "requested_at": local_time(), "api_calls_authorized": 0, "supported_window": binding}
        write_package_files(stage, {"job.json": _bytes(child), "status.json": state, REQUEST: request_raw,
            "perception/parent-status.json": state, "perception/preflight-request.json": _bytes(pre),
            "perception/status.json": _bytes({"status": "queued_preflight", "stage": "supported_window",
                                              "message": SCOPE})})
        _parent(root / parent_id)
        os.rename(stage, root / identity)
        return child
    except Exception as error:
        write_package_files(stage, {"publication-failure.json": _bytes({"published": False,
            "error": str(error)[:1000], "partial_evidence_preserved": True})})
        raise


def prepare_queued_supported_window(child, *, deadline):
    """Run once in the keyless worker: pin and copy the parent's recorded derivative."""
    _require(time.monotonic() < deadline, "no time left to stage the windowed source")
    child = _directory(child)
    job = _json(_read(child, "job.json", maximum=64 << 10))
    binding = job.get("supported_window")
    _require(isinstance(binding, dict) and binding.get("profile") == PROFILE, "window child binding required")
    request = _json(_read(child, REQUEST, pin=binding["request_sha256"]))
    _require(request.get("child_id") == child.name == job.get("id")
             and request.get("parent_id") == binding.get("parent_id")
             and request.get("source_sha256") == job.get("source_sha256"), "child request identity differs")
    _require(not (child / "source.mp4").exists() and not (child / "source.mp4").is_symlink(),
             "started window work cannot be retried")
    parent = child.parent / request["parent_id"]
    parent_job, snapshots, record = _parent(parent)
    _require({n: sha256(v).hexdigest() for n, v in snapshots.items()} == request["parent_files"]
             and record["derivative"]["sha256"] == request["source_sha256"],
             "failed parent or its recorded derivative changed after child queueing")
    video = _read(parent, DERIVATIVE, maximum=MAX_SOURCE_BYTES, pin=request["source_sha256"])
    write_package_files(child, {"source.mp4": video})
    return {}
