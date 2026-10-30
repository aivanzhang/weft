"""Exercise the real lab control/recording path without a server or new inference."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from weft.io import digest, local_time, write_json
from weft.lab import Lab
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording
from weft.shared_exercise_lab import prepare_shared_exercise_lab
from weft.shared_world_packaging import read_package_file, read_package_manifest


MAX_OUTPUT_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_FILES = 2048
TERMINAL_INPUT_STATUSES = {"ready", "trained", "evaluated"}


def _manifest(root):
    raw, files = read_package_manifest(root)
    if len(raw) > 1024 * 1024:
        raise ValueError("Study input manifest exceeds 1 MiB")
    if not 1 <= len(files) <= 1024:
        raise ValueError("Study input requires a bounded manifest")
    manifest = json.loads(raw)
    if "status" in manifest and (not isinstance(manifest["status"], str)
                                 or manifest["status"] not in TERMINAL_INPUT_STATUSES):
        raise ValueError("Study input requires a supported terminal status or historical absent status")
    return hashlib.sha256(raw).hexdigest(), files


def _bounded(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Live lab integration exceeded its cooperative 120-second bound")


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Live lab integration exceeded its cooperative 120-second bound")
    return remaining


def _check_files(root, files, deadline):
    total = 0
    for name, expected in files.items():
        _bounded(deadline)
        raw = read_package_file(root, name, max_bytes=min(64 * 1024 * 1024, max(1, 128 * 1024 * 1024 - total)),
                                expected_sha256=expected)
        total += len(raw)
        if total > 128 * 1024 * 1024:
            raise ValueError("Study inputs exceed file/aggregate byte bounds")
    _bounded(deadline)


def validate_attempt(final, steps, horizon):
    if type(horizon) is not int or not 1 <= horizon <= 180 or not 1 <= steps <= horizon:
        raise ValueError("Exercise attempt requires actions within the complete declared interval")
    metrics = final["task_metrics"]
    if (not (final["terminated"] or final["truncated"])
            or not (metrics["completed_interval"] or metrics["failure"] is not None)
            or metrics["completed_interval"] and steps != horizon):
        raise ValueError("Exercise attempt must finish its interval or preserve an explicit terminal failure")


def validate_baseline_episode(baseline):
    events = baseline["events"]
    if baseline["scene"].get("profile") != "shared-exercise-recording-v1":
        raise ValueError("Study baseline must preserve a human-only exercise task")
    horizon = baseline["scene"]["exercise_task"]["interval"]["horizon_steps"]
    if (type(horizon) is not int or not 1 <= horizon <= 180 or not 2 <= len(events) <= horizon + 1
            or events[0]["type"] != "start" or any(event["type"] != "step" for event in events[1:])):
        raise ValueError("Matched baseline requires one bounded episode with actions and no resets")
    validate_attempt(events[-1]["snapshot"], len(events) - 1, horizon)
    if events[0]["snapshot"]["resolved_reset_options"] != {"jitter": False, "jitter_scale": 0.0}:
        raise ValueError("Matched baseline must use the exact nominal lab reset protocol")


def validate_resumed_recording(restored, task_sha256, directory):
    saved = next((row for row in restored["sessions"] if row["name"] == directory), None)
    if restored["snapshot"]["task_sha256"] != task_sha256 or saved is None or saved.get("complete") is not True:
        raise ValueError("Resume lost the exact task or newly completed recording")


def _selected_task(source, source_files, task_path):
    if task_path is not None:
        raw = read_package_file(task_path.parent, task_path.name, max_bytes=64 * 1024)
        return json.loads(raw), hashlib.sha256(raw).hexdigest(), "explicit_file"
    if "exercise-task.json" in source_files:
        raw = read_package_file(source, "exercise-task.json", max_bytes=64 * 1024,
                                expected_sha256=source_files["exercise-task.json"])
        return json.loads(raw), hashlib.sha256(raw).hexdigest(), "sealed_input"
    return None, None, "default_authored_task"


def _seal(out, deadline):
    files, paths, size = {}, [], 0
    for path in out.rglob("*"):
        _bounded(deadline)
        if path.is_symlink():
            raise ValueError("Integration output contains an unsealed symbolic link")
        if not path.is_file() or path == out / "manifest.json":
            continue
        paths.append(path)
        size += path.stat().st_size
        if len(paths) >= MAX_OUTPUT_FILES or size > MAX_OUTPUT_BYTES:
            raise ValueError("Integration output exceeds its file/byte bound; preserved for inspection")
    for path in sorted(paths):
        _bounded(deadline)
        files[path.relative_to(out).as_posix()] = digest(path)
    raw = json.dumps({"schema_version": 1, "status": "verified", "files": files},
                     indent=2, sort_keys=True, allow_nan=False) + "\n"
    size += len(raw.encode())
    if size > MAX_OUTPUT_BYTES:
        raise ValueError("Integration manifest exceeds the aggregate byte bound; preserved for inspection")
    _bounded(deadline)
    (out / "manifest.json").write_text(raw)
    _bounded(deadline)
    return size


def _failure_receipt(out, error):
    write_json(out / "failure.json", {"type": type(error).__name__, "message": str(error), "time": local_time()})
    # Failure custody stays bounded even when the partial output exceeded its limits.
    write_json(out / "manifest.json", {"schema_version": 1, "status": "failed",
        "files": {"failure.json": digest(out / "failure.json")},
        "custody": "failure_receipt_only", "unsealed_outputs_preserved": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", type=Path, help="Optional explicit task; otherwise preserve the sealed input task")
    parser.add_argument("--baseline-recording", type=Path,
                        help="Optional complete nominal authored episode for exact matched comparison")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.run.resolve(), args.out.resolve()
    if (source / "run/manifest.json").is_file():
        source = source / "run"
    task_path = args.task.resolve() if args.task is not None else None
    baseline_root = args.baseline_recording.resolve() if args.baseline_recording is not None else None
    inputs = [source, args.run.resolve()] + ([baseline_root] if baseline_root else []) + ([task_path] if task_path else [])
    if out.exists() or any(out == p or out.is_relative_to(p) or p.is_relative_to(out)
                           for p in inputs):
        raise ValueError("Study output must be new and outside saved inputs")
    started, deadline = time.monotonic(), time.monotonic() + 120
    source_hash, source_files = _manifest(source)
    _check_files(source, source_files, deadline)
    baseline, baseline_hash, baseline_files = None, None, {}
    if baseline_root is not None:
        baseline_hash, baseline_files = _manifest(baseline_root)
        _check_files(baseline_root, baseline_files, deadline)
        verify_recording(baseline_root, deadline=deadline)
        baseline = load_recording(baseline_root, deadline=deadline)
        validate_baseline_episode(baseline)
    task, task_hash, task_source = _selected_task(source, source_files, task_path)
    if baseline is not None and task is not None and task != baseline["scene"]["exercise_task"]:
        raise ValueError("Use the exact frozen baseline task")
    seed = baseline["events"][0]["seed"] if baseline is not None else 0
    initial = baseline["events"][0]["snapshot"] if baseline is not None else None
    out.mkdir(parents=True)
    lab = None
    status, report = "failed", None
    timings, batch_sizes, rejected = [], [], []

    try:
        (out / "study.py").write_bytes(Path(__file__).read_bytes())
        write_json(out / "settings.json", {
            "created_at": local_time(), "source": str(source), "source_manifest_sha256": source_hash,
            "task_file_sha256": task_hash, "task_source": task_source,
            "baseline": str(baseline_root) if baseline_root else None,
            "baseline_manifest_sha256": baseline_hash, "baseline_seed": seed,
            "maximum_seconds": 120, "maximum_output_bytes": MAX_OUTPUT_BYTES,
            "maximum_output_files": MAX_OUTPUT_FILES,
            "maximum_actions": 189, "maximum_horizon_steps": 180,
            "controls_before_reset": "Up to 3 manual, 3 coast, 3 authored; retain early failures and reset between modes if needed",
            "comparison": "Exact whole baseline episode" if baseline is not None else "No earlier baseline comparison requested",
            "scope": "Real saved-source lab API integration, not browser/device QA or physical validation",
            "paid_calls": 0, "inference_calls": 0, "training": False,
        })
        remaining = _remaining(deadline)
        if remaining < 1:
            raise TimeoutError("Insufficient time remains to prepare the lab")
        prepared = prepare_shared_exercise_lab(source, out / "lab", task_path=task_path, max_seconds=min(60., remaining))
        write_json(out / "preparation.json", prepared)
        _bounded(deadline)
        lab = Lab(out / "lab")
        ready_started = time.perf_counter()
        ready = lab.summary()
        ready_bytes = len(json.dumps(ready, allow_nan=False).encode())
        ready_ms = (time.perf_counter() - ready_started) * 1000
        if ready.get("control_profile") != "shared_exercise":
            raise ValueError("Prepared lab did not select the explicit exercise task")
        if ready["learned_policy_available"]:
            raise ValueError("Untrained task must not expose a legacy policy")
        if digest(out / "lab/run/scene.json") != source_files["scene.json"]:
            raise ValueError("Lab preparation rewrote the pinned engine scene")
        if task is not None and ready["exercise_task"] != task:
            raise ValueError("Lab preparation replaced the supplied or sealed task")
        if baseline is not None and (ready["scene"] != baseline["scene"]["engine_scene"]
                                     or ready["exercise_task"] != baseline["scene"]["exercise_task"]):
            raise ValueError("Lab scene/task differs from the exact baseline")
        horizon = ready["exercise_task"]["interval"]["horizon_steps"]
        if type(horizon) is not int or not 1 <= horizon <= 180:
            raise ValueError("Live lab study requires the complete bounded exercise interval")
        write_json(out / "initial-status.json", ready)
        warm_status = []
        for _ in range(20):
            _bounded(deadline)
            tick = time.perf_counter()
            json.dumps(lab.summary(), allow_nan=False)
            warm_status.append((time.perf_counter() - tick) * 1000)
        lab.inspection()
        inspection = lab.inspection_future.result(timeout=_remaining(deadline))
        write_json(out / "inspection.json", inspection)
        session = lab.load_session()
        count = session.env.action_count
        if count not in (19, 22):
            raise ValueError("Lab actions are not human-only")
        before = session.snapshot()
        for path, payload in (("/api/actor", {"actor": ready["scene"]["objects"][0]["id"]}),
                              ("/api/step", {"action": 0}),
                              ("/api/step", {"policy": True})):
            try:
                lab.control(path, payload)
            except (ValueError, RuntimeError) as error:
                rejected.append({"endpoint": path, "payload": payload, "message": str(error)})
            else:
                raise ValueError("Prohibited actor/action/policy was admitted")
            if session.snapshot() != before:
                raise ValueError("Rejected control advanced the world")
        lab.control("/api/record/start", {})
        manual = [0.] * count
        manual[min(7, count - 1)] = .1
        expected_controllers, smoke_trials = [], []
        for controller, payload in (("manual", {"action": manual, "steps": 3}),
                                    ("coast", {"controller": "coast", "steps": 3}),
                                    ("authored", {"controller": "reference", "steps": 3})):
            _bounded(deadline)
            current = session.snapshot()
            if current["terminated"] or current["truncated"]:
                lab.control("/api/reset", {"seed": seed})
            result = lab.control("/api/step", payload)
            if not 1 <= len(result["transitions"]) <= 3:
                raise ValueError("Interactive controller did not produce bounded actions")
            last = result["transitions"][-1]
            if len(result["transitions"]) < 3 and not (last["terminated"] or last["truncated"]):
                raise ValueError("Interactive controller stopped without an explicit terminal outcome")
            expected_controllers.extend([controller] * len(result["transitions"]))
            smoke_trials.append({"controller": controller, "steps": len(result["transitions"]),
                "terminated": last["terminated"], "truncated": last["truncated"],
                "completed_interval": last["snapshot"]["task_metrics"]["completed_interval"],
                "task_success": last["snapshot"]["task_metrics"]["success"],
                "failure": last["snapshot"]["task_metrics"]["failure"]})
        reset = lab.control("/api/reset", {"seed": seed})["snapshot"]
        if baseline is not None and reset != initial:
            raise ValueError("In-recording reset differs from the frozen baseline reset")
        compared = []
        expected = baseline["events"][1:] if baseline is not None else None
        limit = len(expected) if expected is not None else horizon
        while len(compared) < limit:
            _bounded(deadline)
            tick = time.perf_counter()
            requested = min(6, limit - len(compared))
            result = lab.control("/api/step", {"controller": "reference", "steps": requested})
            json.dumps(result, allow_nan=False)
            timings.append((time.perf_counter() - tick) * 1000)
            batch_sizes.append(len(result["transitions"]))
            if not 1 <= len(result["transitions"]) <= requested:
                raise ValueError("Lab stopped without a bounded authored action")
            compared.extend(result["transitions"])
            last = result["transitions"][-1]
            if last["terminated"] or last["truncated"]:
                break
        if expected is not None:
            if len(compared) != len(expected):
                raise ValueError("Lab stopped before completing the exact baseline episode")
            for actual, prior in zip(compared, expected, strict=True):
                for field in ("action", "reward", "terminated", "truncated", "snapshot"):
                    if actual[field] != prior[field]:
                        raise ValueError(f"Lab differs from baseline in {field}, event {prior['index']}")
        final = session.snapshot()
        validate_attempt(final, len(compared), horizon)
        expected_controllers.extend(["authored"] * len(compared))
        stopped = lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=_remaining(deadline))
        recording = lab.root / "sessions" / stopped["directory"]
        verified = verify_recording(recording, deadline=deadline)
        replay = exported.get("action_replay")
        if (not isinstance(replay, dict) or replay.get("exact_state_replay") is not True
                or replay.get("events") != verified["events"] or replay.get("steps") != verified["steps"]
                or Path(exported["directory"]).resolve() != recording.resolve()):
            raise ValueError("Exporter did not verify this complete recorded action stream")
        saved = load_recording(recording, deadline=deadline)
        controllers = [event.get("controller") for event in saved["events"] if event["type"] == "step"]
        if controllers != expected_controllers:
            raise ValueError("Journal lost the selected manual/coast/authored controllers")
        if any(any(event["snapshot"]["engine_state"]["normalized_action"][count:]) for event in saved["events"]):
            raise ValueError("Exercise interaction applied a forbidden root or equipment force")
        write_json(out / "final-task-state.json", final)
        write_json(out / "recording-verification.json", {"export": exported, "files": verified, "actions": replay})
        lab.close()
        lab = None
        resumed = Lab(out / "lab")
        try:
            restored = resumed.summary()
            validate_resumed_recording(restored, final["task_sha256"], stopped["directory"])
        finally:
            resumed.close()
        run_integrity = verify_artifacts(out / "lab" / "run")
        _check_files(source, source_files, deadline)
        if _manifest(source)[0] != source_hash:
            raise ValueError("Source manifest changed during lab integration")
        if task_path is not None:
            read_package_file(task_path.parent, task_path.name, max_bytes=64 * 1024, expected_sha256=task_hash)
        if baseline_root is not None:
            _check_files(baseline_root, baseline_files, deadline)
            if _manifest(baseline_root)[0] != baseline_hash:
                raise ValueError("Baseline manifest changed during lab integration")
        _bounded(deadline)
        report = {
            "status": "integration_verified", "prepared_lab": str(out / "lab"),
            "recording": str(recording), "recorded_steps": verified["steps"], "recorded_resets": verified["resets"],
            "events": verified["events"], "video_frames": verified["video_frames"],
            "matched_baseline_steps": len(compared) if baseline is not None else None,
            "exact_baseline_state_outcomes": True if baseline is not None else None,
            "baseline_comparison": "exact_match" if baseline is not None else "not_requested",
            "post_reset_authored_steps": len(compared), "declared_horizon_steps": horizon,
            "smoke_trials": smoke_trials,
            "recorded_controllers": {name: controllers.count(name) for name in sorted(set(controllers))},
            "forbidden_controls_rejected_without_advance": rejected, "recording_actions": replay,
            "task_success": final["task_metrics"]["success"], "failed_gates": final["task_metrics"]["failed_gates"],
            "completed_interval": final["task_metrics"]["completed_interval"],
            "task_failure": final["task_metrics"]["failure"], "terminated": final["terminated"],
            "truncated": final["truncated"],
            "run_integrity": run_integrity, "sources_and_baseline_unchanged": True,
            "control_record_json_ms": {"batch_steps": batch_sizes, "samples": len(timings),
                                       "median": float(np.median(timings)),
                                       "p95": float(np.percentile(timings, 95)), "maximum": max(timings)},
            "ready_status": {"bytes": ready_bytes, "first_load_and_serialize_ms": ready_ms,
                             "warm_serialized_samples": len(warm_status),
                             "warm_median_ms": float(np.median(warm_status)), "warm_maximum_ms": max(warm_status),
                             "scope": "Local server-side metadata/state serialization; no browser, disk-cache eviction or sustained-use claim"},
            "wall_seconds": time.monotonic() - started,
            "scope": "Recorded local Lab API interaction; no browser, new inference, learning or physical validation",
        }
        write_json(out / "report.json", report)
        size = _seal(out, deadline)
        status = "verified"
    except Exception as error:
        if report is not None:
            report["status"] = "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            write_json(out / "report.json", report)
        _failure_receipt(out, error)
        raise
    finally:
        if lab is not None:
            lab.close()
    print(json.dumps({"out": str(out), "manifest_sha256": digest(out / "manifest.json"),
                      "bytes": size, "status": status, "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
