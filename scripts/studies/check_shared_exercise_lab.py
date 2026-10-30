"""Exercise the real lab control/recording path without a server or new inference."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from weft.io import digest, local_time, write_json
from weft.lab import Lab
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording
from weft.shared_exercise_lab import prepare_shared_exercise_lab
from weft.shared_exercise_recording import verify_action_replay


ROOT = Path(__file__).resolve().parents[2]
MAX_OUTPUT_BYTES = 256 * 1024 * 1024


def _manifest(root):
    path = root / "manifest.json"
    if path.stat().st_size > 1024 * 1024:
        raise ValueError("Study input manifest exceeds 1 MiB")
    files = json.loads(path.read_text())["files"]
    if not isinstance(files, dict) or not 1 <= len(files) <= 1024:
        raise ValueError("Study input requires a bounded manifest")
    return digest(path), files


def _check_files(root, files):
    total = 0
    for name, expected in files.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Saved input path is invalid: {name}")
        size = path.stat().st_size
        total += size
        if size > 64 * 1024 * 1024 or total > 128 * 1024 * 1024:
            raise ValueError("Study inputs exceed file/aggregate byte bounds")
        if digest(path) != expected:
            raise ValueError(f"Saved input changed: {name}")


def _seal(out, status):
    files = {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob("*"))
             if p.is_file() and p != out / "manifest.json"}
    write_json(out / "manifest.json", {"schema_version": 1, "status": status, "files": files})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--baseline-recording", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source, task_path, baseline_root, out = (p.resolve() for p in
                                           (args.run, args.task, args.baseline_recording, args.out))
    if out.exists() or any(out == p or out.is_relative_to(p) or p.is_relative_to(out)
                           for p in (source, baseline_root)):
        raise ValueError("Study output must be new and outside saved inputs")
    started, deadline = time.monotonic(), time.monotonic() + 120
    source_hash, source_files = _manifest(source)
    baseline_hash, baseline_files = _manifest(baseline_root)
    _check_files(source, source_files)
    _check_files(baseline_root, baseline_files)
    baseline = load_recording(baseline_root, deadline=deadline)
    if baseline["scene"].get("profile") != "shared-exercise-recording-v1":
        raise ValueError("Study baseline must preserve a human-only exercise task")
    if task_path.stat().st_size > 64 * 1024:
        raise ValueError("Study task exceeds 64 KiB")
    task = json.loads(task_path.read_text())
    if task != baseline["scene"]["exercise_task"]:
        raise ValueError("Use the exact frozen baseline task")
    seed = baseline["events"][0]["seed"]
    if any(event["type"] == "reset" for event in baseline["events"]):
        raise ValueError("Matched baseline must have one episode")
    initial = baseline["events"][0]["snapshot"]
    if initial["resolved_reset_options"] != {"jitter": False, "jitter_scale": 0.0}:
        raise ValueError("This study does not silently remove baseline reset jitter")
    out.mkdir(parents=True)
    lab = None
    status = "failed"
    timings, batch_sizes, rejected = [], [], []

    def bounded():
        if time.monotonic() > deadline:
            raise TimeoutError("Live lab integration exceeded its cooperative 120-second bound")

    try:
        (out / "study.py").write_bytes(Path(__file__).read_bytes())
        write_json(out / "settings.json", {
            "created_at": local_time(), "source": str(source), "source_manifest_sha256": source_hash,
            "task_sha256": digest(task_path), "baseline": str(baseline_root),
            "baseline_manifest_sha256": baseline_hash, "baseline_seed": seed,
            "maximum_seconds": 120, "maximum_output_bytes": MAX_OUTPUT_BYTES,
            "controls_before_reset": "3 manual, 3 zero/coast, 3 authored reference",
            "comparison": "Exact whole baseline episode after an in-recording reset",
            "scope": "Real saved-source lab API integration, not browser/device QA or physical validation",
            "paid_calls": 0, "inference_calls": 0, "training": False,
        })
        prepared = prepare_shared_exercise_lab(source, out / "lab", task_path=task_path)
        write_json(out / "preparation.json", prepared)
        bounded()
        lab = Lab(out / "lab")
        ready_started = time.perf_counter()
        ready = lab.summary()
        ready_bytes = len(json.dumps(ready, allow_nan=False).encode())
        ready_ms = (time.perf_counter() - ready_started) * 1000
        if ready.get("control_profile") != "shared_exercise":
            raise ValueError("Prepared lab did not select the explicit exercise task")
        if ready["learned_policy_available"]:
            raise ValueError("Untrained task must not expose a legacy policy")
        if ready["scene"] != baseline["scene"]["engine_scene"]:
            raise ValueError("Lab preparation rewrote the pinned engine scene")
        write_json(out / "initial-status.json", ready)
        warm_status = []
        for _ in range(20):
            tick = time.perf_counter()
            json.dumps(lab.summary(), allow_nan=False)
            warm_status.append((time.perf_counter() - tick) * 1000)
        lab.inspection()
        inspection = lab.inspection_future.result(timeout=max(.1, deadline - time.monotonic()))
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
        for payload in ({"action": manual, "steps": 3}, {"controller": "coast", "steps": 3},
                        {"controller": "reference", "steps": 3}):
            bounded()
            result = lab.control("/api/step", payload)
            if len(result["transitions"]) != 3:
                raise ValueError("Initial interactive actions terminated unexpectedly")
        reset = lab.control("/api/reset", {"seed": seed})["snapshot"]
        if reset != initial:
            raise ValueError("In-recording reset differs from the frozen baseline reset")
        compared = []
        expected = baseline["events"][1:]
        while len(compared) < len(expected):
            bounded()
            tick = time.perf_counter()
            result = lab.control("/api/step", {"controller": "reference", "steps": min(6, len(expected) - len(compared))})
            json.dumps(result, allow_nan=False)
            timings.append((time.perf_counter() - tick) * 1000)
            batch_sizes.append(len(result["transitions"]))
            if not result["transitions"]:
                raise ValueError("Lab stopped before completing the frozen baseline")
            compared.extend(result["transitions"])
        for actual, prior in zip(compared, expected, strict=True):
            for field in ("action", "reward", "terminated", "truncated", "snapshot"):
                if actual[field] != prior[field]:
                    raise ValueError(f"Lab differs from baseline in {field}, event {prior['index']}")
            if any(any(obj["virtual_force"]) for obj in actual["snapshot"]["engine_state"]["objects"]):
                raise ValueError("Exercise interaction applied an object force")
        final = session.snapshot()
        stopped = lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=max(.1, deadline - time.monotonic()))
        recording = lab.root / "sessions" / stopped["directory"]
        verified = verify_recording(recording, deadline=deadline)
        replay = verify_action_replay(recording, max_seconds=min(60., deadline - time.monotonic()))
        saved = load_recording(recording, deadline=deadline)
        controllers = [event.get("controller") for event in saved["events"] if event["type"] == "step"]
        if controllers != ["manual"] * 3 + ["coast"] * 3 + ["authored"] * (3 + len(expected)):
            raise ValueError("Journal lost the selected manual/coast/authored controllers")
        write_json(out / "final-task-state.json", final)
        write_json(out / "recording-verification.json", {"export": exported, "files": verified, "actions": replay})
        lab.close()
        lab = None
        resumed = Lab(out / "lab")
        try:
            restored = resumed.summary()
            if restored["snapshot"]["task_sha256"] != final["task_sha256"]:
                raise ValueError("Resume changed the task identity")
            if not any(row["name"] == stopped["directory"] and row["complete"] for row in restored["sessions"]):
                raise ValueError("Resume lost the completed recording")
        finally:
            resumed.close()
        run_integrity = verify_artifacts(out / "lab" / "run")
        _check_files(source, source_files)
        _check_files(baseline_root, baseline_files)
        bounded()
        report = {
            "status": "integration_verified", "prepared_lab": str(out / "lab"),
            "recording": str(recording), "recorded_steps": verified["steps"], "recorded_resets": verified["resets"],
            "events": verified["events"], "video_frames": verified["video_frames"],
            "matched_baseline_steps": len(compared), "exact_baseline_state_outcomes": True,
            "recorded_controllers": {name: controllers.count(name) for name in sorted(set(controllers))},
            "forbidden_controls_rejected_without_advance": rejected, "recording_actions": replay,
            "task_success": final["task_metrics"]["success"], "failed_gates": final["task_metrics"]["failed_gates"],
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
        status = "verified"
    except Exception as error:
        write_json(out / "failure.json", {"type": type(error).__name__, "message": str(error), "time": local_time()})
        raise
    finally:
        if lab is not None:
            lab.close()
        _seal(out, status)
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        if size > MAX_OUTPUT_BYTES:
            raise ValueError("Integration output exceeds its byte bound; preserved for inspection")
    print(json.dumps({"out": str(out), "manifest_sha256": digest(out / "manifest.json"),
                      "bytes": size, "status": status, "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
