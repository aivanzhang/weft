"""Record a pinned policy through the live API and verify its preserved task trajectory."""

import argparse
import json
from pathlib import Path
import time

from weft.io import digest, local_time, write_json
from weft.lab import Lab
from weft.recording import load_recording, verify_recording


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lab", type=Path, required=True)
    parser.add_argument("--baseline-recording", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, baseline_root, out = (path.resolve() for path in (args.lab, args.baseline_recording, args.out))
    if out.exists() or any(out.is_relative_to(path) or path.is_relative_to(out) for path in (root, baseline_root)):
        raise ValueError("Use a new evidence directory outside the lab and baseline")
    started, deadline = time.monotonic(), time.monotonic() + 120
    baseline = load_recording(baseline_root, deadline=deadline)
    if any(event["type"] == "reset" for event in baseline["events"]):
        raise ValueError("Matched baseline requires one complete episode")
    baseline_manifest, run_manifest = digest(baseline_root / "manifest.json"), digest(root / "run/manifest.json")
    out.mkdir(parents=True)
    write_json(out / "settings.json", {"created_at": local_time(), "lab_directory": str(root),
        "run_manifest_sha256": run_manifest, "baseline_directory": str(baseline_root),
        "baseline_manifest_sha256": baseline_manifest, "maximum_seconds": 120,
        "maximum_actions": 8 + len(baseline["events"]) - 1, "paid_calls": 0, "training": False,
        "scope": "Live API policy provenance and exact recording; no browser, inference or new policy evaluation."})
    (out / "study.py").write_bytes(Path(__file__).read_bytes())
    lab, report = None, {"status": "failed"}
    try:
        lab = Lab(root)
        status = lab.summary()
        if (not status["learned_policy_available"] or status["exercise_policy"]["selected_feedback_kind"] != "authored_unchanged"
                or status["scene"] != baseline["scene"]["engine_scene"]
                or status["exercise_task"] != baseline["scene"]["exercise_task"]):
            raise ValueError("This matched study requires the exact unchanged authored selection and baseline task")
        write_json(out / "initial-status.json", status)
        lab.inspection()
        write_json(out / "inspection.json", lab.inspection_future.result(timeout=10))
        before = lab.load_session().snapshot()
        for payload in ({"policy": True, "controller": "coast"}, {"policy": True, "action": 0},
                        {"policy": True, "actor": "sam-2"}):
            try:
                lab.control("/api/step", payload)
            except ValueError:
                pass
            else:
                raise ValueError("An ambiguous or object-control request was accepted")
            if lab.load_session().snapshot() != before:
                raise ValueError("Rejected control advanced the simulation")
        lab.control("/api/record/start", {})
        action = [0.] * len(status["action_contract"]["names"])
        action[0] = .1
        for payload in ({"action": action}, {"controller": "coast"}, {"controller": "reference"}, {"policy": True}):
            result = lab.control("/api/step", {**payload, "steps": 2})
            if len(result["transitions"]) != 2:
                raise ValueError("Mixed-mode trial ended before its declared interaction")
        reset = lab.control("/api/reset", {"seed": baseline["events"][0]["seed"]})["snapshot"]
        if reset != baseline["events"][0]["snapshot"]:
            raise ValueError("Reset differs from the frozen nominal baseline")
        matched, timings = 0, []
        while matched < len(baseline["events"]) - 1:
            if time.monotonic() >= deadline:
                raise TimeoutError("Policy interaction exceeded its cooperative bound")
            tick = time.perf_counter()
            result = lab.control("/api/step", {"policy": True, "steps": min(6, len(baseline["events"]) - 1 - matched)})
            json.dumps(result, allow_nan=False)
            timings.append((time.perf_counter() - tick) * 1000)
            if not result["transitions"]:
                raise ValueError("Policy interaction stopped before the matched baseline")
            for event in result["transitions"]:
                matched += 1
                expected = baseline["events"][matched]
                if any(event[key] != expected[key] for key in ("action", "reward", "terminated", "truncated", "snapshot")):
                    raise ValueError("Selected unchanged policy differs from the preserved baseline")
        final = lab.load_session().snapshot()
        write_json(out / "final-task-state.json", final)
        lab.control("/api/record/stop", {})
        exported = lab.export_future.result(timeout=max(.1, deadline - time.monotonic()))
        recording = Path(exported["directory"])
        integrity = verify_recording(recording)
        events = load_recording(recording, deadline=deadline)["events"]
        controllers = {name: sum(event.get("controller") == name for event in events)
                       for name in ("manual", "coast", "authored", "learned")}
        if not exported["action_replay"]["exact_state_replay"]:
            raise ValueError("Exporter did not verify exact policy actions")
        for event in events:
            if event.get("controller") == "learned" and event["policy_sha256"] != status["exercise_policy"]["policy_sha256"]:
                raise ValueError("Recorded policy identity differs")
        lab.close(); lab = Lab(root)
        resumed = lab.summary()
        if resumed["exercise_policy"] != status["exercise_policy"] or not resumed["sessions"]:
            raise ValueError("Resume lost the pinned policy or recording")
        if digest(root / "run/manifest.json") != run_manifest or digest(baseline_root / "manifest.json") != baseline_manifest:
            raise ValueError("Immutable input manifest changed")
        report = {"status": "verified", "lab_directory": str(root), "recording_directory": str(recording),
            "matched_baseline_steps": matched, "events": len(events), "video_frames": exported["video_frames"],
            "controllers": controllers, "recorded_resets": sum(event["type"] == "reset" for event in events),
            "policy": status["exercise_policy"], "exact_state_replay": exported["action_replay"],
            "recording_integrity": integrity, "task_success": final["task_metrics"]["success"],
            "failed_gates": final["task_metrics"]["failed_gates"], "control_record_json_ms": timings,
            "input_manifests_unchanged": True, "seconds": time.monotonic() - started}
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if lab is not None:
            lab.close()
        write_json(out / "report.json", report)
        write_json(out / "manifest.json", {"schema_version": 1, "status": report["status"], "files": {
            path.relative_to(out).as_posix(): digest(path) for path in sorted(out.rglob("*"))
            if path.is_file() and path.name != "manifest.json"}})
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
