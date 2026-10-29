"""Pinned video studies with explicit fidelity gates, not a universal accuracy score."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import re
import time

from weft.io import canonical_hash, digest, local_time, write_json
from weft.lab import Lab, prepare_lab
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording


METRICS = {"learned_success_rate", "random_success_rate", "coast_success_rate",
           "learned_minus_random", "reconstruction_rmse_px", "usable_pose_fraction",
           "fully_tracked_pose_fraction", "reference_rmse_rad", "spatial_reference_rmse_rad", "mobile_reference_rmse_rad",
           "root_reference_rmse_units", "floor_penetration_units", "action_effect",
           "recorded_transitions"}


def load_suite(path):
    path = Path(path).resolve()
    if path.stat().st_size > 1_000_000:
        raise ValueError("Suite manifest exceeds 1 MB")
    spec = json.loads(path.read_text())
    if not isinstance(spec, dict) or set(spec) != {"schema_version", "name", "purpose", "cases"} or type(spec["schema_version"]) is not int or spec["schema_version"] != 1:
        raise ValueError("Suite requires schema_version=1, name, purpose, and cases")
    if any(not isinstance(spec[key], str) or not 1 <= len(spec[key].strip()) <= 2000 for key in ("name", "purpose")):
        raise ValueError("Suite name and purpose must be bounded nonempty text")
    if not isinstance(spec["cases"], list) or not 1 <= len(spec["cases"]) <= 24:
        raise ValueError("A bounded suite contains 1–24 cases")
    cases, ids = [], set()
    required = {"id", "profile", "video", "source_sha256", "source_kind", "rights", "conditions", "expected", "gates"}
    optional = {"config", "pose_model", "pose_mode", "body_model", "pose_identity", "episodes", "eval_episodes", "seed"}
    for item in spec["cases"]:
        if not isinstance(item, dict) or not required <= item.keys() or item.keys() - required - optional:
            raise ValueError("Case fields do not match the benchmark contract")
        case = dict(item)
        if not isinstance(case["id"], str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", case["id"]) or case["id"] in ids:
            raise ValueError("Case IDs must be unique short lowercase slugs")
        ids.add(case["id"])
        if not isinstance(case["profile"], str) or case["profile"] not in {"exercise", "planar", "appearance", "rolling"}:
            raise ValueError("Select an implemented profile; arbitrary-video support is not implied")
        if not isinstance(case["source_kind"], str) or case["source_kind"] not in {"physical_recording", "analytic_synthetic", "simulator_synthetic"}:
            raise ValueError("Declare the source's evidence class")
        if not isinstance(case["rights"], str) or not case["rights"].strip():
            raise ValueError("Source rights/provenance must be recorded")
        if not isinstance(case["conditions"], list) or not case["conditions"] or any(not isinstance(x, str) or not x for x in case["conditions"]):
            raise ValueError("Record at least one video condition")
        expected = case["expected"]
        if expected != {"status": "ready"} and not (isinstance(expected, dict) and set(expected) == {"status", "error_contains"} and expected["status"] == "rejected" and isinstance(expected["error_contains"], str) and len(expected["error_contains"].strip()) >= 8):
            raise ValueError("Expected rejection needs a specific error fragment; crashes are not abstention")
        gates = case["gates"]
        if not isinstance(gates, list) or len(gates) > 20:
            raise ValueError("Use a bounded list of explicit metric gates")
        for gate in gates:
            if not isinstance(gate, dict) or set(gate) != {"metric", "operator", "value"} or gate["metric"] not in METRICS or gate["operator"] not in {"min", "max"} or type(gate["value"]) not in (int, float) or not math.isfinite(gate["value"]):
                raise ValueError("Unknown metric or non-finite gate")
        if expected["status"] == "rejected" and gates:
            raise ValueError("Rejected cases cannot claim reconstruction or learning metrics")
        for key, default, maximum in (("episodes", 1000, 5000), ("eval_episodes", 50, 200), ("seed", 1729, 2 ** 31 - 1)):
            value = case.setdefault(key, default)
            if type(value) is not int or not (0 if key == "seed" else 1) <= value <= maximum or (key == "seed" and value == 938401):
                raise ValueError(f"Invalid {key} budget")
        for key in ("video", "config", "pose_model"):
            if key in case:
                if not isinstance(case[key], str) or not case[key]:
                    raise ValueError(f"Expected a local {key} path")
                case[key] = str((path.parent / case[key]).resolve())
                if not Path(case[key]).is_file():
                    raise ValueError(f"Missing local {key}: {case['id']}")
        if Path(case["video"]).stat().st_size > 100_000_000:
            raise ValueError("Source exceeds 100 MB")
        if case.get("config") and Path(case["config"]).stat().st_size > 64_000:
            raise ValueError("Scene config exceeds 64 KB")
        if not isinstance(case["source_sha256"], str) or not re.fullmatch(r"[a-f0-9]{64}", case["source_sha256"]) or digest(case["video"]) != case["source_sha256"]:
            raise ValueError(f"Source identity mismatch: {case['id']}")
        if case.get("config") and case["profile"] not in {"planar", "appearance"}:
            raise ValueError("Only planar and appearance cases accept scene config")
        if case["profile"] == "appearance" and not case.get("config"):
            raise ValueError("Appearance cases require source-pinned region config")
        if "pose_mode" in case and (case["profile"] != "exercise" or case["pose_mode"] not in ("video", "image")):
            raise ValueError("Explicit pose mode must be video or image for exercise")
        if "body_model" in case and (case["profile"] != "exercise" or case["body_model"] not in ("hinge", "spatial", "mobile")):
            raise ValueError("Explicit body model must be hinge or spatial for exercise")
        if "pose_identity" in case and (case["profile"] != "exercise" or case["pose_identity"] not in ("raw", "temporal")):
            raise ValueError("Explicit pose identity must be raw or temporal for exercise")
        cases.append(case)
    return spec, cases


def grade_case(case, status, metrics, error="", *, failure_phase=None, error_type=None):
    expected = case["expected"]
    if expected["status"] == "rejected":
        matched = (status == "failed" and failure_phase == "perception" and error_type == "UnsupportedVideo"
                   and expected["error_contains"] in error)
        return {"outcome": "expected_rejection" if matched else "unexpected_outcome", "passed": matched,
                "gates": [], "accuracy_claim": False}
    if status != "ready":
        return {"outcome": "execution_failed", "passed": False, "gates": [], "accuracy_claim": False}
    gates = []
    for gate in case["gates"]:
        value = metrics.get(gate["metric"])
        available = type(value) in (float, int) and math.isfinite(value)
        passed = available and (value >= gate["value"] if gate["operator"] == "min" else value <= gate["value"])
        gates.append({**gate, "observed": value, "passed": passed, "reason": "measured" if available else "metric unavailable"})
    return {"outcome": "gates_passed" if gates and all(g["passed"] for g in gates) else "gates_failed" if gates else "execution_only",
            "passed": bool(gates) and all(g["passed"] for g in gates), "gates": gates, "accuracy_claim": False}


def _read(root, name):
    path = root / name
    return json.loads(path.read_text()) if path.is_file() else {}


def extract_metrics(root):
    evaluation = _read(root, "rl-evaluation.json")
    policies = evaluation.get("policies", {})
    metrics = {}
    for label, name in (("learned", "q_learning"), ("random", "random"), ("coast", "coast")):
        score = policies.get(name)
        if score and score.get("episodes", 0):
            metrics[label + "_success_rate"] = score["successes"] / score["episodes"]
    if {"learned_success_rate", "random_success_rate"} <= metrics.keys():
        metrics["learned_minus_random"] = metrics["learned_success_rate"] - metrics["random_success_rate"]
    reconstruction = _read(root, "reconstruction.json")
    rmse = reconstruction.get("after", reconstruction.get("selected", {})).get("rmse_px")
    if rmse is not None:
        metrics["reconstruction_rmse_px"] = rmse
    quality = _read(root, "pose-quality.json")
    # Keep unknown schemas unavailable rather than fabricating a passing zero.
    sampled = quality.get("sampled_frames")
    if sampled:
        if "tracked_frames" in quality:
            metrics["usable_pose_fraction"] = quality["tracked_frames"] / sampled
        if "status_counts" in quality:
            metrics["fully_tracked_pose_fraction"] = quality["status_counts"].get("tracked", 0) / sampled
    reference = _read(root, "human-evaluation.json")
    if "reference_servo_rmse_radians" in reference:
        metric = {"exercise-v2": "spatial_reference_rmse_rad", "exercise-v3": "mobile_reference_rmse_rad"}.get(
            reference.get("model_profile"), "reference_rmse_rad")
        metrics[metric] = reference["reference_servo_rmse_radians"]
    if reference.get("model_profile") == "exercise-v3":
        for source, metric in (("root_reference_rmse_model_units", "root_reference_rmse_units"),
                               ("maximum_floor_penetration_model_units", "floor_penetration_units")):
            if source in reference:
                metrics[metric] = reference[source]
    if not reconstruction:
        report = _read(root, "report.json").get("reconstruction", {})
        value = report.get("after", {}).get("rmse_px")
        if value is not None:
            metrics["reconstruction_rmse_px"] = value
    return metrics


def _coordinates(snapshot):
    if "human" in snapshot:
        return list(snapshot["human"]["angles"].values())
    return [coordinate for body in snapshot["objects"] for coordinate in body["position"]]


def probe_interaction(lab):
    session = lab.load_session()
    alternatives = (0, 1, 2)
    states = []
    for action in alternatives:
        session.reset(0)
        for _ in range(6):
            if session.done:
                break
            session.step(action)
        state = session.snapshot()
        values = _coordinates(state)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Non-finite interactive state")
        states.append(values)
    effect = max(math.dist(states[0], row) for row in states[1:])
    lab.control("/api/record/start", {})
    transitions = []
    for action in (1, 2, 0, 1):
        if lab.session.done:
            lab.control("/api/reset", {})
        data = lab.control("/api/step", {"action": action, "steps": 3})
        transitions.extend(data["transitions"])
    recorder, lab.recorder = lab.recorder, None
    recorder.close()
    recording_path = recorder.root
    # A thread timeout cannot kill a stuck encoder; the study uses a bounded child.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    with (lab.root / "benchmark-export.log").open("xb") as log:
        subprocess.run([sys.executable, "-m", "weft", "replay", str(recording_path), "--export"],
                       env=env, stdout=log, stderr=subprocess.STDOUT, timeout=90, check=True)
    verification = verify_recording(recording_path)
    recorded = [event for event in load_recording(recording_path)["events"] if event["type"] == "step"]
    keys = ("action", "reward", "terminated", "truncated", "snapshot")
    simplify = lambda events: [{key: event[key] for key in keys} for event in events]
    if verification["steps"] != len(transitions) or canonical_hash(simplify(recorded)) != canonical_hash(simplify(transitions)):
        raise RuntimeError("Recorded transitions differ from the authoritative API results")
    return {"action_effect": effect, "recorded_transitions": len(transitions)}, {
        "recording": str(recording_path), "export_verification": verification,
        "transitions_sha256": canonical_hash(simplify(transitions)),
        "scope": "Executable simulated interventions and recording integrity, not physical accuracy"}


def run_case(case, output):
    started = time.monotonic()
    result = {"id": case["id"], "profile": case["profile"], "conditions": case["conditions"],
              "source_kind": case["source_kind"], "source_sha256": case["source_sha256"],
              "metrics": {}, "status": "failed", "error": ""}
    lab = None
    phase = "preparation"
    try:
        if digest(case["video"]) != case["source_sha256"]:
            raise ValueError("Source changed after suite validation")
        root = prepare_lab(case["video"], output, profile=case["profile"], config=case.get("config"),
                           pose_model=case.get("pose_model"), pose_mode=case.get("pose_mode", "video"), episodes=case["episodes"],
                           body_model=case.get("body_model", "hinge"),
                           pose_identity=case.get("pose_identity", "raw"),
                           eval_episodes=case["eval_episodes"], seed=case["seed"], source_note=case["rights"])
        lab = Lab(root)
        lab.start_worker()
        lab.worker.wait(timeout=370)
        state = lab.status()
        result["status"] = state["status"]
        result["error"] = state.get("error", {}).get("message", "")
        result["error_type"] = state.get("error", {}).get("type")
        stages = [event.get("stage") for event in state.get("events", []) if event.get("stage") != "failed"]
        result["failure_phase"] = "perception" if stages and stages[-1] in {"observe", "tracking"} else "pipeline"
        phase = "integrity"
        if state["status"] == "ready":
            initial_manifest = digest(root / "run/manifest.json")
            result["verification"] = verify_artifacts(root / "run")
            source = _read(root / "run", "source.json")
            if source.get("sha256", source.get("source_sha256")) != case["source_sha256"]:
                raise ValueError("Copied source does not match the pinned benchmark input")
            result["metrics"] = extract_metrics(root / "run")
            phase = "interaction"
            metrics, interaction = probe_interaction(lab)
            result["metrics"].update(metrics)
            result["interaction"] = interaction
            result["manifest_sha256"] = digest(root / "run" / "manifest.json")
            phase = "integrity"
            result["verification_after_interaction"] = verify_artifacts(root / "run")
            if result["manifest_sha256"] != initial_manifest:
                raise RuntimeError("Run manifest changed during interaction")
    except Exception as error:
        result["status"] = "failed"
        result.update(error=f"{type(error).__name__}: {error}", error_type=type(error).__name__, failure_phase=phase)
    finally:
        if lab:
            try:
                lab.close()
            except Exception as error:
                result.update(status="failed", error=f"Cleanup failed: {error}", error_type=type(error).__name__, failure_phase="cleanup")
    result["grading"] = grade_case(case, result["status"], result["metrics"], result["error"],
                                   failure_phase=result.get("failure_phase"), error_type=result.get("error_type"))
    result["seconds"] = round(time.monotonic() - started, 3)
    write_json(output.parent / (case["id"] + "-result.json"), result)
    return result


def run_suite(manifest, out, workers=2):
    spec, cases = load_suite(manifest)
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError("Use one or two local workers")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "suite-input.json", spec)
    write_json(out / "resolved-cases.json", cases)
    envelope = {"cases": len(cases), "max_parallel_workers": workers,
                "worker_wall_seconds_per_case": 360, "export_wait_seconds_per_case": 90,
                "upper_worker_cpu_seconds": 300 * len(cases), "external_api_usd": 0,
                "recurring_services": 0, "scope": "Local development suite; electricity/storage not priced"}
    write_json(out / "budget.json", envelope)
    started = time.monotonic()
    results = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="weft-benchmark") as executor:
        futures = {executor.submit(run_case, case, out / case["id"]): case for case in cases}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"[{len(results)}/{len(cases)}] {result['id']}: {result['grading']['outcome']}", flush=True)
    results.sort(key=lambda result: next(i for i, case in enumerate(cases) if case["id"] == result["id"]))
    report = {"schema_version": 1, "name": spec["name"], "purpose": spec["purpose"],
              "created_at": local_time(), "seconds": round(time.monotonic() - started, 3),
              "budget": envelope, "cases": results,
              "counts": {"total": len(results), "passed_declared_gates": sum(r["grading"]["outcome"] == "gates_passed" for r in results),
                         "execution_ready": sum(r["status"] == "ready" for r in results),
                         "expected_rejections": sum(r["grading"]["outcome"] == "expected_rejection" for r in results)},
              "limits": ["Development evidence, not an untouched final holdout", "Passes cover only the declared gates",
                         "Simulation success and source fit do not establish physical intervention accuracy",
                         "Unsupported video families remain gaps, not generalized-video success"]}
    write_json(out / "report.json", report)
    lines = [f"# {spec['name']}", "", spec["purpose"], "", "| Case | Source | Outcome | Seconds |", "| --- | --- | --- | --- |"]
    lines.extend(f"| {r['id']} | {r['source_kind']} | {r['grading']['outcome']} | {r['seconds']} |" for r in results)
    lines += ["", "## Evidence limits", ""] + [f"- {limit}" for limit in report["limits"]]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    verification = seal_suite(out)
    return {"directory": str(out), "report": str(out / "report.json"), "counts": report["counts"],
            "verification": verification}


def seal_suite(out):
    """Seal completed study evidence without modifying any nested run or recording."""
    out = Path(out)
    if not (out / "report.json").is_file() or (out / "manifest.json").exists():
        raise ValueError("Only a completed, not-yet-sealed study can be sealed")
    write_json(out / "manifest.json", {"schema": 1, "created_at": local_time(),
               "files": {str(path.relative_to(out)): digest(path)
                         for path in sorted(out.rglob("*")) if path.is_file()}})
    return verify_artifacts(out)
