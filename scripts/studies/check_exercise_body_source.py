"""One pinned three-source construction comparison; no new inference or dynamic trials."""

import argparse
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import selectors
import signal
import subprocess
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from weft.exercise_body_source import prepare_exercise_body_source_from_evidence
from weft.io import canonical_hash, local_time
from weft.observations import adapt_exercise
from weft.shared_world_lab import _adapted_shared_world, prepare_shared_world_seed
from weft.shared_world_packaging import read_package_file, read_package_manifest, write_package_files


ROOT = Path(__file__).resolve().parents[2]
MAX_SECONDS = 90
CASE_SECONDS = 30
FINALIZE_SECONDS = 4
MAX_BYTES = 384 * 1024 * 1024
STAGE_BYTES = 128 * 1024 * 1024
RESERVE_BYTES = 1024 * 1024
MAX_FILES = 4096
LOG_BYTES = 64 * 1024


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("Study plan paths must be relative to this repository")
    path = ROOT / path
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("Study paths must not pass through links")
    return path


def remaining(deadline):
    value = deadline - time.monotonic()
    if value < 1:
        raise TimeoutError("Body-source study reached its shared deadline")
    return value


def inventory(root, *, reserve=0):
    files, total = {}, 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Study output contains a link")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.stat().st_size
            total += path.stat().st_size
            if len(files) > MAX_FILES or total + reserve > MAX_BYTES:
                raise ValueError("Study exceeds its 4096-file/384-MiB aggregate reservation")
    return files, total


def verify(directory, deadline):
    manifest_raw, files = read_package_manifest(directory)
    total = len(manifest_raw)
    for name, expected in files.items():
        remaining(deadline)
        raw = read_package_file(directory, name, max_bytes=64 * 1024 * 1024, expected_sha256=expected)
        total += len(raw)
    return {"manifest_sha256": sha(manifest_raw), "files_including_manifest": len(files) + 1,
            "bytes_including_manifest": total}


def preflight(plan, deadline):
    if (plan.get("profile") != "exercise-body-source-equivalence-study-v1"
            or plan.get("bounds") != {"seconds": MAX_SECONDS, "external_seconds": 95,
                "case_seconds": CASE_SECONDS, "finalization_seconds": FINALIZE_SECONDS,
                "output_bytes": MAX_BYTES, "workers": 1,
                "maximum_floor_forward_calls": 3, "physics_steps": 0, "inference_calls": 0,
                "video_decodes": 0, "training_episodes": 0, "api_calls": 0}
            or len(plan.get("cases", [])) != 3
            or [row.get("method") for row in plan["cases"]] != ["cli", "api", "api"]):
        raise ValueError("Study requires its exact approved three-case bounds and invocation plan")
    if len({row.get("id") for row in plan["cases"]}) != 3 or any(
            not isinstance(row.get("id"), str) or not re.fullmatch(r"[a-z0-9-]{1,60}", row["id"]) for row in plan["cases"]):
        raise ValueError("Study case identifiers must be distinct safe names")
    versions = {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in
                 ("numpy", "mujoco", "mediapipe", "opencv-contrib-python", "Pillow")}}
    if plan.get("runtime_versions") != versions:
        raise ValueError("Study dependency versions differ from the frozen plan")
    initial = {}
    for name, expected in plan["implementation_sha256"].items():
        remaining(deadline)
        path = safe(name)
        raw = read_package_file(path.parent, path.name, max_bytes=64 * 1024 * 1024, expected_sha256=expected)
        initial["implementation/" + name] = raw
    for case in plan["cases"]:
        remaining(deadline)
        directory = safe(case["input"])
        manifest_raw, files = read_package_manifest(directory)
        if sha(manifest_raw) != case["manifest_sha256"]:
            raise ValueError("Original source seal differs from the frozen plan")
        initial[f"inputs/{case['id']}/manifest.json"] = manifest_raw
        for name, expected in case["retained_sha256"].items():
            remaining(deadline)
            if files.get(name) != expected:
                raise ValueError("Retained pose/body pin differs from the original source seal")
            raw = read_package_file(directory, name, max_bytes=64 * 1024 * 1024, expected_sha256=expected)
            if name in {"human.json", "scene.json", "root-motion.json", "floor-assumption.json"}:
                initial[f"inputs/{case['id']}/{name}"] = raw
    return initial


def cli_body(source, out, case_root, deadline):
    command = [sys.executable, "-m", "weft", "exercise-body", "--evidence-run", str(source),
               "--out", str(out), "--max-seconds", str(remaining(deadline))]
    with (case_root / "cli.stdout.json").open("xb") as stdout, (case_root / "cli.stderr.txt").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True, bufsize=0)
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, stdout)
        selector.register(process.stderr, selectors.EVENT_READ, stderr)
        sizes = {stdout: 0, stderr: 0}
        try:
            while selector.get_map():
                remaining(deadline)
                for key, _ in selector.select(.025):
                    block = os.read(key.fileobj.fileno(), 4096)
                    if not block:
                        selector.unregister(key.fileobj)
                        continue
                    sizes[key.data] += len(block)
                    if sizes[key.data] > LOG_BYTES:
                        raise ValueError("Public CLI output exceeded its 64-KiB-per-stream bound")
                    key.data.write(block)
            process.wait(timeout=remaining(deadline))
            if process.returncode:
                raise RuntimeError(f"Public exercise-body CLI failed with exit {process.returncode}; logs retained")
        finally:
            selector.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
            process.stdout.close(); process.stderr.close()
    response = json.loads((case_root / "cli.stdout.json").read_bytes())
    if response.get("status") != "construction_only" or response.get("simulation_ready") is not False:
        raise ValueError("Public body CLI did not return a construction-only package")
    return {"command": command, "response": response,
            "stderr_bytes": (case_root / "cli.stderr.txt").stat().st_size}


def physical_scene(scene):
    value = deepcopy(scene)
    value.pop("source_binding")
    for obj in value["objects"]:
        obj.pop("provenance", None)
    return value


def run_case(case, out, deadline):
    directory = safe(case["input"])
    root = out / case["id"]
    root.mkdir()
    started = time.monotonic()
    case_deadline = min(deadline, started + CASE_SECONDS)
    row = {"id": case["id"], "status": "failed", "method": case["method"]}
    try:
        inventory(out, reserve=STAGE_BYTES + RESERVE_BYTES)
        if case["method"] == "cli":
            row["cli"] = cli_body(directory, root / "body", root, case_deadline)
        else:
            prepare_exercise_body_source_from_evidence(directory, root / "body", max_seconds=remaining(case_deadline))
        row["body_seal"] = verify(root / "body", case_deadline)
        marker = json.loads((root / "body/body-source.json").read_bytes())
        if marker["work"] != {"video_decodes": 0, "inference_calls": 0, "floor_forward_calls": 1,
                              "physics_steps": 0, "training_episodes": 0, "simulation_renders": 0,
                              "api_calls": 0, "observation_media": "saved bytes reused"}:
            raise ValueError("Body stage performed work outside the approved saved-evidence scope")
        body = json.loads((root / "body/human.json").read_bytes())
        original = json.loads((directory / "human.json").read_bytes())
        row["canonical_body_equal"] = canonical_hash(body) == canonical_hash(original)
        row["derived_equal"] = {name: canonical_hash(json.loads((root / "body" / name).read_bytes())) ==
            canonical_hash(json.loads((directory / name).read_bytes())) for name in
            ("root-motion.json", "floor-assumption.json", "body-frame.json", "spatial-pose.json")}
        exact_names = ("evidence.json", "pose-landmarks.json", "model.json", "identity-audit.json", "run-config.json", "source.json")
        row["exact_original_bytes"] = {name: sha((root / "body" / name).read_bytes()) == case["retained_sha256"][name]
                                       for name in (*exact_names, marker["source_artifact"])}
        evidence = json.loads((directory / "evidence.json").read_bytes())
        clocks = json.loads((root / "body/source-clock.json").read_bytes())
        row["exact_native_clock_samples"] = clocks["samples"] == [
            {key: sample[key] for key in ("frame", "time_s", "decoder_timestamp_s")} for sample in evidence["frames"]]
        observations = json.loads((root / "body/observations.json").read_bytes())
        row["source_slots"] = len(observations["observations"])
        row["sampled_slots"] = len(evidence["frames"])
        row["references"] = len(body["reference"])
        if (not row["canonical_body_equal"] or not all(row["derived_equal"].values())
                or not all(row["exact_original_bytes"].values()) or not row["exact_native_clock_samples"]
                or [sample["frame"] for sample in observations["observations"]] != list(range(evidence["decoded_frames"]))):
            raise ValueError("Construction mismatch retained; no normalization or forced equivalence")
        inventory(out, reserve=STAGE_BYTES + RESERVE_BYTES)
        prepare_shared_world_seed(root / "body", root / "seed", deadline=case_deadline)
        row["seed_seal"] = verify(root / "seed", case_deadline)
        old_observations = adapt_exercise(evidence, scene=json.loads((directory / "scene.json").read_bytes()))
        old_scene, _, _ = _adapted_shared_world(original, old_observations, directory, deadline=case_deadline)
        new_scene = json.loads((root / "seed/scene.json").read_bytes())
        row["physical_world_arithmetic_equal"] = physical_scene(old_scene) == physical_scene(new_scene)
        row["comparison_scope"] = "Exact body/world arithmetic; source-binding and object-source provenance intentionally differ"
        if not row["physical_world_arithmetic_equal"]:
            raise ValueError("Shared-world construction arithmetic differs from the legacy pure factory")
        remaining(case_deadline)
        row["status"] = "passed"
    except Exception as error:
        row["error"] = {"type": type(error).__name__, "message": str(error)[:2000]}
    row["seconds"] = time.monotonic() - started
    write_package_files(root, {"comparison.json": encoded(row)})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan_path = safe(args.plan)
    plan_raw = read_package_file(plan_path.parent, plan_path.name, max_bytes=1024 * 1024)
    plan = json.loads(plan_raw)
    out = safe(args.out)
    if args.out != plan.get("output") or out.exists():
        raise ValueError("Study output must match the frozen plan and must not exist")
    started = time.monotonic()
    deadline = started + MAX_SECONDS
    work_deadline = deadline - FINALIZE_SECONDS
    if any(out.is_relative_to(safe(case["input"])) or safe(case["input"]).is_relative_to(out)
           for case in plan.get("cases", [])):
        raise ValueError("Study output must be outside every immutable input")
    initial = preflight(plan, work_deadline)
    remaining(work_deadline)
    if sum(map(len, initial.values())) + len(plan_raw) > MAX_BYTES - STAGE_BYTES - RESERVE_BYTES:
        raise ValueError("Frozen inputs cannot reserve the first bounded stage")
    out.mkdir(parents=True)
    rows = []
    phase = "preserve_plan"
    try:
        write_package_files(out, {"plan.json": plan_raw, **initial})
        for case in plan["cases"]:
            phase = "case:" + case["id"]
            remaining(work_deadline)
            rows.append(run_case(case, out, work_deadline))
        phase = "recheck_inputs"
        preflight(plan, work_deadline)
        report = {"schema_version": 1, "profile": plan["profile"], "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
                  "created_at": local_time(), "seconds_before_seal": time.monotonic() - started, "plan_sha256": sha(plan_raw),
                  "cases": rows, "physical_validation": "unverified", "paid_calls": 0,
                  "terminal_rule": "Requires the matching final manifest and absence of study-failure.json",
                  "scope": "Construction handoff and byte/arithmetic equivalence on exposed development evidence; not new reconstruction accuracy, control qualification or a final holdout"}
        phase = "finalize"
        files, used = inventory(out, reserve=RESERVE_BYTES)
        hashes = {}
        for name in files:
            remaining(deadline)
            hashes[name] = sha(read_package_file(out, name, max_bytes=64 * 1024 * 1024))
        report_raw = encoded(report)
        hashes["report.json"] = sha(report_raw)
        manifest = encoded({"schema_version": 1, "status": report["status"], "files": hashes})
        if used + len(report_raw) + len(manifest) > MAX_BYTES or len(files) + 2 > MAX_FILES:
            raise ValueError("Study final metadata exceeds its retained-output bound")
        remaining(deadline)
        write_package_files(out, {"report.json": report_raw, "manifest.json": manifest})
        result = {"status": report["status"], "output": str(out), "manifest_sha256": sha(manifest),
                  "cases_passed": sum(row["status"] == "passed" for row in rows), "seconds": time.monotonic() - started}
    except Exception as error:
        result = {"schema_version": 1, "status": "failed", "output": str(out), "phase": phase,
                  "plan_sha256": sha(plan_raw), "seconds": time.monotonic() - started,
                  "completed_cases": rows, "error": {"type": type(error).__name__, "message": str(error)[:2000]},
                  "scope": "Authoritative failed/incomplete study receipt; any retained passed case/report is not an admitted study result. No retries."}
        failure_raw = encoded(result)
        if len(failure_raw) > RESERVE_BYTES:
            raise RuntimeError("Study failure metadata exceeds its reserved capacity") from error
        write_package_files(out, {"study-failure.json": failure_raw})
    print(json.dumps(result))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
