"""Bounded coupled-runtime and source-binding study; no browser, server, learning or API."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from weft.io import digest, local_time, write_json
from weft.lab import Lab
from weft.pipeline import verify_artifacts
from weft.recording import verify_recording
from weft.shared_world_lab import adapt_to_shared_world, prepare_shared_world_lab, write_sealed_shared_world_fixture
from weft.shared_world_evidence import verify_shared_world_sources
from weft.shared_world import render_snapshot


def inspect_lab(source, destination):
    packaged = prepare_shared_world_lab(source, destination)
    lab = Lab(destination)
    try:
        summary = lab.summary()
        session = lab.load_session()
        source_check = verify_shared_world_sources(session.scene, destination / "run")
        latencies = []
        for _ in range(30):
            session.reset(1729)
            started = time.perf_counter()
            result = lab.control("/api/step", {"controller": "reference", "steps": 3})
            json.dumps(result, allow_nan=False)
            latencies.append((time.perf_counter() - started) * 1000)
        recordings = []
        for actor in ("human", "ball"):
            lab.control("/api/actor", {"actor": actor})
            lab.control("/api/record/start", {})
            action = {"controller": "reference"} if actor == "human" else {"action": 4}
            for count in (6, 3):
                lab.control("/api/step", {**action, "steps": count})
            lab.control("/api/reset", {})
            lab.control("/api/step", {**action, "steps": 3})
            stopped = lab.control("/api/record/stop", {})
            lab.export_future.result(timeout=30)
            saved = destination / "sessions" / stopped["directory"]
            recordings.append({"actor": actor, "directory": str(saved), "verification": verify_recording(saved)})
        image = render_snapshot(session.snapshot(), session.scene)
        Image.fromarray(image).save(destination / "inspection.png")
        return {"status": "verified_runtime_not_physics", "packaged": packaged,
                "source_binding": source_check, "recordings": recordings,
                "reference_feasibility": summary.get("reference_feasibility"),
                "latency": {"three_steps_plus_json_samples": len(latencies),
                            "p50_ms": float(np.percentile(latencies, 50)),
                            "p95_ms": float(np.percentile(latencies, 95)), "max_ms": max(latencies),
                            "scope": "Local control dispatch and serialization; not browser/UI latency or sustained-use profiling."}}
    finally:
        lab.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-study", type=Path, help="Reuse verified candidates from a preserved earlier integration study")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    source_verified = verify_artifacts(args.human_run)
    if args.candidate_study is not None:
        verify_artifacts(args.candidate_study)
        previous = json.loads((args.candidate_study / "pre-run.json").read_text())
        if previous["source_verification"] != source_verified:
            raise ValueError("Reusable candidates must bind this exact human input manifest")
    write_json(args.out / "pre-run.json", {
        "created_at": local_time(), "hypothesis": "Coupled human/rigid control and exact recording work with complete source bindings.",
        "source": str(args.human_run), "source_verification": source_verified,
        "candidate_study": str(args.candidate_study) if args.candidate_study else None,
        "bounds": {"cases": 2, "recordings_per_case": 2, "timed_three_step_batches": 30,
                   "export_timeout_seconds": 30, "api_calls": 0, "training_runs": 0, "external_cost_usd": 0},
        "scope": "Analytic fixture and source-linked body with authored props; no claim of recovered kettlebell geometry."})
    shutil.copyfile(__file__, args.out / Path(__file__).name)
    snapshot = args.out / "implementation-source"
    snapshot.mkdir()
    for path in sorted((root / "src/weft").rglob("*")):
        if path.is_file() and path.suffix in {".py", ".js", ".html", ".css", ".json"} and "__pycache__" not in path.parts:
            target = snapshot / path.relative_to(root / "src/weft")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    results = []
    for name in ("authored", "source_body_authored_props"):
        started = time.perf_counter()
        candidate = (args.candidate_study or args.out) / f"{name}-candidate"
        try:
            if args.candidate_study is not None:
                verify_artifacts(candidate)
            elif name == "authored":
                write_sealed_shared_world_fixture(candidate)
            else:
                adapt_to_shared_world(args.human_run, candidate)
            row = inspect_lab(candidate, args.out / f"{name}-lab")
        except Exception as error:
            row = {"status": "failed_preserved", "error_type": type(error).__name__, "reason": str(error)}
        row.update(case=name, seconds=time.perf_counter() - started)
        results.append(row)
        write_json(args.out / "results.json", {"cases": results, "physical_validation": "unverified"})
    assert verify_artifacts(args.human_run) == source_verified
    manifest = {"created_at": local_time(), "files": {
        str(p.relative_to(args.out)): digest(p) for p in sorted(args.out.rglob("*")) if p.is_file()}}
    write_json(args.out / "manifest.json", manifest)
    verified = verify_artifacts(args.out)
    print({"cases": [{"case": row["case"], "status": row["status"], "reason": row.get("reason")} for row in results], **verified})


if __name__ == "__main__":
    main()
