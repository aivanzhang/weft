"""Finish saved-recording inspection after a study driver failed to encode its PNG."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from weft.io import digest, local_time, write_json
from weft.pipeline import verify_artifacts
from weft.recording import load_recording, verify_recording
from weft.shared_world import SharedWorldEnv, render_snapshot
from weft.shared_world_evidence import verify_shared_world_sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-study", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    original = verify_artifacts(args.source_study)
    failed = json.loads((args.source_study / "results.json").read_text())
    if any(row.get("reason") != "'numpy.ndarray' object has no attribute 'save'" for row in failed["cases"]):
        raise ValueError("Recovery is scoped to the preserved post-recording PNG driver failure")
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.out / Path(__file__).name)
    root = Path(__file__).resolve().parents[2]
    for snapshot in (args.source_study / "implementation-source").rglob("*"):
        if snapshot.is_file() and digest(snapshot) != digest(root / "src/weft" / snapshot.relative_to(args.source_study / "implementation-source")):
            raise ValueError("Runtime changed since the recording study; preserve explicit versioned verification")
    results = []
    for row in failed["cases"]:
        source = args.source_study / f"{row['case']}-lab"
        scene = json.loads((source / "run/scene.json").read_text())
        recordings = []
        image_paths = []
        for path in sorted((source / "sessions").iterdir()):
            if not (path / "recording.json").is_file():
                continue
            verified = verify_recording(path)
            recording = load_recording(path)
            last = recording["events"][-1]["snapshot"]
            name = f"{row['case']}-{recording['scene']['controlled_id']}.png"
            Image.fromarray(render_snapshot(last, recording["scene"])).save(args.out / name)
            image_paths.append(name)
            recordings.append({"directory": str(path), "actor": recording["scene"]["controlled_id"], "verification": verified})
        if len(recordings) != 2:
            raise ValueError("Expected the two already-completed human/object recordings")
        env = SharedWorldEnv(scene)
        timings = []
        try:
            for _ in range(30):
                env.reset(seed=1729)
                started = time.perf_counter()
                for _ in range(3):
                    env.step(env.authored_action())
                json.dumps(env.snapshot(), allow_nan=False)
                timings.append((time.perf_counter() - started) * 1000)
        finally:
            env.close()
        results.append({"case": row["case"], "status": "verified_saved_runtime_not_physics",
                        "lab": str(source), "run_verification": verify_artifacts(source / "run"),
                        "source_binding": verify_shared_world_sources(scene, source / "run"),
                        "recordings": recordings, "inspection_images": image_paths,
                        "reference_feasibility": json.loads((source / "run/reference-feasibility.json").read_text())["summary"],
                        "new_core_timing": {"samples": 30, "steps_per_sample": 3,
                                            "p50_ms": float(np.percentile(timings, 50)), "p95_ms": float(np.percentile(timings, 95)),
                                            "scope": "Fresh core step+snapshot serialization measurement, not UI latency; original unsaved timings are unavailable."}})
    assert verify_artifacts(args.source_study) == original
    write_json(args.out / "report.json", {"created_at": local_time(), "source_study": str(args.source_study),
        "source_verification": original, "cases": results, "api_calls": 0, "training_runs": 0,
        "reused": "Both sealed candidates, packaged labs and all four existing recordings. No source inference, construction or recording repeated.",
        "physical_validation": "unverified"})
    write_json(args.out / "manifest.json", {"files": {str(p.relative_to(args.out)): digest(p) for p in sorted(args.out.rglob('*')) if p.is_file()}})
    print(verify_artifacts(args.out))


if __name__ == "__main__":
    main()
