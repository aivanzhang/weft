"""Bound the one parent-released local derivative and independent CPU verifier."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources/host"))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, require, sha256, write_json


def main():
    require(not (HERE / "repair-started.json").exists(), "CPU derivative already attempted")
    prepared = json.loads((HERE / "repair-prepared.json").read_text())
    for row in prepared["frozen_files"]:
        require(sha256(HERE / row["path"]) == row["sha256"], "Frozen CPU preparation changed")
    command = [sys.executable, str(HERE / "repair_and_verify.py")]
    write_json(HERE / "repair-started.json", {"command": command,
        "prepared_sha256": sha256(HERE / "repair-prepared.json"), "native_jobs": 0})
    job = run_bounded(command, HERE / "repair-cpu.log", HERE, timeout=60, memory_mb=768, output_mb=128)
    write_json(HERE / "repair-cpu-job.json", job)
    passed = job["exit_code"] == 0 and not job["failure"]
    write_json(HERE / "diagnosis-manifest.json", {"schema_version": 1,
        "status": "analytic_derivative_verified" if passed else "derivative_failed_preserved",
        "passed": passed, "new_native_jobs": 0, "provider_calls": 0, "quality_approved": False,
        "artifacts": artifacts(HERE)})
    print(json.dumps({"passed": passed, "exit_code": job["exit_code"], "elapsed_seconds": job["elapsed_seconds"],
                      "sampled_peak_rss_bytes": job["sampled_peak_rss_bytes"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
