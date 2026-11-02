"""Run the single read-only diagnostic under its released CPU budget."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources/host"))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, require, sha256, write_json


def main():
    require(not (HERE / "execution-started.json").exists(), "Diagnostic already attempted")
    prepared = json.loads((HERE / "prepared.json").read_text())
    for row in prepared["frozen_files"]:
        require(sha256(HERE / row["path"]) == row["sha256"], "Frozen diagnostic changed")
    command = [sys.executable, str(HERE / "diagnose.py")]
    write_json(HERE / "execution-started.json", {"command": command, "prepared_sha256": sha256(HERE / "prepared.json")})
    result = run_bounded(command, HERE / "cpu.log", HERE, timeout=60, memory_mb=512, output_mb=16)
    write_json(HERE / "cpu-job.json", result)
    passed = result["exit_code"] == 0 and not result["failure"]
    write_json(HERE / "manifest.json", {"schema_version": 1,
        "status": "diagnosis_complete_no_repair" if passed else "diagnostic_failed_preserved",
        "native_jobs": 0, "provider_calls": 0, "quality_approved": False, "artifacts": artifacts(HERE)})
    print(json.dumps({"diagnosis_completed": passed, "elapsed_seconds": result["elapsed_seconds"],
                      "sampled_peak_rss_bytes": result["sampled_peak_rss_bytes"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
