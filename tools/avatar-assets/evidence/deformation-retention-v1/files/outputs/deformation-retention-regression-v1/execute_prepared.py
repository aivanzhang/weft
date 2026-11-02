"""Parent-released single native numeric replay; no renderer or model calls."""

import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"sources/host"))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, read_json, require, sha256, write_json


def main():
    started = time.monotonic()
    prepared = read_json(HERE/"prepared.json")
    require(not (HERE/"execution-started.json").exists(), "Replay output is append-only and already started")
    for row in prepared["frozen_files"]:
        require(sha256(HERE/row["path"]) == row["sha256"], "Frozen replay input changed")
    for row in prepared["original_inputs"]:
        require(sha256(row["path"]) == row["sha256"], "Original diagnostic source changed")
    require(sha256(prepared["blender_executable"]) == prepared["blender_executable_sha256"], "Blender executable changed")
    command = [prepared["blender_executable"], "--background", "--factory-startup", "--disable-autoexec",
               "--python-exit-code", "1", "--threads", "4", "--python", str(HERE/"native_replay.py")]
    require(command == prepared["native_command"], "Replay command changed")
    write_json(HERE/"execution-started.json", {"prepared_sha256": sha256(HERE/"prepared.json"), "command": command})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    job = run_bounded(command, HERE/"native-job.log", HERE, timeout=60-(time.monotonic()-started),
                      memory_mb=2048, output_mb=128)
    write_json(HERE/"native-job.json", job)
    require(job["exit_code"] == 0 and not job["failure"], "Native replay failed; preserve log/monitor")
    report = read_json(HERE/"native-replay.json")
    require(report["source_sha256"] == prepared["model_sha256"] and report["passed_retention_regression"],
            "Replay did not verify corrected witness retention")
    for row in prepared["frozen_files"]:
        require(sha256(HERE/row["path"]) == row["sha256"], "Frozen replay input changed during execution")
    for row in prepared["original_inputs"]:
        require(sha256(row["path"]) == row["sha256"], "Original source changed during execution")
    result = {"schema_version": 1, "status": "native_numeric_regression_passed_asset_still_rejected",
              "source_sha256": prepared["model_sha256"], "passed_retention_regression": True,
              "model_quality_approved": False, "new_renders": 0, "new_paid_calls": 0,
              "elapsed_seconds": time.monotonic()-started, "job": "native-job.json"}
    write_json(HERE/"result.json", result)
    write_json(HERE/"manifest.json", {**result, "artifacts": artifacts(HERE)})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
