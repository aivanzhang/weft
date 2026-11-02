"""One parent-released native analytic proof and CPU correction, with no automatic retry."""

import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources/host"))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, read_json, require, sha256, write_json


def preserve_prior():
    rows = []
    for path in sorted((HERE / "preserved").glob("*-inventory.json")):
        inventory = read_json(path)
        root = Path(inventory.get("root", inventory.get("source")))
        files = inventory.get("files", inventory.get("artifacts"))
        for entry in files:
            require(sha256(root / entry["path"]) == entry["sha256"], "Prior artifact changed: " + str(root / entry["path"]))
        rows.append({"root": str(root), "files": len(files), "all_unchanged": True})
    return rows


def main():
    started = time.monotonic()
    require(not (HERE / "execution-started.json").exists(), "Fixture already started; preserve all attempts")
    prepared = read_json(HERE / "prepared.json")
    for row in prepared["frozen_files"]:
        require(sha256(HERE / row["path"]) == row["sha256"], "Frozen fixture input changed")
    require(sha256(prepared["blender_executable"]) == prepared["blender_executable_sha256"], "Blender changed")
    prior = preserve_prior()
    command = [prepared["blender_executable"], "--background", "--factory-startup", "--disable-autoexec",
               "--python-exit-code", "1", "--threads", "4", "--python", str(HERE / "build_fixture.py")]
    require(command == prepared["native_command"], "Prepared native command changed")
    write_json(HERE / "execution-started.json", {"command": command, "prepared_sha256": sha256(HERE / "prepared.json")})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        native = run_bounded(command, HERE / "native.log", HERE,
                             timeout=120 - (time.monotonic() - started), memory_mb=1024, output_mb=128)
        write_json(HERE / "native-job.json", native)
        require(native["exit_code"] == 0 and not native["failure"], "Native fixture failed; inspect preserved witness/log")
        remaining = 120 - (time.monotonic() - started)
        require(remaining > 0, "Aggregate fixture deadline exhausted")
        host = run_bounded([sys.executable, str(HERE / "postprocess.py")], HERE / "host.log", HERE,
                           timeout=min(30, remaining), memory_mb=1024, output_mb=128)
        write_json(HERE / "host-job.json", host)
        require(host["exit_code"] == 0 and not host["failure"], "Independent native-Mikk/PNG export verification failed")
        require(read_json(HERE / "result.json")["status"] == "analytic_native_mikk_fixture_passed", "Fixture did not pass")
        for row in prepared["frozen_files"]:
            require(sha256(HERE / row["path"]) == row["sha256"], "Frozen source changed during proof")
        require(preserve_prior() == prior, "Prior artifact inventory changed")
        elapsed = time.monotonic() - started
        require(elapsed <= 120, "Aggregate native/host fixture deadline exceeded")
        write_json(HERE / "completion.json", {"status": "analytic_native_mikk_fixture_passed", "prior_preservation": prior,
            "elapsed_seconds": elapsed, "native_jobs": 1, "renders": 0, "provider_calls": 0, "quality_approved": False})
        write_json(HERE / "manifest.json", {"status": "analytic_native_mikk_fixture_passed", "artifacts": artifacts(HERE)})
        print(json.dumps({"passed": True, "elapsed_seconds": elapsed, "model_sha256": sha256(HERE / "model.glb")}))
    except Exception as error:
        write_json(HERE / "execution-failure.json", {"status": "failed_preserved", "error": type(error).__name__ + ": " + str(error),
            "automatic_retry": False, "quality_approved": False, "provider_calls": 0})
        write_json(HERE / "failure-manifest.json", {"status": "failed_preserved", "artifacts": artifacts(HERE)})
        raise


if __name__ == "__main__":
    main()
