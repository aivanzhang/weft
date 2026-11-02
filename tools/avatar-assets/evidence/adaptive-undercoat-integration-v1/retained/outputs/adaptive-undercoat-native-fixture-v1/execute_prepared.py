"""Parent-released bounded analytic adapter/export check; no automatic retry."""

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
    require(not (HERE/"execution-started.json").exists(), "Fixture already started; preserve its outputs")
    for row in prepared["frozen_files"]:
        require(sha256(HERE/row["path"]) == row["sha256"], "Frozen fixture input changed: "+row["path"])
    require(sha256(prepared["blender_executable"]) == prepared["blender_executable_sha256"], "Blender executable changed")
    command = [prepared["blender_executable"], "--background", "--factory-startup", "--disable-autoexec",
               "--python-exit-code", "1", "--threads", "4", "--python", str(HERE/"build_fixture.py")]
    require(command == prepared["native_command"], "Prepared native command changed")
    write_json(HERE/"execution-started.json", {"prepared_sha256": sha256(HERE/"prepared.json"), "command": command})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        native = run_bounded(command, HERE/"native.log", HERE, timeout=120-(time.monotonic()-started),
                             memory_mb=1024, output_mb=128)
        write_json(HERE/"native-job.json", native)
        require(native["exit_code"] == 0 and not native["failure"], "Native fixture failed; inspect preserved log/monitor")
        remaining = 120-(time.monotonic()-started)
        require(remaining > 0, "Fixture exhausted its aggregate120-second budget")
        host = run_bounded([sys.executable, str(HERE/"verify_fixture.py")], HERE/"host.log", HERE,
                           timeout=min(30, remaining), memory_mb=1024, output_mb=128)
        write_json(HERE/"host-job.json", host)
        require(host["exit_code"] == 0 and not host["failure"], "Export verification failed; preserve the exact fixture")
        verified = read_json(HERE/"host-verification.json")
        require(verified["status"] == "analytic_adapter_export_verified", "Host verification did not pass")
        for row in prepared["frozen_files"]:
            require(sha256(HERE/row["path"]) == row["sha256"], "Frozen input changed during the fixture")
        result = {"schema_version": 1, "status": "analytic_native_export_fixture_passed",
                  "model_sha256": sha256(HERE/"model.glb"), "selected_uv_tangent_derivatives_verified": True,
                  "quality_approved": False, "complete_avatar_pack_accepted": False, "new_renders": 0,
                  "new_paid_calls": 0, "elapsed_seconds": time.monotonic()-started,
                  "scope": "Small analytic adapter/export fixture only; no generated subject or visual-quality proof."}
        write_json(HERE/"result.json", result)
        write_json(HERE/"manifest.json", {**result, "artifacts": artifacts(HERE)})
        print(json.dumps(result, sort_keys=True))
    except Exception as error:
        write_json(HERE/"execution-failure.json", {"schema_version": 1, "status": "failed_preserved",
                   "error": type(error).__name__+": "+str(error), "quality_approved": False,
                   "automatic_retry": False, "new_paid_calls": 0})
        write_json(HERE/"failure-manifest.json", {"status": "failed_preserved", "artifacts": artifacts(HERE)})
        raise


if __name__ == "__main__":
    main()
