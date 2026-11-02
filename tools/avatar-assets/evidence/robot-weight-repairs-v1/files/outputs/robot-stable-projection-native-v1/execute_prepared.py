"""Run only this frozen local diagnostic after the parent releases the native slot."""

import json
import os
from pathlib import Path
import sys
import time

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"sources/host"))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, read_json, require, sha256, write_json


def main():
    started = time.monotonic()
    require(not (HERE/"execution-started.json").exists(), "This append-only diagnostic already started")
    prepared = read_json(HERE/"prepared.json")
    require(prepared["status"] == "prepared_not_executed" and prepared["limits"] ==
            {"wall_seconds": 600, "rss_mib": 2048, "output_mib": 256, "cpu_threads": 4}, "Prepared limits changed")
    for item in prepared["frozen_files"]:
        require(sha256(HERE/item["path"]) == item["sha256"], "Frozen diagnostic input changed: "+item["path"])
    for item in prepared["original_inputs"]:
        require(sha256(item["path"]) == item["sha256"], "Original input changed: "+item["path"])
    executable = Path(prepared["blender_executable"])
    require(executable.is_file() and sha256(executable) == prepared["blender_executable_sha256"], "Local Blender executable changed")
    expected = [str(executable), "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
                "--threads", "4", "--python", str(HERE/"native_inspection.py")]
    require(prepared["native_command"] == expected, "Prepared command changed")
    write_json(HERE/"execution-started.json", {"schema_version": 1, "prepared_sha256": sha256(HERE/"prepared.json"),
                                              "command": expected, "new_paid_calls": 0})
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        job = run_bounded(expected, HERE/"native-job.log", HERE, timeout=600-(time.monotonic()-started),
                          memory_mb=2048, output_mb=256)
        write_json(HERE/"native-job.json", job)
        require(job["exit_code"] == 0 and not job["failure"], "Native diagnostic failed; inspect preserved log/monitor")
        report = read_json(HERE/"native-deformation.json")
        inspection = read_json(HERE/"native-inspection.json")
        require(report["source_sha256"] == inspection["source_sha256"] == prepared["model_sha256"],
                "Native evidence refers to a different source")
        require([row["view"] for row in inspection["renders"]] == ["front", "three-quarter", "back", "side"],
                "Rest-view evidence incomplete")
        schedule = read_json(HERE/"inputs/motion-states.json")
        actual = [{"id": row["id"], "seconds": [pose["second"] for pose in row["poses"]]}
                  for row in inspection["motion_states"]]
        require(actual == schedule == [{"id": state["id"], "seconds": state["seconds"]} for state in report["states"]],
                "Rendered and measured action times differ")
        image_rows = inspection["renders"]+[pose for state in inspection["motion_states"] for pose in state["poses"]]
        require(len(image_rows) == 12, "Expected four rest views and eight motion poses")
        for row in image_rows:
            path = HERE/row["path"]
            require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 16*1024*1024
                    and sha256(path) == row["sha256"], "Rendered image hash/size mismatch")
            with Image.open(path) as image:
                require(image.format == "PNG" and image.size == (prepared["resolution"], prepared["resolution"])
                        and image.mode == "RGB" and getattr(image, "n_frames", 1) == 1, "Rendered format/dimensions mismatch")
                image.verify()
        for item in prepared["frozen_files"]:
            require(sha256(HERE/item["path"]) == item["sha256"], "Frozen input changed during execution")
        for item in prepared["original_inputs"]:
            require(sha256(item["path"]) == item["sha256"], "Original input changed during execution")
        result = {"schema_version": 1, "status": "diagnostic_complete_needs_visual_review", "source_sha256": prepared["model_sha256"],
                  "deformation_status": report["status"], "passed_no_sampled_severe_spike": report["passed"],
                  "rest_views": 4, "motion_poses": sum(len(row["seconds"]) for row in schedule),
                  "resolution": prepared["resolution"], "samples": prepared["samples"],
                  "image_format_dimensions_and_hashes_verified": True,
                  "visual_quality_approved": False, "automatic_promotion": False, "new_paid_calls": 0,
                  "source_inputs_unchanged": True, "job": "native-job.json", "elapsed_seconds": time.monotonic()-started}
        write_json(HERE/"result.json", result)
        write_json(HERE/"manifest.json", {**result, "artifacts": artifacts(HERE)})
        print(json.dumps(result, sort_keys=True))
    except Exception as error:
        if not (HERE/"execution-failure.json").exists():
            write_json(HERE/"execution-failure.json", {"schema_version": 1, "status": "failed_preserved",
                                                       "error": type(error).__name__+": "+str(error),
                                                       "new_paid_calls": 0, "automatic_retry": False})
        raise


if __name__ == "__main__":
    main()
