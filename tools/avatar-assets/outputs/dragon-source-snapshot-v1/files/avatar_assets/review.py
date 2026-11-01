"""Immutable human/agent review records bound to exact inputs and evidence."""

import shutil
from pathlib import Path

from avatar_assets.build3d import ROOT
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, local_file, new_directory, read_json, require, sha256, write_json


def reload_3d(folder, output, blender=None):
    from avatar_assets.cli import verify_pack
    verify_pack(folder)
    root, out = Path(folder).resolve(), new_directory(output)
    exe = Path(blender) if blender else ROOT / ".tools/Blender.app/Contents/MacOS/Blender"
    source = out / "blender_verify.py"
    shutil.copyfile(ROOT / "scripts/blender_verify.py", source)
    cmd = [str(exe.resolve()), "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
           "--threads", "4", "--python", str(source), "--", str(root / "model.glb"), str(out)]
    job = run_bounded(cmd, out / "reload.log", out, timeout=180, memory_mb=2048)
    write_json(out / "job.json", job)
    report = read_json(out / "independent_reload.json") if (out / "independent_reload.json").exists() else {}
    passed = job["exit_code"] == 0 and not job["failure"] and report.get("status") == "passed"
    result = {"schema_version": 1, "kind": "independent_3d_reload", "status": "passed" if passed else "failed",
              "source_manifest_sha256": sha256(root / "manifest.json"), "source_model_sha256": sha256(root / "model.glb"),
              "job": job, "artifacts": artifacts(out)}
    write_json(out / "manifest.json", result)
    require(passed, f"Native GLB reload failed; evidence preserved at {out}")
    return result


def record_review(folder, review_path, output):
    from avatar_assets.cli import verify_pack
    verify_pack(folder)
    source = Path(folder).resolve()
    review = read_json(review_path)
    require(review.get("schema_version") == 1 and review.get("verdict") in ("accepted", "rejected", "partial"), "Invalid review verdict")
    require(review.get("manifest_sha256") == sha256(source / "manifest.json"), "Review must bind exact manifest hash")
    require(isinstance(review.get("reviewer"), str) and review["reviewer"], "Name the reviewer")
    require(isinstance(review.get("observations"), list) and review["observations"], "Review needs concrete observations")
    require(isinstance(review.get("evidence"), list) and review["evidence"], "Review needs inspected evidence paths")
    evidence = []
    for path in review["evidence"]:
        file = local_file(path)
        evidence.append({"path": str(file), "sha256": sha256(file)})
    out = new_directory(output)
    write_json(out / "review.json", {**review, "evidence_hashes": evidence})
    result = {"schema_version": 1, "kind": "visual_review", "status": review["verdict"],
              "source_manifest_sha256": review["manifest_sha256"], "artifacts": artifacts(out)}
    write_json(out / "manifest.json", result)
    return result
