"""Render and encode avatar media while retaining sources and measured declarations."""

import shutil
import sys
from pathlib import Path

from avatar_assets.build3d import ROOT
from avatar_assets.runner import run_bounded
from avatar_assets.scene import read_scene
from avatar_assets.safety import artifacts, new_directory, preserve_failure, read_json, require, sha256, write_json

PRESETS = {
    "muse": {"size": 720, "portrait": 1600, "seconds": 8, "samples": 8},
    "preview": {"size": 320, "portrait": 640, "seconds": 4, "samples": 8},
}


def export_2d(parent, output, preset="muse", blender=None, size=None, portrait=None, seconds=None, samples=None):
    existed = Path(output).exists()
    try:
        return _export_2d(parent, output, preset, blender, size, portrait, seconds, samples)
    except Exception as error:
        if not existed:
            preserve_failure(output, error)
        raise


def _export_2d(parent, output, preset, blender, size, portrait, seconds, samples):
    from avatar_assets.cli import verify_pack
    verify_pack(parent)
    root = Path(parent).resolve()
    parent_manifest = read_json(root / "manifest.json")
    require(parent_manifest["kind"] == "authored_3d_avatar", "Expected a verified authored 3D pack")
    require(parent_manifest["rig_verification"]["motion"]["animation_verified"], "Rig has no verified animation")
    spec = dict(PRESETS[preset])
    for k, v in (("size", size), ("portrait", portrait), ("seconds", seconds), ("samples", samples)):
        if v is not None:
            spec[k] = v
    require(128 <= spec["size"] <= 1024 and spec["size"] % 2 == 0, "Video size outside 128–1024 even pixels")
    require(128 <= spec["portrait"] <= 2048, "Portrait outside 128–2048 pixels")
    require(1 <= spec["seconds"] <= 20 and 1 <= spec["samples"] <= 64, "Duration/samples exceed limit")
    scene = read_scene(root / "sources/scene.json")
    out = new_directory(output)
    (out / "sources").mkdir()
    for name in ("scene.json", "prompt.txt"):
        shutil.copyfile(root / "sources" / name, out / "sources" / name)
    shutil.copyfile(ROOT / "scripts/blender_render.py", out / "sources/blender_render.py")
    exe = Path(blender) if blender else ROOT / ".tools/Blender.app/Contents/MacOS/Blender"
    cmd = [str(exe.resolve()), "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
           "--threads", "4", "--python", str(out / "sources/blender_render.py"), "--",
           str(root / "source.blend"), str(out / "sources/scene.json"), str(out),
           str(spec["size"]), str(spec["portrait"]), str(spec["seconds"]), str(spec["samples"]), "false"]
    job = run_bounded(cmd, out / "render.log", out, timeout=1200, memory_mb=2048)
    write_json(out / "job.json", job)
    if job["exit_code"] or job["failure"] or not (out / "render-complete.json").exists():
        failure = {"schema_version": 1, "status": "failed", "job": job, "artifacts": artifacts(out)}
        write_json(out / "manifest.json", failure)
        raise ValueError(f"Render failed; attempt preserved at {out}")
    write_json(out / "sources/output-spec.json", spec)
    shutil.copyfile(ROOT / "avatar_assets/packaging.py", out / "sources/packaging.py")
    package_job = run_bounded([sys.executable, "-m", "avatar_assets.packaging", str(out), str(out),
                              str(out / "sources/scene.json"), str(out / "sources/output-spec.json")],
                             out / "packaging.log", out, timeout=180, memory_mb=768)
    write_json(out / "packaging-job.json", package_job)
    if package_job["exit_code"] or package_job["failure"] or not (out / "packaging-result.json").exists():
        write_json(out / "manifest.json", {"schema_version": 1, "status": "failed", "job": job,
                   "packaging_job": package_job, "artifacts": artifacts(out)})
        raise ValueError(f"Packaging failed; attempt preserved at {out}")
    packaged = read_json(out / "packaging-result.json")
    result = {"schema_version": 1, "kind": "rendered_2d_avatar", "status": "needs_visual_review",
              "id": scene["id"], "subject": scene["subject"], "style": scene["style"], "preset": preset,
              "output_spec": spec, "origin": "Native 3D skeletal rendering at 24fps; not stock video or repeated stills",
              "parent_3d_manifest_sha256": sha256(root / "manifest.json"), "parent_3d": str(root),
              "portrait": packaged["portrait"], "clips": packaged["clips"], "job": job, "packaging_job": package_job,
              "has_transparency": False, "reduced_motion": "Use still PNG fallback; do not autoplay video",
              "provider_calls": 0, "provider_cost_usd": 0, "rejected_attempts": [],
              "rights": parent_manifest["rights"], "artifacts": artifacts(out)}
    write_json(out / "manifest.json", result)
    return result
