"""Check real four-view camera framing with a bounded, unanimated analytic cube."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent.parent
VIEWS = ("front", "three-quarter", "back", "side")
MIN_MARGIN = .05


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def native(output):
    import bpy
    from bpy.app.handlers import persistent
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import blender_provider_preview as provider_preview
    from blender_verify import source_header

    out = Path(output)
    source = out / "analytic-cube.glb"
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.mesh.primitive_cube_add(size=2, location=(.7, -.35, 1.2))
    bpy.context.object.name = "AnalyticFramingCube"
    bpy.ops.export_scene.gltf(
        filepath=str(source), export_format="GLB", export_materials="NONE",
        export_animations=False, export_extras=False,
    )
    doc = source_header(source)
    if doc.get("animations") or doc.get("skins") or len(doc.get("meshes", [])) != 1:
        raise ValueError("Fixture must be one embedded, unrigged, unanimated cube")
    source_hash = digest(source)
    measurements = []

    # Factory reset inside the real renderer must not remove the measurement hook.
    @persistent
    def measure(scene, _depsgraph=None):
        try:
            bpy.context.view_layer.update()
            meshes = [obj for obj in scene.objects if obj.type == "MESH"]
            corners = sorted({tuple(round(float(value), 7) for value in obj.matrix_world @ Vector(corner))
                              for obj in meshes for corner in obj.bound_box})
            camera = scene.camera
            projected = [world_to_camera_view(scene, camera, Vector(point)) for point in corners]
            xy_min = [min(p[i] for p in projected) for i in (0, 1)]
            xy_max = [max(p[i] for p in projected) for i in (0, 1)]
            margin = min(*xy_min, *(1 - value for value in xy_max))
            extent = max(max(p[i] for p in corners) - min(p[i] for p in corners) for i in range(3))
            old_scale = extent * 1.18
            ratio = camera.data.ortho_scale / old_scale
            legacy = [[.5 + (p[i] - .5) * ratio for i in (0, 1)] for p in projected]
            old_margin = min(*(p[i] for p in legacy for i in (0, 1)),
                             *(1 - p[i] for p in legacy for i in (0, 1)))
            clip_pass = all(camera.data.clip_start < p.z < camera.data.clip_end for p in projected)
            valid = len(meshes) == 1 and len(corners) == 8 and margin >= MIN_MARGIN and clip_pass
            measurements.append({
                "view": Path(scene.render.filepath).stem,
                "status": "passed" if valid else "failed",
                "cube_corner_count": len(corners), "mesh_count": len(meshes),
                "world_corners": corners,
                "camera_corners_normalized_xy_depth_m": [list(p) for p in projected],
                "normalized_xy_min": xy_min, "normalized_xy_max": xy_max,
                "minimum_margin_fraction": margin,
                "minimum_margin_pixels": margin * scene.render.resolution_x,
                "required_margin_fraction": MIN_MARGIN, "all_depths_inside_clip": clip_pass,
                "resolution": [scene.render.resolution_x, scene.render.resolution_y],
                "ortho_scale_m": camera.data.ortho_scale,
                "camera_world_matrix": [list(row) for row in camera.matrix_world],
                "legacy_world_extent_scale_m": old_scale,
                "legacy_minimum_margin_fraction": old_margin,
                "legacy_negative_control": "Analytic projection at the old extent*1.18 scale; no second render",
            })
        except Exception as error:
            measurements.append({"status": "failed", "error": f"{type(error).__name__}: {error}"})

    preview = out / "preview"
    preview.mkdir()
    bpy.app.handlers.render_pre.append(measure)
    try:
        provider_preview.main(source, preview, 128, 2)
    finally:
        bpy.app.handlers.render_pre.remove(measure)
    three_quarter = next((item for item in measurements if item.get("view") == "three-quarter"), {})
    passed = (tuple(item.get("view") for item in measurements) == VIEWS
              and all(item["status"] == "passed" for item in measurements)
              and three_quarter.get("legacy_minimum_margin_fraction", 0) < -.01
              and digest(source) == source_hash)
    report = {
        "schema_version": 1, "provenance": "analytic_fixture",
        "status": "passed" if passed else "failed", "animation_verified": False,
        "scope": "Actual provider preview four-rest-view camera fit, not avatar quality or motion",
        "fixture": {"path": source.name, "sha256": source_hash, "geometry": "Axis-aligned 2m cube",
                    "center_blender_z_up": [.7, -.35, 1.2], "embedded_glb": True,
                    "source_unchanged": digest(source) == source_hash},
        "blender_version": bpy.app.version_string,
        "measurements": measurements,
        "limitations": ["A cube tests projected extent framing, not arbitrary animated or deformed geometry",
                        "The legacy comparison is calculated, not rendered",
                        "No rig, animation, avatar identity, material, fur or visual quality claim"],
    }
    write_json(out / "framing-validation.json", report)
    if not passed:
        raise ValueError("Framing regression failed; inspect preserved measurements")


def launch(output):
    sys.path.insert(0, str(ROOT))
    from avatar_assets.runner import run_bounded

    proposed = Path(output)
    if not proposed.is_absolute():
        proposed = ROOT / proposed
    out = proposed.absolute()
    if out != out.resolve() or not out.is_relative_to(ROOT / "outputs"):
        raise ValueError("Output must be a new, nonsymlink path inside tools/avatar-assets/outputs")
    if not out.parent.is_dir():
        raise ValueError("Output parent must already exist")
    executable = ROOT / ".tools/Blender.app/Contents/MacOS/Blender"
    if not executable.is_file():
        raise ValueError("Existing local Blender is unavailable")
    out.mkdir()
    sources = out / "sources"
    sources.mkdir()
    paths = [Path(__file__).resolve(), ROOT / "scripts/blender_provider_preview.py",
             ROOT / "scripts/blender_verify.py", ROOT / "avatar_assets/runner.py",
             ROOT / "avatar_assets/safety.py"]
    source_records = []
    for path in paths:
        target = sources / path.name
        shutil.copyfile(path, target)
        if digest(path) != digest(target):
            raise ValueError("Source changed during snapshot")
        source_records.append({"path": str(path.relative_to(ROOT)),
                               "snapshot": str(target.relative_to(out)), "sha256": digest(target)})
    command = [str(executable), "--background", "--factory-startup", "--disable-autoexec",
               "--python-exit-code", "1", "--threads", "4", "--python",
               str(sources / "validate_studio_framing.py"), "--", "--native", str(out)]
    write_json(out / "request.json", {
        "schema_version": 1, "provenance": "analytic_fixture", "command": command,
        "sources": source_records, "resolution": 128, "samples": 2, "views": list(VIEWS),
        "limits": {"wall_seconds": 120, "rss_mb": 1024, "output_mb": 64, "render_threads": 4},
        "paid_calls": 0, "provider_calls": 0, "purpose": "Orthographic camera framing regression",
    })
    job = run_bounded(command, out / "native.log", out, timeout=120, memory_mb=1024, output_mb=64)
    write_json(out / "job.json", job)
    validation = out / "framing-validation.json"
    passed = (job.get("exit_code") == 0 and job.get("process_exit_verified") is True
              and not job.get("failure") and validation.is_file()
              and json.loads(validation.read_text())["status"] == "passed")
    artifacts = [{"path": str(path.relative_to(out)), "bytes": path.stat().st_size,
                  "sha256": digest(path)} for path in sorted(out.rglob("*")) if path.is_file()]
    write_json(out / "manifest.json", {
        "schema_version": 1, "status": "passed" if passed else "failed",
        "provenance": "analytic_fixture", "animation_verified": False,
        "sources": source_records, "artifacts": artifacts,
        "process_exit_verified": job.get("process_exit_verified", False),
        "paid_calls": 0, "provider_calls": 0,
        "scope": "Actual four-view renderer projection only, not avatar or Studio acceptance",
    })
    print(json.dumps({"status": "passed" if passed else "failed", "output": str(out),
                      "elapsed_seconds": job["elapsed_seconds"],
                      "sampled_peak_rss_bytes": job["sampled_peak_rss_bytes"],
                      "process_exit_verified": job.get("process_exit_verified", False)}))
    return 0 if passed else 1


if __name__ == "__main__":
    if "--" in sys.argv:
        args = sys.argv[sys.argv.index("--") + 1:]
        if len(args) != 2 or args[0] != "--native":
            raise ValueError("Expected the trusted native fixture mode")
        native(args[1])
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("output", help="New directory beneath outputs/")
        raise SystemExit(launch(parser.parse_args().output))
