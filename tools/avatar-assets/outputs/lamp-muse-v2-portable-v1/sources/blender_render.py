"""Render genuine skeletal animation from the authored editable source."""

import json
import sys
from pathlib import Path

import bpy


def render(source, config_path, output, size, portrait_size, seconds, samples, transparent):
    out = Path(output)
    config = json.loads(Path(config_path).read_text())
    bpy.ops.wm.open_mainfile(filepath=source)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = max(samples, 32)
    # A full-resolution denoiser exceeded 2 GiB; the still uses more samples without it.
    scene.cycles.use_denoising = False
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.film_transparent = transparent
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    rig = bpy.data.objects["AvatarRig"]
    rig.animation_data.action = bpy.data.actions[config["states"][0]["id"]]
    scene.frame_set(0)
    scene.render.resolution_x = scene.render.resolution_y = portrait_size
    scene.render.filepath = str(out / "portrait.png")
    bpy.ops.render.render(write_still=True)
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.denoising_use_gpu = False
    scene.render.resolution_x = scene.render.resolution_y = size
    for state in config["states"]:
        folder = out / "frames" / state["id"]
        folder.mkdir(parents=True)
        rig.animation_data.action = bpy.data.actions[state["id"]]
        for i in range(seconds * 24 + 1):
            frame = i * state["duration"] / seconds
            scene.frame_set(int(frame), subframe=frame - int(frame))
            scene.render.filepath = str(folder / f"{i:04d}.png")
            bpy.ops.render.render(write_still=True)
    (out / "render-complete.json").write_text(json.dumps({"blender": bpy.app.version_string,
        "rendered_frames_per_state": seconds * 24 + 1, "video_frames_per_state": seconds * 24,
        "endpoint_extra_frame": True, "duration_seconds": seconds, "fps": 24,
        "samples": samples, "native_pixels": size, "portrait_native_pixels": portrait_size,
        "transparent": transparent, "source_animations_retimed": True}, indent=2) + "\n")


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:]
    render(a[0], a[1], a[2], int(a[3]), int(a[4]), int(a[5]), int(a[6]), a[7] == "true")
