"""Render genuine skeletal animation from the authored editable source."""

import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from blender_verify import bind_action, clear_actions


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
    generated = config.get('kind') == 'generated_render'
    rigs = [obj for obj in scene.objects if obj.type == 'ARMATURE']
    if len(rigs) != 1:
        raise ValueError('Rendering requires exactly one skin armature')
    rig = rigs[0]
    rig.data.pose_position = 'POSE'
    actions = [bpy.data.actions[state['id']] for state in config['states']]
    bindings = {a.name: [(rig, next(iter(a.slots), None))] for a in actions}
    def activate(action):
        if generated:
            bind_action(action, bindings, [rig], [rig])
        else:
            rig.animation_data.action = action
    activate(actions[0])
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
        action = bpy.data.actions[state['id']]
        activate(action)
        first, last = map(float, action.frame_range)
        for i in range(seconds * 24 + 1):
            frame = first+i*(last-first)/(seconds*24)
            scene.frame_set(int(frame), subframe=frame - int(frame))
            scene.render.filepath = str(folder / f"{i:04d}.png")
            bpy.ops.render.render(write_still=True)
    (out / "render-complete.json").write_text(json.dumps({"blender": bpy.app.version_string,
        "rendered_frames_per_state": seconds * 24 + 1, "video_frames_per_state": seconds * 24,
        "endpoint_extra_frame": True, "duration_seconds": seconds, "fps": 24,
        "samples": samples, "portrait_samples": max(samples, 32),
        "denoising": {"portrait": False, "video": "OpenImageDenoise CPU"},
        "native_pixels": size, "portrait_native_pixels": portrait_size,
        "transparent": transparent, "source_animations_retimed": True}, indent=2) + "\n")


if __name__ == "__main__":
    a = sys.argv[sys.argv.index("--") + 1:]
    render(a[0], a[1], a[2], int(a[3]), int(a[4]), int(a[5]), int(a[6]), a[7] == "true")
