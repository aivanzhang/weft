"""Render saved body references as neutral, single-camera presentation clips."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FPS = 30
SIZE = 640
EXAMPLES = [
    ("bodyweight-squats", "Bodyweight squat", "Follow the hip and knee bend through a full squat.", "Danielflefil", "Bodyweight_Squats.gif"),
    ("incline-press", "Incline press", "Inspect shoulder and elbow motion. The bench and weights are not modeled.", "FitnessScape", "Incline_press_-_exercise_demonstration_video.webm"),
    ("deadlift", "Deadlift", "Compare the hip hinge with the reconstructed torso and leg motion. The bar is not modeled.", "FitnessScape", "Deadlift_-_exercise_demonstration_video.webm"),
    ("jumping-jacks", "Jumping jacks and floor transition", "Follow the change from wide arm-and-leg movement into a floor transition.", "Taco Fleur", "Jumping_jacks_and_burpees.webm"),
    ("overhead-squat", "Overhead squat", "Inspect the relationship between raised arms, the torso and a deep knee bend.", "Taco Fleur / Cavemantraining", "Overhead_Kettlebell_Squat.webm"),
    ("waiters-squat", "Kettlebell waiters squat", "Use the overhead hold and squat as a repeatable body-motion reference. Kettlebell contacts are not modeled.", "Taco Fleur / Cavemantraining", "Double_Overhead_Kettlebell_Waiters_Squat.webm"),
    ("sots-press", "Sots press", "Inspect the shoulder movement of a press from a low squat. The weights are not modeled.", "Taco Fleur", "Kettlebell_Sots_Press.webm"),
    ("lunge-twist", "Lunge and twist", "Compare torso rotation and the changing stance through a lunge.", "Taco Fleur / Cavemantraining", "Kettlebell_Lunge_and_Twist.webm"),
    ("single-leg-squat", "Single-leg squat", "Inspect an asymmetric knee bend without claiming unassisted balance.", "RickyBennison", "Basic_single_leg_squat.webm"),
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_example(identity, hosted):
    if identity in {"lunge-twist", "single-leg-squat", "overhead-squat"}:
        folder = ROOT / ({
            "lunge-twist": ".local/demo-portal-04/d1dd0007c45a4a19a39403542e813188/lab/run",
            "single-leg-squat": ".local/portal/4e0f1fe859664bb5ba788214e129451d/build/learning_workflow/lab/run",
            "overhead-squat": ".local/tracking-study-01/overhead-squat-06-assisted-lab/run",
        }[identity])
        raw = (folder / "human-reference.json").read_bytes()
        source = folder / ("source.webm" if identity == "single-leg-squat" else "source.mp4")
        return json.loads(raw)["frames"], source, 0, {"reference_sha256": sha(raw), "input": identity}
    folder = hosted / identity
    with zipfile.ZipFile(folder / "environment.zip") as archive:
        raw = archive.read("appearance/human-reference.json")
        receipt = json.loads(archive.read("appearance/receipt.json"))
        offset = float((receipt.get("supported_window") or {}).get("start_s", 0))
        if "supported-window.json" in archive.namelist():
            offset += float(json.loads(archive.read("supported-window.json"))["start_s"])
        provenance = {"reference_sha256": sha(raw), "archive_sha256": sha((folder / "environment.zip").read_bytes()),
                      "input": identity, "reference_member": "appearance/human-reference.json"}
        return json.loads(raw)["frames"], folder / "source.mp4", offset, provenance


def render_reference(frames, destination, poster, *, duration):
    segments = frames[0]["human"]["segments"]
    geometries = "".join(f'<body mocap="true"><geom name="part{i}" type="{"sphere" if part["id"] == "head" else "capsule"}" size="{part["radius"]} .1" rgba=".24 .43 .43 1" material="body"/></body>'
                         for i, part in enumerate(segments))
    xml = f'''<mujoco><statistic extent="3"/><visual><global offwidth="{SIZE}" offheight="{SIZE}"/><quality shadowsize="2048" offsamples="4"/>
      <headlight ambient=".35 .35 .35" diffuse=".35 .35 .35" specular=".15 .15 .15"/></visual>
      <asset><texture type="skybox" builtin="flat" rgb1=".97 .96 .93" rgb2=".97 .96 .93" width="32" height="192"/>
      <material name="body" specular=".25" shininess=".35" reflectance="0"/></asset>
      <worldbody><light pos="-3 -4 7" dir=".3 .4 -1" directional="true" diffuse=".5 .5 .5"/>
      <geom name="floor" type="plane" size="100 100 .1" rgba=".91 .89 .84 1"/>{geometries}</worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    points = np.array([[part[k] for part in frame["human"]["segments"] for k in ("a", "b")] for frame in frames])
    lower, upper = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (lower + upper) / 2
    camera.distance = max(2.3, float(np.max(upper - lower)) * 1.8)
    shoulders = next(part for part in segments if part["id"] == "shoulders")
    axis = np.array(shoulders["b"]) - np.array(shoulders["a"])
    camera.azimuth = float(np.degrees(np.arctan2(axis[1], axis[0])) + 105)
    camera.elevation = -12
    times = np.array([f["source_time_s"] for f in frames])
    renderer = mujoco.Renderer(model, height=SIZE, width=SIZE)
    encoder = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-n", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", f"{SIZE}x{SIZE}", "-framerate", str(FPS), "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)], stdin=subprocess.PIPE)
    try:
        for frame_index in range(math.ceil(duration * FPS)):
            timestamp = times[0] + frame_index / FPS
            index = min(len(times)-2, max(0, int(np.searchsorted(times, timestamp, side="right"))-1))
            fraction = float(np.clip((timestamp-times[index]) / (times[index+1]-times[index]), 0, 1))
            positions = points[index] * (1-fraction) + points[index+1] * fraction
            for i, part in enumerate(segments):
                a, b = positions[i*2:i*2+2]
                data.mocap_pos[i] = b if part["id"] == "head" else (a+b)/2
                model.geom_size[i+1, 1] = np.linalg.norm(b-a)/2
                mujoco.mju_quatZ2Vec(data.mocap_quat[i], b-a)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            pixels = renderer.render()
            encoder.stdin.write(pixels.tobytes())
            if frame_index == 0:
                Image.fromarray(pixels).save(poster)
    finally:
        encoder.stdin.close()
        renderer.close()
    if encoder.wait(timeout=30):
        raise RuntimeError("Preview encoder failed")
    return {"rendered_frames": math.ceil(duration * FPS), "fps": FPS, "camera": {"azimuth": camera.azimuth, "elevation": -12},
            "presentation": "Saved kinematic segment endpoints with linear display interpolation and a spherical head marker; no new simulation or learned motion"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosted-catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", choices=[e[0] for e in EXAMPLES])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = []
    for identity, title, caption, author, source_file in EXAMPLES:
        if args.only and identity != args.only:
            continue
        folder = args.out / identity
        folder.mkdir(exist_ok=False)
        frames, source, offset, receipt = load_example(identity, args.hosted_catalog)
        duration = min(8.0, frames[-1]["source_time_s"] - frames[0]["source_time_s"])
        receipt.update(source_sha256=sha(source.read_bytes()), source_start_s=offset + frames[0]["source_time_s"], duration_s=duration)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-n", "-ss", str(receipt["source_start_s"]),
            "-i", str(source), "-t", str(duration), "-vf", "scale=640:-2,fps=30", "-an", "-c:v", "libx264",
            "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(folder / "source.mp4")], check=True, timeout=60)
        receipt.update(render_reference(frames, folder / "result.mp4", folder / "poster.png", duration=duration))
        receipt["files"] = {name: sha((folder/name).read_bytes()) for name in ["source.mp4", "result.mp4", "poster.png"]}
        (folder / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        items.append({"id": identity, "title": title, "caption": caption, "creator": author, "source_file": source_file})
        print(json.dumps({"finished": identity, "seconds": duration, "frames": receipt["rendered_frames"]}), flush=True)
    index = f"exports-{args.only}.json" if args.only else "exports.json"
    (args.out / index).write_text(json.dumps(items, indent=2) + "\n")


if __name__ == "__main__":
    main()
