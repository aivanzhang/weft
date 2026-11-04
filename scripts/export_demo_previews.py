"""Make presentation clips from saved evidence; never rerun learning or modify results."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import imageio_ffmpeg
from PIL import Image, ImageDraw

from weft.world import World, advance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Choose an empty output directory; existing evidence is never overwritten")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    records = []

    def encode(name, source, filters, *, start=0, duration=None):
        command = [ffmpeg, "-v", "error", "-n", "-ss", str(start), "-i", str(source)]
        if duration is not None:
            command += ["-t", str(duration)]
        command += ["-an", "-vf", filters, "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out / name)]
        subprocess.run(command, check=True, timeout=60)
        records.append({"output": name, "source": str(source.relative_to(root)),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "start_seconds": start, "output_duration_limit": duration, "filters": filters})

    catalog = json.loads((root / "website/environments/catalog.js").read_text().split(" = ", 1)[1].rstrip(";\n"))[0]
    world = World.from_dict(catalog["models"][0]["world"])
    segment = next(s for s in catalog["segments"] if s["primary"])
    duration = segment["end"] - segment["start"]
    # Slow both sides equally so the short measured pass can be inspected.
    encode("rolling-source.mp4", root / "data/rolling-ball.mp4",
           f"trim=duration={duration},setpts=4*(PTS-STARTPTS),fps=30", start=segment["start"])
    width, height, fps = catalog["width"], catalog["height"], 30
    command = [ffmpeg, "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", f"{width}x{height}", "-framerate", str(fps), "-i", "pipe:0",
               "-an", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(out / "rolling-result.mp4")]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    x, velocity = segment["x0"], world.v0
    try:
        for _ in range(math.ceil(duration * 4 * fps)):
            frame = Image.new("RGB", (width, height), "#eee9e0")
            draw = ImageDraw.Draw(frame)
            left = world.support_intercept * width
            right = (world.support_intercept + world.support_slope) * width
            draw.polygon([(0, left), (width, right), (width, height), (0, height)], fill="#e1dacd")
            draw.line([(0, left), (width, right)], fill="#bcb5a7", width=2)
            cx, cy, radius = x * width, world.center_y(x) * width, world.radius * width
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill="#eac34c", outline="#b58c23", width=2)
            process.stdin.write(frame.tobytes())
            x, velocity = advance(world, x, velocity, 0, 1 / (4 * fps))
    finally:
        process.stdin.close()
    if process.wait(timeout=30):
        raise RuntimeError("Preview encoder failed")
    records.append({"output": "rolling-result.mp4", "world": world.to_dict(),
                    "source_manifest_sha256": catalog["manifest_sha256"],
                    "source_interval": [segment["start"], segment["end"]], "playback_speed": 0.25,
                    "scope": "Existing reconstructed one-dimensional model; no new fit or evaluation"})
    lab = root / ".local/verify-video-01/portal/9eadcc9655864070b0039fa3feb93fbc/lab"
    encode("movement-source.mp4", lab / "run/source.mp4", "scale=640:-2")
    encode("controller-result.mp4", lab / "sessions/session-9f8a60b20993/avatar-video.mp4", "crop=360:420:0:0")
    encode("character-source.mp4", lab / "run/source.mp4", "scale=640:-2", start=4, duration=4.16)
    encode("character-result.mp4", lab / "sessions/session-1de315688ae1/avatar-video.mp4",
           "crop=360:420:0:0", start=4, duration=4.16)
    for row in records:
        path = out / row["output"]
        row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    (out / "manifest.json").write_text(json.dumps({"profile": "weft-demo-previews-v1", "files": records}, indent=2) + "\n")
    print(json.dumps({"clips": len(records), "bytes": sum(r["bytes"] for r in records)}))


if __name__ == "__main__":
    main()
