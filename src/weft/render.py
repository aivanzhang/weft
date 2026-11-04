import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from PIL import Image, ImageDraw

from weft.browser_video import open_writer
from weft.brand import COLORS
from weft.world import passive_position


def replay_video(video, evidence, before, after, out):
    width, height = evidence["width"], evidence["height"]
    cap = cv2.VideoCapture(str(video))
    size = (width * 3, height + 58)
    writer = open_writer(cv2, out / "replay.mp4", evidence["fps"], size)
    if not writer.isOpened():
        raise RuntimeError("MP4 writer is not available")
    thumbnails, gif_frames = [], []
    primary = evidence["generation"] + evidence["development_validation"]
    episodes = [primary] + evidence["audit_episodes"]
    rendered = 0
    try:
        for episode_number, rows in enumerate(episodes):
            origin = rows[0]["time_s"]
            x0 = rows[0]["x_px"] / width
            dt = rows[2]["time_s"] - origin
            raw_v0 = (rows[2]["x_px"] - rows[0]["x_px"]) / width / dt
            for j, row in enumerate(rows):
                cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Replay frame missing")
                image = Image.new("RGB", size, COLORS["canvas"])
                image.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, 58))
                draw = ImageDraw.Draw(image)
                labels = ["RECORDED VIDEO", "INITIAL MODEL", "SELECTED MODEL"]
                for k, label in enumerate(labels):
                    draw.text((k * width + 12, 8), label, fill=COLORS["ink"])
                    draw.text((k * width + 12, 27), f"pass {episode_number + 1} | source frame {row['frame']} | {row['time_s']:.3f}s", fill=COLORS["ink"])
                for panel, world in enumerate((before, after), start=1):
                    v0 = world.v0 if episode_number == 0 else raw_v0 + 0.5 * world.drag * dt
                    x = float(passive_position(world, [row["time_s"] - origin], x0=x0, v0=v0)[0])
                    y = world.center_y(x)
                    left = panel * width
                    draw.line((left, 58 + world.support_intercept * width,
                               left + width, 58 + (world.support_intercept + world.support_slope) * width), fill=COLORS["ink"], width=3)
                    radius = world.radius * width
                    cx, cy = left + x * width, 58 + y * width
                    # The source ball stays yellow; scene colors are not brand accents.
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="#C5A059", outline=COLORS["ink"], width=2)
                    ox, oy = left + row["x_px"], 58 + row["y_px"]
                    draw.line((ox - 8, oy, ox + 8, oy), fill="#B3261E", width=2)
                    draw.line((ox, oy - 8, ox, oy + 8), fill="#B3261E", width=2)
                    draw.text((left + 12, height + 34), "red cross = recorded center; circle = simulation", fill=COLORS["ink"])
                writer.write(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
                thumbnail = image.copy()
                thumbnail.thumbnail((960, 240))
                gif_frames.append(thumbnail)
                if j in (0, len(rows) - 1):
                    thumbnails.append(thumbnail)
                rendered += 1
    finally:
        writer.release()
        cap.release()
    gif_frames[0].save(out / "replay-slow.gif", save_all=True, append_images=gif_frames[1:], duration=125, loop=0)
    sheet = Image.new("RGB", (960, sum(i.height for i in thumbnails)), COLORS["canvas"])
    y = 0
    for thumbnail in thumbnails:
        sheet.paste(thumbnail, (0, y)); y += thumbnail.height
    sheet.save(out / "replay-contact-sheet.png")
    check = cv2.VideoCapture(str(out / "replay.mp4"))
    decoded = 0
    while check.read()[0]:
        decoded += 1
    check.release()
    if decoded != rendered:
        raise RuntimeError("Generated replay failed full decode verification")
    return {"frames": rendered, "decoded_frames": decoded, "fps": evidence["fps"],
            "note": "Visible complete-object frames only; pass gap omitted; GIF is slowed to 8 fps"}


def write_viewer(out, report, world, graph, interventions):
    assets = Path(__file__).parent / "assets"
    data = json.dumps({"report": report, "world": world.to_dict(), "graph": graph,
                       "interventions": interventions, "palette": dict(COLORS)}, allow_nan=False).replace("<", "\\u003c")
    template = (assets / "viewer.html").read_text()
    (out / "index.html").write_text(template.replace("__WEFT_DATA__", data))
    (out / "engine.js").write_text((assets / "engine.js").read_text())
    for name in ("brand.css", "weft-logo.png"):
        shutil.copyfile(assets / name, out / name)
