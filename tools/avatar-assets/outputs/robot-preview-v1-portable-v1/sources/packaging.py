"""Stream rendered frames into checked media inside a separately bounded worker."""

import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from avatar_assets.media import encode_video, image_info, video_info
from avatar_assets.safety import read_json, require, write_json


def package_frames(out, frame_root, scene, spec):
    out, frame_root = Path(out), Path(frame_root)
    (out / "fallbacks").mkdir()
    (out / "animations").mkdir()
    (out / "contact-sheets").mkdir()
    portrait = frame_root / "portrait.png"
    image_info(portrait)
    with Image.open(portrait) as im:
        require(im.size == (spec["portrait"], spec["portrait"]), "Portrait dimensions differ from spec")
        im.convert("RGB").save(out / "portrait.webp", lossless=True)
    clips = []
    for state in scene["states"]:
        frames = sorted((frame_root / "frames" / state["id"]).glob("*.png"))
        require(len(frames) == spec["seconds"] * 24 + 1, "Render frame count mismatch")
        require([p.stem for p in frames] == [f"{i:04d}" for i in range(len(frames))], "Noncontiguous render frames")
        path = out / "animations" / (state["id"] + ".mp4")
        def inputs():
            for frame in frames[:-1]:
                require(not frame.is_symlink(), "Linked input frame rejected")
                with Image.open(frame) as im:
                    require(im.size == (spec["size"], spec["size"]), "Rendered frame dimensions differ")
                    yield im.copy()
        encode_video(inputs(), path, 24, spec["size"], spec["size"], "#ffffff")
        report = video_info(path, out / "contact-sheets" / (state["id"] + ".png"))
        require(report["motion_detected"], f"Repeated stills rejected: {state['id']}")
        require(report["frames"] == spec["seconds"] * 24, "Encoded frame count mismatch")
        fallback = out / "fallbacks" / (state["id"] + ".png")
        shutil.copyfile(frames[0], fallback)
        with Image.open(frames[0]) as a, Image.open(frames[-1]) as b:
            seam = float(np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean() / 255)
        clips.append({"id": state["id"], "meaning": state["meaning"], "path": str(path.relative_to(out)),
                      "fallback": str(fallback.relative_to(out)), "fallback_inspection": image_info(fallback), "inspection": report,
                      "source_clip_seconds": state["duration"], "loop_requested": state["loop"],
                      "exact_endpoint_mae": round(seam, 8), "loop_declaration": "not_certified; visual seam review required"})
    result = {"portrait": image_info(out / "portrait.webp"), "clips": clips}
    write_json(out / "packaging-result.json", result)
    return result


if __name__ == "__main__":
    out, frame_root, scene, spec = sys.argv[1:]
    package_frames(out, frame_root, read_json(scene), read_json(spec))
