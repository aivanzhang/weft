"""Derive distance-readable browser fur without changing native groom or geometry."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avatar_assets.safety import sha256, slug
from avatar_assets.scene import read_scene, validate_scene


def derive(source, identifier):
    original = read_scene(source)
    scene = deepcopy(original)
    scene["id"] = slug(identifier)
    materials = sorted({part["material"] for part in scene["parts"]
                        if part.get("fur", {}).get("mode") == "cards" and part["fur"]["count"]})
    if not materials:
        raise ValueError("Source has no hair cards")
    for name in materials:
        scene["materials"][name]["fur_atlas"] = "distance-v1"
    edit = ("Use the distance-v1 masked fur atlas on every card material: ten wider tapered locks "
            "per tile, sixteen seeded tiles, 1024px RGBA atlas and alpha cutoff 0.18. Preserve all "
            "geometry, skin weights, native groom, textures on base surfaces, attachments, actions, "
            "lighting and camera. This browser-only visibility correction is not Muse-quality parity.")
    scene["prompt"] = edit
    scene["browser_fur_revision"] = {"source_scene_sha256": sha256(source),
        "source_prompt": original["prompt"], "edit_prompt": edit, "materials": materials,
        "reason": "fine-v1 alpha coverage approaches zero at 16px and smaller tiles after minification",
        "provider_calls": 0, "provider_cost_usd": 0}
    scene["authored_assumptions"].append(edit)
    return validate_scene(scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scene = derive(args.source, args.id)
    raw = (json.dumps(scene, separators=(",", ":"), allow_nan=False)+"\n").encode()
    path = Path(args.out)
    if args.check:
        if path.read_bytes() != raw:
            raise ValueError("Derived recipe differs from saved bytes")
    else:
        with path.open("xb") as handle:
            handle.write(raw)
    print(json.dumps({"scene": str(path), "sha256": sha256(path), "check": args.check}))


if __name__ == "__main__":
    main()
