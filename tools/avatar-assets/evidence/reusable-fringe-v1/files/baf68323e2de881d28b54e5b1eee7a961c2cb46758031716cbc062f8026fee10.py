"""Derive a new directional native groom from any validated furry scene recipe."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.scene import read_scene, validate_scene
from avatar_assets.safety import require, slug


def author(source, identifier, amount=.65):
    original = read_scene(source)
    scene = deepcopy(original)
    slug(identifier)
    scene["id"] = identifier
    scene["source_uncombed_prompt"] = original["prompt"]
    scene["prompt"] += (
        f" Comb the existing native parent hairs toward each emitter's explicitly authored fur direction "
        f"with amount {amount:g}, preserving every root, strand arc length, skin weight, rig and action. "
        "Use Blender's persistent editable hair keys, not a painted replacement image. Browser cutout "
        "cards keep their existing directional geometry. Preserve the raw render and inspect both "
        "representations; no physical hair simulation or recovered Muse groom is claimed."
    )
    emitters = 0
    for part in scene["parts"]:
        if "native_groom" in part:
            part["native_groom"].update(direction=part.get("fur", {}).get("direction", [0, 0, -1]), comb=amount)
            emitters += 1
    require(emitters > 0, "Source scene has no native groom to comb")
    scene["directional_groom_derivation"] = {
        "source_recipe": Path(source).name,
        "source_scene_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
        "source_id": original["id"], "emitters": emitters, "amount": amount,
        "direction_source": "Existing fur.direction per emitter, or explicit down [0,0,-1]",
        "base_geometry_rig_actions_unchanged": True, "provider_calls": 0, "provider_cost_usd": 0,
        "native_hair_exported_to_glb": False, "muse_parity_verified": False,
    }
    scene["authored_assumptions"].append(
        "Native hair is authored into directional rest-space curves, retaining the emitter skin; "
        "there is no dynamic hair simulation. The GLB contains its separate masked-card coat."
    )
    return validate_scene(scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--amount", type=float, default=.65)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scene = author(args.source, args.id, args.amount)
    encoded = (json.dumps(scene, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()
    out = Path(args.out)
    if args.check:
        read_scene(out)
        require(out.read_bytes() == encoded, "Directional groom recipe differs from deterministic authoring")
    else:
        with out.open("xb") as handle:
            handle.write(encoded)
    print(json.dumps({"path": str(out), "check": args.check, "sha256": hashlib.sha256(encoded).hexdigest(),
                      "emitters": scene["directional_groom_derivation"]["emitters"],
                      "native_fiber_upper_bound": sum(p["native_groom"]["parents"]*(p["native_groom"]["children"]+1)
                                                      for p in scene["parts"] if "native_groom" in p)}))


if __name__ == "__main__":
    main()
