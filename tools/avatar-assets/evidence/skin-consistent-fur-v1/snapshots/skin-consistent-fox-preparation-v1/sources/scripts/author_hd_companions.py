"""Derive bounded HD companion recipes without changing their accepted source scenes."""

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avatar_assets.scene import read_scene, validate_scene

USER_REQUEST = "you're working on making these avatars better right? I want the same textures and hd quality as the muse ones for example fura and hari"
SOURCES = {
    "fox": "fox-quality-v4.json",
    "workshop-robot": "workshop-robot-quality-v2.json",
    "lamp": "lamp-quality-v3.json",
}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _allocate_counts(parts, target):
    total = sum(part["fur"]["count"] for part in parts)
    exact = [part["fur"]["count"]*target/total for part in parts]
    counts = [max(16, math.floor(value)) for value in exact]
    if sum(counts) > target:
        raise ValueError("Fur allocation cannot meet its minimum parent counts")
    order = sorted(range(len(parts)), key=lambda index: (-(exact[index]-math.floor(exact[index])), parts[index]["name"]))
    for index in order[:target-sum(counts)]:
        counts[index] += 1
    if sum(counts) != target:
        raise ValueError("Fur count allocation failed")
    return counts


def _part_min_z(part):
    rx, ry, _ = part.get("rotation", [0, 0, 0])
    row = [-math.sin(ry), math.cos(ry)*math.sin(rx), math.cos(ry)*math.cos(rx)]
    scaled = [row[axis]*part["scale"][axis] for axis in range(3)]
    if part["shape"] == "mesh":
        local = min(sum(scaled[axis]*point[axis] for axis in range(3)) for point in part["vertices"])
    elif part["shape"] == "sphere":
        local = -math.sqrt(sum(value*value for value in scaled))
    elif part["shape"] == "cube":
        local = -sum(abs(value) for value in scaled)
    elif part["shape"] == "cylinder":
        local = -math.hypot(*scaled[:2])-abs(scaled[2])
    else:
        raise ValueError("Ground contact needs explicit mesh, sphere, cube or cylinder bounds")
    extension = max(part.get("fur", {}).get("length", 0), part.get("native_groom", {}).get("length", 0))
    return part["location"][2]+local-extension


def _studio(scene, kind):
    if kind == "fox":
        names = [part["name"] for part in scene["parts"] if "foot" in part["name"] or "toe" in part["name"]]
    elif kind == "workshop-robot":
        names = [part["name"] for part in scene["parts"] if "sole" in part["name"] or "toe" in part["name"]]
    else:
        names = [part["name"] for part in scene["parts"] if part["name"].startswith(("rubber_foot_", "underside_screw_"))]
    contacts = [part for part in scene["parts"] if part["name"] in names]
    if not contacts:
        raise ValueError("No authored ground-contact parts")
    height = round(min(_part_min_z(part) for part in contacts), 6)
    return {"ground_z": height, "world_strength": .3, "exposure": .55}, {
        "method": "Conservative authored rest bounds of named contact parts, including configured maximum fur length; not simulated ground contact",
        "parts": names, "ground_z": height,
    }


def derive(kind):
    source_path = ROOT / "examples/scenes" / SOURCES[kind]
    original = read_scene(source_path)
    scene = deepcopy(original)
    scene["id"] = kind+"-hd-v1"
    scene["source_prompt"] = original["prompt"]
    edit = (
        "Preserve this character/object's exact base mesh components, proportions, palettes, accessories, bone rests "
        "and four skeletal actions. Add native-resolution 1024/2048 procedural basecolor and tangent-normal textures "
        "with irregular fine surface detail, not an enlarged 512px bitmap. Use a grounded soft studio with world "
        "strength0.3 and exposure0.55. Do not claim exact Muse reconstruction or equal visual quality before rendering "
        "and comparison. Keep all sources and all prior recipes unchanged. "
    )
    if kind == "fox":
        edit += (
            "Replace sparse opaque triangle fur with two explicitly different deliverable representations: 5,000 "
            "six-vertex alpha-masked GLB hair ribbons and a native Blender groom with 5,000 parents and ten children "
            "per parent (55,000 configured fibers maximum). Keep prior fur lengths, use ribbon width0.022m, native "
            "hair radius0.00025m and roughness0.001m. Groom/card roots inherit the existing skin and preserve props. "
            "Keep hair static relative to its deforming surface; no hair dynamics or reconstructed real anatomy."
        )
        furry = [part for part in scene["parts"] if "fur" in part]
        counts = _allocate_counts(furry, 5000)
        for part, count in zip(furry, counts):
            old = part["fur"]
            direction = [1, .1, .4] if "tail" in part["name"] else [0, 0, 1] if "ear" in part["name"] else [0, 0, -1]
            part["fur"] = {"count": count, "length": old["length"], "width": .022,
                           "seed": old["seed"], "mode": "cards", "direction": direction}
            part["native_groom"] = {"parents": count, "children": 10, "length": old["length"],
                                    "radius": .00025, "roughness": .001, "seed": old["seed"]+10000}
        for name, resolution, strength in (("coat", 2048, .5), ("cream", 2048, .35), ("dark", 1024, .4), ("scarf", 1024, .22)):
            scene["materials"][name]["surface"].update(resolution=resolution, strength=strength)
        scene["authored_assumptions"] = [text for text in scene["authored_assumptions"] if not text.startswith("Short fur is bounded opaque")]
        scene["authored_assumptions"].extend([
            "Browser GLB fur is 5,000 authored six-vertex alpha-masked hair ribbons; the 55,000-fiber upper-bound native Blender groom is a separate render/edit representation and is not exported as GLB strands.",
            "Both fur representations retain the old configured lengths and inherited surface skinning; no strand dynamics, measured fur, recovered groom, or Muse-level visual parity is claimed.",
            "The fox iris remains its original glossy colored geometry: a radial iris map is not applied to an incompatible latitude/longitude UV sphere.",
        ])
    else:
        edit += "Use two bounded HD brushed-metal finishes only; retain enamel, rubber, glass-like and paper material identities. Do not add animal fur."
        finishes = (("metal", 2048, .3, 173), ("brass", 1024, .28, 271)) if kind == "workshop-robot" else (("brass", 2048, .28, 71), ("steel", 1024, .25, 73))
        for name, resolution, strength, seed in finishes:
            scene["materials"][name]["surface"] = {"pattern": "brushed", "resolution": resolution,
                                                 "strength": strength, "seed": seed}
    scene["prompt"] = original["prompt"]+"\n\nExact follow-up request: "+USER_REQUEST+"\n\nHD edit: "+edit
    scene["style"] = original["style"]+"; native-HD procedural surface and grounded-studio edition"
    scene["studio"], grounding = _studio(scene, kind)
    scene["authored_assumptions"].append(
        "HD textures are seeded, locally authored procedural patterns at native1024/2048 resolution, not neural output, reference-extracted texture maps or evidence of photorealism. Studio floor and exposure are render settings, not simulation geometry."
    )
    card_parts = [part for part in scene["parts"] if part.get("fur", {}).get("mode") == "cards"]
    groom_parts = [part for part in scene["parts"] if "native_groom" in part]
    pixels = sum(2*material["surface"]["resolution"]**2 for material in scene["materials"].values() if "surface" in material)
    pixels += len({part["material"] for part in card_parts})*1024**2
    scene["hd_upgrade"] = {
        "status": "authored_recipe_not_yet_built_or_visually_accepted",
        "source_scene": "examples/scenes/"+SOURCES[kind], "source_scene_sha256": _digest(source_path),
        "source_scene_id": original["id"], "source_prompt": original["prompt"],
        "prior_source_prompt": original.get("source_prompt"), "exact_user_request": USER_REQUEST,
        "edit_prompt": edit, "studio_grounding": grounding,
        "representations": {
            "glb": {"hair_cards": sum(part["fur"]["count"] for part in card_parts),
                    "hair_card_vertices": sum(part["fur"]["count"]*6 for part in card_parts),
                    "native_strands_exported": False, "skeletal_clips": "All four original clips preserved unchanged"},
            "native_blender": {"groom_parents": sum(part["native_groom"]["parents"] for part in groom_parts),
                               "fiber_upper_bound": sum(part["native_groom"]["parents"]*(part["native_groom"]["children"]+1) for part in groom_parts),
                               "simulation": False},
        },
        "texture_pixels_including_card_atlases": pixels,
        "paid_calls": 0, "provider_cost_usd": 0,
    }
    if scene["bones"] != original["bones"] or scene["states"] != original["states"]:
        raise ValueError("HD edit changed the original rig or actions")
    geometry_fields = {"name", "shape", "vertices", "faces", "uv", "weights", "location", "rotation", "scale", "bone", "material", "bevel", "bevel_segments", "segments", "rings"}
    for before, after in zip(original["parts"], scene["parts"]):
        if {key: before[key] for key in geometry_fields if key in before} != {key: after[key] for key in geometry_fields if key in after}:
            raise ValueError("HD edit changed accepted base geometry")
    return validate_scene(scene)


def encoded(scene):
    return (json.dumps(scene, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", *SOURCES), default="all")
    parser.add_argument("--check", action="store_true", help="Check validation and exact deterministic recipe bytes without writing")
    args = parser.parse_args()
    selected = list(SOURCES) if args.only == "all" else [args.only]
    prepared = [(derive(kind), ROOT / "examples/scenes" / (kind+"-hd-v1.json")) for kind in selected]
    if not args.check and any(path.exists() or path.is_symlink() for _, path in prepared):
        parser.error("Output already exists; use --check to verify without overwriting")
    for scene, path in prepared:
        raw = encoded(scene)
        if args.check:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != raw:
                parser.error(f"Generated bytes differ: {path}")
            read_scene(path)
        else:
            with path.open("xb") as handle:
                handle.write(raw)
        detail = scene["hd_upgrade"]
        print(json.dumps({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(),
                          "check": args.check, "bytes": len(raw), "texture_pixels": detail["texture_pixels_including_card_atlases"],
                          "representations": detail["representations"], "studio": scene["studio"]}))


if __name__ == "__main__":
    main()
