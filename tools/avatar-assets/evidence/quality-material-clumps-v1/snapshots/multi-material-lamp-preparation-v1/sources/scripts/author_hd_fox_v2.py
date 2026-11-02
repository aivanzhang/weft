"""Derive the plush fox revision while preserving the v1 recipe and four actions."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avatar_assets.scene import read_scene, validate_scene
from scripts.author_hd_companions import derive as derive_v1, encoded


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def derive():
    source_path = ROOT / "examples/scenes/fox-hd-v1.json"
    original = derive_v1("fox")
    if encoded(original) != source_path.read_bytes():
        raise ValueError("The preserved v1 fox does not match its authoring recipe")
    scene = deepcopy(original)
    scene["id"] = "fox-hd-v2"
    scene["source_prompt"] = original["prompt"]
    edit = (
        "The v1 640px portrait still looks too smooth. Preserve the existing fox anatomy, ears, head, facial "
        "features, teal scarf, book, pencil, 20-bone rig and four action clips. Refine only the existing twenty "
        "furry skin surfaces: native Blender groom children30, original parent counts totaling5,000, "
        "configured fiber upper bound155,000; native length1.6times v1 capped at0.035m, radius1.5times v1 "
        "(0.000375m), roughness0.0015m. Keep5,000 GLB hair cards, length1.4times v1, width1.25times v1 "
        "(0.0275m, below0.04m). Native groom and GLB ribbons are different representations; the full155,000 "
        "native fibers are not GLB strand geometry. Comb the body downward, only explicit outer ears upward "
        "and the tail toward its tip. Do not apply hair to eyes, nose, inner-ear skin, scarf, book or tools, "
        "and do not invent a separate forelock. Use the plush studio preset (900W/3m key,140W fill,600W/2m rim, "
        "AgX Medium High Contrast), world strength0.07 and exposure0.4; retain the established ground height. "
        "These settings supersede v1's55,000-fiber/short-hair settings. Require a new native render and visual "
        "comparison before claiming Muse-like fur quality."
    )
    scene["prompt"] = original["prompt"]+"\n\nPlush fox revision: "+edit
    scene["style"] = original["style"]+"; denser plush-fur revision"
    hairy_names = []
    lengths = {"browser_cards": [], "native_groom": []}
    for part in scene["parts"]:
        if "native_groom" not in part:
            continue
        if "fur" not in part or part["material"] not in {"coat", "cream", "dark"}:
            raise ValueError("Unexpected non-animal groom surface in the v1 source")
        hairy_names.append(part["name"])
        groom = part["native_groom"]
        groom.update(children=30, length=round(min(.035, groom["length"]*1.6), 9),
                     radius=round(groom["radius"]*1.5, 9), roughness=.0015)
        cards = part["fur"]
        cards.update(length=round(cards["length"]*1.4, 9), width=round(cards["width"]*1.25, 9))
        if cards["width"] > .04 or cards["length"] > .04:
            raise ValueError("Plush browser fur exceeds its width or length bound")
        cards["direction"] = [1, .1, .4] if part["name"] in {"curved_tail", "tapered_tail_tip"} else (
            [0, 0, 1] if part["name"].startswith("cupped_ear_") else [0, 0, -1])
        lengths["browser_cards"].append(cards["length"])
        lengths["native_groom"].append(groom["length"])
    if len(hairy_names) != 20:
        raise ValueError("The fox's reviewed furry-surface inventory changed")
    scene["studio"] = {**original["studio"], "preset": "plush", "world_strength": .07, "exposure": .4}
    scene["authored_assumptions"] = [text for text in scene["authored_assumptions"]
                                      if not text.startswith(("Browser GLB fur is", "Both fur representations retain"))]
    scene["authored_assumptions"].extend([
        "Plush browser fur remains5,000 six-vertex alpha-masked skinned hair cards; the separately configured native Blender groom has5,000 parents and30 children, a155,000-fiber upper bound, not an exported GLB strand count.",
        "The native groom length is1.6times the v1 length capped at0.035m, and browser ribbons are1.4times longer and1.25times wider. No hair dynamics, recovered real groom or measured animal anatomy is claimed.",
        "Only the same twenty furry coat/cream/dark skin surfaces receive hair. Inner ears, eyes, nose, accessories and tools remain bare; no separate forelock geometry is invented.",
        "The plush studio changes render lighting and contrast only. Its floor stays at the previous authored rest height; it is not collision geometry or a physical contact solver.",
    ])
    previous = deepcopy(original["hd_upgrade"])
    scene["hd_upgrade"].update({
        "status": "authored_recipe_not_yet_built_or_visually_accepted",
        "source_scene": "examples/scenes/fox-hd-v1.json", "source_scene_sha256": _sha(source_path),
        "source_scene_id": original["id"], "source_prompt": original["prompt"],
        "prior_source_prompt": original.get("source_prompt"), "edit_prompt": edit,
        "previous_upgrade": previous,
    })
    scene["hd_upgrade"]["representations"]["native_blender"]["fiber_upper_bound"] = 155000
    scene["plush_upgrade"] = {
        "source_authoring_script": "scripts/author_hd_companions.py",
        "source_authoring_script_sha256": _sha(ROOT / "scripts/author_hd_companions.py"),
        "source_scene_sha256": _sha(source_path), "exact_edit_prompt": edit,
        "fur_surface_names": hairy_names,
        "card_count": sum(part.get("fur", {}).get("count", 0) for part in scene["parts"]),
        "native_groom_parents": sum(part.get("native_groom", {}).get("parents", 0) for part in scene["parts"]),
        "native_fiber_upper_bound": sum(part["native_groom"]["parents"]*(part["native_groom"]["children"]+1)
                                        for part in scene["parts"] if "native_groom" in part),
        "length_ranges_m": {kind: [min(values), max(values)] for kind, values in lengths.items()},
        "studio_source": "Authored plush preset; no lighting or geometry recovered from Muse",
        "preserved_ground_z": original["studio"]["ground_z"],
        "provider_calls": 0, "provider_cost_usd": 0,
    }
    if scene["bones"] != original["bones"] or scene["states"] != original["states"] or scene["materials"] != original["materials"]:
        raise ValueError("Plush revision changed the rig, clips or HD materials")
    for before, after in zip(original["parts"], scene["parts"]):
        if {key: value for key, value in before.items() if key not in {"fur", "native_groom"}} != {
                key: value for key, value in after.items() if key not in {"fur", "native_groom"}}:
            raise ValueError("Plush revision changed base geometry")
    return validate_scene(scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate and compare exact recipe bytes without writing")
    args = parser.parse_args()
    scene = derive()
    path = ROOT / "examples/scenes/fox-hd-v2.json"
    raw = encoded(scene)
    if args.check:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != raw:
            parser.error("Generated fox-hd-v2 bytes differ from the saved recipe")
        read_scene(path)
    else:
        with path.open("xb") as handle:
            handle.write(raw)
    print(json.dumps({"path": str(path.relative_to(ROOT)), "check": args.check,
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                      "card_count": scene["plush_upgrade"]["card_count"],
                      "native_fiber_upper_bound": scene["plush_upgrade"]["native_fiber_upper_bound"],
                      "length_ranges_m": scene["plush_upgrade"]["length_ranges_m"], "studio": scene["studio"]}))


if __name__ == "__main__":
    main()
