"""Derive darker plush shading and complete paw coverage without changing v2 geometry."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from avatar_assets.scene import read_scene, validate_scene
from author_hd_panda_v2 import build_scene as build_v2_scene

DESTINATION = ROOT / "examples/scenes/muse-panda-hd-v3.json"
USER_REQUEST = "you're working on making these avatars better right? I want the same textures and hd quality as the muse ones for example fura and hari"
ROOT_FINDING = "Rootseesnewv2earfurbetterbutblackcoatgrayandpatchy(corroded/velvet) vsreferencefineblackdenseplush; whiteheadfurlargelydenoisedaway; pawslookbare."
REVIEW_INPUTS = (
    {"path": "outputs/muse-panda-hd-build-v2/portrait.png", "sha256": "6a9ec78732fcc9930c69ef3b2aebe21f838a7b9ed5f0d43754abe7732d1c0161"},
    {"path": "outputs/hd-fur-look-v1/dense-directed-1600.png", "sha256": "0364b027c172bbbfe9285dda858212d73885f6d56a781c19e7a309410c23cbb4"},
    {"path": "outputs/reference-v1/originals/avatar-1792879257185832992-2.webp", "sha256": "c072bc988b0116ba6ac02641467cde3cddf65ac76a98be5a19c0c54553c9f3b2"},
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_scene():
    source_path = ROOT / "examples/scenes/muse-panda-hd-v2.json"
    original = read_scene(source_path)
    if build_v2_scene() != original:
        raise ValueError("The preserved v2 panda does not match its authoring recipe")
    # Review renders inform the edit but are not required inputs when reproducing it elsewhere.
    for item in REVIEW_INPUTS:
        reviewed = ROOT / item["path"]
        if reviewed.exists() and (reviewed.is_symlink() or not reviewed.is_file() or _sha(reviewed) != item["sha256"]):
            raise ValueError("A recorded visual-review input changed")
    scene = deepcopy(original)
    scene["id"] = "muse-panda-hd-v3"
    scene["source_hd_v2_prompt"] = original["prompt"]
    scene["user_hd_request"] = USER_REQUEST
    edit = (
        "Refine the preserved Muse-inspired HD panda after direct comparison of the actual768px v2 portrait, "
        "the raw1600px dense-fur look test and the original Muse reference. The v2 ears show more fur, but "
        "black regions read as gray, mottled velvet rather than fine dark plush. Feet, toes and paw digits "
        "have no native groom. White-head detail is less visible in the smaller denoised portrait; resolution "
        "and denoising both differ, so do not attribute that loss to denoising alone. Make only two bounded "
        "local edits: set charcoal sRGB RGBA to[0.04,0.038,0.034,1] and its existing HD fur-surface strength "
        "to0.06; add short hair to the fourteen currently bare panda foot/toe/digit surfaces. Each of two "
        "feet gets150 native parents at0.012m length, each of six toes35 parents at0.010m, and each of six "
        "paw digits16 parents at0.008m. All new grooms use30 children, radius0.00025m, roughness0.001m and "
        "recorded seeds. Add matching exported GLB cards:30 per foot,10 per toe and10 per paw digit, for180 "
        "new six-vertex cards; width0.012m and length0.010m. Keep every existing groom unchanged. Do not "
        "apply panda fur to dragon feet or any other part. Preserve all base geometry, face and eye shapes, "
        "ivory and other materials, UVs, normal/texture channels, full skeleton, dragon attachment, four "
        "authored clips, camera, ground and plush lighting. Native groom and GLB cutout cards remain distinct. "
        "This is a new authored candidate requiring rendered review, not verified Muse-quality parity."
    )
    scene["prompt"] = "Exact user request: "+USER_REQUEST+"\n\nPanda v3 edit prompt: "+edit
    scene["style"] = original["style"]+"; darker charcoal undercoat and complete short-fur paw coverage"
    scene["materials"]["charcoal"]["color"] = [.04, .038, .034, 1]
    scene["materials"]["charcoal"]["surface"]["strength"] = .06
    additions = {}
    for side_index, side in enumerate(("left", "right")):
        additions[f"foot_{side}"] = {"parents": 150, "length": .012, "cards": 30, "seed": 301+side_index}
        for index in range(3):
            additions[f"toe_{side}_{index}"] = {"parents": 35, "length": .010, "cards": 10,
                                                  "seed": 310+side_index*3+index}
            additions[f"paw_digit_{side}_{index}"] = {"parents": 16, "length": .008, "cards": 10,
                                                        "seed": 320+side_index*3+index}
    found = set()
    for part in scene["parts"]:
        if part["name"] not in additions:
            continue
        if part["material"] != "charcoal" or "fur" in part or "native_groom" in part:
            raise ValueError("Expected an ungroomed charcoal panda contact part")
        planned = additions[part["name"]]
        part["native_groom"] = {"parents": planned["parents"], "children": 30, "length": planned["length"],
                                "radius": .00025, "roughness": .001, "seed": planned["seed"]}
        part["fur"] = {"mode": "cards", "count": planned["cards"], "length": .01, "width": .012,
                       "seed": planned["seed"], "direction": [0, 0, -1] if part["name"].startswith("paw_digit_") else [0, -1, 0]}
        found.add(part["name"])
    if found != set(additions):
        raise ValueError("The fourteen reviewed bare panda contact surfaces changed")
    cards = sum(part.get("fur", {}).get("count", 0) for part in scene["parts"])
    parents = sum(part.get("native_groom", {}).get("parents", 0) for part in scene["parts"])
    fibers = sum(part["native_groom"]["parents"]*(part["native_groom"]["children"]+1)
                 for part in scene["parts"] if "native_groom" in part)
    if (cards, parents, fibers) != (7688, 7688, 238328):
        raise ValueError("The expected bounded panda fur totals changed")
    scene["authored_assumptions"] = [text for text in scene["authored_assumptions"] if not text.startswith("This v2 recipe requests")]
    scene["authored_assumptions"].extend([
        "The v3 charcoal material changes sRGB color to[0.04,0.038,0.034,1] and existing procedural HD fur strength to0.06. This is an authored look adjustment, not reference-color calibration or a rendered improvement claim.",
        "The fourteen previously bare panda foot/toe/paw-digit surfaces receive606 native parents with30 children, adding18,786 configured fibers and180 exported six-vertex hair cards. Dragon feet and all other surfaces are unchanged.",
        "The complete v3 recipe requests7,688 skinned hair cards and at most238,328 configured native parent/child hairs. Native Blender strands are not exported as GLB hairs, and these counts are not independently counted rendered fibers.",
        "All v2 geometry, existing grooms, ivory/facial materials, UVs, rig, attachments, four clips and presentation settings are preserved. The shorter added contact fur can intersect the unchanged authored floor; it is not contact simulation.",
    ])
    checks = scene["authoring_checks"]
    checks.update(glb_hair_cards=cards, glb_card_vertices=cards*6,
                  native_hair_configured_upper_bound=fibers, native_groom_parent_count=parents,
                  additional_panda_furry_parts=14, additional_card_vertices=1080,
                  native_render_performed=False, muse_quality_parity_verified=False,
                  source_hd_v2_scene_sha256=_sha(source_path))
    checks["recipe_dependencies"]["author_hd_panda_v2.py"] = _sha(ROOT / "scripts/author_hd_panda_v2.py")
    scene["v3_visual_revision"] = {
        "status": "authored_candidate_pending_native_render_and_visual_review",
        "source_scene": "examples/scenes/muse-panda-hd-v2.json", "source_scene_sha256": _sha(source_path),
        "source_prompt": original["prompt"], "exact_user_request": USER_REQUEST,
        "root_visual_finding_exact": ROOT_FINDING, "exact_edit_prompt": edit,
        "review_inputs": deepcopy(list(REVIEW_INPUTS)),
        "charcoal_before": deepcopy(original["materials"]["charcoal"]),
        "charcoal_after": deepcopy(scene["materials"]["charcoal"]),
        "new_fur_parts": additions, "provider_calls": 0, "provider_cost_usd": 0,
    }
    for key in ("bones", "states", "attachments", "camera", "studio", "background"):
        if scene[key] != original[key]:
            raise ValueError(f"Preserve v2 {key} exactly")
    if len(scene["parts"]) != len(original["parts"]):
        raise ValueError("The v3 revision cannot add or remove base mesh parts")
    for before, after in zip(original["parts"], scene["parts"]):
        expected = deepcopy(after)
        if before["name"] in additions:
            expected.pop("fur")
            expected.pop("native_groom")
        if before != expected:
            raise ValueError("The v3 revision changed base geometry or an existing groom")
    for name, material in original["materials"].items():
        if name != "charcoal" and scene["materials"][name] != material:
            raise ValueError("The v3 revision changed a non-charcoal material")
    return validate_scene(scene)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate and compare exact saved bytes without writing")
    args = parser.parse_args()
    scene = build_scene()
    raw = (json.dumps(scene, separators=(",", ":"), allow_nan=False)+"\n").encode()
    if args.check:
        if DESTINATION.is_symlink() or not DESTINATION.is_file() or DESTINATION.read_bytes() != raw:
            parser.error("Panda v3 recipe differs from saved bytes")
        read_scene(DESTINATION)
    else:
        with DESTINATION.open("xb") as handle:
            handle.write(raw)
    print(json.dumps({"path": str(DESTINATION.relative_to(ROOT)), "check": args.check,
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                      "cards": scene["authoring_checks"]["glb_hair_cards"],
                      "native_fiber_upper_bound": scene["authoring_checks"]["native_hair_configured_upper_bound"],
                      "new_furry_parts": scene["authoring_checks"]["additional_panda_furry_parts"]}))


if __name__ == "__main__":
    main()
