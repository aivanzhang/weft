"""Reproduce the second, longer-coated HD panda recipe without editing v1."""

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

from author_hd_panda import build_scene as build_v1_scene
from author_hd_panda import dragon_front, head_front, lens
from author_quality_companions import color, primitive

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "examples/scenes/muse-panda-hd-v2.json"


def fuller_eyes():
    replacements = {}
    for label, sign in (("left", 1), ("right", -1)):
        for dragon in (False, True):
            prefix, bone = ("dragon_", "dragon_head") if dragon else ("", "head")
            x, z = (-.64+sign*.075, 1.753) if dragon else (sign*.216, 1.851)
            surface = dragon_front(x, z) if dragon else head_front(x, z)
            center = [x-sign*(.003 if dragon else .005), surface-(.032 if dragon else .056), z-.001 if dragon else z-.002]
            # The wider rim meets the white globe; the apex keeps v1's forward depth.
            rx, rz, depth, ratio = (.0365, .0415, .016, .72) if dragon else (.0715, .0795, .029, .70)
            name = f"{prefix}iris_{label}"
            replacements[name] = lens(name, center, rx, rz, depth, "iris", bone, 32 if dragon else 48, 4 if dragon else 5)
            pupil_center = [center[0], center[1]-depth*(1-ratio*ratio)-.0007, center[2]]
            name = f"{prefix}pupil_{label}"
            replacements[name] = lens(name, pupil_center, rx*ratio, rz*ratio,
                                       depth*ratio*ratio+(.0008 if dragon else .0012), "pupil", bone,
                                       24 if dragon else 32, 3 if dragon else 4)
    return replacements


def groom_revision(parts):
    for part in parts:
        if "fur" in part:
            part["fur"]["length"] = round(min(.04, part["fur"]["length"]*1.4), 6)
            part["fur"]["width"] = round(min(.04, part["fur"]["width"]*1.25), 6)
        if "native_groom" in part:
            groom = part["native_groom"]
            groom.update(children=30, length=round(min(.035, groom["length"]*1.8), 6),
                         radius=round(groom["radius"]*1.5, 7), roughness=.0015)
        if part["name"].startswith("inner_ear_"):
            seed = 181 if part["name"].endswith("left") else 182
            part["material"] = "charcoal"
            part["scale"] = [.098, .020, .117]
            part["fur"] = {"mode":"cards","count":120,"length":.019,"width":.022,
                           "seed":seed,"direction":[0,0,1]}
            part["native_groom"] = {"parents":140,"children":30,"length":.023,
                                    "radius":.00030,"roughness":.0015,"seed":seed}


def crown_emitter():
    # A buried rounded emitter replaces exposed solid spikes; its silhouette comes from hair.
    part = primitive("crown_groom", "sphere", [-.020,.018,2.283], [.048,.035,.010],
                     "ivory", "head", segments=24, rings=16)
    part["fur"] = {"mode":"cards","count":140,"length":.038,"width":.022,
                   "seed":183,"direction":[-.35,.15,.92]}
    part["native_groom"] = {"parents":300,"children":30,"length":.035,
                            "radius":.00030,"roughness":.0015,"seed":183}
    return part


def build_scene():
    scene = build_v1_scene()
    preserved = {key:deepcopy(scene[key]) for key in ("bones","states","attachments")}
    scene["id"] = "muse-panda-hd-v2"
    scene["source_hd_v1_prompt"] = scene["prompt"]
    scene["prompt"] = (
        "Refine the preserved Muse-inspired HD panda and its green anatomical-right/viewer-left shoulder "
        "dragon after reviewing the actual v1 and dense-fur portraits against the supplied Muse reference. "
        "Keep the complete three-dimensional geometry, twenty-two-bone skeleton, attachment hierarchy and "
        "all four authored clips. Give the panda a visibly longer, denser fine coat: native parent counts "
        "unchanged on existing emitters, thirty interpolated children, length 1.8 times v1 capped at .035m, "
        "radius 1.5 times v1 and roughness .0015. Keep exported six-vertex hair ribbons distinct from native "
        "hair: length 1.4 times v1, width 1.25 times v1, each capped at .04m. Replace the flat brown inner "
        "ears with charcoal-furred surfaces. Remove the three exposed white plastic forelock spikes; use "
        "a small buried rounded ivory emitter with fine groomed hair for the crown tuft. Increase the "
        "warm radial-UV iris and black pupil shells so the white rim is narrower, while retaining opaque "
        "convex glossy depth and real light reflections. Use softly off-white sclera instead of a bright "
        "toy-like white ring. Raise the black eye-patch layer enough to cover the lengthened white head "
        "groom. Use the plush studio preset and a modest three-quarter camera from viewer-left. Preserve "
        "the dragon, crafting orb, palette and motion meaning. These are authored visual changes, not "
        "recovered anatomy, hair dynamics, runtime physics or verified Muse-quality parity."
    )
    replacements = fuller_eyes()
    parts = []
    for part in scene["parts"]:
        if part["name"].startswith("forehead_tuft_"):
            continue
        part = replacements.get(part["name"], part)
        if part["name"].startswith("eye_patch_"):
            for vertex in part["vertices"]:
                vertex[1] = round(vertex[1]-.010, 6)
        parts.append(part)
    groom_revision(parts)
    parts.append(crown_emitter())
    scene["parts"] = parts
    scene["materials"]["sclera"] = color("EEE8D9", .24)
    scene["materials"]["pupil"] = color("050708", .035)
    # No part uses the removed brown ear material; avoid generating an unused texture pair.
    if not any(part["material"] == "ear_inner" for part in parts):
        scene["materials"].pop("ear_inner", None)
    scene["camera"]["position"] = [-1.3,-7,2.8]
    scene["studio"] = {"preset":"plush","ground_z":0,"world_strength":.07,"exposure":.4}
    cards = sum(part.get("fur",{}).get("count",0) for part in parts)
    fibers = sum(part.get("native_groom",{}).get("parents",0)*(part.get("native_groom",{}).get("children",0)+1) for part in parts)
    controls = sum(len(part.get("vertices",[])) for part in parts)
    primitives = sum(part["segments"]*(part["rings"]-1)+2 for part in parts if part["shape"] == "sphere")
    texture_pixels = sum(2*mat["surface"]["resolution"]**2 for mat in scene["materials"].values() if "surface" in mat)
    atlas_materials = {part["material"] for part in parts if part.get("fur",{}).get("mode") == "cards"}
    scene["authored_assumptions"] = [text for text in scene["authored_assumptions"]
                                     if not text.startswith(("The retained emitters previously", "The two-channel surface plan"))]
    scene["authored_assumptions"].extend([
        f"This v2 recipe requests {cards:,} skinned six-vertex fiber ribbons and at most {fibers:,} configured native parent/child hairs; neither count is an independently counted rendered fiber measurement.",
        "Existing native hair lengths are 1.8 times v1 and ribbon lengths 1.4 times v1. Hair remains static authored grooming following skin weights, without hair dynamics.",
        "The solid three-spike forelock is removed. A low rounded emitter, mostly buried in the existing head, supplies a separate fine crown groom; no bare cone or triangle is used as a hair strand.",
        "The new eye-patch offset is .033m from the authored face surface to cover the longer white head groom. Hair overlap and final facial-line visibility still require rendered inspection.",
        "Panda iris radii are .0715/.0795m and pupil radius is 70% of the iris; dragon iris radii are .0365/.0415m and pupil radius is 72%. Radial UV0 and opaque glossy eye shells remain unchanged in meaning.",
        f"The surface plan is {texture_pixels:,} pixels for two channels, plus one 1024-square card atlas per fur material; no external texture acquisition is involved.",
        "The plush lighting and viewer-left camera are authored presentation choices, not a reference-camera calibration. Organic facial form still differs from Muse and no parity is asserted."
    ])
    checks = scene["authoring_checks"]
    checks.update(parts=len(parts), explicit_mesh_control_vertices=controls,
                  primitive_vertex_estimate_before_uv_splits=primitives,
                  surface_vertex_estimate_before_uv_splits=controls+primitives,
                  glb_hair_cards=cards, glb_card_vertices=cards*6,
                  native_hair_configured_upper_bound=fibers, two_channel_texture_pixels=texture_pixels,
                  texture_pixels_including_card_atlases=texture_pixels+len(atlas_materials)*1024**2,
                  native_render_performed=False, muse_quality_parity_verified=False,
                  native_children_per_parent=30, solid_forelock_parts_removed=3)
    checks["recipe_dependencies"]["author_hd_panda.py"] = hashlib.sha256((ROOT/"scripts/author_hd_panda.py").read_bytes()).hexdigest()
    checks["source_hd_v1_scene_sha256"] = hashlib.sha256((ROOT/"examples/scenes/muse-panda-hd-v1.json").read_bytes()).hexdigest()
    for key in preserved:
        assert scene[key] == preserved[key], f"Preserve v1 {key} exactly"
    assert len(parts) <= 128 and controls <= 30000
    assert fibers <= 240000 and cards*6+controls+primitives < 90000
    assert all(part.get("native_groom") for part in parts if part.get("fur"))
    return scene


def main():
    parser = argparse.ArgumentParser(description="Author the longer-coated HD panda v2 or check exact saved bytes")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    scene = build_scene()
    raw = (json.dumps(scene,separators=(",",":"),allow_nan=False)+"\n").encode()
    if args.check:
        if DESTINATION.read_bytes() != raw:
            raise SystemExit("HD panda v2 recipe differs from the saved scene bytes")
    else:
        with DESTINATION.open("xb") as handle:
            handle.write(raw)
    print(json.dumps({"path":str(DESTINATION),"mode":"check" if args.check else "author",
                      "sha256":hashlib.sha256(raw).hexdigest(),"bytes":len(raw),
                      "checks":scene["authoring_checks"]},sort_keys=True))


if __name__ == "__main__":
    main()
