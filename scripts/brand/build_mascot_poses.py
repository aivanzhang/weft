"""Derive raster-export poses from the unchanged layered vector master."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("out", type=Path)
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
for name in ("idle", "blink", "wave_a", "wave_b"):
    tree = ET.parse(ROOT / "assets/brand/source/weft-mascot-rig.svg")
    nodes = {node.get("id"): node for node in tree.iter() if node.get("id")}
    if name == "blink":
        nodes["eyes-open"].set("opacity", "0")
        nodes["eyes-closed"].set("opacity", "1")
    elif name.startswith("wave"):
        angle = 30 if name == "wave_a" else 45
        nodes["arm-left"].set("transform", f"rotate({angle} 121 420)")
    tree.write(args.out / (name + ".svg"), encoding="unicode")

# Center the complete silhouette inside square and circular avatar crops.
avatar = ET.parse(ROOT / "assets/brand/source/weft-mascot-rig.svg")
root = avatar.getroot()
root.set("viewBox", "0 0 640 640")
palette = json.loads((ROOT / "assets/brand/source/palette.json").read_text())["colors"]
root.find(f"{{{SVG}}}title").text = "Weft publisher avatar"
root.find(f"{{{SVG}}}desc").text = "The approved woven Weft mascot on a warm ivory background."
root.insert(2, ET.Element(f"{{{SVG}}}rect", {
    "width": "640", "height": "640", "fill": palette["surface"],
}))
root.find(f"{{{SVG}}}g").set("transform", "translate(32 15) scale(0.9)")
avatar.write(args.out / "publisher-avatar.svg", encoding="unicode")

# Keep the developer cue in the stitching, not a competing foreground object.
root.find(f"{{{SVG}}}title").text = "Weft developer publisher avatar"
root.find(f"{{{SVG}}}desc").text = "The woven Weft mascot with a small terracotta code stitch on its lower corner."
ET.SubElement(root, f"{{{SVG}}}path", {
    "id": "publisher-code-stitch",
    "d": "M432 461 L420 472 L432 483 M453 459 L443 485 M465 461 L477 472 L465 483",
    "transform": "translate(-8 -12) rotate(5 449 472)",
    "fill": "none", "stroke": palette["accent"], "stroke-width": "4.5",
    "stroke-linecap": "round", "stroke-linejoin": "round",
})
avatar.write(args.out / "publisher-developer.svg", encoding="unicode")
