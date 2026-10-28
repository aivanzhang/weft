"""Derive raster-export poses from the unchanged layered vector master."""
import argparse
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
