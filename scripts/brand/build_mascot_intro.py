"""Build the one-shot weave entrance from the editable mascot SVG."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/brand/source/weft-mascot-rig.svg"
OUTPUT = ROOT / "assets/brand/exports/weft-mascot-forming.svg"
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("inkscape", "http://www.inkscape.org/namespaces/inkscape")

# Bounds include strokes; clipping must reveal strands without thinning their edges.
STRANDS = [
    ("weave-rear", 0, "up", (390, 100, 120, 86), 200, 400),
    ("weave-rear", 1, "down", (460, 145, 130, 275), 300, 620),
    ("weave-rear", 2, "down", (474, 150, 116, 128), 420, 430),
    ("weave-rear", 3, "left", (460, 225, 125, 124), 900, 420),
    ("weave-rear", 4, "down", (460, 297, 98, 120), 1100, 390),
    ("weave-front", 0, "down", (104, 93, 162, 124), 60, 500),
    ("weave-front", 1, "right", (194, 110, 80, 109), 450, 380),
    ("weave-front", 2, "right", (76, 186, 203, 118), 560, 550),
    ("weave-front", 3, "down", (237, 86, 173, 214), 340, 740),
    ("weave-front", 4, "right", (337, 130, 175, 108), 940, 510),
    ("weave-front", 5, "down", (377, 203, 109, 200), 790, 650),
]
LAYERS = [
    ("body", "down", (88, 192, 462, 363), 850, 650),
    ("foot-left", "down", (157, 505, 116, 82), 1380, 280),
    ("foot-right", "down", (340, 512, 119, 79), 1440, 280),
    ("arm-left", "left", (50, 377, 109, 109), 1460, 280),
    ("arm-right", "right", (489, 384, 104, 104), 1510, 280),
    ("mend-stitch", "right", (436, 324, 85, 54), 1570, 330),
]
FACE = [("eyes-open", 1780, 220), ("cheeks", 1820, 200), ("mouth", 1890, 260)]
DURATION_MS = max(delay + duration for _, delay, duration in FACE)


def progress(at_ms, delay, duration):
    return max(0.0, min(1.0, (at_ms - delay) / duration))


def build_svg(at_ms=None, reverse=False):
    root = ET.fromstring(SOURCE.read_text())
    root.set("class", "weft-forming")
    root.set("data-duration-ms", str(DURATION_MS))
    root.find(f"{{{SVG}}}title").text = "Weft — unweaving" if reverse else "Weft — forming weave"
    root.find(f"{{{SVG}}}desc").text = (
        "The woven strands form, the stitch closes, and the mascot appears. "
        "Plays once in 2.15 seconds. Reduced motion applies the endpoint immediately.")
    nodes = {node.get("id"): node for node in root.iter() if node.get("id")}
    definitions = ET.Element(f"{{{SVG}}}defs")
    root.insert(2, definitions)
    rules = []
    serial = 0
    if reverse and at_ms is not None:
        at_ms = DURATION_MS - at_ms

    def timing(delay, duration):
        return f"{duration}ms linear {DURATION_MS-delay-duration if reverse else delay}ms 1 {'reverse' if reverse else 'normal'} both"

    def reveal(node, direction, bounds, delay, duration):
        nonlocal serial
        serial += 1
        name = f"weft-forming-reveal-{serial}"
        clip = ET.SubElement(definitions, f"{{{SVG}}}clipPath", {
            "id": name, "clipPathUnits": "userSpaceOnUse"})
        x, y, width, height = bounds
        if at_ms is not None:
            value = progress(at_ms, delay, duration)
            if direction in ("left", "right"):
                if direction == "left":
                    x += width * (1 - value)
                width *= value
            else:
                if direction == "up":
                    y += height * (1 - value)
                height *= value
        ET.SubElement(clip, f"{{{SVG}}}rect", {
            "id": name + "-mask", "x": str(x), "y": str(y),
            "width": str(width), "height": str(height)})
        node.set("clip-path", f"url(#{name})")
        if at_ms is None:
            axis = "x" if direction in ("left", "right") else "y"
            origin = {"left": "right center", "right": "left center",
                      "down": "center top", "up": "center bottom"}[direction]
            rules.append(f"#{name}-mask{{transform-box:fill-box;transform-origin:{origin};"
                         f"animation:weft-forming-{axis} {timing(delay, duration)}}}")

    for group, index, direction, bounds, delay, duration in STRANDS:
        reveal(list(nodes[group])[index], direction, bounds, delay, duration)
    for name, direction, bounds, delay, duration in LAYERS:
        reveal(nodes[name], direction, bounds, delay, duration)
    for name, delay, duration in FACE:
        if at_ms is None:
            rules.append(f"#{name}{{animation:weft-forming-face {timing(delay, duration)}}}")
        else:
            nodes[name].set("opacity", str(progress(at_ms, delay, duration)))
    if at_ms is None:
        style = ET.Element(f"{{{SVG}}}style")
        style.text = ("\n@keyframes weft-forming-x{from{transform:scaleX(0)}to{transform:scaleX(1)}}\n"
                      "@keyframes weft-forming-y{from{transform:scaleY(0)}to{transform:scaleY(1)}}\n"
                      "@keyframes weft-forming-face{from{opacity:0}to{opacity:1}}\n"
                      + "\n".join(rules)
                      + "\n@media(prefers-reduced-motion:reduce){.weft-forming *{animation:none!important}"
                      + ("#mascot{visibility:hidden}" if reverse else "") + "}\n")
        root.insert(3, style)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify generated SVG without writing")
    parser.add_argument("--snapshots", type=Path, help="Write static structural keyframes for native visual QA")
    args = parser.parse_args()
    metadata = {"image": OUTPUT.name, "reverse": "weft-mascot-unweaving.svg", "empty": "weft-mascot-empty.svg", "still": "../source/" + SOURCE.name, "duration_ms": DURATION_MS}
    outputs = {OUTPUT: build_svg(), OUTPUT.parent / metadata["reverse"]: build_svg(reverse=True),
               OUTPUT.parent / metadata["empty"]: build_svg(0), OUTPUT.with_suffix(".js"):
               "// Generated by scripts/brand/build_mascot_intro.py.\nwindow.WEFT_MASCOT_INTRO = "
               + json.dumps(metadata, separators=(",", ":")) + ";\n"}
    stale = [path for path, content in outputs.items() if not path.is_file() or path.read_text() != content]
    if args.check and stale:
        raise SystemExit("Stale formation assets: run python3 scripts/brand/build_mascot_intro.py")
    if args.check:
        print("Formation assets are current")
    else:
        for path in stale:
            path.write_text(outputs[path])
            print(path.relative_to(ROOT))
    if args.snapshots:
        args.snapshots.mkdir(parents=True, exist_ok=True)
        for time_ms in (0, 500, 1000, 1500, DURATION_MS):
            (args.snapshots / f"forming-{time_ms:04}.svg").write_text(build_svg(time_ms))


if __name__ == "__main__":
    main()
