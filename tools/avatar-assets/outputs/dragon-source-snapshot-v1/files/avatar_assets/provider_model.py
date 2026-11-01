"""Local-only inspection of generated geometry before rigging or visual acceptance."""

import io
import json
from pathlib import Path
import shutil

from PIL import Image

from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, local_file, new_directory, preserve_failure, read_json, require, sha256, write_json
from avatar_assets.three_d import _GLB

ROOT = Path(__file__).resolve().parents[1]


def preflight(path):
    glb = _GLB(path)
    vertices, triangles, attributes = 0, 0, []
    for mesh in glb.doc.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            require(primitive.get("mode", 4) == 4 and not primitive.get("targets"), "Require triangles without morph targets")
            attrs = primitive.get("attributes", {})
            points = glb.accessor(attrs.get("POSITION"), "VEC3", (5126,))
            require(points, "Empty generated geometry")
            vertices += len(points)
            require(vertices <= 100000, "Generated model exceeds the 100000-vertex local profile")
            indices = glb.accessor(primitive.get("indices"), "SCALAR", (5121, 5123, 5125), count_limit=300000)
            require(len(indices) % 3 == 0 and all(0 <= row[0] < len(points) for row in indices), "Invalid triangle indices")
            triangles += len(indices)//3
            require(triangles <= 100000, "Generated model exceeds 100000 triangles")
            for name, kind, types in (("NORMAL", "VEC3", (5126,)), ("TEXCOORD_0", "VEC2", (5126,))):
                if name in attrs:
                    require(len(glb.accessor(attrs[name], kind, types)) == len(points), "Attribute count mismatch")
            attributes.append(sorted(attrs))
    images = []
    for source in glb.doc.get("images", []):
        view = glb.doc["bufferViews"][source["bufferView"]]
        start = view.get("byteOffset", 0)
        data = glb.binary[start:start+view["byteLength"]]
        try:
            with Image.open(io.BytesIO(data)) as image:
                require(image.width*image.height <= 16_777_216 and getattr(image, "n_frames", 1) == 1,
                        "Embedded texture exceeds the 16MP single-image limit")
                images.append({"format": image.format, "width": image.width, "height": image.height})
                image.verify()
        except Image.DecompressionBombError:
            raise ValueError("Embedded texture exceeds the 16MP or stricter process safety limit") from None
    require(vertices and triangles, "Generated GLB has no renderable geometry")
    return {"schema_version": 1, "sha256": glb.hash, "vertices": vertices, "triangles": triangles,
            "skins": len(glb.doc.get("skins", [])), "animations": len(glb.doc.get("animations", [])),
            "attributes": attributes, "textures": images, "source_unchanged": True,
            "scope": "Bounded geometry/embedded-image preflight; not likeness, rig or motion acceptance"}


def preview(model, output, resolution=512, samples=16):
    require(type(resolution) is int and 128 <= resolution <= 1600, "Invalid preview size")
    require(type(samples) is int and 1 <= samples <= 64, "Invalid preview samples")
    model = local_file(model, 64*1024*1024)
    facts = preflight(model)
    out = new_directory(output)
    try:
        (out/"sources").mkdir()
        shutil.copyfile(model, out/"original.glb")
        for name in ("blender_provider_preview.py", "blender_verify.py"):
            shutil.copyfile(ROOT/"scripts"/name, out/"sources"/name)
        write_json(out/"preflight.json", facts)
        exe = ROOT/".tools/Blender.app/Contents/MacOS/Blender"
        require(exe.is_file(), "Local Blender missing")
        command = [str(exe), "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
                   "--threads", "4", "--python", str(out/"sources/blender_provider_preview.py"), "--",
                   str(out/"original.glb"), str(out), str(resolution), str(samples)]
        job = run_bounded(command, out/"preview.log", out, timeout=300, memory_mb=2048, output_mb=256)
        write_json(out/"job.json", job)
        require(job["exit_code"] == 0 and not job["failure"], "Native preview failed; inspect retained job evidence")
        require(sha256(model) == facts["sha256"], "Original model changed")
        report = read_json(out/"native-inspection.json")
        write_json(out/"manifest.json", {"schema_version": 1, "kind": "generated_geometry_preview",
                   "status": "needs_visual_review", "source_glb_sha256": facts["sha256"], "preflight": facts,
                   "native": report, "provider_calls": 0, "new_provider_cost_usd": 0, "artifacts": artifacts(out)})
        return {"status": "needs_visual_review", "out": str(out), "vertices": facts["vertices"], "triangles": facts["triangles"]}
    except Exception as error:
        preserve_failure(out, error)
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(preview(args.model, args.out, args.resolution, args.samples), indent=2))
