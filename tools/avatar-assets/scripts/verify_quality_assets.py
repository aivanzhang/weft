"""Read-only quality evidence using the existing GLB validator; no renderer or importer."""

from array import array
import argparse
from collections import Counter
import gc
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import resource
import shutil
import signal
import sys
import time

from PIL import Image, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.safety import artifacts, contained, local_file, new_directory, read_json, require, sha256, write_json
from avatar_assets.scene import read_scene
from avatar_assets.three_d import _GLB, _WIDTHS, MAX_VERTICES, validate_glb, validate_render_attributes

UNIQUE_SCALARS = 3_000_000
DECODE_WORK_SCALARS = 6_000_000
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_TEXTURE_PIXELS = 16_777_216


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value*1024


class Budget:
    def __init__(self, seconds=120, memory_mb=1024):
        self.start = time.monotonic()
        self.seconds, self.memory_bytes = seconds, memory_mb*1024*1024

    def check(self):
        require(time.monotonic()-self.start < self.seconds, "Proof runtime limit exceeded")
        require(peak_rss_bytes() <= self.memory_bytes, "Proof process RSS limit exceeded")


class AttributeReader:
    """Reuse one-accessor decoding, with explicit whole-file work and residency bounds."""
    def __init__(self, glb, budget):
        self.glb, self.budget = glb, budget
        self.seen, self.unique, self.work = set(), 0, 0
        total = 0
        for accessor in glb.doc.get("accessors", []):
            count, width = accessor.get("count"), _WIDTHS.get(accessor.get("type"))
            require(type(count) is int and 0 < count <= MAX_VERTICES and width, "Invalid accessor bound")
            total += count*width
        require(total <= UNIQUE_SCALARS, "Quality proof unique-scalar limit exceeded")

    def rows(self, index, kind=None, components=None):
        self.budget.check()
        accessor = self.glb.doc["accessors"][index]
        amount = accessor["count"]*_WIDTHS[accessor["type"]]
        self.work += amount
        require(self.work <= DECODE_WORK_SCALARS, "Quality proof decoded-work limit exceeded")
        if index not in self.seen:
            self.seen.add(index)
            self.unique += amount
        # Clearing the cache releases Python tuples; compact arrays retain only needed cross-attribute data.
        self.glb.cache.clear()
        self.glb.value_count = 0
        return self.glb.accessor(index, kind, components)


def _tracked_file(pack, inventory, relative, maximum=64*1024*1024):
    require(relative in inventory, f"File absent from pack inventory: {relative}")
    path = local_file(contained(pack, relative), maximum)
    record = inventory[relative]
    require(path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"],
            f"Preserved artifact hash differs: {relative}")
    return path


def _triangle_features(positions, indices, budget):
    uses = Counter(indices)
    isolated, slender, zero_area = 0, 0, 0
    for offset in range(0, len(indices), 3):
        if offset % 12288 == 0:
            budget.check()
        tri = indices[offset:offset+3]
        points = [positions[3*index:3*index+3] for index in tri]
        a, b, c = points
        u, v = [b[i]-a[i] for i in range(3)], [c[i]-a[i] for i in range(3)]
        cross = (u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0])
        zero_area += sum(value*value for value in cross) <= 1e-20
        if all(uses[index] == 1 for index in tri):
            isolated += 1
            edges = sorted(math.dist(points[i], points[(i+1) % 3]) for i in range(3))
            slender += .0002 <= edges[0] <= .005 and .001 <= edges[-1] <= .052 and edges[-1] >= edges[0]*3
    return {"isolated_triangles": isolated, "isolated_slender_triangles": slender, "near_zero_area_triangles": zero_area}


def _primitive(reader, primitive, joints_count, budget):
    require(primitive.get("mode", 4) == 4, "Quality proof expects triangle geometry")
    attrs = primitive["attributes"]
    require({"POSITION", "NORMAL", "TEXCOORD_0", "TANGENT", "JOINTS_0", "WEIGHTS_0"} <= attrs.keys(),
            "Missing portable surface or skin attribute")
    require(not any(name.startswith(("JOINTS_", "WEIGHTS_")) and name not in ("JOINTS_0", "WEIGHTS_0") for name in attrs),
            "More than four exported skin influences")
    positions = array("f", (value for row in reader.rows(attrs["POSITION"], "VEC3", (5126,)) for value in row))
    count = len(positions)//3
    normals = array("f", (value for row in reader.rows(attrs["NORMAL"], "VEC3", (5126,)) for value in row))
    require(len(normals) == 3*count, "Normal count differs from positions")
    normal_error = max(abs(sum(v*v for v in normals[i:i+3])-1) for i in range(0, len(normals), 3))
    require(normal_error < .001, "Invalid exported normal length")
    rows = reader.rows(attrs["TEXCOORD_0"], "VEC2")
    require(len(rows) == count, "UV count differs from positions")
    uv_bounds = [[min(row[axis] for row in rows), max(row[axis] for row in rows)] for axis in range(2)]
    del rows
    tangent_error, orthogonality = 0., 0.
    invalid_tangents, zero_tangents = 0, 0
    rows = reader.rows(attrs["TANGENT"], "VEC4", (5126,))
    require(len(rows) == count, "Tangent count differs from positions")
    for index, row in enumerate(rows):
        length_squared = sum(v*v for v in row[:3])
        length_error = abs(length_squared-1)
        dot_error = abs(sum(normals[3*index+i]*row[i] for i in range(3)))
        tangent_error = max(tangent_error, length_error)
        orthogonality = max(orthogonality, dot_error)
        invalid_tangents += length_error >= .001 or dot_error >= .002
        zero_tangents += length_squared < 1e-12
        require(abs(abs(row[3])-1) < 1e-6, "Invalid tangent handedness")
    del rows, normals
    joints = array("H", (value for row in reader.rows(attrs["JOINTS_0"], "VEC4", (5121, 5123)) for value in row))
    require(len(joints) == 4*count and all(index < joints_count for index in joints), "Joint index outside skeleton")
    rows = reader.rows(attrs["WEIGHTS_0"], "VEC4", (5121, 5123, 5126))
    require(len(rows) == count, "Weight count differs from positions")
    influences, error = Counter(), 0.
    for index, row in enumerate(rows):
        error = max(error, abs(sum(row)-1))
        require(min(row) >= 0 and max(row) <= 1, "Skin weight outside [0,1]")
        distinct = {joints[4*index+i] for i, weight in enumerate(row) if weight > 1e-6}
        require(1 <= len(distinct) <= 4, "Unweighted or excessive-influence vertex")
        influences[len(distinct)] += 1
    require(error < .001, "Exported weights are not normalized")
    del rows, joints
    indices = array("I", (row[0] for row in reader.rows(primitive["indices"], "SCALAR", (5121, 5123, 5125)))) if "indices" in primitive else array("I", range(count))
    require(len(indices) % 3 == 0 and all(index < count for index in indices), "Invalid triangle index")
    features = _triangle_features(positions, indices, budget)
    return {"material": primitive["material"], "vertices": count, "triangles": len(indices)//3,
            "attributes_finite": True, "uv0_bounds": uv_bounds, "normal_squared_length_error": normal_error,
            "tangent_squared_length_error": tangent_error, "normal_tangent_dot_error": orthogonality,
            "invalid_tangent_vertices": invalid_tangents, "zero_tangent_vertices": zero_tangents,
            "weight_sum_error": error, "influence_histogram": dict(sorted(influences.items())),
            "mixed_weight_vertices": sum(number for influences, number in influences.items() if influences > 1),
            **features}


def verify_pack(identifier, pack, budget):
    started = time.monotonic()
    pack = Path(pack).resolve()
    require(pack.is_relative_to(ROOT) and pack.is_dir(), "Proof packs must be inside this workspace")
    manifest_path = local_file(pack/"manifest.json", 1024*1024)
    manifest = read_json(manifest_path)
    require(manifest.get("kind") == "authored_3d_avatar" and manifest.get("status") != "failed", "Pack is not a completed authored 3D build")
    entries = manifest.get("artifacts")
    require(isinstance(entries, list) and 0 < len(entries) <= 8192, "Invalid artifact inventory")
    inventory = {item["path"]: item for item in entries}
    require(len(inventory) == len(entries), "Duplicate inventory path")
    model = _tracked_file(pack, inventory, "model.glb")
    contract_path = _tracked_file(pack, inventory, "rig-contract.json")
    report_path = _tracked_file(pack, inventory, "blender-build.json")
    scene_path = _tracked_file(pack, inventory, "sources/scene.json", 8*1024*1024)
    blend = _tracked_file(pack, inventory, "source.blend", 128*1024*1024)
    scene, backend = read_scene(scene_path), read_json(report_path)
    budget.check()
    structural = validate_glb(model, read_json(contract_path))
    budget.check()
    require(structural["motion"]["animation_verified"], "Existing validator did not observe all clip motions")
    glb = _GLB(model)
    require(glb.hash == inventory["model.glb"]["sha256"], "GLB changed during verification")
    names = [animation.get("name") for animation in glb.doc.get("animations", [])]
    require(len(names) == 4 and set(names) == {state["id"] for state in scene["states"]}, "Expected all four authored clips")
    reader = AttributeReader(glb, budget)
    primitives = []
    for node in glb.doc["nodes"]:
        if "mesh" not in node:
            continue
        require("skin" in node, "Unskinned exported mesh")
        joints_count = len(glb.doc["skins"][node["skin"]]["joints"])
        for primitive in glb.doc["meshes"][node["mesh"]]["primitives"]:
            primitives.append(_primitive(reader, primitive, joints_count, budget))
    for index in range(len(glb.doc["accessors"])):
        if index not in reader.seen:
            reader.rows(index)
    glb.cache.clear()
    vertices, triangles = sum(item["vertices"] for item in primitives), sum(item["triangles"] for item in primitives)
    require(vertices == structural["mesh_vertices"] == backend["exported_vertices"] <= MAX_VERTICES,
            "Exported vertex evidence disagrees")
    textures, expected_hashes, pixels = [], Counter(), 0
    for texture in backend.get("textures", []):
        source = _tracked_file(pack, inventory, texture["path"], MAX_IMAGE_BYTES)
        digest = sha256(source)
        require(digest == texture["sha256"], "Backend texture hash differs")
        expected_hashes[digest] += 1
        textures.append({**texture, "absolute_path": str(source)})
    embedded, actual_hashes = [], Counter()
    for image in glb.doc.get("images", []):
        require(image.get("mimeType") == "image/png" and "uri" not in image, "Texture is not embedded PNG")
        view = glb.doc["bufferViews"][image["bufferView"]]
        require(view["byteLength"] <= MAX_IMAGE_BYTES, "Embedded PNG exceeds byte limit")
        start = view.get("byteOffset", 0)
        raw = glb.binary[start:start+view["byteLength"]]
        digest = hashlib.sha256(raw).hexdigest()
        actual_hashes[digest] += 1
        with Image.open(BytesIO(raw)) as decoded:
            require(decoded.format == "PNG" and decoded.width in (256, 512) and decoded.height == decoded.width,
                    "Unexpected baked PNG dimensions")
            pixels += decoded.width*decoded.height
            require(pixels <= MAX_TEXTURE_PIXELS, "Decoded PNG pixel budget exceeded")
            decoded.load()
            embedded.append({"name": image.get("name"), "sha256": digest, "bytes": len(raw),
                             "width": decoded.width, "height": decoded.height, "mode": decoded.mode})
    require(actual_hashes == expected_hashes, "Embedded PNG bytes differ from preserved texture sources")
    materials = []
    for material in glb.doc["materials"]:
        source = scene["materials"][material["name"]]
        pbr = material.get("pbrMetallicRoughness", {})
        roughness, metallic = pbr.get("roughnessFactor", 1), pbr.get("metallicFactor", 1)
        require(abs(roughness-source.get("roughness", .5)) < 1e-6 and abs(metallic-source.get("metallic", 0)) < 1e-6,
                "Exported PBR settings differ from source")
        require(material.get("alphaMode", "OPAQUE") == "OPAQUE", "Unexpected transparent material")
        normal, base = "normalTexture" in material, "baseColorTexture" in pbr
        require("surface" not in source or normal and base, "Baked surface is missing a portable material texture")
        for container, field in ((pbr, "baseColorTexture"), (material, "normalTexture")):
            if field in container:
                require(container[field].get("texCoord", 0) == 0, "Texture does not use verified UV0")
        materials.append({"name": material["name"], "roughness": roughness, "metallic": metallic,
                          "basecolor_texture": base, "normal_texture": normal, "alpha_mode": "OPAQUE"})
    require(len(embedded) == backend["embedded_images"], "Embedded image count differs from backend report")
    require(sha256(model) == glb.hash, "GLB changed during verification")
    budget.check()
    invalid_tangents = sum(item["invalid_tangent_vertices"] for item in primitives)
    try:
        surface_gate = validate_render_attributes(model)
    except ValueError as error:
        surface_gate = {"status": "failed", "error": str(error)}
    failures = [f"{invalid_tangents} vertices have invalid tangent frames"] if invalid_tangents else []
    if surface_gate["status"] == "failed":
        failures.append(surface_gate["error"])
    return {"schema_version": 1, "id": identifier, "status": "failed" if failures else "passed", "pack": str(pack),
            "validation_failures": failures,
            "input_hashes": {"manifest": sha256(manifest_path), "model": glb.hash,
                             "scene": sha256(scene_path), "backend_report": sha256(report_path), "blend": sha256(blend)},
            "model_bytes": model.stat().st_size, "vertices": vertices, "triangles": triangles,
            "joints": len(structural["bones"]), "clips": structural["motion"]["animations"],
            "primitives": primitives, "materials": materials, "pbr_material_count": len(materials),
            "normal_mapped_material_count": sum(material["normal_texture"] for material in materials),
            "embedded_images": embedded, "preserved_textures": textures,
            "embedded_png_hashes_equal_preserved_sources": True,
            "mixed_weight_vertices_exported": sum(item["mixed_weight_vertices"] for item in primitives),
            "automatic_surface_gate": surface_gate,
            "fur": {"source_reported_strands": backend["fur_strands"],
                    "source_reported_strand_vertices": sum(part["fur_vertices"] for part in backend["parts"]),
                    "independent_isolated_slender_triangles": sum(item["isolated_slender_triangles"] for item in primitives),
                    "distinction": "Strand count comes from hash-verified backend metadata. Slender isolated triangles are independent geometry evidence, not recovered strand labels; small non-fur details can also match.",
                    "geometry_criterion": "Unshared triangle indices; shortest edge0.0002–0.005m, longest0.001–0.052m, aspect>=3"},
            "reader": {"unique_decoded_scalars": reader.unique, "total_decoded_work_scalars": reader.work,
                       "method": "Existing _GLB.accessor decoder with cache released per attribute; compact cross-attribute arrays only."},
            "elapsed_seconds": round(time.monotonic()-started, 4),
            "process_cumulative_peak_rss_bytes": peak_rss_bytes(),
            "not_verified_here": ["Native render appearance", "Reference likeness", "Weft runtime alignment", "Seamless loops"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", action="append", required=True, metavar="ID=PATH")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--memory-mb", type=int, default=1024)
    args = parser.parse_args()
    require(1 <= len(args.pack) <= 4 and 10 <= args.timeout <= 120 and 256 <= args.memory_mb <= 2048, "Invalid proof bounds")
    require(args.output.resolve().is_relative_to(ROOT), "Proof output must stay inside this workspace")
    packs = []
    for value in args.pack:
        identifier, separator, path = value.partition("=")
        require(separator and identifier.replace("-", "").isalnum() and len(identifier) <= 64, "Use ID=PATH")
        require(identifier not in {item[0] for item in packs}, "Duplicate proof ID")
        packs.append((identifier, path))
    out = new_directory(args.output)
    budget = Budget(args.timeout, args.memory_mb)
    sources = out/"sources"
    sources.mkdir()
    for source in (Path(__file__), ROOT/"avatar_assets/three_d.py", ROOT/"avatar_assets/scene.py", ROOT/"avatar_assets/safety.py"):
        shutil.copyfile(source, sources/source.name)
    reports = []

    def timeout(*_):
        raise TimeoutError("Quality proof wall-time budget exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, args.timeout)
    try:
        for identifier, pack in packs:
            report = verify_pack(identifier, pack, budget)
            write_json(out/(identifier+".json"), report)
            reports.append({"id": identifier, "report": identifier+".json", "model_sha256": report["input_hashes"]["model"],
                            "vertices": report["vertices"], "triangles": report["triangles"], "status": report["status"]})
            gc.collect()
        status = "passed" if all(report["status"] == "passed" for report in reports) else "failed"
        error = None if status == "passed" else "See asset reports for failed attribute checks"
    except Exception as failure:
        status, error = "failed", str(failure)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    result = {"schema_version": 1, "kind": "independent_quality_attribute_proof", "status": status,
              "command": sys.argv, "reports": reports, "error": error,
              "limits": {"wall_seconds": args.timeout, "process_rss_mb": args.memory_mb,
                         "unique_scalars_per_glb": UNIQUE_SCALARS, "decoded_work_scalars_per_glb": DECODE_WORK_SCALARS,
                         "png_bytes_each": MAX_IMAGE_BYTES, "png_pixels_total_per_glb": MAX_TEXTURE_PIXELS,
                         "exported_vertices": MAX_VERTICES, "existing_glb_validator_limits_unchanged": True},
              "elapsed_seconds": round(time.monotonic()-budget.start, 4), "process_peak_rss_bytes": peak_rss_bytes(),
              "tools": {"python": sys.version, "Pillow": PILLOW_VERSION}, "provider_calls": 0, "provider_cost_usd": 0,
              "artifacts": artifacts(out)}
    write_json(out/"manifest.json", result)
    print(json.dumps({key: result[key] for key in ("status", "reports", "error", "elapsed_seconds", "process_peak_rss_bytes")}))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
