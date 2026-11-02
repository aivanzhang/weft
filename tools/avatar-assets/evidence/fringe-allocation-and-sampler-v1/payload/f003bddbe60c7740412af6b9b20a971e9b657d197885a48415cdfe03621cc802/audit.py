"""Independent area-candidate audit; preserved color defects prevent acceptance."""
import copy
import hashlib
import json
import mmap
import struct
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DTYPES = {5121: np.dtype("u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(name, value):
    with (HERE / name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


class GLB:
    def __init__(self, path):
        self.path = Path(path)
        require(self.path.is_file() and not self.path.is_symlink() and 28 <= self.path.stat().st_size <= 64 * 1024**2,
                "Invalid GLB file bound")
        self.handle = self.path.open("rb")
        self.data = mmap.mmap(self.handle.fileno(), 0, access=mmap.ACCESS_READ)
        magic, version, total, js, jt = struct.unpack_from("<4sIIII", self.data)
        require(magic == b"glTF" and version == 2 and total == len(self.data) and jt == 0x4E4F534A
                and js <= 2*1024**2 and js % 4 == 0, "Invalid GLB JSON header")
        self.doc = json.loads(self.data[20:20+js])
        size, kind = struct.unpack_from("<II", self.data, 20+js)
        self.offset = 28+js
        self.bin_size = size
        require(kind == 0x004E4942 and size % 4 == 0 and self.offset+size == len(self.data), "Invalid BIN chunk")
        require(len(self.doc["buffers"]) == 1 and "uri" not in self.doc["buffers"][0], "External buffer")
        self.declared = self.doc["buffers"][0]["byteLength"]
        require(type(self.declared) is int and 0 <= size-self.declared <= 3, "Invalid embedded buffer size")
        self.decoded = 0

    def array(self, index, kind, components, limit=100000):
        require(type(index) is int and 0 <= index < len(self.doc["accessors"]), "Invalid accessor index")
        a = self.doc["accessors"][index]
        require(a["type"] == kind and a["componentType"] in components and not a.get("sparse")
                and not a.get("normalized", False), "Unexpected accessor representation")
        count, width, dtype = a["count"], WIDTHS[kind], DTYPES[a["componentType"]]
        require(type(count) is int and 0 < count <= limit, "Accessor bound")
        view = self.doc["bufferViews"][a["bufferView"]]
        offset, length, stride = view.get("byteOffset", 0), view["byteLength"], view.get("byteStride", width*dtype.itemsize)
        start = a.get("byteOffset", 0)
        require(view["buffer"] == 0 and all(type(x) is int and x >= 0 for x in (offset, length, stride, start))
                and stride >= width*dtype.itemsize and stride <= 252 and stride % dtype.itemsize == 0
                and offset+length <= self.declared and start+(count-1)*stride+width*dtype.itemsize <= length,
                "Accessor storage exceeds bound")
        result = np.ndarray((count, width), dtype=dtype, buffer=self.data, offset=self.offset+offset+start,
                            strides=(stride, dtype.itemsize)).copy()
        self.decoded += count*width
        require(self.decoded <= 2000000 and np.isfinite(result).all(), "Nonfinite or oversized decoded values")
        return result

    def view_bytes(self, index):
        view = self.doc["bufferViews"][index]
        require(view["buffer"] == 0 and view.get("byteOffset", 0)+view["byteLength"] <= self.declared, "Invalid view")
        start = self.offset+view.get("byteOffset", 0)
        return self.data[start:start+view["byteLength"]]


def npz(path, required):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(len(entries) == len(required) and sum(e.file_size for e in entries) <= 4*1024**2
                and all(e.file_size <= 2*1024**2 for e in entries), "NPZ expansion bound")
    with np.load(path, allow_pickle=False) as values:
        require(set(values.files) == set(required), "Unexpected correspondence arrays")
        result = {name: values[name] for name in required}
    require(all(a.dtype.kind in "iuf" and np.isfinite(a).all() for a in result.values()), "Invalid NPZ array")
    return result


def main():
    started = time.monotonic()
    prepared = json.loads((HERE / "prepared.json").read_text())
    for item in prepared["inputs"] + prepared["sources"]:
        require(sha(item["path"]) == item["sha256"], "Pinned input changed")
    source, target = GLB(prepared["source"]), GLB(prepared["candidate"])
    s, t = source.doc, target.doc
    require(source.declared == source.bin_size, "Original BIN padding requires separate explicit accounting")
    require(target.declared > source.declared, "No appended data")
    original_hash = hashlib.sha256()
    for start in range(0, source.declared, 1024**2):
        end = min(start+1024**2, source.declared)
        old = source.data[source.offset+start:source.offset+end]
        new = target.data[target.offset+start:target.offset+end]
        require(old == new, "Original BIN prefix changed")
        original_hash.update(old)
    appended_counts = {"accessors": 7, "bufferViews": 8, "images": 1, "materials": 1, "textures": 1, "samplers": 1}
    expected = copy.deepcopy(s)
    for key, added in appended_counts.items():
        require(len(t[key]) == len(s[key])+added and t[key][:len(s[key])] == s[key], "Original JSON prefix changed: "+key)
        expected[key] += copy.deepcopy(t[key][len(s[key]):])
    require(len(s["meshes"]) == 3 and len(t["meshes"]) == 1 and len(s["nodes"]) == 23, "Unexpected source structure")
    require(len(t["meshes"][0]["primitives"]) == len(s["meshes"][0]["primitives"])+1, "Unexpected new primitive count")
    new_primitive = t["meshes"][0]["primitives"][-1]
    expected["meshes"] = expected["meshes"][:1]
    expected["meshes"][0]["primitives"].append(copy.deepcopy(new_primitive))
    for node_id, old_mesh in ((21, 1), (22, 2)):
        require(s["nodes"][node_id].get("mesh") == old_mesh and s["nodes"][node_id].get("skin") == 0,
                "Unexpected old coat node")
        expected["nodes"][node_id].pop("mesh")
        expected["nodes"][node_id].pop("skin")
    expected["buffers"][0]["byteLength"] = target.declared
    expected["asset"]["generator"] = "Local experimental edge-fringe derivative; source body, rig and motion unchanged"
    require(t == expected, "Candidate JSON exceeds explicit allowlist")
    require(t["skins"] == s["skins"] and t["animations"] == s["animations"] and t["scenes"] == s["scenes"], "Rig/action/scene changed")
    old_views, old_accessors = len(s["bufferViews"]), len(s["accessors"])
    cursor = source.declared
    for view in t["bufferViews"][old_views:]:
        offset = view.get("byteOffset", 0)
        require(view["buffer"] == 0 and offset == (cursor+3)//4*4 and view["byteLength"] > 0,
                "Appended view aliases old data, another view, or unexpected gap")
        require(target.data[target.offset+cursor:target.offset+offset] == b"\0"*(offset-cursor), "Nonzero alignment padding")
        cursor = offset+view["byteLength"]
    require(cursor == target.declared and target.data[target.offset+cursor:] == b"\0"*(target.bin_size-cursor), "Unexpected trailing BIN payload")
    fields = {"POSITION": ("VEC3", 5126), "NORMAL": ("VEC3", 5126), "TEXCOORD_0": ("VEC2", 5126),
              "COLOR_0": ("VEC3", 5126), "JOINTS_0": ("VEC4", 5123), "WEIGHTS_0": ("VEC4", 5126)}
    require(new_primitive == {"attributes": {name: old_accessors+i for i, name in enumerate(fields)},
                              "indices": old_accessors+6, "mode": 4, "material": len(s["materials"])}, "Unexpected fin primitive binding")
    fin_arrays = {}
    for i, (name, (kind, component)) in enumerate(fields.items()):
        a = t["accessors"][old_accessors+i]
        require(a["bufferView"] == old_views+i and not a.get("byteOffset", 0), "Fin accessor not exclusive")
        fin_arrays[name] = target.array(old_accessors+i, kind, (component,))
    indices = target.array(old_accessors+6, "SCALAR", (5125,), 300000).reshape(-1, 3)
    require(t["accessors"][old_accessors+6]["bufferView"] == old_views+6, "Index view is not exclusive")
    points = fin_arrays["POSITION"]
    require(len(points) % 4 == 0 and all(len(a) == len(points) for a in fin_arrays.values()), "Fin attribute counts mismatch")
    count = len(points)//4
    require(1 <= count <= 4800 and len(indices) == count*2, "Fin count bound")
    expected_triangles = (np.arange(count)[:, None, None]*4 + np.array([[0, 1, 3], [0, 3, 2]])).reshape(-1, 3)
    require(np.array_equal(indices, expected_triangles), "Fin winding/connectivity differs from two quad triangles")
    for key, value in fin_arrays.items():
        a = t["accessors"][new_primitive["attributes"][key]]
        if "min" in a:
            require(np.array_equal(a["min"], value.min(axis=0)) and np.array_equal(a["max"], value.max(axis=0)), "Accessor bounds mismatch")
    require((fin_arrays["TEXCOORD_0"] >= 0).all() and (fin_arrays["TEXCOORD_0"] <= 1).all()
            and (fin_arrays["COLOR_0"] >= 0).all() and (fin_arrays["COLOR_0"] <= 1).all(), "UV/color range")
    uv4 = fin_arrays["TEXCOORD_0"].reshape(count, 4, 2)
    require(np.array_equal(uv4[:, [0, 2], 0], np.repeat(uv4[:, :1, 0], 2, axis=1))
            and np.array_equal(uv4[:, [1, 3], 0], np.repeat(uv4[:, 1:2, 0], 2, axis=1))
            and np.array_equal(uv4[:, :2, 1], np.repeat(uv4[:, :1, 1], 2, axis=1))
            and np.array_equal(uv4[:, 2:, 1], np.repeat(uv4[:, 2:3, 1], 2, axis=1))
            and (uv4[:, 1, 0] > uv4[:, 0, 0]).all() and (uv4[:, 0, 1] > uv4[:, 2, 1]).all(), "Atlas rectangle orientation")
    material = t["materials"][-1]
    require(material == {"name": "ExperimentalEdgeFringe", "doubleSided": True, "alphaMode": "MASK", "alphaCutoff": .18,
                         "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1],
                         "baseColorTexture": {"index": len(s["textures"]), "texCoord": 0}, "metallicFactor": 0, "roughnessFactor": .92}}, "Unexpected new material")
    require(t["textures"][-1] == {"source": len(s["images"]), "sampler": len(s["samplers"])}
            and t["samplers"][-1] == {"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}
            and t["images"][-1] == {"bufferView": old_views+7, "mimeType": "image/png", "name": "EdgeFringeTwoTips"}, "New image routing mismatch")
    image = target.view_bytes(old_views+7)
    require(image == Path(prepared["atlas"]).read_bytes() and image[:8] == b"\x89PNG\r\n\x1a\n"
            and image[12:16] == b"IHDR" and struct.unpack_from(">IIBB", image, 16) == (1024, 1024, 8, 6), "Atlas bytes/header mismatch")
    correspondence = npz(prepared["correspondence"], ["root_vertex_ids", "root_face_ids", "root_corner_ids", "root_source_uv",
                                                        "root_centers", "outward_normals", "fin_directions"])
    whitelist = npz(prepared["whitelist"], ["primitive_5_edges", "primitive_5_faces"])
    root_ids, face_ids, corner_ids = (correspondence[k] for k in ("root_vertex_ids", "root_face_ids", "root_corner_ids"))
    require(root_ids.shape == (count, 2) and face_ids.shape == (count,) and corner_ids.shape == (count, 2)
            and all(a.dtype.kind in "iu" for a in (root_ids, face_ids, corner_ids)), "Invalid root correspondence schema")
    primitive = s["meshes"][0]["primitives"][5]
    require(s["materials"][primitive["material"]]["name"] == "body_surface__short_pile_adaptive_v1", "Wrong source material")
    source_arrays = {name: source.array(primitive["attributes"][name], kind, (5121, 5123) if name == "JOINTS_0" else (component,))
                     for name, (kind, component) in fields.items() if name != "COLOR_0"}
    source_points, source_weights, source_joints = (source_arrays[k] for k in ("POSITION", "WEIGHTS_0", "JOINTS_0"))
    source_faces = source.array(primitive["indices"], "SCALAR", (5121, 5123, 5125), 300000).reshape(-1, 3)
    require((root_ids >= 0).all() and (root_ids < len(source_points)).all() and (root_ids[:, 0] < root_ids[:, 1]).all()
            and (face_ids >= 0).all() and (face_ids < len(source_faces)).all() and (corner_ids >= 0).all()
            and (corner_ids < 3).all(), "Source root/corner indices outside bounds")
    require(np.array_equal(source_faces[face_ids[:, None], corner_ids], root_ids), "Root vertex/face/corner correspondence differs")
    root_uv = source_arrays["TEXCOORD_0"][root_ids]
    require(correspondence["root_source_uv"].shape == root_uv.shape and correspondence["root_source_uv"].dtype == root_uv.dtype
            and correspondence["root_source_uv"].tobytes() == root_uv.tobytes(), "Source UV0 correspondence changed")
    root_points = source_points[root_ids]
    require(points.reshape(count, 4, 3)[:, :2].tobytes() == root_points.tobytes(), "Actual fin roots differ from source endpoints")
    require(correspondence["root_centers"].shape == (count, 3)
            and correspondence["root_centers"].tobytes() == root_points.mean(axis=1).tobytes(), "Root center correspondence differs")
    spans = np.linalg.norm(root_points[:, 1]-root_points[:, 0], axis=1)
    require((spans >= .0085).all() and (spans <= .020).all(), "Actual anchors exceed prepared edge range")
    # The cropped atlas changes UV span, not source endpoints or physical tuft width.
    tile = np.floor(uv4[:, 0]*4).astype(int)
    local_uv = uv4.astype(np.float64)*4-tile[:, None]
    half_span = spans.astype(np.float64)/(2*.0208)
    expected_uv = np.zeros((count, 4, 2))
    expected_uv[:, [0, 2], 0] = .5-half_span[:, None]
    expected_uv[:, [1, 3], 0] = .5+half_span[:, None]
    expected_uv[:, :2, 1] = .985
    expected_uv[:, 2:, 1] = .015
    crop_error = float(np.max(np.abs(local_uv-expected_uv)))
    require(crop_error < 5e-7 and (tile >= 0).all() and (tile < 4).all(), "Physical atlas crop differs from actual root span")
    nv = len(source_points)
    safe_edges, safe_faces = whitelist["primitive_5_edges"], whitelist["primitive_5_faces"]
    require(safe_edges.shape == safe_faces.shape and safe_edges.ndim == 2 and safe_edges.shape[1] == 2, "Invalid safe-edge schema")
    safe_codes = safe_edges[:, 0].astype(np.int64)*nv+safe_edges[:, 1]
    order = np.argsort(safe_codes)
    safe_codes = safe_codes[order]
    codes = root_ids[:, 0].astype(np.int64)*nv+root_ids[:, 1]
    lookup = np.searchsorted(safe_codes, codes)
    require(len(np.unique(codes)) == count and (lookup < len(safe_codes)).all()
            and np.array_equal(safe_codes[lookup], codes), "Fin edge absent from independent safe whitelist or duplicated")
    require((safe_faces[order[lookup]] == face_ids[:, None]).any(axis=1).all(), "Recorded root face not one of whitelisted adjacent faces")
    inherited_joints = np.tile(source_joints[root_ids], (1, 2, 1)).astype("<u2").reshape(-1, 4)
    original_weights = source_weights[root_ids].astype(np.float64)
    require((original_weights >= 0).all() and np.all(np.abs(original_weights.sum(axis=2)-1) < 1e-6), "Invalid source weights")
    normalized = original_weights / original_weights.sum(axis=2)[:, :, None]
    inherited_weights = np.tile(normalized, (1, 2, 1)).astype("<f4").reshape(-1, 4)
    require(fin_arrays["JOINTS_0"].tobytes() == inherited_joints.tobytes(), "Fin joint IDs not exact inherited slots")
    require(fin_arrays["WEIGHTS_0"].tobytes() == inherited_weights.tobytes(), "Fin weights not exact normalized inherited values")
    require((fin_arrays["JOINTS_0"] < len(s["skins"][0]["joints"])).all()
            and np.max(np.abs(fin_arrays["WEIGHTS_0"].astype(float).sum(axis=1)-1)) < 1e-6, "Invalid final skin support")
    colors = fin_arrays["COLOR_0"].reshape(count, 4, 3)
    require(colors[:, :2].tobytes() == colors[:, 2:].tobytes(), "Fin tips changed endpoint colors")
    directions = correspondence["fin_directions"]
    require(directions.shape == (count, 3) and np.max(np.abs(np.linalg.norm(directions, axis=1)-1)) < 1e-10, "Invalid authored direction")
    expected_tips = (root_points + directions[:, None, :]*.008).astype("<f4")
    require(points.reshape(count, 4, 3)[:, 2:].tobytes() == expected_tips.tobytes(), "Fin tips differ from recorded 8 mm direction")
    xyz = points.astype(np.float64)[indices]
    crosses = np.cross(xyz[:, 1]-xyz[:, 0], xyz[:, 2]-xyz[:, 0])
    cross_size = np.linalg.norm(crosses, axis=1)
    require((cross_size > 1e-12).all(), "Degenerate new fin triangle")
    face_normals = crosses/cross_size[:, None]
    new_normals = fin_arrays["NORMAL"].astype(np.float64)
    unit_error = float(np.max(np.abs(np.linalg.norm(new_normals, axis=1)-1)))
    normal_dots = np.einsum("fvc,fc->fv", new_normals[indices], face_normals)
    require(unit_error < 1e-5 and (normal_dots > .99999).all(), "New normals disagree with actual geometric winding")
    outward = correspondence["outward_normals"]
    require(outward.shape == (count, 3) and np.max(np.abs(np.linalg.norm(outward, axis=1)-1)) < 1e-10, "Invalid outward normals")
    normalized_source_normals = source_arrays["NORMAL"].astype(np.float64)
    normalized_source_normals /= np.linalg.norm(normalized_source_normals, axis=1)[:, None]
    expected_outward = normalized_source_normals[root_ids].sum(axis=1)
    expected_outward /= np.linalg.norm(expected_outward, axis=1)[:, None]
    require(np.max(np.abs(outward-expected_outward)) < 1e-12 and (np.einsum("nc,nc->n", directions, outward) > 0).all(), "Fin direction/outward source correspondence")
    vertex_count = sum(t["accessors"][p["attributes"]["POSITION"]]["count"] for p in t["meshes"][0]["primitives"])
    index_count = sum(t["accessors"][p["indices"]]["count"] for p in t["meshes"][0]["primitives"])
    require(vertex_count <= 100000 and index_count <= 300000 and index_count % 3 == 0, "Final mesh budget exceeded")
    active_mesh_nodes = [i for i, n in enumerate(t["nodes"]) if "mesh" in n]
    require(active_mesh_nodes == [19] and t["nodes"][19]["mesh"] == 0 and t["nodes"][19]["skin"] == 0, "Unexpected mesh instance/skin route")
    require(time.monotonic()-started < 28, "Internal audit deadline")
    for item in prepared["inputs"] + prepared["sources"]:
        require(sha(item["path"]) == item["sha256"], "Pinned input changed during audit")
    build = json.loads(Path(prepared["build_report"]).read_text())
    require(build["candidate_sha256"] == prepared["candidate_sha256"] and build["exported_vertices"] == vertex_count
            and build["exported_triangles"] == index_count//3 and build["fin_report"]["selected_fins"] == count, "Build receipt mismatch")
    color_defect = build["long_edge_color_interpolation"]
    require(color_defect["tip_count"] == count*2 and color_defect["tips_over_010"] == 194
            and color_defect["tips_over_005"] == 277 and color_defect["maximum_linear_component_error"] > .1,
            "Known color-defect receipt changed; do not silently accept")
    reference = json.loads(Path(prepared["allocation_reference"]).read_text())
    require(reference["source_sha256"] == prepared["source_sha256"]
            and reference["selection"]["source_primitive"] == 5
            and reference["settings"]["linear_luminance_breaks"] == [.1, .5], "Source allocation reference mismatch")
    luminance = colors[:, :2].astype(np.float64).mean(axis=1)@np.array([.2126, .7152, .0722])
    band_ids = np.searchsorted([.1, .5], luminance, side="right")
    band_counts = np.bincount(band_ids, minlength=3)
    area = reference["selection"]["eligible_surface_area_m2"]
    band_rows = []
    for i, old in enumerate(reference["luminance_bands"]):
        band_area = old["eligible_area_m2"]
        band_rows.append({"band": old["band"], "reference_eligible_area_m2": band_area,
                          "reference_area_fraction": old["area_fraction"], "new_fin_roots": int(band_counts[i]),
                          "new_root_fraction": float(band_counts[i]/count),
                          "new_roots_per_m2": float(band_counts[i]/band_area),
                          "new_relative_density": float((band_counts[i]/count)/old["area_fraction"]),
                          "old_root_fraction": old["old_root_fraction"], "old_roots_per_m2": old["old_roots_per_m2"]})
    low = np.asarray(reference["sampling"]["original_bounds_blender"]["min"])
    high = np.asarray(reference["sampling"]["original_bounds_blender"]["max"])
    native = root_points.astype(np.float64).mean(axis=1)[:, [0, 2, 1]]*np.array([1, -1, 1])
    cells = np.clip(np.floor((native-low)/(high-low)*4).astype(int), 0, 3)
    cell_ids = (cells[:, 0]*4+cells[:, 1])*4+cells[:, 2]
    grid_counts = np.bincount(cell_ids, minlength=64)
    grid_area, old_counts = np.zeros(64), np.zeros(64, dtype=int)
    for old in reference["grid_cells"]:
        x, y, z = old["cell"]
        idx = (x*4+y)*4+z
        grid_area[idx], old_counts[idx] = old["eligible_area_m2"], old["old_fin_roots"]
    grid_rows = [{"cell": [int(i//16), int(i//4%4), int(i%4)], "reference_eligible_area_m2": float(grid_area[i]),
                  "old_fin_roots": int(old_counts[i]), "new_fin_roots": int(grid_counts[i])}
                 for i in np.flatnonzero((grid_area > 0)|(grid_counts > 0))]
    allocation = {"method": "Actual stored linear COLOR_0 endpoint average, fixed luminance bands; source area/grid estimates reused unchanged from the pinned prior census.",
                  "reference_report_sha256": sha(prepared["allocation_reference"]), "source_area_m2": area,
                  "old_roots": int(old_counts.sum()), "new_roots": count, "bands": band_rows, "grid_cells": grid_rows,
                  "new_grid_distribution_total_variation_vs_area": float(np.abs(grid_counts/count-grid_area/area).sum()/2),
                  "old_grid_distribution_total_variation_vs_area": reference["old_allocation_summary"]["grid_distribution_total_variation_vs_area"],
                  "new_luminance_distribution_total_variation_vs_area": float(np.abs(band_counts/count-np.asarray([b["area_fraction"] for b in reference["luminance_bands"]])).sum()/2),
                  "new_low_high_roots_per_area_ratio": band_rows[0]["new_roots_per_m2"]/band_rows[2]["new_roots_per_m2"],
                  "old_low_high_roots_per_area_ratio": band_rows[0]["old_roots_per_m2"]/band_rows[2]["old_roots_per_m2"],
                  "new_texture_decodes": 0,
                  "limitations": ["These are actual endpoint-color/root allocations, not verified color at the visible tip centers; known interpolation errors remain.",
                                  "Source-area color/grid bins use the original three-point triangle quadrature; no anatomy labels.",
                                  "Root totals differ (3231 versus 4800); fraction/density-ratio comparisons do not prove visible coverage or appearance."]}
    result = {"schema_version": 1, "status": "measured_with_known_color_defect", "quality_approved": False,
              "attribute_checks_passed": True, "candidate_accepted": False, "passed": False,
              "source_sha256": prepared["source_sha256"], "candidate_sha256": prepared["candidate_sha256"],
              "inputs_unchanged": True, "elapsed_seconds": round(time.monotonic()-started, 6),
              "original_bin_prefix": {"bytes": source.declared, "sha256": original_hash.hexdigest(), "byte_exact": True},
              "json_allowlist": {"original_base_primitives_accessors_views_images_textures_materials_samplers_skins_animations_exact": True,
                                 "append_counts": appended_counts, "removed_mesh_suffix": [1, 2], "only_detached_nodes": [21, 22],
                                 "appended_mesh_0_primitives": 1, "buffer_byte_length_extended": True,
                                 "metadata_only_generator_change": {"before": s["asset"]["generator"], "after": t["asset"]["generator"]},
                                 "all_other_json_exact": True},
              "geometry": {"fins": count, "new_vertices": len(points), "new_triangles": len(indices),
                           "all_root_endpoint_bytes_exact": True, "all_edges_unique_and_whitelisted": True,
                           "all_source_face_corner_indices_correspond": True, "all_source_uv0_bytes_exact": True,
                           "actual_tip_positions_match_recorded_direction": True, "length_m": .008,
                           "physical_atlas_domain_m": .0208, "maximum_local_crop_uv_error": crop_error,
                           "source_edge_span_min_max_m": [float(spans.min()), float(spans.max())],
                           "minimum_triangle_area_m2": float(cross_size.min()/2), "normal_unit_error": unit_error,
                           "minimum_geometric_normal_dot": float(normal_dots.min()),
                           "minimum_outward_direction_dot": float(np.einsum("nc,nc->n", directions, outward).min())},
              "skin": {"source_skin": 0, "joints": len(s["skins"][0]["joints"]), "all_new_joint_ids_exact": True,
                       "all_new_weights_exact_float32_normalized_source": True, "source_and_final_four_slot_order_unchanged": True,
                       "maximum_new_weight_sum_error": float(np.max(np.abs(fin_arrays["WEIGHTS_0"].astype(float).sum(axis=1)-1)))},
              "atlas": {"embedded_sha256": hashlib.sha256(image).hexdigest(), "bytes": len(image), "size": [1024, 1024],
                        "embedded_equals_external_png": True, "header_only_not_full_decode": True, "source_root_uv_verified": True,
                        "root_colors_resampled": False, "endpoint_colors_repeated_at_tips_exact": True},
              "allocation": allocation,
              "known_color_defect": {"status": "unresolved_not_accepted", "measurement_source": "Hash-bound candidate build report; not independently texture-resampled in this audit",
                                     "evidence": color_defect, "candidate_approval": False,
                                     "reason": "194 of 9600 nominal visible-tip colors exceed 0.1 linear RGB component error; endpoint preservation does not establish whole-fin color fidelity."},
              "budgets": {"file_bytes": len(target.data), "file_cap": 64*1024**2, "active_mesh_vertices": vertex_count,
                          "active_mesh_triangles": index_count//3, "vertex_cap": 100000, "triangle_cap": 100000,
                          "source_decoded_scalar_values": source.decoded, "candidate_decoded_scalar_values": target.decoded},
              "limitations": ["No appearance, softness, dense-fur, silhouette, GPU alpha/lighting, or simulation approval.",
                              "No repeated whole-body or 102-pose validation; parent-owned motion receipts remain separate evidence.",
                              "Original embedded image bytes and exact source UV correspondence are verified; root color texture filtering is not independently rerun.",
                              "Normal checks use actual serialized geometry, finite unit normals, positive winding, and dot > .99999; no texture-normal or physics claim.",
                              "Source rig is authored data, not recovered anatomy; Weft normal-map/vertex-budget compatibility remains unsupported."],
              "limits": {"wall_seconds": 30, "rss_mb": 512, "output_mb": 16, "attempts": 1}, "native_calls": 0, "provider_calls": 0}
    write("audit.json", result)
    print(json.dumps({"status": result["status"], "fins": count, "vertices": vertex_count, "triangles": index_count//3}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write("failure.json", {"status": "rejected", "type": type(error).__name__, "message": str(error)})
        raise
