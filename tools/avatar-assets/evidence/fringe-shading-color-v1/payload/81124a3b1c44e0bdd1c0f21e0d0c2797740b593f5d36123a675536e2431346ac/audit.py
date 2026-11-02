"""Independent derivative audit for declared source normals and opposed culled faces."""
import copy
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from glb_read import GLB, require

HERE = Path(__file__).resolve().parent
POLICY = "source-surface-normals-opposed-culled-faces-v1"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(name, data):
    with (HERE/name).open("x") as handle:
        json.dump(data, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main():
    start = time.monotonic()
    prepared = json.loads((HERE/"prepared.json").read_text())
    for row in prepared["inputs"]+prepared["sources"]:
        require(sha(row["path"]) == row["sha256"], "Pinned input changed")
    source, target = GLB(prepared["source"]), GLB(prepared["candidate"])
    old, new = source.doc, target.doc
    build = json.loads(Path(prepared["build_report"]).read_text())
    require(build["policy"] == POLICY and build["source_sha256"] == prepared["source_sha256"]
            and build["candidate_sha256"] == prepared["candidate_sha256"], "Authored policy/hash binding mismatch")
    require(source.bin_size-source.declared == 2, "Unexpected source padding")
    prefix_hash, payload_hash = hashlib.sha256(), hashlib.sha256()
    for begin in range(0, source.bin_size, 1024**2):
        end = min(begin+1024**2, source.bin_size)
        a = source.data[source.offset+begin:source.offset+end]
        b = target.data[target.offset+begin:target.offset+end]
        require(a == b, "Original padded BIN prefix changed")
        prefix_hash.update(a)
        if begin < source.declared:
            payload_hash.update(a[:min(end, source.declared)-begin])
    require(source.data[source.offset+source.declared:source.offset+source.bin_size] == b"\0\0", "Source padding is not zero")
    require(len(old["meshes"]) == 1 and len(old["meshes"][0]["primitives"]) == 8, "Unexpected source mesh layout")
    old_fin = old["meshes"][0]["primitives"][7]
    root_primitive = old["meshes"][0]["primitives"][5]
    require(set(old_fin["attributes"]) == {"POSITION", "NORMAL", "TEXCOORD_0", "COLOR_0", "JOINTS_0", "WEIGHTS_0"}
            and old_fin.get("mode", 4) == 4, "Unexpected source fin schema")
    with zipfile.ZipFile(prepared["correspondence"]) as archive:
        require(len(archive.infolist()) <= 8 and sum(i.file_size for i in archive.infolist()) <= 4*1024**2, "NPZ expansion bound")
    with np.load(prepared["correspondence"], allow_pickle=False) as arrays:
        root_ids = arrays["root_vertex_ids"]
    require(root_ids.shape == (4800, 2) and root_ids.dtype.kind in "iu", "Invalid pinned root correspondence")
    source_normals = source.array(root_primitive["attributes"]["NORMAL"], "VEC3", (5126,))
    require((root_ids >= 0).all() and (root_ids < len(source_normals)).all(), "Source normal index bound")
    source_lengths = np.linalg.norm(source_normals, axis=1)
    require(np.max(np.abs(source_lengths-1)) < .001 and (source_lengths > 0).all(), "Source normals not unit")
    # Float32 normalization is part of this versioned policy, not an arbitrary tolerance.
    normalized_source = source_normals/source_lengths[:, None]
    expected_normals = np.tile(normalized_source[root_ids], (1, 2, 1)).reshape(-1, 3).astype("<f4")
    count = len(expected_normals)
    require(count == 19200, "Normal count mismatch")
    old_indices = source.array(old_fin["indices"], "SCALAR", (5125,), 300000).reshape(-1, 3)
    expected_old = (np.arange(len(root_ids))[:, None, None]*4+np.array([[0, 1, 3], [0, 3, 2]])).reshape(-1, 3)
    require(np.array_equal(old_indices, expected_old), "Unexpected baseline fin connectivity")
    expected_indices = np.concatenate((old_indices, old_indices[:, ::-1])).astype("<u4")
    old_views, old_accessors = len(old["bufferViews"]), len(old["accessors"])
    normal_offset = (source.declared+3)//4*4
    index_offset = normal_offset+expected_normals.nbytes
    expected = copy.deepcopy(old)
    expected["bufferViews"] += [
        {"buffer": 0, "byteOffset": normal_offset, "byteLength": expected_normals.nbytes, "target": 34962},
        {"buffer": 0, "byteOffset": index_offset, "byteLength": expected_indices.nbytes, "target": 34963}]
    expected["accessors"] += [
        {"bufferView": old_views, "componentType": 5126, "count": count, "type": "VEC3",
         "min": expected_normals.min(axis=0).tolist(), "max": expected_normals.max(axis=0).tolist()},
        {"bufferView": old_views+1, "componentType": 5125, "count": expected_indices.size, "type": "SCALAR"}]
    old_material = old["materials"][old_fin["material"]]
    require(old_material.get("doubleSided") is True and old_material.get("alphaMode") == "MASK"
            and "normalTexture" not in old_material, "Unexpected frozen source material")
    material = copy.deepcopy(old_material)
    material.update(name="ExperimentalSourceSurfaceFinNormals", doubleSided=False)
    expected["materials"].append(material)
    expected_fin = expected["meshes"][0]["primitives"][7]
    expected_fin["attributes"]["NORMAL"] = old_accessors
    expected_fin["indices"] = old_accessors+1
    expected_fin["material"] = len(old["materials"])
    expected["buffers"][0]["byteLength"] = index_offset+expected_indices.nbytes
    expected["asset"]["generator"] = "Experimental source-surface fin shading; original geometry and motion unchanged"
    require(new == expected, "JSON differs from exact authored-policy allowlist")
    actual_normals = target.array(old_accessors, "VEC3", (5126,))
    actual_indices = target.array(old_accessors+1, "SCALAR", (5125,), 300000).reshape(-1, 3)
    require(actual_normals.tobytes() == expected_normals.tobytes(), "New normals differ from exact normalized source endpoints")
    require(actual_indices.tobytes() == expected_indices.tobytes(), "New indices are not original then exact reversed winding")
    require(target.declared == index_offset+expected_indices.nbytes and target.bin_size == target.declared,
            "Unexpected appended/trailing data")
    require(target.view_bytes(old_views) == expected_normals.tobytes()
            and target.view_bytes(old_views+1) == expected_indices.tobytes(), "Appended view payload mismatch")
    unit_error = float(np.max(np.abs(np.linalg.norm(actual_normals.astype(np.float64), axis=1)-1)))
    require(unit_error < 1e-6, "New shading normals are not finite unit vectors")
    protected = {}
    for name in ("POSITION", "TEXCOORD_0", "COLOR_0", "JOINTS_0", "WEIGHTS_0"):
        index = old_fin["attributes"][name]
        require(new["meshes"][0]["primitives"][7]["attributes"][name] == index, "Protected fin attribute rebound")
        view_id = old["accessors"][index]["bufferView"]
        original_bytes = source.view_bytes(view_id)
        require(original_bytes == target.view_bytes(view_id), "Protected fin bytes changed")
        protected[name] = {"accessor": index, "bufferView": view_id, "bytes": len(original_bytes),
                           "sha256": hashlib.sha256(original_bytes).hexdigest(), "exact": True}
    source_points = source.array(root_primitive["attributes"]["POSITION"], "VEC3", (5126,))
    fin_points = target.array(old_fin["attributes"]["POSITION"], "VEC3", (5126,)).reshape(4800, 4, 3)
    require(fin_points[:, :2].tobytes() == source_points[root_ids].tobytes(), "Unchanged fin anchors do not match source endpoints")
    vertex_count = sum(new["accessors"][p["attributes"]["POSITION"]]["count"] for m in new["meshes"] for p in m["primitives"])
    index_count = sum(new["accessors"][p["indices"]]["count"] for m in new["meshes"] for p in m["primitives"])
    require(vertex_count == 70567 and index_count == 81781*3 and vertex_count <= 100000 and index_count <= 300000,
            "Actual mesh exceeds unchanged counts/bounds")
    require(len(target.data) <= 64*1024**2 and len(new["skins"][0]["joints"]) <= 64, "File/joint budget")
    require(material["doubleSided"] is False and material["alphaMode"] == "MASK" and material["alphaCutoff"] == .18
            and material["pbrMetallicRoughness"]["metallicFactor"] == 0
            and material["pbrMetallicRoughness"]["roughnessFactor"] == .92, "Material contract mismatch")
    require(build["vertices"] == vertex_count and build["triangles"] == index_count//3 and build["fins"] == 4800,
            "Build receipt counts mismatch")
    old_build = json.loads(Path(prepared["baseline_build_report"]).read_text())
    require(old_build["long_edge_color_interpolation"]["tips_over_010"] == 194, "Known defect receipt changed")
    require(time.monotonic()-start < 14, "Internal audit deadline")
    for row in prepared["inputs"]+prepared["sources"]:
        require(sha(row["path"]) == row["sha256"], "Pinned evidence changed during audit")
    raw_repeated = np.tile(source_normals[root_ids], (1, 2, 1)).reshape(-1, 3)
    result = {"schema_version": 1, "status": "passed_authored_normal_derivative_audit", "attribute_checks_passed": True,
        "quality_approved": False, "candidate_accepted": False, "policy": POLICY,
        "source_sha256": prepared["source_sha256"], "candidate_sha256": prepared["candidate_sha256"],
        "inputs_unchanged": True, "elapsed_seconds": round(time.monotonic()-start, 6),
        "preservation": {"original_declared_payload_bytes": source.declared, "original_payload_sha256": payload_hash.hexdigest(),
            "original_padded_BIN_bytes": source.bin_size, "original_padded_BIN_sha256": prefix_hash.hexdigest(), "all_prefix_bytes_exact": True,
            "source_padding_bytes_preserved": 2, "all_original_accessors_views_materials_images_textures_samplers_exact": True,
            "original_base_primitives_skins_nodes_scenes_animations_exact": True, "protected_fin_attributes": protected,
            "animation_json_sha256": canonical_hash(old["animations"]), "skin_json_sha256": canonical_hash(old["skins"])},
        "exact_JSON_allowlist": {"appended_accessors": 2, "appended_views": 2, "appended_materials": 1,
            "changed_primitive": [0, 7], "changed_primitive_fields": ["attributes.NORMAL", "indices", "material"],
            "buffer_byte_length_extended": True, "generator_metadata_changed": True, "other_fields_exact": True},
        "normals": {"new_accessor": old_accessors, "vertices": count, "source_primitive": [0, 5],
            "formula": "Stored source NORMAL float32 divided by its float32 Euclidean length; root endpoint pairs repeat at corresponding fin tips; float32 serialization.",
            "byte_exact_against_formula": True, "maximum_float64_unit_error": unit_error,
            "changed_components_vs_unnormalized_stored_source": int((actual_normals != raw_repeated).sum()),
            "maximum_component_change_from_unit_normalization": float(np.abs(actual_normals-raw_repeated).max()),
            "geometric_face_normal_agreement_required": False,
            "reason": "This explicit policy intentionally supplies authored source-surface shading normals, not fin-plane geometric normals."},
        "topology": {"old_fin_triangles": len(old_indices), "new_fin_triangles": len(actual_indices),
            "ordering": "All original fin triangles, then each original triangle with indices reversed [2,1,0] in original row order",
            "indices_byte_exact": True, "no_new_vertices": True, "root_positions_exact": True,
            "positions_unchanged_not_triangle_count": True, "volumetric_fur": False},
        "material": {"new_index": len(old["materials"]), "definition": material,
            "original_materials_preserved": True, "only_clone_name_and_doubleSided_changed": True},
        "budgets": {"vertices": vertex_count, "triangles": index_count//3, "file_bytes": len(target.data),
            "joint_count": len(new["skins"][0]["joints"]), "vertex_cap": 100000, "triangle_cap": 100000, "file_cap": 64*1024**2},
        "known_color_defect_preserved": {"status": "unresolved", "evidence_scope": "Hash-bound baseline measurement; colors unchanged, no new texture decode",
            "reported_tips_over_0_1": 194, "reported_tip_count": 9600},
        "limitations": ["No native render, texture decode, 102-pose rerun, browser, provider, or AI-role call was made.",
            "Parent verified a separate native control for EEVEE culling; this audit does not independently repeat that result or claim Cycles compatibility.",
            "Opposed coincident triangles add no thickness or recovered hair; source normals are an authored shading policy.",
            "Structural preservation is not appearance, dense-fur, Muse parity, fresh-prompt success, physics or Weft consumer acceptance."],
        "limits": {"wall_seconds": 15, "rss_mb": 256, "output_mb": 8, "attempts": 1}}
    write("audit.json", result)
    print(json.dumps({"status": result["status"], "vertices": vertex_count, "triangles": index_count//3,
                      "quality_approved": False, "candidate_accepted": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write("failure.json", {"status": "rejected", "type": type(error).__name__, "message": str(error)})
        raise
