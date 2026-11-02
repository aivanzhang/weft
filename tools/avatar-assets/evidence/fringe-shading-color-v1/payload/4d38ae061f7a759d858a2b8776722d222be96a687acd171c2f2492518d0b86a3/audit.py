"""Read-only authored part ownership; no texture decode or appearance verdict."""
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np
from glb_read import GLB, require

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    require(Path(path).stat().st_size <= 2*1024**2, "JSON size bound")
    return json.loads(Path(path).read_text())


def write(name, value):
    with (HERE/name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def artifact_binding(manifest, relative, expected):
    rows = [row for row in manifest["artifacts"] if row["path"] == relative]
    require(len(rows) == 1 and rows[0]["sha256"] == expected, "Manifest binding: "+relative)


def descendants(parent, root):
    require(root in parent and len(parent) <= 64, "Missing part root or joint bound")
    result = {root}
    for _ in range(len(parent)):
        result |= {name for name, p in parent.items() if p in result}
    return result


def ownership(joints, weights, slots):
    selected = np.isin(joints, sorted(slots))
    positive = weights > 0  # Tiny stored influences still count as part membership.
    inside = (positive & selected).any(axis=1)
    outside = (positive & ~selected).any(axis=1)
    share = (weights.astype(np.float64)*selected).sum(axis=1)/weights.astype(np.float64).sum(axis=1)
    return {"exported_vertices": len(weights), "any_positive_part_weight": int(inside.sum()),
            "part_only_positive_weights": int((inside & ~outside).sum()),
            "mixed_part_and_outside_weights": int((inside & outside).sum()),
            "no_positive_part_weight": int((~inside).sum()),
            "maximum_normalized_part_weight": float(share.max(initial=0)),
            "minimum_positive_part_weight": float(weights[positive & selected].min()) if inside.any() else None}, inside


def main():
    started = time.monotonic()
    prepared = read(HERE/"prepared.json")
    pins = {row["path"]: row["sha256"] for row in prepared["inputs"]}
    cfg, finish, contract = (read(prepared[key]) for key in ("config", "finish_report", "rig_contract"))
    source_manifest = read(prepared["finish_manifest"])
    for key in ("config", "finish_report", "rig_contract", "original_model"):
        artifact_binding(source_manifest, Path(prepared[key]).name, pins[prepared[key]])
    source_build, area_build = read(prepared["normal_build"]), read(prepared["area_build"])
    require(source_build["candidate_sha256"] == pins[prepared["model"]]
            and source_build["source_sha256"] == area_build["candidate_sha256"]
            and area_build["source_glb_sha256"] == pins[prepared["original_model"]], "Derivative lineage mismatch")
    prior = read(prepared["prior_audit"])
    require(prior["attribute_checks_passed"] and prior["candidate_sha256"] == pins[prepared["model"]], "Prior attribute binding")
    artifact_binding(read(prepared["prior_manifest"]), "audit.json", pins[prepared["prior_audit"]])
    preview = read(prepared["preview_manifest"])
    require(preview["source_glb_sha256"] == pins[prepared["model"]], "Wrong preview model")
    for key in ("front_image", "pose_image"):
        relative = str(Path(prepared[key]).relative_to(Path(prepared["preview_manifest"]).parent))
        artifact_binding(preview, relative, pins[prepared[key]])

    glb = GLB(prepared["model"])
    doc = glb.doc
    require(len(doc["meshes"]) == 1 and len(doc["skins"]) == 1, "Unexpected mesh/skin inventory")
    mesh_nodes = [(i, n) for i, n in enumerate(doc["nodes"]) if "mesh" in n]
    require(len(mesh_nodes) == 1 and mesh_nodes[0][1]["mesh"] == 0 and mesh_nodes[0][1]["skin"] == 0,
            "Unexpected active mesh instance")
    nodes, joint_nodes = doc["nodes"], doc["skins"][0]["joints"]
    require(len(joint_nodes) <= 64 and len(set(joint_nodes)) == len(joint_nodes), "Joint bound/duplication")
    names = [nodes[i]["name"] for i in joint_nodes]
    require(len(set(names)) == len(names), "Duplicate exported joint names")
    parent_nodes = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            require(type(child) is int and 0 <= child < len(nodes) and child not in parent_nodes, "Invalid node graph")
            parent_nodes[child] = i
    actual_parent = {}
    for i, name in zip(joint_nodes, names):
        visited, node = {i}, parent_nodes.get(i)
        while node is not None and node not in joint_nodes:
            require(node not in visited, "Cyclic node graph")
            visited.add(node)
            node = parent_nodes.get(node)
        actual_parent[name] = nodes[node]["name"] if node is not None else None
    authored_parent = {bone["name"]: bone["parent"] for bone in cfg["authored_rig"]["bones"]}
    require(actual_parent == authored_parent and set(contract["bone_to_simulated_joint"]) == set(names), "Authored/exported rig mismatch")
    parts = {part["name"]: part for part in cfg["part_rigs"]}
    require(1 <= len(parts) <= 4 and set(cfg["groom_excluded_parts"]) <= set(parts), "Unknown excluded part")
    recorded_parts = {part["name"]: part for part in finish["part_rigs"]["parts"]}
    excluded = {}
    for name in cfg["groom_excluded_parts"]:
        part, recorded = parts[name], recorded_parts[name]
        bones = descendants(authored_parent, part["root_bone"])
        require(bones == descendants(actual_parent, part["root_bone"])
                and bones == set(recorded["bone_names"])
                and part["materials"] == recorded["materials"], "Part lineage mismatch")
        excluded[name] = {"root_bone": part["root_bone"], "bones": sorted(bones),
                          "joint_slots": [i for i, bone in enumerate(names) if bone in bones],
                          "source_materials": part["materials"],
                          "recorded_native_vertices": recorded["selected_vertices"],
                          "recorded_faces_by_material": recorded["faces_by_material"]}
    slots = {slot for part in excluded.values() for slot in part["joint_slots"]}
    part_materials = {name for part in excluded.values() for name in part["source_materials"]}
    undercoat = {row["material"]: row for row in finish["fur_undercoat"]["materials"]}
    authored_image_hashes = {row["texture_sha256"] for row in undercoat.values()}
    image_cache = {}

    def normal_image(material):
        texture = material.get("normalTexture")
        if texture is None:
            return None
        route = doc["textures"][texture["index"]]
        require("extensions" not in route, "Unknown normal texture route")
        index = route["source"]
        image = doc["images"][index]
        require("uri" not in image and "bufferView" in image, "External normal image")
        if index not in image_cache:
            image_cache[index] = hashlib.sha256(glb.view_bytes(image["bufferView"])).hexdigest()
        return {"image_index": index, "sha256": image_cache[index], "texcoord": texture.get("texCoord", 0)}

    primitives, arrays, rows = doc["meshes"][0]["primitives"], {}, []
    vertex_count = sum(doc["accessors"][p["attributes"]["POSITION"]]["count"] for p in primitives)
    index_count = sum(doc["accessors"][p["indices"]]["count"] for p in primitives)
    require(vertex_count <= 100000 and index_count <= 300000, "Geometry bound")
    for pidx, primitive in enumerate(primitives):
        require(primitive.get("mode", 4) == 4, "Nontriangle primitive")
        attrs = primitive["attributes"]
        joints = glb.array(attrs["JOINTS_0"], "VEC4", (5121, 5123))
        weights = glb.array(attrs["WEIGHTS_0"], "VEC4", (5126,))
        indices = glb.array(primitive["indices"], "SCALAR", (5121, 5123, 5125), 300000).reshape(-1, 3)
        require(len(joints) == len(weights) == doc["accessors"][attrs["POSITION"]]["count"]
                and (joints < len(names)).all() and (weights >= 0).all()
                and np.abs(weights.astype(float).sum(axis=1)-1).max() < 1e-6
                and indices.max() < len(weights), "Invalid skin/index storage")
        used = np.unique(indices)
        require(len(used) == len(weights), "Unreferenced exported vertices need separate accounting")
        material = doc["materials"][primitive["material"]]
        name, image = material["name"], normal_image(material)
        own, positive = ownership(joints, weights, slots)
        coat = undercoat.get(name)
        if coat is not None:
            require(image is not None and image["sha256"] == coat["texture_sha256"]
                    and image["texcoord"] == coat["normal_texcoord"]
                    and len(indices) == coat["selected_faces"], "Authored undercoat binding mismatch")
        arrays[pidx] = (joints, weights, indices)
        rows.append({"primitive": [0, pidx], "material_index": primitive["material"], "material": name,
                     "triangles": len(indices), "declared_excluded_part_material": name in part_materials,
                     "recorded_undercoat_source_material": coat["source_material"] if coat else None,
                     "normal_image": image,
                     "authored_undercoat_image": bool(image and image["sha256"] in authored_image_hashes),
                     "excluded_part_ownership": own,
                     "triangles_with_any_excluded_part_weight": int(positive[indices].any(axis=1).sum())})
    for part in excluded.values():
        observed = {row["material"]: row["triangles"] for row in rows if row["material"] in part["source_materials"]}
        require(observed == part["recorded_faces_by_material"], "Part material face inventory changed")

    design = read(prepared["derivative_prepared"])
    require(design["root_primitive"][0] == design["fin_primitive"][0] == 0, "Unexpected root mesh")
    require(design["correspondence"] == prepared["correspondence"], "Wrong root correspondence source")
    root_idx, fin_idx = design["root_primitive"][1], design["fin_primitive"][1]
    with zipfile.ZipFile(prepared["correspondence"]) as archive:
        require(len(archive.infolist()) <= 8 and sum(x.file_size for x in archive.infolist()) <= 4*1024**2,
                "Correspondence size bound")
    with np.load(prepared["correspondence"], allow_pickle=False) as archive:
        root_ids = archive["root_vertex_ids"].copy()
        saved_uv = archive["root_source_uv"].copy()
    rj, rw, _ = arrays[root_idx]
    fj, fw, _ = arrays[fin_idx]
    require(root_ids.dtype.kind in "iu" and root_ids.shape == (len(fw)//4, 2) and len(root_ids) <= 4800
            and root_ids.min() >= 0 and root_ids.max() < len(rw), "Invalid fin correspondence")
    expected_joints = np.tile(rj[root_ids], (1, 2, 1)).astype(fj.dtype).reshape(-1, 4)
    source_weights = rw[root_ids].astype(np.float64)
    expected_weights = np.tile(source_weights/source_weights.sum(axis=2, keepdims=True), (1, 2, 1)).astype(fw.dtype).reshape(-1, 4)
    require(fj.tobytes() == expected_joints.tobytes() and fw.tobytes() == expected_weights.tobytes(), "Fin skin inheritance mismatch")
    root_attrs, fin_attrs = primitives[root_idx]["attributes"], primitives[fin_idx]["attributes"]
    rp = glb.array(root_attrs["POSITION"], "VEC3", (5126,))
    fp = glb.array(fin_attrs["POSITION"], "VEC3", (5126,))
    uv = glb.array(root_attrs["TEXCOORD_0"], "VEC2", (5126,))
    require(fp.reshape(-1, 4, 3)[:, :2].tobytes() == rp[root_ids].tobytes()
            and saved_uv.tobytes() == uv[root_ids].tobytes(), "Fin source roots/UV correspondence changed")
    endpoint_ownership, root_positive = ownership(rj[root_ids].reshape(-1, 4), rw[root_ids].reshape(-1, 4), slots)
    fin_leak = int(root_positive.reshape(-1, 2).any(axis=1).sum())
    coat_rows = [row for row in rows if row["authored_undercoat_image"]]
    require({row["material"] for row in coat_rows} == set(undercoat), "Missing/unrecorded authored undercoat binding")
    coat_leak = sum(row["excluded_part_ownership"]["any_positive_part_weight"] for row in coat_rows)
    material_leak = [row["primitive"] for row in coat_rows if row["declared_excluded_part_material"]]
    result = {"schema_version": 1, "status": "ownership_leak_found" if fin_leak or coat_leak or material_leak else "no_exported_assignment_leak_found",
              "scope": "Declared authored part materials and actual positive exported bone weights; not recovered anatomy",
              "model_sha256": pins[prepared["model"]], "excluded_parts": excluded,
              "joint_names_in_skin_slot_order": names, "authored_exported_parent_graph_exact": True,
              "active_mesh_node": mesh_nodes[0][0], "active_meshes": 1, "exported_vertices": vertex_count,
              "exported_triangles": index_count//3, "primitive_inventory": rows,
              "fins": {"count": len(root_ids), "source_primitive": design["root_primitive"],
                       "fin_primitive": design["fin_primitive"], "root_endpoints": endpoint_ownership,
                       "root_count_with_any_positive_excluded_part_weight": fin_leak,
                       "all_fin_vertex_ownership": rows[fin_idx]["excluded_part_ownership"],
                       "root_position_uv_and_normalized_skin_inheritance_exact": True},
              "undercoat": {"primitives": [row["primitive"] for row in coat_rows],
                            "triangles": sum(row["triangles"] for row in coat_rows),
                            "vertices_with_any_positive_excluded_part_weight": coat_leak,
                            "excluded_part_material_primitives_using_authored_undercoat_images": material_leak,
                            "embedded_png_hashes_match_recorded_bakes": True},
              "membership_threshold": "Exact stored weight > 0; no positive influence discarded",
              "native_groom": {"report_excluded_part_vertices": finish["groom_excluded_parts"]["vertices"],
                               "report_names": finish["groom_excluded_parts"]["names"],
                               "current_export_has_only_audited_base_and_fin_primitives": True,
                               "native_particle_density_not_reopened_or_independently_evaluated": True},
              "limitations": ["No source-color sampling, texture decode, rendering, native execution or pose replay.",
                              "Exported UV/normal seam copies are counted separately; counts are not distinct anatomical points.",
                              "No assignment leak does not exclude projected overlap/intersection, source partition errors or source texture/normal/shading defects.",
                              "Visible light flecks remain unexplained by this ownership test; no appearance, Muse quality, physics or Weft acceptance."],
              "quality_approved": False, "decoded_scalar_values": glb.decoded,
              "elapsed_seconds": round(time.monotonic()-started, 6), "limits": prepared["limits"]}
    require(time.monotonic()-started < 14, "Internal deadline")
    write("ownership.json", result)
    print(json.dumps({"status": result["status"], "fins": len(root_ids), "fin_root_leaks": fin_leak,
                      "undercoat_vertex_leaks": coat_leak, "quality_approved": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write("failure.json", {"error_type": type(error).__name__, "error": str(error), "retry_authorized": False})
        raise
