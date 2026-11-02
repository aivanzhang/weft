"""Read-only indexed-edge census; candidate indices are not authored geometry."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources"))
from avatar_assets.three_d import _GLB, _local_matrix, _world_matrices


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write(name, value):
    with (HERE / name).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def quantiles(values):
    if not len(values):
        return None
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "p99", "max"),
                    np.quantile(values, [0, .1, .25, .5, .75, .9, .99, 1]).tolist()))


def histogram(values):
    bounds = [0, .001, .003, .006, .01, .02, .04, float("inf")]
    return {label: int(n) for label, n in zip(
        ["0_to_1_mm", "1_to_3_mm", "3_to_6_mm", "6_to_10_mm", "10_to_20_mm", "20_to_40_mm", "40_mm_plus"],
        np.histogram(values, bounds)[0])}


def main():
    started = time.monotonic()
    prepared = json.loads((HERE / "prepared.json").read_text())
    for entry in prepared["inputs"] + prepared["sources"]:
        require(sha(entry["path"]) == entry["sha256"], "Pinned input changed: " + entry["path"])
    model = Path(prepared["model"])
    finish = json.loads(Path(prepared["finish_report"]).read_text())
    config = json.loads(Path(prepared["finish_config"]).read_text())
    glb = _GLB(model)
    doc = glb.doc
    nodes = doc["nodes"]
    parents = {}
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            require(type(child) is int and 0 <= child < len(nodes) and child not in parents, "Invalid node parent")
            parents[child] = i
    for i in range(len(nodes)):
        seen = set()
        while i in parents:
            require(i not in seen, "Node cycle")
            seen.add(i)
            i = parents[i]
    worlds = [np.array(m).reshape(4, 4, order="F") for m in _world_matrices(nodes, parents)]
    body_nodes = [i for i, n in enumerate(nodes) if "mesh" in n and not n.get("name", "").startswith(("PortableCoat_", "NativeGroom_"))]
    require(len(body_nodes) == 1, "Census requires one actual body mesh node")
    node_id = body_nodes[0]
    node = nodes[node_id]
    mesh_id = node["mesh"]
    mesh = doc["meshes"][mesh_id]
    require(len(mesh["primitives"]) <= 32, "Too many body primitives")
    skin_id = node["skin"]
    skin = doc["skins"][skin_id]
    joints = skin["joints"]
    require(1 <= len(joints) <= 64 and len(set(joints)) == len(joints), "Invalid skin joints")
    inverse_bind = np.asarray(glb.accessor(skin["inverseBindMatrices"], "MAT4", (5126,)), dtype=np.float64)
    require(len(inverse_bind) == len(joints), "Bind count mismatch")
    binds = inverse_bind.reshape(-1, 4, 4).transpose(0, 2, 1)
    bind_error = max(float(np.abs(worlds[j] @ binds[i] - worlds[node_id]).max()) for i, j in enumerate(joints))
    require(bind_error < 1e-4, "Rest skin binding is not mesh rest transform")
    target_names = sorted({name for row in finish["normal_continuity"]["corrections"] for name in row["materials"]})
    undercoat = finish["fur_undercoat"]["materials"]
    require(len({r["material"] for r in undercoat}) == len(undercoat), "Duplicate undercoat binding")
    selected_bindings = {r["material"]: r for r in undercoat if r["source_material"] not in target_names}
    excluded_bindings = [r for r in undercoat if r["source_material"] in target_names]
    require(selected_bindings, "No eligible undercoat material")
    material_names = [m.get("name", "") for m in doc["materials"]]
    require(all(material_names.count(n) == 1 for n in selected_bindings), "Ambiguous final material name")
    prims = []
    positions_all, normals_all, uv_all, joints_all, weights_all = [], [], [], [], []
    vertex_offset = 0
    for pi, primitive in enumerate(mesh["primitives"]):
        require(primitive.get("mode", 4) == 4 and not primitive.get("targets"), "Require ordinary indexed triangles")
        attrs = primitive["attributes"]
        arrays = {}
        for name, kind, components in (("POSITION", "VEC3", (5126,)), ("NORMAL", "VEC3", (5126,)),
                                      ("TEXCOORD_0", "VEC2", (5121, 5123, 5126)),
                                      ("JOINTS_0", "VEC4", (5121, 5123)), ("WEIGHTS_0", "VEC4", (5121, 5123, 5126))):
            arrays[name] = np.asarray(glb.accessor(attrs[name], kind, components), dtype=np.float64)
        n = len(arrays["POSITION"])
        require(all(len(a) == n for a in arrays.values()) and vertex_offset + n <= 100000, "Attribute counts exceed bounds")
        require(not doc["accessors"][attrs["JOINTS_0"]].get("normalized", False), "Normalized joint IDs")
        ji = arrays["JOINTS_0"].astype(np.int32)
        wt = arrays["WEIGHTS_0"]
        sums = wt.sum(axis=1)
        require(np.array_equal(ji, arrays["JOINTS_0"]) and (ji >= 0).all() and (ji < len(joints)).all(), "Invalid joint IDs")
        require((wt >= 0).all() and (wt <= 1).all() and np.max(np.abs(sums - 1)) < .001, "Invalid skin weights")
        normalized_weights = wt / sums[:, None]
        idx = np.asarray(glb.accessor(primitive["indices"], "SCALAR", (5121, 5123, 5125), count_limit=300000), dtype=np.int32).reshape(-1)
        require(len(idx) % 3 == 0 and (idx >= 0).all() and (idx < n).all(), "Invalid triangle indices")
        faces = idx.reshape(-1, 3)
        p = arrays["POSITION"] @ worlds[node_id][:3, :3].T + worlds[node_id][:3, 3]
        normal_lengths = np.linalg.norm(arrays["NORMAL"], axis=1)
        require(np.max(np.abs(normal_lengths - 1)) < .001, "Nonunit source normals")
        normals = arrays["NORMAL"] @ np.linalg.inv(worlds[node_id][:3, :3])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        directed = np.stack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]), axis=1).reshape(-1, 2)
        edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
        order = np.argsort(inverse, kind="stable")
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        first = order[starts]
        second = order[starts + np.minimum(counts - 1, 1)]
        adj = np.stack((first // 3, second // 3), axis=1)
        material = doc["materials"][primitive["material"]]
        name = material["name"]
        selected = name in selected_bindings
        if selected:
            require(len(faces) == selected_bindings[name]["selected_faces"], "Reported selected faces differ from final primitive")
        prims.append({"id": pi, "name": name, "primitive": primitive, "p": p, "n": normals, "uv": arrays["TEXCOORD_0"],
                      "joints": ji, "weights": normalized_weights, "faces": faces, "edges": edges, "counts": counts,
                      "directed": directed, "first": first, "second": second, "adj": adj, "offset": vertex_offset,
                      "selected": selected, "weight_sum_error": float(np.max(np.abs(sums - 1)))})
        positions_all.append(p); normals_all.append(normals); uv_all.append(arrays["TEXCOORD_0"])
        joints_all.append(ji); weights_all.append(normalized_weights)
        vertex_offset += n
        require(time.monotonic() - started < 25, "Internal census deadline")
    require(sum(len(p["faces"]) for p in prims) <= 100000, "Body triangle bound")
    points = np.concatenate(positions_all)
    normals = np.concatenate(normals_all)
    uvs = np.concatenate(uv_all)
    joint_rows = np.concatenate(joints_all)
    weight_rows = np.concatenate(weights_all)
    dense_weights = np.zeros((len(points), len(joints)))
    np.add.at(dense_weights, (np.arange(len(points))[:, None], joint_rows), weight_rows)
    extent = np.ptp(points, axis=0)
    scale = float(extent.max())
    require(scale > 1e-8, "Degenerate body bounds")
    length_epsilon = scale * 1e-9
    cross_epsilon = scale * scale * 1e-12
    _, position_ids = np.unique(points, axis=0, return_inverse=True)
    # Coincident positions classify splits only; no geometry is welded.
    edge_vertices = np.concatenate([p["edges"] + p["offset"] for p in prims])
    edge_primitive = np.concatenate([np.full(len(p["edges"]), p["id"], dtype=np.int32) for p in prims])
    position_pairs = position_ids[edge_vertices]
    flip = position_pairs[:, 0] > position_pairs[:, 1]
    edge_vertices[flip] = edge_vertices[flip, ::-1]
    keys, group, group_counts = np.unique(np.sort(position_pairs, axis=1), axis=0, return_inverse=True, return_counts=True)
    group_order = np.argsort(group, kind="stable")
    group_starts = np.r_[0, np.cumsum(group_counts)[:-1]]
    edge_offset = 0
    reports, candidates = [], {}
    for p in prims:
        faces, edges, counts, adj = (p[k] for k in ("faces", "edges", "counts", "adj"))
        stop = edge_offset + len(edges)
        groups_here = group[edge_offset:stop]
        if not p["selected"]:
            edge_offset = stop
            continue
        xyz = p["p"][faces]
        cross = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        good_face = twice_area > cross_epsilon
        face_normals = cross / np.maximum(twice_area[:, None], cross_epsilon)
        corner_dots = np.einsum("fvc,fc->fv", p["n"][faces], face_normals)
        length = np.linalg.norm(p["p"][edges[:, 1]] - p["p"][edges[:, 0]], axis=1)
        shared = counts == 2
        distinct = adj[:, 0] != adj[:, 1]
        reverse = np.all(p["directed"][p["first"]] == p["directed"][p["second"]][:, ::-1], axis=1)
        coherent_faces = np.einsum("ec,ec->e", face_normals[adj[:, 0]], face_normals[adj[:, 1]]) >= .5
        coherent_vertices = np.einsum("ec,ec->e", p["n"][edges[:, 0]], p["n"][edges[:, 1]]) >= .5
        corner_consistency = (corner_dots[adj] > .05).all(axis=(1, 2))
        nondegenerate = (length > length_epsilon) & good_face[adj].all(axis=1)
        unique_position_edge = group_counts[groups_here] == 1
        safe = shared & distinct & reverse & coherent_faces & coherent_vertices & corner_consistency & nondegenerate & unique_position_edge
        boundary_stats = {k: 0 for k in ("unmatched_exact_position_edge", "same_primitive_index_split", "other_primitive_coincidence", "coincident_group_more_than_two", "uv0_differs", "normal_differs", "skin_differs")}
        seam_witnesses = []
        for local_edge in np.flatnonzero(counts == 1):
            global_edge = edge_offset + int(local_edge)
            g = int(group[global_edge])
            members = group_order[group_starts[g]:group_starts[g] + group_counts[g]]
            others = members[members != global_edge]
            flags = {
                "unmatched_exact_position_edge": len(others) == 0,
                "same_primitive_index_split": bool(np.any(edge_primitive[others] == p["id"])),
                "other_primitive_coincidence": bool(np.any(edge_primitive[others] != p["id"])),
                "coincident_group_more_than_two": len(members) > 2,
                "uv0_differs": False, "normal_differs": False, "skin_differs": False}
            v = edge_vertices[global_edge]
            for other in others:
                w = edge_vertices[other]
                flags["uv0_differs"] |= bool(np.max(np.abs(uvs[v] - uvs[w])) > 1e-7)
                flags["normal_differs"] |= bool(np.max(np.abs(normals[v] - normals[w])) > 1e-6)
                flags["skin_differs"] |= bool(np.max(np.abs(dense_weights[v] - dense_weights[w])) > 1e-6)
            for key, flag in flags.items():
                boundary_stats[key] += int(flag)
            if len(seam_witnesses) < 8 and len(others):
                seam_witnesses.append({"edge_vertex_indices": edges[local_edge].tolist(), "adjacent_face": int(adj[local_edge, 0]),
                                       "coincident_primitive_ids": sorted(set(edge_primitive[others].tolist())), "flags": flags})
        safe_ids = np.flatnonzero(safe)
        candidates[f"primitive_{p['id']}_edges"] = edges[safe_ids].astype("<i4")
        candidates[f"primitive_{p['id']}_faces"] = adj[safe_ids].astype("<i4")
        witnesses = [{"edge_vertex_indices": edges[e].tolist(), "face_indices": adj[e].tolist(),
                      "world_endpoints_m": p["p"][edges[e]].tolist(), "length_m": float(length[e]),
                      "uv0": p["uv"][edges[e]].tolist(), "joint_indices": p["joints"][edges[e]].tolist(),
                      "normalized_weights": p["weights"][edges[e]].tolist(),
                      "minimum_corner_face_normal_dot": float(corner_dots[adj[e]].min())} for e in safe_ids[:8]]
        reports.append({"primitive": p["id"], "material_index": p["primitive"]["material"], "material": p["name"],
                        "source_binding": selected_bindings[p["name"]], "vertices": len(p["p"]), "triangles": len(faces),
                        "unique_indexed_edges": len(edges), "shared_two_face_edges": int(shared.sum()),
                        "boundary_indexed_edges": int((counts == 1).sum()), "nonmanifold_indexed_edges": int((counts > 2).sum()),
                        "self_edges": int((edges[:, 0] == edges[:, 1]).sum()), "zero_or_tiny_edges": int((length <= length_epsilon).sum()),
                        "degenerate_faces": int((~good_face).sum()), "shared_repeated_same_face": int((shared & ~distinct).sum()),
                        "shared_winding_conflicts": int((shared & ~reverse).sum()),
                        "opposed_corner_normals": int(((corner_dots < 0) & good_face[:, None]).sum()),
                        "faces_with_opposed_corner_normal": int(((corner_dots < 0).any(axis=1) & good_face).sum()),
                        "faces_with_corner_normal_dot_at_most_005": int(((corner_dots <= .05).any(axis=1) & good_face).sum()),
                        "shared_face_crease_over_60_degrees": int((shared & ~coherent_faces & nondegenerate).sum()),
                        "shared_endpoint_normal_angle_over_60_degrees": int((shared & ~coherent_vertices).sum()),
                        "shared_exact_position_edge_duplicates": int((shared & ~unique_position_edge).sum()),
                        "boundary_coincidence_categories_overlapping": boundary_stats, "boundary_witnesses": seam_witnesses,
                        "edge_length_m": quantiles(length), "shared_edge_length_m": quantiles(length[shared]),
                        "conservative_candidate_edges": len(safe_ids), "candidate_edge_length_m": quantiles(length[safe]),
                        "candidate_length_histogram": histogram(length[safe]), "candidate_witnesses": witnesses})
        edge_offset = stop
        require(time.monotonic() - started < 25, "Internal census deadline")
    require(reports, "No selected primitive found")
    candidate_path = HERE / "candidate-indices.npz"
    require(not candidate_path.exists(), "Candidate output already exists")
    np.savez_compressed(candidate_path, **candidates)
    schemas = []
    for p in prims:
        primitive = p["primitive"]
        schemas.append({"primitive": p["id"], "material_index": primitive["material"], "material": p["name"], "selected": p["selected"],
                        "vertices": len(p["p"]), "triangles": len(p["faces"]), "max_weight_sum_error": p["weight_sum_error"],
                        "nonzero_influences_per_vertex": [int((p["weights"] > 0).sum(axis=1).min()), int((p["weights"] > 0).sum(axis=1).max())],
                        "uv0_range": {"min": p["uv"].min(axis=0).tolist(), "max": p["uv"].max(axis=0).tolist()},
                        "accessors": {name: {"index": index, **doc["accessors"][index]} for name, index in {**primitive["attributes"], "INDICES": primitive["indices"]}.items()},
                        "material_definition": doc["materials"][primitive["material"]]})
    for entry in prepared["inputs"] + prepared["sources"]:
        require(sha(entry["path"]) == entry["sha256"], "Pinned input changed during census")
    result = {"schema_version": 1, "status": "measured_source_census", "quality_approved": False,
              "model_sha256": glb.hash, "inputs_unchanged": True, "elapsed_seconds": round(time.monotonic() - started, 6),
              "body_node": node_id, "mesh": mesh_id, "mesh_name": mesh.get("name"), "body_exported_vertices": len(points),
              "body_triangles": sum(len(p["faces"]) for p in prims), "decoded_scalar_values": glb.value_count,
              "world_bounds": {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()},
              "coordinate_space": "glTF right-handed +Y-up world rest positions, authored meters; Blender conversion [x,-z,y]",
              "authored_height_meters": config["authored_rig"]["height_meters"],
              "selection": {"source": "finish-report.fur_undercoat.materials", "eligible_materials": list(selected_bindings),
                            "normal_continuity_target_source_names_excluded": target_names,
                            "excluded_undercoat_bindings": excluded_bindings,
                            "rule": "Exclude recorded normal_continuity target materials by source name, not UV set or coordinates; all other non-undercoat materials remain unselected."},
              "candidate_filter": {"exactly_two_distinct_faces": True, "opposite_directed_edges": True,
                                   "nondegenerate_edge_and_faces": True, "minimum_adjacent_face_normal_dot": .5,
                                   "minimum_endpoint_normal_dot": .5, "minimum_corner_face_normal_dot_exclusive": .05,
                                   "no_other_exact_position_edge": True, "length_epsilon_m": length_epsilon,
                                   "twice_face_area_epsilon_m2": cross_epsilon,
                                   "scope": "Conservative read-only candidate screen, not an established quality threshold or geometry guarantee."},
              "primitives": reports, "source_primitive_schema": schemas,
              "skin": {"index": skin_id, "name": skin.get("name"), "joint_count": len(joints),
                       "joint_node_indices_in_slot_order": joints, "joint_names_in_slot_order": [nodes[j].get("name") for j in joints],
                       "inverse_bind_accessor": {"index": skin["inverseBindMatrices"], **doc["accessors"][skin["inverseBindMatrices"]]},
                       "inverse_bind_float64_sha256": hashlib.sha256(inverse_bind.astype("<f8").tobytes()).hexdigest(),
                       "max_rest_world_joint_times_inverse_bind_minus_mesh_world": bind_error},
              "nodes": [{"index": i, "name": n.get("name"), "parent": parents.get(i), "mesh": n.get("mesh"), "skin": n.get("skin"),
                         "local_matrix_column_major": list(_local_matrix(n)), "world_matrix_column_major": worlds[i].reshape(-1, order="F").tolist()} for i, n in enumerate(nodes)],
              "candidate_index_artifact": {"path": candidate_path.name, "sha256": sha(candidate_path), "bytes": candidate_path.stat().st_size,
                                           "schema": "primitive_N_edges: int32 [K,2] primitive-local exported vertex indices; primitive_N_faces: int32 [K,2] triangle ordinals; rows correspond; no positions or geometry authored."},
              "limits": {"wall_seconds": 30, "internal_seconds": 25, "rss_mb": 512, "output_mb": 16, "vertices": 100000, "triangles": 100000, "decoded_values": 2000000},
              "limitations": ["No welding, geometry generation, image edit, render, provider call, or production edit.",
                              "Exact coincident positions indicate possible seams/splits, not proven shared physical topology; they never become candidate shared edges.",
                              "Candidate counts do not prove silhouette visibility, groom direction, density, animation deformation, or appearance quality.",
                              "Indices are exported primitive-local indices, not anatomical or original source-vertex identities.",
                              "Normal/skin coherence is inspected only at rest; no self-intersection, motion or physical-validity claim."]}
    write("census.json", result)
    print(json.dumps({"status": result["status"], "candidates": sum(r["conservative_candidate_edges"] for r in reports), "elapsed_seconds": result["elapsed_seconds"]}))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write("failure.json", {"status": "rejected", "type": type(error).__name__, "message": str(error)})
        raise
