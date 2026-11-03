"""Probe-guided local skin-weight repair; the source asset, geometry and rig stay unchanged.

Generated sources can blend far-apart bones (opposite feet, chest with a hand) into one
vertex. Those defects are localized while most authored weights are sound, so wholesale
filtering or geometric re-authoring replaces good data and creates new seams. This repair
is surgical instead: deterministic pose probes flag the exact vertices whose edges stretch
severely, and only those vertices take new weights from a distance-weighted consensus of
nearby clean vertices, restricted to bones chain-local to the consensus dominant. The loop
repeats until probes stop flagging or a bounded iteration limit. Only JOINTS_0/WEIGHTS_0
bytes are spliced into a new GLB. Acceptance belongs to the recorded-pose pipeline, not
this file alone.
"""

import math
from pathlib import Path
import struct
import time

import numpy as np

from avatar_assets.safety import require
from avatar_assets.three_d import _GLB, MAX_TRIANGLE_INDICES

PROFILE = "probe-guided-local-consensus-skin-repair-v1"
MAX_GLB_BYTES = 64 * 1024 * 1024
DEFAULT_REPAIR_CONFIG = {
    "chain_distance": 2, "consensus_neighbors": 16, "max_iterations": 8,
    "distance_epsilon": 1e-4, "coincident_round": 1e-6, "minimum_rest_edge": 0.001,
    "severe_ratio": 4.0, "probe_angle_rad": math.pi / 4,
}


def _config(overrides):
    config = dict(DEFAULT_REPAIR_CONFIG)
    if overrides is not None:
        require(isinstance(overrides, dict) and not set(overrides) - set(config),
                "Unknown skin repair configuration")
        config.update(overrides)
    require(type(config["chain_distance"]) is int and 1 <= config["chain_distance"] <= 3,
            "Chain distance must stay local (1-3 bones)")
    require(type(config["consensus_neighbors"]) is int and 4 <= config["consensus_neighbors"] <= 64,
            "Consensus neighborhood must be 4-64 clean groups")
    require(type(config["max_iterations"]) is int and 1 <= config["max_iterations"] <= 16,
            "Repair iterations must be 1-16")
    for key, low, high in (("distance_epsilon", 1e-9, 1e-2), ("coincident_round", 1e-9, 1e-3),
                           ("minimum_rest_edge", 1e-6, 1.0), ("severe_ratio", 2.0, 16.0),
                           ("probe_angle_rad", 0.1, math.pi / 2)):
        value = config[key]
        require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
                f"Skin repair bound out of range: {key}")
    return config


def _parents(doc):
    parents = {}
    for index, node in enumerate(doc.get("nodes", [])):
        for child in node.get("children", []):
            require(type(child) is int and child not in parents, "Node has multiple parents")
            parents[child] = index
    return parents


def joint_tree_distances(doc, joints):
    """Hop distance between skin joints through the node tree; non-joint ancestors count as hops."""
    parents = _parents(doc)
    paths = {}
    for joint in joints:
        chain, current = [], joint
        while current is not None:
            require(current not in chain, "Cyclic node ancestry")
            chain.append(current)
            current = parents.get(current)
        paths[joint] = chain
    count = len(joints)
    distances = np.zeros((count, count), dtype=np.int64)
    for a in range(count):
        for b in range(a + 1, count):
            pa, pb = paths[joints[a]], paths[joints[b]]
            ancestors = {node: depth for depth, node in enumerate(pa)}
            shared = next(((depth, ancestors[node]) for depth, node in enumerate(pb) if node in ancestors), None)
            require(shared is not None, "Skin joints do not share a tree root")
            distances[a, b] = distances[b, a] = shared[0] + shared[1]
    return distances


def _trs_matrix(node):
    require("matrix" not in node or not {"translation", "rotation", "scale"} & set(node),
            "Mixed node matrix/TRS is unsupported")
    if "matrix" in node:
        values = node["matrix"]
        require(isinstance(values, list) and len(values) == 16
                and all(type(x) in (int, float) and math.isfinite(x) for x in values), "Invalid node matrix")
        return np.array(values, dtype=np.float64).reshape(4, 4).T
    t = node.get("translation", [0.0, 0.0, 0.0])
    q = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    s = node.get("scale", [1.0, 1.0, 1.0])
    for value, width in ((t, 3), (q, 4), (s, 3)):
        require(isinstance(value, list) and len(value) == width
                and all(type(x) in (int, float) and math.isfinite(x) for x in value), "Invalid node TRS")
    x, y, z, w = q
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.array(s, dtype=np.float64)
    matrix[:3, 3] = t
    return matrix


def _world_matrices(doc, local_overrides=None, world_pins=None):
    parents = _parents(doc)
    nodes = doc.get("nodes", [])
    world = [None] * len(nodes)

    def resolve(index):
        if world[index] is not None:
            return world[index]
        pin = (world_pins or {}).get(index)
        if pin is not None:
            world[index] = pin
            return pin
        local = (local_overrides or {}).get(index)
        if local is None:
            local = _trs_matrix(nodes[index])
        parent = parents.get(index)
        world[index] = local if parent is None else resolve(parent) @ local
        return world[index]

    for index in range(len(nodes)):
        resolve(index)
    return world


def _skinned_primitives(glb):
    doc = glb.doc
    require(len(doc.get("skins", [])) == 1, "Skin repair expects exactly one skin")
    skin = doc["skins"][0]
    joints = skin.get("joints")
    require(isinstance(joints, list) and 1 <= len(joints) <= 64 and len(set(joints)) == len(joints),
            "Invalid skin joint set")
    rows = []
    for mesh_index, mesh in enumerate(doc.get("meshes", [])):
        for primitive_index, primitive in enumerate(mesh.get("primitives", [])):
            attributes = primitive.get("attributes", {})
            if "JOINTS_0" not in attributes:
                continue
            require({"POSITION", "JOINTS_0", "WEIGHTS_0"} <= set(attributes)
                    and not any(key.startswith(("JOINTS_", "WEIGHTS_")) and key not in ("JOINTS_0", "WEIGHTS_0")
                                for key in attributes), "Every skinned primitive requires one four-slot skin")
            positions = np.array(glb.accessor(attributes["POSITION"], "VEC3", {5126}), dtype=np.float64)
            joints_array = np.array(glb.accessor(attributes["JOINTS_0"], "VEC4", {5121, 5123}), dtype=np.int64)
            weights = np.array(glb.accessor(attributes["WEIGHTS_0"], "VEC4", {5126}), dtype=np.float64)
            indices = np.array(glb.accessor(primitive["indices"], "SCALAR", {5121, 5123, 5125},
                                            count_limit=MAX_TRIANGLE_INDICES), dtype=np.int64).ravel()
            require(len(positions) == len(joints_array) == len(weights) and len(indices) % 3 == 0
                    and indices.max() < len(positions), "Skinned primitive arrays disagree")
            require(joints_array.max() < len(joints) and np.all(weights >= -1e-6)
                    and np.abs(weights.sum(axis=1) - 1).max() <= 1e-3, "Invalid source skin weights")
            rows.append({"mesh": mesh_index, "primitive": primitive_index, "attributes": attributes,
                         "positions": positions, "joints": joints_array, "weights": np.clip(weights, 0, None),
                         "indices": indices})
    require(rows, "No skinned primitives found")
    return skin, rows


def _skin_palette(glb, skin, local_overrides=None, world_pins=None):
    joints = skin["joints"]
    inverse = np.array(glb.accessor(skin["inverseBindMatrices"], "MAT4", {5126}),
                       dtype=np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)
    require(len(inverse) == len(joints), "Inverse bind matrices disagree with skin joints")
    world = _world_matrices(glb.doc, local_overrides, world_pins)
    return np.stack([world[j] @ inverse[slot] for slot, j in enumerate(joints)]), world


def _lbs(row, palette):
    skinned = np.einsum("vk,vkab->vab", row["weights"], palette[row["joints"]])
    homogeneous = np.concatenate((row["positions"], np.ones((len(row["positions"]), 1))), axis=1)
    return np.einsum("vab,vb->va", skinned, homogeneous)[:, :3]


def _axis_rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _probe_palettes(glb, skin, probe_nodes, pinned, config):
    """Weight-independent joint palettes for every probe pose (node x axis x sign)."""
    _, rest_world = _skin_palette(glb, skin)
    palettes = []
    for node in probe_nodes:
        pins = {other: rest_world[other] for other in pinned if other != node}
        for axis in (0, 2):
            for sign in (1.0, -1.0):
                local = _trs_matrix(glb.doc["nodes"][node])
                probe = np.eye(4)
                probe[:3, :3] = _axis_rotation(axis, sign * config["probe_angle_rad"])
                palette, _ = _skin_palette(glb, skin, {node: local @ probe}, pins)
                palettes.append(palette)
    return palettes


def _grouping(row, config):
    """Coincident rest vertices share one weight row so seams never split."""
    keys = np.round(row["positions"] / config["coincident_round"]).astype(np.int64)
    _, group_of, group_sizes = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    palette_width = int(row["joints"].max()) + 1
    return group_of, group_sizes, palette_width


def _group_dense(row, group_of, group_sizes, palette):
    dense = np.zeros((len(group_sizes), palette), dtype=np.float64)
    np.add.at(dense, (np.repeat(group_of, 4), row["joints"].ravel()), row["weights"].ravel())
    dense /= group_sizes[:, None]
    sums = dense.sum(axis=1)
    require(np.all(sums > 0), "Source skin group lost all influence")
    return dense / sums[:, None]


def _eligible_edges(row, config):
    tri = row["indices"].reshape(-1, 3)
    edges = np.unique(np.sort(np.concatenate((tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]])), axis=1), axis=0)
    lengths = np.linalg.norm(row["positions"][edges[:, 0]] - row["positions"][edges[:, 1]], axis=1)
    keep = lengths >= config["minimum_rest_edge"]
    return edges[keep], lengths[keep]


def _severe_flags(row, group_of, dense, palettes, edges, lengths, config):
    """Groups incident to any severely stretched eligible edge across the probe poses."""
    homogeneous = np.concatenate((row["positions"], np.ones((len(row["positions"]), 1))), axis=1)
    flagged = np.zeros(len(dense), dtype=bool)
    severe_total = 0
    worst = 1.0
    for palette in palettes:
        group_matrices = np.einsum("gp,pab->gab", dense, palette)
        posed = np.einsum("vab,vb->va", group_matrices[group_of], homogeneous)[:, :3]
        ratios = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1) / lengths
        severe = ratios > config["severe_ratio"]
        severe_total += int(np.count_nonzero(severe))
        if len(ratios):
            worst = max(worst, float(ratios.max()))
        if severe.any():
            flagged[group_of[edges[severe].ravel()]] = True
    return flagged, severe_total, worst


def _heal(dense, points, defective, original_dominant, distances, config):
    """Replace defective rows with a chain-compatible, distance-weighted clean consensus.

    A defective row's dominant bone is usually right even when its long-range mix is wrong,
    so donors must be clean groups whose own dominant stays chain-local to that anchor;
    Euclidean proximity alone could import the opposite limb across a gap. A row without
    compatible donors keeps its own weights masked to the anchor's chain neighborhood.
    """
    clean = np.flatnonzero(~defective)
    bad = np.flatnonzero(defective)
    require(len(clean) >= 1, "Too few clean groups to support consensus repair")
    healed = dense.copy()
    clean_points = points[clean]
    clean_dominant = dense[clean].argmax(axis=1)
    for g in bad:
        anchor = original_dominant[g]
        allowed = distances[anchor] <= config["chain_distance"]
        compatible = np.flatnonzero(allowed[clean_dominant])
        consensus = None
        if len(compatible):
            offsets = clean_points[compatible] - points[g]
            squared = np.einsum("nd,nd->n", offsets, offsets)
            keep = min(config["consensus_neighbors"], len(compatible))
            nearest = np.argpartition(squared, keep - 1)[:keep]
            # Far donors would still import unrelated regions; keep the neighborhood local.
            nearest = nearest[squared[nearest] <= 16.0 * max(float(squared[nearest].min()),
                                                             config["distance_epsilon"] ** 2)]
            falloff = 1.0 / (squared[nearest] + config["distance_epsilon"] ** 2)
            consensus = (falloff @ dense[clean[compatible[nearest]]]) * allowed
        if consensus is None or consensus.sum() <= 0:
            consensus = dense[g] * allowed
        total = consensus.sum()
        if total <= 0:
            consensus = np.zeros_like(dense[g])
            consensus[anchor] = 1.0
            total = 1.0
        healed[g] = consensus / total
    return healed


def _dense_to_vertex_arrays(dense, group_of, joints_dtype):
    top_order = np.argsort(-dense, axis=1)[:, :4]
    top = np.take_along_axis(dense, top_order, axis=1)
    sums = top.sum(axis=1)
    require(np.all(sums > 0), "Repaired weights lost all influence")
    top /= sums[:, None]
    new_joints = top_order[group_of].astype(joints_dtype)
    new_weights = top[group_of]
    new_joints[new_weights <= 0] = 0
    new_weights[new_weights <= 0] = 0.0
    return new_joints, new_weights


def _accessor_layout(doc, index, binary_size):
    accessor = doc["accessors"][index]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    size = {5121: 1, 5123: 2, 5125: 4, 5126: 4}[accessor["componentType"]]
    view = doc["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", width * size)
    count = accessor["count"]
    require(start + (count - 1) * stride + width * size <= binary_size, "Accessor exceeds binary bytes")
    return start, stride, count, width, accessor["componentType"]


def _splice(binary, doc, accessor_index, values):
    start, stride, count, width, component = _accessor_layout(doc, accessor_index, len(binary))
    require(len(values) == count and values.shape[1] == width, "Splice shape differs from accessor")
    formats = {5121: "B", 5123: "H", 5125: "I", 5126: "f"}
    packer = struct.Struct("<" + formats[component] * width)
    for row_index in range(count):
        packer.pack_into(binary, start + row_index * stride, *values[row_index])


def repair_skinned_glb(source, target, *, mapped_nodes=None, config=None):
    """Write a repaired-GLB derivative beside an audit; the source file never changes.

    With mapped_nodes, probes articulate one mapped role at a time with the others pinned
    at rest, matching how the viewer drives a retargeted avatar; otherwise every skin joint
    is probed independently (stricter than any retargeted use).
    """
    config = _config(config)
    source, target = Path(source), Path(target)
    require(source.is_file() and not source.is_symlink() and source.stat().st_size <= MAX_GLB_BYTES,
            "Source GLB must be a bounded regular file")
    require(not target.exists() and target.parent.is_dir(), "Require a fresh repair target")
    started = time.monotonic()
    raw = source.read_bytes()
    glb = _GLB(source)
    skin, primitives = _skinned_primitives(glb)
    palette_width = len(skin["joints"])
    distances = joint_tree_distances(glb.doc, skin["joints"])
    if mapped_nodes is None:
        probe_nodes, pinned = list(skin["joints"]), ()
    else:
        require(isinstance(mapped_nodes, list) and 1 <= len(mapped_nodes) <= 16
                and all(type(n) is int and 0 <= n < len(glb.doc.get("nodes", [])) for n in mapped_nodes)
                and len(set(mapped_nodes)) == len(mapped_nodes), "Invalid mapped node list")
        probe_nodes, pinned = list(mapped_nodes), tuple(mapped_nodes)
    palettes = _probe_palettes(glb, skin, probe_nodes, pinned, config)

    json_size = struct.unpack_from("<I", raw, 12)[0]
    binary_offset = 12 + 8 + json_size + 8
    binary = bytearray(raw)
    audit_rows = []
    converged = True
    for row in primitives:
        group_of, group_sizes, _ = _grouping(row, config)
        dense = _group_dense(row, group_of, group_sizes, palette_width)
        points = np.zeros((len(group_sizes), 3), dtype=np.float64)
        np.add.at(points, group_of, row["positions"])
        points /= group_sizes[:, None]
        edges, lengths = _eligible_edges(row, config)
        original_dominant = dense.argmax(axis=1)
        defective = np.zeros(len(group_sizes), dtype=bool)
        iterations = 0
        baseline = final = None
        for iteration in range(config["max_iterations"]):
            flagged, severe, worst = _severe_flags(row, group_of, dense, palettes, edges, lengths, config)
            if baseline is None:
                baseline = (severe, worst)
            final = (severe, worst)
            fresh = flagged & ~defective
            if not fresh.any():
                break
            defective |= fresh
            dense = _heal(dense, points, defective, original_dominant, distances, config)
            iterations += 1
        else:
            converged = False
        if iterations:
            flagged, severe, worst = _severe_flags(row, group_of, dense, palettes, edges, lengths, config)
            final = (severe, worst)
            converged = converged and severe == 0
        new_joints, new_weights = _dense_to_vertex_arrays(dense, group_of, row["joints"].dtype)
        changed = int(np.count_nonzero(np.any((new_joints != row["joints"])
                                              | (np.abs(new_weights - row["weights"]) > 1e-7), axis=1)))
        view = memoryview(binary)[binary_offset:]
        _splice(view, glb.doc, row["attributes"]["JOINTS_0"], new_joints)
        _splice(view, glb.doc, row["attributes"]["WEIGHTS_0"], new_weights.astype("<f4"))
        audit_rows.append({"mesh": row["mesh"], "primitive": row["primitive"],
                           "vertices": len(row["positions"]), "coincident_groups": int(len(group_sizes)),
                           "repaired_groups": int(np.count_nonzero(defective)),
                           "changed_vertices": changed, "heal_iterations": iterations,
                           "probe_severe_edges_before": baseline[0], "probe_worst_ratio_before": baseline[1],
                           "probe_severe_edges_after": final[0], "probe_worst_ratio_after": final[1]})
    target.write_bytes(bytes(binary))
    repaired = _GLB(target)
    require(repaired.doc == glb.doc, "Repair must leave the GLB document unchanged")
    joint_names = [glb.doc["nodes"][j].get("name") for j in skin["joints"]]
    return {"schema_version": 1, "profile": PROFILE, "config": config,
            "source_sha256": glb.hash, "target_sha256": repaired.hash,
            "skin_joints": joint_names,
            "probe_mode": "mapped_roles_pinned" if mapped_nodes is not None else "independent_joints",
            "probe_nodes": probe_nodes, "primitives": audit_rows, "converged": converged,
            "seconds": time.monotonic() - started,
            "scope": ("Probe-flagged vertices take chain-local consensus weights from nearby clean "
                      "vertices; all other authored weights, texture, geometry, rig and physics are "
                      "unchanged. This is an authored derivative, not a visual-quality certification; "
                      "posed validation decides acceptance.")}


def _probe_metrics(path, probe_nodes, pinned, profile, config):
    glb = _GLB(Path(path))
    skin, primitives = _skinned_primitives(glb)
    palettes = _probe_palettes(glb, skin, probe_nodes, pinned, config)
    labels = []
    for node in probe_nodes:
        for axis in (0, 2):
            for sign in (1, -1):
                labels.append({"node": node, "name": glb.doc["nodes"][node].get("name"),
                               "axis": "xz"[axis == 2], "sign": sign})
    results = [dict(label, primitives=[]) for label in labels]
    for row in primitives:
        edges, lengths = _eligible_edges(row, config)
        homogeneous = np.concatenate((row["positions"], np.ones((len(row["positions"]), 1))), axis=1)
        for index, palette in enumerate(palettes):
            skinned = np.einsum("vk,vkab->vab", row["weights"], palette[row["joints"]])
            posed = np.einsum("vab,vb->va", skinned, homogeneous)[:, :3]
            ratios = np.linalg.norm(posed[edges[:, 0]] - posed[edges[:, 1]], axis=1) / lengths
            results[index]["primitives"].append({
                "mesh": row["mesh"], "primitive": row["primitive"], "eligible_edges": int(len(ratios)),
                "severe_edges": int(np.count_nonzero(ratios > config["severe_ratio"])),
                "max_edge_ratio": float(ratios.max()) if len(ratios) else 1.0,
            })
    worst = max((p["max_edge_ratio"] for row in results for p in row["primitives"]), default=1.0)
    severe = sum(p["severe_edges"] for row in results for p in row["primitives"])
    return {"schema_version": 1, "profile": profile, "config": config,
            "glb_sha256": glb.hash, "probes": results,
            "total_severe_edges": severe, "worst_edge_ratio": worst,
            "scope": ("Deterministic linear-blend probes against the asset's own rest pose; "
                      "not a renderer, recorded-pose or visual-quality result.")}


def limb_probe_metrics(path, *, config=None):
    """Pose each skin joint independently and measure linear-blend edge stretch."""
    config = _config(config)
    glb = _GLB(Path(path))
    skin, _ = _skinned_primitives(glb)
    return _probe_metrics(path, list(skin["joints"]), (), "chain-local-limb-probes-v1", config)


def mapped_role_probe_metrics(path, mapped_nodes, *, config=None):
    """Rotate one mapped role while the other mapped roles stay pinned at rest.

    This mirrors how the viewer drives a retargeted avatar: mapped nodes receive absolute
    body frames and everything else follows rest-relative, so only boundaries between
    mapped-role groups ever move relative to each other in the product.
    """
    config = _config(config)
    glb = _GLB(Path(path))
    require(isinstance(mapped_nodes, list) and 1 <= len(mapped_nodes) <= 16
            and all(type(n) is int and 0 <= n < len(glb.doc.get("nodes", [])) for n in mapped_nodes)
            and len(set(mapped_nodes)) == len(mapped_nodes), "Invalid mapped node list")
    return _probe_metrics(path, list(mapped_nodes), tuple(mapped_nodes), "mapped-role-probes-v1", config)
