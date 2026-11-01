"""Bounded CPU skin measurements from actual GLB animation; no rendering or simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import numpy as np

from avatar_assets import three_d
from avatar_assets.action_intent import evaluate_action_evidence, validate_action_bindings, validate_state_intents
from avatar_assets.safety import local_file, require, sha256


MAX_VERTICES = 100000
MAX_JOINTS = 64
MAX_PROBES = 1024
PROBES_PER_CONTROL = 16
MAX_STATES = 8
MAX_FRAMES_PER_STATE = 97
MAX_TOTAL_FRAMES = 776
MAX_FULL_FRAMES = 8
MAX_SECONDS = 50
MAX_REPORT_BYTES = 4*1024*1024
WEIGHT_FLOOR = .05
RESERVED_SURFACES = ("PortableCoat_", "NativeGroom_")
VERSION = "weighted-glb-action-audit-v2"


def _deadline(start):
    require(time.monotonic()-start <= MAX_SECONDS, "Action audit exceeded its CPU wall-time bound")


def _matrix(flat):
    return np.asarray(flat, dtype=np.float64).reshape(4, 4, order="F")


def _points(matrix, points):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _hierarchy(glb):
    doc, nodes = glb.doc, glb.doc.get("nodes", [])
    require(nodes and doc.get("skins") and doc.get("meshes"), "Action audit requires a real skinned mesh")
    parents, names = {}, {}
    for index, node in enumerate(nodes):
        three_d._local_matrix(node)
        name = node.get("name")
        if name is not None:
            require(isinstance(name, str) and name and name not in names, "GLB nodes require unique names when named")
            names[name] = index
        children = node.get("children", [])
        require(isinstance(children, list) and len(children) <= three_d.MAX_NODES, "Invalid child list")
        for child in children:
            three_d._index(child, len(nodes), "child")
            require(child not in parents, "Node has multiple parents")
            parents[child] = index
    for index in range(len(nodes)):
        seen, current = set(), index
        while current in parents:
            require(current not in seen, "Cyclic node hierarchy")
            seen.add(current)
            current = parents[current]
    scenes = doc.get("scenes", [])
    scene = scenes[three_d._index(doc.get("scene", 0), len(scenes), "scene")]
    roots = scene.get("nodes", [])
    require(isinstance(roots, list) and roots and len(roots) <= three_d.MAX_NODES, "Invalid default scene roots")
    reachable, pending = set(), list(roots)
    for root in roots:
        three_d._index(root, len(nodes), "scene root")
        require(root not in parents, "Scene root has a parent")
    while pending:
        current = three_d._index(pending.pop(), len(nodes), "scene node")
        if current not in reachable:
            reachable.add(current)
            pending.extend(nodes[current].get("children", []))
    joint_indices = set()
    for skin in doc["skins"]:
        joints = skin.get("joints")
        require(isinstance(joints, list) and 1 <= len(joints) <= MAX_JOINTS
                and all(type(joint) is int for joint in joints) and len(set(joints)) == len(joints), "Invalid skin joints")
        for joint in joints:
            three_d._index(joint, len(nodes), "skin joint")
            require(joint in reachable and isinstance(nodes[joint].get("name"), str), "Skin joint is unnamed or outside the scene")
        joint_indices.update(joints)
    require(len(joint_indices) <= MAX_JOINTS, "Aggregate joint count exceeds 64")
    bone_parents = {}
    for joint in sorted(joint_indices):
        parent = parents.get(joint)
        while parent is not None and parent not in joint_indices:
            parent = parents.get(parent)
        bone_parents[nodes[joint]["name"]] = nodes[parent]["name"] if parent is not None else None
    return nodes, parents, names, reachable, joint_indices, bone_parents


def actual_bone_parents(model_path):
    """Nearest actual skin-joint parents; intermediary non-joint transforms stay in FK."""
    return _hierarchy(three_d._GLB(model_path))[-1]


def _load_geometry(glb, nodes, reachable, start):
    doc = glb.doc
    palettes, inverse, skin_offsets = [], [], []
    for skin in doc["skins"]:
        skin_offsets.append(len(palettes))
        joints = skin["joints"]
        matrices = glb.accessor(skin.get("inverseBindMatrices"), "MAT4", (5126,))
        require(len(matrices) == len(joints), "Inverse-bind/joint count mismatch")
        for value in matrices:
            matrix = _matrix(value)
            require(np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6, rtol=0)
                    and abs(np.linalg.det(matrix[:3, :3])) > 1e-12, "Invalid inverse-bind matrix")
            inverse.append(matrix)
        palettes.extend(joints)
    positions, weights, indices, excluded, primitive_ids = [], [], [], [], []
    total, primitives = 0, 0
    for node_index, node in enumerate(nodes):
        if "mesh" not in node:
            require("skin" not in node, "Skin without mesh")
            continue
        require(node_index in reachable, "Mesh is outside the default scene")
        mesh = doc["meshes"][three_d._index(node["mesh"], len(doc["meshes"]), "mesh")]
        entries = mesh.get("primitives", [])
        require(isinstance(entries, list) and 1 <= len(entries) <= 32, "Invalid mesh primitives")
        for primitive_index, primitive in enumerate(entries):
            _deadline(start)
            primitives += 1
            require(primitives <= 128 and primitive.get("mode", 4) == 4 and not primitive.get("targets"),
                    "Only bounded triangle primitives without morph targets are supported")
            attrs = primitive.get("attributes", {})
            require(isinstance(attrs, dict), "Invalid primitive attributes")
            accessor = doc["accessors"][three_d._index(attrs.get("POSITION"), len(doc["accessors"]), "position")]
            count = accessor.get("count")
            require(type(count) is int and 0 < count <= MAX_VERTICES, "Invalid vertex count")
            total += count
            require(total <= MAX_VERTICES, "Aggregate vertex count exceeds 100000")
            material = doc.get("materials", [])[three_d._index(primitive.get("material"), len(doc.get("materials", [])), "material")]
            labels = [node.get("name", ""), mesh.get("name", ""), material.get("name", "")]
            if any(isinstance(label, str) and label.startswith(RESERVED_SURFACES) for label in labels):
                excluded.append({"node": node_index, "primitive": primitive_index, "vertices": count, "names": labels})
                continue
            require("skin" in node, "Anatomical surface has no skin")
            skin_index = three_d._index(node["skin"], len(doc["skins"]), "skin")
            require(not any(key.startswith(("JOINTS_", "WEIGHTS_")) and key not in ("JOINTS_0", "WEIGHTS_0") for key in attrs),
                    "More than four skin influences is unsupported")
            xyz = np.asarray(glb.accessor(attrs["POSITION"], "VEC3", (5126,)), dtype=np.float64)
            joints = glb.accessor(attrs.get("JOINTS_0"), "VEC4", (5121, 5123))
            require(all(all(type(value) is int and 0 <= value < len(doc["skins"][skin_index]["joints"]) for value in row) for row in joints),
                    "Skin joint index outside skin")
            joint_array = np.asarray(joints, dtype=np.int32)
            weight_array = np.asarray(glb.accessor(attrs.get("WEIGHTS_0"), "VEC4", (5121, 5123, 5126)), dtype=np.float64)
            weight_accessor = doc["accessors"][attrs["WEIGHTS_0"]]
            require(weight_accessor["componentType"] == 5126 or weight_accessor.get("normalized") is True, "Integer skin weights must be normalized")
            require(len(xyz) == len(joint_array) == len(weight_array) and np.all(weight_array >= 0)
                    and np.all(np.abs(weight_array.sum(axis=1)-1) <= .001), "Invalid skin weight coverage")
            sum_error = float(np.abs(weight_array.sum(axis=1)-1).max())
            # Normalize tolerated encoding error so rigid travel cannot leak into articulation.
            weight_array /= weight_array.sum(axis=1, keepdims=True)
            if "indices" in primitive:
                triangles = glb.accessor(primitive["indices"], "SCALAR", (5121, 5123, 5125), count_limit=three_d.MAX_TRIANGLE_INDICES)
                require(len(triangles) % 3 == 0 and all(type(row[0]) is int and 0 <= row[0] < count for row in triangles), "Invalid triangles")
            else:
                require(count % 3 == 0, "Nonindexed triangles need three vertices per face")
            anatomical_start = sum(len(value) for value in positions)
            positions.append(xyz)
            weights.append(weight_array)
            indices.append(joint_array + skin_offsets[skin_index])
            primitive_ids.append({"node": node_index, "primitive": primitive_index, "vertices": count,
                                  "anatomical_start_index": anatomical_start, "max_input_weight_sum_error": sum_error})
    require(positions, "No anatomical skinned surface remains after coat exclusion")
    return (np.concatenate(positions), np.concatenate(weights), np.concatenate(indices),
            np.asarray(palettes, dtype=np.int32), np.asarray(inverse), excluded, primitive_ids, total)


def _skin(world, points, weights, joints, palette, inverse):
    transforms = world[palette] @ inverse
    result = np.zeros_like(points)
    for column in range(4):
        matrices = transforms[joints[:, column]]
        result += weights[:, column, None] * (np.einsum("nij,nj->ni", matrices[:, :3, :3], points) + matrices[:, :3, 3])
    require(np.isfinite(result).all() and np.abs(result).max(initial=0) <= 100, "Skinned positions exceed finite world bounds")
    return result


def _select_probes(ids, rest, coverage, limit=PROBES_PER_CONTROL):
    if len(ids) <= limit:
        return ids
    selected = [int(ids[np.argmax(coverage[ids])])]
    cloud = rest[ids]
    for axis in range(3):
        selected.extend((int(ids[np.argmin(cloud[:, axis])]), int(ids[np.argmax(cloud[:, axis])])))
    selected = list(dict.fromkeys(selected))
    # Farthest-point sampling adds geometric coverage, not just high-weight neighbors.
    distance = np.min(np.sum((cloud[:, None, :]-rest[selected][None, :, :])**2, axis=2), axis=1)
    while len(selected) < limit:
        distance[np.isin(ids, selected)] = -1
        chosen = int(ids[int(np.argmax(distance))])
        selected.append(chosen)
        distance = np.minimum(distance, np.sum((cloud-rest[chosen])**2, axis=1))
    return np.asarray(selected, dtype=np.int32)


def _reversals(values, tolerance):
    anchor, extreme, direction, count = float(values[0]), float(values[0]), 0, 0
    for value in values[1:]:
        value = float(value)
        if not direction:
            if abs(value-anchor) >= tolerance:
                direction, extreme = (1 if value > anchor else -1), value
        elif direction*(value-extreme) >= 0:
            extreme = value
        elif direction*(extreme-value) >= tolerance:
            direction, extreme, count = -direction, value, count+1
    return count


def _sample_times(tracks):
    start = min(track[3][0] for track in tracks)
    end = max(track[3][-1] for track in tracks)
    require(abs(start) <= 1e-5 and 0 < end <= 4.00001, "Audit actions must start at zero and last at most four seconds")
    times = [frame/24 for frame in range(int(math.floor(end*24+1e-5))+1)]
    if abs(times[-1]-end) > 1e-5:
        times.append(end)
    else:
        times[-1] = end
    require(len(times) <= MAX_FRAMES_PER_STATE, "Action frame count exceeds 97")
    return times


def audit_glb(model_path, states, action_bindings, height_meters):
    """Measure probes at 24fps and exact moving-vertex counts at informative frames."""
    start = time.monotonic()
    try:
        return _audit_glb(model_path, states, action_bindings, height_meters, start)
    except (KeyError, TypeError, IndexError, struct.error, RecursionError, OverflowError, np.linalg.LinAlgError) as error:
        raise ValueError("Malformed or unsupported action audit input: " + type(error).__name__) from error


def _audit_glb(model_path, states, action_bindings, height_meters, start):
    require(type(height_meters) in (int, float) and math.isfinite(height_meters) and .05 <= height_meters <= 10,
            "Invalid authored height")
    require(isinstance(states, list) and 1 <= len(states) <= MAX_STATES, "Action audit requires one to eight states")
    exact = [state.get("user_request", "") for state in states]
    validate_state_intents(states, exact if any(exact) else [], [state["id"] for state in states])
    require(all(state.get("requirements") for state in states), "Action audit requires structured state intent")
    glb = three_d._GLB(model_path)
    nodes, parents, names, reachable, joint_nodes, bone_parents = _hierarchy(glb)
    validate_action_bindings(action_bindings, states, bone_parents)
    points, weights, joint_slots, palette, inverse, excluded, primitives, total = _load_geometry(glb, nodes, reachable, start)
    rest_world = np.asarray([_matrix(matrix) for matrix in three_d._world_matrices(nodes, parents)])
    rest = _skin(rest_world, points, weights, joint_slots, palette, inverse)
    influence_nodes = palette[joint_slots]
    target_names = sorted({bone for action in action_bindings for role in action["roles"] for bone in role["bones"]})
    stable_roles = {(state["id"], requirement["role"]) for state in states for requirement in state["requirements"]
                    if requirement["kind"] == "stability"}
    # Reserve half the probe budget for up to32 direct-surface stability roles.
    control_probe_limit = 8 if stable_roles else PROBES_PER_CONTROL
    controls, selected = {}, []
    for name in target_names:
        _deadline(start)
        descendants = []
        for joint in joint_nodes:
            current = joint
            while current in parents and current != names[name]:
                current = parents[current]
            if current == names[name]:
                descendants.append(joint)
        coverage = np.sum(weights*np.isin(influence_nodes, descendants), axis=1)
        ids = np.flatnonzero(coverage >= WEIGHT_FLOOR)
        require(len(ids) > 0, "Action target has no anatomical skin coverage: " + name)
        probes = _select_probes(ids, rest, coverage, control_probe_limit)
        controls[name] = {"node": names[name], "descendants": descendants, "ids": ids, "coverage": coverage, "probes": probes}
        selected.extend(map(int, probes))
    stable_selections = {}
    for action in action_bindings:
        for role in action["roles"]:
            key = (action["state_id"], role["id"])
            if key not in stable_roles:
                continue
            direct = [names[bone] for bone in role["bones"]]
            coverage = np.sum(weights*np.isin(influence_nodes, direct), axis=1)
            ids = np.flatnonzero(coverage >= .5)
            require(len(ids) >= 8, "Stability role lacks eight predominantly weighted direct target vertices")
            probes = _select_probes(ids, rest, coverage)
            stable_selections[key] = {"ids": ids, "probes": probes}
            selected.extend(map(int, probes))
    probe_ids = np.asarray(sorted(set(selected)), dtype=np.int32)
    require(0 < len(probe_ids) <= MAX_PROBES, "Motion probe bound exceeded")
    lookup = {int(vertex): index for index, vertex in enumerate(probe_ids)}
    for control in controls.values():
        control["probe_indices"] = np.asarray([lookup[int(vertex)] for vertex in control["probes"]], dtype=np.int32)
        control["bone_local"] = _points(np.linalg.inv(rest_world[control["node"]]), rest[control["probes"]])
    animations = three_d._animation_tracks(glb, nodes, joint_nodes)
    tracks_by_name = {}
    for name, tracks, _ in animations:
        require(isinstance(name, str) and name not in tracks_by_name, "Animations require unique names")
        tracks_by_name[name] = tracks
    require(all(state["id"] in tracks_by_name for state in states), "Requested action is missing from actual GLB")
    schedules = {state["id"]: _sample_times(tracks_by_name[state["id"]]) for state in states}
    require(sum(map(len, schedules.values())) <= MAX_TOTAL_FRAMES, "Total action frame bound exceeded")
    report = {"schema_version": 1, "audit_version": VERSION, "status": "measured", "passed": False,
              "model_sha256": glb.hash, "actual_bone_parents": bone_parents,
              "aggregate_vertices": total, "anatomical_vertices": len(points), "anatomical_primitives": primitives,
              "excluded_surfaces": excluded, "probe_vertex_ids": probe_ids.tolist(), "probe_count": len(probe_ids),
              "probe_index_space": "Concatenated anatomical primitives in anatomical_primitives order",
              "fps": 24, "sampled_frames": sum(map(len, schedules.values())), "states": [],
              "control_probe_limit_used": control_probe_limit,
              "limits": {"vertices": MAX_VERTICES, "joints": MAX_JOINTS, "states": MAX_STATES,
                         "probes": MAX_PROBES, "probes_per_control": PROBES_PER_CONTROL,
                         "frames_per_state": MAX_FRAMES_PER_STATE, "total_frames": MAX_TOTAL_FRAMES,
                         "full_skin_frames_per_state": MAX_FULL_FRAMES, "seconds": MAX_SECONDS,
                         "report_bytes": MAX_REPORT_BYTES, "file_bytes": three_d.MAX_FILE_BYTES},
              "scope": "Actual GLB linear-blend skinning. Travel uses bound root origin; articulation removes animated root transform; stability measures world displacement.",
              "limitations": ["No semantic, visual, collision, physical, Weft or seamless-loop approval.",
                              "24fps joint-targeted probes are sampled evidence, not all vertices at every time.",
                              "Moving-vertex counts are exact conservative counts at listed full-skin frames, never extrapolated from probes.",
                              "Roles include target-joint descendants with at least 0.05 cumulative weight; UV-seam vertices remain separate.",
                              "Stability instead selects direct target-bone surfaces with at least0.5 summed weight, excluding unrelated descendants.",
                              "Stability is sampled world-space stillness, not ground contact, balance or a continuous-time upper bound.",
                              "Input weight-sum encoding errors up to 0.001 are reported and normalized for CPU skinning.",
                              "Reserved coat/groom surfaces are excluded; source mesh naming outside those prefixes is retained.",
                              "Travel/rise/descent use glTF meter coordinates and +Y up, not ground/contact measurements."],
              "semantic_approval": False, "physics_verified": False, "numpy_version": np.__version__}
    for state, bound in zip(states, action_bindings):
        _deadline(start)
        times, tracks = schedules[state["id"]], tracks_by_name[state["id"]]
        role_data = {}
        for role in bound["roles"]:
            stable = stable_selections.get((state["id"], role["id"]))
            if stable is not None:
                ids = stable["ids"]
                probes = np.asarray([lookup[int(vertex)] for vertex in stable["probes"]], dtype=np.int32)
            else:
                descendants = set().union(*(controls[name]["descendants"] for name in role["bones"]))
                covered = np.sum(weights*np.isin(influence_nodes, list(descendants)), axis=1)
                ids = np.flatnonzero(covered >= WEIGHT_FLOOR)
                probes = np.asarray(sorted({lookup[int(vertex)] for bone in role["bones"] for vertex in controls[bone]["probes"]}), dtype=np.int32)
            require(len(ids) >= 3, "Action role has insufficient anatomical vertices")
            scale = float(np.ptp(rest[ids], axis=0).max())
            require(.0001 <= scale <= 100, "Action role has degenerate or oversized geometry")
            root = names[role["root_bone"]]
            role_data[role["id"]] = {"design": role, "ids": ids, "scale": scale, "root": root, "probes": probes,
                "stable": stable is not None, "root_positions": [], "relative": [], "world_displacement": [],
                "full_world_max": 0., "own_max": {name: 0. for name in role["bones"]},
                "moving": np.zeros(len(ids), dtype=bool)}
        worlds, own_baseline = [], {}
        for second in times:
            _deadline(start)
            world = np.asarray([_matrix(matrix) for matrix in three_d._world_matrices(nodes, parents, three_d._pose(tracks, second))])
            worlds.append(world)
            posed = _skin(world, points[probe_ids], weights[probe_ids], joint_slots[probe_ids], palette, inverse)
            own_changes = {}
            for name, control in controls.items():
                joint, parent = control["node"], parents.get(control["node"])
                own = np.linalg.solve(world[parent], world[joint]) if parent is not None else world[joint]
                rest_parent = rest_world[parent] if parent is not None else np.eye(4)
                shifted = _points(rest_parent@own, control["bone_local"])
                if name not in own_baseline:
                    own_baseline[name] = shifted.copy()
                displacement = np.linalg.norm(shifted-own_baseline[name], axis=1) * control["coverage"][control["probes"]]
                own_changes[name] = float(displacement.max(initial=0))
            for item in role_data.values():
                root = item["root"]
                item["root_positions"].append(world[root, :3, 3].copy())
                local = _points(np.linalg.inv(world[root]), posed[item["probes"]])
                if "baseline_local" not in item:
                    item["baseline_local"] = local.copy()
                    item["baseline_world"] = posed[item["probes"]].copy()
                relative = (local-item["baseline_local"]) @ worlds[0][root, :3, :3].T
                item["relative"].append(relative)
                item["world_displacement"].append(float(np.linalg.norm(posed[item["probes"]]-item["baseline_world"], axis=1).max(initial=0)))
                for name in item["design"]["bones"]:
                    item["own_max"][name] = max(item["own_max"][name], own_changes[name])
        # Each role gets its relevant peak; stability must not miss inherited root travel.
        full_frames = {0, len(times)-1, len(times)//2}
        for item in role_data.values():
            relative = np.asarray(item["relative"])
            item["relative"] = relative
            frame_peaks = np.linalg.norm(relative, axis=2).max(axis=1)
            full_frames.add(int(np.argmax(item["world_displacement"] if item["stable"] else frame_peaks)))
        require(len(full_frames) <= MAX_FULL_FRAMES, "Full-skin frame budget exceeded")
        for frame in sorted(full_frames):
            _deadline(start)
            posed = _skin(worlds[frame], points, weights, joint_slots, palette, inverse)
            for item in role_data.values():
                root, ids = item["root"], item["ids"]
                current_local = _points(np.linalg.inv(worlds[frame][root]), posed[ids])
                if "full_baseline_local" not in item:
                    item["full_baseline_local"] = current_local.copy()
                    item["full_baseline_world"] = posed[ids].copy()
                shift = np.linalg.norm((current_local-item["full_baseline_local"]) @ worlds[0][root, :3, :3].T, axis=1)
                item["moving"] |= shift > max(1e-5, .001*item["scale"])
                item["full_world_max"] = max(item["full_world_max"], float(np.linalg.norm(posed[ids]-item["full_baseline_world"], axis=1).max(initial=0)))
        metrics, detail = {}, {}
        for identifier, item in role_data.items():
            scale, relative = item["scale"], item["relative"]
            travel = np.asarray(item["root_positions"])
            root_shift = travel-travel[0]
            traces = [root_shift[:, axis] for axis in range(3)]
            for axis in range(3):
                strongest = np.argsort(np.ptp(relative[:, :, axis], axis=0))[-min(3, relative.shape[1]):]
                traces.extend(relative[:, index, axis] for index in strongest)
            metrics[identifier] = {"scale_m": scale, "weighted_vertices": len(item["ids"]),
                "moving_vertices": int(item["moving"].sum()),
                "max_root_relative_displacement_m": float(np.linalg.norm(relative, axis=2).max(initial=0)),
                "max_world_displacement_m": max(item["full_world_max"], max(item["world_displacement"])),
                "max_travel_m": float(np.linalg.norm(root_shift, axis=1).max(initial=0)),
                "max_vertical_rise_m": float(max(0, root_shift[:, 1].max())),
                "descent_from_peak_m": float(max(0, travel[:, 1].max()-travel[-1, 1])),
                "rhythmic_reversals": max(_reversals(trace, .005*scale) for trace in traces),
                "active_control_count": sum(value >= .002*scale for value in item["own_max"].values())}
            detail[identifier] = {"root_bone": item["design"]["root_bone"], "bones": item["design"]["bones"],
                "probe_count": len(item["probes"]), "own_control_displacement_m": item["own_max"],
                "surface_selection": "direct_target_bone_weight_sum_at_least_0.5" if item["stable"] else "target_and_descendant_weight_sum_at_least_0.05",
                "selected_vertices": len(item["ids"]),
                "moving_vertex_threshold_m": max(1e-5, .001*scale), "rest_world_bounds_m": {
                    "min": rest[item["ids"]].min(axis=0).tolist(), "max": rest[item["ids"]].max(axis=0).tolist()}}
        evidence = {"state_id": state["id"], "evidence_id": "action-audit-"+state["id"], "roles": metrics}
        result = evaluate_action_evidence(state["requirements"], evidence, height_meters)
        report["states"].append({"id": state["id"], "evidence": evidence, "requirement_results": result,
                                 "sample_seconds": times, "full_skin_sample_seconds": [times[frame] for frame in sorted(full_frames)],
                                 "role_details": detail})
    _deadline(start)
    require(sha256(model_path) == glb.hash, "Source model changed during audit")
    report["passed"] = all(state["requirement_results"]["passed"] for state in report["states"])
    report["input_contract_sha256"] = hashlib.sha256(json.dumps({"states": states, "action_bindings": action_bindings,
        "height_meters": height_meters}, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    report["implementation_sha256"] = {path.name: sha256(path) for path in (
        Path(__file__), Path(three_d.__file__), Path(__file__).with_name("action_intent.py"))}
    report["source_unchanged"] = True
    report["elapsed_seconds"] = time.monotonic()-start
    require(len(json.dumps(report, allow_nan=False).encode()) <= MAX_REPORT_BYTES, "Action audit report exceeds output bound")
    return report


def write_audit(model_path, intent_path, report_path):
    """Persist once; source_sha256 identifies this exported GLB, not its provider ancestor."""
    model = local_file(model_path, three_d.MAX_FILE_BYTES)
    intent_file = local_file(intent_path, 128*1024)
    intent_hash = sha256(intent_file)
    intent = json.loads(intent_file.read_text())
    require(isinstance(intent, dict) and set(intent) == {"schema_version", "states", "action_bindings", "height_meters", "source_sha256"}
            and type(intent["schema_version"]) is int and intent["schema_version"] == 1, "Invalid action audit intent envelope")
    require(isinstance(intent["source_sha256"], str) and sha256(model) == intent["source_sha256"], "Action audit model hash mismatch")
    output = Path(report_path)
    require(output.parent.is_dir() and not output.exists() and not output.is_symlink(), "Action audit report already exists or has no parent")
    report = audit_glb(model, intent["states"], intent["action_bindings"], intent["height_meters"])
    require(sha256(intent_file) == intent_hash, "Action intent changed during audit")
    report["intent_sha256"] = intent_hash
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False)+"\n"
    require(len(payload.encode()) <= MAX_REPORT_BYTES, "Action audit report exceeds output bound")
    with output.open("x") as handle:
        handle.write(payload)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("intent")
    parser.add_argument("report")
    args = parser.parse_args()
    result = write_audit(args.model, args.intent, args.report)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "states": len(result["states"]),
                      "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
