"""Fresh bounded import/render/deformation evidence for one pinned diagnostic GLB."""

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"sources/native"))
from blender_provider_preview import main as provider_preview
from blender_verify import bind_action, clear_actions, hash_file, imported_bindings, inspect_geometry, require, set_frame, source_header, write_json
from deformation_diagnostics import diagnose_stretched_edges


MAX_VERTICES, MAX_EDGES, MAX_TRIANGLES = 100000, 300000, 100000
RATIO_LIMIT, EXCESS_HEIGHT_RATIO = 8., .03


def array_hash(value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes(order="C")).hexdigest()


def measure_edges(rest, posed, edges, height):
    outliers = diagnose_stretched_edges(rest, posed, edges)
    require(type(height) in (int, float) and math.isfinite(height) and height > 0, "Invalid measured rest height")
    scale = float(np.ptp(rest, axis=0).max())
    original = np.hypot.reduce(rest[edges[:, 1]]-rest[edges[:, 0]], axis=1)
    deformed = np.hypot.reduce(posed[edges[:, 1]]-posed[edges[:, 0]], axis=1)
    valid = original > max(scale*1e-6, 1e-12)
    ratios = np.divide(deformed, original, out=np.zeros(len(edges)), where=valid)
    excess = deformed-original
    severe = np.flatnonzero(valid & (ratios > RATIO_LIMIT) & (excess > EXCESS_HEIGHT_RATIO*height))
    order = severe[np.lexsort((severe, -ratios[severe], -excess[severe]))]
    low, span = rest.min(axis=0), np.ptp(rest, axis=0)
    rows = [{"edge_index": int(index), "endpoint_indices": edges[index].tolist(),
             "rest_length": float(original[index]), "posed_length": float(deformed[index]),
             "excess_length": float(excess[index]), "stretch_ratio": float(ratios[index]),
             "rest_normalized_bounds": ((rest[edges[index]]-low)/np.where(span > 0, span, 1)).tolist()}
            for index in order[:4]]
    return {"deformation_outliers": outliers, "severe_edge_count": len(severe), "severe_edges": rows,
            "eligible_edges": int(valid.sum()), "ignored_tiny_rest_edges": int((~valid).sum()),
            "maximum_eligible_excess_m": max(0., float(excess[valid].max())) if valid.any() else 0.,
            "passed_no_sampled_severe_spike": len(severe) == 0}


def capture(obj):
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    try:
        require(0 < len(mesh.vertices) <= MAX_VERTICES and 0 < len(mesh.edges) <= MAX_EDGES
                and 0 < len(mesh.polygons) <= MAX_TRIANGLES and len(mesh.loops) == len(mesh.polygons)*3,
                "Imported evaluated geometry exceeds diagnostic bounds")
        points = np.empty((len(mesh.vertices), 3), dtype=np.float64)
        mesh.vertices.foreach_get("co", points.ravel())
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        points = points@matrix[:3, :3].T+matrix[:3, 3]
        edges = np.empty((len(mesh.edges), 2), dtype=np.int32)
        mesh.edges.foreach_get("vertices", edges.ravel())
        loops = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get("vertex_index", loops)
        starts, totals = np.empty(len(mesh.polygons), dtype=np.int32), np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get("loop_start", starts)
        mesh.polygons.foreach_get("loop_total", totals)
        require(np.all(totals == 3) and np.isfinite(points).all() and np.max(np.abs(points)) <= 100,
                "Nonfinite/unbounded positions or nontriangle topology")
        require(edges.min() >= 0 and edges.max() < len(points) and loops.min() >= 0 and loops.max() < len(points),
                "Invalid imported topology indices")
        return points, edges, loops, starts, totals
    finally:
        evaluated.to_mesh_clear()


def attach_weights(rows, obj, bone_names):
    group_names = {group.index: group.name for group in obj.vertex_groups}
    for row in rows:
        row["endpoint_influences"] = []
        for index in row["endpoint_indices"]:
            influences = [{"bone": group_names[group.group], "weight": float(group.weight)}
                          for group in obj.data.vertices[index].groups
                          if group_names.get(group.group) in bone_names and group.weight > 0]
            require(0 < len(influences) <= 4 and all(math.isfinite(x["weight"]) for x in influences),
                    "Invalid imported endpoint weights")
            row["endpoint_influences"].append(influences)


def main():
    started = time.monotonic()
    prepared = json.loads((HERE/"prepared.json").read_text())
    for item in prepared["frozen_files"]:
        require(hash_file(HERE/item["path"]) == item["sha256"], "Prepared native input changed: "+item["path"])
    model = HERE/"inputs/model.glb"
    design_path, motion_path = HERE/"inputs/original-finish.json", HERE/"inputs/motion-states.json"
    require(hash_file(model) == prepared["model_sha256"], "Derivative model pin mismatch")
    design = json.loads(design_path.read_text())
    schedule = json.loads(motion_path.read_text())
    require(len(schedule) == 1 and len(design["authored_actions"]) == 1,
            "This diagnostic is bounded to its one original action")
    action_design = design["authored_actions"][0]
    require(schedule == [{"id": action_design["id"], "seconds": [key["second"] for key in action_design["keys"]]}]
            and 2 <= len(schedule[0]["seconds"]) <= 8, "Sample schedule differs from original authored keys")
    doc = source_header(model)
    require(len(doc["skins"]) == 1, "Exactly one actual skin is required")
    joint_ids = doc["skins"][0]["joints"]
    expected_names = {doc["nodes"][index]["name"] for index in joint_ids}
    require(len(expected_names) == prepared["expected_bones"] and
            all(channel["target"]["node"] in joint_ids for animation in doc["animations"] for channel in animation["channels"]),
            "Unexpected bone count or nonbone animation target")
    # Reuse the unmodified provider renderer; no geometry or material edits occur here.
    provider_preview(model, HERE, prepared["resolution"], prepared["samples"], False, motion_path)
    # A fresh import preserves actual action ownership after the preview clears its actions.
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    bpy.context.scene.render.fps, bpy.context.scene.render.fps_base = 24, 1
    bpy.ops.import_scene.gltf(filepath=str(model))
    scene = bpy.context.scene
    scene.render.fps, scene.render.fps_base = 24, 1
    rigs = [obj for obj in scene.objects if obj.type == "ARMATURE"]
    require(len(rigs) == 1 and set(rigs[0].data.bones.keys()) == expected_names, "Imported bone set changed")
    rig = rigs[0]
    helpers = {bone.custom_shape for bone in rig.pose.bones if bone.custom_shape}
    meshes = [obj for obj in scene.objects if obj.type == "MESH" and obj not in helpers]
    require(len(meshes) == 1, "Diagnostic requires exactly one imported base mesh")
    obj = meshes[0]
    require(not obj.data.shape_keys and len(obj.modifiers) == 1 and obj.modifiers[0].type == "ARMATURE"
            and obj.modifiers[0].object == rig, "Unexpected imported deformation mechanism")
    records, _, _, _, _ = inspect_geometry(meshes, rigs)
    actions = {action.name: action for action in bpy.data.actions}
    require(set(actions) == {row["id"] for row in schedule}, "Imported actions differ from original state list")
    bindings, owners = imported_bindings(list(scene.objects), list(actions.values()))
    for action in actions.values():
        require(bindings[action.name] and all(owner == rig for owner, _ in bindings[action.name]),
                "Actual imported action owner/slot is missing or unexpected")
    clear_actions(owners, rigs)
    rig.data.pose_position = "REST"
    set_frame(0)
    rest, edges, loops, starts, totals = capture(obj)
    require(len(rest) == len(obj.data.vertices), "Rest evaluation changed native imported vertex indices")
    low, high = rest.min(axis=0), rest.max(axis=0)
    span, scale = high-low, float(np.max(high-low))
    require(scale > 1e-6 and np.isfinite(scale), "Degenerate measured rest bounds")
    height = float(span[2]) if span[2] > scale*1e-6 else scale
    rest_hash, edge_hash = array_hash(rest, "<f8"), array_hash(edges, "<i4")
    report = {"schema_version": 1, "status": "measured", "source_sha256": hash_file(model),
              "original_design_sha256": hash_file(design_path), "motion_states_sha256": hash_file(motion_path),
              "wrapper_sha256": hash_file(Path(__file__)), "blender_version": bpy.app.version_string,
              "coordinate_space": "Actual imported mesh world coordinates, Blender XYZ +Z up, meters",
              "index_space": "Native imported mesh vertex/edge indices; not exported indices or recovered anatomy",
              "measurement_import": "Independent fresh import of the identical pinned GLB after unmodified preview rendering",
              "rest_pose": "Actual imported armature set to REST; NLA/actions disabled before evaluated baseline",
              "measured_rest_bounds": {"min": low.tolist(), "max": high.tolist(), "height_m": height,
                                       "height_basis": "z_span" if height == float(span[2]) else "maximum_span"},
              "rest_sha256": rest_hash, "edges_sha256": edge_hash,
              "loop_indices_sha256": array_hash(loops, "<i4"), "polygon_loop_starts_sha256": array_hash(starts, "<i4"),
              "hash_encodings": {"rest": "little-endian float64 XYZ world meters", "topology": "little-endian int32"},
              "native_mesh": records[0], "bone_names": sorted(expected_names),
              "vertices": len(rest), "edges": len(edges), "triangles": len(starts),
              "thresholds": {"ratio_strictly_greater_than": RATIO_LIMIT, "excess_height_ratio_strictly_greater_than": EXCESS_HEIGHT_RATIO,
                             "actual_excess_limit_m": height*EXCESS_HEIGHT_RATIO,
                             "tiny_rest_edge_exclusion_m": max(scale*1e-6, 1e-12)},
              "states": [], "passed": True, "visual_quality_approved": False, "physics_verified": False,
              "source_deformation_report_reused": False,
              "limits": {"vertices": MAX_VERTICES, "edges": MAX_EDGES, "triangles": MAX_TRIANGLES,
                         "action_samples": 8, "cpu_threads": 4, "wall_seconds": 600, "rss_mib": 2048, "output_mib": 256},
              "limitations": ["Every eligible edge is checked only at the eight listed authored-key times, not all time.",
                              "No severe sampled spike is not a silhouette, joint, identity or visual-quality approval.",
                              "Multi-bone protected subsets are not independently certified co-rigid.",
                              "No collision, ground-contact, physics, Weft or seamless-loop claim."]}
    rig.data.pose_position = "POSE"
    for state in schedule:
        action = actions[state["id"]]
        bind_action(action, bindings, owners, rigs)
        require(abs(float(action.frame_range[0])-state["seconds"][0]*24) <= .01 and
                abs(float(action.frame_range[1])-state["seconds"][-1]*24) <= .01,
                "Imported action range differs from original design")
        rows = []
        for number, second in enumerate(state["seconds"]):
            require(time.monotonic()-started < 590, "Native diagnostic exhausted its internal wall budget")
            set_frame(second*24)
            posed, pose_edges, pose_loops, pose_starts, pose_totals = capture(obj)
            require(posed.shape == rest.shape and np.array_equal(pose_edges, edges)
                    and np.array_equal(pose_loops, loops) and np.array_equal(pose_starts, starts)
                    and np.array_equal(pose_totals, totals), "Evaluated sample changed exact imported topology")
            measured = measure_edges(rest, posed, edges, height)
            attach_weights(measured["deformation_outliers"], obj, expected_names)
            attach_weights(measured["severe_edges"], obj, expected_names)
            path = HERE/"renders"/state["id"]/f"{number:04d}.png"
            measured.update({"second": second, "frame": second*24, "pose_sha256": array_hash(posed, "<f8"),
                             "render": str(path.relative_to(HERE)), "render_sha256": hash_file(path),
                             "unchanged_topology": True})
            rows.append(measured)
        passed = all(row["passed_no_sampled_severe_spike"] for row in rows)
        report["states"].append({"id": state["id"], "seconds": state["seconds"], "samples": rows, "passed": passed,
                                 "action_owners": [{"name": owner.name, "slot": slot.identifier if slot else None}
                                                   for owner, slot in bindings[state["id"]]]})
        report["passed"] &= passed
    clear_actions(owners, rigs)
    rig.data.pose_position = "REST"
    set_frame(0)
    require(hash_file(model) == prepared["model_sha256"], "Original diagnostic GLB changed")
    report["elapsed_seconds"] = time.monotonic()-started
    report["status"] = "no_sampled_severe_spike" if report["passed"] else "sampled_severe_spike_present"
    write_json(HERE/"native-deformation.json", report)
    print(json.dumps({"status": report["status"], "passed": report["passed"], "source_sha256": report["source_sha256"],
                      "sample_count": sum(len(state["samples"]) for state in report["states"]),
                      "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
