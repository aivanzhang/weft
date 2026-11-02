"""No-render native replay of corrected four-row deformation retention."""

import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"sources/native"))
from blender_verify import bind_action, clear_actions, hash_file, imported_bindings, require, set_frame, source_header, write_json
from capture import capture
from deformation_diagnostics import SEVERE_HEIGHT_FRACTION, SEVERE_RATIO, diagnose_stretched_edges, retain_stretched_edges


def digest(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes(order="C")).hexdigest()


def main():
    started = time.monotonic()
    prepared = json.loads((HERE/"prepared.json").read_text())
    for row in prepared["frozen_files"]:
        require(hash_file(HERE/row["path"]) == row["sha256"], "Frozen replay input changed")
    model = HERE/"inputs/model.glb"
    require(hash_file(model) == prepared["model_sha256"], "Replay model hash changed")
    previous = json.loads((HERE/"inputs/original-native-deformation.json").read_text())
    require(previous["source_sha256"] == prepared["model_sha256"], "Prior native report has a different model")
    doc = source_header(model)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    bpy.context.scene.render.fps, bpy.context.scene.render.fps_base = 24, 1
    bpy.ops.import_scene.gltf(filepath=str(model))
    objects = list(bpy.context.scene.objects)
    rigs = [obj for obj in objects if obj.type == "ARMATURE"]
    require(len(rigs) == 1 and len(rigs[0].data.bones) == 24, "Unexpected imported rig")
    rig = rigs[0]
    helpers = {bone.custom_shape for bone in rig.pose.bones if bone.custom_shape}
    meshes = [obj for obj in objects if obj.type == "MESH" and obj not in helpers]
    require(len(meshes) == 1, "Exactly one imported body is required")
    obj = meshes[0]
    names = {doc["nodes"][i]["name"] for i in doc["skins"][0]["joints"]}
    require(set(rig.data.bones.keys()) == names and not obj.data.shape_keys and len(obj.modifiers) == 1
            and obj.modifiers[0].type == "ARMATURE", "Native rig or modifier structure changed")
    actions = {action.name: action for action in bpy.data.actions}
    require(set(actions) == {state["id"] for state in previous["states"]}, "Actual action set changed")
    bindings, owners = imported_bindings(objects, list(actions.values()))
    require(all(bindings[name] and all(owner == rig for owner, _ in bindings[name]) for name in actions),
            "Imported action owner/slot missing or changed")
    clear_actions(owners, rigs)
    rig.data.pose_position = "REST"
    set_frame(0)
    rest, edges, loops, starts, totals = capture(obj)
    require(digest(rest, "<f8") == previous["rest_sha256"] and digest(edges, "<i4") == previous["edges_sha256"],
            "Fresh import rest geometry differs from the original native diagnosis")
    span = np.ptp(rest, axis=0)
    height = float(span[2]) if span[2] > float(span.max())*1e-6 else float(span.max())
    require(height == previous["measured_rest_bounds"]["height_m"] and SEVERE_RATIO == 8. and SEVERE_HEIGHT_FRACTION == .03,
            "Measured height or fixed thresholds changed")
    group_names = {group.index: group.name for group in obj.vertex_groups}
    result = {"schema_version": 1, "status": "fresh_native_retention_replay", "source_sha256": hash_file(model),
              "prior_native_report_sha256": hash_file(HERE/"inputs/original-native-deformation.json"),
              "corrected_helper_sha256": hash_file(HERE/"sources/native/deformation_diagnostics.py"),
              "finisher_source_sha256": hash_file(HERE/"sources/blender_finish_generated.py"),
              "rest_sha256": digest(rest, "<f8"), "edges_sha256": digest(edges, "<i4"),
              "measured_rest_height_m": height, "states": [], "passed_retention_regression": True,
              "model_quality_approved": False, "new_renders": 0, "new_paid_calls": 0,
              "scope": "Fresh imported skin samples using corrected per-pose and cross-pose helper; not a full finisher rebuild or quality approval."}
    rig.data.pose_position = "POSE"
    for state in previous["states"]:
        bind_action(actions[state["id"]], bindings, owners, rigs)
        collected, samples = [], []
        for old in state["samples"]:
            require(time.monotonic()-started < 55, "Numeric replay exceeded its internal time budget")
            set_frame(old["second"]*24)
            posed, pe, pl, ps, pt = capture(obj)
            require(posed.shape == rest.shape and all(np.array_equal(a, b) for a, b in
                    ((pe, edges), (pl, loops), (ps, starts), (pt, totals))), "Evaluated topology changed")
            require(digest(posed, "<f8") == old["pose_sha256"], "Fresh pose differs from the original native diagnosis")
            rows = diagnose_stretched_edges(rest, posed, edges)
            for row in rows:
                row["second"] = old["second"]
                row["endpoint_influences"] = [{group_names[g.group]: float(g.weight)
                    for g in obj.data.vertices[index].groups if g.weight > 1e-6} for index in row["endpoint_indices"]]
            severe = [row for row in rows if row["stretch_ratio"] > SEVERE_RATIO
                      and row["excess_length"] > SEVERE_HEIGHT_FRACTION*height]
            expected = min(4, old["severe_edge_count"])
            matched = len(severe) == expected
            result["passed_retention_regression"] &= matched
            samples.append({"second": old["second"], "unchanged_pose_and_topology": True,
                            "previous_every_edge_severe_count": old["severe_edge_count"],
                            "previous_retained_severe_count": sum(r["stretch_ratio"] > 8 and r["excess_length"] > .03*height
                                                                  for r in old["deformation_outliers"]),
                            "corrected_retained_severe_count": len(severe), "matched_expected_cap": matched,
                            "deformation_outliers": rows})
            collected.extend(rows)
        retained = retain_stretched_edges(collected, height)
        severe = [row for row in retained if row["stretch_ratio"] > SEVERE_RATIO
                  and row["excess_length"] > SEVERE_HEIGHT_FRACTION*height]
        require(not any(sample["previous_every_edge_severe_count"] for sample in samples) or severe,
                "Cross-pose aggregation lost every severe witness")
        result["states"].append({"id": state["id"], "samples": samples, "deformation_outliers": retained,
                                 "retained_severe_count": len(severe), "asset_rejected_by_unchanged_thresholds": bool(severe)})
    clear_actions(owners, rigs)
    rig.data.pose_position = "REST"
    require(hash_file(model) == prepared["model_sha256"], "Input GLB changed")
    result["blender_version"] = bpy.app.version_string
    result["elapsed_seconds"] = time.monotonic()-started
    write_json(HERE/"native-replay.json", result)
    require(result["passed_retention_regression"], "Corrected helper failed to retain known severe witnesses")
    print(json.dumps({"status": result["status"], "passed_retention_regression": result["passed_retention_regression"],
                      "asset_approved": False, "elapsed_seconds": result["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
