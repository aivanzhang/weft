"""One append-only CPU weight diagnostic; never render or promote its result."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avatar_assets import action_audit, three_d
from avatar_assets.action_intent import validate_action_bindings, validate_state_intents
from avatar_assets.safety import artifacts, require, sha256, write_json
from avatar_assets.stable_role_projection import project_stable_roles


def digest(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def layout(glb, index):
    accessor = glb.doc["accessors"][index]
    require(not accessor.get("sparse"), "Sparse accessor is unsupported")
    width = three_d._WIDTHS[accessor["type"]]
    code, size = three_d._COMPONENTS[accessor["componentType"]]
    view = glb.doc["bufferViews"][accessor["bufferView"]]
    count, stride = accessor["count"], view.get("byteStride", width*size)
    start = view.get("byteOffset", 0)+accessor.get("byteOffset", 0)
    require(type(count) is int and 0 < count <= 300000 and type(stride) is int
            and width*size <= stride <= 252 and stride % size == 0
            and accessor.get("byteOffset", 0)+(count-1)*stride+width*size <= view["byteLength"],
            "Accessor layout exceeds diagnostic bounds")
    return np.ndarray((count, width), dtype=np.dtype("<"+code), buffer=glb.binary,
                      offset=start, strides=(stride, size)), start, stride, width*size


def inventory(glb, raw):
    values = []
    for index, accessor in enumerate(glb.doc["accessors"]):
        data, _, _, _ = layout(glb, index)
        values.append({"index": index, "metadata_sha256": digest(canonical(accessor)),
                       "raw_components_sha256": digest(data.tobytes(order="C")),
                       "component_bytes": data.nbytes, "metadata": accessor})
    attributes = [{"mesh": mi, "primitive": pi, "attributes": p["attributes"], "indices": p.get("indices")}
                  for mi, mesh in enumerate(glb.doc["meshes"]) for pi, p in enumerate(mesh["primitives"])]
    animations = []
    for animation in glb.doc.get("animations", []):
        indices = sorted({s[key] for s in animation["samplers"] for key in ("input", "output")})
        animations.append({"name": animation["name"], "metadata_sha256": digest(canonical(animation)),
                           "accessors": [{"index": i, "raw_components_sha256": values[i]["raw_components_sha256"]}
                                         for i in indices]})
    images = []
    for image in glb.doc.get("images", []):
        view = glb.doc["bufferViews"][image["bufferView"]]
        begin = view.get("byteOffset", 0)
        images.append({"metadata": image, "bytes": view["byteLength"],
                       "sha256": digest(glb.binary[begin:begin+view["byteLength"]])})
    json_size = struct.unpack_from("<I", raw, 12)[0]
    return {"glb_sha256": digest(raw), "file_bytes": len(raw),
            "json_chunk_sha256": digest(raw[20:20+json_size]), "accessors": values,
            "vertex_attributes": attributes, "animations": animations, "images": images,
            "nodes_sha256": digest(canonical(glb.doc["nodes"])),
            "skins_sha256": digest(canonical(glb.doc["skins"])),
            "materials_sha256": digest(canonical(glb.doc["materials"]))}


def mask(weights, joints, selected, threshold):
    normal = weights/weights.sum(axis=1, keepdims=True)
    return np.sum(normal*np.isin(joints, selected), axis=1) >= threshold


def main():
    started = time.monotonic()
    output = Path(__file__).resolve().parent
    settings = json.loads((output/"settings.json").read_text())
    source = ROOT/settings["source"]
    intent_path = ROOT/settings["intent"]
    require(sha256(source) == settings["source_sha256"], "Pinned source changed")
    require(sha256(intent_path) == settings["intent_sha256"], "Pinned intent changed")
    target = output/"diagnostic-model.glb"
    require(not target.exists(), "Never overwrite a diagnostic model")
    for item in settings["source_files"]:
        require(sha256(ROOT/item["path"]) == item["sha256"], "Bound source file changed")
    intent = json.loads(intent_path.read_text())
    require(intent["source_sha256"] == settings["source_sha256"], "Intent is not bound to this GLB")
    glb = three_d._GLB(source)
    nodes, parents, _, reachable, bone_ids, bone_parents = action_audit._hierarchy(glb)
    require(len(bone_ids) == settings["expected_bones"], "Unexpected actual bone count")
    exact = [state.get("user_request", "") for state in intent["states"]]
    validate_state_intents(intent["states"], exact if any(exact) else [], [s["id"] for s in intent["states"]])
    validate_action_bindings(intent["action_bindings"], intent["states"], bone_parents)
    geometry = action_audit._load_geometry(glb, nodes, reachable, started)
    points, _, joints, palette, _, excluded, primitives, _, original = geometry
    require(not excluded and len(glb.doc["skins"]) == 1, "Diagnostic profile requires one body skin without separate coats")
    names = [nodes[index]["name"] for index in palette]
    require(len(set(names)) == len(names), "Skin palette reuses actual joints")
    stable = {(state["id"], req["role"]) for state in intent["states"] for req in state["requirements"]
              if req["kind"] == "stability"}
    roles = [{"state_id": state["state_id"], "role_id": role["id"], "bones": role["bones"],
              "stable": (state["state_id"], role["id"]) in stable}
             for state in intent["action_bindings"] for role in state["roles"]]
    write_json(output/"derived-roles.json", {"intent_sha256": sha256(intent_path), "roles": roles,
                                           "bone_names": names, "bone_parents": bone_parents})
    projected, report = project_stable_roles(joints, original, names, roles, bone_parents=bone_parents,
                                            surface_kind=settings["surface_kind"])
    write_json(output/"projection-report.json", report)
    # Check the actual float32 representation before creating a derivative.
    encoded = projected.astype("<f4").astype(np.float64)
    index = {name: i for i, name in enumerate(names)}
    for row in report["roles"]:
        for label, bones, threshold in (("core", row["bones"], .5), ("fringe", row["fringe_bones"], .05)):
            selected = np.flatnonzero(mask(encoded, joints, [index[name] for name in bones], threshold))
            require(digest(np.asarray(selected, dtype="<i4").tobytes()) == row[label+"_indices_after_sha256"],
                    "Float32 serialization changed a verified role membership")
    require(np.all(np.isfinite(encoded)) and np.all(np.abs(encoded.sum(axis=1)-1) <= .001),
            "Encoded weights invalid")
    raw = source.read_bytes()
    updated = bytearray(raw)
    binary_start = 28+struct.unpack_from("<I", raw, 12)[0]
    intervals, weight_ids, serialized_changed = [], set(), []
    for item in primitives:
        primitive = glb.doc["meshes"][nodes[item["node"]]["mesh"]]["primitives"][item["primitive"]]
        accessor_id = primitive["attributes"]["WEIGHTS_0"]
        accessor = glb.doc["accessors"][accessor_id]
        require(accessor_id not in weight_ids and accessor["componentType"] == 5126
                and accessor["type"] == "VEC4" and "min" not in accessor and "max" not in accessor,
                "Require unique float32 weight accessors without stale extrema")
        weight_ids.add(accessor_id)
        source_rows, offset, stride, row_size = layout(glb, accessor_id)
        start, count = item["anatomical_start_index"], item["vertices"]
        desired = encoded[start:start+count].astype("<f4")
        require(len(source_rows) == count and row_size == 16, "Weight accessor count mismatch")
        for vertex in np.flatnonzero(np.any(desired != source_rows, axis=1)):
            begin = binary_start+offset+int(vertex)*stride
            updated[begin:begin+16] = desired[vertex].tobytes()
            intervals.append((begin, begin+16))
            serialized_changed.append(start+int(vertex))
    intervals.sort()
    require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])), "Overlapping weight writes")
    preserved = hashlib.sha256()
    cursor = 0
    for begin, end in intervals:
        require(raw[cursor:begin] == updated[cursor:begin], "Unexpected bytes outside selected weight rows")
        preserved.update(raw[cursor:begin])
        cursor = end
    require(raw[cursor:] == updated[cursor:], "Unexpected bytes after weight rows")
    preserved.update(raw[cursor:])
    before = inventory(glb, raw)
    with target.open("xb") as handle:
        handle.write(updated)
    derivative = three_d._GLB(target)
    after = inventory(derivative, updated)
    require(glb.doc == derivative.doc, "GLB JSON changed")
    changed_accessors = []
    for old, new in zip(before["accessors"], after["accessors"]):
        if old != new:
            require(old["index"] in weight_ids and old["metadata"] == new["metadata"], "Nonweight accessor changed")
            changed_accessors.append(old["index"])
    for field in ("animations", "images", "nodes_sha256", "skins_sha256", "materials_sha256", "json_chunk_sha256"):
        require(before[field] == after[field], "Protected GLB field changed: "+field)
    reloaded = action_audit._load_geometry(derivative, nodes, reachable, time.monotonic())[-1]
    require(np.array_equal(reloaded, encoded), "Reloaded weights differ from verified float32 output")
    write_json(output/"attribute-inventory-before.json", before)
    write_json(output/"attribute-inventory-after.json", after)
    intent_after = deepcopy(intent)
    intent_after["source_sha256"] = derivative.hash
    write_json(output/"action-intent.json", intent_after)
    require({k: v for k, v in intent.items() if k != "source_sha256"}
            == {k: v for k, v in intent_after.items() if k != "source_sha256"}, "Action contract changed")
    after_audit = action_audit.write_audit(target, output/"action-intent.json", output/"action-audit.json")
    require(sha256(source) == settings["source_sha256"] and sha256(intent_path) == settings["intent_sha256"],
            "An original input changed during diagnostic")
    for item in settings["source_files"]:
        require(sha256(ROOT/item["path"]) == item["sha256"], "Bound source changed during diagnostic")
    source_manifest = json.loads((source.parent/"manifest.json").read_text())
    proof = {"schema_version": 1, "status": "cpu_diagnostic_pending_native_and_visual_review",
             "source": settings["source"], "source_sha256": glb.hash, "result_sha256": derivative.hash,
             "source_bones": [{"node_index": i, "name": nodes[i]["name"]} for i in sorted(bone_ids)],
             "source_bone_count": len(bone_ids), "changed_accessors": changed_accessors,
             "serialized_changed_vertices": len(serialized_changed),
             "serialized_changed_indices_sha256": digest(np.asarray(serialized_changed, dtype="<i4").tobytes()),
             "modified_weight_byte_ranges": intervals,
             "unchanged_bytes_outside_modified_weight_rows_sha256": preserved.hexdigest(),
             "actual_changed_bytes": sum(a != b for a, b in zip(raw, updated)),
             "all_nonweight_accessors_unchanged": True, "json_and_all_24_bone_ids_unchanged": True,
             "geometry_uv_normals_joints_images_materials_rest_pose_and_curves_unchanged": True,
             "original_states_roles_bindings_thresholds_unchanged": True,
             "all_role_core_memberships_unchanged": True,
             "float32_memberships_verified": True, "source_inputs_unchanged": True,
             "action_audit_passed": after_audit["passed"], "native_deformation_tested": False,
             "visual_quality_approved": False, "automatic_promotion": False, "new_paid_calls": 0,
             "rights": source_manifest.get("rights"),
             "authoring_declaration": "Explicit diagnostic hard-surface weight edit; not inferred anatomy or organic skin repair.",
             "limitations": report["limitations"]+["Old native deformation results do not apply to these changed weights."],
             "elapsed_seconds": time.monotonic()-started}
    write_json(output/"preservation-proof.json", proof)
    write_json(output/"cpu-artifacts.json", {"schema_version": 1, "artifacts": artifacts(output)})
    print(json.dumps({"status": proof["status"], "result_sha256": derivative.hash,
                      "changed_vertices": len(serialized_changed), "action_audit_passed": after_audit["passed"],
                      "elapsed_seconds": proof["elapsed_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
