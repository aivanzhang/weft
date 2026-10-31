"""Author consumer metadata; Weft's frozen importer remains the sole integration validator."""

import math
import shutil
from pathlib import Path

import numpy as np

from avatar_assets.safety import artifacts, new_directory, read_json, require, sha256, slug, write_json
from avatar_assets.three_d import _GLB, _local_matrix, _world_matrices

FRAMES = {"mobile_root", "torso", *(f"{side}_{joint}_body" for side in ("left", "right")
                                 for joint in ("shoulder", "elbow", "hip", "knee"))}


def matrix_quaternion(rotation):
    trace = np.trace(rotation)
    if trace > 0:
        s = math.sqrt(trace + 1) * 2
        q = [(rotation[2, 1]-rotation[1, 2])/s, (rotation[0, 2]-rotation[2, 0])/s,
             (rotation[1, 0]-rotation[0, 1])/s, s/4]
    else:
        i = int(np.argmax(np.diag(rotation)))
        j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(1 + rotation[i, i] - rotation[j, j] - rotation[k, k]) * 2
        q = [0., 0., 0., 0.]
        q[i], q[j], q[k], q[3] = s/4, (rotation[j, i]+rotation[i, j])/s, (rotation[k, i]+rotation[i, k])/s, (rotation[k, j]-rotation[j, k])/s
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return [float(x) for x in q]


def create_metadata(folder, design_path, neutral_path, output):
    from avatar_assets.cli import verify_pack
    verify_pack(folder)
    source = Path(folder).resolve()
    manifest = read_json(source / "manifest.json")
    require(manifest.get("output_profile") == "weft-v1", "Build a separate explicit weft-v1 material variant first")
    scene = read_json(source / "sources/scene.json")
    design, neutral = read_json(design_path), read_json(neutral_path)
    require(set(neutral["frames"]) == FRAMES, "Neutral proposal must define all ten frames")
    require(neutral["status"] == "authored_proposal_not_weft_measurement", "This authoring route requires explicitly provisional neutral data")
    model = _GLB(source / "model.glb")
    nodes = model.doc["nodes"]
    names = {node["name"]: i for i, node in enumerate(nodes) if "name" in node}
    require(len(names) == len(nodes), "Unique node names required for unambiguous authoring metadata")
    parents = {child: i for i, node in enumerate(nodes) for child in node.get("children", [])}
    world = [np.asarray(m).reshape(4, 4, order="F") for m in _world_matrices(nodes, parents)]
    bindings = design["bindings"]
    require(isinstance(bindings, dict) and set(bindings.values()) == FRAMES and all(k in names for k in bindings),
            "Explicit bindings must cover all ten target frames using exported node names")
    root_name = design.get("root_node", "AvatarRig")
    require(root_name in bindings and bindings[root_name] == "mobile_root", "Map the containing rig root to mobile_root")
    scale = neutral["scale"]
    require(type(scale) in (int, float) and 0 < scale <= 1000, "Invalid scale")
    orient = neutral["orientation_xyzw"]
    G = np.asarray(_local_matrix({"rotation": orient, "scale": [scale]*3})).reshape(4, 4, order="F")
    proposed = {}
    for name, frame in neutral["frames"].items():
        proposed[name] = np.asarray(_local_matrix({"translation": frame["translation"],
                            "rotation": frame["rotation_xyzw"]})).reshape(4, 4, order="F")
    result_bindings, residuals = [], []
    for name, body in bindings.items():
        i = names[name]
        C = np.linalg.inv(proposed[body]) @ G @ world[i] @ np.linalg.inv(G)
        require(np.allclose(C[:3, :3].T @ C[:3, :3], np.eye(3), atol=1e-6)
                and np.linalg.det(C[:3, :3]) > .99999, "Binding has unrepresentable scale/shear")
        quaternion = matrix_quaternion(C[:3, :3])
        result_bindings.append({"node": i, "body": body, "translation": C[:3, 3].tolist(), "rotation_xyzw": quaternion})
        reconstructed = proposed[body] @ np.asarray(_local_matrix({"translation": C[:3, 3].tolist(), "rotation": quaternion})).reshape(4, 4, order="F") @ G
        residuals.append(float(np.max(np.abs(reconstructed - G @ world[i]))))
    require(max(residuals) < 1e-6, "Proposed-neutral equation verification failed")
    skin = model.doc["skins"][0]
    ibms = model.accessor(skin["inverseBindMatrices"], "MAT4", (5126,))
    bind_residuals = [float(np.max(np.abs(world[j] @ np.asarray(m).reshape(4, 4, order="F") - np.eye(4))))
                      for j, m in zip(skin["joints"], ibms)]
    require(max(bind_residuals) < 1e-5, "Authored world bind and inverse bind differ")
    out = new_directory(output)
    shutil.copyfile(source / "model.glb", out / "model.glb")
    shutil.copyfile(neutral_path, out / "neutral-proposal.json")
    shutil.copyfile(design_path, out / "mapping-design.json")
    metadata = {"asset_id": slug(design.get("asset_id", scene["id"])),
                "display_name": design.get("display_name", scene["subject"][:128]),
                "rights": {"holder": "unknown", "license": "unknown",
                    "source": "New locally authored procedural model; selected reference hashes in provenance",
                    "declaration": "User requested local authoring and integration. No independent ownership, license or redistribution permission for referenced designs is asserted."},
                "provenance": {"tools": [{"name": "Blender", "version": read_json(source / "blender-build.json")["blender"]},
                                           {"name": "avatar-assets", "version": "0.1.0", "source_manifest_sha256": sha256(source / "manifest.json")}],
                    "prompt_history": [scene["prompt"]],
                    "references": manifest.get("references", []),
                    "assumptions": scene["authored_assumptions"] + [
                        "Neutral frame matrices and Z-up/-Y-front orientation are a user-authorized authoring proposal, NOT observed Weft state or metric calibration.",
                        "Visual alignment with Weft remains unverified. Recompute offsets using its actual neutral frame matrices before claiming aligned integration.",
                        "Weft has 19 joint angles plus root translation. Unmapped head, hands, fingers and accessories do not acquire independent simulated anatomy.",
                        "Authored clips are provenance; they must not override simulated pose."],
                    "neutral_proposal_sha256": sha256(out / "neutral-proposal.json")},
                "retargeting": {"profile": "weft-avatar-retarget-v1", "root_node": names[root_name],
                                "scale": scale, "orientation_xyzw": orient, "bindings": result_bindings}}
    write_json(out / "mapping.json", metadata)
    write_json(out / "node-table.json", [{"node": i, "name": node.get("name"), "parent": parents.get(i),
                "is_skin_joint": i in skin["joints"], "global_rest_column_major": world[i].flatten(order="F").tolist(),
                "mapped_body": bindings.get(node.get("name")),
                "unmapped_behavior": None if node.get("name") in bindings else "Keep authored local rest transform"}
               for i, node in enumerate(nodes)])
    result = {"schema_version": 1, "kind": "weft_authored_model_and_mapping", "status": "provisional_alignment",
              "source_manifest_sha256": sha256(source / "manifest.json"), "source_pack": str(source),
              "target_runtime_verified": False, "frozen_importer_verified": False,
              "proposal_equation_max_residual": max(residuals), "bind_inverse_max_residual": max(bind_residuals),
              "geometry": "Real GLB skinned triangle mesh; authored, not recovered reference geometry",
              "artifacts": artifacts(out)}
    write_json(out / "manifest.json", result)
    return result
