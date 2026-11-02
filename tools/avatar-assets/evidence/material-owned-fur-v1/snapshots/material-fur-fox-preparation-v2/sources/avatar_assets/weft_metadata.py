"""Author consumer metadata; Weft's frozen importer remains the sole integration validator."""

import math
import re
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


def _rigid_frames(frames, *, captured=False):
    require(isinstance(frames, dict) and set(frames) == FRAMES, "Neutral data must define all ten frames")
    result = {}
    for name, frame in frames.items():
        require(isinstance(frame, dict), "Invalid neutral frame")
        translation = np.asarray(frame.get("translation"), dtype=float)
        rotation = np.asarray(frame.get("rotation_xyzw"), dtype=float)
        require(translation.shape == (3,) and np.isfinite(translation).all() and np.max(np.abs(translation)) <= 100,
                "Invalid neutral translation")
        require(rotation.shape == (4,) and np.isfinite(rotation).all() and abs(np.linalg.norm(rotation)-1) < 1e-6,
                "Invalid neutral quaternion")
        matrix = np.asarray(_local_matrix({"translation": translation.tolist(), "rotation": rotation.tolist()})).reshape(4, 4, order="F")
        if captured:
            native = np.asarray(frame.get("matrix_column_major"), dtype=float)
            require(native.shape == (16,) and np.isfinite(native).all()
                    and np.allclose(native.reshape(4, 4, order="F"), matrix, atol=1e-8, rtol=0),
                    "Captured matrix disagrees with its transform")
        result[name] = matrix
    return result


def _alignment_context(neutral, design, names, world):
    captured = neutral.get("status") == "weft_runtime_frame_capture"
    if not captured:
        require(neutral.get("status") == "authored_proposal_not_weft_measurement",
                "Expected an authored proposal or verified runtime frame capture")
        scale, orient = neutral["scale"], neutral["orientation_xyzw"]
        frames = _rigid_frames(neutral["frames"])
        anchor = None
    else:
        require(neutral.get("schema_version") == 1 and isinstance(neutral.get("scene_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", neutral["scene_sha256"]), "Missing captured scene identity")
        calibration = neutral.get("calibration", {})
        require(calibration.get("kind") == "zero_angle_root_zero_not_lab_reset", "Unsupported calibration posture")
        angles = calibration.get("angles")
        joint_names = neutral.get("joint_names")
        require(isinstance(joint_names, list) and all(isinstance(name, str) for name in joint_names)
                and len(joint_names) == len(set(joint_names)) == 19
                and isinstance(angles, dict) and set(angles) == set(joint_names)
                and all(type(value) in (int, float) and value == 0 for value in angles.values())
                and isinstance(calibration.get("root_translation"), list)
                and len(calibration["root_translation"]) == 3
                and all(type(value) in (int, float) and value == 0 for value in calibration["root_translation"]),
                "Calibration must be explicitly zero angle/root")
        parity = neutral.get("parity", {})
        error = parity.get("max_abs_matrix_error")
        require(parity.get("status") == "passed" and type(error) in (int, float)
                and math.isfinite(error) and 0 <= error <= 1e-8, "Native/consumer frame parity is not verified")
        frames = _rigid_frames(calibration.get("frames"), captured=True)
        alignment = design.get("alignment")
        require(isinstance(alignment, dict) and set(alignment) == {
            "scale", "source_to_target_body_xyzw", "source_anchor_node", "target_anchor_body", "rationale"},
            "Captured frames require an explicit visual alignment design")
        require(isinstance(alignment["rationale"], str) and 1 <= len(alignment["rationale"]) <= 2000,
                "Visual alignment must disclose its rationale")
        scale = alignment["scale"]
        body = np.asarray(neutral.get("target_body_rotation3x3"), dtype=float)
        require(body.shape == (3, 3) and np.isfinite(body).all()
                and np.allclose(body.T @ body, np.eye(3), atol=1e-6, rtol=0)
                and abs(np.linalg.det(body)-1) < 1e-6, "Invalid captured body orientation")
        local = np.asarray(alignment["source_to_target_body_xyzw"], dtype=float)
        require(local.shape == (4,) and np.isfinite(local).all() and abs(np.linalg.norm(local)-1) < 1e-6,
                "Invalid authored source-to-body orientation")
        local_matrix = np.asarray(_local_matrix({"rotation": local.tolist()})).reshape(4, 4, order="F")
        orient = matrix_quaternion(body @ local_matrix[:3, :3])
        source_anchor, target_anchor = alignment["source_anchor_node"], alignment["target_anchor_body"]
        require(isinstance(source_anchor, str) and isinstance(target_anchor, str)
                and source_anchor in names and target_anchor in FRAMES, "Invalid visual anchor")
        anchor = (source_anchor, target_anchor)
    require(type(scale) in (int, float) and math.isfinite(scale) and 0 < scale <= 1000, "Invalid scale")
    orientation = np.asarray(orient, dtype=float)
    require(orientation.shape == (4,) and np.isfinite(orientation).all()
            and abs(np.linalg.norm(orientation)-1) < 1e-6, "Invalid alignment orientation")
    basis = np.asarray(_local_matrix({"rotation": orient, "scale": [scale]*3})).reshape(4, 4, order="F")
    model_alignment = basis.copy()
    if anchor:
        model_alignment[:3, 3] = frames[anchor[1]][:3, 3] - (basis @ world[names[anchor[0]]])[:3, 3]
    return captured, frames, scale, orient, basis, model_alignment


def create_metadata(folder, design_path, neutral_path, output):
    from avatar_assets.cli import verify_pack
    verify_pack(folder)
    source = Path(folder).resolve()
    manifest = read_json(source / "manifest.json")
    require(manifest.get("output_profile") == "weft-v1", "Build a separate explicit weft-v1 material variant first")
    scene = read_json(source / "sources/scene.json")
    design, neutral = read_json(design_path), read_json(neutral_path)
    if neutral.get("status") == "weft_runtime_frame_capture":
        verify_pack(Path(neutral_path).resolve().parent)
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
    captured, proposed, scale, orient, G, M = _alignment_context(neutral, design, names, world)
    result_bindings, residuals, offsets = [], [], []
    for name, body in bindings.items():
        i = names[name]
        C = np.linalg.inv(proposed[body]) @ M @ world[i] @ np.linalg.inv(G)
        require(np.allclose(C[:3, :3].T @ C[:3, :3], np.eye(3), atol=1e-6)
                and np.linalg.det(C[:3, :3]) > .99999, "Binding has unrepresentable scale/shear")
        quaternion = matrix_quaternion(C[:3, :3])
        result_bindings.append({"node": i, "body": body, "translation": C[:3, 3].tolist(), "rotation_xyzw": quaternion})
        reconstructed = proposed[body] @ np.asarray(_local_matrix({"translation": C[:3, 3].tolist(), "rotation": quaternion})).reshape(4, 4, order="F") @ G
        residuals.append(float(np.max(np.abs(reconstructed - M @ world[i]))))
        offsets.append({"node": i, "body": body,
                        "pivot_offset_world_units": float(np.linalg.norm((M @ world[i])[:3, 3] - proposed[body][:3, 3]))})
    require(max(residuals) < 1e-6, "Proposed-neutral equation verification failed")
    skin = model.doc["skins"][0]
    ibms = model.accessor(skin["inverseBindMatrices"], "MAT4", (5126,))
    bind_residuals = [float(np.max(np.abs(world[j] @ np.asarray(m).reshape(4, 4, order="F") - np.eye(4))))
                      for j, m in zip(skin["joints"], ibms)]
    require(max(bind_residuals) < 1e-5, "Authored world bind and inverse bind differ")
    out = new_directory(output)
    shutil.copyfile(source / "model.glb", out / "model.glb")
    neutral_name = "runtime-frames.json" if captured else "neutral-proposal.json"
    shutil.copyfile(neutral_path, out / neutral_name)
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
                    "neutral_proposal_sha256": sha256(out / neutral_name)},
                "retargeting": {"profile": "weft-avatar-retarget-v1", "root_node": names[root_name],
                                "scale": scale, "orientation_xyzw": orient, "bindings": result_bindings}}
    if captured:
        provenance = metadata["provenance"]
        provenance["runtime_frame_capture_sha256"] = provenance.pop("neutral_proposal_sha256")
        provenance["target_scene_sha256"] = neutral["scene_sha256"]
        provenance["alignment_design"] = design["alignment"]
        provenance["assumptions"] = scene["authored_assumptions"] + [
            "Calibration uses actual simulator zero-angle/root-zero frames, not the lab's reset pose or measured anatomy.",
            "Uniform visual scale, facing and anchor are authored choices; simulation units are not calibrated physical meters.",
            "Calibration reproduces the authored bind shape; mismatched bone pivots/proportions still require articulated visual review.",
            "Only the existing ten body frames drive visuals. No independent head, fingers or dragon simulation is added.",
            "Authored clips are provenance; they do not replace simulated poses."]
        write_json(out / "alignment.json", {"schema_version": 1, "status": "calibrated_visual_review_required",
            "target_scene_sha256": neutral["scene_sha256"], "model_alignment_column_major": M.flatten(order="F").tolist(),
            "binding_equation": "C = inverse(F0) * M * B_world * inverse(G); J = F_body * C * G",
            "native_consumer_parity": neutral["parity"], "pivot_offsets": offsets,
            "scope": "Bind-shape calibration is not joint-center coincidence, metric calibration or motion-quality acceptance."})
    write_json(out / "mapping.json", metadata)
    write_json(out / "node-table.json", [{"node": i, "name": node.get("name"), "parent": parents.get(i),
                "is_skin_joint": i in skin["joints"], "global_rest_column_major": world[i].flatten(order="F").tolist(),
                "mapped_body": bindings.get(node.get("name")),
                "unmapped_behavior": None if node.get("name") in bindings else "Keep authored local rest transform"}
               for i, node in enumerate(nodes)])
    result = {"schema_version": 1, "kind": "weft_authored_model_and_mapping",
              "status": "calibrated_visual_review_required" if captured else "provisional_alignment",
              "source_manifest_sha256": sha256(source / "manifest.json"), "source_pack": str(source),
              "target_runtime_verified": False, "frozen_importer_verified": False,
              "proposal_equation_max_residual": max(residuals), "bind_inverse_max_residual": max(bind_residuals),
              "geometry": "Real GLB skinned triangle mesh; authored, not recovered reference geometry",
              "artifacts": artifacts(out)}
    write_json(out / "manifest.json", result)
    return result
