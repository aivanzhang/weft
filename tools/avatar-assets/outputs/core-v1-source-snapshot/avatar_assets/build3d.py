"""Compile authored scene data, preserving editable sources and proof separately."""

import json
import shutil
import struct
from pathlib import Path

from PIL import Image

from avatar_assets.media import contact_sheet, image_info, tools_info
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, new_directory, preserve_failure, read_json, require, sha256, write_json
from avatar_assets.scene import validate_scene
from avatar_assets.three_d import validate_glb

ROOT = Path(__file__).resolve().parent.parent


def glb_document(path):
    with Path(path).open("rb") as h:
        header = h.read(20)
        require(header[:4] == b"glTF", "Bad GLB magic")
        length = struct.unpack_from("<I", header, 12)[0]
        require(length < 2 * 1024 * 1024, "GLB JSON limit exceeded")
        return json.loads(h.read(length))


def contract_for(scene, glb):
    doc = glb_document(glb)
    joints = sorted({doc["nodes"][i]["name"] for skin in doc["skins"] for i in skin["joints"]})
    nodes = {n["name"]: n for n in doc["nodes"] if "name" in n}
    attachments = []
    for design in scene.get("attachments", []):
        child = nodes[design["child_bone"]]
        attachments.append({"node": design["child_bone"], "parent_bone": design["parent_bone"],
                            "mode": "rigid_child", "translation_m": child.get("translation", [0, 0, 0]),
                            "rotation_xyzw": child.get("rotation", [0, 0, 0, 1]),
                            "placement": design["placement"]})
    return {"schema_version": 1, "units": "meters",
            "axes": {"up": "+Y", "forward": "+Z", "handedness": "right"},
            "rest_pose": "Authored default bone transforms. Blender Z-up/-Y-front exported to glTF Y-up/+Z-front.",
            "bone_to_simulated_joint": {j: j for j in joints},
            "target": {"name": "Local authored bone controls v1; NOT a Weft joint schema", "joint_names": joints},
            "authored_assumptions": scene["authored_assumptions"], "attachments": attachments,
            "attachment_design": scene.get("attachments", []),
            "weft_compatibility": "Not evaluated by this local-control contract; use separate Weft metadata and its frozen importer"}


def build_3d(config_path, output, blender=None, resolution=384, samples=8, timeout=300):
    existed = Path(output).exists()
    try:
        return _build_3d(config_path, output, blender, resolution, samples, timeout)
    except Exception as error:
        if not existed:
            preserve_failure(output, error)
        raise


def _build_3d(config_path, output, blender, resolution, samples, timeout):
    scene = validate_scene(read_json(config_path))
    require(type(resolution) is int and 128 <= resolution <= 1024 and resolution % 2 == 0, "Invalid render resolution")
    require(type(samples) is int and 1 <= samples <= 64, "Samples outside 1–64")
    require(30 <= timeout <= 1800, "Timeout outside 30–1800 seconds")
    exe = Path(blender) if blender else ROOT / ".tools/Blender.app/Contents/MacOS/Blender"
    require(exe.is_file(), "Blender not found; pass --blender PATH")
    out = new_directory(output)
    sources = out / "sources"
    sources.mkdir()
    write_json(sources / "scene.json", scene)
    shutil.copyfile(ROOT / "scripts/blender_build.py", sources / "blender_build.py")
    (sources / "prompt.txt").write_text(scene["prompt"] + "\n")
    references = []
    for index, reference in enumerate(scene.get("references", [])):
        require(index < 4, "At most four references")
        path = Path(config_path).resolve().parent / reference["path"]
        info = image_info(path)
        require(info["sha256"] == reference["sha256"], "Reference hash differs from declared input")
        require(isinstance(reference.get("role"), str) and reference["role"], "Reference role required")
        destination = sources / (f"reference-{index}" + path.suffix.lower())
        shutil.copyfile(path, destination)
        references.append({"path": str(destination.relative_to(out)), "role": reference["role"], "inspection": info})
    command = [str(exe.resolve()), "--background", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1",
               "--threads", "4", "--python", str(sources / "blender_build.py"), "--",
               str(sources / "scene.json"), str(out), str(resolution), str(samples)]
    job = run_bounded(command, out / "build.log", out, timeout=timeout, memory_mb=2048)
    write_json(out / "job.json", job)
    success = job["exit_code"] == 0 and not job["failure"] and (out / "blender-build.json").exists()
    result = {"schema_version": 1, "kind": "authored_3d_avatar", "id": scene["id"],
              "subject": scene["subject"], "style": scene["style"], "scene_sha256": sha256(sources / "scene.json"),
              "origin": "Agent-authored procedural geometry and skeletal keyframes; not neural reconstruction or an analytic fixture",
              "generation": {"provider_calls": 0, "provider_cost_usd": 0, "recurring_service_cost_usd": 0,
                             "local_compute": "CPU, up to 4 threads; resource usage in job.json"},
              "rights": "New locally authored shapes/materials; reference resemblance is not independent rights verification",
              "states": [{"id": s["id"], "meaning": s["meaning"], "duration_seconds": s["duration"],
                          "loop_requested": s["loop"], "loop_verified": False} for s in scene["states"]],
              "status": "needs_visual_review" if success else "failed", "job": job,
              "tools": tools_info(), "authored_assumptions": scene["authored_assumptions"],
              "output_profile": scene.get("output_profile", "general"), "references": references,
              "rejected_attempts": [] if success else [{"reason": "Backend failed; inspect build.log and job.json", "preserved": True}]}
    if success:
        contract = contract_for(scene, out / "model.glb")
        write_json(out / "rig-contract.json", contract)
        verification = validate_glb(out / "model.glb", contract)
        write_json(out / "rig-verification.json", verification)
        result["rig_verification"] = verification
        result["portrait"] = image_info(out / "portrait.png")
        for s in scene["states"]:
            files = sorted((out / "renders" / s["id"]).glob("*.png"))
            contact_sheet([Image.open(p) for p in files], [p.stem for p in files], out / (s["id"] + "-contact.png"))
    result["artifacts"] = artifacts(out)
    write_json(out / "manifest.json", result)
    require(success, f"3D build failed; preserved output: {out}")
    return result
