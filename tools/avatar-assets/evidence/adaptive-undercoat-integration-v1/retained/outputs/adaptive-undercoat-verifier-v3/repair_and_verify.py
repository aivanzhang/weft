"""One CPU-only derivative of the preserved native fixture, with independent byte/frame proof."""

import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources/host"))
from avatar_assets.generated_tangents import repair
from avatar_assets.safety import require, sha256
from avatar_assets.three_d import _GLB
from verify_fixture_v3 import verify


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def unchanged_prior_sources():
    inventory = json.loads((HERE / "source-v2-inventory.json").read_text())
    source = Path(inventory["source"])
    for row in inventory["files"]:
        require(sha256(source / row["path"]) == row["sha256"], "Original v2 artifact changed")
    prior = json.loads((source / "previous-v1/complete-v1-inventory.json").read_text())
    prior_files = prior.get("files", prior.get("artifacts")) if isinstance(prior, dict) else prior
    source_v1 = source.parent / "adaptive-undercoat-native-fixture-v1"
    for row in prior_files:
        require(sha256(source_v1 / row["path"]) == row["sha256"], "Original v1 artifact changed")
    return {"v1_files": len(prior_files), "v2_files": len(inventory["files"]), "all_unchanged": True}


def byte_proof(source, target):
    original, result = source.read_bytes(), target.read_bytes()
    require(len(original) == len(result), "Correction changed GLB length")
    before, after = _GLB(source), _GLB(target)
    require(before.doc == after.doc, "Correction changed scene/material/accessor JSON")
    binary_start = 28 + struct.unpack_from("<I", original, 12)[0]
    allowed = np.zeros(len(original), dtype=bool)
    ranges, untouched_tangents = [], []
    for mesh_index, mesh in enumerate(before.doc["meshes"]):
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            accessor = before.doc["accessors"][primitive["attributes"]["TANGENT"]]
            view = before.doc["bufferViews"][accessor["bufferView"]]
            start = binary_start + view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
            length = accessor["count"] * 16
            require(view.get("byteStride", 16) == 16, "Fixture tangent view is not contiguous")
            normal = before.doc["materials"][primitive["material"]].get("normalTexture", {})
            row = {"mesh": mesh_index, "primitive": primitive_index, "file_offset": start, "bytes": length,
                   "normal_texcoord": normal.get("texCoord", 0) if normal else None,
                   "before_sha256": hashlib.sha256(original[start:start + length]).hexdigest(),
                   "after_sha256": hashlib.sha256(result[start:start + length]).hexdigest()}
            if normal.get("texCoord", 0) == 1:
                allowed[start:start + length] = True
                ranges.append(row)
            else:
                require(original[start:start + length] == result[start:start + length], "Healthy/excluded tangent changed")
                untouched_tangents.append(row)
    before_bytes, after_bytes = np.frombuffer(original, dtype=np.uint8), np.frombuffer(result, dtype=np.uint8)
    changed = before_bytes != after_bytes
    require(np.any(changed) and not np.any(changed & ~allowed), "Correction changed bytes outside UV1 tangents")
    unchanged_hash = hashlib.sha256(before_bytes[~allowed].tobytes()).hexdigest()
    require(unchanged_hash == hashlib.sha256(after_bytes[~allowed].tobytes()).hexdigest(), "Non-tangent hash changed")
    return {"source_sha256": before.hash, "result_sha256": after.hash,
            "changed_bytes": int(changed.sum()), "allowed_tangent_bytes": int(allowed.sum()),
            "unchanged_nonselected_bytes": int((~allowed).sum()), "nonselected_bytes_sha256": unchanged_hash,
            "only_uv1_tangent_float4_storage_changed": True, "scene_json_exact": True,
            "geometry_all_uvs_normals_materials_images_exact": True,
            "healthy_uv0_and_excluded_tangents_exact": True, "selected_ranges": ranges,
            "untouched_tangent_ranges": untouched_tangents,
            "source_skins": len(before.doc.get("skins", [])), "source_animations": len(before.doc.get("animations", [])),
            "weight_and_motion_scope": "Fixture contains no skin weights or actions; none added. All bytes outside UV1 tangent storage remain exact."}


def main():
    prepared = json.loads((HERE / "repair-prepared.json").read_text())
    for row in prepared["frozen_files"]:
        require(sha256(HERE / row["path"]) == row["sha256"], "Frozen CPU input changed: " + row["path"])
    require(sha256(HERE / "sources/host/avatar_assets/generated_tangents.py") == prepared["repair_helper_sha256"],
            "Repair helper hash changed")
    prior = unchanged_prior_sources()
    raw = json.loads((HERE / "raw-diagnosis.json").read_text())
    require(raw["passed"] is False and len(raw["triangles"]) == 4, "Raw rejection evidence is missing")
    source = HERE / "source-v2/model.glb"
    output = HERE / "derived"
    output.mkdir()
    target = output / "model.glb"
    report = repair(source, target, correct_normal_uv1=True)
    write(output / "repair-report.json", report)
    preservation = byte_proof(source, target)
    write(output / "byte-preservation.json", preservation)
    verified = verify(target, sha256(target))
    write(output / "verification.json", verified)
    prior_after = unchanged_prior_sources()
    require(prior == prior_after, "Prior inventory changed during verification")
    require(verified["passed"], "Derived selected-UV native bake-frame proof failed")
    write(output / "result.json", {"status": "analytic_uv1_derivative_verified", "passed": True,
        "source_sha256": sha256(source), "model_sha256": sha256(target),
        "repair_helper_sha256": prepared["repair_helper_sha256"], "verifier_sha256": verified["verifier_sha256"],
        "prior_preservation": prior_after, "raw_diagnosis_sha256": sha256(HERE / "raw-diagnosis.json"),
        "byte_proof_sha256": sha256(output / "byte-preservation.json"),
        "verification_sha256": sha256(output / "verification.json"),
        "native_jobs": 0, "renders": 0, "provider_calls": 0, "quality_approved": False,
        "scope": "Four-triangle analytic tangent export proof only, not an accepted avatar or visual-quality claim."})
    print(json.dumps({"status": "analytic_uv1_derivative_verified", "model_sha256": sha256(target),
                      "changed_bytes": preservation["changed_bytes"], "native_jobs": 0}))


if __name__ == "__main__":
    main()
