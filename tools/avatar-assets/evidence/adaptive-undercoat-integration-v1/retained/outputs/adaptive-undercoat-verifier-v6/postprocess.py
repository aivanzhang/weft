"""One CPU-only correction after exact exporter quantization diagnosis; preserve the failed native attempt."""

import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources/host"))
from avatar_assets.generated_tangents import repair, NORMAL_UV1_FRAME_POLICY
from avatar_assets.safety import require, sha256, write_json
from avatar_assets.three_d import _GLB
from verify_fixture import verify


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
    prior_v5 = json.loads((HERE / 'source-v5-inventory.json').read_text())
    for row in prior_v5['files']:
        require(sha256(Path(prior_v5['source']) / row['path']) == row['sha256'], 'Original v5 artifact changed')
    inventory = json.loads((HERE / 'source-v4-inventory.json').read_text())
    for row in inventory['files']:
        require(sha256(Path(inventory['source']) / row['path']) == row['sha256'], 'Original v4 artifact changed')
    native = json.loads((HERE / "source-v4/native-fixture.json").read_text())
    require(native["frame_policy"] == NORMAL_UV1_FRAME_POLICY, "Native bake policy mismatch")
    require(native["undercoat"]["frame_policy"] == NORMAL_UV1_FRAME_POLICY, "Native adapter policy mismatch")
    require(native["undercoat"]["plan"]["frame_policy"] == NORMAL_UV1_FRAME_POLICY, "Native planner policy mismatch")
    require(all(row['frame_policy'] == (NORMAL_UV1_FRAME_POLICY if row['normal_texcoord'] == 1 else None)
                for row in native['undercoat']['materials']), 'Native material bake policy mismatch')
    source, target = HERE / "source-v4/model-native-export.glb", HERE / "model.glb"
    require(native["model_sha256"] == sha256(source), "Raw native model changed")
    raw = verify(source, sha256(source))
    write_json(HERE / "raw-verification.json", raw)
    require(raw['normal_quantization']['exact_installed_exporter_transform_matched_all_corners'],
            'Exact installed-exporter quantization does not explain all normals; do not correct the model')
    write_json(HERE / 'normal-quantization-diagnosis.json', {
        'schema_version': 1, 'source_sha256': sha256(source),
        'previous_host_failure': 'V4 used2e-6 component preservation, but exporter rounds float32 to4 decimals and renormalizes; V5 proved exact numeric matching but rejected10 signed-zero bit differences.',
        'all_six_faces_measured': True, 'summary': raw['normal_quantization'],
        'faces': [{'source_face': row['source_face'], 'case': row['case'], **row['normal_quantization']} for row in raw['triangles']],
        'tangent_and_rgb_dot_threshold_unchanged': .99999, 'native_jobs': 0})
    require(raw["passed"] is False, "Fixture did not expose the unchanged exporter active-UV mismatch")
    report = repair(source, target, correct_normal_uv1=True, normal_uv1_frame_policy=native["frame_policy"])
    write_json(HERE / "tangent-repair.json", report)
    preservation = byte_proof(source, target)
    write_json(HERE / "byte-preservation.json", preservation)
    result = verify(target, sha256(target))
    write_json(HERE / "host-verification.json", result)
    require(result["passed"], "Corrected export differs from actual Blender Mikk bake frames")
    for row in inventory['files']:
        require(sha256(Path(inventory['source']) / row['path']) == row['sha256'], 'Original v4 artifact changed during proof')
    for row in prior_v5['files']:
        require(sha256(Path(prior_v5['source']) / row['path']) == row['sha256'], 'Original v5 artifact changed during proof')
    write_json(HERE / "result.json", {"schema_version": 6, "status": "analytic_native_mikk_fixture_passed",
        "frame_policy": NORMAL_UV1_FRAME_POLICY, "model_sha256": sha256(target), "raw_model_sha256": sha256(source),
        "baker_vs_actual_blender_mikk_verified": True, "new_png_world_normal_vectors_verified": True,
        "nonselected_bytes_preserved": True, "quality_approved": False, "avatar_pack_accepted": False,
        "renders": 0, "provider_calls": 0, 'new_native_jobs': 0, 'original_v4_files_preserved': len(inventory['files']),
        'original_v5_files_preserved': len(prior_v5['files']),
        'normal_quantization_diagnosis_sha256': sha256(HERE / 'normal-quantization-diagnosis.json'),
        "scope": "Six analytic triangles only; custom-normal/new-bake frame proof, not rendered avatar quality."})
    print(json.dumps({"passed": True, "model_sha256": sha256(target), "changed_bytes": preservation["changed_bytes"]}))


if __name__ == "__main__":
    main()
