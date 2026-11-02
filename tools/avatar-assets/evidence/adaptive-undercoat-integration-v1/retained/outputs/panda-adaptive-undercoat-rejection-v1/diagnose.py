"""Read-only diagnosis of selected-UV tangent parity; never imports the native scene."""

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "sources/host"))
sys.path.insert(0, str(HERE / "source-snapshot/sources"))
from avatar_assets.safety import require
from avatar_assets.three_d import _GLB
from undercoat_uv_atlas import atlas_tangents


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def unchanged(inventory):
    root = Path(inventory["root"])
    for row in inventory["files"]:
        require(sha(root / row["path"]) == row["sha256"], "Original failed artifact changed: " + row["path"])


def read_array(glb, index, kind, types):
    require(type(index) is int and 0 <= index < len(glb.doc["accessors"]), "Invalid diagnostic accessor")
    count = glb.doc["accessors"][index].get("count")
    require(type(count) is int and 1 <= count <= 6144, "UV1 diagnostic accessor exceeds 2048-face bound")
    return np.asarray(glb.accessor(index, kind, types), dtype=np.float64 if 5126 in types else np.int32)


def main():
    started = time.monotonic()
    prepared = json.loads((HERE / "prepared.json").read_text())
    for row in prepared["frozen_files"]:
        require(sha(HERE / row["path"]) == row["sha256"], "Frozen diagnostic source changed")
    inventory = json.loads((HERE / "source-inventory.json").read_text())
    unchanged(inventory)
    source = Path(inventory["root"]) / "model-native-export.glb"
    require(sha(source) == prepared["model_sha256"], "Native GLB hash differs from prepared source")
    glb = _GLB(source)
    finish = json.loads((HERE / "source-snapshot/finish-report.json").read_text())
    reports = {row["material"]: row for row in finish["fur_undercoat"]["materials"]}
    regions, witnesses = [], []
    for mesh_index, mesh in enumerate(glb.doc["meshes"]):
        for primitive_index, primitive in enumerate(mesh["primitives"]):
            material = glb.doc["materials"][primitive["material"]]
            if material.get("normalTexture", {}).get("texCoord", 0) != 1:
                continue
            require(len(regions) < 4, "UV1 diagnostic exceeds four regions")
            attrs = primitive["attributes"]
            points = read_array(glb, attrs["POSITION"], "VEC3", (5126,))
            normals = read_array(glb, attrs["NORMAL"], "VEC3", (5126,))
            uv_export = read_array(glb, attrs["TEXCOORD_1"], "VEC2", (5126,))
            faces = read_array(glb, primitive["indices"], "SCALAR", (5121, 5123, 5125)).reshape(-1, 3)
            require(len(points) == len(normals) == len(uv_export) and 1 <= len(faces) <= 2048
                    and faces.min() >= 0 and faces.max() < len(points), "Invalid UV1 diagnostic topology")
            xyz = points[faces]
            original_n = normals[faces]
            lengths = np.linalg.norm(original_n, axis=2)
            require(np.all(np.abs(lengths - 1) < .001), "Exported corner normals are not unit")
            n = original_n / lengths[:, :, None]
            uv_up = uv_export.copy()
            uv_up[:, 1] = 1 - uv_up[:, 1]
            chart = uv_up[faces]
            e1, e2 = xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]
            area_vector = np.cross(e1, e2)
            area2 = np.linalg.norm(area_vector, axis=1)
            require(np.all(area2 > 1e-12), "Selected geometry contains degenerate triangles")
            face_n = area_vector / area2[:, None]
            a, b = chart[:, 1] - chart[:, 0], chart[:, 2] - chart[:, 0]
            det = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            require(np.all(np.abs(det) > 1e-12), "Selected UV1 chart is degenerate")
            du = (e1 * b[:, 1, None] - e2 * a[:, 1, None]) / det[:, None]
            dv = (e2 * a[:, 0, None] - e1 * b[:, 0, None]) / det[:, None]
            tangent = du[:, None] - n * np.sum(n * du[:, None], axis=2)[:, :, None]
            tangent_lengths = np.linalg.norm(tangent, axis=2)
            require(np.all(tangent_lengths > 1e-8), "Source normal cannot support selected tangent")
            tangent /= tangent_lengths[:, :, None]
            orientation = np.sum(np.cross(n, tangent) * dv[:, None], axis=2)
            signs = np.sign(orientation)
            require(np.all(np.abs(signs) == 1), "Undefined tangent sign")
            dots = np.sum(n * face_n[:, None], axis=2)
            expected_signs = np.sign(dots) * np.sign(det)[:, None]
            require(np.array_equal(signs, expected_signs), "Per-corner sign/normal-hemisphere identity failed")
            mixed = np.any(signs != signs[:, :1], axis=1)
            straddling = (dots.min(axis=1) < 0) & (dots.max(axis=1) > 0)
            require(np.array_equal(mixed, straddling), "Mixed signs are not explained by normal hemispheres")
            native_frames = atlas_tangents(points, faces, np.arange(len(faces), dtype=np.int32), chart, original_n)
            require(np.array_equal(native_frames[:, :, 3], signs), "Preserved native atlas formula differs")
            payload = HERE / ("uv1-region-" + str(len(regions)) + ".npz")
            require(not payload.exists(), "Diagnostic payload already exists")
            np.savez_compressed(payload, points=points, faces=faces,
                face_ids=np.arange(len(faces), dtype=np.int32), uv1=chart, corner_normals=original_n)
            require(payload.stat().st_size <= 1024 * 1024, "Diagnostic payload exceeds 1MiB")
            info = reports[material["name"]]
            region = {"mesh": mesh_index, "primitive": primitive_index, "material": material["name"],
                "source_material": info["source_material"], "triangles": len(faces), "exported_vertices": len(points),
                "mixed_handedness_faces": int(mixed.sum()), "mixed_face_indices": np.flatnonzero(mixed).tolist(),
                "all_corner_normals_against_winding_faces": int(np.all(dots < 0, axis=1).sum()),
                "all_corner_normals_with_winding_faces": int(np.all(dots > 0, axis=1).sum()),
                "opposed_normal_corners": int((dots < 0).sum()),
                "near_tangent_normal_corners_abs_dot_below_1e_6": int((np.abs(dots) < 1e-6).sum()),
                "corner_normal_dot_geometric_normal_min_max": [float(dots.min()), float(dots.max())],
                "uv_determinant_min_max": [float(det.min()), float(det.max())],
                "triangle_area_m2_min_max": [float(area2.min() / 2), float(area2.max() / 2)],
                "minimum_tangent_projection_length": float(tangent_lengths.min()),
                "normal_hemisphere_exactly_explains_mixed_faces": True,
                "preserved_native_atlas_formula_reproduces_signs": True,
                "uniform_v_flip_still_mixed_faces": int(np.any(-signs != -signs[:, :1], axis=1).sum()),
                "native_bake_authored_tangent_hash": info["authored_tangents_sha256"],
                "replay": {"path": payload.name, "sha256": sha(payload), "bytes": payload.stat().st_size,
                    "coordinate_space": "GLB mesh-local XYZ; UV1 restored to native V-up; corner normals from exported NORMAL",
                    "call": "atlas_tangents(points, faces, face_ids, uv1, corner_normals)",
                    "scope": "Actual exported region, not byte-identical pre-export native arrays; V round-trip and exporter ordering may differ."}}
            regions.append(region)
            for face in np.flatnonzero(mixed):
                witnesses.append({"mesh": mesh_index, "primitive": primitive_index, "material": material["name"],
                    "triangle_index_in_primitive": int(face), "exported_vertex_indices": faces[face].tolist(),
                    "position_mesh_local_gltf_m": xyz[face].tolist(), "normal_mesh_local_gltf": n[face].tolist(),
                    "geometric_face_normal_mesh_local_gltf": face_n[face].tolist(),
                    "uv1_export_v_down": uv_export[faces[face]].tolist(), "uv1_native_bake_v_up": chart[face].tolist(),
                    "dPdu_mesh_local_gltf": du[face].tolist(), "dPdv_mesh_local_gltf": dv[face].tolist(),
                    "uv_determinant": float(det[face]), "triangle_area_m2": float(area2[face] / 2),
                    "corner_normal_dot_geometric_face_normal": dots[face].tolist(),
                    "projected_tangent_mesh_local_gltf": tangent[face].tolist(),
                    "cross_N_T_dot_dPdv": orientation[face].tolist(), "corner_handedness": signs[face].tolist(),
                    "minimum_corner_normal_dot_face": float(dots[face].min())})
    require(regions, "No UV1 region present")
    witnesses.sort(key=lambda row: (row["minimum_corner_normal_dot_face"], row["mesh"], row["primitive"], row["triangle_index_in_primitive"]))
    unchanged(inventory)
    report = {"schema_version": 1, "status": "mixed_handedness_diagnosed_no_repair", "model_sha256": glb.hash,
        "source_file": str(source), "finish_report_sha256": sha(HERE / "source-snapshot/finish-report.json"),
        "native_atlas_source_sha256": sha(HERE / "source-snapshot/sources/undercoat_uv_atlas.py"),
        "strict_export_helper_sha256": sha(HERE / "source-snapshot/sources/generated_tangents.py"),
        "diagnostic_script_sha256": sha(Path(__file__)), "regions": regions, "worst_witnesses": witnesses[:8],
        "witness_selection": "At most8 mixed faces, ordered by most negative corner N dot geometric face N, then stable mesh/primitive/face indices.",
        "total_uv1_faces": sum(row["triangles"] for row in regions),
        "total_mixed_faces": sum(row["mixed_handedness_faces"] for row in regions),
        "preserved_failed_files": len(inventory["files"]), "all_original_files_unchanged": True,
        "normal_continuity_context": finish.get("normal_continuity"),
        "interpretation": [
            "For the current projected tangent, sign(cross(N,T) dot dPdv) = sign(N dot geometric face normal) times sign(UV determinant).",
            "Mixed per-corner signs therefore identify normals on opposite sides of a triangle plane; a uniform V-axis sign change cannot fix them.",
            "The native bake uses this per-corner sign formula, whereas a per-triangle UV orientation is a different frame policy. Changing only exported W would reinterpret unchanged PNG vectors.",
            "Transferred cap shading normals and cap geometry are not mutually hemisphere-coherent under this strict authored frame profile. This does not prove the GLB is universally invalid or identify a measured physical surface.",
            "The preserved normal-continuity report is context, not a before/after causal proof for each exported corner. No source.blend execution or native inspection occurred."
        ],
        "next_safe_action": "Reject this unsupported region before bake/render with the same consistency guard. Any future geometry/normal/frame-policy repair needs new bounded source-bound evidence and a new bake; do not force W after the current bake.",
        "native_jobs": 0, "provider_calls": 0, "model_calls": 0, "quality_approved": False,
        "elapsed_seconds": time.monotonic() - started}
    write(HERE / "diagnosis.json", report)
    print(json.dumps({"status": report["status"], "uv1_faces": report["total_uv1_faces"],
                      "mixed_faces": report["total_mixed_faces"], "witnesses": len(report["worst_witnesses"])}))


if __name__ == "__main__":
    main()
