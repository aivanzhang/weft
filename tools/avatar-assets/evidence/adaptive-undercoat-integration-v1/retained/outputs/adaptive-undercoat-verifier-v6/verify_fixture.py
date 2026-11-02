"""Exact float32 exporter quantization plus native-Mikk/RGB proof; signed zero is numerically equal."""

from io import BytesIO
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source-v4"
sys.path.insert(0, str(HERE / "sources/host"))
from avatar_assets.safety import require, sha256
from avatar_assets.three_d import _GLB, _world_matrices, validate_render_attributes


def image_pixels(data):
    with Image.open(BytesIO(data)) as image:
        require(image.format == "PNG" and image.width * image.height <= 1024 ** 2
                and getattr(image, "n_frames", 1) == 1, "Unexpected fixture image format/size")
        return np.asarray(image.convert("RGBA"))


def frame(points, uv, normal):
    edge1, edge2 = points[1] - points[0], points[2] - points[0]
    delta1, delta2 = uv[1] - uv[0], uv[2] - uv[0]
    determinant = delta1[0] * delta2[1] - delta1[1] * delta2[0]
    require(abs(determinant) > 1e-10, "Fixture UV triangle is degenerate")
    du = (edge1 * delta2[1] - edge2 * delta1[1]) / determinant
    dv = (edge2 * delta1[0] - edge1 * delta2[0]) / determinant
    tangent = du - normal * np.dot(du, normal)
    require(np.linalg.norm(tangent) > 1e-8, "Fixture tangent derivative is degenerate")
    tangent /= np.linalg.norm(tangent)
    sign = float(np.sign(np.dot(np.cross(normal, tangent), dv)))
    require(abs(sign) == 1, "Fixture handedness is undefined")
    return tangent, sign


def unit(value):
    value = np.asarray(value, dtype=float)
    require(np.all(np.isfinite(value)) and np.linalg.norm(value) > 1e-8, "Invalid finite vector")
    return value / np.linalg.norm(value)


def world_normal(rgb, normal, tangent, sign):
    # PNG vectors retain the native V-up bake frame; the exported UV V flip is addressing only.
    vector = unit(np.asarray(rgb, dtype=float) / 255 * 2 - 1)
    bitangent = np.cross(normal, tangent) * sign
    return unit(tangent * vector[0] + bitangent * vector[1] + normal * vector[2])


def blender_to_gltf(value):
    return np.asarray(value)[:, [0, 2, 1]] * np.array([1, 1, -1])


def exported_normal_bytes(native_blender_normals):
    # Installed Blender4.5 exporter: round float32 to4 decimals, normalize in place, then rotate axes.
    values = np.asarray(native_blender_normals, dtype=np.float32)
    values = np.round(values, 4)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    require(np.all(norms > 0), 'Fixture normal unexpectedly rounded to zero')
    np.divide(values, norms, out=values, where=norms != 0)
    values[:, [1, 2]] = values[:, [2, 1]]
    values[:, 2] *= -1
    return values


def interpolated_frame(normals, frames, barycentric):
    normal = unit(np.asarray(barycentric) @ normals)
    tangent = np.asarray(barycentric) @ frames[:, :3]
    tangent = unit(tangent - normal * np.dot(tangent, normal))
    require(np.all(frames[:, 3] == frames[0, 3]), "Native policy requires per-face handedness")
    return normal, tangent, float(frames[0, 3])


def verify(model, expected_sha):
    started = time.monotonic()
    require(sha256(model) == expected_sha, "Requested model hash binding failed")
    pins = json.loads((HERE / "prepared.json").read_text())
    for row in pins["frozen_files"]:
        require(sha256(HERE / row["path"]) == row["sha256"], "Frozen verifier input changed: " + row["path"])
    native = json.loads((SOURCE / "native-fixture.json").read_text())
    require(native["model_sha256"] == sha256(SOURCE / "model-native-export.glb") and native["original_data_unchanged"],
            "Native source binding failed")
    require(native['frame_policy'] == 'mikk-unshared-triangle-v-up-v2', 'Require an explicitly new-policy bake')
    witness = json.loads((SOURCE / 'native-frame-witness.json').read_text())
    require(sha256(SOURCE / 'native-frame-witness.json') == native['native_frame_witness_sha256']
            and witness['passed'] and witness['uv0_active_render_defaults_restored'], 'Native Mikk witness failed')
    glb = _GLB(model)
    attributes = validate_render_attributes(model)
    doc, nodes = glb.doc, glb.doc.get("nodes", [])
    require(all(type(a.get("count")) is int and 1 <= a["count"] <= 64 for a in doc.get("accessors", [])),
            "Analytic fixture accessor exceeds 64 elements")
    require(len(nodes) <= 4 and len(doc.get("materials", [])) <= 8, "Analytic fixture exceeds scope")
    parents = {child: index for index, node in enumerate(nodes) for child in node.get("children", [])}
    matrices = [np.asarray(value).reshape(4, 4, order="F") for value in _world_matrices(nodes, parents)]
    expected = native["faces"]
    require(len(expected) == 6, "Fixture must contain exactly six source faces")
    sources = {row["material"]: row for row in native["undercoat"]["materials"]}
    source_color = image_pixels((SOURCE / "textures/original-basecolor.png").read_bytes())
    checked_images, cache = set(), {}

    def texture_pixels(info):
        texture = doc["textures"][info["index"]]
        image_id = texture["source"]
        if image_id not in cache:
            image = doc["images"][image_id]
            require(image["mimeType"] == "image/png", "Fixture exporter must preserve PNG textures")
            view = doc["bufferViews"][image["bufferView"]]
            start = view.get("byteOffset", 0)
            cache[image_id] = image_pixels(glb.binary[start:start + view["byteLength"]])
        checked_images.add(image_id)
        return cache[image_id]

    seen, rows, signs, uv1_angles, control_angles = set(), [], set(), [], []
    for node_index, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        matrix = matrices[node_index]
        basis = matrix[:3, :3]
        require(np.allclose(basis.T @ basis, np.eye(3), atol=1e-6) and np.linalg.det(basis) > 0,
                "Fixture unexpectedly acquired scale/shear/reflection")
        for primitive_index, primitive in enumerate(doc["meshes"][node["mesh"]]["primitives"]):
            attrs = primitive["attributes"]
            points = np.asarray(glb.accessor(attrs["POSITION"], "VEC3", (5126,))) @ basis.T + matrix[:3, 3]
            normal_attribute = np.asarray(glb.accessor(attrs["NORMAL"], "VEC3", (5126,)), dtype=np.float32)
            normals = normal_attribute.astype(float) @ basis.T
            uv0 = np.asarray(glb.accessor(attrs["TEXCOORD_0"], "VEC2", (5126,)))
            indices = np.asarray(glb.accessor(primitive["indices"], "SCALAR", (5121, 5123, 5125), count_limit=300000)).reshape(-1, 3)
            material = doc["materials"][primitive["material"]]
            base = material["pbrMetallicRoughness"]["baseColorTexture"]
            require(type(base.get("texCoord", 0)) is int and base.get("texCoord", 0) == 0,
                    "Original color texture no longer uses UV0")
            require(np.array_equal(texture_pixels(base), source_color), "Export changed basecolor PNG pixels")
            normal_map = material.get("normalTexture")
            normal_uv = normal_map.get("texCoord", 0) if normal_map else None
            tangent_data = None
            if "TANGENT" in attrs:
                tangent_data = np.asarray(glb.accessor(attrs["TANGENT"], "VEC4", (5126,)))
                actual_tangents = tangent_data[:, :3] @ basis.T
            if normal_map:
                require(type(normal_uv) is int and normal_uv in (0, 1), "Invalid normal UV selector")
                source = sources[material["name"]]
                require(normal_uv == source["normal_texcoord"], "Normal binding differs from adapter")
                path = SOURCE / source["texture_path"]
                normal_pixels = texture_pixels(normal_map)
                require(sha256(path) == source["texture_sha256"]
                        and np.array_equal(normal_pixels, image_pixels(path.read_bytes())), "Normal PNG pixels changed")
                selected_uv = np.asarray(glb.accessor(attrs["TEXCOORD_" + str(normal_uv)], "VEC2", (5126,)))
                require(tangent_data is not None, "Normal-mapped region lacks tangents")
            for ids in indices:
                triangle = points[ids]
                blender = triangle[:, [0, 2, 1]] * np.array([1, -1, 1])
                distances = [np.linalg.norm(blender.mean(axis=0) - np.asarray(face["positions_blender_m"]).mean(axis=0))
                             for face in expected]
                match = int(np.argmin(distances))
                require(distances[match] < 1e-7 and match not in seen, "Triangle changed, duplicated or ambiguous")
                seen.add(match)
                face = expected[match]
                require(material["name"] == face["material"] and normal_uv == face["expected_normal_uv"],
                        "Selected/excluded material or normal UV ownership changed")
                source_points = np.asarray(face["positions_blender_m"])
                correspondence = np.linalg.norm(blender[:, None] - source_points[None, :], axis=2)
                order = np.argmin(correspondence, axis=1)
                require(len(set(map(int, order))) == 3 and np.max(correspondence[np.arange(3), order]) < 1e-7,
                        "Source triangle positions changed")
                native_uv0 = np.asarray(face["uv0_blender"])[order]
                color_uv = uv0[ids].copy()
                color_uv[:, 1] = 1 - color_uv[:, 1]
                require(np.allclose(color_uv, native_uv0, atol=1e-7, rtol=0), "Original color UV0 changed")
                row = {"source_face": match, "case": face['case'], "primitive_index": primitive_index, "material": material["name"],
                       "normal_uv": normal_uv, "positions_and_uv0_preserved": True,
                       "tangent_w_exported": tangent_data[ids, 3].tolist() if tangent_data is not None else None,
                       "tangent_xyz_world_exported": actual_tangents[ids].tolist() if tangent_data is not None else None}
                original_normals = np.asarray(face['native_corner_normals_blender'])[order]
                predicted_normals = exported_normal_bytes(original_normals)
                actual_normals = normal_attribute[ids]
                exact_normals = bool(np.array_equal(actual_normals, predicted_normals))
                bit_equal = actual_normals.view(np.uint32) == predicted_normals.view(np.uint32)
                zero_sign_difference = ((actual_normals == 0) & (predicted_normals == 0)
                    & (np.signbit(actual_normals) != np.signbit(predicted_normals)))
                nonzero = (actual_normals != 0) | (predicted_normals != 0)
                nonzero_bit_exact = bool(np.all(bit_equal[nonzero]))
                zero_only_bit_difference = bool(np.all(bit_equal | zero_sign_difference))
                native_rotated = blender_to_gltf(original_normals)
                angular_errors = [float(np.degrees(np.arccos(np.clip(np.dot(unit(before), unit(after)), -1, 1))))
                                  for before, after in zip(native_rotated, actual_normals)]
                row['normal_quantization'] = {'method': 'float32 round4 -> float32 normalize -> Blender Z-up to glTF Y-up',
                    'expected_exported_float32': predicted_normals.tolist(), 'actual_exported_float32': actual_normals.tolist(),
                    'exact_float32_values_match': exact_normals, 'exact_bytes_match': bool(np.all(bit_equal)),
                    'nonzero_components_bit_exact': nonzero_bit_exact,
                    'bit_differences_only_signed_zero': zero_only_bit_difference,
                    'signed_zero_difference_count': int(zero_sign_difference.sum()),
                    'maximum_component_error_from_native': float(np.max(np.abs(actual_normals - native_rotated))),
                    'maximum_angular_error_degrees_from_native': max(angular_errors),
                    'angular_errors_degrees_from_native': angular_errors,
                    'mismatched_components_after_exact_exporter_transform': int(np.count_nonzero(actual_normals != predicted_normals))}
                if normal_map:
                    source_uv = np.asarray(face["uv1_blender"] if normal_uv == 1 else face["uv0_blender"])[order]
                    exported_uv = selected_uv[ids].copy()
                    exported_uv[:, 1] = 1 - exported_uv[:, 1]
                    require(np.allclose(exported_uv, source_uv, atol=1e-7, rtol=0), "Selected normal UV changed")
                    native_normals = blender_to_gltf(np.asarray(face['native_corner_normals_blender'])[order])
                    native_frames = np.asarray(face['blender_mikk_frame_blender'])[order].copy()
                    native_frames[:, :3] = blender_to_gltf(native_frames[:, :3])
                    baker_frames = np.asarray(face['baker_frame_blender'])[order].copy()
                    baker_frames[:, :3] = blender_to_gltf(baker_frames[:, :3])
                    uv0_frames = np.asarray(face['blender_mikk_uv0_frame_blender'])[order].copy()
                    uv0_frames[:, :3] = blender_to_gltf(uv0_frames[:, :3])
                    corners, frames = [], []
                    for corner, index in enumerate(ids):
                        normal = unit(normals[index])
                        target, sign = unit(native_frames[corner, :3]), float(native_frames[corner, 3])
                        original, original_sign = unit(uv0_frames[corner, :3]), float(uv0_frames[corner, 3])
                        baker_target, baker_sign = unit(baker_frames[corner, :3]), float(baker_frames[corner, 3])
                        require(np.dot(target, baker_target) > .99999 and sign == baker_sign,
                                'Actual native baker frame differs from actual Blender Mikk frame')
                        _, old_corner_sign = frame(triangle, source_uv, normal)
                        vdown_target, vdown_sign = frame(triangle, selected_uv[ids], normal)
                        actual = unit(actual_tangents[index])
                        dot = float(np.dot(target, actual))
                        angle = float(np.degrees(np.arccos(np.clip(np.dot(target, original), -1, 1))))
                        signs.add(sign)
                        if normal_uv == 1:
                            uv1_angles.append(angle)
                            if match in (0, 1):
                                control_angles.append(angle)
                        corners.append({"corner": corner, "exported_vertex_index": int(index),
                            "native_bake_tangent_world": target.tolist(), "native_bake_w": sign,
                            "oracle": "Actual Blender calc_tangents selected UV layer", "old_per_corner_sign": old_corner_sign,
                            "native_uv0_w": original_sign, "exported_w": float(tangent_data[index, 3]),
                            "native_bake_tangent_dot": dot, "native_uv0_tangent_dot": float(np.dot(original, actual)),
                            "native_bake_tangent_angle_degrees": float(np.degrees(np.arccos(np.clip(dot, -1, 1)))),
                            "bake_w_matches": bool(abs(tangent_data[index, 3] - sign) < 1e-6),
                            "vdown_derivative_w": vdown_sign,
                            "old_vdown_w_assertion_matches": bool(abs(tangent_data[index, 3] - vdown_sign) < 1e-6),
                            "vdown_tangent_dot": float(np.dot(vdown_target, actual)),
                            "uv1_vs_uv0_angle_degrees": angle})
                        frames.append((normal, target, sign, actual, float(tangent_data[index, 3])))
                    samples = []
                    for bary in ((1/3, 1/3, 1/3), (.5, .25, .25), (.25, .5, .25), (.25, .25, .5)):
                        uv = np.asarray(bary) @ source_uv
                        height, width = normal_pixels.shape[:2]
                        x = int(np.clip(np.floor(uv[0] * width), 0, width - 1))
                        y = int(np.clip(np.floor((1 - uv[1]) * height), 0, height - 1))
                        rgb = normal_pixels[y, x, :3]
                        center_uv = np.asarray([(x + .5) / width, 1 - (y + .5) / height])
                        fractions = np.linalg.solve((source_uv[1:] - source_uv[0]).T, center_uv - source_uv[0])
                        pixel_bary = np.asarray([1 - fractions.sum(), *fractions])
                        require(np.all(pixel_bary > 0), 'RGB sample fell outside its chart')
                        normal, target, sign = interpolated_frame(native_normals, native_frames, pixel_bary)
                        actual_frames = tangent_data[ids].copy()
                        actual_frames[:, :3] = actual_tangents[ids]
                        actual_normal, actual, actual_sign = interpolated_frame(normals[ids], actual_frames, pixel_bary)
                        expected_world = world_normal(rgb, normal, target, sign)
                        actual_world = world_normal(rgb, actual_normal, actual, actual_sign)
                        normal_dot = float(np.dot(expected_world, actual_world))
                        samples.append({"requested_barycentric": list(bary), "texel_center_barycentric": pixel_bary.tolist(),
                            "native_uv_v_up": center_uv.tolist(),
                            "png_pixel_top_left_origin": [x, y], "rgb8": rgb.tolist(),
                            "native_bake_world_normal": expected_world.tolist(),
                            "exported_world_normal": actual_world.tolist(), "world_normal_dot": normal_dot,
                            "world_normal_angle_degrees": float(np.degrees(np.arccos(np.clip(normal_dot, -1, 1))))})
                    passed = all(c["native_bake_tangent_dot"] > .99999 and c["bake_w_matches"] for c in corners)
                    passed = passed and all(s["world_normal_dot"] > .99999 for s in samples)
                    row.update(corners=corners, rgb_world_normal_samples=samples, native_bake_frame_preserved=passed,
                               failure=(None if passed else "Selected-UV tangent frame differs from unchanged-PNG native bake"))
                else:
                    row.update(native_bake_frame_preserved=None, failure=None,
                               excluded_original_material_preserved=True, normal_frame_check="not_applicable_no_normal_texture")
                rows.append(row)
    require(seen == set(range(6)) and signs == {-1., 1.} and uv1_angles and control_angles and min(control_angles) > 15,
            "Fixture does not exercise six faces, both bake signs and rotated UV1")
    require(native["active_uv_index"] == 0 and native["active_render_flags"] == [True, False], "UV flags changed")
    require({row["case"] for row in native["negative_cases"] if row["rejected"]} ==
            {"existing_uv1", "conflicting_attribute"}, "Refusal cases missing")
    quantized_exact = all(row['normal_quantization']['exact_float32_values_match']
        and row['normal_quantization']['nonzero_components_bit_exact']
        and row['normal_quantization']['bit_differences_only_signed_zero'] for row in rows)
    passed = quantized_exact and all(row["native_bake_frame_preserved"] is not False for row in rows)
    return {"schema_version": 6, "status": "analytic_export_verified" if passed else "analytic_export_rejected",
        "passed": passed, "model_sha256": glb.hash, "raw_source_model_sha256": native["model_sha256"],
        "native_report_sha256": sha256(SOURCE / "native-fixture.json"), "verifier_sha256": sha256(Path(__file__)),
        "frame_policy": native['frame_policy'],
        "normal_quantization": {'exact_installed_exporter_transform_matched_all_corners': quantized_exact,
            'exact_nonzero_float32_bits_all_corners': all(row['normal_quantization']['nonzero_components_bit_exact'] for row in rows),
            'signed_zero_difference_count': sum(row['normal_quantization']['signed_zero_difference_count'] for row in rows),
            'other_bit_difference_count': sum(row['normal_quantization']['mismatched_components_after_exact_exporter_transform'] for row in rows),
            'verified_corners': sum(len(row['normal_quantization']['actual_exported_float32']) for row in rows),
            'maximum_component_error_from_native': max(row['normal_quantization']['maximum_component_error_from_native'] for row in rows),
            'maximum_angular_error_degrees_from_native': max(row['normal_quantization']['maximum_angular_error_degrees_from_native'] for row in rows),
            'round_digits': 4, 'working_dtype': 'float32', 'normalization': 'np.linalg.norm(axis=1, keepdims=True); np.divide(..., out=values)',
            'primary_code_sha256': sha256(HERE / 'exporter-primary/blender/exp/primitive_extract.py'),
            'constant_source_sha256': sha256(HERE / 'exporter-primary/io/com/constants.py'),
            'tolerance_widened': False},
        "tangent_convention": "Actual Blender Mikk selected-UV frames and newly baked V-up PNG vectors; XYZ rotated to glTF Y-up; B=cross(N,T)*W. No old per-corner-sign PNG reuse.",
        "native_frame_witness_sha256": native['native_frame_witness_sha256'],
        "coordinate_space": "glTF world Y-up", "triangles": sorted(rows, key=lambda row: row["source_face"]),
        "all_six_faces_measured": True, "normal_uv_regions": [0, 1], "bake_handedness_signs_checked": sorted(signs),
        "rotated_uv1_min_angle_degrees": min(uv1_angles), "material_png_images_checked": len(checked_images),
        "rotated_uv1_original_control_min_angle_degrees": min(control_angles),
        "hd_render_attributes": attributes, "source_geometry_uv0_basecolor_and_active_flags_preserved": True,
        "excluded_triangle_original_material_preserved": True, "native_negative_cases": native["negative_cases"],
        "native_jobs": 0, "new_renders": 0, "provider_calls": 0, "quality_approved": False, "weft_compatible": False,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "Six analytic triangles only, including custom normals. No avatar, rendered appearance or broad tangent proof."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("model_sha256")
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    require(not args.report.exists(), "Append-only report already exists")
    report = verify(args.model.resolve(), args.model_sha256)
    with args.report.open("x") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "passed": report["passed"], "faces": len(report["triangles"])}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
