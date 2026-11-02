"""Independently compare exported triangles, color data and selected-UV tangent derivatives."""

from io import BytesIO
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/"sources/host"))
from avatar_assets.safety import require, sha256, write_json
from avatar_assets.three_d import _GLB, _world_matrices, validate_render_attributes


def image_pixels(data):
    with Image.open(BytesIO(data)) as image:
        require(image.format == "PNG" and image.width*image.height <= 1024**2 and getattr(image, "n_frames", 1) == 1,
                "Unexpected exported fixture image format/size")
        return np.asarray(image.convert("RGBA"))


def frame(points, uv, normal):
    edge1, edge2 = points[1]-points[0], points[2]-points[0]
    delta1, delta2 = uv[1]-uv[0], uv[2]-uv[0]
    determinant = delta1[0]*delta2[1]-delta1[1]*delta2[0]
    require(abs(determinant) > 1e-10, "Fixture UV triangle is degenerate")
    du = (edge1*delta2[1]-edge2*delta1[1])/determinant
    dv = (edge2*delta1[0]-edge1*delta2[0])/determinant
    tangent = du-normal*np.dot(du, normal)
    magnitude = np.linalg.norm(tangent)
    require(magnitude > 1e-8, "Fixture tangent derivative is degenerate")
    tangent /= magnitude
    sign = float(np.sign(np.dot(np.cross(normal, tangent), dv)))
    require(abs(sign) == 1, "Fixture handedness is undefined")
    return tangent, sign


def main():
    native = json.loads((HERE/"native-fixture.json").read_text())
    require(native["model_sha256"] == sha256(HERE/"model.glb") and native["original_data_unchanged"], "Native source binding failed")
    glb = _GLB(HERE/"model.glb")
    attributes = validate_render_attributes(HERE/"model.glb")
    doc, nodes = glb.doc, glb.doc.get("nodes", [])
    parents = {child: index for index, node in enumerate(nodes) for child in node.get("children", [])}
    matrices = [np.asarray(value).reshape(4, 4, order="F") for value in _world_matrices(nodes, parents)]
    expected = native["faces"]
    sources = {row["material"]: row for row in native["undercoat"]["materials"]}
    source_color = image_pixels((HERE/"textures/original-basecolor.png").read_bytes())
    checked_images = set()

    def texture_pixels(info):
        texture = doc["textures"][info["index"]]
        image = doc["images"][texture["source"]]
        require(image["mimeType"] == "image/png", "Fixture exporter must preserve PNG textures")
        view = doc["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        checked_images.add(texture["source"])
        return image_pixels(glb.binary[start:start+view["byteLength"]])

    seen, rows, signs, uv1_angle = set(), [], set(), []
    for node_index, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        matrix = matrices[node_index]
        basis = matrix[:3, :3]
        require(np.allclose(basis.T@basis, np.eye(3), atol=1e-6) and np.linalg.det(basis) > 0,
                "Fixture unexpectedly acquired scale/shear/reflection")
        for primitive in doc["meshes"][node["mesh"]]["primitives"]:
            attrs = primitive["attributes"]
            points = np.asarray(glb.accessor(attrs["POSITION"], "VEC3", (5126,)))@basis.T+matrix[:3, 3]
            normals = np.asarray(glb.accessor(attrs["NORMAL"], "VEC3", (5126,)))@basis.T
            uv0 = np.asarray(glb.accessor(attrs["TEXCOORD_0"], "VEC2", (5126,)))
            indices = np.asarray(glb.accessor(primitive["indices"], "SCALAR", (5121, 5123, 5125), count_limit=300000)).reshape(-1, 3)
            material = doc["materials"][primitive["material"]]
            base = material["pbrMetallicRoughness"]["baseColorTexture"]
            require(type(base.get("texCoord", 0)) is int and base.get("texCoord", 0) == 0,
                    "Original color texture no longer uses UV0")
            require(np.array_equal(texture_pixels(base), source_color), "Export changed original basecolor PNG pixels")
            normal_map = material.get("normalTexture")
            normal_uv = normal_map.get("texCoord", 0) if normal_map else None
            if normal_map:
                require(type(normal_uv) is int and normal_uv in (0, 1), "Invalid exported normal UV selector")
                source = sources[material["name"]]
                require(normal_uv == source["normal_texcoord"], "Exported normal binding differs from adapter")
                path = HERE/source["texture_path"]
                require(sha256(path) == source["texture_sha256"] and
                        np.array_equal(texture_pixels(normal_map), image_pixels(path.read_bytes())),
                        "Export changed selected material normal PNG pixels")
                selected_uv = np.asarray(glb.accessor(attrs["TEXCOORD_"+str(normal_uv)], "VEC2", (5126,)))
                tangent_data = np.asarray(glb.accessor(attrs["TANGENT"], "VEC4", (5126,)))
                actual_tangents = tangent_data[:, :3]@basis.T
            for ids in indices:
                triangle = points[ids]
                blender = triangle[:, [0, 2, 1]]*np.array([1, -1, 1])
                distances = [np.linalg.norm(blender.mean(axis=0)-np.asarray(face["positions_blender_m"]).mean(axis=0))
                             for face in expected]
                match = int(np.argmin(distances))
                require(distances[match] < 1e-7 and match not in seen, "Exported triangle changed, duplicated or ambiguous")
                seen.add(match)
                face = expected[match]
                require(material["name"] == face["material"] and normal_uv == face["expected_normal_uv"],
                        "Selected/excluded material or normal UV ownership changed")
                source_points, source_uv0 = np.asarray(face["positions_blender_m"]), np.asarray(face["uv0_blender"])
                correspondence = np.linalg.norm(blender[:, None]-source_points[None, :], axis=2)
                order = np.argmin(correspondence, axis=1)
                require(len(set(map(int, order))) == 3 and np.max(correspondence[np.arange(3), order]) < 1e-7,
                        "Source triangle positions changed")
                color_uv = uv0[ids].copy()
                color_uv[:, 1] = 1-color_uv[:, 1]
                require(np.allclose(color_uv, source_uv0[order], atol=1e-7, rtol=0), "Original per-corner color UV0 changed")
                row = {"source_face": match, "material": material["name"], "normal_uv": normal_uv,
                       "positions_and_uv0_preserved": True}
                if normal_map:
                    exported_uv = selected_uv[ids].copy()
                    exported_uv[:, 1] = 1-exported_uv[:, 1]
                    source_uv = np.asarray(face["uv1_blender"] if normal_uv == 1 else face["uv0_blender"])
                    require(np.allclose(exported_uv, source_uv[order], atol=1e-7, rtol=0),
                            "Selected normal UV coordinates changed during export")
                    dots, expected_signs, angles = [], [], []
                    for corner, index in enumerate(ids):
                        normal = normals[index]/np.linalg.norm(normals[index])
                        target, sign = frame(triangle, selected_uv[ids], normal)
                        actual = actual_tangents[index]/np.linalg.norm(actual_tangents[index])
                        dot = float(np.dot(target, actual))
                        require(dot > .99999 and abs(tangent_data[index, 3]-sign) < 1e-6,
                                "Exporter tangent direction/handedness does not match selected normal UV derivatives")
                        dots.append(dot)
                        expected_signs.append(sign)
                        signs.add(sign)
                        if normal_uv == 1:
                            original, _ = frame(triangle, uv0[ids], normal)
                            angle = float(np.degrees(np.arccos(np.clip(np.dot(target, original), -1, 1))))
                            angles.append(angle)
                            uv1_angle.append(angle)
                    row.update(tangent_direction_dot=dots, tangent_handedness=expected_signs,
                               uv1_vs_uv0_angle_degrees=angles, selected_uv_derivatives_verified=True)
                rows.append(row)
    require(seen == set(range(4)) and signs == {-1., 1.} and uv1_angle and min(uv1_angle) > 15,
            "Fixture did not exercise all faces, both handedness signs or a materially rotated UV1 basis")
    require(native["active_uv_index"] == 0 and native["active_render_flags"] == [True, False], "Original UV selection flags changed")
    require({row["case"] for row in native["negative_cases"] if row["rejected"]} ==
            {"existing_uv1", "conflicting_attribute"}, "Required refusal cases were not observed")
    result = {"schema_version": 1, "status": "analytic_adapter_export_verified", "model_sha256": glb.hash,
              "triangles": rows, "normal_uv_regions": [0, 1], "material_png_images_checked": len(checked_images),
              "rotated_uv1_min_angle_degrees": min(uv1_angle), "handedness_signs_checked": sorted(signs),
              "hd_render_attributes": attributes, "source_geometry_uv0_basecolor_and_active_flags_preserved": True,
              "excluded_triangle_original_material_preserved": True, "selected_uv_tangent_derivatives_verified": True,
              "native_negative_cases": native["negative_cases"], "new_renders": 0, "quality_approved": False,
              "weft_compatible": False, "scope": "One small analytic Blender adapter/export fixture, not a generated avatar or complete accepted pack."}
    write_json(HERE/"host-verification.json", result)
    print(json.dumps({"status": result["status"], "rotated_uv1_min_angle_degrees": min(uv1_angle),
                      "triangles": len(rows), "quality_approved": False}, sort_keys=True))


if __name__ == "__main__":
    main()
