"""Analytic fixtures verify v2 data/geometry behavior, not finished visual quality."""

import copy
import json
import math
import struct
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from avatar_assets.scene import read_scene, validate_scene
from avatar_assets.three_d import _GLB, MAX_ACCESSOR_VALUES, validate_render_attributes
from scripts.blender_geometry import fur_geometry, normalized_weights, surface_pixels, planar_triangle_uv
from tests.test_scene import analytic_scene
from tests.test_three_d import analytic_rig, glb_bytes


def quality_scene():
    scene = analytic_scene()
    scene["schema_version"] = 2
    scene["parts"][0] = {
        "name": "custom-body", "shape": "mesh", "location": [0, 0, 0], "scale": [1, 1, 1],
        "material": "body", "bone": "root", "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "faces": [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        "uv": [[0, 0], [1, 0], [0, 1], [1, 1]],
        "weights": [{"root": 1}, {"root": .75, "head": .25}, {"root": .5, "head": .5}, {"head": 1}],
        "fur": {"count": 20, "length": .02, "width": .001, "seed": 7},
    }
    scene["materials"]["body"]["surface"] = {"pattern": "fur", "resolution": 256, "strength": .5, "seed": 11}
    return scene


class QualitySceneTests(unittest.TestCase):
    def reject(self, change):
        scene = quality_scene()
        change(scene)
        with self.assertRaises(ValueError):
            validate_scene(scene)

    def test_valid_mesh_surface_weights_are_preserved_without_mutation(self):
        scene = quality_scene()
        original = copy.deepcopy(scene)
        result = validate_scene(scene)
        self.assertEqual(result, original)
        result["parts"][0]["weights"][0]["root"] = .5
        self.assertEqual(scene, original)

    def test_quads_and_optional_mesh_channels(self):
        scene = quality_scene()
        mesh = scene["parts"][0]
        mesh["vertices"] = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
        mesh["faces"] = [[0, 1, 2, 3]]
        mesh.pop("weights")
        mesh.pop("uv")
        self.assertEqual(validate_scene(scene), scene)

    def test_v1_does_not_silently_ignore_quality_requests(self):
        self.reject(lambda scene: scene.update(schema_version=1))
        for field, value in (("segments", 32), ("bevel", .03), ("fur", {"count": 0})):
            with self.subTest(field=field):
                scene = analytic_scene()
                scene["parts"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_scene(scene)

    def test_weft_quality_conversion_is_explicitly_unsupported(self):
        self.reject(lambda scene: scene.update(output_profile="weft-v1"))

    def test_bad_topology_is_rejected(self):
        for faces in ([[0, 1]], [[0, 1, 99]], [[0, 1, 1]], [[0, True, 2]], [[0, 1, 2, 3, 0]]):
            with self.subTest(faces=faces):
                self.reject(lambda scene: scene["parts"][0].update(faces=faces))
        self.reject(lambda scene: scene["parts"][0]["vertices"].__setitem__(2, [2, 0, 0]))
        self.reject(lambda scene: scene["parts"][0]["vertices"].__setitem__(0, [math.inf, 0, 0]))

    def test_uv_count_and_finiteness(self):
        self.reject(lambda scene: scene["parts"][0].update(uv=[[0, 0]]))
        self.reject(lambda scene: scene["parts"][0]["uv"].__setitem__(0, [math.nan, 0]))

    def test_weights_need_known_positive_normalized_influences(self):
        for invalid in ({}, {"root": .9}, {"root": -1, "head": 2}, {"root": 0, "head": 1},
                        {"missing": 1}, {"root": True}, {"root": math.nan}):
            with self.subTest(weights=invalid):
                self.reject(lambda scene: scene["parts"][0]["weights"].__setitem__(0, invalid))
        self.reject(lambda scene: scene["parts"][0].update(weights=[{"root": 1}]))
        scene = quality_scene()
        for i in range(3):
            scene["bones"].append({"name": f"extra-{i}", "parent": "root", "head": [i, 0, 0], "tail": [i, 0, 1]})
        scene["parts"][0]["weights"][0] = {bone["name"]: .2 for bone in scene["bones"]}
        with self.assertRaises(ValueError):
            validate_scene(scene)

    def test_mesh_aggregate_vertex_and_face_bounds(self):
        scene = quality_scene()
        part = scene["parts"][0]
        part["vertices"] += [[0, 0, 0]]*15000
        part.pop("uv")
        part.pop("weights")
        other = copy.deepcopy(part)
        other["name"] = "second-surface"
        scene["parts"] = [part, other]
        with self.assertRaisesRegex(ValueError, "Aggregate mesh"):
            validate_scene(scene)
        self.reject(lambda scene: scene["parts"][0].update(faces=[[0, 1, 2]]*60001))

    def test_part_and_primitive_detail_limits(self):
        scene = quality_scene()
        scene["parts"] = [{**scene["parts"][1], "name": f"sphere-{i}", "segments": 8, "rings": 6} for i in range(128)]
        self.assertEqual(len(validate_scene(scene)["parts"]), 128)
        scene["parts"].append({**scene["parts"][0], "name": "excess"})
        with self.assertRaises(ValueError):
            validate_scene(scene)
        for field, value in (("segments", 65), ("rings", 49), ("bevel_segments", 5), ("bevel", .301), ("bevel", True)):
            with self.subTest(field=field):
                self.reject(lambda scene: scene["parts"][1].update({field: value}))

    def test_fur_bounds_and_aggregate_budget(self):
        for field, value in (("count", 10001), ("length", .041), ("width", .0041), ("seed", -1)):
            with self.subTest(field=field):
                self.reject(lambda scene: scene["parts"][0]["fur"].update({field: value}))
        scene = quality_scene()
        scene["parts"][0]["fur"]["count"] = 10000
        for i, count in enumerate((10000, 4000)):
            scene["parts"].append({**scene["parts"][1], "name": f"fur-part-{i}",
                                   "fur": {"count": count, "length": .02, "width": .001, "seed": i}})
        self.assertEqual(sum(part.get("fur", {}).get("count", 0) for part in validate_scene(scene)["parts"]), 24000)
        scene["parts"][-1]["fur"]["count"] += 1
        with self.assertRaisesRegex(ValueError, "Aggregate fur"):
            validate_scene(scene)

    def test_surface_fields_are_bounded_and_data_only(self):
        for field, value in (("pattern", "python"), ("resolution", 1024), ("strength", 1.1), ("seed", "1")):
            with self.subTest(field=field):
                self.reject(lambda scene: scene["materials"]["body"]["surface"].update({field: value}))
        self.reject(lambda scene: scene["materials"]["body"]["surface"].update(expression="execute"))

    def test_large_v2_reader_and_original_v1_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"scene.json"
            scene = quality_scene()
            scene["provenance_notes"] = "a"*(1024*1024)
            path.write_text(json.dumps(scene))
            self.assertEqual(read_scene(path)["schema_version"], 2)
            scene = analytic_scene()
            scene["provenance_notes"] = "a"*(1024*1024)
            path.write_text(json.dumps(scene))
            with self.assertRaisesRegex(ValueError, "Version 1 scene"):
                read_scene(path)
            for raw in ('{"schema_version":2,"schema_version":2}', '{"schema_version":NaN}'):
                path.write_text(raw)
                with self.assertRaises(ValueError):
                    read_scene(path)


class DeterministicGeometryTests(unittest.TestCase):
    def test_collapsed_triangle_uv_uses_geometric_plane_and_keeps_neighborhood(self):
        points = [(0, 0, 0), (1, 0, 1), (0, 2, 0)]
        original = [(.2, .4), (.5, .4), (.8, .4)]
        uv = planar_triangle_uv(points, original)
        determinant = (uv[1][0]-uv[0][0])*(uv[2][1]-uv[0][1])-(uv[2][0]-uv[0][0])*(uv[1][1]-uv[0][1])
        self.assertGreater(abs(determinant), 1e-10)
        self.assertAlmostEqual(sum(value[0] for value in uv)/3, .5)
        self.assertAlmostEqual(sum(value[1] for value in uv)/3, .4)
        self.assertAlmostEqual(math.dist(uv[0], uv[1])/math.dist(uv[0], uv[2]),
                               math.dist(points[0], points[1])/math.dist(points[0], points[2]))

    def test_fur_has_real_tapered_geometry_and_inherited_mixed_weights(self):
        settings = {"count": 40, "length": .02, "width": .001, "seed": 5}
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        weights = [{"root": 1}, {"tip": 1}, {"root": .5, "tip": .5}]
        uv = [(0, 0), (1, 0), (0, 1)]
        first = fur_geometry(vertices, [(0, 1, 2)], weights, uv, settings)
        self.assertEqual(first, fur_geometry(vertices, [(0, 1, 2)], weights, uv, settings))
        self.assertEqual((len(first["vertices"]), len(first["faces"])), (120, 40))
        for i in range(0, 120, 3):
            a, b, tip = first["vertices"][i:i+3]
            self.assertEqual(a[2], 0)
            self.assertEqual(b[2], 0)
            self.assertGreater(tip[2], .014)
            midpoint = [(a[j]+b[j])/2 for j in range(2)]
            self.assertGreaterEqual(min(midpoint), 0)
            self.assertLessEqual(sum(midpoint), 1)
            influence = first["weights"][i]
            self.assertEqual(set(influence), {"root", "tip"})
            self.assertAlmostEqual(sum(influence.values()), 1)
            self.assertAlmostEqual(influence["tip"], midpoint[0]+midpoint[1]*.5)
        changed = fur_geometry(vertices, [(0, 1, 2)], weights, uv, {**settings, "seed": 6})
        self.assertNotEqual(first["vertices"], changed["vertices"])

    def test_fur_samples_surface_area_not_triangle_count(self):
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (10, 0, 0), (14, 0, 0), (10, 4, 0)]
        result = fur_geometry(vertices, [(0, 1, 2), (3, 4, 5)], [{"root": 1}]*6, [(0, 0)]*6,
                              {"count": 1700, "length": .01, "width": .001, "seed": 2})
        large = sum(result["vertices"][i][0] > 5 for i in range(0, len(result["vertices"]), 3))
        self.assertTrue(1550 <= large <= 1650, large)

    def test_interpolated_weight_budget_is_normalized_top_four(self):
        result = normalized_weights({"a": .4, "b": .3, "c": .1, "d": .09, "e": .08, "f": .03})
        self.assertEqual(set(result), {"a", "b", "c", "d"})
        self.assertAlmostEqual(sum(result.values()), 1)

    def test_baked_patterns_are_reproducible_finite_and_distinct(self):
        outputs = []
        for pattern in ("fur", "scales", "brushed"):
            settings = {"pattern": pattern, "resolution": 256, "strength": .7, "seed": 17}
            color, normal = surface_pixels([.8, .4, .2, 1], settings)
            second = surface_pixels([.8, .4, .2, 1], settings)
            np.testing.assert_array_equal(color, second[0])
            np.testing.assert_array_equal(normal, second[1])
            self.assertEqual(color.shape, (256, 256, 4))
            self.assertTrue(np.isfinite(color).all() and np.isfinite(normal).all())
            self.assertTrue((color >= 0).all() and (color <= 1).all())
            np.testing.assert_allclose(np.linalg.norm(normal[:, :, :3]*2-1, axis=2), 1, atol=2e-7)
            outputs.append(color)
        self.assertFalse(np.array_equal(outputs[0], outputs[1]))
        self.assertFalse(np.array_equal(outputs[1], outputs[2]))

    def test_zero_surface_strength_keeps_flat_normal_and_base_color(self):
        color, normal = surface_pixels([.5, .25, .1, 1], {"pattern": "fur", "resolution": 256, "strength": 0, "seed": 0})
        np.testing.assert_allclose(normal[:, :, :3], np.broadcast_to([.5, .5, 1], (256, 256, 3)))
        self.assertEqual(float(color[:, :, 0].max()), float(color[:, :, 0].min()))
        np.testing.assert_allclose(color[0, 0], [.5, .25, .1, 1], atol=1e-7)

    def test_fur_texture_has_fine_directional_detail_without_dominant_repeated_bands(self):
        for resolution in (256, 512):
            with self.subTest(resolution=resolution):
                color, _ = surface_pixels([.6, .6, .6, 1],
                    {"pattern": "fur", "resolution": resolution, "strength": 1, "seed": 17})
                field = color[:, :, 0]-color[:, :, 0].mean()
                power = np.abs(np.fft.rfft2(field))**2
                power[0, 0] = 0
                self.assertLess(float(power.max()/power.sum()), .02)
                across = np.mean(np.abs(np.roll(field, 1, axis=1)-field))
                along = np.mean(np.abs(np.roll(field, 1, axis=0)-field))
                self.assertGreater(float(across/along), 1.8)


class QualityAccessorBudgetTests(unittest.TestCase):
    def test_two_million_values_fit_but_one_more_is_rejected_before_decoding(self):
        self.assertEqual(MAX_ACCESSOR_VALUES, 2_000_000)
        binary = struct.pack("<f", 1.)*(2_000_000+1)
        document = {"asset": {"version": "2.0"}, "buffers": [{"byteLength": len(binary)}],
                    "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 6_400_000},
                                    {"buffer": 0, "byteOffset": 6_400_000, "byteLength": 1_600_000},
                                    {"buffer": 0, "byteOffset": 8_000_000, "byteLength": 4}],
                    "accessors": [{"bufferView": 0, "componentType": 5126, "count": 100_000, "type": "MAT4"},
                                  {"bufferView": 1, "componentType": 5126, "count": 25_000, "type": "MAT4"},
                                  {"bufferView": 2, "componentType": 5126, "count": 1, "type": "SCALAR"}]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"analytic-accessor-budget.glb"
            path.write_bytes(glb_bytes(document, binary))
            glb = _GLB(path)
            first = glb.accessor(0)
            self.assertEqual(len(first), 100_000)
            self.assertEqual(len(glb.accessor(1)), 25_000)
            self.assertEqual(glb.value_count, 2_000_000)
            self.assertIs(glb.accessor(0), first)
            self.assertEqual(glb.value_count, 2_000_000)
            with self.assertRaisesRegex(ValueError, "Decoded accessor values exceed"):
                glb.accessor(2)
            self.assertNotIn(2, glb.cache)


class RenderAttributeGateTests(unittest.TestCase):
    def fixture(self, tangent=(1, 0, 0, 1), normal=(0, 0, 1), uv=(.2, .4)):
        document, binary, _ = analytic_rig()
        attrs = document["meshes"][0]["primitives"][0]["attributes"]
        for name, value, kind in (("NORMAL", normal, "VEC3"), ("TANGENT", tangent, "VEC4"), ("TEXCOORD_0", uv, "VEC2")):
            offset = len(binary)
            binary.extend(struct.pack("<"+"f"*len(value), *value)*3)
            document["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary)-offset})
            document["accessors"].append({"bufferView": len(document["bufferViews"])-1, "componentType": 5126,
                                          "count": 3, "type": kind})
            attrs[name] = len(document["accessors"])-1
        document["buffers"][0]["byteLength"] = len(binary)
        return document, binary

    def test_valid_frames_use_a_separate_surface_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"surface.glb"
            path.write_bytes(glb_bytes(*self.fixture()))
            result = validate_render_attributes(path)
            self.assertEqual(result["status"], "validated_render_attributes")
            self.assertEqual(result["decoded_scalar_values"], 27)

    def test_rejects_nonunit_parallel_and_invalid_handedness_tangents(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"surface.glb"
            for tangent in ((0, 0, 0, 1), (2, 0, 0, 1), (0, 0, 1, 1), (1, 0, 0, .5)):
                with self.subTest(tangent=tangent):
                    path.write_bytes(glb_bytes(*self.fixture(tangent=tangent)))
                    with self.assertRaises(ValueError):
                        validate_render_attributes(path)

    def test_rejects_bad_normals_nonfinite_uv_and_attribute_count_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"surface.glb"
            for kwargs in ({"normal": (0, 0, 2)}, {"uv": (math.nan, 0)}):
                path.write_bytes(glb_bytes(*self.fixture(**kwargs)))
                with self.assertRaises(ValueError):
                    validate_render_attributes(path)
            document, binary = self.fixture()
            document["accessors"][-1]["count"] = 2
            path.write_bytes(glb_bytes(document, binary))
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                validate_render_attributes(path)

    def test_normal_map_requires_verified_uv0_and_tangent_binding(self):
        document, binary = self.fixture()
        image = BytesIO()
        Image.new("RGB", (1, 1), (128, 128, 255)).save(image, format="PNG")
        offset = len(binary)
        binary.extend(image.getvalue())
        document["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary)-offset})
        document["images"] = [{"bufferView": len(document["bufferViews"])-1, "mimeType": "image/png"}]
        document["textures"] = [{"source": 0}]
        document["materials"][0]["normalTexture"] = {"index": 0}
        document["buffers"][0]["byteLength"] = len(binary)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"surface.glb"
            path.write_bytes(glb_bytes(document, binary))
            self.assertTrue(validate_render_attributes(path)["primitives"][0]["normal_mapped"])
            for name in ("TEXCOORD_0", "TANGENT"):
                changed = copy.deepcopy(document)
                changed["meshes"][0]["primitives"][0]["attributes"].pop(name)
                path.write_bytes(glb_bytes(changed, binary))
                with self.assertRaisesRegex(ValueError, "requires UV0 and tangents"):
                    validate_render_attributes(path)
            document["materials"][0]["normalTexture"]["texCoord"] = 1
            path.write_bytes(glb_bytes(document, binary))
            with self.assertRaisesRegex(ValueError, "verified UV0"):
                validate_render_attributes(path)


if __name__ == "__main__":
    unittest.main()
