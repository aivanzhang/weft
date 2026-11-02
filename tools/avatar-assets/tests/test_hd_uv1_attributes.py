"""HD surface binding fixtures, not native-export, appearance or Weft acceptance."""

from io import BytesIO
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from avatar_assets.three_d import MAX_ACCESSOR_VALUES, MAX_FILE_BYTES, MAX_VERTICES, validate_render_attributes
from tests.test_three_d import glb_bytes


def surface_fixture(count=3, *, uv0=True, uv1=True, normal_uv=1, integer_uv1=False):
    binary, views, accessors, attrs = bytearray(), [], [], {}

    def add(name, values, kind, component=5126, normalized=False):
        binary.extend(b"\0"*(-len(binary) % 4))
        offset = len(binary)
        binary.extend(struct.pack("<"+({5126: "f", 5123: "H"}[component])*len(values), *values)*count)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary)-offset})
        accessors.append({"bufferView": len(views)-1, "count": count, "type": kind, "componentType": component})
        if normalized:
            accessors[-1]["normalized"] = True
        attrs[name] = len(accessors)-1

    add("POSITION", (0, 0, 0), "VEC3")
    add("NORMAL", (0, 0, 1), "VEC3")
    add("TANGENT", (1, 0, 0, 1), "VEC4")
    if uv0:
        add("TEXCOORD_0", (.2, .4), "VEC2")
    if uv1:
        add("TEXCOORD_1", (20000, 40000) if integer_uv1 else (.6, .8), "VEC2",
            5123 if integer_uv1 else 5126, integer_uv1)
    image = BytesIO()
    Image.new("RGB", (1, 1), (128, 128, 255)).save(image, format="PNG")
    offset = len(binary)
    binary.extend(image.getvalue())
    views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary)-offset})
    normal = {"index": 0}
    if normal_uv is not None:
        normal["texCoord"] = normal_uv
    document = {"asset": {"version": "2.0", "generator": "analytic-HD-frame-fixture"},
                "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
                "meshes": [{"primitives": [{"attributes": attrs, "material": 0}]}],
                "materials": [{"name": "HD normal mapped", "normalTexture": normal}],
                "textures": [{"source": 0}], "images": [{"bufferView": len(views)-1, "mimeType": "image/png"}]}
    return document, binary


class HDUV1AttributeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"surface.glb"

    def validate(self, document, binary):
        self.path.write_bytes(glb_bytes(document, binary))
        return validate_render_attributes(self.path)

    def test_uv0_default_and_explicit_uv1_select_matching_verified_accessor(self):
        for uv in (None, 0, 1):
            with self.subTest(uv=uv):
                report = self.validate(*surface_fixture(normal_uv=uv))
                self.assertEqual(report["primitives"][0]["normal_map_uv_set"], 0 if uv is None else uv)
                self.assertEqual(report["primitives"][0]["verified_uv_sets"], [0, 1])
                self.assertEqual(report["decoded_scalar_values"], 33)
                self.assertFalse(report["tangent_uv_derivation_verified"])
                self.assertFalse(report["weft_runtime_verified"])
                self.assertIn("remain unsupported", report["limitations"][1])

    def test_uv1_only_normal_map_does_not_require_unreferenced_uv0(self):
        report = self.validate(*surface_fixture(uv0=False))
        self.assertEqual(report["decoded_scalar_values"], 27)
        self.assertEqual(report["primitives"][0]["verified_uv_sets"], [1])
        for name in ("TEXCOORD_1", "TANGENT"):
            document, binary = surface_fixture()
            document["meshes"][0]["primitives"][0]["attributes"].pop(name)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "requires UV1 and tangents"):
                self.validate(document, binary)

    def test_selector_requires_exact_nonboolean_integer_zero_or_one(self):
        for invalid in (True, False, 0., 1., "1", None, -1, 2, 100, math.nan):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "exact integer UV index 0 or 1"):
                document, binary = surface_fixture()
                document["materials"][0]["normalTexture"]["texCoord"] = invalid
                self.validate(document, binary)

    def test_uv1_requires_finite_matching_vec2_values(self):
        for defect in ("count", "shape", "nan", "infinity"):
            document, binary = surface_fixture()
            uv = document["accessors"][document["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_1"]]
            if defect == "count":
                uv["count"] = 2
            elif defect == "shape":
                uv["type"] = "VEC3"
            else:
                struct.pack_into("<f", binary, document["bufferViews"][uv["bufferView"]]["byteOffset"],
                                 math.nan if defect == "nan" else math.inf)
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                self.validate(document, binary)

    def test_uv1_still_requires_valid_frames_and_handedness(self):
        for name, values in (("NORMAL", (0, 0, 2)), ("TANGENT", (0, 0, 0, 1)),
                             ("TANGENT", (0, 0, 1, 1)), ("TANGENT", (1, 0, 0, .5)),
                             ("TANGENT", (1, 0, 0, math.nan))):
            document, binary = surface_fixture()
            spec = document["accessors"][document["meshes"][0]["primitives"][0]["attributes"][name]]
            offset = document["bufferViews"][spec["bufferView"]]["byteOffset"]
            struct.pack_into("<"+"f"*len(values), binary, offset, *values)
            with self.subTest(name=name, values=values), self.assertRaises(ValueError):
                self.validate(document, binary)
        document, binary = surface_fixture()
        spec = document["accessors"][document["meshes"][0]["primitives"][0]["attributes"]["TANGENT"]]
        struct.pack_into("<f", binary, document["bufferViews"][spec["bufferView"]]["byteOffset"]+12, -1.)
        self.assertEqual(self.validate(document, binary)["status"], "validated_render_attributes")

    def test_extension_overrides_and_texture_transforms_remain_rejected(self):
        for extension in ({"texCoord": 0}, {"texCoord": 1}, {"texCoord": 2}, {"rotation": .5}):
            document, binary = surface_fixture()
            document["materials"][0]["normalTexture"]["extensions"] = {"KHR_texture_transform": extension}
            with self.subTest(extension=extension), self.assertRaisesRegex(ValueError, "extension"):
                self.validate(document, binary)
        document, binary = surface_fixture()
        document["extensionsUsed"] = ["KHR_texture_transform"]
        with self.assertRaisesRegex(ValueError, "extensions"):
            self.validate(document, binary)

    def test_uv1_normalized_integer_accessor_is_supported(self):
        report = self.validate(*surface_fixture(integer_uv1=True))
        self.assertEqual(report["primitives"][0]["normal_map_uv_set"], 1)
        self.assertEqual(report["decoded_scalar_values"], 33)

    def test_existing_vertex_file_global_decode_bounds_are_not_widened(self):
        report = self.validate(*surface_fixture())
        self.assertEqual(report["limits"]["vertices"], MAX_VERTICES)
        self.assertEqual(report["limits"]["file_bytes"], MAX_FILE_BYTES)
        self.assertEqual(report["limits"]["decoded_global_values"], MAX_ACCESSOR_VALUES)
        self.assertEqual((MAX_VERTICES, MAX_FILE_BYTES, MAX_ACCESSOR_VALUES), (100000, 64*1024*1024, 2000000))
        self.assertEqual(report["limits"]["decoded_surface_values"], 1100000)
        document, binary = surface_fixture()
        document["accessors"][0]["count"] = MAX_VERTICES+1
        with patch("avatar_assets.three_d._GLB.accessor", side_effect=AssertionError("Decoded oversized surface")):
            with self.assertRaisesRegex(ValueError, "surface vertex count"):
                self.validate(document, binary)

    def test_no_normal_texture_does_not_claim_a_selected_uv_or_require_tangent(self):
        document, binary = surface_fixture()
        document["materials"][0].pop("normalTexture")
        document["meshes"][0]["primitives"][0]["attributes"].pop("TANGENT")
        report = self.validate(document, binary)
        self.assertIsNone(report["primitives"][0]["normal_map_uv_set"])
        self.assertFalse(report["primitives"][0]["normal_mapped"])


if __name__ == "__main__":
    unittest.main()
