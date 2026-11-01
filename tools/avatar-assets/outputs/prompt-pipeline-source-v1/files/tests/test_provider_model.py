"""Hostile generated GLB checks using explicitly analytic triangles."""

import io
import struct
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from avatar_assets.provider_model import preflight
from test_three_d import analytic_rig, glb_bytes


class ProviderModelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/"fixture.glb"
        self.doc, self.binary, _ = analytic_rig()
        offset = len(self.binary)
        self.binary.extend(struct.pack("<HHH", 0, 1, 2))
        self.doc["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": 6})
        self.doc["accessors"].append({"bufferView": len(self.doc["bufferViews"])-1,
                                      "componentType": 5123, "count": 3, "type": "SCALAR"})
        self.doc["buffers"][0]["byteLength"] = len(self.binary)
        self.doc["meshes"][0]["primitives"][0]["indices"] = len(self.doc["accessors"])-1

    def tearDown(self):
        self.temp.cleanup()

    def check(self):
        self.path.write_bytes(glb_bytes(self.doc, self.binary))
        return preflight(self.path)

    def test_static_geometry_is_not_called_animation(self):
        self.doc.pop("skins")
        self.doc.pop("animations")
        result = self.check()
        self.assertEqual(result["vertices"], 3)
        self.assertEqual(result["animations"], 0)
        self.assertEqual(result["skins"], 0)

    def test_oversized_accessor_rejected_before_native_import(self):
        self.doc["accessors"][0]["count"] = 100001
        with self.assertRaisesRegex(ValueError, "count outside bounds"):
            self.check()

    def test_external_texture_is_not_fetched(self):
        self.doc["images"] = [{"uri": "https://example.invalid/texture.png"}]
        with self.assertRaisesRegex(ValueError, "URIs are unsupported"):
            self.check()

    def test_oversized_embedded_image_header_rejected(self):
        data = io.BytesIO()
        Image.new("1", (4097, 4096)).save(data, format="PNG")
        offset = len(self.binary)
        self.binary.extend(data.getvalue())
        self.doc["buffers"][0]["byteLength"] = len(self.binary)
        self.doc["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(data.getvalue())})
        self.doc["images"] = [{"bufferView": len(self.doc["bufferViews"])-1, "mimeType": "image/png"}]
        with self.assertRaisesRegex(ValueError, "16MP"):
            self.check()

    def test_out_of_range_indices_rejected(self):
        self.binary[-2:] = struct.pack("<H", 3)
        with self.assertRaisesRegex(ValueError, "triangle indices"):
            self.check()

    def test_compression_and_morphs_rejected(self):
        primitive = self.doc["meshes"][0]["primitives"][0]
        primitive["targets"] = [{"POSITION": 0}]
        with self.assertRaisesRegex(ValueError, "morph"):
            self.check()


if __name__ == "__main__":
    unittest.main()
