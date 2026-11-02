"""Analytic triangles test validation; they are not generated character assets."""

import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

from avatar_assets.three_d import validate_glb, unsupported_3d


def analytic_rig():
    binary, views, accessors = bytearray(), [], []

    def add(rows, kind, component=5126):
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        fmt = {5126: "f", 5123: "H"}[component]
        for row in rows:
            binary.extend(struct.pack("<" + fmt*len(row), *row))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary)-offset})
        accessors.append({"bufferView": len(views)-1, "componentType": component,
                          "count": len(rows), "type": kind})
        return len(accessors)-1

    position = add([(0, 0, 0), (1, 0, 0), (1, 1, 0)], "VEC3")
    joints = add([(0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)], "VEC4", 5123)
    weights = add([(1, 0, 0, 0)]*3, "VEC4")
    identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    tip_inverse = identity.copy()
    tip_inverse[13] = -1
    inverse = add([identity, tip_inverse], "MAT4")
    time = add([(0,), (.5,), (1,)], "SCALAR")
    angle = math.sqrt(.5)
    rotation = add([(0, 0, 0, 1), (0, 0, angle, angle), (0, 0, 0, 1)], "VEC4")
    doc = {
        "asset": {"version": "2.0", "generator": "analytic-test-fixture"},
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
        "scene": 0, "scenes": [{"nodes": [0, 1]}],
        "nodes": [{"name": "triangle", "mesh": 0, "skin": 0},
                  {"name": "base", "children": [2]},
                  {"name": "tip", "translation": [0, 1, 0], "children": [3]},
                  {"name": "accessory", "translation": [.1, 0, 0]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": position, "JOINTS_0": joints,
                                                    "WEIGHTS_0": weights}, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [.1, .5, .3, 1]}}],
        "skins": [{"joints": [1, 2], "skeleton": 1, "inverseBindMatrices": inverse}],
        "animations": [{"name": "bend-and-return", "samplers": [{"input": time, "output": rotation}],
                        "channels": [{"sampler": 0, "target": {"node": 2, "path": "rotation"}}]}],
    }
    contract = {
        "schema_version": 1, "units": "meters",
        "axes": {"up": "+Y", "forward": "+Z", "handedness": "right"},
        "rest_pose": "Analytic triangle default node transforms; not a character",
        "bone_to_simulated_joint": {"base": "fixture_root", "tip": "fixture_hinge"},
        "target": {"name": "analytic fixture, NOT Weft", "joint_names": ["fixture_root", "fixture_hinge"]},
        "authored_assumptions": ["All geometry and rig structure are analytic test data"],
        "attachments": [{"node": "accessory", "parent_bone": "tip", "mode": "rigid_child",
                         "translation_m": [.1, 0, 0], "rotation_xyzw": [0, 0, 0, 1],
                         "placement": "Analytic anchor, not a dragon"}],
    }
    return doc, binary, contract


def glb_bytes(doc, binary):
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    binary = bytes(binary) + b"\x00" * (-len(binary) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28+len(encoded)+len(binary))
            + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


class ThreeDTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/"analytic.glb"
        self.doc, self.binary, self.contract = analytic_rig()

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        self.path.write_bytes(glb_bytes(self.doc, self.binary))
        return validate_glb(self.path, self.contract)

    def test_load_and_observe_midpoint_motion_when_endpoints_match(self):
        result = self.validate()
        self.assertTrue(result["loading_verified"])
        self.assertTrue(result["motion"]["animation_verified"])
        self.assertGreater(result["motion"]["animations"][0]["max_sampled_displacement_m"], 1)
        self.assertFalse(result["weft_runtime_verified"])
        self.assertEqual(result["mesh_vertices"], 3)

    def test_rig_without_animation_does_not_claim_animation(self):
        self.doc["animations"] = []
        result = self.validate()
        self.assertFalse(result["motion"]["animation_verified"])

    def test_optional_skeleton_is_derived_from_joint_hierarchy(self):
        self.doc["skins"][0].pop("skeleton")
        result = self.validate()
        self.assertEqual(result["skeleton_roots"][0],
                         {"node": 1, "name": "base", "source": "derived_common_ancestor"})
        self.assertTrue(result["motion"]["animation_verified"])

    def test_single_constant_key_does_not_claim_motion(self):
        self.doc["accessors"][4]["count"] = 1
        self.doc["accessors"][5]["count"] = 1
        result = self.validate()
        self.assertFalse(result["motion"]["animation_verified"])
        self.assertEqual(result["motion"]["animations"][0]["sample_times_seconds"], [0.])

    def test_static_keyframes_do_not_claim_animation(self):
        view = self.doc["bufferViews"][5]
        for i in range(3):
            struct.pack_into("<4f", self.binary, view["byteOffset"]+16*i, 0, 0, 0, 1)
        self.assertFalse(self.validate()["motion"]["animation_verified"])

    def test_unrigged_mesh_is_rejected(self):
        self.doc.pop("skins")
        with self.assertRaisesRegex(ValueError, "skeleton"):
            self.validate()

    def test_incomplete_and_unknown_joint_mapping(self):
        for mapping in ({"base": "fixture_root"}, {"base": "fixture_root", "tip": "made_up"}):
            with self.subTest(mapping=mapping):
                self.contract["bone_to_simulated_joint"] = mapping
                with self.assertRaises(ValueError):
                    self.validate()

    def test_attachment_requires_actual_parent_and_transform(self):
        self.contract["attachments"][0]["parent_bone"] = "base"
        with self.assertRaisesRegex(ValueError, "direct child"):
            self.validate()

    def test_rejects_invalid_weights_and_joint_indices(self):
        for index, fmt, value in ((2, "f", .2), (1, "H", 8)):
            with self.subTest(index=index):
                self.doc, self.binary, self.contract = analytic_rig()
                struct.pack_into("<"+fmt, self.binary, self.doc["bufferViews"][index]["byteOffset"], value)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_rejects_nonfinite_values(self):
        struct.pack_into("<f", self.binary, 0, float("nan"))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.validate()

    def test_rejects_cycles(self):
        self.doc["nodes"][2]["children"].append(1)
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            self.validate()

    def test_rejects_external_and_data_uris_without_loading(self):
        for uri in ("https://example.invalid/mesh.bin", "file:///etc/passwd", "data:application/octet-stream;base64,AA=="):
            with self.subTest(uri=uri):
                self.doc["buffers"][0]["uri"] = uri
                with self.assertRaisesRegex(ValueError, "URIs"):
                    self.validate()

    def test_rejects_huge_accessor_before_decoding(self):
        self.doc["accessors"][0]["count"] = 2**30
        with self.assertRaisesRegex(ValueError, "bounds"):
            self.validate()

    def test_rejects_out_of_range_accessor(self):
        self.doc["accessors"][0]["byteOffset"] = 400
        with self.assertRaisesRegex(ValueError, "exceeds buffer"):
            self.validate()

    def test_rejects_large_file_before_reading(self):
        with self.path.open("wb") as file:
            file.truncate(64*1024*1024+1)
        with self.assertRaisesRegex(ValueError, "file size"):
            validate_glb(self.path, self.contract)

    def test_rejects_truncated_glb(self):
        self.path.write_bytes(glb_bytes(self.doc, self.binary)[:-1])
        with self.assertRaisesRegex(ValueError, "header"):
            validate_glb(self.path, self.contract)

    def test_rejects_undeclared_extension_and_extra_weights(self):
        self.doc["nodes"][2]["extensions"] = {"made_up_extension": {"url": "https://example.invalid"}}
        with self.assertRaisesRegex(ValueError, "extension"):
            self.validate()
        self.doc["nodes"][2].pop("extensions")
        self.doc["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_2"] = 2
        with self.assertRaisesRegex(ValueError, "four skin influences"):
            self.validate()

    def test_rejects_singular_inverse_bind(self):
        view = self.doc["bufferViews"][3]
        struct.pack_into("<f", self.binary, view["byteOffset"], 0.)
        with self.assertRaisesRegex(ValueError, "Singular inverse"):
            self.validate()

    def test_rejects_ambiguous_bone_name(self):
        self.doc["nodes"][3]["name"] = "tip"
        with self.assertRaisesRegex(ValueError, "unique names"):
            self.validate()

    def test_unsupported_is_explicit(self):
        result = unsupported_3d("panda with dragon", "Reference contains no mesh or rig")
        self.assertEqual(result["status"], "unsupported")
        self.assertFalse(result["geometry_created"])
        self.assertFalse(result["rig_created"])
        self.assertFalse(result["animation_verified"])


if __name__ == "__main__":
    unittest.main()
