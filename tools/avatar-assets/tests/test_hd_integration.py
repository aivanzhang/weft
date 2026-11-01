"""Regression coverage for material identity, mixed grooms and bounded indexed topology."""

from copy import deepcopy
import ast
import math
from pathlib import Path
import struct
import tempfile
import unittest

from avatar_assets.scene import validate_scene
from avatar_assets.three_d import _GLB, MAX_ACCESSOR_VALUES, MAX_TRIANGLE_INDICES, validate_glb
from scripts.blender_geometry import validate_card_material_export
from scripts.verify_quality_assets import AttributeReader, Budget as ProofBudget, UnsupportedProof, _material_proof
from tests.test_scene import analytic_scene
from tests.test_scene_quality import quality_scene
from tests.test_three_d import analytic_rig, glb_bytes


def indexed_grid():
    """A 20,301-vertex grid has 40,000 distinct triangles and 120,000 indices."""
    document, binary, contract = analytic_rig()
    width, height = 201, 101

    def add(rows, kind, component, replace=None):
        rows = list(rows)
        binary.extend(b"\0" * (-len(binary) % 4))
        offset = len(binary)
        packer = struct.Struct("<" + {5126: "f", 5125: "I", 5123: "H"}[component] * len(rows[0]))
        for row in rows:
            binary.extend(packer.pack(*row))
        document["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary) - offset})
        accessor = {"bufferView": len(document["bufferViews"]) - 1, "componentType": component,
                    "count": len(rows), "type": kind}
        if replace is None:
            document["accessors"].append(accessor)
            return len(document["accessors"]) - 1
        document["accessors"][replace] = accessor
        return replace

    add(((x / (width - 1), y / (height - 1), 0) for y in range(height) for x in range(width)), "VEC3", 5126, 0)
    add(((0, 1, 0, 0) for _ in range(width * height)), "VEC4", 5123, 1)
    add(((1 - x / (width - 1), x / (width - 1), 0, 0) for y in range(height) for x in range(width)), "VEC4", 5126, 2)
    indices = []
    for y in range(height - 1):
        for x in range(width - 1):
            start = y * width + x
            indices.extend((start, start + 1, start + width, start + 1, start + width + 1, start + width))
    index = add(((value,) for value in indices), "SCALAR", 5125)
    document["meshes"][0]["primitives"][0]["indices"] = index
    document["buffers"][0]["byteLength"] = len(binary)
    return document, binary, contract, index


class IndexedTopologyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document, cls.binary, cls.contract, cls.index = indexed_grid()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "analytic-grid.glb"
        self.path.write_bytes(glb_bytes(self.document, self.binary))

    def tearDown(self):
        self.temporary.cleanup()

    def test_many_indices_with_fewer_vertices_load_and_skin_correctly(self):
        result = validate_glb(self.path, self.contract)
        self.assertEqual(result["mesh_vertices"], 20301)
        self.assertTrue(result["motion"]["animation_verified"])
        self.assertEqual(result["limits"]["triangle_index_elements_per_accessor"], 300000)

    def test_wide_index_cache_cannot_bypass_default_vertex_count_limit(self):
        glb = _GLB(self.path)
        values = glb.accessor(self.index, "SCALAR", (5125,), count_limit=MAX_TRIANGLE_INDICES)
        self.assertEqual(len(values), 120000)
        charged = glb.value_count
        self.assertIs(glb.accessor(self.index, "SCALAR", (5125,), count_limit=MAX_TRIANGLE_INDICES), values)
        self.assertEqual(glb.value_count, charged)
        with self.assertRaisesRegex(ValueError, "Accessor count"):
            glb.accessor(self.index, "SCALAR", (5125,))
        with self.assertRaisesRegex(ValueError, "only for triangle indices"):
            glb.accessor(0, "VEC3", (5126,), count_limit=MAX_TRIANGLE_INDICES)

    def test_index_bound_and_vertex_bound_remain_distinct(self):
        for index, count in ((self.index, MAX_TRIANGLE_INDICES + 1), (0, 100001)):
            document = deepcopy(self.document)
            document["accessors"][index]["count"] = count
            self.path.write_bytes(glb_bytes(document, self.binary))
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Accessor count"):
                validate_glb(self.path, self.contract)

    def test_wide_indices_still_consume_the_two_million_value_budget(self):
        glb = _GLB(self.path)
        glb.value_count = MAX_ACCESSOR_VALUES - 120000
        self.assertEqual(len(glb.accessor(self.index, "SCALAR", (5125,), count_limit=MAX_TRIANGLE_INDICES)), 120000)
        self.assertEqual(glb.value_count, MAX_ACCESSOR_VALUES)
        other = _GLB(self.path)
        other.value_count = MAX_ACCESSOR_VALUES - 120000 + 1
        with self.assertRaisesRegex(ValueError, "Decoded accessor values"):
            other.accessor(self.index, "SCALAR", (5125,), count_limit=MAX_TRIANGLE_INDICES)

    def test_release_proof_uses_the_same_index_only_limit(self):
        reader = AttributeReader(_GLB(self.path), ProofBudget())
        self.assertEqual(len(reader.rows(self.index, "SCALAR", (5125,), count_limit=MAX_TRIANGLE_INDICES)), 120000)
        with self.assertRaisesRegex(ValueError, "Accessor count"):
            reader.rows(self.index, "SCALAR", (5125,))
        document = deepcopy(self.document)
        document["accessors"][self.index]["componentType"] = 5126
        self.path.write_bytes(glb_bytes(document, self.binary))
        with self.assertRaisesRegex(ValueError, "Invalid accessor bound"):
            AttributeReader(_GLB(self.path), ProofBudget())


class MaterialIdentityTests(unittest.TestCase):
    def test_one_sided_shell_material_requires_explicit_export_expectation(self):
        material = {"name": "fur", "alphaMode": "MASK",
                    "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        validate_card_material_export({"materials": [material]}, {"fur"}, {"fur": False})
        with self.assertRaisesRegex(ValueError, "Hair-card export"):
            validate_card_material_export({"materials": [material]}, {"fur"})

    def test_legacy_suffix_material_is_not_mistaken_for_generated_fur(self):
        scene = analytic_scene()
        scene["materials"]["body_cutout_fur"] = scene["materials"].pop("body")
        for part in scene["parts"]:
            if part["material"] == "body":
                part["material"] = "body_cutout_fur"
        self.assertEqual(validate_scene(scene), scene)
        validate_card_material_export({"materials": [{"name": "body_cutout_fur", "alphaMode": "OPAQUE"}]}, set())

    def test_generated_identity_handles_blender_collision_suffix(self):
        original = {"name": "body_cutout_fur", "alphaMode": "OPAQUE"}
        generated = {"name": "body_cutout_fur.001", "alphaMode": "MASK", "doubleSided": True,
                     "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        document = {"materials": [original, generated]}
        before = deepcopy(document)
        validate_card_material_export(document, {generated["name"]})
        self.assertEqual(document, before)
        for field, value in (("alphaMode", "OPAQUE"), ("doubleSided", False), ("pbrMetallicRoughness", {})):
            invalid = deepcopy(document)
            invalid["materials"][1][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Hair-card export"):
                validate_card_material_export(invalid, {generated["name"]})
        with self.assertRaisesRegex(ValueError, "Hair-card export"):
            validate_card_material_export({"materials": [original]}, {generated["name"]})

    def test_user_names_formerly_used_as_dictionary_keys_remain_allowed(self):
        scene = quality_scene()
        scene["materials"]["body__fur_cards"] = deepcopy(scene["materials"]["body"])
        scene["parts"][0]["fur"].update(mode="cards", width=.025)
        scene["parts"][1]["material"] = "body__fur_cards"
        self.assertEqual(validate_scene(scene), scene)

    def test_mixed_native_coverage_requires_separate_nonempty_card_materials(self):
        scene = quality_scene()
        scene["parts"][0]["fur"].update(mode="cards", width=.025)
        scene["parts"][1]["fur"] = deepcopy(scene["parts"][0]["fur"])
        scene["parts"][0]["native_groom"] = {"parents": 100, "children": 2, "length": .02,
                                             "radius": .0004, "roughness": .003, "seed": 8}
        with self.assertRaisesRegex(ValueError, "must all use native_groom"):
            validate_scene(scene)
        scene["parts"][1]["fur"]["count"] = 0
        validate_scene(scene)
        scene["parts"][1]["fur"]["count"] = 20
        scene["materials"]["other"] = deepcopy(scene["materials"]["body"])
        scene["parts"][1]["material"] = "other"
        validate_scene(scene)


class HDMaterialProofTests(unittest.TestCase):
    def test_card_binding_is_explicit_and_cannot_swap_another_preserved_image(self):
        scene = {"materials": {"body": {"color": [1, 1, 1, 1]}, "body_cutout_fur": {"roughness": .5}}}
        material = {"name": "body_cutout_fur.001", "alphaMode": "MASK", "alphaCutoff": .45, "doubleSided": True,
                    "pbrMetallicRoughness": {"roughnessFactor": .78, "metallicFactor": 0, "baseColorTexture": {"index": 0}}}
        document = {"materials": [material], "textures": [{"source": 0}, {"source": 1}]}
        backend = {"textures": [{"kind": "fur_atlas", "material": material["name"], "source_material": "body",
                                 "alpha_cutoff": .45, "sha256": "a" * 64}]}
        embedded = [{"sha256": "a" * 64}, {"sha256": "b" * 64}]
        result = _material_proof(document, scene, backend, embedded)
        self.assertTrue(result[0]["generated_hair_cards"])
        self.assertEqual(result[0]["source_material"], "body")
        material["pbrMetallicRoughness"]["baseColorTexture"]["index"] = 1
        with self.assertRaisesRegex(ValueError, "wrong preserved image"):
            _material_proof(document, scene, backend, embedded)
        material["pbrMetallicRoughness"]["baseColorTexture"]["index"] = 0
        material["alphaMode"] = "OPAQUE"
        with self.assertRaisesRegex(ValueError, "alpha mode"):
            _material_proof(document, scene, backend, embedded)
        material["alphaMode"] = "MASK"
        del backend["textures"][0]["source_material"]
        with self.assertRaisesRegex(UnsupportedProof, "explicit source-material"):
            _material_proof(document, scene, backend, embedded)
        scene["parts"] = [{"material": "body", "fur": {"mode": "cards", "count": 20}}]
        backend["textures"][0]["path"] = "textures/body-fur-atlas.png"
        with self.assertRaisesRegex(UnsupportedProof, "do not infer"):
            _material_proof(document, scene, backend, embedded)

    def test_legacy_named_material_stays_opaque(self):
        name = "body_cutout_fur"
        scene = {"materials": {name: {"roughness": .5, "metallic": 0}}}
        material = {"name": name, "pbrMetallicRoughness": {"roughnessFactor": .5, "metallicFactor": 0}}
        result = _material_proof({"materials": [material]}, scene, {"textures": []}, [])
        self.assertFalse(result[0]["generated_hair_cards"])
        self.assertEqual(result[0]["alpha_mode"], "OPAQUE")


class NativeQualityOptionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Test the helper's pure validator without importing Blender or starting a native process.
        path = Path(__file__).resolve().parents[1] / "scripts/blender_native_groom.py"
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_require", "validate_groom_settings")
                 or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MAX_FIBERS"
                                                         for target in node.targets)]
        namespace = {"math": math}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        cls.helper_validate = staticmethod(namespace["validate_groom_settings"])

    def test_native_helper_and_schema_allow_the_measured_density_envelope(self):
        base = {"parents": 10000, "children": 23, "length": .02, "radius": .0004, "roughness": .003, "seed": 9}
        for groom in (base, dict(base, parents=7741, children=30)):
            self.assertEqual(self.helper_validate(groom), groom)
            scene = quality_scene()
            scene["parts"][0]["native_groom"] = groom
            validate_scene(scene)
        for groom in (dict(base, children=24), dict(base, parents=10001, children=0), dict(base, parents=16, children=31)):
            with self.subTest(groom=groom), self.assertRaises(ValueError):
                self.helper_validate(groom)
            scene = quality_scene()
            scene["parts"][0]["native_groom"] = groom
            with self.assertRaises(ValueError):
                validate_scene(scene)

    def test_plush_is_opt_in_and_unknown_studio_fields_are_rejected(self):
        scene = quality_scene()
        original = deepcopy(scene)
        self.assertEqual(validate_scene(scene), original)
        studio = {"ground_z": 0, "world_strength": .2, "exposure": -.5}
        scene["studio"] = dict(studio)
        self.assertNotIn("preset", validate_scene(scene)["studio"])
        scene["studio"] = dict(studio, preset="plush")
        self.assertEqual(validate_scene(scene)["studio"]["preset"], "plush")
        for invalid in (dict(studio, preset="unknown"), dict(studio, arbitrary_shader="no")):
            scene["studio"] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_scene(scene)


if __name__ == "__main__":
    unittest.main()
