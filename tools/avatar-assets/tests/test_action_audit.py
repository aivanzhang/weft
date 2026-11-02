"""Real analytic GLB skinning tests; no generated-character or visual-quality claims."""

from copy import deepcopy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from avatar_assets.action_audit import actual_bone_parents, audit_glb, write_audit
from avatar_assets.safety import sha256
from avatar_assets.three_d import _local_matrix
from test_three_d import glb_bytes


def fixture(*, travel=True, flap=True, rotate_root=False, constant_wing=False, coat=False, body_triangles=4,
            scale=(1, 1, 1), body_wing_weight=0., weight_scale=1., uv_seams=False):
    binary, views, accessors = bytearray(), [], []

    def add(rows, kind, component=5126):
        binary.extend(b"\0"*(-len(binary) % 4))
        offset = len(binary)
        fmt = {5126: "f", 5123: "H"}[component]
        for row in rows:
            binary.extend(struct.pack("<"+fmt*len(row), *row))
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(binary)-offset})
        accessors.append({"bufferView": len(views)-1, "componentType": component, "count": len(rows), "type": kind})
        return len(accessors)-1

    body = []
    for index in range(body_triangles):
        x = -.3+.6*(index % 17)/16
        y = .9*(index % 23)/22
        body.extend([(x, y, -.1), (x+.03, y+.08, .1), (x-.03, y+.1, 0)])
    wing = [(x, y, z) for x, y, z in ((.3, .7, 0), (.6, .7, .05), (.7, .85, 0),
             (.3, .7, 0), (.7, .85, 0), (.45, .9, -.05), (.3, .7, 0), (.45, .9, -.05), (.4, .75, -.1),
             (.3, .7, 0), (.4, .75, -.1), (.6, .7, .05))]
    positions = add(body+wing, "VEC3")
    joints = add([(0, 1, 0, 0)]*len(body)+[(1, 0, 0, 0)]*len(wing), "VEC4", 5123)
    rows = [(1-body_wing_weight, body_wing_weight, 0, 0)]*len(body)+[(1, 0, 0, 0)]*len(wing)
    weights = add([tuple(value*weight_scale for value in row) for row in rows], "VEC4")
    uv = add([(index % 2, (index//2) % 2) for index in range(len(body)+len(wing))], 'VEC2') if uv_seams else None
    root = {"name": "Root", "children": [2], "scale": list(scale)}
    wing_node = {"name": "Wing", "translation": [.3, .7, 0]}
    root_matrix = np.asarray(_local_matrix(root)).reshape(4, 4, order="F")
    wing_matrix = root_matrix@np.asarray(_local_matrix(wing_node)).reshape(4, 4, order="F")
    inverse = add([np.linalg.inv(root_matrix).reshape(-1, order="F"), np.linalg.inv(wing_matrix).reshape(-1, order="F")], "MAT4")
    seconds = [index*.5 for index in range(9)]
    times = add([(value,) for value in seconds], "SCALAR")
    root_positions = [(0 if not travel else .1*math.sin(index*math.pi/2),
                       .15*(index % 2) if travel else 0, 0) for index in range(9)]
    translated = add(root_positions, "VEC3")
    wing_angles = ([.4]*9 if constant_wing else [(.35*math.sin(index*math.pi/2) if flap else 0) for index in range(9)])
    rotated = add([(0, 0, math.sin(angle/2), math.cos(angle/2)) for angle in wing_angles], "VEC4")
    root_angles = [.3*math.sin(index*math.pi/2) if rotate_root else 0 for index in range(9)]
    root_rotated = add([(0, 0, math.sin(angle/2), math.cos(angle/2)) for angle in root_angles], "VEC4")
    primitive = {"attributes": {"POSITION": positions, "JOINTS_0": joints, "WEIGHTS_0": weights}, "material": 0}
    if uv is not None:
        primitive['attributes']['TEXCOORD_0'] = uv
    primitives = [primitive]
    materials = [{"name": "AnatomicalSurface", "pbrMetallicRoughness": {"baseColorFactor": [.4, .6, .3, 1]}}]
    if coat:
        coat_position = add([(50, 50, 50), (50, 50, 51), (51, 50, 50)], "VEC3")
        materials.append({"name": "PortableCoat_0", "pbrMetallicRoughness": {}})
        primitives.append({"attributes": {"POSITION": coat_position}, "material": 1})
    doc = {"asset": {"version": "2.0", "generator": "analytic-weighted-audit-fixture"},
           "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
           "scene": 0, "scenes": [{"nodes": [0, 1]}],
           "nodes": [{"name": "Anatomy", "mesh": 0, "skin": 0}, root, wing_node],
           "meshes": [{"name": "AnalyticBodyAndWing", "primitives": primitives}], "materials": materials,
           "skins": [{"joints": [1, 2], "skeleton": 1, "inverseBindMatrices": inverse}],
           "animations": [{"name": "move", "samplers": [{"input": times, "output": translated},
                           {"input": times, "output": rotated}, {"input": times, "output": root_rotated}],
                           "channels": [{"sampler": 0, "target": {"node": 1, "path": "translation"}},
                                        {"sampler": 1, "target": {"node": 2, "path": "rotation"}},
                                        {"sampler": 2, "target": {"node": 1, "path": "rotation"}}]}]}
    return doc, binary


def intent(kinds=("travel", "articulation"), *, body=False):
    state = {"id": "move", "meaning": "Analytic articulated motion, not a character action", "user_request": "",
             "roles": [{"id": "subject", "description": "Analytic body" if body else "Analytic wing"}],
             "requirements": [{"id": name, "kind": name, "role": "subject", "description": "Observed " + name} for name in kinds]}
    bindings = [{"state_id": "move", "roles": [{"id": "subject", "root_bone": "Root",
                                                "bones": ["Root", "Wing"] if body else ["Wing"]}]}]
    return [state], bindings


class ActionAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"analytic.glb"

    def save(self, doc, binary):
        self.path.write_bytes(glb_bytes(doc, binary))

    def audit(self, kinds=("travel", "articulation"), body=False, **options):
        self.save(*fixture(**options))
        states, bindings = intent(kinds, body=body)
        return audit_glb(self.path, states, bindings, 1.)

    def test_real_skin_travel_and_wing_articulation_are_independent(self):
        result = self.audit()
        self.assertTrue(result["passed"])
        metrics = result["states"][0]["evidence"]["roles"]["subject"]
        self.assertGreater(metrics["max_travel_m"], .15)
        self.assertGreater(metrics["max_root_relative_displacement_m"], .1)
        self.assertGreaterEqual(metrics["moving_vertices"], 8)
        self.assertEqual(result["actual_bone_parents"], {"Root": None, "Wing": "Root"})
        self.assertEqual(result["sampled_frames"], 97)
        self.assertLessEqual(len(result["states"][0]["full_skin_sample_seconds"]), 8)
        self.assertFalse(result["semantic_approval"])
        self.assertFalse(result["physics_verified"])

    def test_stationary_body_can_pass_while_its_child_appendage_moves(self):
        self.save(*fixture(travel=False, flap=True))
        states, bindings = intent(("articulation",))
        states[0]["roles"].append({"id": "base", "description": "Stationary supporting body"})
        states[0]["requirements"].append({"id": "steady-base", "kind": "stability", "role": "base",
                                          "description": "Keep the body steady throughout the appendage gesture."})
        bindings[0]["roles"].append({"id": "base", "root_bone": "Root", "bones": ["Root"]})
        result = audit_glb(self.path, states, bindings, 1.)
        self.assertTrue(result["passed"])
        metrics = result["states"][0]["evidence"]["roles"]
        self.assertEqual(metrics["base"]["weighted_vertices"], 12)
        self.assertEqual(metrics["base"]["active_control_count"], 0)
        self.assertLess(metrics["base"]["max_world_displacement_m"], 1e-12)
        self.assertGreater(metrics["subject"]["max_root_relative_displacement_m"], .1)
        detail = result["states"][0]["role_details"]["base"]
        self.assertEqual(detail["surface_selection"], "direct_target_bone_weight_sum_at_least_0.5")

    def test_stability_detects_parent_carried_feet_and_parent_rotation(self):
        for translate, rotate in ((True, False), (False, True)):
            with self.subTest(translate=translate, rotate=rotate):
                result = self.audit(("stability",), travel=translate, rotate_root=rotate, flap=False)
                metrics = result["states"][0]["evidence"]["roles"]["subject"]
                self.assertLess(metrics["max_root_relative_displacement_m"], 1e-12)
                self.assertEqual(metrics["active_control_count"], 0)
                self.assertGreater(metrics["max_world_displacement_m"], .01)
                self.assertFalse(result["passed"])

    def test_fully_stationary_weighted_roles_do_not_need_animation(self):
        result = self.audit(("stability",), travel=False, flap=False)
        self.assertTrue(result["passed"])
        metrics = result["states"][0]["evidence"]["roles"]["subject"]
        self.assertEqual(metrics["moving_vertices"], 0)
        self.assertEqual(metrics["active_control_count"], 0)
        result = self.audit(("stability",), travel=False, constant_wing=True)
        self.assertTrue(result["passed"])

    def test_stability_requires_meaningful_direct_weighted_coverage(self):
        self.save(*fixture(travel=False, body_triangles=1))
        states, bindings = intent(("stability",))
        bindings[0]["roles"][0]["bones"] = ["Root"]
        with self.assertRaisesRegex(ValueError, "eight predominantly weighted direct target vertices"):
            audit_glb(self.path, states, bindings, 1.)

    def test_stability_witnesses_keep_selection_threshold_and_original_weights(self):
        self.save(*fixture(travel=False, body_wing_weight=.2, weight_scale=.99995))
        states, bindings = intent(('stability',))
        bindings[0]['roles'][0]['bones'] = ['Root']
        result = audit_glb(self.path, states, bindings, 1.)
        state = result['states'][0]
        evidence = state['role_details']['subject']['stability_diagnostics']
        self.assertFalse(result['passed'])
        self.assertEqual(evidence['selected_exported_vertices'], 12)
        self.assertEqual(evidence['threshold_m'], state['requirement_results']['requirements'][0]['maximum_displacement_m'])
        self.assertEqual(len(evidence['vertices']), 4)
        self.assertGreaterEqual(evidence['violating_exported_vertices'], 4)
        self.assertIsNone(evidence['layer_matches'])
        for row in evidence['vertices']:
            self.assertLess(row['exported_vertex_index'], 12)
            self.assertEqual(row['node_index'], 0)
            self.assertEqual(row['primitive_index'], 0)
            self.assertEqual(row['primitive_vertex_index'], row['exported_vertex_index'])
            self.assertEqual([w['bone'] for w in row['influences']], ['Root', 'Wing'])
            self.assertEqual([w['slot'] for w in row['influences']], [0, 1])
            self.assertAlmostEqual(sum(w['source_weight'] for w in row['influences']), .99995, places=6)
            self.assertAlmostEqual(sum(w['audit_weight'] for w in row['influences']), 1., places=12)
            measured = np.linalg.norm(np.array(row['peak_world_gltf_m'])-row['initial_world_gltf_m'])
            self.assertAlmostEqual(measured, row['max_world_displacement_m'], places=12)
            self.assertGreater(measured, evidence['threshold_m'])
            self.assertIn(row['peak_second'], evidence['examined_seconds'])

    def test_stability_witness_peak_time_order_and_uv_seam_copies(self):
        doc, binary = fixture(travel=False, uv_seams=True)
        accessor = doc['accessors'][doc['animations'][0]['samplers'][1]['output']]
        view = doc['bufferViews'][accessor['bufferView']]
        for index, angle in enumerate((0, .2, 0, .6, 0, .2, 0, .1, 0)):
            struct.pack_into('<ffff', binary, view['byteOffset']+index*16, 0, 0, math.sin(angle/2), math.cos(angle/2))
        self.save(doc, binary)
        before = sha256(self.path)
        result = audit_glb(self.path, *intent(('stability',)), 1.)
        evidence = result['states'][0]['role_details']['subject']['stability_diagnostics']
        rows = evidence['vertices']
        self.assertEqual([row['exported_vertex_index'] for row in rows[:2]], [14, 16])
        self.assertEqual(rows[0]['rest_world_gltf_m'], rows[1]['rest_world_gltf_m'])
        self.assertEqual(rows[0]['influences'], rows[1]['influences'])
        self.assertEqual([row['peak_second'] for row in rows], [1.5]*len(rows))
        self.assertEqual(rows, sorted(rows, key=lambda row: (-row['max_world_displacement_m'], row['exported_vertex_index'])))
        self.assertIn('not distinct physical points', evidence['index_space'])
        self.assertEqual(sha256(self.path), before)

    def test_stability_witness_coordinates_convert_asymmetric_bounds_to_blender(self):
        doc, binary = fixture(travel=False)
        accessor = doc['accessors'][0]; view = doc['bufferViews'][accessor['bufferView']]
        positions = []
        for index in range(accessor['count']):
            offset = view['byteOffset']+index*12
            x, y, z = struct.unpack_from('<fff', binary, offset)
            struct.pack_into('<fff', binary, offset, x, y, z*2+.35)
            positions.append(struct.unpack_from('<fff', binary, offset))
        self.save(doc, binary)
        result = audit_glb(self.path, *intent(('stability',)), 1.)
        evidence = result['states'][0]['role_details']['subject']['stability_diagnostics']
        blender = np.array(positions)[:, [0, 2, 1]]*np.array([1, -1, 1])
        low, span = blender.min(axis=0), np.ptp(blender, axis=0)
        np.testing.assert_allclose(evidence['rest_bounds_blender_z_up_m']['min'], low, atol=1e-7)
        np.testing.assert_allclose(evidence['rest_bounds_blender_z_up_m']['max'], low+span, atol=1e-7)
        differs_from_unconverted = False
        gltf = np.array(positions)
        for row in evidence['vertices']:
            world = np.array(row['rest_world_gltf_m'])
            expected = (world[[0, 2, 1]]*np.array([1, -1, 1])-low)/span
            np.testing.assert_allclose(row['rest_normalized_blender_z_up'], expected, atol=1e-7)
            wrong = (world-gltf.min(axis=0))/np.ptp(gltf, axis=0)
            differs_from_unconverted |= abs(expected[1]-wrong[2]) > .1
        self.assertTrue(differs_from_unconverted)
        self.assertIn('glTF X,-Z,Y', evidence['coordinate_space'])

    def test_stability_witness_rows_are_empty_when_fixed_and_bounded_when_large(self):
        result = self.audit(('stability',), travel=False, flap=False)
        evidence = result['states'][0]['role_details']['subject']['stability_diagnostics']
        self.assertEqual(evidence['vertices'], [])
        self.assertEqual(evidence['violating_exported_vertices'], 0)
        result = self.audit(('stability',), body=True, travel=True, flap=False, body_triangles=1000)
        evidence = result['states'][0]['role_details']['subject']['stability_diagnostics']
        self.assertEqual(evidence['selected_exported_vertices'], 3012)
        self.assertGreater(evidence['violating_exported_vertices'], 3000)
        self.assertEqual(evidence['maximum_rows'], 4)
        self.assertEqual(len(evidence['vertices']), 4)
        self.assertEqual(evidence['examined_seconds'], result['states'][0]['full_skin_sample_seconds'])

    def test_rigid_global_translation_rotation_scale_do_not_become_flapping(self):
        result = self.audit(flap=False, rotate_root=True, scale=(2, 1.2, .8))
        metrics = result["states"][0]["evidence"]["roles"]["subject"]
        self.assertGreater(metrics["max_travel_m"], .15)
        self.assertLess(metrics["max_root_relative_displacement_m"], 1e-12)
        self.assertEqual(metrics["moving_vertices"], 0)
        self.assertEqual(metrics["active_control_count"], 0)
        self.assertFalse(result["passed"])

    def test_constant_preposed_wing_is_not_animation(self):
        result = self.audit(constant_wing=True, rotate_root=True)
        metrics = result["states"][0]["evidence"]["roles"]["subject"]
        self.assertLess(metrics["max_root_relative_displacement_m"], 1e-12)
        self.assertEqual(metrics["moving_vertices"], 0)
        self.assertEqual(metrics["active_control_count"], 0)
        self.assertFalse(result["passed"])

    def test_standing_flaps_cannot_satisfy_flight_travel(self):
        result = self.audit(travel=False)
        rows = result["states"][0]["requirement_results"]["requirements"]
        self.assertEqual([row["status"] for row in rows], ["failed", "passed"])
        self.assertFalse(result["passed"])

    def test_dance_gate_requires_rhythm_and_two_independent_controls(self):
        result = self.audit(("rhythm",), body=True)
        self.assertTrue(result["passed"])
        metrics = result["states"][0]["evidence"]["roles"]["subject"]
        self.assertGreaterEqual(metrics["rhythmic_reversals"], 3)
        self.assertEqual(metrics["active_control_count"], 2)
        root_only = self.audit(("rhythm",), body=True, flap=False)
        self.assertFalse(root_only["passed"])
        self.assertEqual(root_only["states"][0]["evidence"]["roles"]["subject"]["active_control_count"], 1)

    def test_joint_targeted_probes_find_small_wing_and_exclude_coat(self):
        result = self.audit(body_triangles=1000, coat=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["anatomical_vertices"], 3012)
        self.assertEqual(result["aggregate_vertices"], 3015)
        self.assertEqual(result["probe_count"], 12)
        self.assertEqual(len(result["excluded_surfaces"]), 1)
        metrics = result["states"][0]["evidence"]["roles"]["subject"]
        self.assertAlmostEqual(metrics["scale_m"], .4, places=5)
        self.assertEqual(metrics["weighted_vertices"], 12)

    def test_missing_invalid_or_oversized_rigs_fail_closed(self):
        for mutate in (lambda d: d.update(skins=[]), lambda d: d.update(animations=[]),
                       lambda d: d["skins"][0].update(joints=[1, 99]),
                       lambda d: d["skins"][0].pop("inverseBindMatrices"),
                       lambda d: d["nodes"][2].update(children=[1]),
                       lambda d: d["accessors"][0].update(count=100001),
                       lambda d: d["animations"][0]["channels"][0]["target"].update(path="weights")):
            doc, binary = fixture(); mutate(doc); self.save(doc, binary)
            with self.assertRaises(ValueError):
                audit_glb(self.path, *intent(), 1.)
        self.save(*fixture())
        states, bindings = intent()
        bindings[0]["roles"][0]["bones"] = ["MissingWing"]
        with self.assertRaisesRegex(ValueError, "Unknown action target"):
            audit_glb(self.path, states, bindings, 1.)
        with self.assertRaisesRegex(ValueError, "height"):
            audit_glb(self.path, *intent(), float("nan"))

    def test_hash_bound_persisted_report_is_not_overwritten(self):
        self.save(*fixture())
        states, bindings = intent()
        request = {"schema_version": 1, "states": states, "action_bindings": bindings,
                   "height_meters": 1., "source_sha256": sha256(self.path)}
        intent_path, report_path = self.path.parent/"intent.json", self.path.parent/"audit.json"
        intent_path.write_text(json.dumps(request))
        result = write_audit(self.path, intent_path, report_path)
        self.assertEqual(result["model_sha256"], request["source_sha256"])
        self.assertEqual(result["intent_sha256"], sha256(intent_path))
        self.assertEqual(json.loads(report_path.read_text())["states"][0]["evidence"], result["states"][0]["evidence"])
        self.assertEqual(actual_bone_parents(self.path), {"Root": None, "Wing": "Root"})
        with self.assertRaisesRegex(ValueError, "already exists"):
            write_audit(self.path, intent_path, report_path)
        request["source_sha256"] = "f"*64
        intent_path.write_text(json.dumps(request))
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            write_audit(self.path, intent_path, self.path.parent/"second.json")


if __name__ == "__main__":
    unittest.main()
