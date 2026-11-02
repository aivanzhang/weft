"""Analytic geometry and retained responses; no generation or native execution."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from avatar_assets.author_geometry import author_diagnostics, _coverage
from avatar_assets.pipeline_prompts import _author_design
from scripts.creature_rig_math import skin_weights

ROOT = Path(__file__).resolve().parents[1]
RETAINED = ROOT/'outputs/studio-f628641c66e74ea09f3b8517'


def fixture(folder, mutate=None):
    # Mechanical block corners plus two interior points; deliberately unlike a biped.
    points = [(0, 0, 0), (2, 0, 0), (0, 3, 0), (2, 3, 0),
              (0, 0, 1), (2, 0, 1), (0, 3, 1), (2, 3, 1), (1, 1.5, .5)]
    triangles = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    binary = b''.join(struct.pack('<3f', *p) for p in points)
    offset = len(binary)
    binary += struct.pack('<9H', *triangles)
    doc = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': len(binary)}],
           'bufferViews': [{'buffer': 0, 'byteLength': offset},
                           {'buffer': 0, 'byteOffset': offset, 'byteLength': 18}],
           'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': 9, 'type': 'VEC3'},
                         {'bufferView': 1, 'componentType': 5123, 'count': 9, 'type': 'SCALAR'}],
           'nodes': [{'mesh': 0}], 'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1}]}],
           'scenes': [{'nodes': [0]}], 'scene': 0}
    if mutate:
        mutate(doc)
    payload = json.dumps(doc).encode()
    payload += b' '*(-len(payload) % 4)
    binary += b'\0'*(-len(binary) % 4)
    raw = (struct.pack('<4sII', b'glTF', 2, 28+len(payload)+len(binary))
           + struct.pack('<II', len(payload), 0x4E4F534A)+payload
           + struct.pack('<II', len(binary), 0x004E4942)+binary)
    path = folder/'analytic.glb'
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    bounds = {'min': [0., -1., 0.], 'max': [2., 0., 3.]}
    native = {'source_sha256': digest, 'source_unchanged': True, 'rigs': [],
              'meshes': [{'vertices': 9, 'faces': 3}], 'bounds_blender_z_up': bounds,
              'rig_guides': {'source_sha256': digest, 'space': 'normalized_bounds_blender_z_up',
                             'bounds_blender_z_up': deepcopy(bounds)}}
    design = {'authored_rig': {'space': 'normalized_bounds_blender_z_up', 'height_meters': 1.,
        'bones': [{'name': 'Base', 'parent': None, 'head': [.5, .5, 0], 'tail': [.5, .5, 1]}],
        'base_bones': ['Base'], 'layers': [
            {'bones': ['Base'], 'min': [0, 0, 0], 'max': [1, 1, 1], 'feather': 0, 'min_vertices': 9},
            {'bones': ['Base'], 'min': [0, 0, 0], 'max': [1, 1, 1], 'feather': .1, 'min_vertices': 2}]},
        'action_bindings': [{'state_id': 'tilt', 'roles': [{'id': 'housing', 'root_bone': 'Base', 'bones': ['Base']}]}]}
    return path, digest, native, design


class AuthorGeometryTests(unittest.TestCase):
    def test_retained_proposals_aggregate_before_any_finishing(self):
        source = RETAINED/'inspection/original.glb'
        native_path = RETAINED/'inspection/native-inspection.json'
        self.assertTrue(source.is_file(), 'Retained source fixture is required')
        native = json.loads(native_path.read_text())
        paths = [source, native_path]+[RETAINED/f'agents/author-{i}/answer.json' for i in range(2)]
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        for attempt, counts, failing in [
            (0, [26957, 11845, 3802, 4000, 1043, 751, 6294, 1386], [1, 4, 5, 6]),
            (1, [26957, 14721, 4069, 4357, 1247, 857, 8213, 1230], [4, 5])]:
            design = _author_design(json.loads(paths[2+attempt].read_text()))
            untouched = deepcopy(design)
            result = author_diagnostics(design, source, before[0], native)
            self.assertEqual(result['status'], 'available', result)
            self.assertEqual([r['selected_vertices'] for r in result['layers']], counts)
            self.assertEqual([r['layer_index'] for r in result['layers'] if not r['passes']], failing)
            self.assertEqual(result['unknown_bone_references'], [] if attempt else [
                {'state_id': 'dance', 'role_id': 'body', 'field': 'bones', 'unknown_names': ['hip_l', 'hip_r']}])
            self.assertEqual(design, untouched)
        self.assertEqual(before, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])

    def test_mechanical_boundary_coverage_matches_native_formula(self):
        with tempfile.TemporaryDirectory() as folder:
            path, digest, native, design = fixture(Path(folder))
            result = author_diagnostics(design, path, digest, native)
            self.assertEqual(result['status'], 'available', result)
            self.assertEqual([r['selected_vertices'] for r in result['layers']], [9, 1])
            self.assertEqual(len(result['failures']), 1)
            # Independent native helper agrees without relaxing the proposal's threshold.
            rig = deepcopy(design['authored_rig'])
            rig['layers'][1]['min_vertices'] = 1
            points = np.array([[x, -z, y] for x, y, z in
                [(0, 0, 0), (2, 0, 0), (0, 3, 0), (2, 3, 0),
                 (0, 0, 1), (2, 0, 1), (0, 3, 1), (2, 3, 1), (1, 1.5, .5)]], float)
            low, high = points.min(axis=0), points.max(axis=0)
            _, proof = skin_weights(points, low, high, rig)
            self.assertEqual([r['selected_vertices'] for r in _coverage(points, low, high, rig)],
                             [r['selected_vertices'] for r in proof['layers']])
            self.assertEqual([r['selected_vertices'] for r in _coverage(points*2+4, low*2+4, high*2+4, rig)], [9, 1])

    def test_mismatched_source_counts_bounds_and_guide_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            path, digest, native, design = fixture(Path(folder))
            for mutate in [lambda n: n.update(source_sha256='0'*64),
                           lambda n: n['meshes'][0].update(vertices=10),
                           lambda n: n['meshes'][0].update(faces=4),
                           lambda n: n['bounds_blender_z_up']['min'].__setitem__(0, .00001),
                           lambda n: n['rig_guides'].update(source_sha256='0'*64)]:
                altered = deepcopy(native); mutate(altered)
                result = author_diagnostics(design, path, digest, altered)
                self.assertEqual(result['status'], 'unavailable')
                self.assertEqual(result['layers'], [])
            self.assertEqual(author_diagnostics(design, path, '0'*64, native)['status'], 'unavailable')

    def test_unsupported_transforms_topology_and_oversized_accessors(self):
        changes = [lambda d: d['nodes'][0].update(translation=[1, 0, 0]),
                   lambda d: d['nodes'].append({'mesh': 0}),
                   lambda d: d['meshes'][0]['primitives'].append(deepcopy(d['meshes'][0]['primitives'][0])),
                   lambda d: d['meshes'][0]['primitives'][0].update(targets=[{'POSITION': 0}]),
                   lambda d: d.update(skins=[{'joints': [0]}]),
                   lambda d: d['accessors'][0].update(count=100001),
                   lambda d: d['meshes'][0]['primitives'][0].update(mode=1)]
        with tempfile.TemporaryDirectory() as folder:
            for change in changes:
                path, digest, native, design = fixture(Path(folder), change)
                result = author_diagnostics(design, path, digest, native)
                self.assertEqual(result['status'], 'unavailable', result)
                self.assertEqual(result['layers'], [])

    def test_unknown_names_survive_unavailable_geometry_and_are_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            path, digest, native, design = fixture(Path(folder))
            role = design['action_bindings'][0]['roles'][0]
            role.update(root_bone='MissingRoot', bones=['MissingTarget', 'Other'])
            native['meshes'][0]['vertices'] = 10
            result = author_diagnostics(design, path, digest, native)
            self.assertEqual(result['status'], 'unavailable')
            self.assertEqual(len(result['unknown_bone_references']), 2)
            self.assertEqual(len(result['failures']), 2)
            design['authored_rig']['layers'] *= 17
            self.assertEqual(author_diagnostics(design, path, digest, native)['status'], 'unavailable')
            design = {'action_bindings': []}
            self.assertEqual(author_diagnostics(design, path, digest, native)['status'], 'not_applicable')
            self.assertEqual(author_diagnostics([], path, digest, native)['status'], 'unavailable')


if __name__ == '__main__':
    unittest.main()
