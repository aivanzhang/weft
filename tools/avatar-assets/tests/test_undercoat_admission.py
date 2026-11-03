"""Analytic custody and UV planning; no Blender process, render or appearance approval."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from avatar_assets import undercoat_admission as host
from avatar_assets.safety import artifacts, read_json, sha256
from scripts import blender_undercoat_admission as native
from scripts.blender_fur_cards import tuft_export_budget


def settings(**updates):
    return {'profile': 'short-pile-v1', 'resolution': 1024, 'seed': 17, 'pitch_m': .004,
            'strength': .22, 'roughness': .92, 'min_faces': 1, 'max_faces': 1,
            'uv_policy': 'material-adaptive-v1', **updates}


def geometry():
    points = np.array([[0, 0, 0], [.02, 0, 0], [.02, .02, 0], [0, .02, 0]], dtype='<f4')
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype='<i4')
    edges = np.array([[0, 1], [1, 2], [2, 0], [2, 3], [3, 0]], dtype='<i4')
    uv = np.array([[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 1], [0, 1]]], dtype='<f4')
    return points, edges, faces, uv


class Collection(list):
    def foreach_get(self, name, target):
        target[:] = np.asarray([getattr(value, name) for value in self]).ravel()


class Layers(list):
    active_index = 0


class Object(NS):
    def get(self, key, default=None):
        return self.properties.get(key, default)


def mesh():
    points, edges, faces, uv = geometry()
    return NS(vertices=Collection(NS(index=i, co=p, groups=[]) for i, p in enumerate(points)),
              edges=Collection(NS(vertices=e) for e in edges),
              polygons=Collection(NS(loop_start=i*3, loop_total=3, material_index=0) for i in range(2)),
              loops=Collection(NS(vertex_index=int(v)) for v in faces.ravel()),
              materials=[NS(name='AnalyticSurface')], attributes={},
              uv_layers=Layers([NS(name='UVMap', active_render=True,
                  data=Collection(NS(uv=v) for v in uv.reshape(-1, 2)))]))


def rest():
    points, edges, _, _ = geometry()
    return {'schema_version': 1, 'status': 'measured', 'coordinate_space': 'mesh_local_after_world_transform_applied',
            'units': 'meters', 'rest_array_encoding': 'little-endian float64 XYZ',
            'edge_array_encoding': 'little-endian int32 pairs', 'unchanged_topology_required': True,
            'vertices': 4, 'edges': 5, 'bone_names': ['root'],
            'rest_sha256': native.array_hash(points.astype('<f8')), 'edges_sha256': native.array_hash(edges)}


def scene():
    rig = Object(name='Rig', type='ARMATURE', data=NS(bones=['root']), properties={})
    body = Object(name='Body', type='MESH', data=mesh(), properties={}, matrix_world=np.eye(4),
                  modifiers=[NS(type='ARMATURE', object=rig)])
    groom = Object(name='NativeGroom_Body', type='MESH', data=mesh(),
                   properties={'native_groom': True, 'native_groom_source': 'Body'}, matrix_world=np.eye(4),
                   modifiers=[NS(type='ARMATURE', object=rig)], vertex_groups={'AuthoredFurDensity': NS(index=0)},
                   particle_systems=[NS(vertex_group_density='AuthoredFurDensity')])
    for vertex in groom.data.vertices[:3]:
        vertex.groups = [NS(group=0, weight=1.)]
    objects = {obj.name: obj for obj in (body, groom, rig)}
    bpy = NS(data=NS(libraries=[], objects=objects), context=NS(scene=NS(objects=objects)))
    prepared = {'expected': {'body': body.name, 'groom': groom.name, 'rig': rig.name,
                            'density_group': 'AuthoredFurDensity', 'density_selected_vertices': 3,
                            'density_excluded_vertices': 1, 'selected_triangles': 1, 'triangles': 2, 'rest': rest()},
                'settings': settings(), 'scope': 'Analytic admission, not rendering',
                'export_vertex_budget': {'source_export_vertices': 4, 'portable_export_vertices': 24, 'reserve_vertices': 4096}}
    return bpy, prepared, body, groom


def glb_bytes():
    points, _, faces, _ = geometry()
    normal = np.tile(np.array([0, 0, 1], dtype='<f4'), (4, 1))
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype='<f4')
    binary = bytearray(); views, accessors = [], []
    for value, kind, component in ((points, 'VEC3', 5126), (normal, 'VEC3', 5126),
                                    (uv, 'VEC2', 5126), (faces.astype('<u4').ravel(), 'SCALAR', 5125)):
        data = value.tobytes()
        views.append({'buffer': 0, 'byteOffset': len(binary), 'byteLength': len(data)})
        accessors.append({'bufferView': len(views)-1, 'count': len(value), 'type': kind, 'componentType': component})
        binary.extend(data)
    doc = {'asset': {'version': '2.0'}, 'buffers': [{'byteLength': len(binary)}], 'bufferViews': views,
           'accessors': accessors, 'meshes': [{'primitives': [{'attributes': {'POSITION': 0, 'NORMAL': 1, 'TEXCOORD_0': 2}, 'indices': 3}]}],
           'nodes': [{'mesh': 0}], 'scenes': [{'nodes': [0]}], 'scene': 0}
    encoded = json.dumps(doc, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    return struct.pack('<III', 0x46546C67, 2, 12+8+len(encoded)+8+len(binary)) + \
        struct.pack('<II', len(encoded), 0x4E4F534A)+encoded+struct.pack('<II', len(binary), 0x004E4942)+binary


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='undercoat-admission-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.pack = self.root/'pack'; self.pack.mkdir()
        self.source = self.root/'source.glb'; self.source.write_bytes(glb_bytes())
        (self.pack/'source.blend').write_bytes(b'ANALYTIC CUSTODY BYTES; NOT A BLENDER SCENE')
        self.cfg = {'source': str(self.source), 'source_sha256': sha256(self.source), 'authored_rig': {'height_meters': 1},
                    'groom_exclusions': [], 'fur_max_chroma': .9,
                    'portable_fur': {'settings': {'profile': 'tuft-v1'}, 'bands': [{'count': 2, 'brightness': [0, 1.01]}]}}
        self.budget = tuft_export_budget(4, 2)
        self.report = {'source_sha256': self.cfg['source_sha256'], 'deformation_measurement': rest(),
            'groom': {'kind': 'editable_native_blender_particle_hair', 'native_only': True,
                      'source_object': 'Body', 'object': 'NativeGroom_Body', 'armature': 'Rig',
                      'density_selected_vertices': 3, 'density_excluded_vertices': 1,
                      'mask_config': [], 'fur_max_chroma': .9},
            'portable_fur_export_budget': self.budget, 'portable_fur': [{'cards': 2, 'root_summary': {
                'binding_selection': {'eligible_triangles': 1, 'excluded_degenerate_triangles': 0}}}]}
        self.job = {'exit_code': 0, 'failure': None, 'process_exit_verified': True,
                    'scope': 'Analytic fixture only; no process launched'}
        self.seal()

    def seal(self):
        for name, value in (('finish-config.json', self.cfg), ('finish-report.json', self.report),
                            ('job.json', self.job), ('tuft-export-budget.json', self.budget)):
            (self.pack/name).write_text(json.dumps(value))
        self.manifest = {'schema_version': 1, 'status': 'needs_visual_review', 'source_path': str(self.source),
            'source_sha256': self.cfg['source_sha256'], 'finish': deepcopy(self.report), 'job': deepcopy(self.job),
            'artifacts': artifacts(self.pack)}
        (self.pack/'manifest.json').write_text(json.dumps(self.manifest))

    def test_exact_pack_source_preflight_and_whole_coat_budget_are_bound(self):
        prepared = host._inputs(self.pack, settings())
        self.assertEqual(prepared['source_preflight']['vertices'], 4)
        self.assertEqual(prepared['expected']['selected_triangles'], 1)
        self.assertEqual(prepared['export_vertex_budget'], {'source_export_vertices': 4, 'portable_export_vertices': 24, 'reserve_vertices': 4096})
        self.assertEqual(prepared['inputs']['source.blend']['sha256'], sha256(self.pack/'source.blend'))

    def test_changed_blend_and_missing_report_fail_custody(self):
        (self.pack/'source.blend').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            host._inputs(self.pack, settings())
        self.seal()
        self.manifest['artifacts'] = [r for r in self.manifest['artifacts'] if r['path'] != 'finish-report.json']
        (self.pack/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'Missing manifest binding'):
            host._inputs(self.pack, settings())

    def test_resealed_embedded_report_and_source_mismatch_still_reject(self):
        self.manifest['finish']['groom']['density_selected_vertices'] = 2
        (self.pack/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'Embedded'):
            host._inputs(self.pack, settings())
        self.seal(); self.source.write_bytes(glb_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'source binding'):
            host._inputs(self.pack, settings())

    def test_resealed_card_budget_and_unfinished_job_reject(self):
        self.cfg['portable_fur']['bands'][0]['count'] = 3; self.seal()
        with self.assertRaisesRegex(ValueError, 'budget differs'):
            host._inputs(self.pack, settings())
        self.cfg['portable_fur']['bands'][0]['count'] = 2
        self.job['process_exit_verified'] = False; self.seal()
        with self.assertRaisesRegex(ValueError, 'did not exit'):
            host._inputs(self.pack, settings())

    def test_symlinks_duplicate_paths_and_missing_rest_binding_reject(self):
        self.manifest['artifacts'].append(deepcopy(self.manifest['artifacts'][0]))
        (self.pack/'manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            host._inputs(self.pack, settings())
        self.report['deformation_measurement']['rest_sha256'] = 'bad'; self.seal()
        with self.assertRaisesRegex(ValueError, 'rest geometry hashes'):
            host._inputs(self.pack, settings())
        self.report['deformation_measurement'] = rest(); self.seal()
        actual = self.pack/'source.blend'; actual.rename(self.pack/'saved.blend')
        actual.symlink_to(self.pack/'saved.blend')
        with self.assertRaisesRegex(ValueError, 'Symlink'):
            host._inputs(self.pack, settings())

    def test_settings_bounds_and_nested_output_stop_before_dispatch(self):
        with patch.object(host, 'run_bounded') as run:
            for bad in ({**settings(), 'resolution': 4096}, {**settings(), 'strength': float('nan')},
                        {**settings(), 'min_faces': True}, {**settings(), 'max_faces': 100001}):
                with self.assertRaises(ValueError):
                    host.inspect(self.pack, bad, self.root/'new')
            with self.assertRaisesRegex(ValueError, 'cannot modify'):
                host.inspect(self.pack, settings(), self.pack/'diagnostic')
            run.assert_not_called()
        self.assertFalse((self.root/'new').exists())

    def test_refusal_is_successful_diagnostic_with_minimal_environment_and_pinned_scripts(self):
        original = {p.name: sha256(p) for p in self.pack.iterdir() if p.is_file()}

        def run(command, log, output, **limits):
            self.assertEqual(command[:2], ['/usr/bin/env', '-i'])
            self.assertIn('--disable-autoexec', command); self.assertIn('--factory-startup', command)
            self.assertEqual(command[command.index('--threads')+1], '4')
            self.assertEqual(limits, {'timeout': 120, 'memory_mb': 2048, 'output_mb': 64})
            self.assertFalse(any(item.startswith(('OPENAI_', 'MESHY_', 'PYTHONPATH=')) for item in command))
            data = {'policy': host.POLICY, 'status': 'refused', 'reason': 'Analytic measured capacity refusal',
                    'inputs_sha256': sha256(output/'admission-inputs.json'), 'settings': settings(), 'input_writes': False}
            (output/'native-admission.json').write_text(json.dumps(data))
            return {'exit_code': 0, 'failure': None, 'process_exit_verified': True, 'scope': 'Mock; no native job'}

        with patch.object(host, 'run_bounded', side_effect=run) as dispatch:
            result = host.inspect(self.pack, settings(), self.root/'new')
        self.assertEqual(dispatch.call_count, 1); self.assertEqual(result['status'], 'refused')
        self.assertTrue(result['inputs_unchanged']); self.assertFalse(result['quality_approved'])
        self.assertEqual(original, {p.name: sha256(p) for p in self.pack.iterdir() if p.is_file()})
        prepared = read_json(self.root/'new/prepared.json')
        self.assertEqual(prepared['sources']['blender_fur_undercoat.py']['sha256'], sha256(host.ROOT/'scripts/blender_fur_undercoat.py'))

    def test_input_change_during_mock_execution_is_not_success(self):
        def run(*args, **kwargs):
            self.source.write_bytes(b'changed during execution')
            return {'exit_code': 0, 'failure': None, 'process_exit_verified': True}
        with patch.object(host, 'run_bounded', side_effect=run), self.assertRaisesRegex(ValueError, 'input changed'):
            host.inspect(self.pack, settings(), self.root/'new')
        self.assertEqual(read_json(self.root/'new/manifest.json')['status'], 'failed')

    def test_final_receipts_are_inside_the_aggregate_output_bound(self):
        def run(command, log, output, **kwargs):
            with (output/'unexpected-output.bin').open('wb') as handle:
                handle.truncate(64*1024*1024)
            (output/'native-admission.json').write_text(json.dumps({'policy': host.POLICY, 'status': 'refused',
                'inputs_sha256': sha256(output/'admission-inputs.json'), 'settings': settings(), 'input_writes': False}))
            return {'exit_code': 0, 'failure': None, 'process_exit_verified': True}
        with patch.object(host, 'run_bounded', side_effect=run), self.assertRaisesRegex(ValueError, 'final receipts'):
            host.inspect(self.pack, settings(), self.root/'new')
        self.assertEqual(read_json(self.root/'new/manifest.json')['status'], 'failed')


class NativeAdmissionTests(unittest.TestCase):
    def test_standalone_snapshot_imports_only_its_siblings_from_unrelated_cwd(self):
        with tempfile.TemporaryDirectory(prefix='undercoat-snapshot-') as temporary:
            root = Path(temporary).resolve(); sources = root/'sources'; sources.mkdir()
            other = root/'unrelated'; other.mkdir()
            for name in host.NATIVE_SOURCES:
                shutil.copyfile(host.ROOT/'scripts'/name, sources/name)
            (other/'blender_fur_undercoat.py').write_text("raise AssertionError('Untrusted CWD import')\n")
            code = ("import runpy,sys,pathlib; "
                    "scope=runpy.run_path(sys.argv[1],run_name='admission_import_test'); "
                    "import blender_fur_undercoat; "
                    "assert pathlib.Path(blender_fur_undercoat.__file__).parent==pathlib.Path(sys.argv[1]).parent; "
                    "assert 'bpy' not in sys.modules; assert callable(scope['inspect_scene']); print('snapshot_import_passed')")
            result = subprocess.run([sys.executable, '-I', '-B', '-c', code,
                                     str(sources/'blender_undercoat_admission.py')], cwd=other,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), 'snapshot_import_passed')

    def test_exact_copy_mask_and_healthy_uv_admit_without_mutation(self):
        bpy, prepared, body, groom = scene()
        before = [value.copy() for value in native._raw_mesh(body.data)]
        result = native.inspect_scene(bpy, prepared)
        self.assertEqual(result['status'], 'supported')
        self.assertEqual(result['counts']['selected_triangles'], 1)
        self.assertEqual(result['counts']['selected_vertices'], 3)
        self.assertEqual(result['plan']['atlas_regions'], [])
        self.assertEqual(result['materials'], [{'slot': 0, 'name': 'AnalyticSurface', 'triangles': 2, 'selected_triangles': 1}])
        for a, b in zip(before, native._raw_mesh(body.data)):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(len(body.data.uv_layers), 1); self.assertFalse(result['baked'])

    def test_different_coordinates_or_topology_never_transfer_density_by_proximity(self):
        for kind in ('point', 'topology'):
            bpy, prepared, body, groom = scene()
            if kind == 'point':
                groom.data.vertices[0].co = [1e-8, 0, 0]
            else:
                groom.data.loops[0].vertex_index = 1
            with self.assertRaisesRegex(ValueError, 'coordinates or topology'):
                native.inspect_scene(bpy, prepared)

    def test_fractional_density_and_changed_report_counts_reject(self):
        for kind in ('fractional', 'count', 'faces', 'rest'):
            bpy, prepared, _, groom = scene()
            if kind == 'fractional': groom.data.vertices[0].groups[0].weight = .99
            elif kind == 'count': prepared['expected']['density_selected_vertices'] = 2
            elif kind == 'faces': prepared['expected']['selected_triangles'] = 2
            else: prepared['expected']['rest']['rest_sha256'] = '0'*64
            with self.assertRaises(ValueError): native.inspect_scene(bpy, prepared)

    def test_oversized_geometry_is_rejected_before_buffer_allocation(self):
        class TooMany(Collection):
            def __len__(self): return 100001
            def foreach_get(self, *args): raise AssertionError('Must reject metadata before decoding')
        bpy, prepared, body, _ = scene()
        body.data.vertices = TooMany()
        with self.assertRaisesRegex(ValueError, 'bounds'):
            native.inspect_scene(bpy, prepared)

    def test_selection_refusal_preserves_actual_counts_and_explicit_settings(self):
        bpy, prepared, _, _ = scene()
        prepared['settings'] = settings(min_faces=2, max_faces=2)
        result = native.inspect_scene(bpy, prepared)
        self.assertEqual(result['status'], 'refused')
        self.assertEqual(result['counts']['selected_triangles'], 1)
        self.assertEqual(result['selection_diagnostic']['selected_faces'], 1)
        self.assertEqual(result['settings']['min_faces'], 2)

    def test_uv_defaults_and_nonfinite_uv_refuse_without_mutation(self):
        for kind in ('active', 'nonfinite'):
            bpy, prepared, body, _ = scene()
            if kind == 'active': body.data.uv_layers.active_index = 1
            else: body.data.uv_layers[0].data[0].uv = [float('nan'), 0]
            result = native.inspect_scene(bpy, prepared)
            self.assertEqual(result['status'], 'refused')
            self.assertFalse(result['saved_scene']); self.assertEqual(len(body.data.uv_layers), 1)

    def test_material_split_cannot_borrow_full_coat_reserve_or_reduce_cards(self):
        bpy, prepared, body, _ = scene()
        prepared['export_vertex_budget'] = {'source_export_vertices': 50004,
                                           'portable_export_vertices': 45900, 'reserve_vertices': 4096}
        before = deepcopy(prepared)
        result = native.inspect_scene(bpy, prepared)
        self.assertEqual(result['status'], 'refused')
        self.assertIn('100000', result['reason'])
        self.assertEqual(prepared, before)
        self.assertEqual(result['counts']['selected_triangles'], 1)
        self.assertEqual(result['export_vertex_budget'], before['export_vertex_budget'])


class MaterialSeamCapacityTests(unittest.TestCase):
    def fixture(self):
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 4], [0, 1, 5]], dtype=np.int32)
        selected = np.array([True, True, False, False])
        materials = np.zeros(4, dtype=np.int32)
        budget = {'source_export_vertices': 50004, 'portable_export_vertices': 45900, 'reserve_vertices': 4096}
        return faces, selected, materials, budget

    def test_corner_multiplicity_and_card_ceiling_are_not_unique_vertex_counts(self):
        faces, selected, materials, budget = self.fixture()
        before = [x.copy() for x in (faces, selected, materials)]; old_budget = dict(budget)
        result = native.material_seam_capacity(6, faces, selected, materials, budget)
        self.assertEqual(result['materials'][0]['shared_source_vertices'], 2)
        self.assertEqual(result['material_boundary_corner_reserve'], 4)
        self.assertEqual(result['current_cards'], 3825)
        self.assertEqual(result['maximum_cards_before_additional_uv_atlas'], 3824)
        self.assertEqual(result['current_vertex_ceiling_before_additional_uv_atlas'], 100004)
        self.assertTrue(result['upper_bound_only']); self.assertFalse(result['admission_verified'])
        self.assertFalse(result['additional_uv_atlas_cost_included'])
        self.assertEqual(result, native.material_seam_capacity(6, faces.astype(np.uint64), selected, materials, budget))
        for actual, old in zip((faces, selected, materials), before): np.testing.assert_array_equal(actual, old)
        self.assertEqual(budget, old_budget)

    def test_material_boundaries_do_not_cross_charge_existing_source_material_splits(self):
        faces, selected, materials, budget = self.fixture()
        materials[2:] = 1
        result = native.material_seam_capacity(6, faces, selected, materials, budget)
        self.assertEqual(result['material_boundary_corner_reserve'], 0)
        self.assertEqual(result['maximum_cards_before_additional_uv_atlas'], 3825)
        self.assertEqual(result['materials'], [{'material_id': 0, 'selected_triangles': 2,
                                               'unselected_triangles': 0, 'shared_source_vertices': 0,
                                               'boundary_corner_reserve': 0}])
        materials[:] = [0, 1, 0, 1]
        result = native.material_seam_capacity(6, faces, selected, materials, budget)
        self.assertEqual([row['boundary_corner_reserve'] for row in result['materials']], [2, 2])
        self.assertEqual(result['material_boundary_corner_reserve'], 4)

    def test_zero_card_capacity_is_distinct_from_base_and_seam_infeasibility(self):
        faces, selected, materials, _ = self.fixture()
        budget = {'source_export_vertices': 95900, 'portable_export_vertices': 0, 'reserve_vertices': 4096}
        result = native.material_seam_capacity(6, faces, selected, materials, budget)
        self.assertEqual(result['maximum_cards_before_additional_uv_atlas'], 0)
        self.assertTrue(result['source_reserve_and_seams_fit_without_cards'])
        budget['source_export_vertices'] += 1
        result = native.material_seam_capacity(6, faces, selected, materials, budget)
        self.assertIsNone(result['maximum_cards_before_additional_uv_atlas'])
        self.assertFalse(result['source_reserve_and_seams_fit_without_cards'])

    def test_invalid_or_oversized_arrays_reject_before_bincount(self):
        faces, selected, materials, budget = self.fixture()
        invalid = [(True, faces, selected, materials), (100001, faces, selected, materials),
                   (6, np.zeros((100001, 3), dtype=np.int32), np.ones(100001, dtype=bool), np.zeros(100001, dtype=np.int32)),
                   (6, faces.astype(float), selected, materials), (6, faces-1, selected, materials),
                   (6, faces+6, selected, materials), (6, faces, selected.astype(int), materials),
                   (6, faces, selected, materials.astype(float)), (6, faces, selected, materials+32),
                   (6, faces[:, :2], selected, materials), (6, faces, selected[:2], materials),
                   (6, faces, selected, np.full(4, np.nan))]
        with patch.object(native.np, 'bincount', side_effect=AssertionError('Invalid metadata reached allocation')):
            for args in invalid:
                with self.subTest(shape=getattr(args[1], 'shape', None)), self.assertRaises(ValueError):
                    native.material_seam_capacity(*args, budget)
            for bad in ({**budget, 'portable_export_vertices': 1}, {**budget, 'reserve_vertices': True}):
                with self.assertRaises(ValueError): native.material_seam_capacity(6, faces, selected, materials, bad)

    def test_diagnostic_calls_planner_once_and_keeps_capacity_when_it_refuses(self):
        bpy, prepared, _, _ = scene()
        with patch.object(native, 'plan_material_undercoat', side_effect=ValueError('Analytic atlas cost refusal')) as planner:
            result = native.inspect_scene(bpy, prepared)
        self.assertEqual(planner.call_count, 1)
        self.assertEqual(result['status'], 'refused')
        self.assertEqual(result['material_seam_capacity']['material_boundary_corner_reserve'], 2)
        self.assertFalse(result['material_seam_capacity']['admission_verified'])


if __name__ == '__main__':
    unittest.main()
