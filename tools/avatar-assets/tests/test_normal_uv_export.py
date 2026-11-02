"""Analytic live-graph admission fixtures; no native export or appearance claim."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts.blender_normal_uv_export import assert_normal_uv_export
from scripts.normal_uv_export import FRAME_POLICY, NormalUVExportError, evaluate_normal_uv_export


PNG = b'\x89PNG\r\n\x1a\nanalytic-packed-byte-witness'
DIGEST = hashlib.sha256(PNG).hexdigest()


def receipt(uv=1, name='Authored', count=1):
    return {'schema_version': 1, 'profile': 'short-pile-v1', 'uv_policy': 'material-adaptive-v1',
        'frame_policy': FRAME_POLICY, 'authored_uv': {'index': uv, 'name': 'AuthoredUV'},
        'materials': [{'material': name, 'normal_texcoord': uv, 'normal_uv_map': 'AuthoredUV',
                       'selected_faces': count, 'texture_sha256': DIGEST, 'frame_policy': FRAME_POLICY}]}


def evidence(uv=0, name='Source', count=1):
    names = ['UVMap', 'AuthoredUV'] if uv <= 1 else ['UVMap', 'SourceUV1', 'AuthoredUV']
    return {'object': 'Body', 'material': name, 'slot': 0, 'polygons': count, 'triangles': count,
        'triangular': True, 'uv_names': names, 'active_uv_index': 0, 'render_uv_indices': [0],
        'normal': {'normal_uv_index': uv, 'texture_uv_index': uv, 'image': 'NormalPNG',
                   'texture_sha256': DIGEST if uv else None, 'packed_png_bytes': len(PNG) if uv else 0}}


def node(kind, inputs=(), **kwargs):
    return NS(type=kind, inputs={name: NS(links=[]) for name in inputs}, mute=False, **kwargs)


def link(source, output, target, input_name):
    target.inputs[input_name].links.append(NS(from_node=source, from_socket=NS(name=output),
                                             is_valid=True, is_muted=False))


class Image:
    name, source = 'NormalPNG', 'FILE'

    def __init__(self, packed=PNG):
        self.packed_file = NS(size=len(packed), data=packed) if packed is not None else None
        self.packed_files = [self.packed_file] if packed is not None else []

    @property
    def pixels(self):
        raise AssertionError('Image pixels must not be read')

    @property
    def filepath(self):
        raise AssertionError('Image paths must not be read')


def material(name='Source', normal_uv='', vector_uv=None, image=None):
    output = node('OUTPUT_MATERIAL', ['Surface'], is_active_output=True)
    shader = node('BSDF_PRINCIPLED', ['Normal'])
    normal = node('NORMAL_MAP', ['Color', 'Strength'], uv_map=normal_uv, space='TANGENT')
    texture = node('TEX_IMAGE', ['Vector'], image=image or Image(), projection='FLAT')
    nodes = [output, shader, normal, texture]
    link(shader, 'BSDF', output, 'Surface')
    link(normal, 'Normal', shader, 'Normal')
    link(texture, 'Color', normal, 'Color')
    if vector_uv is not None:
        uv = node('UVMAP', uv_map=vector_uv, from_instancer=False)
        nodes.append(uv)
        link(uv, 'UV', texture, 'Vector')
    return NS(name=name, use_nodes=True, node_tree=NS(nodes=nodes, links=[]))


class Layers(list):
    active_index = 0


def owner(materials, slots=(0,), names=('UVMap',), name='Body'):
    layers = Layers(NS(name=value, active_render=index == 0) for index, value in enumerate(names))
    polygons = [NS(material_index=slot, loop_total=3) for slot in slots]
    mesh = NS(vertices=range(3*len(slots)), polygons=polygons, loops=range(3*len(slots)), uv_layers=layers)
    return NS(type='MESH', mode='OBJECT', name=name, data=mesh,
              material_slots=[NS(material=value) for value in materials])


class EvidencePolicyTests(unittest.TestCase):
    def reject(self, rows, code, source=None):
        report = evaluate_normal_uv_export(rows, source)
        self.assertEqual(report['status'], 'unsupported')
        self.assertEqual(report['errors'][0]['code'], code)
        return report

    def test_uv0_and_non_normal_surfaces_need_no_author_receipt(self):
        row = evidence()
        report = evaluate_normal_uv_export([row])
        self.assertEqual(report['status'], 'supported')
        self.assertEqual(report['materials'][0]['admission'], 'coherent_uv0')
        row.update(normal=None, uv_names=[], active_uv_index=-1, render_uv_indices=[])
        self.assertEqual(evaluate_normal_uv_export([row])['status'], 'supported')

    def test_uv1_and_uv2_are_bound_to_exact_current_authored_receipt(self):
        for uv in (1, 2):
            row, source = evidence(uv, 'Authored'), receipt(uv)
            before = copy.deepcopy((row, source))
            report = evaluate_normal_uv_export([row], source)
            self.assertEqual(report['status'], 'supported')
            self.assertFalse(report['tangent_restoration_performed'])
            self.assertFalse(report['quality_approved'])
            self.assertEqual((row, source), before)
            report['materials'][0]['normal']['image'] = 'Changed'
            self.assertEqual(row['normal']['image'], 'NormalPNG')

    def test_unreceipted_source_uv1_is_not_generic_preservation(self):
        self.reject([evidence(1)], 'nonzero_normal_uv_without_receipt')
        row = evidence(1, 'Renamed')
        self.reject([row], 'nonzero_normal_uv_without_receipt', receipt())

    def test_live_route_mismatch_and_inactive_uv0_are_distinct_failures(self):
        row = evidence(1, 'Authored')
        row['normal']['texture_uv_index'] = 0
        self.reject([row], 'normal_uv_route_mismatch', receipt())
        for key, value in (('active_uv_index', 1), ('render_uv_indices', [1]),
                           ('render_uv_indices', [0, 1]), ('render_uv_indices', [False])):
            row = evidence()
            row[key] = value
            self.reject([row], 'uv0_incoherent')

    def test_wrong_hash_uv_name_count_and_nontriangular_ownership_reject(self):
        for change in ('hash', 'name', 'count', 'polygon'):
            row = evidence(1, 'Authored')
            if change == 'hash':
                row['normal']['texture_sha256'] = '0'*64
            elif change == 'name':
                row['uv_names'][1] = 'OtherUV'
            elif change == 'count':
                row.update(polygons=2, triangles=2)
            else:
                row.update(triangular=False, triangles=2)
            with self.subTest(change=change):
                self.reject([row], 'authored_receipt_mismatch', receipt())

    def test_stale_policy_or_layer_allocation_receipt_rejects(self):
        for location, key, value in (('top', 'frame_policy', 'old-policy'),
                                   ('row', 'frame_policy', 'old-policy'),
                                   ('top', 'uv_policy', 'legacy'),
                                   ('top', 'schema_version', True),
                                   ('uv', 'index', 2)):
            source = receipt()
            target = source if location == 'top' else source['materials'][0] if location == 'row' else source['authored_uv']
            target[key] = value
            self.reject([evidence(1, 'Authored')], 'authored_receipt_invalid', source)

    def test_receipted_counts_aggregate_actual_used_slots_and_missing_targets_reject(self):
        rows = [evidence(1, 'Authored'), evidence(1, 'Authored')]
        rows[1]['object'] = 'SecondOwner'
        self.assertEqual(evaluate_normal_uv_export(rows, receipt(count=2))['status'], 'supported')
        self.reject([evidence()], 'authored_receipt_mismatch', receipt())

    def test_invalid_types_duplicate_slots_and_aggregate_work_reject(self):
        self.reject([evidence(), evidence()], 'invalid_evidence')
        self.reject([evidence()]*257, 'resource_limit')
        rows = [evidence(count=100000), evidence()]
        rows[1]['object'] = 'Other'
        self.reject(rows, 'resource_limit')
        for bad in (True, float('nan'), -1):
            row = evidence()
            row['normal']['normal_uv_index'] = bad
            self.reject([row], 'invalid_evidence')
        row = evidence()
        row['normal']['unexpected'] = object()
        self.reject([row], 'invalid_evidence')


class NativeAdapterTests(unittest.TestCase):
    def reject(self, obj, code, source=None, path=None):
        with self.assertRaises(NormalUVExportError) as caught:
            assert_normal_uv_export([obj], source, report_path=path)
        self.assertEqual(caught.exception.report['errors'][0]['code'], code)
        return caught.exception.report

    def test_default_and_explicit_uv0_routes_pass_without_pixels_or_paths(self):
        for uv, vector in (('', None), ('UVMap', 'UVMap'), ('UVMap', None)):
            obj = owner([material(normal_uv=uv, vector_uv=vector)])
            report = assert_normal_uv_export([obj])
            self.assertEqual(report['status'], 'supported')
            self.assertEqual(report['packed_png_bytes_hashed'], 0)
            self.assertFalse(report['texture_pixels_read'])

    def test_dangling_uv1_normal_and_unused_invalid_material_slots_are_harmless(self):
        live = material()
        dangling = node('NORMAL_MAP', ['Color'], uv_map='MissingUV1', space='TANGENT')
        live.node_tree.nodes.append(dangling)
        obj = owner([live, NS(name='UnusedBroken', use_nodes=True, node_tree=None), None])
        self.assertEqual(assert_normal_uv_export([obj])['used_materials'], 1)
        live.node_tree.nodes[1].inputs['Normal'].links.clear()
        self.assertEqual(assert_normal_uv_export([obj])['materials'][0]['admission'], 'no_live_normal_map')

    def test_authored_uv2_reads_only_packed_bytes_and_leaves_native_data_unchanged(self):
        mat = material('Authored', 'AuthoredUV', 'AuthoredUV')
        obj = owner([mat], names=('UVMap', 'OriginalUV1', 'AuthoredUV'))
        before = copy.deepcopy(obj)
        report = assert_normal_uv_export([obj], receipt(2))
        self.assertEqual(report['packed_png_bytes_hashed'], len(PNG))
        self.assertEqual(report['materials'][0]['normal']['texture_sha256'], DIGEST)
        self.assertEqual(obj.data, before.data)
        self.assertEqual(len(mat.node_tree.nodes), len(before.material_slots[0].material.node_tree.nodes))

    def test_source_uv1_without_receipt_rejects_without_reading_unpacked_image(self):
        mat = material('SourceUV1', 'UV1', 'UV1', Image(None))
        report = self.reject(owner([mat], names=('UVMap', 'UV1')),
                             'nonzero_normal_uv_without_receipt')
        self.assertEqual(report['packed_png_bytes_hashed'], 0)

    def test_vector_uv_mismatch_and_absent_uv_fail(self):
        self.reject(owner([material('Authored', 'AuthoredUV', None)], names=('UVMap', 'AuthoredUV')),
                    'normal_uv_route_mismatch', receipt())
        self.reject(owner([material(normal_uv='Missing')]), 'unsupported_graph')

    def test_mixed_surface_non_tangent_normal_mapping_and_invalid_links_fail(self):
        for change in ('surface', 'normal', 'space', 'vector', 'muted', 'invalid', 'alpha'):
            mat = material()
            output, shader, normal, texture = mat.node_tree.nodes
            if change == 'surface':
                shader.type = 'MIX_SHADER'
            elif change == 'normal':
                normal.type = 'BUMP'
            elif change == 'space':
                normal.space = 'OBJECT'
            elif change == 'vector':
                mapping = node('MAPPING')
                link(mapping, 'Vector', texture, 'Vector')
            elif change == 'muted':
                normal.mute = True
            elif change == 'invalid':
                shader.inputs['Normal'].links[0].is_valid = False
            else:
                normal.inputs['Color'].links[0].from_socket.name = 'Alpha'
            with self.subTest(change=change):
                self.reject(owner([mat]), 'unsupported_graph')

    def test_multiple_active_outputs_fail_but_inactive_output_is_ignored(self):
        mat = material()
        extra = node('OUTPUT_MATERIAL', ['Surface'], is_active_output=False)
        mat.node_tree.nodes.append(extra)
        self.assertEqual(assert_normal_uv_export([owner([mat])])['status'], 'supported')
        extra.is_active_output = True
        self.reject(owner([mat]), 'unsupported_graph')

    def test_wrong_packed_bytes_missing_png_and_oversized_storage_fail_before_pixels(self):
        for image, code in ((Image(PNG+b'changed'), 'authored_receipt_mismatch'),
                            (Image(b'not-a-png'), 'packed_png_required'),
                            (Image(None), 'packed_png_required')):
            mat = material('Authored', 'AuthoredUV', 'AuthoredUV', image)
            self.reject(owner([mat], names=('UVMap', 'AuthoredUV')), code, receipt())
        image = Image()
        image.packed_file = NS(size=32*1024**2+1)
        mat = material('Authored', 'AuthoredUV', 'AuthoredUV', image)
        self.reject(owner([mat], names=('UVMap', 'AuthoredUV')), 'resource_limit', receipt())

    def test_shared_image_is_hashed_once_across_actual_owner_counts(self):
        mat = material('Authored', 'AuthoredUV', 'AuthoredUV')
        objects = [owner([mat], names=('UVMap', 'AuthoredUV'), name=name) for name in ('Body', 'Other')]
        report = assert_normal_uv_export(objects, receipt(count=2))
        self.assertEqual(report['packed_png_bytes_hashed'], len(PNG))
        self.assertEqual(report['triangles'], 2)

    def test_dirty_authored_image_cannot_reuse_a_stale_packed_png_receipt(self):
        image = Image()
        image.is_dirty = True
        mat = material('Authored', 'AuthoredUV', 'AuthoredUV', image)
        report = self.reject(owner([mat], names=('UVMap', 'AuthoredUV')), 'packed_png_required', receipt())
        self.assertEqual(report['packed_png_bytes_hashed'], 0)

    def test_work_limits_and_deadline_fail_with_detached_reports(self):
        obj = owner([material()])
        obj.data.vertices = range(100001)
        self.reject(obj, 'resource_limit')
        obj = owner([material()])
        obj.data.polygons[0].material_index = 7
        self.reject(obj, 'invalid_evidence')
        with patch('scripts.blender_normal_uv_export.time.monotonic', side_effect=[0, 11, 11]):
            self.reject(owner([material()]), 'resource_limit')

    def test_failure_report_is_exclusive_json_and_carries_material_identity(self):
        obj = owner([material('OriginalUV1', 'UV1', 'UV1')], names=('UVMap', 'UV1'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'admission.json'
            report = self.reject(obj, 'nonzero_normal_uv_without_receipt', path=path)
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertEqual(report['errors'][0]['material'], 'OriginalUV1')
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                assert_normal_uv_export([obj], report_path=path)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
