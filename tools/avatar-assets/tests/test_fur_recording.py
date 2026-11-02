"""Analytic binding capture and native identity guards; no avatar quality claims."""

from types import SimpleNamespace as NS
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from scripts.blender_fur_cards import card_geometry
from scripts.blender_fur_identity import ATTRIBUTE, POLICY, stamp_fur_identities
from scripts.groom_flow import prepare_flow


def geometry(profile, record):
    points = [(0., 0., 0.), (.1, 0., 0.), (0., .1, 0.),
              (.2, 0., 0.), (.3, 0., 0.), (.2, .1, 0.)]
    # A skipped first face exposes accidental eligible-list index substitution.
    faces = [(0, 0, 0), (0, 1, 2), (3, 4, 5)]
    weights = [{'root': 1.} for _ in points]
    cfg = {'seed': 27, 'count': 35, 'length': .012, 'width': .004, 'direction': [0., 1., 0.]}
    args = {'profile': profile, 'record_bindings': record, 'normals': [(0., 0., 1.)]*len(points)}
    if profile == 'tuft-v1':
        args.update(prepared_flow=prepare_flow([0, 0, -.01], [.4, .2, .02], [0, 1, 0], .4,
                    {'regions': [{'name': 'uniform', 'center': [.5, .5, .5], 'radius': [1, 1, 1],
                                  'feather': .5, 'direction': [0, 1, 0], 'comb': .4, 'length_scale': 1}]}),
                    root_color_sampler=lambda *_: (.2, .4, .6))
    else:
        args['colors'] = [[.2, .4, .6]]*len(points)
    return card_geometry(points, faces, weights, cfg, **args), np.asarray(points), np.asarray(faces)


class Values:
    def __init__(self, count):
        self.values = np.zeros(count, dtype=np.int32)

    def foreach_set(self, field, values):
        assert field == 'value'
        self.values[:] = values

    def foreach_get(self, field, target):
        assert field == 'value'
        target[:] = self.values


class Attributes(list):
    def __init__(self, count):
        super().__init__()
        self.count = count
        self.rename = False

    def new(self, *, name, type, domain):
        layer = NS(name=name+'.001' if self.rename else name, data_type=type,
                   domain=domain, data=Values(self.count))
        self.append(layer)
        return layer


def owner(name, vertices, triangles, rig):
    mesh = NS(vertices=[NS(index=i) for i in range(vertices)], polygons=[None]*triangles,
              loop_triangles=[None]*triangles, attributes=Attributes(vertices), calc_loop_triangles=lambda: None)
    modifier = NS(type='ARMATURE', object=rig, show_render=True, show_viewport=True,
                  use_vertex_groups=True, use_bone_envelopes=False, use_deform_preserve_volume=False)
    return NS(name=name, type='MESH', mode='OBJECT', data=mesh, modifiers=[modifier],
              matrix_world=np.eye(4), matrix_basis=np.eye(4))


def fixture():
    rig = NS(type='ARMATURE', data=NS(pose_position='REST', bones=[None, None]),
             matrix_world=np.eye(4), matrix_basis=np.eye(4))
    body = owner('different subject body', 3, 1, rig)
    coat = owner('PortableCoat_7', 12, 8, rig)
    meta = {'policy': POLICY, 'attribute': ATTRIBUTE, 'source_node': body.name,
            'coat_node': coat.name, 'source_vertex_count': 3, 'source_triangle_count': 1, 'coat_vertex_count': 12}
    reports = [{'geometry_profile': 'combed-v1', 'cards': 2,
                'root_summary': {'binding_archive': {'schema_version': 2, 'roots': 2,
                                                     'export_correspondence': meta}}}]
    return body, [coat], rig, reports


class RecordingTests(unittest.TestCase):
    def test_unrecorded_profiles_equal_saved_prechange_implementation(self):
        path = Path(__file__).resolve().parents[1]/'evidence/material-owned-fur-v1/snapshots/material-pipeline-tests-v1/sources/scripts/blender_fur_cards.py'
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         'b80936f4dab8d6c3546a72fa63f170173d99a90cd61a49c31b646eb04c6c68e7')
        spec = importlib.util.spec_from_file_location('scripts._prior_fur_recording', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        def historical(*args, **kwargs):
            kwargs.pop('record_bindings')
            return module.card_geometry(*args, **kwargs)
        for profile in ('legacy', 'combed-v1', 'tuft-v1'):
            current, _, _ = geometry(profile, False)
            with patch(__name__+'.card_geometry', side_effect=historical):
                prior, _, _ = geometry(profile, False)
            self.assertEqual(json.dumps(current, sort_keys=True), json.dumps(prior, sort_keys=True))

    def test_capture_keeps_geometry_rng_colors_and_all_original_fields(self):
        for profile in ('legacy', 'combed-v1', 'tuft-v1'):
            with self.subTest(profile=profile):
                old, _, _ = geometry(profile, False)
                new, _, _ = geometry(profile, True)
                for key, value in old.items():
                    if key == 'root_summary':
                        self.assertEqual({k: new[key][k] for k in value}, value)
                    else:
                        self.assertEqual(new[key], value)
                self.assertEqual(len(new['root_bindings']['triangle_index']), 35)

    def test_all_profiles_record_exact_source_triangles_and_bottom_insets(self):
        for profile in ('legacy', 'combed-v1', 'tuft-v1'):
            data, points, faces = geometry(profile, True)
            bindings = data['root_bindings']
            self.assertEqual(set(bindings['triangle_index']), {1, 2})
            for root, (face, bary, inset) in enumerate(zip(bindings['triangle_index'], bindings['barycentric'],
                                                         bindings['root_inset'])):
                midpoint = np.asarray(data['vertices'][6*root:6*root+2]).mean(axis=0)
                np.testing.assert_allclose(midpoint+inset, np.asarray(bary)@points[faces[face]], atol=1e-15, rtol=0)

    def test_record_flag_rejects_truthy_coercion(self):
        for flag in (1, 'true', None):
            with self.assertRaisesRegex(ValueError, 'boolean'):
                geometry('legacy', flag)


class IdentityTests(unittest.TestCase):
    def test_exact_ids_and_bound_owner_metadata(self):
        body, coats, rig, reports = fixture()
        receipt = stamp_fur_identities(body, coats, rig, reports)
        self.assertEqual(receipt['recorded_roots'], 2)
        self.assertFalse(receipt['attachment_verified'])
        for obj in [body, *coats]:
            self.assertEqual(obj.data.attributes[0].name, ATTRIBUTE)
            np.testing.assert_array_equal(obj.data.attributes[0].data.values, np.arange(len(obj.data.vertices)))

    def test_all_owners_validate_before_first_attribute_write(self):
        edits = [lambda b, c, r, p: setattr(c[0], 'matrix_world', np.diag([2., 1., 1., 1.])),
                 lambda b, c, r, p: c[0].data.attributes.append(NS(name='_UNRELATED')),
                 lambda b, c, r, p: p[0]['root_summary']['binding_archive']['export_correspondence'].update(source_node='other'),
                 lambda b, c, r, p: setattr(c[0].modifiers[0], 'use_deform_preserve_volume', True),
                 lambda b, c, r, p: p[0].update(cards=3),
                 lambda b, c, r, p: setattr(c[0], 'data', b.data)]
        for edit in edits:
            args = fixture(); edit(*args)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                stamp_fur_identities(*args)
            self.assertFalse(any(a.name == ATTRIBUTE for a in args[0].data.attributes))

    def test_only_new_owned_attributes_roll_back_on_native_rename(self):
        args = fixture(); args[1][0].data.attributes.rename = True
        with self.assertRaisesRegex(ValueError, 'renamed'):
            stamp_fur_identities(*args)
        self.assertEqual(list(args[0].data.attributes), [])
        self.assertEqual(list(args[1][0].data.attributes), [])

    def test_existing_experimental_id_requires_explicit_ownership(self):
        args = fixture(); layer = NS(name='_AUTHOR_VERTEX_ID'); args[0].data.attributes.append(layer)
        with self.assertRaisesRegex(ValueError, 'unowned'):
            stamp_fur_identities(*args)
        receipt = stamp_fur_identities(*args, allow_source_skin_ids=True)
        self.assertEqual(receipt['source_node'], args[0].name)
        self.assertIs(args[0].data.attributes[0], layer)


if __name__ == '__main__':
    unittest.main()
