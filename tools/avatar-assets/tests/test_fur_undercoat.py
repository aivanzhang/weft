"""Analytic UV/mask/resource fixtures, not appearance or generated-hair evidence."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import blender_fur_undercoat as undercoat
from scripts.groom_flow import prepare_flow


def settings():
    return {'profile':'short-pile-v1', 'resolution':1024, 'seed':41, 'pitch_m':.004,
            'strength':.22, 'roughness':.92, 'min_faces':1, 'max_faces':10}


def fixture():
    points = np.array([[0,0,0],[.02,0,0],[.02,.02,0],[0,.02,0]], dtype=np.float32)
    faces = np.array([[0,1,2],[0,2,3]], dtype=np.int32)
    uv = np.array([[[0,0],[1,0],[1,1]],[[0,0],[1,1],[0,1]]], dtype=np.float32)
    normals = np.tile([0.,0.,1.], (2,3,1)).astype(np.float32)
    tangents = np.tile([1.,0.,0.], (2,3,1)).astype(np.float32)
    signs = np.ones((2,3), dtype=np.float32)
    selected = np.array([True,True])
    flow = prepare_flow([0,0,-.01],[.02,.02,.02],[0,1,0],.6, {'regions':[
        {'name':'analytic-flow','center':[.5,.5,.5],'radius':[1.,1.,1.],
         'feather':.5,'direction':[0,1,0],'comb':.6,'length_scale':1.}]})
    return points, faces, uv, normals, tangents, signs, selected, flow, settings()


class UndercoatTests(unittest.TestCase):
    def test_package_import_cannot_capture_a_different_top_level_flow_class(self):
        source=Path(undercoat.__file__)
        spec=importlib.util.spec_from_file_location('scripts._undercoat_import_fixture',source)
        module=importlib.util.module_from_spec(spec)
        foreign=SimpleNamespace(PreparedFlow=type('ForeignPreparedFlow',(),{}),evaluate_prepared_flow=None)
        with patch.dict(sys.modules, {'groom_flow':foreign}):
            spec.loader.exec_module(module)
        self.assertIsInstance(fixture()[7],module.PreparedFlow)
        self.assertIs(module.PreparedFlow,undercoat.PreparedFlow)

    def test_bake_preserves_input_and_is_chunk_independent(self):
        values = fixture()
        originals = [value.copy() for value in values[:7]]
        image, report = undercoat.bake_short_pile(*values)
        with patch.object(undercoat, 'CHUNK', 65536):
            repeated, second = undercoat.bake_short_pile(*values)
        np.testing.assert_array_equal(image, repeated)
        for actual, before in zip(values[:7], originals):
            np.testing.assert_array_equal(actual, before)
        self.assertEqual(report['normal_float32_sha256'], second['normal_float32_sha256'])
        self.assertEqual(report['normal_float32_sha256'],
                         '64953c8d89139043b8cb1bfaed441af49ed649c25e8e769e368aac85d1b9a0fc')
        self.assertEqual(report['selected_triangles'], 2)
        self.assertEqual(report['covered_texels'], 1024**2)
        self.assertEqual(report['overlapping_different_surface_samples'], 0)
        self.assertEqual(image.dtype, np.float32)
        self.assertGreater(report['maximum_normal_tilt_degrees'], 1)
        self.assertLessEqual(report['maximum_normal_tilt_degrees'], 24.23)
        np.testing.assert_allclose(np.linalg.norm(image[:,:,:3]*2-1, axis=2), 1, atol=2e-7)
        self.assertTrue(np.all(image[:,:,3] == 1))
        self.assertFalse(report['palette_changed'])
        self.assertFalse(report['geometry_displaced'])

    def test_mask_excludes_nonfur_and_zero_strength_is_neutral(self):
        values = list(fixture());values[6] = np.array([True,False])
        image, report = undercoat.bake_short_pile(*values)
        self.assertEqual(report['selected_triangles'], 1)
        self.assertEqual(report['excluded_triangles'], 1)
        np.testing.assert_array_equal(image[800,100], [.5,.5,1,1])
        self.assertFalse(np.array_equal(image[100,800], [.5,.5,1,1]))
        values[-1] = {**settings(), 'strength':0}
        image, report = undercoat.bake_short_pile(*values)
        self.assertTrue(np.all(image == np.array([.5,.5,1,1])))
        self.assertEqual(report['maximum_normal_tilt_degrees'], 0)

    def test_rejects_conflicting_uv_surfaces_instead_of_silent_overpaint(self):
        values = list(fixture())
        values[0] = np.vstack((values[0], values[0]+[0,0,.01]))
        values[1] = np.vstack((values[1], values[1]+4))
        for index in (2,3,4,5):
            values[index] = np.concatenate((values[index], values[index]))
        values[6] = np.ones(4, dtype=bool)
        with self.assertRaisesRegex(ValueError, 'overlap cannot preserve'):
            undercoat.bake_short_pile(*values)

    def test_invalid_geometry_is_rejected_before_raster(self):
        changes = [(0, np.full((4,3), np.nan)), (1, np.array([[0,1,9],[0,2,3]])),
                   (2, np.full((2,3,2), 1.1)), (3, np.zeros((2,3,3))),
                   (5, np.full((2,3), .5)), (6, np.ones(2, dtype=int)), (7, None),
                   (0, np.broadcast_to(np.zeros((1,3)), (100001,3))),
                   (1, np.broadcast_to(np.zeros((1,3), dtype=int), (100001,3)))]
        for index, changed in changes:
            values = list(fixture());values[index] = changed
            with self.subTest(index=index,shape=getattr(changed,'shape',None)), \
                 patch.object(undercoat, '_raster') as raster, self.assertRaises(ValueError):
                undercoat.bake_short_pile(*values)
            raster.assert_not_called()

    def test_raster_and_deadline_bounds_and_degenerate_uv(self):
        with patch.object(undercoat, 'MAX_RASTER_WORK', 8), self.assertRaisesRegex(ValueError, 'raster work'):
            undercoat.bake_short_pile(*fixture())
        with patch.object(undercoat, 'MAX_SECONDS', -1), self.assertRaisesRegex(RuntimeError, 'deadline'):
            undercoat.bake_short_pile(*fixture())
        values = list(fixture());values[2][0] = [[0,0],[0,0],[0,0]]
        with self.assertRaisesRegex(ValueError, 'degenerate UV'):
            undercoat.bake_short_pile(*values)

    def test_settings_are_exact_bounded_and_nonmutating(self):
        cfg=settings();before=deepcopy(cfg)
        self.assertEqual(undercoat.validate_undercoat(cfg), cfg)
        self.assertEqual(cfg,before)
        for field, value in [('profile','fur'),('resolution',4096),('resolution',True),
                             ('seed',True),('seed',2**31),('strength',float('nan')),
                             ('strength',True),('strength',.51),('pitch_m',.001),
                             ('roughness',.1),('min_faces',0),('max_faces',100001),
                             ('max_faces',False),('unexpected',0)]:
            with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                undercoat.validate_undercoat({**cfg,field:value})
        values=list(fixture());values[-1]={**cfg,'min_faces':3}
        with patch.object(undercoat, '_raster') as raster, self.assertRaisesRegex(ValueError,'face count'):
            undercoat.bake_short_pile(*values)
        raster.assert_not_called()


if __name__ == '__main__':
    unittest.main()
