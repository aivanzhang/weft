"""Analytic finishing contracts; no native rendering or generated asset claims."""

import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch
import sys
import numpy as np

from avatar_assets.finish_generated import ROOT, finish, validate, reuse_native, _undercoat_frame_policy, _portable_fringe_export
from avatar_assets.safety import LIMITS, artifacts, sha256, write_json
from scripts import surface_integrity, blender_fur_cards, fur_root_sampling, groom_flow


def finish_config():
    return {'schema_version': 1, 'id': 'analytic-quality-contract',
            'subject': 'Analytic creature', 'style': 'Analytic fixture',
            'rights': 'Analytic fixture, not a generated asset', 'prompt': 'Test bounded finishing settings',
            'source': 'analytic.glb', 'source_sha256': '0'*64,
            'normal_strength': .6, 'resolution': 512, 'samples': 8,
            'states': [{'id': 'idle', 'meaning': 'Analytic idle state'}],
            'groom': {'parents': 16, 'children': 0, 'length': .01, 'radius': .0001,
                      'roughness': .001, 'seed': 3}}


def tuft_config():
    config = finish_config()
    config['groom'].update(direction=[0, 0, -1], comb=.6)
    config['groom_flow'] = {'regions': [{'name':'surface','center':[.5,.5,.5],'radius':[.8,.8,.8],
        'feather':.5,'direction':[1,0,0],'comb':.65,'length_scale':1.2}]}
    config['portable_fur'] = {'settings': {'seed':1,'length':.01,'width':.01,
        'direction':[0,0,-1],'profile':'tuft-v1','sample_root_color':True},
        'bands':[{'brightness':[0,1.01],'color':[1,1,1,1],'count':100}]}
    return config


def material(image, *, route=(), projection='FLAT', extension='REPEAT'):
    links = [SimpleNamespace(from_node=SimpleNamespace(type=kind, name=name, uv_map=uv_map),
                             from_socket=SimpleNamespace(name=socket))
             for kind, name, socket, uv_map in route]
    texture = SimpleNamespace(type='TEX_IMAGE', image=image, name='BaseColor', projection=projection,
                              extension=extension, inputs={'Vector': SimpleNamespace(links=links)})
    shader = SimpleNamespace(type='BSDF_PRINCIPLED', inputs={
        'Base Color': SimpleNamespace(links=[SimpleNamespace(from_node=texture)])})
    return SimpleNamespace(use_nodes=True, node_tree=SimpleNamespace(nodes=[shader, texture])), texture


class FinishQualityConfigTests(unittest.TestCase):
    def test_unverified_fringe_finishing_fails_before_output_or_native_work(self):
        cfg = tuft_config()
        cfg.pop('portable_fur')
        cfg['portable_fringe'] = {'profile': 'edge-fringe-v1', 'settings': {'seed': 1, 'min_fins': 1,
            'max_fins': 10, 'length_m': .008, 'min_spacing_m': .003, 'min_edge_m': .0085,
            'max_edge_m': .02, 'comb': .15}, 'design_views': [[0, -1, 0]], 'excluded_materials': []}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'analytic.glb'; source.write_bytes(b'ANALYTIC PREFLIGHT FIXTURE')
            cfg.update(source=str(source), source_sha256=sha256(source))
            write_json(root/'config.json', cfg)
            with patch('avatar_assets.finish_generated.preflight', return_value={'vertices': 100, 'triangles': 50, 'skins': 1}), \
                 patch('avatar_assets.finish_generated.run_bounded') as native, \
                 self.assertRaisesRegex(ValueError, 'Unsupported portable_fringe'):
                finish(root/'config.json', root/'new')
            native.assert_not_called()
            self.assertFalse((root/'new').exists())

    def test_native_reuse_retains_hash_bound_review_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'old'; root.mkdir()
            out = Path(temporary)/'new'; out.mkdir()
            (root/'sources').mkdir()
            for folder in ('avatar_assets', 'scripts'):
                for path in (ROOT/folder).glob('*.py'):
                    shutil.copyfile(path, root/'sources'/path.name)
            (root/'views').mkdir()
            image = root/'views/front.png'
            image.write_bytes(b'ANALYTIC COPY-CUSTODY FIXTURE, NOT AN IMAGE')
            cfg = {'review_views': True, 'portable_fringe': {'profile': 'edge-fringe-v1'}}
            report = {'review_views': [{'path': 'views/front.png', 'sha256': sha256(image)}]}
            write_json(root/'finish-config.json', cfg)
            write_json(root/'finish-report.json', report)
            write_json(root/'fringe-report.json', {'provenance': 'analytic copy-custody fixture'})
            write_json(root/'job.json', {'exit_code': 0, 'failure': None})
            for name in ('source.blend', 'portrait.png', 'finish.log', 'finish-monitor.json',
                         'finish-handle.json', 'model-native-export.glb', 'fringe-correspondence.npz'):
                (root/name).write_bytes(b'ANALYTIC COPY-CUSTODY FIXTURE')
            (root/'phase-memory.jsonl').write_text('{"provenance":"analytic fixture","rss_bytes":null}\n')
            write_json(root/'manifest.json', {'artifacts': artifacts(root)})
            reuse_native(root, out, cfg)
            self.assertEqual(sha256(out/'views/front.png'), report['review_views'][0]['sha256'])
            self.assertEqual(json.loads((out/'finish-report.json').read_text()), report)
            self.assertEqual(sha256(out/'phase-memory.jsonl'), sha256(root/'phase-memory.jsonl'))
            self.assertFalse(json.loads((out/'native-reuse.json').read_text())['native_execution_repeated'])

    def test_fringe_postprocessing_rejects_unbound_material_and_reserves_both_glbs(self):
        fringe = {'geometry_profile': 'edge-fringe-v1', 'material': {'material': 'ObservedCoat'}}
        report = {'portable_fringe': fringe, 'portable_fur': [fringe]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'fringe-report.json').write_text(json.dumps(fringe))
            with patch('avatar_assets.fringe_export.omit_unused_fringe_tangents') as derive:
                mismatched = deepcopy(report)
                mismatched['portable_fringe']['material']['material'] = 'UnrelatedSurface'
                with self.assertRaisesRegex(ValueError, 'exact native material report'):
                    _portable_fringe_export(root, mismatched)
                # Sparse bytes exercise actual directory accounting without allocating a large fixture.
                with (root/'preserved-native.bin').open('wb') as handle:
                    handle.truncate(LIMITS['output_bytes']-2*LIMITS['input_bytes'])
                with self.assertRaisesRegex(ValueError, 'pack byte budget'):
                    _portable_fringe_export(root, report)
                derive.assert_not_called()
                self.assertFalse((root/'model-fringe-export.glb').exists())

    def test_uv1_export_requires_the_saved_bake_policy_not_just_an_opt_in(self):
        from avatar_assets.generated_tangents import NORMAL_UV1_FRAME_POLICY
        policy = NORMAL_UV1_FRAME_POLICY
        current = {'fur_undercoat': {'frame_policy': policy, 'plan': {'frame_policy': policy},
                                    'materials': [{'normal_texcoord': 0, 'frame_policy': None},
                                                  {'normal_texcoord': 1, 'frame_policy': policy}]}}
        self.assertEqual(_undercoat_frame_policy(current, True), policy)
        self.assertIsNone(_undercoat_frame_policy({}, False))
        for defect in ('missing_top', 'old_top', 'missing_plan', 'old_material', 'boolean_uv', 'no_materials'):
            bad = deepcopy(current); undercoat = bad['fur_undercoat']
            if defect == 'missing_top': undercoat.pop('frame_policy')
            if defect == 'old_top': undercoat['frame_policy'] = 'authored-unshared-atlas-blender-v-up-v1'
            if defect == 'missing_plan': undercoat.pop('plan')
            if defect == 'old_material': undercoat['materials'][1]['frame_policy'] = 'old'
            if defect == 'boolean_uv': undercoat['materials'][1]['normal_texcoord'] = True
            if defect == 'no_materials': undercoat['materials'] = []
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                _undercoat_frame_policy(bad, True)

    def test_adaptive_undercoat_budget_is_private_and_reserved_before_native(self):
        config = tuft_config()
        config['portable_fur']['settings']['profile'] = 'combed-v1'
        config['portable_fur']['bands'][0]['count'] = 3600
        config['fur_undercoat'] = {'profile':'short-pile-v1', 'resolution':2048, 'seed':41,
            'pitch_m':.004, 'strength':.22, 'roughness':.92, 'min_faces':1, 'max_faces':100000,
            'uv_policy':'material-adaptive-v1'}
        validate(config)
        with self.assertRaisesRegex(ValueError, 'Invalid generated finishing config'):
            validate({**config, 'undercoat_export_budget': {'source_export_vertices':1,
                      'portable_export_vertices':0, 'reserve_vertices':0}})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'analytic.glb'
            source.write_bytes(b'ANALYTIC PREFLIGHT FIXTURE ONLY')
            config.update(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            path = root/'config.json';path.write_text(json.dumps(config))
            with patch('avatar_assets.finish_generated.preflight', return_value={'vertices':55000,'skins':1}), \
                 patch('avatar_assets.finish_generated.run_bounded') as native, \
                 self.assertRaisesRegex(ValueError, 'Adaptive undercoat base, coat and reserve'):
                finish(path, root/'new-pack')
            native.assert_not_called()
            self.assertFalse((root/'new-pack').exists())

    def test_undercoat_is_opt_in_and_requires_shared_measured_groom_context(self):
        config = tuft_config()
        config['fur_undercoat'] = {'profile':'short-pile-v1','resolution':2048,'seed':41,
            'pitch_m':.004,'strength':.22,'roughness':.92,'min_faces':1,'max_faces':100000}
        original = deepcopy(config)
        validate(config)
        self.assertEqual(original, config)
        for key in ('groom', 'groom_flow'):
            bad=deepcopy(config);bad.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(bad)
        bad=deepcopy(config);bad['fur_undercoat']['resolution']=4096
        with self.assertRaisesRegex(ValueError, '1024 or2048'):
            validate(bad)

    def test_tuft_requires_exact_root_color_and_shared_native_flow(self):
        config = tuft_config();original = deepcopy(config)
        validate(config)
        self.assertEqual(config, original)
        for field in ('groom_flow', 'groom'):
            bad = deepcopy(config);bad.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(bad)
        for value in (False, None, 1):
            bad = deepcopy(config);bad['portable_fur']['settings']['sample_root_color'] = value
            with self.subTest(color=value), self.assertRaises(ValueError):
                validate(bad)
        bad = deepcopy(config);bad['portable_fur']['settings']['direction'] = [1, 0, 0]
        with self.assertRaisesRegex(ValueError, 'match the shared'):
            validate(bad)

    def test_tuft_budget_sums_bands_and_reserves_corners_before_native_or_output_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'analytic.glb';source.write_bytes(b'ANALYTIC PREFLIGHT FIXTURE ONLY')
            config = tuft_config()
            config.update(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
            config['portable_fur']['bands'] = [
                {'brightness':[0,.5],'color':[1,1,1,1],'count':2000},
                {'brightness':[.5,1.01],'color':[1,1,1,1],'count':2000}]
            path = root/'config.json';path.write_text(json.dumps(config))
            with patch('avatar_assets.finish_generated.preflight', return_value={'vertices':52000,'skins':1}), \
                 patch('avatar_assets.finish_generated.run_bounded') as native, \
                 self.assertRaisesRegex(ValueError, 'aggregate 100000'):
                finish(path, root/'new-pack')
            native.assert_not_called()
            self.assertFalse((root/'new-pack').exists())

    def test_full_resolution_review_controls_are_bounded(self):
        settings = finish_config()
        validate({**settings, 'pose_resolution': 720, 'review_views': True})
        for fields in ({'pose_resolution': True}, {'pose_resolution': 1025}, {'pose_resolution': 127},
                       {'pose_resolution': 720.0}, {'review_views': 1}, {'review_views': 'true'}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate({**settings, **fields})

    def test_flow_and_combed_root_colors_are_bounded_optional_controls(self):
        config = finish_config()
        flow = {'regions': [{'name':'crown','center':[.5,.5,.8],'radius':[.3,.4,.2],
                  'feather':.5,'direction':[0,0,1],'comb':.3,'length_scale':1.2}]}
        config['groom_flow'] = flow
        with self.assertRaisesRegex(ValueError, 'direction and comb'):
            validate(config)
        config['groom'].update(direction=[0,0,-1],comb=.6)
        config['portable_fur'] = {'settings':{'seed':1,'length':.01,'width':.004,
            'direction':[0,0,-1],'profile':'combed-v1','sample_root_color':True},
            'bands':[{'brightness':[0,1.01],'color':[1,1,1,1],'count':100}]}
        self.assertEqual(validate(config)['groom_flow'], flow)
        for key, value in [('profile','unknown'),('sample_root_color',1),('script','run.sh')]:
            bad = deepcopy(config);bad['portable_fur']['settings'][key] = value
            with self.assertRaises(ValueError):
                validate(bad)

    def test_groom_shader_accepts_both_inclusive_roughness_limits(self):
        for roughness, radial in ((.04, 1), (1, .04)):
            settings = finish_config()
            settings['groom_shader'] = {'roughness': roughness, 'radial_roughness': radial}
            before = deepcopy(settings)
            validate(settings)
            self.assertEqual(settings, before)

    def test_groom_shader_requires_a_valid_native_groom(self):
        for groom in (None, {}, {'parents': 0, 'children': 0, 'length': .01, 'radius': .0001,
                                 'roughness': .001, 'seed': 3}):
            settings = finish_config()
            settings['groom'] = groom
            settings['groom_shader'] = {'roughness': .6, 'radial_roughness': .7}
            with self.subTest(groom=groom), self.assertRaises(ValueError):
                validate(settings)
        settings = finish_config()
        del settings['groom']
        settings['groom_shader'] = {'roughness': .6, 'radial_roughness': .7}
        with self.assertRaisesRegex(ValueError, 'requires native hair'):
            validate(settings)

    def test_groom_shader_requires_exactly_the_two_named_settings(self):
        for shader in (None, [], {}, {'roughness': .5}, {'radial_roughness': .5},
                       {'roughness': .5, 'radial_roughness': .5, 'specular': 1}):
            settings = finish_config()
            settings['groom_shader'] = shader
            with self.subTest(shader=shader), self.assertRaisesRegex(ValueError, 'two explicit roughness settings'):
                validate(settings)

    def test_both_groom_roughness_values_reject_nonfinite_boolean_and_out_of_bounds(self):
        for key in ('roughness', 'radial_roughness'):
            for value in (False, True, float('nan'), float('inf'), -float('inf'), 0, .039, 1.001, '0.5', None):
                settings = finish_config()
                settings['groom_shader'] = {'roughness': .5, 'radial_roughness': .5, key: value}
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    validate(settings)

    def test_legacy_and_spatial_smoothing_are_individually_valid_but_exclusive(self):
        legacy = {'iterations': 4, 'strength': .2}
        spatial = {'iterations': 4, 'strength': .2, 'max_displacement_ratio': .01,
                   'feather': .25, 'exclusions': []}
        for key, value in (('smoothing', legacy), ('surface_smoothing', spatial),
                           ('surface_smoothing', {**spatial, 'method': 'laplacian'})):
            settings = finish_config()
            settings[key] = value
            validate(settings)
        for legacy_value in (legacy, {}, None):
            settings = finish_config()
            settings.update(smoothing=legacy_value, surface_smoothing=spatial)
            with self.subTest(legacy=legacy_value), self.assertRaisesRegex(ValueError, 'Choose one smoothing method'):
                validate(settings)


class SharedColorTextureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1]/'scripts/blender_finish_generated.py'
        tree = ast.parse(source.read_text(), filename=str(source))
        helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'shared_color_texture')
        namespace = {}
        # Execute only the actual data helper, never the bpy imports or native entrypoint.
        exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), 'exec'), namespace)
        cls.shared_texture = staticmethod(namespace['shared_color_texture'])

    def test_identical_direct_active_uv_textures_are_accepted(self):
        image = object()
        first, _ = material(image)
        second, texture = material(image)
        self.assertIs(self.shared_texture([first, second]), texture)

    def test_different_images_are_rejected_even_with_identical_routes(self):
        first, _ = material(object())
        second, _ = material(object())
        with self.assertRaises(ValueError):
            self.shared_texture([first, second])

    def test_same_name_different_uv_layers_and_identical_explicit_routes_are_rejected(self):
        image = object()
        first_route = [('UVMAP', 'UV Map', 'UV', 'UV_Main')]
        for second_route in (first_route, [('UVMAP', 'UV Map', 'UV', 'UV_Alternate')]):
            first, _ = material(image, route=first_route)
            second, _ = material(image, route=second_route)
            with self.subTest(route=second_route), self.assertRaises(ValueError):
                self.shared_texture([first, second])
        mapped, _ = material(image, route=[('MAPPING', 'Mapping', 'Vector', None)])
        with self.assertRaises(ValueError):
            self.shared_texture([mapped])

    def test_nonflat_projection_and_nonrepeat_wrapping_are_rejected(self):
        for changes in ({'projection': 'BOX'}, {'projection': 'SPHERE'}, {'extension': 'CLIP'}, {'extension': 'EXTEND'}):
            source, _ = material(object(), **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.shared_texture([source])

    def test_missing_or_indirect_base_color_sources_are_rejected(self):
        untextured, _ = material(None)
        indirect, texture = material(object())
        texture.type = 'MIX_RGB'
        no_link, _ = material(object())
        no_link.node_tree.nodes[0].inputs['Base Color'].links.clear()
        ambiguous, _ = material(object())
        ambiguous.node_tree.nodes.append(deepcopy(ambiguous.node_tree.nodes[0]))
        for sources in ([], [None], [untextured], [indirect], [no_link], [ambiguous]):
            with self.subTest(count=len(sources)), self.assertRaises(ValueError):
                self.shared_texture(sources)


class TuftSamplingAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1]/'scripts/blender_finish_generated.py'
        tree = ast.parse(source.read_text(), filename=str(source))
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in ('exact_root_texture_contract', 'tuft_sampling_context', 'portable_coat')]
        cls.scope = {'np': np}
        exec(compile(ast.Module(body=helpers, type_ignores=[]), str(source), 'exec'), cls.scope)

    def fixture(self):
        vertices = [SimpleNamespace(co=co,normal=(0,0,1),groups=[SimpleNamespace(group=0,weight=1.)])
                    for co in ((0,0,0),(1,0,0),(0,1,1),(1,1,1))]
        triangles = [SimpleNamespace(vertices=(0,1,2),loops=(0,1,2),material_index=0),
                     SimpleNamespace(vertices=(1,3,2),loops=(3,4,5),material_index=1)]
        uv = SimpleNamespace(name='UVMap',active_render=True,
                             data=[SimpleNamespace(uv=value) for value in [(0.125,.25)]*3+[(.875,.75)]*3])
        class UVLayers(list): pass
        layers = UVLayers([uv]);layers.active = uv
        obj = SimpleNamespace(vertex_groups=[SimpleNamespace(index=0,name='Root')],data=SimpleNamespace(
            vertices=vertices,loop_triangles=triangles,calc_loop_triangles=Mock(),
            uv_layers=layers,attributes=SimpleNamespace(get=lambda _:None)))
        pixels = np.array([[[1,0,0,1]]*2+[[0,0,1,1]]*2]*2,dtype=np.float32)
        image = SimpleNamespace(size=(4,2),colorspace_settings=SimpleNamespace(name='sRGB'),
                                pixels=SimpleNamespace(foreach_get=Mock(side_effect=lambda target:target.__setitem__(slice(None),pixels.ravel()))))
        obj.data.materials = []
        for _ in range(2):
            source, texture = material(image)
            texture.interpolation = 'Linear'
            obj.data.materials.append(source)
        return obj, triangles, image

    def modules(self):
        return patch.dict(sys.modules, {'fur_root_sampling':fur_root_sampling,'groom_flow':groom_flow})

    def test_context_pairs_real_loop_uvs_and_native_full_emitter_bounds(self):
        obj, triangles, image = self.fixture();config = tuft_config()
        with self.modules():
            sampler, field, report = self.scope['tuft_sampling_context'](obj,image,triangles,config['groom'],config['groom_flow'])
        self.assertEqual(sampler(0,[.2,.3,.5]),(1.,0.,0.))
        self.assertEqual(sampler(1,[.2,.3,.5]),(0.,0.,1.))
        self.assertEqual(field.low,(0.,0.,0.));self.assertEqual(field.inverse_span,(1.,1.,1.))
        self.assertEqual(report['borrowed_pixel_buffer_bytes'],4*2*4*4)
        self.assertEqual(sampler.pixels.dtype,np.float32)
        image.pixels.foreach_get.assert_called_once()

    def test_band_triangle_ordinal_remains_paired_with_its_actual_uv_island(self):
        obj, _, image = self.fixture();config = tuft_config()
        config['portable_fur']['bands'] = [{'brightness':[.5,1.01],'color':[1,1,1,1],'count':3}]
        colors = np.array([[0,0,0],[.4,.4,.4],[.4,.4,.4],[1,1,1]])
        captured = []
        def geometry(*args,**kwargs):
            result = blender_fur_cards.card_geometry(*args,**kwargs);captured.append(result)
            return result
        class SceneBoundary(Exception): pass
        self.scope['bpy'] = SimpleNamespace(data=SimpleNamespace(meshes=SimpleNamespace(new=Mock(side_effect=SceneBoundary))))
        with self.modules(), patch.dict(sys.modules, {'blender_fur_cards':SimpleNamespace(
                card_geometry=geometry,attach_fur_material=Mock())}), self.assertRaises(SceneBoundary):
            self.scope['portable_coat'](obj,None,[True]*4,colors,config['portable_fur'],Path('/unused'),
                tuft_image=image,native_groom=config['groom'],groom_flow=config['groom_flow'])
        self.assertEqual(len(captured),1)
        self.assertEqual(captured[0]['colors'],[(0.,0.,1.)]*18)

    def test_oversized_pixels_fail_before_allocating_or_reading(self):
        obj, triangles, image = self.fixture();config = tuft_config();image.size=(4097,4096)
        with self.modules(), self.assertRaisesRegex(ValueError, 'bounded explicit sRGB'):
            self.scope['tuft_sampling_context'](obj,image,triangles,config['groom'],config['groom_flow'])
        image.pixels.foreach_get.assert_not_called()


class StoredSurfaceTests(unittest.TestCase):
    def test_float32_collapse_is_rejected_before_writing_mesh(self):
        path = Path(__file__).resolve().parents[1]/'scripts/blender_finish_generated.py'
        tree = ast.parse(path.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'set_surface_positions')
        write = Mock()
        scope = {'np': np, 'write_json': write}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
        step = 2**-23
        original = np.array([[1.,0.,0.], [1.+step,0.,0.], [1.,1.,0.]])
        refined = original.copy()
        refined[:2,0] = [1.+.48*step, 1.+.499*step]
        triangles = np.array([[0,1,2]])
        self.assertEqual(surface_integrity.enforce_surface_change(original, refined, triangles)
                         ['regression_gate']['status'], 'passed')
        data = SimpleNamespace(calc_loop_triangles=Mock(), vertices=Mock(),
                               loop_triangles=[SimpleNamespace(vertices=(0,1,2))])
        with patch.dict(sys.modules, {'surface_integrity': surface_integrity}):
            with self.assertRaises(surface_integrity.SurfaceChangeError):
                scope['set_surface_positions'](SimpleNamespace(data=data), original, refined, Path('/unused'))
        data.vertices.foreach_set.assert_not_called()
        self.assertEqual(write.call_args.args[1]['new_degenerate_count'], 1)


class DeformationBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'scripts/blender_finish_generated.py'
        tree=ast.parse(path.read_text())
        helper=next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name=='deformation_baseline')
        cls.scope={'np':np,'math':math,'hashlib':hashlib}
        exec(compile(ast.Module(body=[helper],type_ignores=[]),str(path),'exec'),cls.scope)

    def fixture(self):
        class Collection:
            def __init__(self, values): self.values=np.asarray(values)
            def __len__(self): return len(self.values)
            def foreach_get(self, key, target): target[:]=self.values.ravel()
        obj=SimpleNamespace(matrix_world=np.eye(4),data=SimpleNamespace(
            vertices=Collection([[0,0,0],[1,0,0],[0,1,2]]),edges=Collection([[0,1],[1,2],[2,0]])))
        rig=SimpleNamespace(data=SimpleNamespace(pose_position='REST',bones={'Root':None,'Arm':None}))
        return obj,rig

    def test_source_skin_and_authored_routes_capture_same_bound_geometry(self):
        obj,rig=self.fixture()
        first=self.scope['deformation_baseline'](obj,rig,'source_skin')
        second=self.scope['deformation_baseline'](obj,rig,'authored')
        np.testing.assert_array_equal(first[0],second[0])
        np.testing.assert_array_equal(first[1],second[1])
        report=first[2]
        self.assertEqual(report['rest_sha256'],hashlib.sha256(first[0].astype('<f8').tobytes()).hexdigest())
        self.assertEqual(report['edges_sha256'],hashlib.sha256(first[1].astype('<i4').tobytes()).hexdigest())
        self.assertEqual(report['bone_names'],['Root','Arm'])
        self.assertEqual(report['rest_bounds']['height'],2)
        self.assertEqual(report['units'],'meters')
        self.assertEqual(report['pose_samples_measured'],0)
        self.assertTrue(report['unchanged_topology_required'])

    def test_ambiguous_pose_space_and_invalid_geometry_are_not_empty_successes(self):
        for scenario in ('posed','transform','nan','edge','rig'):
            obj,rig=self.fixture();route='source_skin'
            if scenario=='posed': rig.data.pose_position='POSE'
            if scenario=='transform': obj.matrix_world[0,0]=.01
            if scenario=='nan': obj.data.vertices.values=obj.data.vertices.values.astype(float);obj.data.vertices.values[0,0]=np.nan
            if scenario=='edge': obj.data.edges.values[0,0]=10
            if scenario=='rig': route='unknown'
            with self.subTest(scenario=scenario),self.assertRaises(ValueError):
                self.scope['deformation_baseline'](obj,rig,route)


if __name__ == '__main__':
    unittest.main()
