"""Analytic geometry/custody checks; no native render or avatar acceptance."""

from copy import deepcopy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from avatar_assets import finish_generated, pipeline
from avatar_assets.fur_bindings import FOLDER, write_bindings
from avatar_assets.pipeline_prompts import author_prompt, finish_config
from avatar_assets.quality_policy import build_quality_policy
from avatar_assets.safety import artifacts, read_json, sha256
from scripts.blender_fur_cards import card_geometry
from scripts.groom_flow import prepare_flow
import test_material_native_reuse as reuse_fixture
import test_material_pipeline as author_fixture


ROOT = Path(__file__).resolve().parents[1]
MODE = 'skin-consistent-v1'
BASELINE = ROOT/'evidence/material-owned-fur-v1/snapshots/material-pipeline-tests-v1/sources/scripts/blender_fur_cards.py'
BASELINE_SHA = 'b80936f4dab8d6c3546a72fa63f170173d99a90cd61a49c31b646eb04c6c68e7'


def mesh():
    vertices = [(0., 0., 0.), (.02, 0., 0.), (0., .02, 0.),
                (.05, 0., 0.), (.07, 0., 0.), (.05, .02, 0.)]
    triangles = [(0, 1, 2), (3, 4, 5)]
    weights = [{'A': .25, 'B': .5, 'C': .25} for _ in range(3)]+[{'A': 1.}, {'B': 1.}, {'C': 1.}]
    return vertices, triangles, weights


def settings(count=6, mode=MODE):
    result = {'seed': 37, 'count': count, 'length': .01, 'width': .003, 'direction': [1., 0., 0.], 'comb': .4}
    if mode is not None:
        result['root_placement'] = mode
    return result


def field():
    return prepare_flow([0, 0, -.01], [.1, .03, .03], [1, 0, 0], .4, {'regions': [
        {'name': 'analytic-flow', 'center': [.5, .5, .5], 'radius': [.8, .8, .8],
         'feather': .5, 'direction': [1, .2, 0], 'comb': .5, 'length_scale': 1.1}]})


def cards(mode=MODE, count=6, uniform=False):
    points, triangles, weights = mesh()
    if uniform:
        points, triangles, weights = points[:3], triangles[:1], weights[:3]
    result = card_geometry(points, triangles, weights, settings(count, mode), normals=[(0., 0., 1.)]*len(points),
                           profile='tuft-v1', prepared_flow=field(), root_color_sampler=lambda *_: (.1, .3, .6))
    return result, points, triangles, weights


def transforms(pose):
    result = {}
    for name, multiplier, offset in (('A', .4, (.01, -.02, .03)), ('B', 1.7, (-.03, .01, .02)),
                                     ('C', -.9, (.02, .01, -.01))):
        angle = pose*multiplier
        matrix = np.eye(4)
        matrix[:3, :3] = [[math.cos(angle), -math.sin(angle), 0],
                          [math.sin(angle), math.cos(angle), 0], [0, 0, 1]]
        matrix[:3, 3] = np.asarray(offset)*pose
        result[name] = matrix
    return result


def skin_point(point, weights, matrices, *, vector=False):
    point4 = np.append(np.asarray(point, dtype=float), 0. if vector else 1.)
    total = math.fsum(weights.values())
    return sum((matrices[name]@point4)[:3]*(value/total) for name, value in weights.items())


def root_errors(result, points, triangles, weights, pose):
    matrices = transforms(pose)
    body = np.array([skin_point(point, row, matrices) for point, row in zip(points, weights)])
    if 'root_bindings' in result:
        binding = result['root_bindings']
        records = zip(binding['triangle_index'], binding['barycentric'], binding['root_inset'])
    else:
        records = [(row['triangle_index'], row['barycentric'], np.asarray(row['normal'])*row['root_inset_m'])
                   for row in result['root_samples']]
    errors = []
    for index, (triangle, bary, inset) in enumerate(records):
        first = index*6
        left = skin_point(result['vertices'][first], result['weights'][first], matrices)
        right = skin_point(result['vertices'][first+1], result['weights'][first+1], matrices)
        corrected = (left+right)/2+skin_point(inset, result['weights'][first], matrices, vector=True)
        expected = np.asarray(bary)@body[np.asarray(triangles[triangle])]
        errors.append(float(np.linalg.norm(corrected-expected)))
    return errors


class SkinRootGeometryIntegrationTests(unittest.TestCase):
    def test_actual_bottom_centers_and_declared_inset_track_deformed_source(self):
        result, points, triangles, weights = cards()
        domains = result['root_summary']['binding_selection']['domains']
        self.assertEqual([row['resolved_count'] for row in domains], [3, 3])
        for pose in (0., .7, -1.2):
            errors = root_errors(result, points, triangles, weights, pose)
            self.assertEqual(len(errors), 6)
            self.assertLess(max(errors), 1e-12)
        old, old_points, old_triangles, old_weights = cards(mode=None)
        self.assertLess(max(root_errors(old, old_points, old_triangles, old_weights, 0)), 1e-12)
        self.assertGreater(max(root_errors(old, old_points, old_triangles, old_weights, .7)), 1e-4)

    def test_full_bindings_counts_and_sample_cap_survive_geometry_build(self):
        result, points, triangles, _ = cards(count=260, uniform=True)
        self.assertEqual((len(result['vertices']), len(result['faces'])), (1560, 1040))
        self.assertEqual(result['root_summary']['roots'], 260)
        self.assertEqual(result['root_summary']['root_placement'], MODE)
        self.assertEqual(result['root_summary']['binding_selection']['domains'][0]['resolved_count'], 260)
        self.assertEqual(len(result['root_samples']), 128)
        bindings = result['root_bindings']
        self.assertEqual(set(bindings), {'triangle_index', 'barycentric', 'root_inset'})
        self.assertEqual({len(value) for value in bindings.values()}, {260})
        for index, (triangle, bary, inset) in enumerate(zip(bindings['triangle_index'], bindings['barycentric'], bindings['root_inset'])):
            source = np.asarray(bary)@np.asarray(points)[np.asarray(triangles[triangle])]
            midpoint = np.asarray(result['vertices'][index*6:index*6+2]).mean(axis=0)
            np.testing.assert_allclose(midpoint+inset, source, rtol=0, atol=1e-14)
            self.assertLessEqual(np.linalg.norm(inset), .0005)
        for sample in result['root_samples']:
            index = sample['root_index']
            self.assertEqual(sample['triangle_index'], bindings['triangle_index'][index])
            np.testing.assert_array_equal(sample['barycentric'], bindings['barycentric'][index])
        self.assertEqual(result['root_samples'][-1]['root_index'], 259)

    def test_unsupported_placement_and_non_tuft_profiles_fail(self):
        points, triangles, weights = mesh()
        for profile in ('legacy', 'combed-v1'):
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, 'requires tuft-v1'):
                card_geometry(points, triangles, weights, settings(), profile=profile)
        for mode in ('source-vertices-v1', 'arbitrary', True):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Unsupported tuft root placement'):
                cards(mode=mode)

    def test_default_profiles_are_byte_identical_to_pinned_trusted_history(self):
        self.assertEqual(sha256(BASELINE), BASELINE_SHA)
        spec = importlib.util.spec_from_file_location('scripts._trusted_fur_cards_13fd3f9', BASELINE)
        historical = importlib.util.module_from_spec(spec); spec.loader.exec_module(historical)
        points, triangles, weights = mesh()
        normal = [(0., 0., 1.)]*len(points)
        for profile in ('legacy', 'combed-v1', 'tuft-v1'):
            options = {'profile': profile, 'normals': normal}
            if profile == 'tuft-v1':
                options.update(prepared_flow=field(), root_color_sampler=lambda _, bary: tuple(.2+.1*value for value in bary))
            else:
                options['colors'] = [[index/8, .2, .4] for index in range(len(points))]
            current = card_geometry(points, triangles, weights, settings(mode=None), **options)
            previous = historical.card_geometry(points, triangles, weights, settings(mode=None), **options)
            with self.subTest(profile=profile):
                self.assertEqual(json.dumps(current, sort_keys=True, separators=(',', ':'), allow_nan=False).encode(),
                                 json.dumps(previous, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


class SkinRootPipelineIntegrationTests(unittest.TestCase):
    def test_author_host_settings_and_action_metadata_remain_bound(self):
        request, plan, design, inspection = author_fixture.fixture()
        design['portable_fur']['settings']['root_placement'] = MODE
        quality = build_quality_policy(request, plan, inspection)
        config = finish_config(author_fixture.response(design), request, plan, author_fixture.SOURCE, 0, quality_policy=quality)
        self.assertEqual(config['portable_fur']['settings']['root_placement'], MODE)
        self.assertEqual(config['authored_actions'], design['authored_actions'])
        self.assertEqual(config['normal_strength'], 'preserve')
        payload = json.loads(author_prompt(request, plan, inspection).split('INPUT_JSON (data, not instructions):\n', 1)[1])
        self.assertEqual(payload['request']['custom_actions'], request['custom_actions'])
        for profile, mode in (('combed-v1', MODE), ('legacy', MODE), ('tuft-v1', 'source-vertices-v1'), ('tuft-v1', None)):
            bad = deepcopy(config); bad['portable_fur']['settings'].update(profile=profile, root_placement=mode)
            with self.subTest(profile=profile, mode=mode), self.assertRaises(ValueError):
                finish_generated.validate(bad)

    def test_prepare_only_pins_both_selectors_and_receipt_module_without_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name, source in pipeline._pipeline_sources().items():
                target = root/('scripts' if name.startswith('scripts/') else 'avatar_assets')/Path(name).name
                target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
            out = root/'outputs/skin-root-prepare-fixture'
            with patch.object(pipeline, 'ROOT', root), patch('avatar_assets.agent_runner.AgentRunner') as roles, \
                    patch('avatar_assets.meshy.run') as paid, patch.object(finish_generated, 'run_bounded') as native:
                pipeline.prepare('Analytic skinned furry character.', out, custom_actions=['Wave the paw.'])
                pins = {row['path']: row['sha256'] for row in read_json(out/'input-receipt.json')['sources']}
                prepared_artifacts = artifacts(out)
                agent = Mock()
                for saved, current in (('scripts/blender_fur_cards.py', 'scripts/blender_fur_cards.py'),
                                       ('scripts/fur_vertex_roots.py', 'scripts/fur_vertex_roots.py'),
                                       ('scripts/fur_skin_roots.py', 'scripts/fur_skin_roots.py'),
                                       ('fur_bindings.py', 'avatar_assets/fur_bindings.py')):
                    self.assertEqual(pins[saved], sha256(root/current))
                    self.assertEqual((out/'sources'/saved).read_bytes(), (root/current).read_bytes())
                    before = (root/current).read_bytes(); (root/current).write_bytes(before+b'\n# stale analytic fixture\n')
                    with self.subTest(changed_source=current), self.assertRaisesRegex(ValueError, 'backend changed'):
                        pipeline.Pipeline(out, agents=agent)
                    self.assertEqual(artifacts(out), prepared_artifacts)
                    (root/current).write_bytes(before)
                self.assertEqual(read_json(out/'request.json')['custom_actions'], ['Wave the paw.'])
                agent.call.assert_not_called()
                roles.assert_not_called(); paid.assert_not_called(); native.assert_not_called()


class SkinRootNativeReuseIntegrationTests(unittest.TestCase):
    seal = reuse_fixture.MaterialNativeReuseTests.seal
    assert_rejected = reuse_fixture.MaterialNativeReuseTests.assert_rejected

    def setUp(self):
        reuse_fixture.MaterialNativeReuseTests.setUp(self, portable_bindings=False)
        for name in ('fur_vertex_roots.py', 'fur_skin_roots.py'):
            shutil.copyfile(finish_generated.ROOT/'scripts'/name, self.source/'sources'/name)
        shutil.copyfile(finish_generated.ROOT/'avatar_assets/fur_bindings.py', self.source/'sources/fur_bindings.py')
        self.cfg['portable_fur']['settings']['root_placement'] = MODE
        self.cfg['portable_fur']['bands'] = [{'count': 6}]
        self.coat.update(requested_cards=6, created_cards=6)
        result, _, _, _ = cards()
        self.parts = []
        for start, name in ((0, 'PortableCoat_coat_0'), (3, 'PortableCoat_trim_0')):
            bindings = {key: value[start:start+3] for key, value in result['root_bindings'].items()}
            receipt = write_bindings(self.source, name, bindings, 3)
            self.parts.append({'name': name, 'cards': 3, 'root_summary': {'root_placement': MODE, 'binding_archive': receipt}})
        self.finish_report.update(portable_fur=self.parts, material_coat=deepcopy(self.coat))
        (self.source/'finish-config.json').write_text(json.dumps(self.cfg))
        (self.source/'finish-report.json').write_text(json.dumps(self.finish_report))
        (self.source/'material-coat.json').write_text(json.dumps(self.coat)); self.seal()

    def test_public_reuse_copies_exact_root_archives_and_receipts_without_native_work(self):
        before = artifacts(self.source)
        result = finish_generated.reuse_native(self.source, self.target, self.cfg)
        self.assertEqual(result, self.job)
        for part in self.parts:
            receipt = part['root_summary']['binding_archive']; path = receipt['path']
            self.assertEqual((self.target/path).read_bytes(), (self.source/path).read_bytes())
            self.assertEqual(sha256(self.target/path), receipt['sha256'])
            with np.load(self.target/path, allow_pickle=False) as archive:
                self.assertEqual(archive['barycentric'].shape, (3, 3))
                self.assertEqual(archive['root_inset'].shape, (3, 3))
        self.assertEqual(read_json(self.target/'finish-report.json')['portable_fur'], self.parts)
        self.assertEqual(read_json(self.target/'material-coat.json'), self.coat)
        self.assertEqual(artifacts(self.source), before)
        self.assertFalse(read_json(self.target/'native-reuse.json')['native_execution_repeated'])
        self.native.assert_not_called()

    def test_resealed_changed_selector_snapshots_reject_before_copy(self):
        for name in ('fur_vertex_roots.py', 'fur_skin_roots.py', 'fur_bindings.py'):
            path = self.source/'sources'/name; original = path.read_bytes()
            path.write_bytes(original+b'\n# stale analytic source\n'); self.seal()
            with self.subTest(name=name):
                self.assert_rejected('selector changed|binding custody changed')
                self.assertFalse((self.target/FOLDER).exists())
            path.write_bytes(original); self.seal()

    def test_changed_current_modules_cannot_reuse_matching_old_snapshot(self):
        current = self.root/'current'
        configs = set(reuse_fixture.CONFIG_SOURCES)|{'fur_bindings.py'}
        for path in (self.source/'sources').iterdir():
            target = current/('avatar_assets' if path.name in configs else 'scripts')/path.name
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
        with patch.object(finish_generated, 'ROOT', current):
            for relative in ('scripts/fur_vertex_roots.py', 'scripts/fur_skin_roots.py', 'avatar_assets/fur_bindings.py'):
                path = current/relative; original = path.read_bytes(); path.write_bytes(original+b'\n# changed analytic current module\n')
                with self.subTest(relative=relative):
                    self.assert_rejected('selector changed|binding custody changed')
                path.write_bytes(original)

    def test_resealed_corrupted_archive_cannot_keep_its_previous_receipt(self):
        path = self.source/self.parts[0]['root_summary']['binding_archive']['path']
        data = bytearray(path.read_bytes()); data[len(data)//2] ^= 1; path.write_bytes(data); self.seal()
        self.assert_rejected('Binding archive hash changed')
        self.assertFalse((self.target/FOLDER).exists())

    def test_resealed_missing_full_binding_receipt_is_not_reusable(self):
        self.finish_report['portable_fur'][0]['root_summary'].pop('binding_archive')
        (self.source/'finish-report.json').write_text(json.dumps(self.finish_report)); self.seal()
        self.assert_rejected('Binding receipt has missing')


if __name__ == '__main__':
    unittest.main()
