"""Analytic fixtures test preservation and rejection; they are not generated assets."""

from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets.assemble_generated import assemble
from avatar_assets.finish_generated import validate
from avatar_assets.generated_tangents import repair
from avatar_assets.export2d import read_render_scene
from avatar_assets.meshy import run
from avatar_assets.normalize_generated import normalize
from avatar_assets.safety import sha256, write_json
from avatar_assets.three_d import _GLB
from test_three_d import analytic_rig, glb_bytes


def triangle():
    doc, binary, _ = analytic_rig()
    def add(rows, kind, component, fmt):
        binary.extend(b'\0'*(-len(binary)%4))
        offset = len(binary)
        for row in rows:
            binary.extend(struct.pack('<'+fmt*len(row), *row))
        doc['bufferViews'].append({'buffer': 0, 'byteOffset': offset, 'byteLength': len(binary)-offset})
        doc['accessors'].append({'bufferView': len(doc['bufferViews'])-1, 'componentType': component, 'count': len(rows), 'type': kind})
        return len(doc['accessors'])-1
    primitive = doc['meshes'][0]['primitives'][0]
    primitive['attributes']['NORMAL'] = add([(0, 0, 1)]*3, 'VEC3', 5126, 'f')
    primitive['attributes']['TEXCOORD_0'] = add([(0, 0), (1, 0), (1, 1)], 'VEC2', 5126, 'f')
    primitive['indices'] = add([(0,), (1,), (2,)], 'SCALAR', 5123, 'H')
    doc['buffers'][0]['byteLength'] = len(binary)
    return doc, binary


class GeneratedPreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.doc, self.binary = triangle()
        self.source = self.root/'input.glb'
        self.source.write_bytes(glb_bytes(self.doc, self.binary))

    def tearDown(self):
        self.temp.cleanup()

    def config(self):
        cfg = {'schema_version': 1, 'id': 'analytic', 'source': 'input.glb', 'source_sha256': sha256(self.source),
               'states': [{'id': 'bend', 'source': 'input.glb', 'sha256': sha256(self.source), 'meaning': 'Analytic triangle bending'}]}
        write_json(self.root/'config.json', cfg)
        return self.root/'config.json'

    def test_assembly_keeps_skin_and_copies_actual_motion(self):
        assemble(self.config(), self.root/'out')
        result = _GLB(self.root/'out/model.glb')
        self.assertEqual(result.doc['skins'], self.doc['skins'])
        self.assertEqual(result.binary[:len(self.binary)], self.binary)
        self.assertEqual([a['name'] for a in result.doc['animations']], ['bend'])
        proof = json.loads((self.root/'out/rig-verification.json').read_text())
        self.assertTrue(proof['motion']['animation_verified'])

    def test_static_clip_cannot_be_relabelled(self):
        self.doc['accessors'][4]['count'] = 1
        self.doc['accessors'][5]['count'] = 1
        self.source.write_bytes(glb_bytes(self.doc, self.binary))
        with self.assertRaisesRegex(ValueError, 'no verified skin motion'):
            assemble(self.config(), self.root/'out')
        self.assertFalse((self.root/'out').exists())

    def test_different_rest_pose_requires_retargeting(self):
        cfg_path = self.config()
        other = deepcopy(self.doc)
        other['nodes'][2]['translation'] = [0, 2, 0]
        motion = self.root/'different.glb'
        motion.write_bytes(glb_bytes(other, self.binary))
        cfg = json.loads(cfg_path.read_text())
        cfg['states'][0].update(source='different.glb', sha256=sha256(motion))
        cfg_path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, 'rest hierarchy differs'):
            assemble(cfg_path, self.root/'out')

    def test_normal_cleanup_keeps_skin_motion_uv_and_positions(self):
        normalize(self.source, self.root/'clean')
        result = _GLB(self.root/'clean/model.glb')
        self.assertEqual(result.doc['skins'], self.doc['skins'])
        self.assertEqual(result.doc['animations'], self.doc['animations'])
        self.assertEqual(result.binary[:len(self.binary)], self.binary)
        before = _GLB(self.source)
        first = before.doc['meshes'][0]['primitives'][0]['attributes']
        second = result.doc['meshes'][0]['primitives'][0]['attributes']
        for name, kind, types in [('POSITION', 'VEC3', (5126,)), ('TEXCOORD_0', 'VEC2', (5126,)),
                                   ('JOINTS_0', 'VEC4', (5123,)), ('WEIGHTS_0', 'VEC4', (5126,))]:
            self.assertEqual(sorted(before.accessor(first[name], kind, types)), sorted(result.accessor(second[name], kind, types)))

    def test_changed_source_hash_rejected(self):
        config = self.config()
        self.source.write_bytes(self.source.read_bytes()+b'x')
        with self.assertRaisesRegex(ValueError, 'source changed'):
            assemble(config, self.root/'out')

    def test_undefined_tangent_derivative_preserves_original(self):
        offset = len(self.binary)
        self.binary.extend(struct.pack('<12f', 0,0,0,1, 1,0,0,1, 1,0,0,1))
        self.doc['bufferViews'].append({'buffer':0,'byteOffset':offset,'byteLength':48})
        self.doc['accessors'].append({'bufferView':len(self.doc['bufferViews'])-1,'componentType':5126,'count':3,'type':'VEC4'})
        self.doc['meshes'][0]['primitives'][0]['attributes']['TANGENT']=len(self.doc['accessors'])-1
        self.doc['buffers'][0]['byteLength']=len(self.binary)
        self.source.write_bytes(glb_bytes(self.doc,self.binary))
        original=self.source.read_bytes()
        report=repair(self.source,self.root/'repaired.glb')
        self.assertEqual(self.source.read_bytes(),original)
        self.assertEqual(len(report['repaired_vertices']),1)
        result=_GLB(self.root/'repaired.glb')
        self.assertEqual(result.doc,self.doc)
        self.assertEqual(result.binary[:offset],self.binary[:offset])
        self.assertEqual(result.binary[offset+16:offset+48],self.binary[offset+16:offset+48])


class GeneratedConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).resolve().parents[1]/'examples/meshy/muse-panda-finish-v1.json').read_text())

    def test_real_finish_designs_validate(self):
        validate(self.config)
        validate(json.loads((Path(__file__).resolve().parents[1]/'examples/meshy/workshop-robot-finish-v1.json').read_text()))

    def test_oversized_groom_rejected(self):
        self.config['groom']['parents'] = 10000
        with self.assertRaisesRegex(ValueError, 'fiber limit'):
            validate(self.config)

    def test_dense_profile_keeps_explicit_fiber_ceiling(self):
        self.config['native_render_children']=90
        validate(self.config)
        self.config['native_render_children']=91
        with self.assertRaisesRegex(ValueError,'600000'):
            validate(self.config)

    def test_invalid_action_and_region_rejected(self):
        self.config['authored_actions'][0]['keys'][1]['second'] = -1
        with self.assertRaises(ValueError):
            validate(self.config)
        self.setUp()
        self.config['groom_exclusions'][0]['radius'][0] = 0
        with self.assertRaises(ValueError):
            validate(self.config)

    def test_unbounded_texture_or_unknown_executable_config_rejected(self):
        self.config['resolution'] = 99999
        with self.assertRaises(ValueError):
            validate(self.config)
        self.setUp()
        self.config['script'] = 'arbitrary.py'
        with self.assertRaisesRegex(ValueError, 'Invalid generated finishing config'):
            validate(self.config)


class MeshyRunTests(unittest.TestCase):
    def test_dependency_order_and_no_quality_claim(self):
        plan = {'config': {'generation': {'mode': 'text'}, 'rig': 'humanoid', 'states': [{'id': 'wave'}]}}
        calls = []
        with patch('avatar_assets.meshy._plan', return_value=(Path('/analytic'), plan)), \
             patch('avatar_assets.meshy.submit', side_effect=lambda p,s,**kw: calls.append(s)), \
             patch('avatar_assets.meshy.status', return_value={'status': 'SUCCEEDED'}), \
             patch('avatar_assets.meshy.fetch', return_value={}):
            result = run('/analytic')
        self.assertEqual(calls, ['mesh', 'texture', 'rig', 'motion_wave', 'animate_wave'])
        self.assertFalse(result['quality_accepted'])

    def test_failed_stage_stops_dependent_paid_work(self):
        plan = {'config': {'generation': {'mode': 'text'}, 'rig': 'none', 'states': []}}
        with patch('avatar_assets.meshy._plan', return_value=(Path('/analytic'), plan)), \
             patch('avatar_assets.meshy.submit') as submit, \
             patch('avatar_assets.meshy.status', return_value={'status': 'FAILED'}):
            with self.assertRaisesRegex(ValueError, 'without paid retry'):
                run('/analytic')
            self.assertEqual(submit.call_count, 1)

    def test_pending_work_returns_resumable_checkpoint(self):
        plan = {'config': {'generation': {'mode': 'text'}, 'rig': 'none', 'states': []}}
        with patch('avatar_assets.meshy._plan', return_value=(Path('/analytic'), plan)), \
             patch('avatar_assets.meshy.submit'), \
             patch('avatar_assets.meshy.status', return_value={'status': 'IN_PROGRESS', 'task_id': 'fixture'}):
            result = run('/analytic', 1)
            self.assertEqual(result['status'], 'checkpointed')
            self.assertEqual(result['task_id'], 'fixture')


class GeneratedMediaDescriptionTests(unittest.TestCase):
    def test_named_short_states_keep_truthful_timing_and_no_loop(self):
        value = {'schema_version':1,'kind':'generated_render','id':'analytic','subject':'Analytic fixture',
                 'style':'Not an avatar','states':[{'id':'bend','meaning':'Test bend','duration':4,'loop':False}]}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'scene.json';write_json(p,value)
            self.assertEqual(read_render_scene(p),value)
            value['states'][0]['loop']=True
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'unverified loop'):
                read_render_scene(p)

    def test_invalid_generated_timing_and_duplicate_states_rejected(self):
        value = {'schema_version':1,'kind':'generated_render','id':'analytic','subject':'Fixture',
                 'style':'Analytic','states':[{'id':'bend','meaning':'Test','duration':99,'loop':False}]}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'scene.json';write_json(p,value)
            with self.assertRaises(ValueError):
                read_render_scene(p)
            value['states'][0]['duration']=4
            value['states']*=2
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                read_render_scene(p)


if __name__ == '__main__':
    unittest.main()
