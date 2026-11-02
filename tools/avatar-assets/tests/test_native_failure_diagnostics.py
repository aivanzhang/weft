"""Analytic exceptions and immutable failed packs; no Blender or provider execution."""
import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from avatar_assets.pipeline import Pipeline, EvidenceCustodyError
from avatar_assets.safety import artifacts, read_json, sha256, write_json
from scripts import blender_fur_undercoat as undercoat
from tests.test_fur_undercoat import fixture, settings

ROOT=Path(__file__).resolve().parents[1]


def native_functions(filename, implementation):
    # Execute the actual pure wrapper/writer bodies, without importing Blender or native modules.
    source=ast.parse((ROOT/'scripts/blender_finish_generated.py').read_text())
    nodes=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in ('_native_failure','finish')]
    scope={'Path':Path,'json':json,'sys':sys,'hash_file':sha256,'write_json':write_json,
           '__file__':str(filename),'_finish':implementation}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(filename),'exec'),scope)
    return scope


class NativeFailureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.pack=Path(self.temp.name).resolve()/'pack';(self.pack/'sources').mkdir(parents=True)
        for name in ('blender_finish_generated.py','blender_fur_undercoat.py'):
            shutil.copyfile(ROOT/'scripts'/name,self.pack/'sources'/name)
        self.config={'source_sha256':'a'*64,'fur_undercoat':{**settings(),'min_faces':3}}
        write_json(self.pack/'finish-config.json',self.config)
        self.worker=Pipeline.__new__(Pipeline)

    def rejection(self, scope='uv0_bake'):
        cfg=deepcopy(self.config['fur_undercoat'])
        if scope=='adaptive_plan':cfg['uv_policy']='material-adaptive-v1'
        return undercoat.UndercoatSelectionError(np.array([True,True]),cfg,scope)

    def run_failure(self, error):
        implementation=Mock(side_effect=error)
        scope=native_functions(self.pack/'sources/blender_finish_generated.py',implementation)
        with self.assertRaises(type(error)) as caught:
            scope['finish'](self.pack/'finish-config.json',self.pack)
        self.assertIs(caught.exception,error)
        implementation.assert_called_once()
        return read_json(self.pack/'native-failure.json')

    def seal(self):
        if not (self.pack/'job.json').exists():
            write_json(self.pack/'job.json',{'exit_code':1,'process_exit_verified':True,'manual_stop_required':False})
        path=self.pack/'manifest.json'
        value={'schema_version':1,'status':'failed','error':'Finishing failed; inspect preserved attempt',
               'artifacts':artifacts(self.pack)}
        if path.exists():path.write_text(json.dumps(value))
        else:write_json(path,value)

    def test_both_count_paths_reject_before_baking_with_exact_counts(self):
        values=list(fixture());values[-1]={**settings(),'min_faces':3}
        before=[item.copy() for item in values[:7]]
        with patch.object(undercoat,'_raster') as raster, self.assertRaises(undercoat.UndercoatSelectionError) as caught:
            undercoat.bake_short_pile(*values)
        raster.assert_not_called()
        self.assertEqual(caught.exception.diagnostic['scope'],'uv0_bake')
        self.assertEqual({key:caught.exception.diagnostic[key] for key in ('selected_faces','min_faces','max_faces','total_faces')},
                         {'selected_faces':2,'min_faces':3,'max_faces':10,'total_faces':2})
        for current,old in zip(values[:7],before):np.testing.assert_array_equal(current,old)
        with patch('scripts.undercoat_uv_atlas.plan_normal_atlases') as atlas, self.assertRaises(undercoat.UndercoatSelectionError) as caught:
            undercoat.plan_material_undercoat(values[0],values[1],values[2],values[6],np.zeros(2,dtype=np.int32),
                {**values[-1],'uv_policy':'material-adaptive-v1'},
                export_vertex_budget={'source_export_vertices':4,'portable_export_vertices':0,'reserve_vertices':0})
        atlas.assert_not_called();self.assertEqual(caught.exception.diagnostic['scope'],'adaptive_plan')

    def test_exact_native_failure_reaches_repair_measurements_with_hashes(self):
        report=self.run_failure(self.rejection());self.seal()
        measured=self.worker._diagnostics(self.pack)['native_failure']
        self.assertEqual(measured['report'],report)
        self.assertEqual(measured['receipt']['sha256'],sha256(self.pack/'native-failure.json'))
        self.assertEqual(measured['manifest_sha256'],sha256(self.pack/'manifest.json'))
        self.assertIn('selected=2, min=3, max=10',report['message'])
        self.assertFalse(report['thresholds_changed']);self.assertFalse(report['automatic_repair_performed'])
        self.assertEqual(read_json(self.pack/'finish-config.json'),self.config)

    def test_generic_native_exception_is_bounded_and_no_log_scraping_occurs(self):
        report=self.run_failure(RuntimeError('Measured shader route rejected '+'x'*2000));self.seal()
        (self.pack/'finish.log').write_text('FAKE undercoat selected=999 and fake success')
        result=self.worker._diagnostics(self.pack)['native_failure']['report']
        self.assertEqual(result,report);self.assertTrue(report['message_truncated'])
        self.assertEqual(len(report['message']),1000);self.assertEqual(report['measurements'],{})

    def test_native_report_never_overwrites_or_swallows_original_failure(self):
        original=self.run_failure(self.rejection());saved=(self.pack/'native-failure.json').read_bytes()
        error=RuntimeError('Second independent error')
        scope=native_functions(self.pack/'sources/blender_finish_generated.py',Mock(side_effect=error))
        with patch('sys.stderr',new=io.StringIO()),self.assertRaises(RuntimeError) as caught:
            scope['finish'](self.pack/'finish-config.json',self.pack)
        self.assertIs(caught.exception,error)
        self.assertEqual((self.pack/'native-failure.json').read_bytes(),saved)
        self.assertEqual(read_json(self.pack/'native-failure.json'),original)

    def test_changed_missing_or_unbound_receipt_blocks_instead_of_repairing(self):
        self.run_failure(self.rejection());self.seal()
        path=self.pack/'native-failure.json';saved=path.read_bytes()
        path.write_bytes(saved+b' ')
        with self.assertRaises(EvidenceCustodyError):self.worker._diagnostics(self.pack)
        path.unlink()
        with self.assertRaises(EvidenceCustodyError):self.worker._diagnostics(self.pack)
        path.write_bytes(saved)
        manifest=read_json(self.pack/'manifest.json')
        manifest['artifacts']=[row for row in manifest['artifacts'] if row['path']!='native-failure.json']
        (self.pack/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(EvidenceCustodyError):self.worker._diagnostics(self.pack)

    def test_inconsistent_counts_settings_source_and_process_custody_are_rejected(self):
        original=self.run_failure(self.rejection());self.seal()
        mutations=[lambda r:r['measurements']['undercoat_selection'].update(selected_faces=True),
                   lambda r:r['measurements']['undercoat_selection'].update(selected_faces=4,total_faces=5),
                   lambda r:r['measurements']['undercoat_selection'].update(min_faces=4),
                   lambda r:r['measurements']['undercoat_selection'].update(scope='adaptive_plan'),
                   lambda r:r['measurements']['undercoat_selection'].update(implementation_sha256='b'*64),
                   lambda r:r.update(source_sha256='b'*64),lambda r:r.update(measurements={}),
                   lambda r:r.update(automatic_repair_performed=True)]
        for mutate in mutations:
            value=deepcopy(original);mutate(value)
            (self.pack/'native-failure.json').write_text(json.dumps(value));self.seal()
            with self.subTest(value=value),self.assertRaises(EvidenceCustodyError):self.worker._diagnostics(self.pack)
        (self.pack/'native-failure.json').write_text(json.dumps(original))
        for job in ({'exit_code':0,'process_exit_verified':True}, {'exit_code':1,'process_exit_verified':False},
                    {'exit_code':1,'process_exit_verified':True,'manual_stop_required':True}):
            (self.pack/'job.json').write_text(json.dumps(job));self.seal()
            with self.subTest(job=job),self.assertRaises(EvidenceCustodyError):self.worker._diagnostics(self.pack)

    def test_old_failed_pack_without_structured_receipt_remains_readable(self):
        (self.pack/'sources/blender_finish_generated.py').write_text('def finish(config, output):\n    pass\n')
        self.seal();self.assertEqual(self.worker._diagnostics(self.pack),{})

    def test_modern_missing_receipt_blocks_startup_crash_and_other_unexplained_failures(self):
        for exit_code in (-11, -9, 1):
            job={'exit_code':exit_code,'process_exit_verified':True,'manual_stop_required':False}
            (self.pack/'job.json').write_text(json.dumps(job));self.seal()
            with self.subTest(exit_code=exit_code),self.assertRaisesRegex(EvidenceCustodyError,'uninformed paid repair'):
                self.worker._diagnostics(self.pack)

    def test_missing_receipt_authenticates_saved_code_before_classifying_old_or_modern(self):
        self.seal()
        source=self.pack/'sources/blender_finish_generated.py'
        source.write_text('def finish(config, output):\n    pass\n')
        with self.assertRaisesRegex(EvidenceCustodyError,'artifact binding changed'):
            self.worker._diagnostics(self.pack)
        source.unlink()
        with self.assertRaises(EvidenceCustodyError):self.worker._diagnostics(self.pack)

    def test_successful_modern_pack_does_not_need_an_exception_receipt(self):
        self.seal();manifest=read_json(self.pack/'manifest.json');manifest['status']='needs_visual_review'
        (self.pack/'manifest.json').write_text(json.dumps(manifest))
        self.assertEqual(self.worker._diagnostics(self.pack),{})


if __name__=='__main__':unittest.main()
