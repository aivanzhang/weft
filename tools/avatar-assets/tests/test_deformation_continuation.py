"""Completed-pack numerical rejection custody; all model/image bytes are analytic fixtures."""
import base64
import json
import shutil
import unittest

from avatar_assets import deformation_gate, pipeline, pipeline_continuation as cont
from avatar_assets.safety import artifacts, read_json, sha256, write_json
import test_pipeline_continuation as fixtures
from test_deformation_gate import outlier


class DeformationContinuationTests(unittest.TestCase):
    setUp = fixtures.ContinuationTests.setUp
    prepare = fixtures.ContinuationTests.prepare
    make_terminal = fixtures.ContinuationTests.make_terminal

    def numeric(self):
        p=self.parent;pack=p/'model-0';config=read_json(p/'finish-0.json')
        shutil.rmtree(pack);(pack/'sources').mkdir(parents=True)
        (p/'stage-failure-model-0.json').unlink()
        write_json(pack/'sources/finish-design.json',config)
        (pack/'model.glb').write_bytes(b'Analytic non-renderable completed geometry')
        states=[]
        for action in config['authored_actions']:
            sample=deformation_gate.motion_samples.sample_action(action,(0,96),24)
            row=outlier();row['endpoint_influences']=[{'Base':1.0},{'Tube':1.0}]
            states.append({'id':action['id'],'duration_seconds':4,**sample,'deformation_outliers':[row]})
        report={'schema_version':1,'source_sha256':config['source_sha256'],'source_unchanged':True,
                'authored_rig':{'design':config['authored_rig'],'height_meters':self.plan['height_meters'],
                                'authored_not_recovered':True},'states':states}
        write_json(pack/'finish-report.json',report)
        write_json(pack/'job.json',{'exit_code':0,'failure':None,'process_exit_verified':True,'manual_stop_required':False})
        write_json(pack/'manifest.json',{'schema_version':1,'status':'needs_visual_review','id':config['id'],
            'source_sha256':config['source_sha256'],'states':config['states'],'finish':report,'artifacts':artifacts(pack)})
        gate=deformation_gate.audit_finish_pack(pack,expected_source_sha256=config['source_sha256'],
            expected_design_sha256=sha256(p/'finish-0.json'),expected_state_ids=[s['id'] for s in self.plan['states']],
            authored_height=self.plan['height_meters'])
        write_json(p/'deformation-gate-0.json',gate)
        error='Severe sampled skin deformation: '+json.dumps(gate['states'])
        (p/'events').mkdir(exist_ok=True)
        for index,event in enumerate(({'stage':'build','status':'complete'},
                                     {'stage':'deformation_gate','status':'failed','reason':error[:4000]},
                                     {'stage':'attempt','status':'rejected','reason':error})):
            write_json(p/f'events/{index:04}.json',{'attempt':0,**event})
        pixel=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=')
        for name in ('guide-front.png','guide-side.png','three-quarter.png'):
            (p/'inspection'/name).write_bytes(pixel)
        evidence=p/'evidence-0';(evidence/'states').mkdir(parents=True)
        for state in states:(evidence/'states'/f"{state['id']}.png").write_bytes(pixel)
        write_json(evidence/'evidence.json',{'finish_report_sha256':sha256(pack/'finish-report.json'),'images':artifacts(evidence)})
        findings={'validation_failure':error,'measurements':pipeline.Pipeline.__new__(pipeline.Pipeline)._diagnostics(pack),
                  'repair_images':[str(evidence/row['path']) for row in read_json(evidence/'evidence.json')['images']]}
        findings['measurements']['author_geometry']=self.geometry
        terminal=read_json(p/'unsupported.json');terminal['findings']=findings
        (p/'unsupported.json').write_text(json.dumps(terminal))
        self.network.reset_mock();self.network.side_effect=AssertionError('No provider calls during collection')
        return p

    def mutate(self,path,change):
        value=read_json(path);change(value);path.write_text(json.dumps(value))

    def test_completed_model_with_exact_failed_gate_is_continuable_without_mutation(self):
        p=self.numeric();before=cont._inventory(p);ledger=sha256(self.root/'.local/openai/ledger.sqlite3')
        captured=cont.collect(self.root,p)
        self.assertEqual(captured['context']['previous_answer'],self.answer)
        self.assertEqual(captured['context']['findings'],read_json(p/'unsupported.json')['findings'])
        self.assertNotIn('native_failure',captured['context']['findings']['measurements'])
        self.assertEqual(cont._inventory(p),before);self.assertEqual(sha256(self.root/'.local/openai/ledger.sqlite3'),ledger)
        self.network.assert_not_called()

    def test_changed_gate_binding_or_threshold_cannot_be_used_as_rejection(self):
        p=self.numeric();path=p/'deformation-gate-0.json';original=path.read_bytes()
        for change in (lambda value:value['bindings'].update(design_sha256='f'*64),
                       lambda value:value['thresholds'].update(stretch_ratio_strictly_greater_than=7),
                       lambda value:value.update(passed=True,status='no_reported_severe_spike')):
            self.mutate(path,change)
            with self.assertRaisesRegex(ValueError,'exact measured rejection'):cont.collect(self.root,p)
            path.write_bytes(original)

    def test_changed_validator_source_never_executes_old_snapshot(self):
        p=self.numeric();self.mutate(p/'deformation-gate-0.json',lambda value:value.update(implementation_sha256='f'*64))
        with self.assertRaisesRegex(ValueError,'validator or sampler source changed'):cont.collect(self.root,p)

    def test_completed_pack_with_missing_gate_is_not_treated_as_native_failure(self):
        p=self.numeric();(p/'deformation-gate-0.json').unlink()
        with self.assertRaises((ValueError,OSError)):cont.collect(self.root,p)

    def test_changed_terminal_findings_and_missing_failure_events_reject(self):
        p=self.numeric();path=p/'unsupported.json';original=path.read_bytes()
        self.mutate(path,lambda value:value['findings'].update(validation_failure='Invented failure'))
        with self.assertRaisesRegex(ValueError,'terminal findings differ'):cont.collect(self.root,p)
        path.write_bytes(original);(p/'events/0001.json').unlink()
        with self.assertRaisesRegex(ValueError,'failed-stage events'):cont.collect(self.root,p)

    def test_resealed_changed_report_cannot_replace_saved_gate(self):
        p=self.numeric();pack=p/'model-0'
        report=read_json(pack/'finish-report.json');report['states'][0]['deformation_outliers']=[]
        (pack/'finish-report.json').write_text(json.dumps(report))
        manifest=read_json(pack/'manifest.json');manifest['finish']=report;manifest['artifacts']=artifacts(pack)
        (pack/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'exact measured rejection'):cont.collect(self.root,p)

    def test_unexplained_later_stage_and_changed_contact_sheet_reject(self):
        p=self.numeric();(p/'reload-0').mkdir()
        with self.assertRaisesRegex(ValueError,'unexplained later-stage'):cont.collect(self.root,p)
        (p/'reload-0').rmdir();(p/'evidence-0/states/idle.png').write_bytes(b'changed analytic pixels')
        with self.assertRaisesRegex(ValueError,'contact sheet changed'):cont.collect(self.root,p)

    def test_failed_native_fixture_still_uses_legacy_custody_route(self):
        self.network.reset_mock();self.network.side_effect=AssertionError('No provider calls')
        captured=cont.collect(self.root,self.parent)
        self.assertIn('native_failure',captured['context']['findings']['measurements'])
        self.network.assert_not_called()


if __name__=='__main__':unittest.main()
