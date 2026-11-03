"""Analytic automatic probe orchestration; mocked native transport, no provider calls."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock,patch

from avatar_assets import repair_admission as repair,undercoat_admission as host,pipeline
from avatar_assets.safety import artifacts,read_json,sha256,write_json
import test_undercoat_admission as fixtures


class RepairAdmissionTests(unittest.TestCase):
    seal=fixtures.AdmissionTests.seal

    def setUp(self):
        fixtures.AdmissionTests.setUp(self)
        self.jobroot=self.root/'job';self.jobroot.mkdir()
        self.pack=self.pack.rename(self.jobroot/'model-0')
        (self.pack/'sources').mkdir();write_json(self.pack/'sources/finish-design.json',self.cfg);self.seal()
        write_json(self.jobroot/'request.json',{})
        self.source_sha=self.cfg['source_sha256'];model=sha256(self.pack/'manifest.json')
        review={'verdict':'repair','summary':'Analytic coat target'}
        receipt={'review':review,'evidence':[],'model_manifest_sha256':model}
        write_json(self.jobroot/'review-0.json',receipt)
        paid=self.jobroot/'agents/reviewer-0';paid.mkdir(parents=True)
        write_json(paid/'answer.json',review)
        write_json(paid/'receipt.json',{'status':'succeeded','answer_sha256':sha256(paid/'answer.json')})
        write_json(paid/'request.json',{'images':[]})
        proof=self.jobroot/'final-evidence-0';proof.mkdir()
        write_json(proof/'manifest.json',{'model_manifest_sha256':model,'images':[],'artifacts':[]})
        self.current={'source_sha256':self.source_sha,'model_manifest_sha256':model,
            'receipt':{'path':'review-0.json','bytes':(self.jobroot/'review-0.json').stat().st_size,'sha256':sha256(self.jobroot/'review-0.json')}}
        self.dispatch=patch.object(host,'run_bounded',side_effect=self.native).start();self.addCleanup(patch.stopall)

    def native(self,command,log,out,**limits):
        self.assertEqual(limits,{'timeout':120,'memory_mb':2048,'output_mb':64})
        settings=read_json(out/'settings.json')
        write_json(out/'native-admission.json',{'policy':host.POLICY,'status':'refused','reason':'Analytic seam capacity refusal',
            'inputs_sha256':sha256(out/'admission-inputs.json'),'settings':settings,'input_writes':False,
            'counts':{'selected_triangles':1},'export_vertex_budget':{'source_export_vertices':4,'portable_export_vertices':24,'reserve_vertices':4096},
            'material_seam_capacity':{'maximum_cards_before_additional_uv_atlas':3753,'upper_bound_only':True,'admission_verified':False}})
        return {'exit_code':0,'failure':None,'process_exit_verified':True,'scope':'Analytic mock, no native process'}

    def feedback(self,**kwargs):
        return repair.feedback(self.root,self.jobroot,{},None,self.current,self.source_sha,**kwargs)

    def test_exact_valid_settings_and_automatic_refusal_cached_once(self):
        before={r['path']:r['sha256'] for r in artifacts(self.pack)};guard=Mock()
        first=self.feedback(before_probe=guard);second=self.feedback(before_probe=guard)
        self.assertEqual(first,second);self.assertEqual(first['status'],'refused')
        self.assertEqual(first['settings'],{**repair.SETTINGS,'min_faces':1,'max_faces':1})
        host.validate_undercoat(first['settings'])
        self.assertEqual(first['settings']['profile'],'short-pile-v1')
        self.assertFalse(first['quality_approved']);self.assertIn('Do not automatically reduce',first['scope'])
        self.assertEqual(first['material_seam_capacity']['maximum_cards_before_additional_uv_atlas'],3753)
        self.dispatch.assert_called_once();guard.assert_called_once()
        self.assertEqual({r['path']:r['sha256'] for r in artifacts(self.pack)},before)
        self.assertLess(len(json.dumps(first).encode()),16*1024)

    def test_no_review_and_nonfur_routes_do_not_dispatch(self):
        result=repair.feedback(self.root,self.jobroot,{},None,None,self.source_sha)
        self.assertEqual(result['status'],'not_applicable')
        self.cfg['fur_undercoat']={'profile':'short-pile-v1'}
        (self.pack/'sources/finish-design.json').write_text(json.dumps(self.cfg));self.seal()
        with patch.object(repair,'_target',return_value=(self.pack,self.manifest)):
            self.assertEqual(self.feedback()['status'],'not_applicable')
        self.dispatch.assert_not_called()

    def test_missing_exact_mask_count_is_not_guessed(self):
        self.report['portable_fur'][0]['root_summary']['binding_selection']['excluded_degenerate_triangles']=1;self.seal()
        with patch.object(repair,'_target',return_value=(self.pack,self.manifest)):
            result=self.feedback()
        self.assertEqual(result['status'],'not_applicable');self.dispatch.assert_not_called()

    def test_changed_paid_review_or_model_stops_before_native(self):
        (self.jobroot/'agents/reviewer-0/answer.json').write_text('{"verdict":"accept"}')
        with self.assertRaisesRegex(ValueError,'paid answer mismatch'):self.feedback()
        self.dispatch.assert_not_called()

    def test_failed_native_has_no_paid_repair_or_automatic_probe_retry(self):
        self.dispatch.side_effect=lambda *a,**kw:{'exit_code':-11,'failure':'Analytic crash','process_exit_verified':True}
        with self.assertRaisesRegex(ValueError,'Native admission failed'):self.feedback()
        with self.assertRaises(ValueError):self.feedback()
        self.dispatch.assert_called_once()

    def test_unknown_process_custody_cannot_be_a_refusal(self):
        self.dispatch.side_effect=lambda *a,**kw:{'exit_code':0,'failure':None,'process_exit_verified':False}
        with self.assertRaisesRegex(ValueError,'Native admission failed'):self.feedback()
        self.dispatch.assert_called_once()

    def test_cache_receipt_and_output_tamper_stop_without_replay(self):
        result=self.feedback();out=self.jobroot/Path(result['result_reference']['path']).parent
        receipt=self.jobroot/result['receipt_reference']['path'];saved=receipt.read_bytes()
        receipt.write_bytes(saved+b' ')
        # JSON whitespace is not an altered claim; semantic receipt fields still bind exact output bytes.
        self.assertEqual(self.feedback()['status'],'refused')
        (out/'result.json').write_bytes((out/'result.json').read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'incomplete, failed or changed'):self.feedback()
        self.dispatch.assert_called_once()

    def test_three_probe_job_bound_includes_incomplete_attempts(self):
        for i in range(3):(self.jobroot/('coat-admission-'+str(i)*64)).mkdir()
        with self.assertRaisesRegex(ValueError,'three probes'):self.feedback()
        self.dispatch.assert_not_called()

    def test_continuation_uses_bound_lineage_and_no_freeform_pack(self):
        summary={'parent_output':'outputs/ancestor','attempt':0,'model_manifest_sha256':self.current['model_manifest_sha256'],
                 'review_reference':{'sha256':self.current['receipt']['sha256']}}
        context={'findings':{'historical_visual_reviews':[summary]}}
        pins={r['path']:r for r in artifacts(self.jobroot)}
        request={'continuation':{'parent':'outputs/ancestor'}}
        with patch('avatar_assets.pipeline_continuation._read_context',return_value=({},context)), \
             patch.object(repair.repair_visual_inputs,'_lineage_inventory',return_value=(self.jobroot,pins,{'reuse_model':{'sha256':self.source_sha}})) as lineage:
            result=repair.feedback(self.root,self.jobroot,request,context,None,self.source_sha)
        self.assertEqual(result['status'],'refused');lineage.assert_called_once()
        self.dispatch.assert_called_once()

    def test_continuation_changed_context_is_not_followed(self):
        context={'findings':{'historical_visual_reviews':[{'parent_output':'/untrusted','attempt':0,'model_manifest_sha256':'a'*64,'review_reference':{}}]}}
        with patch('avatar_assets.pipeline_continuation._read_context',return_value=({},{})):
            with self.assertRaisesRegex(ValueError,'context differs'):
                repair.feedback(self.root,self.jobroot,{},context,None,self.source_sha)
        self.dispatch.assert_not_called()

    def test_pipeline_wraps_custody_error_and_never_calls_author(self):
        worker=pipeline.Pipeline.__new__(pipeline.Pipeline)
        worker.root=self.jobroot;worker.request={};worker.continuation=None;worker.guard=Mock();worker.event=Mock();worker.agents=Mock()
        with patch.object(repair,'feedback',side_effect=ValueError('Analytic unresolved native custody')):
            with self.assertRaises(pipeline.EvidenceCustodyError):worker._coat_admission_feedback(self.source_sha,None,0)
        worker.agents.call.assert_not_called()
        def success(*a,**kw):kw['before_probe']();return {'status':'refused'}
        with patch.object(repair,'feedback',side_effect=success):
            self.assertEqual(worker._coat_admission_feedback(self.source_sha,None,0),{'status':'refused'})
        self.assertEqual(worker.event.call_args_list[0].kwargs['reason'],'Checking the exported coat')
        worker.guard.assert_called_once_with(64)

    def test_actual_coordinator_stops_before_author_on_probe_custody_failure(self):
        import test_pipeline_progress as progress
        case=progress.PipelineProgressTests();case.setUp();self.addCleanup(case.doCleanups)
        with patch.object(repair,'feedback',side_effect=ValueError('Analytic native crash before receipt')):
            with self.assertRaises(pipeline.EvidenceCustodyError):case._coordinator()
        case.worker.agents.call.assert_not_called()
        self.assertEqual(read_json(case.root/'status.json')['status'],'checkpointed_error')

    def test_request_source_pins_cover_host_native_and_feedback_dependencies(self):
        pins=pipeline._pipeline_sources()
        for name in ['repair_admission.py','undercoat_admission.py','provider_model.py','runner.py','safety.py',
                     *['scripts/'+name for name in host.NATIVE_SOURCES]]:
            self.assertIn(name,pins);self.assertTrue(pins[name].is_file())


if __name__=='__main__':unittest.main()
