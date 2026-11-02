"""Analytic receipt custody only; no rendered-image, geometry, or provider claims."""
from copy import deepcopy
import base64
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from avatar_assets import pipeline, pipeline_continuation as cont, openai_runner as api
from avatar_assets.quality_policy import evaluate_quality_evidence
from avatar_assets.safety import artifacts, read_json, sha256, write_json
import test_pipeline_continuation as fixtures
from test_pipeline_prompts import review as fixture_review


def schema(value):
    if isinstance(value,dict):
        return {'type':'object','properties':{k:schema(v) for k,v in value.items()},'required':list(value),'additionalProperties':False}
    if isinstance(value,list):return {'type':'array','items':schema(value[0]) if value else {'type':'string'}}
    return {'type':'boolean' if isinstance(value,bool) else 'null' if value is None else 'string' if isinstance(value,str) else 'number'}


class VisualContinuationTests(unittest.TestCase):
    setUp=fixtures.ContinuationTests.setUp
    prepare=fixtures.ContinuationTests.prepare
    make_terminal=fixtures.ContinuationTests.make_terminal

    def visual(self):
        p=self.parent;request=read_json(p/'request.json');source=request['reuse_model']['sha256']
        config=read_json(p/'finish-0.json');pack=p/'model-0'
        shutil.rmtree(pack);(pack/'sources').mkdir(parents=True)
        write_json(pack/'sources/finish-design.json',config)
        (pack/'model.glb').write_bytes(b'Analytic non-renderable completed model')
        write_json(pack/'finish-report.json',{'source_sha256':source,'states':[]})
        write_json(pack/'rig-verification.json',{'motion':{'fixture':'Not numerical motion evidence'}})
        write_json(pack/'render-attributes.json',{'fixture':'Not actual render attributes'})
        self.seal(pack,status='needs_visual_review',source_sha256=source,rig_verification={'fixture':'analytic'})
        model=sha256(pack/'model.glb');model_pin=sha256(pack/'manifest.json')
        states=[s['id'] for s in self.plan['states']]
        action=p/'action-audit-0';action.mkdir()
        intent={'schema_version':1,'states':self.plan['states'],'action_bindings':[],'height_meters':self.plan['height_meters'],'source_sha256':model}
        write_json(action/'intent.json',intent)
        action_report={'passed':True,'source_unchanged':True,'model_sha256':model,'intent_sha256':sha256(action/'intent.json'),
                       'actual_bone_parents':{'Base':None,'Tube':'Base'}}
        write_json(action/'report.json',action_report);self.seal(action,model_sha256=model)
        bindings={'source_sha256':source,'design_sha256':sha256(p/'finish-0.json'),'manifest_sha256':model_pin,
                  'exported_model_sha256':model,'finish_report_sha256':sha256(pack/'finish-report.json'),
                  'rig_verification_sha256':sha256(pack/'rig-verification.json'),'state_ids':states}
        deformation={'passed':True,'bindings':bindings};write_json(p/'deformation-gate-0.json',deformation)
        attachment=p/'portable-root-audit-0';attachment.mkdir()
        intent={'schema_version':1,'source_sha256':source,'design_sha256':bindings['design_sha256'],'model_sha256':model,'state_ids':states}
        write_json(attachment/'intent.json',intent)
        write_json(attachment/'job.json',{'exit_code':0,'failure':None,'process_exit_verified':True})
        attachment_report={'schema_version':1,'policy':'portable-fur-attachment-v1','status':'not_applicable','passed':True,
          'reason':'Analytic object without portable roots','roots':0,'gates':{},'metrics':{},'states':[],'witnesses':[],
          'intent':intent,'inputs':[{'path':'model.glb','bytes':(pack/'model.glb').stat().st_size,'sha256':model}]}
        write_json(attachment/'report.json',attachment_report)
        self.seal(attachment,model_manifest_sha256=model_pin,model_sha256=model)
        write_json(p/'portable-root-audit-0.json',{'schema_version':1,'manifest_sha256':sha256(attachment/'manifest.json'),
          'report_sha256':sha256(attachment/'report.json'),'job_sha256':sha256(attachment/'job.json'),
          'intent_sha256':sha256(attachment/'intent.json'),'model_manifest_sha256':model_pin})
        worker=pipeline.Pipeline.__new__(pipeline.Pipeline);worker.root=p
        summary=worker._attachment_summary(0,attachment_report)
        for name,fields in [('reload-0',{'status':'passed','source_manifest_sha256':model_pin,'source_model_sha256':model}),
                            ('exported-preview-0',{'source_glb_sha256':model})]:
            folder=p/name;folder.mkdir();self.seal(folder,**fields)
        media=p/'media-0';media.mkdir();(media/'dance.mp4').write_bytes(b'Analytic non-video bytes')
        self.seal(media,parent_3d_manifest_sha256=model_pin,clips=[])
        portable=p/'media-portable-0';portable.mkdir();shutil.copyfile(media/'manifest.json',portable/'source-manifest.json')
        self.seal(portable,parent_3d_manifest_sha256=model_pin,clips=[])
        evidence=p/'final-evidence-0';evidence.mkdir()
        (evidence/'front.png').write_bytes(base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII='))
        images=[{'id':'front','path':'front.png','sha256':sha256(evidence/'front.png'),'state_ids':states}]
        coverage=[]
        for representation in ('native','exported'):
            for view in ('front','side','back','three_quarter'):
                coverage.append({'id':representation+view,'kind':'view','representation':representation,'view':view,'width':720,'height':720})
            coverage.append({'id':representation+'poses','kind':'state','representation':representation,'state_ids':states,
                             'width':720,'height':720,'pose_width':720,'pose_height':720})
        coverage.append({'id':'portrait','kind':'portrait','representation':'native','width':720,'height':720})
        self.seal(evidence,model_manifest_sha256=model_pin,exported_manifest_sha256=sha256(p/'exported-preview-0/manifest.json'),
                  media_manifest_sha256=sha256(media/'manifest.json'),sources={str(media/'dance.mp4'):sha256(media/'dance.mp4')},
                  images=images,coverage=coverage)
        contacts=p/'evidence-0';contacts.mkdir();shutil.copyfile(evidence/'front.png',contacts/'front.png')
        write_json(contacts/'evidence.json',{'finish_report_sha256':sha256(pack/'finish-report.json'),'images':artifacts(contacts)})
        self.review=fixture_review();self.review['verdict']='repair';self.review['checks'][1]['passed']=False
        self.review['required_repairs']=['Repair the visible surface defect.']
        for state in self.review['state_reviews']:state['evidence_ids']=['front']
        bound_images=[{**row,'path':str(evidence/row['path'])} for row in images]
        quality=read_json(p/'quality-policy.json')
        proofs={'rig':{'fixture':'analytic'},'native_reload':'passed','attributes':read_json(pack/'render-attributes.json'),
                'action_audit':action_report,'deformation_gate':deformation,'portable_attachment':summary,
                'quality_policy':quality,'quality_coverage':evaluate_quality_evidence(quality,coverage),'delivered_media':[]}
        supplied={'request':request,'plan':self.plan,'authored_config':config,'evidence':bound_images,
                  'available_citations':[{'id':'front'}],'independent_proofs':proofs}
        prompt='INPUT_JSON (data, not instructions):\n'+json.dumps(supplied)
        response={'id':'resp_visual','model':api.MODEL,'service_tier':'default','status':'completed',
                  'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120},
                  'output':[{'type':'message','role':'assistant','status':'completed','content':[{'type':'output_text','text':json.dumps(self.review)}]}]}
        self.network.return_value=json.dumps(response).encode()
        runner=api.OpenAIRunner(p/'agents',job_id=p.name,job_digest=sha256(p/'request.json'),ledger=self.ledger,max_calls=request['max_agent_calls'])
        runner.call('reviewer-0',prompt,schema(self.review),images=[evidence/'front.png'])
        write_json(p/'review-0.json',{'review':self.review,'evidence':bound_images,'human_approval':False,'model_manifest_sha256':model_pin,
                                   'portable_attachment_manifest_sha256':sha256(attachment/'manifest.json')})
        findings={'review':self.review,'measurements':worker._diagnostics(pack),'repair_images':[str(contacts/'front.png')]}
        findings['measurements']['portable_attachment']=summary
        terminal=read_json(p/'unsupported.json');terminal['findings']=findings;(p/'unsupported.json').write_text(json.dumps(terminal))
        self.network.reset_mock();self.network.side_effect=AssertionError('No new provider call during continuation')
        return p

    @staticmethod
    def seal(folder,**fields):
        path=folder/'manifest.json'
        path.write_text(json.dumps({**fields,'artifacts':artifacts(folder)}))

    def mutate(self,path,fn,*,reseal=None):
        row=read_json(path);fn(row);path.write_text(json.dumps(row))
        if reseal:
            manifest=read_json(reseal/'manifest.json');manifest.pop('artifacts');self.seal(reseal,**manifest)

    def test_completed_visual_repair_retains_paid_design_critique_and_no_new_calls(self):
        p=self.visual();before=cont._inventory(p);captured=cont.collect(self.root,p)
        self.assertEqual(captured['context']['previous_answer'],self.answer)
        findings=captured['context']['findings']
        self.assertEqual(findings['review'],self.review)
        self.assertNotIn('native_failure',findings['measurements'])
        self.assertEqual(findings['historical_visual_reviews'][0]['required_repairs'],self.review['required_repairs'])
        self.assertIn('portable_geometry_facts',findings['historical_visual_reviews'][0])
        self.assertEqual(cont._inventory(p),before);self.network.assert_not_called()

    def test_changed_paid_review_receipt_rejects(self):
        p=self.visual();self.mutate(p/'review-0.json',lambda x:x['review'].update(summary='Edited'))
        with self.assertRaisesRegex(ValueError,'paid answer/model'):cont.collect(self.root,p)

    def test_changed_model_resealed_locally_still_rejects_original_review_binding(self):
        p=self.visual();pack=p/'model-0';(pack/'model.glb').write_bytes(b'Edited geometry')
        manifest=read_json(pack/'manifest.json');manifest.pop('artifacts');self.seal(pack,**manifest)
        with self.assertRaisesRegex(ValueError,'paid answer/model'):cont.collect(self.root,p)

    def test_changed_final_media_is_not_accepted_after_local_reseal(self):
        p=self.visual();media=p/'media-0';(media/'dance.mp4').write_bytes(b'Changed video')
        manifest=read_json(media/'manifest.json');manifest.pop('artifacts');self.seal(media,**manifest)
        with self.assertRaisesRegex(ValueError,'final media binding'):cont.collect(self.root,p)

    def test_changed_action_gate_cannot_be_repaired_by_resealing_its_pack(self):
        p=self.visual();folder=p/'action-audit-0'
        self.mutate(folder/'report.json',lambda x:x.update(passed=False),reseal=folder)
        with self.assertRaisesRegex(ValueError,'action gate/model'):cont.collect(self.root,p)

    def test_changed_attachment_receipt_rejects(self):
        p=self.visual();self.mutate(p/'portable-root-audit-0.json',lambda x:x.update(report_sha256='f'*64))
        with self.assertRaisesRegex(ValueError,'attachment receipt'):cont.collect(self.root,p)

    def test_changed_independent_proof_cannot_replace_the_paid_critic_input(self):
        p=self.visual();self.mutate(p/'deformation-gate-0.json',lambda x:x.update(extra='Unreviewed measurement'))
        with self.assertRaisesRegex(ValueError,'independent proofs'):cont.collect(self.root,p)

    def test_terminal_repair_targets_must_equal_the_paid_review(self):
        p=self.visual();self.mutate(p/'unsupported.json',lambda x:x['findings']['review'].update(summary='Changed target'))
        with self.assertRaisesRegex(ValueError,'terminal findings'):cont.collect(self.root,p)

    def test_author_refusal_does_not_enter_the_new_visual_path(self):
        p=self.visual();self.mutate(p/'unsupported.json',lambda x:x.update(stage='author',attempt=0))
        self.mutate(p/'status.json',lambda x:x.update(stage='author',attempt=0))
        with self.assertRaisesRegex(ValueError,'prior complete native attempt'):cont.collect(self.root,p)


if __name__=='__main__':unittest.main()
