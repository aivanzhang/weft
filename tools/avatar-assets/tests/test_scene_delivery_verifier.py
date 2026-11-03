"""Analytic custody fixtures use non-renderable bytes; no provider or native jobs."""

from copy import deepcopy
from contextlib import closing, contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets.studio_jobs import _digest
from avatar_assets.studio_worker import MEDIA_SPECS, PIPELINE_LIMITS
from scripts import verify_scene_delivery as verifier


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pin(path, base):
    return {'path':path.relative_to(base).as_posix(), 'bytes':path.stat().st_size, 'sha256':checksum(path)}


@contextmanager
def database(path):
    with closing(sqlite3.connect(path)) as db, db:
        yield db


def seal(folder, fields):
    if not folder.exists() or not any(p.is_file() for p in folder.rglob('*')):
        write(folder/'fixture.txt', b'Analytic sealed pack; no computation executed.\n')
    write(folder/'manifest.json', {'schema_version':1, **fields, 'artifacts':[
        pin(p, folder) for p in sorted(folder.rglob('*')) if p.is_file() and p != folder/'manifest.json']})
    return checksum(folder/'manifest.json')


class SceneDeliveryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.state = self.root/'.local/studio'
        self.state.mkdir(parents=True)
        self.db = self.state/'jobs.sqlite3'
        self.scene_id = 'a'*24
        self.ids = ['b'*24, 'c'*24]
        self.assets = [dict(id='owl', name='Owl', prompt='One owl only.', style='Soft feathers',
                           custom_actions=['Flap both wings.'], preset='preview', reference=None),
                       dict(id='rabbit', name='Rabbit', prompt='One rabbit only.', style='Short plush',
                           custom_actions=['Bend both ears.'], preset='preview', reference=None)]
        self.scene = {'schema_version':1, 'title':'Two separate animals', 'mode':'explicit_independent_assets', 'assets':self.assets}
        write(self.root/'.local/studio-registry.json', {'schema_version':1,'state_dir':str(self.state)})
        with database(self.db) as db:
            db.executescript('CREATE TABLE scenes(id TEXT,request TEXT,request_digest TEXT);'
                'CREATE TABLE scene_assets(scene TEXT,asset TEXT,job TEXT);'
                'CREATE TABLE jobs(id TEXT,request TEXT,request_digest TEXT,data TEXT,output TEXT,status TEXT);'
                'CREATE TABLE active(job TEXT);')
            db.execute('INSERT INTO scenes VALUES(?,?,?)', (self.scene_id,json.dumps(self.scene),_digest(self.scene)))

    def child(self, index, status='complete', *, source_bytes=None, model_bytes=None):
        asset, identifier = self.assets[index], self.ids[index]
        request = {k:deepcopy(v) for k,v in asset.items() if k not in ('id','name')}
        request['reuse_model'] = None
        output = 'studio-'+identifier
        folder = self.root/'outputs'/output
        data = {'scene':{'id':self.scene_id,'title':self.scene['title'],'asset_id':asset['id'],'asset_name':asset['name']}}
        with database(self.db) as db:
            db.execute('INSERT INTO scene_assets VALUES(?,?,?)', (self.scene_id,asset['id'],identifier))
            db.execute('INSERT INTO jobs VALUES(?,?,?,?,?,?)', (identifier,json.dumps(request),_digest(request),json.dumps(data),output,status))
        write(self.state/'jobs'/identifier/'request.json', request)
        if status == 'created':
            return folder
        prepared = {'schema_version':1,'id':output, **request, 'states':[], 'media_spec':MEDIA_SPECS['preview'],
                    'max_credits':35, **PIPELINE_LIMITS, 'created':'analytic fixture'}
        write(folder/'request.json', prepared)
        write(self.state/'jobs'/identifier/'pipeline-request.json', {'schema_version':1,
            'studio_request_digest':_digest(request), 'pipeline_request_sha256':checksum(folder/'request.json')})
        write(folder/'sources/fixture.py', b'# Analytic source; never executed.\n')
        write(folder/'input-receipt.json', {'request_sha256':checksum(folder/'request.json'),
            'sources':[pin(folder/'sources/fixture.py', folder/'sources')]})
        if status == 'unsupported':
            write(folder/'unsupported.json', {'status':'unsupported','stage':'author','reason':'Fixture cannot meet the requested motion.',
                'request_sha256':checksum(folder/'request.json')})
            return folder
        state = {'id':'move', 'meaning':asset['custom_actions'][0], 'user_request':asset['custom_actions'][0],
                 'roles':[{'id':'appendages','description':'The requested moving parts.'}],
                 'requirements':[{'id':'articulate','kind':'articulation','role':'appendages','description':'Visible independent movement.'}]}
        plan = {'status':'ready','reason':'','subject':asset['name'],'style':asset['style'],
                'generation_prompt':asset['prompt'],'height_meters':1,'states':[state],
                'appearance_checks':['The requested subject and style are preserved.'],'limitations':[]}
        write(folder/'plan.json', plan)
        generation_config = {'id':output+'-mesh','subject':plan['subject'],'style':plan['style'],
                             'prompt':plan['generation_prompt'],'rig':'none','states':[]}
        write(folder/'generation-config.json', generation_config)
        write(folder/'generation/inputs/config.json', generation_config)
        write(folder/'generation/plan.json', {'config':generation_config})
        write(folder/'generation-receipt.json', {'plan_sha256':checksum(folder/'generation/plan.json')})
        source = folder/'generation/stages/texture/downloads'
        write(source/'model.glb', source_bytes or ('NONRENDERABLE SOURCE '+str(index)).encode())
        seal(source, {'kind':'meshy_original_downloads','stage':'texture','plan_sha256':checksum(folder/'generation/plan.json')})
        source_sha = checksum(source/'model.glb')
        model = folder/'model-0'
        write(model/'model.glb', model_bytes or ('NONRENDERABLE RIGGED MODEL '+str(index)).encode())
        write(model/'source.blend', b'NONRENDERABLE editable scene')
        model_sha = checksum(model/'model.glb')
        config = {'action_bindings':[{'state_id':'move','roles':[{'id':'appendages','bones':['tip'],'root_bone':'root'}]}]}
        write(folder/'finish-0.json', config)
        write(model/'sources/finish-design.json', config)
        design_sha = checksum(folder/'finish-0.json')
        write(model/'finish-report.json', {'states':[{'id':'move'}]})
        rig = {'sha256':model_sha, 'status':'validated_local_structure','loading_verified':True,'skin_count':1,
               'motion':{'animation_verified':True,'animations':[{'name':'move'}]}}
        write(model/'rig-verification.json', rig)
        model_manifest = seal(model, {'source_sha256':source_sha,'states':[state],'rig_verification':rig})
        gate = {'passed':True,'states':[{'id':'move','passed':True}], 'bindings':{
            'source_sha256':source_sha,'exported_model_sha256':model_sha,'manifest_sha256':model_manifest,
            'design_sha256':design_sha,'state_ids':['move'],'finish_report_sha256':checksum(model/'finish-report.json'),
            'rig_verification_sha256':checksum(model/'rig-verification.json')}}
        write(folder/'deformation-gate-0.json', gate)
        audit = folder/'action-audit-0'
        intent = {'states':[state],'action_bindings':config['action_bindings'],'source_sha256':model_sha}
        write(audit/'intent.json', intent)
        action = {'passed':True,'source_unchanged':True,'model_sha256':model_sha,'intent_sha256':checksum(audit/'intent.json'),
                  'states':[{'id':'move','evidence':{'state_id':'move','evidence_id':'audit-move'},
                             'requirement_results':{'passed':True,'requirements':[{'id':'articulate','status':'passed'}]}}]}
        write(audit/'report.json', action);seal(audit, {'status':'measured'})
        seal(folder/'reload-0', {'status':'passed','source_manifest_sha256':model_manifest,'source_model_sha256':model_sha})
        attachment = folder/'portable-root-audit-0'
        attach_intent = {'schema_version':1,'source_sha256':source_sha,'model_sha256':model_sha,
                         'design_sha256':design_sha,'state_ids':['move']}
        write(attachment/'intent.json', attach_intent)
        write(attachment/'report.json', {'passed':True,'status':'not_applicable','intent':attach_intent})
        write(attachment/'job.json', {'process_exit_verified':True})
        seal(attachment, {'status':'not_applicable'})
        custody = {'schema_version':1,'model_manifest_sha256':model_manifest,
                   **{key:checksum(attachment/name) for key,name in [('manifest_sha256','manifest.json'),
                       ('report_sha256','report.json'),('intent_sha256','intent.json'),('job_sha256','job.json')]}}
        write(folder/'portable-root-audit-0.json', custody)
        exported_sha = seal(folder/'exported-preview-0', {'status':'needs_visual_review','source_glb_sha256':model_sha})
        media_sha = seal(folder/'media-0', {'status':'needs_visual_review'})
        media = folder/'media-portable-0'
        write(media/'animations/move.mp4', b'ANALYTIC VIDEO; NOT DECODED')
        write(media/'fallbacks/move.png', b'ANALYTIC STILL; NOT DECODED')
        seal(media, {'parent_3d_manifest_sha256':model_manifest,'clips':[{
            'id':'move','path':'animations/move.mp4','fallback':'fallbacks/move.png',
            'inspection':{'sha256':checksum(media/'animations/move.mp4')},
            'fallback_inspection':{'sha256':checksum(media/'fallbacks/move.png')}}]})
        evidence = folder/'final-evidence-0'
        write(evidence/'motion.png', b'ANALYTIC IMAGE; NOT DECODED')
        images = [{'id':'motion','kind':'native_exported_and_delivered_motion','state_ids':['move'],
                   'path':'motion.png','sha256':checksum(evidence/'motion.png')}]
        seal(evidence, {'images':images,'model_manifest_sha256':model_manifest,
                        'exported_manifest_sha256':exported_sha,'media_manifest_sha256':media_sha})
        review = {'verdict':'accept','summary':'Analytic recorded acceptance, not actual media quality.',
                  'checks':[{'name':name,'passed':True,'reason':'Analytic check.','evidence_ids':['motion']}
                            for name in ('identity','appearance','framing','deformation')],
                  'state_reviews':[{'id':'move','meaning_passed':True,'motion_passed':True,'reason':'Analytic motion.',
                      'evidence_ids':['motion'],'requirement_reviews':[{'id':'articulate','passed':True,
                          'reason':'Analytic articulation.','evidence_ids':['motion']}]}],
                  'required_repairs':[],'limitations':['Fixture only.']}
        write(folder/'review-0.json', {'review':review,'model_manifest_sha256':model_manifest,
              'portable_attachment_manifest_sha256':custody['manifest_sha256'],
              'evidence':[{**item,'path':str(evidence/item['path'])} for item in images]})
        write(folder/'gallery.json', {'schema_version':1,'assets':[{'id':output,'kind':'generated_3d',
            'pack':f'outputs/{output}/model-0','media_pack':f'outputs/{output}/media-portable-0'}]})
        write(folder/'quality-policy.json', {'scope':'analytic fixture'})
        write(folder/'delivery.json', {'status':'complete','selected_attempt':0,'request_sha256':checksum(folder/'request.json'),
            'plan_sha256':checksum(folder/'plan.json'),'model':'model-0/model.glb','editable_source':'model-0/source.blend',
            'gallery_catalog':f'outputs/{output}/gallery.json',
            'review':review,'portable_attachment':{'passed':True,'status':'not_applicable','evidence':{
                'sha256':custody['report_sha256'],'manifest_sha256':custody['manifest_sha256']}},
            'packs':[{'path':name,'manifest_sha256':checksum(folder/name/'manifest.json')} for name in (
                'model-0','reload-0','media-0','media-portable-0','exported-preview-0','action-audit-0','final-evidence-0','portable-root-audit-0')]})
        self.reseal_delivery(folder)
        return folder

    def reseal_delivery(self, folder):
        write(folder/'delivery-receipt.json', {name:checksum(folder/name) for name in (
            'delivery.json','plan.json','gallery.json','quality-policy.json','review-0.json',
            'deformation-gate-0.json','portable-root-audit-0.json')})

    def test_two_independent_complete_deliveries_no_import_dispatch_or_mutation(self):
        self.child(0);self.child(1)
        before = {str(p):checksum(p) for p in self.root.rglob('*') if p.is_file()}
        with patch('avatar_assets.studio_jobs.StudioJobs', side_effect=AssertionError('No manager construction')), \
             patch('avatar_assets.three_d.validate_glb', side_effect=AssertionError('No model importer')), \
             patch('avatar_assets.meshy.run', side_effect=AssertionError('No provider')):
            result = verifier.inspect_scene(self.root, self.scene_id)
        self.assertTrue(result['complete']);self.assertEqual(result['status'],'verified_complete')
        self.assertTrue(result['claims']['separate_delivered_files'])
        self.assertFalse(any(v for k,v in result['claims'].items() if k != 'separate_delivered_files'))
        self.assertEqual(result['children'][0]['request']['custom_actions'],['Flap both wings.'])
        self.assertEqual(result['children'][1]['request']['custom_actions'],['Bend both ears.'])
        self.assertTrue(all(c['media'][0]['video_available'] and c['media'][0]['static_fallback_available'] for c in result['children']))
        self.assertEqual(before,{str(p):checksum(p) for p in self.root.rglob('*') if p.is_file()})

    def test_unsupported_child_is_partial_not_complete(self):
        self.child(0);self.child(1,'unsupported')
        result = verifier.inspect_scene(self.root,self.scene_id)
        self.assertEqual(result['status'],'partial');self.assertFalse(result['complete'])
        self.assertEqual(result['children'][1]['status'],'unsupported')
        self.assertNotIn('model',result['children'][1])

    def test_unstarted_and_missing_child_are_explicit(self):
        self.child(0,'created')
        result = verifier.inspect_scene(self.root,self.scene_id)
        self.assertEqual(result['status'],'not_complete')
        self.assertEqual([r['status'] for r in result['children']],['not_complete','missing_child'])

    def test_scene_action_or_membership_swap_is_rejected(self):
        self.child(0,'created');self.child(1,'created')
        with database(self.db) as db:
            db.execute('UPDATE scene_assets SET job=CASE asset WHEN ? THEN ? ELSE ? END',('owl',self.ids[1],self.ids[0]))
        with self.assertRaisesRegex(ValueError,'swapped or changed'):
            verifier.inspect_scene(self.root,self.scene_id)

    def test_duplicate_sources_rejected_even_with_distinct_final_models(self):
        self.child(0,source_bytes=b'SAME ORIGINAL SOURCE');self.child(1,source_bytes=b'SAME ORIGINAL SOURCE')
        with self.assertRaisesRegex(ValueError,'same source GLB'):
            verifier.inspect_scene(self.root,self.scene_id)

    def test_duplicate_final_models_rejected_with_distinct_sources(self):
        self.child(0,model_bytes=b'SAME DELIVERED MODEL');self.child(1,model_bytes=b'SAME DELIVERED MODEL')
        with self.assertRaisesRegex(ValueError,'same final GLB'):
            verifier.inspect_scene(self.root,self.scene_id)

    def test_model_tamper_and_unsealed_pack_file_are_rejected(self):
        first=self.child(0);self.child(1)
        write(first/'model-0/model.glb',b'CHANGED')
        with self.assertRaisesRegex(ValueError,'inventory differs'):
            verifier.inspect_scene(self.root,self.scene_id)
        write(first/'model-0/model.glb',b'NONRENDERABLE RIGGED MODEL 0')
        write(first/'model-0/extra.txt',b'UNSEALED')
        with self.assertRaisesRegex(ValueError,'unsealed artifacts'):
            verifier.inspect_scene(self.root,self.scene_id)

    def test_resealed_false_gate_is_still_rejected(self):
        first=self.child(0);self.child(1)
        gate=json.loads((first/'deformation-gate-0.json').read_text());gate['passed']=False
        write(first/'deformation-gate-0.json',gate);self.reseal_delivery(first)
        with self.assertRaisesRegex(ValueError,'Deformation gate'):
            verifier.inspect_scene(self.root,self.scene_id)

    def test_receipt_omission_and_active_complete_are_rejected(self):
        first=self.child(0);self.child(1)
        receipt=json.loads((first/'delivery-receipt.json').read_text());del receipt['review-0.json']
        write(first/'delivery-receipt.json',receipt)
        with self.assertRaisesRegex(ValueError,'omits required evidence'):
            verifier.inspect_scene(self.root,self.scene_id)
        self.reseal_delivery(first)
        with database(self.db) as db:db.execute('INSERT INTO active VALUES(?)',(self.ids[0],))
        with self.assertRaisesRegex(ValueError,'active worker'):
            verifier.inspect_scene(self.root,self.scene_id)

    def test_missing_static_fallback_is_not_claimed_available(self):
        first=self.child(0);self.child(1)
        (first/'media-portable-0/fallbacks/move.png').unlink()
        with self.assertRaises(ValueError):verifier.inspect_scene(self.root,self.scene_id)

    def test_resealed_gallery_cannot_point_to_the_other_child(self):
        first=self.child(0);second=self.child(1)
        gallery=json.loads((first/'gallery.json').read_text())
        gallery['assets'][0]['pack']=str((second/'model-0').relative_to(self.root))
        write(first/'gallery.json',gallery);self.reseal_delivery(first)
        with self.assertRaisesRegex(ValueError,'Gallery points to a different child'):
            verifier.inspect_scene(self.root,self.scene_id)

    def test_resealed_critic_duplicate_does_not_replace_required_check(self):
        first=self.child(0);self.child(1)
        delivery=json.loads((first/'delivery.json').read_text())
        delivery['review']['checks'][-1]=deepcopy(delivery['review']['checks'][1])
        review=json.loads((first/'review-0.json').read_text());review['review']=delivery['review']
        write(first/'review-0.json',review);write(first/'delivery.json',delivery);self.reseal_delivery(first)
        with self.assertRaisesRegex(ValueError,'Duplicate review check'):
            verifier.inspect_scene(self.root,self.scene_id)

    def test_traversal_symlink_and_bounded_read_fail_closed(self):
        self.child(0,'created')
        reader=verifier.Reader(self.root)
        for name in ('../escape','/absolute','a//b','./a'):
            with self.assertRaises(ValueError):reader.path(self.root,name)
        target=self.root/'target';write(target,b'1234')
        (self.root/'link').symlink_to(target)
        with self.assertRaises(ValueError):reader.pin(self.root/'link')
        with self.assertRaises(ValueError):reader.pin(target,3)
        with patch.dict(verifier.LIMITS,seconds=-1):
            with self.assertRaisesRegex(ValueError,'deadline'):reader.pin(target)

    def test_script_keeps_failure_receipt_and_never_overwrites_output(self):
        self.child(0,'created')
        out=self.root/'outputs/receipt'
        with patch.object(verifier,'ROOT',self.root):
            self.assertEqual(verifier.main(['--scene',self.scene_id,'--out',str(out)]),2)
            self.assertEqual(json.loads((out/'result.json').read_text())['status'],'not_complete')
            with self.assertRaisesRegex(ValueError,'fresh canonical'):
                verifier.main(['--scene',self.scene_id,'--out',str(out)])


if __name__ == '__main__':
    unittest.main()
