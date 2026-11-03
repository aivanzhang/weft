"""Analytic motion cues are not measured animation or quality acceptance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from avatar_assets.pipeline_evidence import motion_samples,_sampling_inputs,decoded_poses
from avatar_assets.safety import artifacts,sha256,write_json
from test_pipeline_evidence import FullEvidenceTests


def clip():
    return {'source_clip_seconds':4.,'inspection':{'frames':96,'fps':24.,'duration_seconds':4.}}


def context(*,vertical=True):
    keys=[{'second':t,'rotations':{},'translations_m':{'root':v}} for t,v in
          [(0,[0,0,0]),(.5,[-.006,0,0]),(1,[-.015,0,.014]),(1.5,[0,0,0]),
           (2,[.08,0,0]),(2.5,[0,0,0]),(3,[-.015,0,.014]),(3.5,[0,0,0]),(4,[0,0,0])]]
    if not vertical:
        keys=[{'second':0,'rotations':{}},{'second':1,'rotations':{'hinge':[0,60,0]}},
              {'second':2,'rotations':{}},{'second':4,'rotations':{}}]
    state={'id':'act','role_details':{'subject':{'root_bone':'root','bones':['root','hinge']}},
           'evidence':{'roles':{'subject':{'max_vertical_rise_m':.014 if vertical else 0.,'max_travel_m':.08 if vertical else 0.,'scale_m':.38}}}}
    return {'config':{'authored_actions':[{'id':'act','interpolation':'LINEAR','keys':keys}]},
            'audit':{'states':[state]},'intent':{'states':[{'id':'act','requirements':[{'role':'subject','kind':'travel' if vertical else 'articulation'}]}]},
            'source_hashes':{'analytic_fixture':'not actual source evidence'}}


class MotionSamplingTests(unittest.TestCase):
    def test_vertical_peak_survives_larger_out_of_phase_horizontal_range(self):
        result=motion_samples('act',clip(),context())
        self.assertEqual(result['indices'],[0,24,36,95])
        self.assertEqual(result['primary_extremum']['second'],1)
        self.assertEqual(result['primary_extremum']['magnitude'],.014)
        self.assertEqual(result['omitted_candidates'][0]['second'],2)
        self.assertFalse(result['semantic_approval']);self.assertIn('unverified',result['foot_clearance'])
        self.assertIn('not a measured peak timestamp',result['observations'][0]['measurement_space'])
        self.assertEqual(result['frames'][1]['source_second'],1)

    def test_hinged_object_uses_rotation_cue_without_fabricating_vertical_motion(self):
        result=motion_samples('act',clip(),context(vertical=False))
        self.assertEqual(result['indices'],[0,24,48,95])
        self.assertEqual(result['primary_extremum']['control'],'authored_euler_control_excursion')
        self.assertEqual(result['observations'][0]['measured_role_root_vertical_rise_m'],0)
        self.assertFalse(result['semantic_approval'])

    def test_retiming_records_source_and_delivered_timestamps(self):
        value=clip();value['inspection'].update(frames=48,duration_seconds=2.)
        result=motion_samples('act',value,context())
        self.assertEqual(result['indices'],[0,12,18,47])
        self.assertEqual(result['frames'][1]['video_second'],.5)
        self.assertEqual(result['frames'][1]['source_second'],1)

    def test_nonzero_baseline_missing_controls_and_no_relevant_motion_are_explicit(self):
        ctx=context()
        for key in ctx['config']['authored_actions'][0]['keys']:
            key['translations_m']['root'][2]+=.2
        self.assertAlmostEqual(motion_samples('act',clip(),ctx)['primary_extremum']['magnitude'],.014)
        ctx=context();ctx['intent']['states'][0]['requirements']=[]
        result=motion_samples('act',clip(),ctx)
        self.assertEqual(result['indices'],[0,32,63,95]);self.assertEqual(result['selection'],'uniform_fallback')
        self.assertIn('not action proof',result['reason'])
        self.assertEqual(motion_samples('act',clip(),None)['selection'],'uniform_fallback')

    def test_missing_or_nonlinear_controls_use_honest_uniform_fallback(self):
        for actions in (None,[],[{**context()['config']['authored_actions'][0],'interpolation':'BEZIER'}]):
            ctx=context();ctx['config']['authored_actions']=actions
            result=motion_samples('act',clip(),ctx)
            self.assertEqual(result['selection'],'uniform_fallback');self.assertFalse(result['semantic_approval'])
            self.assertIn('source_hashes',result)

    def test_invalid_cues_and_timing_fail_closed(self):
        for mutate in [lambda c:c['config']['authored_actions'][0]['keys'][1].update(second=float('nan')),
                       lambda c:c['audit']['states'][0]['evidence']['roles']['subject'].update(max_vertical_rise_m=float('inf'))]:
            ctx=context();mutate(ctx)
            with self.assertRaises(ValueError):motion_samples('act',clip(),ctx)
        value=clip();value['source_clip_seconds']=8
        with self.assertRaisesRegex(ValueError,'timing differs'):motion_samples('act',value,context())

    def test_decode_uses_selected_real_frames_and_preserves_small_bounds(self):
        fixture=FullEvidenceTests('runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        rows=decoded_poses(fixture.media/'dance.mp4',fixture.facts,[0,2])
        self.assertEqual([row[0] for row in rows],[0,2]);self.assertEqual(rows[0][1].size,(128,128))
        for invalid in ([0,0],[0,1,2,3,4],[-1],[True]):
            with self.assertRaisesRegex(ValueError,'indices'):decoded_poses(fixture.media/'dance.mp4',fixture.facts,invalid)

    def test_motion_cues_reach_review_image_metadata_without_changing_native_poses(self):
        from avatar_assets.pipeline_evidence import build_evidence
        fixture=FullEvidenceTests('runTest');fixture.setUp();self.addCleanup(fixture.doCleanups)
        ctx=context()
        for section in ('config','audit','intent'):
            rows=ctx[section]['authored_actions' if section=='config' else 'states'];rows[0]['id']='dance'
        before=sha256(fixture.pack/'renders/dance/0001.png')
        with patch('avatar_assets.pipeline_evidence._sampling_inputs',return_value=ctx):
            report=build_evidence(fixture.pack,fixture.exported,fixture.media,fixture.root/'new-evidence')
        motion=next(row for row in report['images'] if row['id']=='evidence-motion-00')
        self.assertEqual(motion['motion_sampling'],report['motion_sampling'])
        self.assertEqual(motion['motion_sampling'][0]['observations'][0]['authored_root_vertical_rise_m'],.014)
        self.assertEqual(sha256(fixture.pack/'renders/dance/0001.png'),before)
        self.assertEqual(len(report['images']),6)


class SamplingCustodyTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name).resolve()
        self.pack=self.root/'model-0';(self.pack/'sources').mkdir(parents=True)
        (self.pack/'model.glb').write_bytes(b'ANALYTIC NONMODEL')
        write_json(self.pack/'sources/finish-design.json',{'source_sha256':'a'*64})
        write_json(self.pack/'manifest.json',{'status':'needs_visual_review','source_sha256':'a'*64,'artifacts':artifacts(self.pack)})
        audit=self.root/'action-audit-0';audit.mkdir();self.audit=audit
        model=sha256(self.pack/'model.glb')
        write_json(audit/'intent.json',{'source_sha256':model,'states':[]})
        write_json(audit/'report.json',{'passed':True,'source_unchanged':True,'model_sha256':model,
            'intent_sha256':sha256(audit/'intent.json'),'states':[]})
        write_json(audit/'job.json',{'process_exit_verified':True,'manual_stop_required':False,'exit_code':0})
        write_json(audit/'manifest.json',{'model_sha256':model,'artifacts':artifacts(audit)})

    def load(self):return _sampling_inputs(self.pack,lambda path:path)

    def test_valid_receipt_and_source_hashes(self):
        result=self.load();self.assertEqual(result['source_hashes']['model_glb'],sha256(self.pack/'model.glb'))

    def test_tampered_config_model_or_audit_rejected(self):
        for path in [self.pack/'sources/finish-design.json',self.pack/'model.glb',self.audit/'report.json']:
            saved=path.read_bytes();path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'artifact changed'):self.load()
            path.write_bytes(saved)

    def test_wrong_model_even_resealed_audit_rejected(self):
        path=self.audit/'report.json';value=json.loads(path.read_text());value['model_sha256']='f'*64
        path.write_text(json.dumps(value));manifest=json.loads((self.audit/'manifest.json').read_text())
        manifest['artifacts']=artifacts(self.audit);(self.audit/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'custody differs'):self.load()

    def test_missing_source_binding_cannot_be_resealed_as_valid(self):
        path=self.pack/'sources/finish-design.json';path.write_text('{}')
        manifest=json.loads((self.pack/'manifest.json').read_text());manifest.pop('source_sha256')
        manifest['artifacts']=artifacts(self.pack);(self.pack/'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'model/source differs'):self.load()

    def test_escaped_or_linked_artifact_rejected(self):
        path=self.pack/'manifest.json';saved=path.read_text();value=json.loads(saved)
        value['artifacts'][0]['path']='../outside';path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'missing|invalid'):self.load()
        path.write_text(saved);source=self.pack/'sources/finish-design.json';source.unlink();source.symlink_to(self.audit/'intent.json')
        with self.assertRaises(ValueError):self.load()


if __name__=='__main__':unittest.main()
