"""Rejected review images stay useful without promoting malformed votes to approval."""
from copy import deepcopy
from pathlib import Path
import json
import unittest

from avatar_assets import repair_visual_inputs as visual
from avatar_assets.safety import artifacts, read_json, sha256
from test_repair_visual_inputs import RepairVisualInputsTests, SOURCE


def write_json(path,value):
    # Tampering is confined to disposable fixtures; production snapshots are write-once.
    Path(path).write_text(json.dumps(value,sort_keys=True)+'\n')


class RejectedReviewImagesTests(unittest.TestCase):
    def setUp(self):
        RepairVisualInputsTests.setUp(self)
        self.invalid={'verdict':'accept','checks':[
            {'name':'appearance','passed':True,'evidence_ids':['invented']},
            {'name':'appearance','passed':False,'evidence_ids':[]}]}
        role=self.parent/'agents/reviewer-0'
        write_json(role/'answer.json',self.invalid)
        write_json(role/'receipt.json',{'status':'succeeded','answer_sha256':sha256(role/'answer.json')})
        reason='Reviewer response failed its evidence contract: Duplicate review check'
        write_json(self.parent/'unsupported.json',{'stage':'reviewer','failure_kind':'review_contract',
            'attempt':0,'reason':reason})
        self.rejected={k:v for k,v in self.summary.items() if k!='review_reference'}
        self.rejected.update(status='invalid_untrusted',role='reviewer-0',source_sha256=SOURCE,
            raw_answer=self.invalid,reason=reason,answer_sha256=sha256(role/'answer.json'),
            receipt_sha256=sha256(role/'receipt.json'))
        self.context['findings']={'rejected_review':self.rejected,'repair_images':['/untrusted/freeform.png']}
        self.reseal()

    def reseal(self):
        write_json(self.child/'continuation/context.json',self.context)
        p=self.child/'continuation/receipt.json';v=read_json(p)
        v.update(inventory=artifacts(self.parent),context_sha256=sha256(self.child/'continuation/context.json'))
        write_json(p,v);self.request['continuation']['sha256']=sha256(p)

    def selected(self,base=None):
        return visual.from_continuation(self.root,self.child,self.request,self.context,base or self.guides)

    def test_invalid_votes_never_choose_evidence_or_become_approval(self):
        result=self.selected()
        self.assertEqual(result['review_status'],'invalid_untrusted')
        self.assertEqual([r['id'] for r in result['images']][:3],
            ['evidence-exported-views','evidence-native-views','evidence-motion-00'])
        self.assertNotIn('verdict',result)
        self.assertNotIn('invented',str(result));self.assertNotIn('/untrusted/freeform.png',str(result))
        self.assertEqual(self.invalid['verdict'],'accept')

    def test_reference_keeps_motion_within_eight_image_limit(self):
        reference=Path(self.evidence[2]['path'])
        result=self.selected([*self.guides,reference])
        self.assertEqual(len(result['images']),4);self.assertEqual(result['total_agent_images'],8)
        self.assertIn('evidence-motion-00',[row['id'] for row in result['images']])

    def test_image_changed_after_binding_rejected(self):
        Path(self.evidence[0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'changed artifact'):self.selected()

    def test_paid_image_inventory_mismatch_rejected(self):
        write_json(self.parent/'agents/reviewer-0/request.json',{'images':[]});self.reseal()
        with self.assertRaisesRegex(ValueError,'paid images differ'):self.selected()

    def test_unknown_call_receipt_cannot_supply_images(self):
        p=self.parent/'agents/reviewer-0/receipt.json';v=read_json(p);v['status']='unknown';write_json(p,v)
        self.rejected['receipt_sha256']=sha256(p);self.reseal()
        with self.assertRaisesRegex(ValueError,'answer/source/terminal differs'):self.selected()

    def test_rejected_status_and_source_cannot_be_relabelled(self):
        for field,value in (('status','accepted'),('source_sha256','b'*64),('raw_answer',{}),('attempt',True)):
            with self.subTest(field=field):
                original=self.rejected[field];self.rejected[field]=value;self.reseal()
                with self.assertRaises(ValueError):self.selected()
                self.rejected[field]=original;self.reseal()

    def test_terminal_error_and_relative_image_inventory_are_bound(self):
        for field,value in (('reason','new reason'),('evidence',[]),('parent_output','../outside')):
            with self.subTest(field=field):
                original=deepcopy(self.rejected[field]);self.rejected[field]=value;self.reseal()
                with self.assertRaises(ValueError):self.selected()
                self.rejected[field]=original;self.reseal()

    def test_valid_typed_paid_answer_matches_unchanged_canonical_receipt(self):
        from avatar_assets.pipeline_prompts import decode_typed_review
        raw={'schema_version':2,'verdict':'repair','summary':'Analytic appearance failure',
            'checks':{name:{'passed':False,'reason':'Analytic fixture','evidence_ids':['evidence-exported-views']}
                      for name in ('identity','appearance','framing','deformation')},
            'state_reviews':[],'required_repairs':['Recheck appearance'],'limitations':[]}
        canonical=decode_typed_review(raw);original=deepcopy(raw)
        role=self.parent/'agents/reviewer-0';write_json(role/'answer.json',raw)
        write_json(role/'receipt.json',{'status':'succeeded','answer_sha256':sha256(role/'answer.json')})
        write_json(self.parent/'review-0.json',{'model_manifest_sha256':self.summary['model_manifest_sha256'],
            'review':canonical,'evidence':self.evidence})
        result=visual.from_current_review(self.parent,SOURCE,0,canonical,self.evidence,self.guides)
        self.assertEqual(result['status'],'available');self.assertEqual(raw,original)
        self.assertNotIn('review_status',result)


if __name__=='__main__':unittest.main()
