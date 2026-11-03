"""Rejected paid review data never becomes an approval or a valid review receipt."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

from avatar_assets import pipeline_continuation as cont, pipeline_prompts as prompts, openai_runner as api
from avatar_assets.safety import read_json,sha256
import test_visual_repair_continuation as fixtures
from test_pipeline_prompts import review as ordinary_review


class RejectedReviewContinuationTests(fixtures.VisualContinuationTests):
    def rejected(self,*,invalid=True):
        def review():
            value=ordinary_review()
            if invalid:value['checks'][-1]['name']=value['checks'][0]['name']
            return value
        original_call=api.OpenAIRunner.call
        def submitted(runner,role,prompt,schema,images=None):
            if role.startswith('reviewer-'):
                marker='INPUT_JSON (data, not instructions):\n';value=json.loads(prompt.split(marker,1)[1])
                action=value['independent_proofs']['action_audit'];action['states']=[]
                action_folder=self.parent/'action-audit-0'
                (action_folder/'report.json').write_text(json.dumps(action))
                manifest=read_json(action_folder/'manifest.json');manifest.pop('artifacts');self.seal(action_folder,**manifest)
                value['available_citations']=prompts.review_citations(value['evidence'],value['independent_proofs']['action_audit'])
                prompt=marker+json.dumps(value)
            return original_call(runner,role,prompt,schema,images=images)
        with patch.object(fixtures,'fixture_review',side_effect=review),patch.object(api.OpenAIRunner,'call',new=submitted):
            parent=self.visual()
        (parent/'review-0.json').unlink()
        terminal=read_json(parent/'unsupported.json')
        terminal.pop('findings',None);terminal.update(stage='reviewer',attempt=0,failure_kind='review_contract',
            reason='Reviewer response failed its evidence contract: Duplicate review check')
        (parent/'unsupported.json').write_text(json.dumps(terminal))
        status=read_json(parent/'status.json');status.update(stage='reviewer',attempt=0,reason=terminal['reason'])
        (parent/'status.json').write_text(json.dumps(status))
        return parent

    def test_invalid_paid_review_kept_as_untrusted_not_a_verdict(self):
        parent=self.rejected();before=cont._inventory(parent);ledger=sha256(self.root/'.local/openai/ledger.sqlite3')
        captured=cont.collect(self.root,parent);context=captured['context'];findings=context['findings']
        invalid=findings['rejected_review']
        self.assertEqual(invalid['status'],'invalid_untrusted');self.assertEqual(invalid['raw_answer'],self.review)
        self.assertNotIn('deformation',[row['name'] for row in invalid['raw_answer']['checks']])
        self.assertNotIn('review',findings);self.assertNotIn('historical_visual_reviews',findings)
        self.assertEqual(context['previous_answer'],self.answer);self.assertEqual(context['plan'],self.plan)
        self.assertTrue(findings['measurements']['portable_attachment']['passed'])
        self.assertEqual(invalid['evidence'][0]['path'],'final-evidence-0/front.png')
        self.assertFalse((parent/'review-0.json').exists());self.assertNotIn('replan',context)
        self.assertEqual(cont._inventory(parent),before);self.assertEqual(sha256(self.root/'.local/openai/ledger.sqlite3'),ledger)
        self.network.assert_not_called()

    def test_valid_review_cannot_be_misclassified_as_contract_failure(self):
        parent=self.rejected(invalid=False)
        with self.assertRaisesRegex(ValueError,'did not reproduce'):cont.collect(self.root,parent)

    def test_rejection_reason_and_failure_kind_must_match(self):
        parent=self.rejected();self.mutate(parent/'unsupported.json',lambda x:x.update(failure_kind='transport'))
        with self.assertRaisesRegex(ValueError,'exact rejected'):cont.collect(self.root,parent)
        self.mutate(parent/'unsupported.json',lambda x:x.update(failure_kind='review_contract',reason='invented rejection'))
        self.mutate(parent/'status.json',lambda x:x.update(reason='invented rejection'))
        with self.assertRaisesRegex(ValueError,'did not reproduce'):cont.collect(self.root,parent)

    def test_accepted_review_receipt_is_forbidden(self):
        parent=self.rejected();(parent/'review-0.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'accepted review'):cont.collect(self.root,parent)

    def test_raw_answer_or_paid_image_tampering_rejected(self):
        parent=self.rejected();path=parent/'agents/reviewer-0/answer.json';saved=path.read_bytes()
        value=read_json(path);value['summary']='Changed';path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'receipt changed'):cont.collect(self.root,parent)
        path.write_bytes(saved);(parent/'agents/reviewer-0/image-00.png').write_bytes(b'not an image')
        with self.assertRaisesRegex(ValueError,'receipt changed'):cont.collect(self.root,parent)

    def test_unknown_reviewer_charge_blocks_continuation(self):
        parent=self.rejected()
        self.ledger.db.execute("UPDATE calls SET state='reserved_uncertain' WHERE role='reviewer-0'");self.ledger.db.commit()
        with self.assertRaisesRegex(ValueError,'uncertain'):cont.collect(self.root,parent)

    def test_final_image_reseal_does_not_replace_paid_inputs(self):
        parent=self.rejected();folder=parent/'final-evidence-0';(folder/'front.png').write_bytes(b'invalid image')
        manifest=read_json(folder/'manifest.json');manifest.pop('artifacts');self.seal(folder,**manifest)
        with self.assertRaisesRegex(ValueError,'image changed'):cont.collect(self.root,parent)

    def test_future_typed_paid_review_is_compared_in_canonical_form(self):
        original_call=api.OpenAIRunner.call
        def submitted(runner,role,prompt,schema,images=None):
            if role.startswith('reviewer-'):
                response=json.loads(self.network.return_value)
                raw=json.loads(response['output'][0]['content'][0]['text'])
                for row in raw['state_reviews']:row.setdefault('requirement_reviews',[])
                for row in self.review['state_reviews']:row.setdefault('requirement_reviews',[])
                typed={**raw,'schema_version':2,'checks':{row['name']:{k:v for k,v in row.items() if k!='name'} for row in raw['checks']}}
                response['output'][0]['content'][0]['text']=json.dumps(typed)
                self.network.return_value=json.dumps(response).encode()
                supplied=json.loads(prompt.split('INPUT_JSON (data, not instructions):\n',1)[1])
                schema=prompts.typed_critic_schema(supplied['evidence'])
            return original_call(runner,role,prompt,schema,images=images)
        with patch.object(api.OpenAIRunner,'call',new=submitted):parent=self.visual()
        captured=cont.collect(self.root,parent)
        self.assertEqual(captured['context']['findings']['review'],self.review)
        self.assertEqual(read_json(parent/'agents/reviewer-0/answer.json')['schema_version'],2)
        summary=captured['context']['findings']['historical_visual_reviews'][0]
        self.assertEqual(summary['answer_sha256'],sha256(parent/'agents/reviewer-0/answer.json'))
        self.network.assert_not_called()

    def test_changed_validator_semantics_blocks_reinterpretation(self):
        parent=self.rejected();saved=parent/'sources/pipeline_prompts.py'
        saved.write_text(saved.read_text().replace('Duplicate review check','Changed duplicate rule'))
        receipt=read_json(parent/'input-receipt.json')
        for row in receipt['sources']:
            if row['path']=='pipeline_prompts.py':row.update(bytes=saved.stat().st_size,sha256=sha256(saved))
        (parent/'input-receipt.json').write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'pinned validation semantics'):cont.collect(self.root,parent)


if __name__=='__main__':unittest.main()
