"""Analytic reference-image planner custody; no real keys, API, process or native calls."""
import json
import unittest
from unittest.mock import patch
import test_studio_api_release as existing
from avatar_assets import studio_inspection as inspection


class ReferencePlanningCustodyTests(unittest.TestCase):
    def setUp(self):
        self.fixture=existing.StudioApiReleaseTests('runTest')
        self.fixture.fixture_reference=True
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)

    def collect(self,profile='planning'):
        f=self.fixture
        return inspection.collect_api_planning(f.root,f.output,f.request,
            f.jobs.jobs_dir/f.job/'pipeline-request.json',profile=profile)

    def test_reference_transport_is_bound_and_analytic_release_preserves_all_bytes_and_cost(self):
        f=self.fixture;proof=self.collect()
        ref=proof['reference']
        self.assertEqual(ref['submitted_image']['sha256'],ref['prepared_reference']['sha256'])
        self.assertEqual(ref['submitted_image']['sha256'],ref['original_reference']['sha256'])
        self.assertEqual(proof['new_provider_calls'],0)
        f.test_settled_semantically_invalid_answer_releases_only_as_failed_without_calls_or_reset()

    def test_original_reference_identity_and_copied_bytes_are_both_required(self):
        f=self.fixture;(f.output/'inputs/reference.png').write_bytes(b'changed copy')
        with self.assertRaisesRegex(ValueError,'reference bytes'):self.collect()

    def test_original_reference_replacement_blocks_release(self):
        f=self.fixture;f.reference.write_bytes(b'changed original')
        with self.assertRaises(ValueError):self.collect()

    def test_submitted_image_must_equal_selected_reference_even_if_artifacts_resealed(self):
        from avatar_assets.agent_runner import _call_artifacts
        f=self.fixture;folder=f.output/'agents/planner'
        (folder/'image-00.png').write_bytes(b'changed API image')
        receipt=json.loads((folder/'receipt.json').read_text());receipt['artifacts']=_call_artifacts(folder)
        existing.write(folder/'receipt.json',receipt)
        with self.assertRaisesRegex(ValueError,'planner image differs'):self.collect()

    def test_reference_payload_hash_must_reconstruct_exactly(self):
        from avatar_assets.agent_runner import _call_artifacts
        f=self.fixture;folder=f.output/'agents/planner'
        payload=json.loads((folder/'payload-metadata.json').read_text());payload['sha256']='0'*64
        existing.write(folder/'payload-metadata.json',payload)
        receipt=json.loads((folder/'receipt.json').read_text());receipt['artifacts']=_call_artifacts(folder)
        existing.write(folder/'receipt.json',receipt)
        with self.assertRaisesRegex(ValueError,'payload metadata differs'):self.collect()

    def test_reference_does_not_expand_author_input_recovery_or_extra_image_scope(self):
        with self.assertRaisesRegex(ValueError,'only completed planning'):self.collect('author_input_bound')
        (self.fixture.output/'agents/planner/image-01.png').write_bytes(b'extra')
        with self.assertRaises(ValueError):self.collect()


if __name__=='__main__':unittest.main()
