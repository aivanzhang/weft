"""Streamed planning receipts are analytic; no real calls, processes or billing."""
import json
import unittest
import test_studio_api_release as existing
from avatar_assets import studio_inspection as inspection


class StreamPlanningCustodyTests(unittest.TestCase):
    def setUp(self):
        self.fixture=existing.StudioApiReleaseTests('runTest')
        self.fixture.fixture_stream=True;self.fixture.fixture_reference=True
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)

    def collect(self):
        f=self.fixture
        return inspection.collect_api_planning(f.root,f.output,f.request,f.jobs.jobs_dir/f.job/'pipeline-request.json')

    def test_settled_stream_planner_releases_as_failed_without_replay(self):
        self.collect()
        self.fixture.test_settled_semantically_invalid_answer_releases_only_as_failed_without_calls_or_reset()

    def test_changed_trace_is_not_billing_or_release_evidence(self):
        f=self.fixture;path=f.output/'agents/planner/transport.json'
        trace=json.loads(path.read_text());trace['response_id']='resp_invented';existing.write(path,trace)
        with self.assertRaises(ValueError):self.collect()

    def test_missing_transport_implementation_pin_rejects(self):
        path=self.fixture.output/'agents/runner.json';settings=json.loads(path.read_text())
        settings['transport_implementation_sha256']='0'*64;existing.write(path,settings)
        with self.assertRaisesRegex(ValueError,'settings'):self.collect()

    def test_stream_completion_cannot_release_an_uncertain_ledger_call(self):
        self.fixture.sql("UPDATE calls SET state='reserved_uncertain',actual=NULL,response_id=NULL,usage=NULL")
        with self.assertRaisesRegex(ValueError,'uncertain'):self.collect()


if __name__=='__main__':unittest.main()
