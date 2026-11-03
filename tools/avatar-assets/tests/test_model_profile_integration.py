"""Analytic model-selection checks; no paid calls or generated assets."""
import hashlib
import unittest
import test_openai_studio_integration as studio_fixture
import test_studio_api_release as release_fixture
from avatar_assets import studio_inspection
from avatar_assets.agent_backend import readiness
from avatar_assets.openai_runner import reserve_microdollars

MODEL = 'gpt-5.2-2025-12-11'

class ModelProfileIntegrationTests(unittest.TestCase):
    def test_new_pipeline_estimate_and_readiness_use_selected_model(self):
        f=studio_fixture.OpenAIStudioIntegrationTests('runTest'); self.addCleanup(f.doCleanups); f.setUp()
        f.settings.update(model=MODEL,transport='sse-v1',timeout=600)
        f.config(); f.key(); result=f.prepare(); f.grant()
        self.assertEqual(result['estimate']['reservation_microdollars_per_call'],929376)
        ready=readiness(f.root,job_id=f.out.name,job_digest=hashlib.sha256((f.out/'request.json').read_bytes()).hexdigest())
        self.assertTrue(ready['can_run'],ready)
        self.assertEqual(ready['reservation_microdollars_per_call'],reserve_microdollars(16384,model=MODEL))

    def test_new_model_streamed_planner_custody(self):
        f=release_fixture.StudioApiReleaseTests('runTest'); f.fixture_model=MODEL; f.fixture_stream=True; f.fixture_reference=True
        self.addCleanup(f.doCleanups); f.setUp()
        proof=studio_inspection.collect_api_planning(f.root,f.output,f.request,f.jobs.jobs_dir/f.job/'pipeline-request.json')
        self.assertIsInstance(proof,dict)

if __name__=='__main__':unittest.main()
