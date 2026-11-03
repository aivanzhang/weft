"""Queue/worker integration over custody fixtures; no decode, inference or provider calls."""

from copy import deepcopy
import threading
from types import SimpleNamespace

import pytest

from test_portal_body_detail import case, create, all_bytes, put
from weft import exercise_perception_pipeline as pipeline
from weft import portal_perception_jobs as jobs
from weft.portal import Portal


def test_worker_prepares_detail_before_source_context_and_only_quotes_paid_work(case, monkeypatch):
    before = all_bytes(case['parent'])
    config = create(case); child = case['root']/config['id']
    seen = []
    def preflight(video, out, **options):
        seen.append(options)
        assert video.read_bytes() == (case['parent']/'source.mp4').read_bytes()
        assert options['body_detail_recipe']['source_sha256'] == config['source_sha256']
        assert options['pose_model'].name == 'pose_landmarker_full.task'
        assert len(options['body_detail_inputs']) == 2
        return {'metadata': {'frame_count': 20}, 'manifest_sha256': 'f'*64}
    monkeypatch.setattr(pipeline, 'prepare_exercise_perception_preflight', preflight)
    monkeypatch.setattr(jobs, '_budget_guard', lambda *a: pytest.fail('No paid preparation authorized'))
    assert jobs.execute_perception_job(child, project=case['project']) == 0
    assert len(seen) == 1 and jobs.read_status(child)['status'] == 'awaiting_consent'
    assert not (child/'perception/request.json').exists()
    assert all_bytes(case['parent']) == before
    assert jobs.execute_perception_job(child, project=case['project']) == 0
    assert len(seen) == 1
    summary = jobs.summary(child, config, {'status':'needs_input','route':'exercise'})
    assert summary['artifacts']['body_detail_recipe'] == 'body-detail/recipe.json'


def test_worker_detail_refusal_preserves_parent_and_never_runs_pose_acquisition(case, monkeypatch):
    config = create(case); child = case['root']/config['id']
    put(case['parent'], 'perception/failure.json', {'status':'failed','error':'changed after queue'})
    before = all_bytes(case['parent'])
    monkeypatch.setattr(pipeline, 'prepare_exercise_perception_preflight', lambda *a, **k: pytest.fail('No inference on changed source'))
    assert jobs.execute_perception_job(child, project=case['project']) == 1
    assert jobs.read_status(child)['status'] == 'failed'
    assert (child/'perception/preflight-start.json').is_file()
    assert not (child/'perception/request.json').exists()
    assert all_bytes(case['parent']) == before


def test_portal_registers_one_child_in_existing_queue_and_indexes_lineage(case):
    p = SimpleNamespace(root=case['root'], lease=case['lease'], jobs=case['jobs'], lock=threading.RLock(),
        active=None, body_detail_children={}, wake=threading.Event(), _stored_bytes=lambda:100000,
        _reserved_growth=lambda:0, _pending_jobs=lambda:0)
    p._folder=lambda identity:p.root/identity
    p.summary=lambda identity:deepcopy(p.jobs[identity])
    payload={'recipe':'full-dense-region-v1','request_id':'b'*32}
    first=Portal.prepare_body_detail(p,case['parent'].name,payload)
    assert p.wake.is_set() and p.body_detail_children[case['parent'].name] == first['id']
    second=Portal.prepare_body_detail(p,case['parent'].name,payload)
    assert first == second and len(p.jobs) == 2
    p.active=case['parent'].name
    with pytest.raises(ValueError,match='worker is finishing'):
        Portal.prepare_body_detail(p,case['parent'].name,payload)


def test_detail_downloads_expose_only_small_request_and_recipe(case):
    config=create(case);child=case['root']/config['id']
    p=SimpleNamespace(_folder=lambda _:child)
    assert Portal.artifact(p,config['id'],'body-detail/request.json') == child/'body-detail/request.json'
    put(child,'body-detail/recipe.json',b' '*(150 << 10))
    assert Portal.artifact(p,config['id'],'body-detail/recipe.json') == child/'body-detail/recipe.json'
    for name in ('body-detail/parent/job.json','body-detail/inputs/pose-landmarks.json','body-detail/../job.json'):
        with pytest.raises(ValueError,match='public body-detail'):
            Portal.artifact(p,config['id'],name)
