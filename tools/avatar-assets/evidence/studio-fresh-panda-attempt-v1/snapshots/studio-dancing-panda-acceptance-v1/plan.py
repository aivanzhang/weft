"""Launch exactly the approved canonical planner; never submit geometry here."""
from pathlib import Path
import sqlite3
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.safety import read_json, sha256, write_json
from avatar_assets.studio_jobs import StudioJobs
from avatar_assets.studio_worker import PIPELINE_LIMITS

assert sys.argv[1:] == ['--execute']
binding = read_json(HERE / 'execution-binding.json')
approval = read_json(HERE / 'approval.json')
assert sha256(HERE / 'approval.json') == binding['approval_sha256']
assert approval['decision'] == 'approved' and approval['maximum_AI_role_calls'] == 5
assert sha256(ROOT / 'avatar_assets/studio_worker.py') == binding['worker_sha256']
assert PIPELINE_LIMITS == binding['hard_limits']
assert not (HERE / 'plan-launch-intent.json').exists()
state = ROOT / '.local/studio'
with sqlite3.connect((state / 'jobs.sqlite3').as_uri()+'?mode=ro', uri=True, timeout=1) as db:
    assert db.execute('SELECT COUNT(*) FROM active').fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0] == 0
    assert db.execute('SELECT status FROM jobs WHERE id=?', (binding['job_id'],)).fetchone()[0] == 'created'
jobs = StudioJobs(ROOT)
assert read_json(jobs.jobs_dir / binding['job_id'] / 'request.json') == read_json(HERE / 'request.json')
write_json(HERE / 'plan-launch-intent.json', {'schema_version': 1, 'job_id': binding['job_id'],
           'approval_sha256': binding['approval_sha256'], 'maximum_total_AI_role_calls': 5,
           'operation': 'plan', 'new_meshy_calls': 0, 'script_sha256': sha256(HERE / 'plan.py')})
result = jobs.plan(binding['job_id'])
write_json(HERE / 'plan-start.json', result)
print({'job_id': result['id'], 'status': result['status'], 'operation': 'plan'})
