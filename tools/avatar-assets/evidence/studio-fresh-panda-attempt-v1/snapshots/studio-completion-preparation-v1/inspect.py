"""Read fixed local metadata without constructing a Studio or provider client."""

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import stat
import time
from zoneinfo import ZoneInfo


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
PROJECT = ROOT.parents[1]
START = time.monotonic()
signal.alarm(20)


def digest(path, limit=2*1024**2):
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit
    data = path.read_bytes()
    assert len(data) == info.st_size
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def write(name, value):
    with (OUT/name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def immutable_db(path):
    # Refuse journals rather than taking an incomplete immutable snapshot.
    assert not any(Path(str(path)+suffix).exists() for suffix in ('-wal', '-journal', '-shm'))
    pin = digest(path, 4*1024**2)
    info = path.stat()
    assert info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) == 0o600
    db = sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1', uri=True, timeout=.2)
    db.row_factory = sqlite3.Row
    return db, pin


sources = ['avatar_assets/studio_jobs.py', 'avatar_assets/studio_worker.py',
           'avatar_assets/studio.py', 'avatar_assets/pipeline.py',
           'avatar_assets/agent_runner.py', 'avatar_assets/meshy.py',
           'avatar_assets/meshy_access.py', 'avatar_assets/quality_policy.py',
           'avatar_assets/pipeline_prompts.py', 'docs/fresh-avatar-acceptance-plan.md',
           '.local/studio-registry.json']
pins = [digest(ROOT/name) for name in sources]
registry = json.loads((ROOT/'.local/studio-registry.json').read_text())
assert registry == {'schema_version': 1, 'state_dir': str(ROOT/'.local/studio')}
database = ROOT/'.local/studio/jobs.sqlite3'
db, studio_pin = immutable_db(database)
try:
    assert db.execute('SELECT count(*) FROM jobs').fetchone()[0] <= 256
    rows = db.execute('SELECT id,status,output,created,data FROM jobs ORDER BY created,id').fetchall()
    jobs = []
    for row in rows:
        data = json.loads(row['data'])
        jobs.append({key: row[key] for key in ('id', 'status', 'output')} | {
            'operation': data.get('operation'), 'invocations': data.get('invocations'),
            'retained_bytes_cached': data.get('retained_bytes', 0),
            'output_bytes_cached': data.get('output_bytes', 0),
            'ended': data.get('ended'), 'owner_pid_recorded': data.get('owner_pid'),
            'provenance': data.get('provenance')})
    active = [row[0] for row in db.execute('SELECT job FROM active ORDER BY id')]
finally:
    db.close()
assert digest(database, 4*1024**2) == studio_pin

ledger_path = PROJECT/'.local/meshy/ledger.sqlite3'
db, ledger_pin = immutable_db(ledger_path)
try:
    assert db.execute('SELECT count(*) FROM calls').fetchone()[0] <= 10000
    base = db.execute('SELECT credit_limit FROM budget WHERE id=1').fetchone()[0]
    by_job = dict(db.execute('SELECT job,SUM(credits) FROM calls GROUP BY job'))
    grants = [dict(row) for row in db.execute('SELECT job,credits,plan_hash FROM budget_grants ORDER BY job')]
    states = [dict(row) for row in db.execute('SELECT state,COUNT(*) AS calls,SUM(credits) AS credits FROM calls GROUP BY state')]
finally:
    db.close()
assert digest(ledger_path, 4*1024**2) == ledger_pin
assert all(digest(Path(pin['path'])) == pin for pin in pins)
grant_map = {row['job']: row['credits'] for row in grants}
base_used = sum(max(0, used-grant_map.get(job, 0)) for job, used in by_job.items())
retained = sum(row['retained_bytes_cached'] for row in jobs)
facts = {
    'schema_version': 1, 'recorded_local': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
    'scope': 'Read-only immutable local metadata; no StudioJobs/Ledger constructor, no pipeline or provider call.',
    'sources': pins, 'database_pins': [studio_pin, ledger_pin],
    'canonical_registry': registry,
    'studio': {'jobs': jobs, 'active_lease_jobs': active,
               'queued_jobs': [row['id'] for row in jobs if row['status'] == 'queued'],
               'cached_retained_bytes': retained, 'storage_limit_bytes': 8*1024**3,
               'new_active_reservation_bytes': 2*1024**3+64*1024**2,
               'cached_capacity_fits_one_new_job': retained+2*1024**3+64*1024**2 <= 8*1024**3,
               'scope': 'Saved registry state and cached storage, not an OS process or fresh filesystem census.'},
    'meshy': {'base_allowance': base, 'job_grants': grants,
              'total_authorized': base+sum(grant_map.values()),
              'used_or_reserved': sum(by_job.values()), 'base_used': base_used,
              'remaining_base': max(0, base-base_used), 'states': states,
              'new_text_job_estimate': 35, 'new_paid_submissions': 2,
              'required_new_job_grant': 35,
              'no_balance_key_account_or_network_access': True},
    'historical_ai_study': {'consumed_calls': 13, 'remaining_calls': 0,
                            'input_tokens_recorded': 358211, 'output_tokens_recorded': 13885,
                            'authority': 'Parent task plus TODO.md rejected-trial handoff; not a new billing/API query.'},
    'proposed_call_flow': {'expected': ['planner', 'author-0', 'reviewer-0'],
                           'maximum': ['planner', 'author-0', 'reviewer-0', 'author-1', 'reviewer-1'],
                           'max_repairs': 1, 'current_studio_persisted_max_agent_calls': 10,
                           'requested_new_hard_cap': 5, 'automatic_retries': 0},
    'elapsed_seconds': time.monotonic()-START,
    'mutations_outside_this_folder': False, 'job_created': False,
    'provider_calls': 0, 'agent_calls': 0, 'native_calls': 0, 'server_browser_calls': 0,
}
write('read-only-facts.json', facts)
print(json.dumps({'status': 'read_only_preparation', 'studio_jobs': len(jobs),
                  'active_lease_jobs': active, 'queued_jobs': facts['studio']['queued_jobs'],
                  'meshy_remaining_base': facts['meshy']['remaining_base'],
                  'new_job_grant_needed': 35, 'configured_agent_ceiling': 10,
                  'flow_max_agent_calls': 5, 'elapsed_seconds': facts['elapsed_seconds']}))
