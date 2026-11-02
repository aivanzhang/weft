from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, sha256, write_json

paths = ['avatar_assets/__init__.py', 'avatar_assets/studio_jobs.py', 'avatar_assets/studio_worker.py',
         'avatar_assets/studio.py', 'avatar_assets/gallery.py', 'avatar_assets/meshy.py',
         'avatar_assets/meshy_access.py', 'avatar_assets/safety.py', 'avatar_assets/runner.py',
         'tests/__init__.py', 'tests/test_studio_jobs.py', 'requirements.lock']
assert sha256(ROOT/'avatar_assets/studio_worker.py') == 'b09cdea70a4c285bd69c2070b035f2199256b315fe7a6214fa58ccff35a9298e'
assert sha256(ROOT/'tests/test_studio_jobs.py') == 'd35199634cafd17bbbf9fa995e0ccd88449659714268daf961fb6ba6ddd1753f'
pins = []
for name in paths:
    source, target = ROOT/name, OUT/'sources'/name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    pins.append({'path': name, 'sha256': sha256(source), 'bytes': source.stat().st_size,
                 'snapshot': str(target.relative_to(OUT))})
(OUT/'tmp').mkdir()
command = ['/usr/bin/env', 'PYTHONDONTWRITEBYTECODE=1', 'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1',
           'TMPDIR='+str(OUT/'tmp'), sys.executable, '-B', str(OUT/'run_tests.py')]
write_json(OUT/'prepared.json', {'schema_version': 1,
    'recorded_local': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
    'command': command, 'equivalent_test_command': '.venv/bin/python -B -m unittest -v tests.test_studio_jobs',
    'expected_unique_methods': 29, 'inputs': pins,
    'scripts': [{'path': name, 'sha256': sha256(OUT/name)} for name in ('execute.py', 'run_tests.py')],
    'limits': {'attempts': 1, 'seconds': 30, 'memory_mb': 512, 'output_mb': 16},
    'change': 'New Studio pipeline requests pin five agent calls, with one repair and240-second per-call deadline unchanged.',
    'compatibility': 'Completed ten-call fixture remains readable without replay; unfinished ten-call request is rejected, never rewritten.'})
job = run_bounded(command, OUT/'tests.log', OUT, timeout=30, memory_mb=512, output_mb=16)
write_json(OUT/'job.json', job)
results = json.loads((OUT/'test-result.json').read_text()) if (OUT/'test-result.json').is_file() else {}
unchanged = [{'path': pin['path'], 'unchanged': sha256(ROOT/pin['path']) == pin['sha256']} for pin in pins]
passed = (results.get('passed') is True and results.get('tests_run') == 29
          and job.get('process_exit_verified') and job.get('exit_code') == 0 and not job.get('failure')
          and all(row['unchanged'] for row in unchanged))
write_json(OUT/'receipt.json', {'schema_version': 1, 'status': 'passed' if passed else 'failed',
    'test_result': results, 'job': job, 'source_unchanged': unchanged,
    'new_agent_calls': 0, 'new_provider_calls': 0, 'new_native_calls': 0,
    'new_server_or_browser_calls': 0, 'canonical_registry_or_ledger_mutated': False,
    'scope': 'Software cap/custody compatibility only, not a completed actual Studio job.'})
write_json(OUT/'manifest.json', {'schema_version': 1, 'status': 'passed' if passed else 'failed',
    'kind': 'studio_five_call_cap_tests', 'artifacts': artifacts(OUT)})
print(json.dumps({'status': 'passed' if passed else 'failed', 'tests': results.get('tests_run'),
                  'elapsed_seconds': job.get('elapsed_seconds'),
                  'sampled_peak_rss_bytes': job.get('sampled_peak_rss_bytes'),
                  'pid': job.get('pid'), 'process_exit_verified': job.get('process_exit_verified'),
                  'manifest_sha256': sha256(OUT/'manifest.json')}))
raise SystemExit(0 if passed else 1)
