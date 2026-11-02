"""Execute the pinned data-only candidate once under fixed resource limits."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources'))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import require, sha256, write_json


def main():
    prepared = json.loads((HERE/'prepared.json').read_text())
    for row in prepared['inputs']:
        require(sha256(ROOT/row['path']) == row['sha256'], 'Prepared input changed: '+row['path'])
    config = json.loads((HERE/'candidate.json').read_text())
    out = ROOT/config['output']
    require(not out.exists() and not (HERE/'execution-started.json').exists(), 'Candidate was already attempted')
    out.mkdir()
    with (HERE/'execution-started.json').open('x') as stream:
        json.dump({'prepared_sha256': sha256(HERE/'prepared.json'), 'attempts': 1, 'generation_calls': 0}, stream)
    job = run_bounded([str(ROOT/'.venv/bin/python'), '-B', str(HERE/'build_candidate.py')],
                      HERE/'cpu.log', out, timeout=30, memory_mb=512, output_mb=96)
    write_json(HERE/'cpu-job.json', job)
    unchanged = all(sha256(ROOT/row['path']) == row['sha256'] for row in prepared['inputs'])
    artifacts = [{'path': str(path.relative_to(out)), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
                 for path in sorted(out.rglob('*')) if path.is_file()]
    status = 'candidate_pending_visual_review' if job['exit_code'] == 0 and not job['failure'] and unchanged else 'failed_preserved'
    write_json(out/'manifest.json', {'schema_version': 1, 'status': status, 'quality_approved': False,
        'source': config['source'], 'source_sha256': config['source_sha256'],
        'prepared_path': str(HERE.relative_to(ROOT)/'prepared.json'), 'prepared_sha256': sha256(HERE/'prepared.json'),
        'inputs_unchanged': unchanged, 'job': job, 'artifacts': artifacts,
        'new_generation_calls': 0, 'native_render_calls': 0, 'automatic_retry': False})
    print(json.dumps({'status': status, **{key: job.get(key) for key in ('exit_code', 'failure', 'elapsed_seconds',
          'sampled_peak_rss_bytes', 'final_output_bytes', 'process_exit_verified')}}))
    if status == 'failed_preserved':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
