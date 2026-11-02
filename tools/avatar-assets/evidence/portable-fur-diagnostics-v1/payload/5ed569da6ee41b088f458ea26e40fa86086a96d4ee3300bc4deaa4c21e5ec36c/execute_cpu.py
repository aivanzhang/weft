"""One bounded read-only visibility audit; preserve failures without retries."""

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'sources/host'))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, require, sha256, write_json


def main():
    require(not (HERE / 'execution-started.json').exists(), 'Visibility audit already attempted')
    prepared = json.loads((HERE / 'prepared.json').read_text())
    for row in prepared['frozen_files']:
        require(sha256(HERE / row['path']) == row['sha256'], 'Frozen audit changed')
    command = [sys.executable, str(HERE / 'audit.py')]
    write_json(HERE / 'execution-started.json', {'command': command, 'prepared_sha256': sha256(HERE / 'prepared.json')})
    job = run_bounded(command, HERE / 'cpu.log', HERE, timeout=59.738, memory_mb=512, output_mb=32)
    write_json(HERE / 'cpu-job.json', job)
    passed = job['exit_code'] == 0 and not job['failure']
    write_json(HERE / 'manifest.json', {'status': 'visibility_audit_complete' if passed else 'failed_preserved',
        'native_jobs': 0, 'provider_calls': 0, 'quality_approved': False, 'artifacts': artifacts(HERE)})
    print(json.dumps({'completed': passed, 'elapsed_seconds': job['elapsed_seconds'], 'sampled_peak_rss_bytes': job['sampled_peak_rss_bytes']}))
    return 0 if passed else 1


if __name__ == '__main__':
    sys.exit(main())
