"""One bounded independent CPU audit of read-only exported artifacts."""

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources'))
from avatar_assets.runner import run_bounded


def main():
    prepared = json.loads((HERE/'prepared.json').read_text())
    for row in prepared['sources']:
        if hashlib.sha256((HERE/row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Prepared audit source changed')
    with (HERE/'execution-started.json').open('x') as stream:
        json.dump({'prepared_sha256': hashlib.sha256((HERE/'prepared.json').read_bytes()).hexdigest()}, stream)
        stream.write('\n')
    job = run_bounded([str(ROOT/'.venv/bin/python'), '-B', str(HERE/'audit.py')],
                      HERE/'cpu.log', HERE, timeout=15, memory_mb=512, output_mb=16)
    for row in prepared['sources']:
        if hashlib.sha256((HERE/row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Audit source changed during execution')
    for row in prepared['inputs']:
        with (ROOT/row['path']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                raise ValueError('Read-only audit input changed during execution')
    with (HERE/'job.json').open('x') as stream:
        json.dump(job, stream, indent=2, sort_keys=True);stream.write('\n')
    print(json.dumps({key: job.get(key) for key in ('exit_code', 'failure', 'elapsed_seconds',
          'sampled_peak_rss_bytes', 'final_output_bytes', 'process_exit_verified')}, indent=2))
    if job['exit_code'] != 0 or job['failure']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
