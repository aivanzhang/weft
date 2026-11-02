"""Run the prepared analytic child once with the existing resource monitor."""

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
        path = HERE/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Prepared source changed: '+row['path'])
    result = run_bounded([str(ROOT/'.venv/bin/python'), '-B', str(HERE/'screen.py')],
                         HERE/'cpu.log', HERE, timeout=15, memory_mb=256, output_mb=16)
    with (HERE/'job.json').open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    for row in prepared['sources']:
        if hashlib.sha256((HERE/row['path']).read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Prepared source changed during execution')
    print(json.dumps({key: result.get(key) for key in ('exit_code', 'failure', 'elapsed_seconds',
          'sampled_peak_rss_bytes', 'final_output_bytes', 'process_exit_verified')}, indent=2))
    if result['exit_code'] != 0 or result['failure']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
