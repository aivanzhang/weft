"""Run the frozen bounded attribution once, with explicit Python error exit."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import new_directory, require, sha256, write_json

prep = ROOT/'outputs/shoulder-subpixel-attribution-preparation-v1'
cfg = json.loads((prep/'prepared.json').read_text())
require(not (HERE/'started.json').exists(), 'One attempt only')
for row in cfg['inputs']:
    require(sha256(row['path']) == row['sha256'], 'Prepared input changed')
for row in cfg['sources']:
    require(sha256(prep/row['path']) == row['sha256'], 'Frozen source changed')
command = cfg['command'].copy()
require(sha256(command[0]) == cfg['blender_sha256'], 'Native tool changed')
command[command.index('--threads'):command.index('--threads')] = ['--python-exit-code', '1']
out = new_directory(cfg['output'])
write_json(HERE/'started.json', {'prepared_sha256': sha256(prep/'prepared.json'), 'command': command,
    'command_change': 'Explicit Python exception exit only; frozen config and probe unchanged',
    'controller_sha256': sha256(__file__)})
allowed = {key: os.environ[key] for key in ('PATH', 'TMPDIR', 'LANG') if key in os.environ}
os.environ.clear(); os.environ.update(allowed); os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
job = run_bounded(command, out/'native.log', out, timeout=30, memory_mb=1024, output_mb=8)
unchanged = all(sha256(r['path']) == r['sha256'] for r in cfg['inputs'])
write_json(HERE/'receipt.json', {'job': job, 'inputs_unchanged': unchanged, 'new_paid_generation_calls': 0})
print(json.dumps(job))
require(job['exit_code'] == 0 and not job['failure'] and unchanged and (out/'subpixel-attribution.json').exists(),
        'Attribution failed; preserve without retry')
