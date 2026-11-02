"""One bounded native regression; pinned code and inputs, no paid work."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources'))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import new_directory, require, sha256, write_json

cfg = json.loads((HERE/'prepared.json').read_text())
require(not (HERE/'started.json').exists(), 'One native attempt only')
for row in cfg['inputs']:
    require(sha256(ROOT/row['path']) == row['sha256'], 'Original input changed')
for row in cfg['snapshots']:
    require(sha256(HERE/row['path']) == row['sha256'], 'Frozen source changed')
exe = ROOT/'.tools/Blender.app/Contents/MacOS/Blender'
require(sha256(exe) == cfg['blender_sha256'], 'Native tool changed')
out = new_directory(ROOT/cfg['output'])
write_json(HERE/'started.json', {'prepared_sha256': sha256(HERE/'prepared.json')})
allowed = {key: os.environ[key] for key in ('PATH', 'TMPDIR', 'LANG') if key in os.environ}
os.environ.clear(); os.environ.update(allowed)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
command = [str(exe), '--background', '--factory-startup', '--disable-autoexec',
    '--python-exit-code', '1', '--threads', '4', '--python', str(HERE/'check.py'), '--', str(out)]
job = run_bounded(command, out/'native.log', out, timeout=30, memory_mb=1536, output_mb=16)
unchanged = all(sha256(ROOT/r['path']) == r['sha256'] for r in cfg['inputs'])
write_json(HERE/'receipt.json', {'job': job, 'inputs_unchanged': unchanged,
    'new_paid_generation_calls': 0, 'quality_approved': False})
print(json.dumps(job))
require(job['exit_code'] == 0 and not job['failure'] and unchanged, 'Native proof failed; preserve without replay')
