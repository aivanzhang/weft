"""One bounded experimental native preparation; no automatic renders or paid calls."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE/'sources'))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, new_directory, require, sha256, write_json

config = json.loads((HERE/'prepared.json').read_text())
require(not (HERE/'started.json').exists(), 'One prepared native attempt only')
for row in config['snapshots']:
    require(sha256(HERE/row['path']) == row['sha256'], 'Frozen native input changed')
for row in config['inputs']:
    require(sha256(ROOT/row['path']) == row['sha256'], 'Original source changed')
exe = ROOT/'.tools/Blender.app/Contents/MacOS/Blender'
require(sha256(exe) == config['blender_sha256'], 'Native tool changed')
out = new_directory(ROOT/config['output'])
write_json(HERE/'started.json', {'prepared_sha256': sha256(HERE/'prepared.json')})
allowed = {key: os.environ[key] for key in ('PATH', 'TMPDIR', 'LANG') if key in os.environ}
os.environ.clear(); os.environ.update(allowed)
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
command = [str(exe), '--background', '--factory-startup', '--disable-autoexec',
           '--python-exit-code', '1', '--threads', '4', '--python', str(HERE/'prepare_entry.py'), '--', str(out)]
job = run_bounded(command, out/'prepare.log', out, timeout=120, memory_mb=2048, output_mb=512)
unchanged = all(sha256(ROOT/row['path']) == row['sha256'] for row in config['inputs'])
write_json(HERE/'receipt.json', {'job': job, 'originals_unchanged': unchanged,
    'new_generation_calls': 0, 'new_pipeline_role_calls': 0, 'quality_approved': False})
write_json(out/'preparation-manifest.json', {'schema_version': 1, 'job': job, 'artifacts': artifacts(out),
    'prepared_sha256': sha256(HERE/'prepared.json'), 'originals_unchanged': unchanged,
    'scope': 'Experimental rig/fur/scene preparation; no rendered images or accepted delivery'})
print(json.dumps(job))
require(unchanged and job['exit_code'] == 0 and not job['failure'], 'Native preparation failed; retain without replay')
