"""One bounded actual-source inspection, without provider calls or native finishing."""
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
require(not (HERE/'started.json').exists(), 'One prepared preflight attempt only')
for row in config['snapshots']:
    require(sha256(HERE/row['path']) == row['sha256'], 'Frozen preflight input changed')
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
           '--python-exit-code', '1', '--threads', '4', '--python', str(HERE/'probe.py'), '--', str(out)]
job = run_bounded(command, out/'native.log', out, timeout=45, memory_mb=1536, output_mb=32)
unchanged = all(sha256(ROOT/row['path']) == row['sha256'] for row in config['inputs'])
write_json(HERE/'receipt.json', {'job': job, 'originals_unchanged': unchanged,
                               'new_generation_calls': 0, 'quality_approved': False})
write_json(out/'manifest.json', {'schema_version': 1, 'job': job, 'artifacts': artifacts(out),
    'prepared_sha256': sha256(HERE/'prepared.json'), 'originals_unchanged': unchanged,
    'scope': 'Actual-source preflight, not full finishing or visual acceptance'})
print(json.dumps(job))
require(unchanged and job['exit_code'] == 0 and not job['failure'], 'Preflight failed; retain without replay')
