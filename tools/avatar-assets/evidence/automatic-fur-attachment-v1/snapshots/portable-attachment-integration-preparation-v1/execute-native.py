"""Execute only the pinned analytic native export once, within retained bounds."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


assert sys.argv[1:] == ['--execute']
request = json.loads((HERE/'native-request.json').read_text())
assert sha(request['blender']) == request['blender_sha256']
assert sha(HERE/'native-fixture.py') == request['script_sha256']
for row in request['inputs']:
    assert sha(ROOT/row['path']) == sha(HERE/row['snapshot']) == row['sha256']
out = ROOT/request['output']
out.mkdir()
env = {key: os.environ[key] for key in ('PATH', 'LANG', 'TMPDIR') if key in os.environ}
env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
os.environ.clear(); os.environ.update(env)
job = run_bounded(request['command'], HERE/'native.log', out, **request['limits'])
with (HERE/'native-job.json').open('x') as stream:
    json.dump(job, stream, indent=2, sort_keys=True); stream.write('\n')
print(json.dumps({key: job.get(key) for key in ('status', 'pid', 'exit_code', 'failure', 'elapsed_seconds',
                                               'sampled_peak_rss_bytes', 'final_output_bytes', 'process_exit_verified')}))
assert job['exit_code'] == 0 and not job['failure'] and job['process_exit_verified']
