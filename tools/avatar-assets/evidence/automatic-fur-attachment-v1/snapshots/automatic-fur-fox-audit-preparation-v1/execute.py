"""One local CPU integration check with a separately bounded audit child."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import read_json, sha256, write_json

assert sys.argv[1:] == ['--execute']
request = read_json(HERE / 'prepared-final.json')
assert sha256(HERE / 'check-stage.py') == request['driver_sha256']
command = ['/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'PYTHONDONTWRITEBYTECODE=1',
           'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'VECLIB_MAXIMUM_THREADS=1',
           str(ROOT / '.venv/bin/python'), '-B', str(HERE / 'check-stage.py')]
job = run_bounded(command, HERE / 'controller.log', ROOT / request['job'],
                  timeout=90, memory_mb=512, output_mb=64)
write_json(HERE / 'controller-job.json', job)
print({k: job[k] for k in ('status', 'pid', 'elapsed_seconds', 'exit_code',
                          'failure', 'sampled_peak_rss_bytes', 'process_exit_verified')})
assert job['exit_code'] == 0 and not job['failure'] and job['process_exit_verified']
