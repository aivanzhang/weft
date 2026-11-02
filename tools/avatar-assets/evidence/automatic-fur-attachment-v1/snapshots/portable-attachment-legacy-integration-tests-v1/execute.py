import json, os
from pathlib import Path
from avatar_assets.runner import run_bounded
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
command=['/Users/ivan/kintsugi/weft/tools/avatar-assets/.venv/bin/python', '-B', '/Users/ivan/kintsugi/weft/tools/avatar-assets/outputs/portable-attachment-legacy-integration-tests-v1/run-tests.py']
os.environ.clear()
os.environ.update({'PATH':'/usr/bin:/bin','LANG':'en_US.UTF-8','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':'/private/tmp'})
job=run_bounded(command,out/'tests.log',out,timeout=60,memory_mb=512,output_mb=16)
with (out/'job.json').open('x') as handle:json.dump(job,handle,sort_keys=True,indent=2);handle.write('\n')
print(json.dumps(job,sort_keys=True,indent=2))
raise SystemExit(0 if job['exit_code']==0 and job['failure'] is None and job['process_exit_verified'] else 1)
