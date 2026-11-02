import json, os, sys
from pathlib import Path
root=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(root))
from avatar_assets.runner import run_bounded
out=Path(__file__).resolve().parent
os.environ.clear()
os.environ.update({'PATH':'/usr/bin:/bin','LANG':'en_US.UTF-8','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':'/private/tmp'})
job=run_bounded(['/Users/ivan/kintsugi/weft/tools/avatar-assets/.venv/bin/python', '-B', '/Users/ivan/kintsugi/weft/tools/avatar-assets/outputs/automatic-fur-fox-preservation-v1/compare.py'],out/'comparison.log',out,timeout=30,memory_mb=384,output_mb=16)
with (out/'job.json').open('x') as stream:json.dump(job,stream,sort_keys=True,indent=2);stream.write('\n')
print(json.dumps(job,sort_keys=True,indent=2))
raise SystemExit(0 if job['exit_code']==0 and job['failure'] is None and job['process_exit_verified'] else 1)
