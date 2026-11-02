import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sources'))
from avatar_assets.runner import run_bounded

def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def write(name,value):
    with (HERE/name).open('x') as handle:
        json.dump(value,handle,sort_keys=True,indent=2,allow_nan=False)
        handle.write('\n')

p=json.loads((HERE/'prepared.json').read_text())
for row in p['inputs']+p['sources']:
    if sha(row['path']) != row['sha256']:
        raise ValueError('Pinned input changed: '+row['path'])
job=run_bounded([p['python'],str(HERE/'compare.py')],HERE/'cpu.log',HERE,timeout=30,memory_mb=512,output_mb=16)
write('cpu-job.json',job)
unchanged=all(sha(row['path']) == row['sha256'] for row in p['inputs']+p['sources'])
write('custody.json',{'all_pinned_inputs_unchanged':unchanged,'native_calls':0,'provider_calls':0})
print(json.dumps({'job':job,'inputs_unchanged':unchanged}))
sys.exit(0 if job['exit_code']==0 and not job['failure'] and unchanged else 1)
