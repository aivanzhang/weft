import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sources/host'))
from avatar_assets.runner import run_bounded
p=json.loads((HERE/'prepared.json').read_text())
for row in p['source_pins']:
    f=HERE/row['path']
    with f.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==row['sha256'] and f.stat().st_size==row['bytes']
r=run_bounded(p['command'],HERE/'replay.log',HERE,timeout=10,memory_mb=512,output_mb=16)
with (HERE/'job.json').open('x') as f:json.dump(r,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(r))
sys.exit(0 if r.get('exit_code')==0 and not r.get('failure') else 1)
