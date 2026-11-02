"""One bounded CPU diagnosis; no retry or native execution."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'sources/host'))
from avatar_assets.runner import run_bounded
prepared=json.loads((HERE/'prepared.json').read_text())
for row in prepared['source_pins']:
    path=HERE/row['path']
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if digest != row['sha256'] or path.stat().st_size != row['bytes']:
        raise ValueError('Frozen diagnostic source changed: '+row['path'])
report=run_bounded(prepared['command'],HERE/'diagnosis.log',HERE,timeout=30,memory_mb=512,output_mb=16)
with (HERE/'job.json').open('x') as stream:
    json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(report))
sys.exit(0 if report.get('exit_code') == 0 and not report.get('failure') else 1)
