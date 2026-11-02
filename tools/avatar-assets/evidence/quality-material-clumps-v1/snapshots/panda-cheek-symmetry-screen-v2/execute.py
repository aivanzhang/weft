import hashlib,json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
prep=json.loads((OUT/'prepared.json').read_text())
for row in prep['inputs']+prep['scripts']:
 with (ROOT/row['path']).open('rb') as stream: actual=hashlib.file_digest(stream,'sha256').hexdigest()
 if actual!=row['sha256']: raise RuntimeError('Changed pinned source: '+row['path'])
report=run_bounded([str(ROOT/'.venv/bin/python'),str(OUT/'screen.py')],OUT/'cpu.log',OUT,timeout=57.399,memory_mb=512,output_mb=16)
with (OUT/'cpu-job.json').open('x') as stream: json.dump(report,stream,indent=2,sort_keys=True);stream.write('\n')
print(json.dumps(report,sort_keys=True))
