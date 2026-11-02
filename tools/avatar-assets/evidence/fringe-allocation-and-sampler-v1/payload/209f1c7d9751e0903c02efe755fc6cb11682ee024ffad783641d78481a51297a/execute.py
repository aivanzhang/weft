"""One bounded focused-test run with exact before/after source pins."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert not (HERE/'execution-started.json').exists()
prepared=json.loads((HERE/'prepared.json').read_text())
for row in prepared['prepared_files']:
    assert sha(HERE/row['path'])==row['sha256']
source=json.loads((HERE/'source-inventory.json').read_text())
for row in source['files']:
    assert sha(ROOT/row['path'])==row['sha256']
command=prepared['command']
(HERE/'execution-started.json').write_text(json.dumps({'command':command,'prepared_sha256':sha(HERE/'prepared.json')},indent=2)+'\n')
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
job=run_bounded(command,HERE/'tests.log',HERE,timeout=30,memory_mb=512,output_mb=32)
(HERE/'job.json').write_text(json.dumps(job,indent=2,sort_keys=True)+'\n')
changed=[row['path'] for row in source['files'] if sha(ROOT/row['path'])!=row['sha256']]
report={'source_count':len(source['files']),'changed_sources':changed,'source_pins_unchanged':not changed}
(HERE/'source-verification.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({'exit_code':job['exit_code'],'failure':job['failure'],'elapsed_seconds':job['elapsed_seconds'],'sampled_peak_rss_bytes':job['sampled_peak_rss_bytes'],'source_pins_unchanged':not changed}))
assert job['exit_code']==0 and not job['failure'] and not changed
