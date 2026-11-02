"""One bounded source-normal derivative; no automatic retries."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import require,sha256,write_json,artifacts
cfg=json.loads((HERE/'prepared.json').read_text())
require(not (HERE/'execution-started.json').exists(),'Already executed')
for row in cfg['inputs']:require(sha256(ROOT/row['path'])==row['sha256'],'Input changed')
require(sha256(HERE/'execute.py')==cfg['executor_sha256'],'Executor changed')
out=ROOT/cfg['output'];require(not out.exists(),'Output exists');out.mkdir()
write_json(HERE/'execution-started.json',{'prepared_sha256':sha256(HERE/'prepared.json'),'attempts':1})
job=run_bounded([str(ROOT/'.venv/bin/python'),'-B',str(HERE/'derive.py')],HERE/'cpu.log',out,timeout=30,memory_mb=512,output_mb=64)
write_json(HERE/'job.json',job)
unchanged=all(sha256(ROOT/row['path'])==row['sha256'] for row in cfg['inputs'])
write_json(out/'manifest.json',{'schema_version':1,'job':job,'prepared_sha256':sha256(HERE/'prepared.json'),'artifacts':artifacts(out),'quality_approved':False,'inputs_unchanged':unchanged,'new_provider_calls':0})
print(json.dumps({k:job[k] for k in ['exit_code','failure','elapsed_seconds','sampled_peak_rss_bytes','process_exit_verified']}))
require(job['exit_code']==0 and not job['failure'] and unchanged,'Preserve failed derivative')
