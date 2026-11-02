"""Rerun only sandbox-denied loopback cases, with unchanged source pins."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import sha256,write_json,require
pins=json.loads((ROOT/'outputs/root-sampler-integration-tests-v1/sources.json').read_text())['files']
require(not (HERE/'execution-started.json').exists(),'Already executed')
require(all(sha256(ROOT/p['path'])==p['sha256'] for p in pins),'Sources changed')
ids=json.loads((HERE/'test-ids.json').read_text())
os.environ['PYTHONPATH']=str(ROOT/'tests')
write_json(HERE/'execution-started.json',{'cases':ids,'source_count':len(pins)})
job=run_bounded([sys.executable,'-m','unittest','-q',*ids],HERE/'tests.txt',HERE,timeout=60,memory_mb=512,output_mb=16)
unchanged=all(sha256(ROOT/p['path'])==p['sha256'] for p in pins)
receipt={'job':job,'sources_unchanged':unchanged,'source_count':len(pins),'case_count':len(ids),'log_sha256':sha256(HERE/'tests.txt'),'scope':'Only 27 loopback cases denied by the sandbox; previous 668 passing cases not rerun.','generation_calls':0}
write_json(HERE/'receipt.json',receipt)
print(json.dumps(receipt,indent=2))
require(job['exit_code']==0 and not job['failure'] and unchanged,'Inspect retained failure')
