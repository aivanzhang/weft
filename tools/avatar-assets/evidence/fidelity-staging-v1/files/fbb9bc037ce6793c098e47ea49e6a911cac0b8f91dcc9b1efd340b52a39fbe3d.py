"""One focused integration run for reusable fringe finishing and existing pipeline contracts."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import require,sha256,write_json
cfg=json.loads((HERE/'prepared.json').read_text());require(not (HERE/'execution-started.json').exists(),'Already executed')
for row in cfg['sources']:require(sha256(ROOT/row['path'])==row['sha256'],'Source changed')
write_json(HERE/'execution-started.json',{'prepared_sha256':sha256(HERE/'prepared.json')})
os.environ['PYTHONPATH']=str(ROOT/'tests')
job=run_bounded([sys.executable,'-m','unittest','-q',*cfg['modules']],HERE/'tests.txt',HERE,timeout=30,memory_mb=512,output_mb=16)
unchanged=all(sha256(ROOT/r['path'])==r['sha256']for r in cfg['sources'])
write_json(HERE/'receipt.json',{'job':job,'sources_unchanged':unchanged,'source_count':len(cfg['sources']),'new_generation_calls':0,'log_sha256':sha256(HERE/'tests.txt'),'modules':cfg['modules']})
print(json.dumps({k:job[k]for k in ['exit_code','failure','elapsed_seconds','sampled_peak_rss_bytes','process_exit_verified']}))
require(job['exit_code']==0 and not job['failure'] and unchanged,'Preserve failed integration evidence')
