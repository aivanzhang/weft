"""One independent exported-GLB preview; exact local inputs and a scoped output cap."""
import json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import sha256,require,write_json,new_directory
cfg=json.loads((HERE/'prepared.json').read_text());require(not (HERE/'started.json').exists(),'One attempt only')
for row in cfg['inputs']:require(sha256(ROOT/row['path'])==row['sha256'],'Input changed')
for row in cfg['snapshots']:require(sha256(HERE/row['path'])==row['sha256'],'Snapshot changed')
require(sha256(cfg['blender'])==cfg['blender_sha256'],'Tool changed')
out=new_directory(ROOT/cfg['output']);write_json(HERE/'started.json',{'prepared_sha256':sha256(HERE/'prepared.json')})
allowed={k:os.environ[k]for k in ('PATH','TMPDIR','LANG')if k in os.environ};allowed.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
os.environ.clear();os.environ.update(allowed)
cmd=[cfg['blender'],'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','--threads','4',
 '--python',str(HERE/'sources/blender_provider_preview.py'),'--',str(ROOT/cfg['model']),str(out),'720','16','false',str(HERE/'motion.json')]
job=run_bounded(cmd,out/'native.log',out,timeout=120,memory_mb=2048,output_mb=128)
unchanged=all(sha256(ROOT/r['path'])==r['sha256']for r in cfg['inputs']) and all(sha256(HERE/r['path'])==r['sha256']for r in cfg['snapshots'])
write_json(HERE/'receipt.json',{'job':job,'inputs_unchanged':unchanged,'paid_generation_calls':0,'quality_approved':False})
print(json.dumps({k:job[k]for k in ('pid','exit_code','failure','elapsed_seconds','sampled_peak_rss_bytes','final_output_bytes','process_exit_verified')}))
require(job['exit_code']==0 and not job['failure'] and unchanged,'Preview failed; preserve it without retry')
