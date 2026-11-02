"""One bounded four-mode native diagnostic; no paid work or retry."""
import os,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
from avatar_assets.safety import require,sha256,write_json,new_directory
from avatar_assets.runner import run_bounded
cfg=json.loads((HERE/'prepared.json').read_text());require(not (HERE/'started.json').exists(),'One attempt only')
for r in cfg['inputs']:require(sha256(ROOT/r['path'])==r['sha256'],'Input changed')
for r in cfg['snapshots']:require(sha256(HERE/r['path'])==r['sha256'],'Snapshot changed')
exe=ROOT/'.tools/Blender.app/Contents/MacOS/Blender';require(sha256(exe)==cfg['blender_sha256'],'Tool changed')
out=new_directory(ROOT/cfg['output']);write_json(HERE/'started.json',{'prepared_sha256':sha256(HERE/'prepared.json')})
allowed={k:os.environ[k]for k in ('PATH','TMPDIR','LANG')if k in os.environ};allowed.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.environ.clear();os.environ.update(allowed)
cmd=[str(exe),'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','--threads','4','--python',str(HERE/'probe.py'),'--',str(HERE/'config.json'),str(out)]
job=run_bounded(cmd,out/'native.log',out,timeout=30,memory_mb=1024,output_mb=8)
unchanged=all(sha256(ROOT/r['path'])==r['sha256']for r in cfg['inputs']) and all(sha256(HERE/r['path'])==r['sha256']for r in cfg['snapshots'])
write_json(HERE/'receipt.json',{'job':job,'inputs_unchanged':unchanged,'generation_calls':0,'quality_approved':False})
print(json.dumps({k:job[k]for k in ('pid','exit_code','failure','elapsed_seconds','sampled_peak_rss_bytes','final_output_bytes')}))
require(job['exit_code']==0 and not job['failure'] and unchanged,'Diagnostic failed; preserve without retry')
