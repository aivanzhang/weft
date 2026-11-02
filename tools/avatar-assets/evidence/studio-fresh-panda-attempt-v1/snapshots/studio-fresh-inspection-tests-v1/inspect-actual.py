import hashlib,json,os,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
from avatar_assets.runner import run_bounded
pins=json.loads((out/'source-pins.json').read_text())
for item in pins:assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256']
command=[str(root/'.venv/bin/python'),'-B',str(root/'scripts/verify_studio_live.py'),'--job','2fa07fbb61ef4da4bc8f7f70']
os.environ.clear();os.environ.update({'PATH':'/usr/bin:/bin','LANG':'en_US.UTF-8','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':'/private/tmp'})
job=run_bounded(command,out/'actual-inspection.log',out,timeout=15,memory_mb=256,output_mb=16)
with (out/'actual-inspection-job.json').open('x') as stream:json.dump(job,stream,indent=2);stream.write('\n')
assert job['exit_code']==0 and job['failure'] is None and job['process_exit_verified'],job
result=json.loads((out/'actual-inspection.log').read_text())
assert result['plan_id']=='9c2f9d62e6dfc883088a897970b15639f89229311cb2d3c548fac5c0da7bc425'
assert result['generation_mode']=='new_meshy_generation' and result['owner']['active_lease_job'] is None and result['owner']['queued_jobs']==[]
for item in pins:assert hashlib.sha256((root/item['path']).read_bytes()).hexdigest()==item['sha256']
print(json.dumps({'status':'passed','generation_mode':result['generation_mode'],'job_status':result['status'],'plan_id':result['plan_id'],'source_generation':result['source_generation'],'native_or_provider_calls':0,'server_or_browser_started':False,'elapsed_seconds':job['elapsed_seconds'],'sampled_peak_rss_bytes':job['sampled_peak_rss_bytes'],'rss_samples':job['rss_samples'],'pid':job['pid'],'process_exit_verified':job['process_exit_verified']},indent=2))
