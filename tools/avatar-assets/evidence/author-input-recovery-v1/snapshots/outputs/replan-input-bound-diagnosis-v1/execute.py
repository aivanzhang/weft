import json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import sha256,write_json,artifacts
p=json.loads((OUT/'prepared.json').read_text());assert sha256(OUT/'check.py')==p['worker_sha256']
job=run_bounded([sys.executable,'-B',str(OUT/'check.py')],OUT/'check.log',OUT,timeout=30,memory_mb=512,output_mb=16)
write_json(OUT/'job.json',job)
passed=job.get('exit_code')==0 and not job.get('failure') and job.get('process_exit_verified')
write_json(OUT/'result.json',{'passed':bool(passed),'job':job,'scope':'Read-only actual Studio/API custody and artifact hashes; no API, native, worker dispatch, release, grant or config mutation'})
write_json(OUT/'manifest.json',{'schema_version':1,'kind':'actual_replan_predispatch_diagnosis','status':'verified_read_only' if passed else 'failure_preserved','artifacts':artifacts(OUT)})
print(json.dumps({'passed':bool(passed),'seconds':job.get('elapsed_seconds'),'pid':job.get('pid'),'sampled_peak_rss_bytes':job.get('sampled_peak_rss_bytes'),'manifest_sha256':sha256(OUT/'manifest.json')}))
raise SystemExit(0 if passed else 1)
