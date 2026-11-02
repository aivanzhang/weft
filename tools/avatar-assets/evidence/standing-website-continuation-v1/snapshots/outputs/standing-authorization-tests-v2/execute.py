import json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import sha256,write_json,artifacts
p=json.loads((OUT/'prepared.json').read_text());assert all(sha256(ROOT/r['path'])==r['sha256'] for r in p['sources'])
assert sha256(OUT/'run.py')==p['worker_sha256']
job=run_bounded([sys.executable,'-B',str(OUT/'run.py')],OUT/'tests.log',OUT,timeout=30,memory_mb=512,output_mb=16)
write_json(OUT/'job.json',job)
unchanged=all(sha256(ROOT/r['path'])==r['sha256'] for r in p['sources'])
passed=job.get('exit_code')==0 and not job.get('failure') and job.get('process_exit_verified') and unchanged
write_json(OUT/'result.json',{'passed':bool(passed),'pinned_sources_unchanged':unchanged,'scope':'Analytic temporary-ledger fixtures only; no actual policy, grant, configuration or provider calls','job':job})
write_json(OUT/'manifest.json',{'schema_version':1,'kind':'standing_authorization_tests','status':'passed' if passed else 'failure_preserved','artifacts':artifacts(OUT)})
print(json.dumps({'passed':bool(passed),'seconds':job.get('elapsed_seconds'),'pid':job.get('pid'),'sampled_peak_rss_bytes':job.get('sampled_peak_rss_bytes'),'manifest_sha256':sha256(OUT/'manifest.json')}))
raise SystemExit(0 if passed else 1)
