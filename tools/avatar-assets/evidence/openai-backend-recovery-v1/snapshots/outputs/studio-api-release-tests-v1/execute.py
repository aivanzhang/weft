import json,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts,sha256,write_json
prepared=json.loads((OUT/'prepared.json').read_text())
assert all(sha256(ROOT/p['path'])==p['sha256'] for p in prepared['sources'])
job=run_bounded([sys.executable,*prepared['command'][1:]],OUT/'tests.log',OUT,timeout=30,memory_mb=512,output_mb=16)
write_json(OUT/'job.json',job)
unchanged=all(sha256(ROOT/p['path'])==p['sha256'] for p in prepared['sources'])
passed=job.get('exit_code')==0 and not job.get('failure') and job.get('process_exit_verified') and unchanged
write_json(OUT/'result.json',{'passed':bool(passed),'sources_unchanged':unchanged,'provenance':'analytic fixtures only; no canonical state mutations','job':job})
write_json(OUT/'manifest.json',{'schema_version':1,'kind':'studio_api_terminal_release_tests','status':'passed' if passed else 'failed_preserved','artifacts':artifacts(OUT)})
print(json.dumps({'passed':bool(passed),'sources_unchanged':unchanged,'elapsed_seconds':job.get('elapsed_seconds'),'pid':job.get('pid'),'sampled_peak_rss_bytes':job.get('sampled_peak_rss_bytes'),'manifest_sha256':sha256(OUT/'manifest.json')}))
raise SystemExit(0 if passed else 1)
