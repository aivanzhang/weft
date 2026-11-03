from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import read_json,write_json,sha256
out=Path(__file__).resolve().parent;prepared=read_json(out/'prepared.json')
assert sha256(out/'ablate.py')==prepared['script_sha256']
for name,expected in prepared['sources'].items():assert sha256(out/'sources'/name)==expected
r=run_bounded(prepared['command'],out/'native.log',out,timeout=60,memory_mb=2048,output_mb=64)
write_json(out/'job.json',r)
print({k:r[k] for k in ('exit_code','failure','elapsed_seconds','sampled_peak_rss_bytes','process_exit_verified')})
print((out/'native.log').read_text()[-2400:])
sys.exit(0 if r['exit_code']==0 and not r['failure'] else 1)
