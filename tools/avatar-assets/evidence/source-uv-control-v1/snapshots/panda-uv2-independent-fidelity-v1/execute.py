import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts,write_json

job=run_bounded([ROOT/'.venv/bin/python',OUT/'check.py',OUT/'request.json'],OUT/'audit.log',OUT,
    timeout=30,memory_mb=512,output_mb=16)
write_json(OUT/'job.json',job)
write_json(OUT/'manifest.json',{'schema_version':1,'quality_approved':False,
    'scope':'Independent final source-ID, byte-custody, skin and sampled attachment audit; no native execution.',
    'artifacts':artifacts(OUT)})
print(json.dumps(job,sort_keys=True))
raise SystemExit(0 if job['exit_code']==0 and not job['failure'] else 1)
