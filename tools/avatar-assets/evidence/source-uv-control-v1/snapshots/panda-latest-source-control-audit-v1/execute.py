import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, write_json

job = run_bounded([ROOT/'.venv/bin/python', OUT/'audit.py'], OUT/'audit.log', OUT,
                  timeout=30, memory_mb=256, output_mb=16)
write_json(OUT/'job.json', job)
write_json(OUT/'manifest.json', {'schema_version':1,'quality_approved':False,
    'scope':'Read-only source/control CPU preparation; no native or texture pixel decode.',
    'artifacts':artifacts(OUT)})
print(json.dumps(job, sort_keys=True))
raise SystemExit(0 if not job['failure'] and job['exit_code'] == 0 else 1)
