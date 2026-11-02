from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, sha256, write_json

paths = [
    'outputs/fox-fringe-component-preparation-v2/inputs/base-no-legacy-cards.glb',
    'outputs/fox-material-fur-finish-v1/finish-config.json',
    'outputs/fox-material-fur-finish-v1/material-groom/report.json',
    'outputs/fox-material-fur-finish-v1/material-groom/correspondence.npz',
    'outputs/fox-material-fur-finish-v1/material-coat.json',
    'outputs/fox-material-fur-finish-v1/sources/blender_material_groom.py',
    'outputs/fox-material-fur-finish-v1/sources/blender_material_coat.py',
    'outputs/fox-material-fur-finish-v1/sources/blender_fur_cards.py',
    'avatar_assets/runner.py', 'avatar_assets/safety.py',
]
inputs = []
for name in paths:
    p = ROOT/name
    assert p.is_file() and not p.is_symlink()
    inputs.append({'path': name, 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    if p.suffix == '.py':
        target = OUT/'sources'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.exists()
        shutil.copyfile(p, target)
assert inputs[0]['sha256'] == 'f34aa798252355fc6e44a03f78cdc7f31eab682a20b591a93c9dcb35ed82f8a3'
for name in ('census.py', 'execute.py'):
    compile((OUT/name).read_text(), str(OUT/name), 'exec')
command = [sys.executable, '-B', str(OUT/'census.py')]
write_json(OUT/'prepared.json', {'schema_version': 1,
    'at': datetime.now(ZoneInfo('America/Los_Angeles')).isoformat(),
    'hypothesis': 'The unchanged material tuft counts may have sufficient vertex capacity but lose spatial uniformity from coincident seams and tessellation.',
    'inputs': inputs, 'scripts': {name: sha256(OUT/name) for name in ('census.py', 'execute.py')},
    'command': command, 'limits': {'attempts': 1, 'seconds': 30, 'memory_mb': 512, 'output_mb': 16},
    'settings': {'grid': [6, 6, 6], 'source_id_binding': 'Exact source-derived arrays against sealed native hashes',
        'coordinate_conversion': 'glTF x,y,z to Blender x,-z,y; optionally canonical numeric zero before hash binding',
        'tuple_sampling': None, 'texture_decode': False, 'native_jobs': 0, 'provider_calls': 0, 'ai_role_calls': 0}})
job = run_bounded(command, OUT/'cpu.log', OUT, timeout=30, memory_mb=512, output_mb=16)
write_json(OUT/'cpu-job.json', job)
unchanged = [{'path': row['path'], 'unchanged': (ROOT/row['path']).stat().st_size == row['bytes']
    and sha256(ROOT/row['path']) == row['sha256']} for row in inputs]
report = json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').is_file() else None
passed = (job.get('exit_code') == 0 and not job.get('failure') and job.get('process_exit_verified')
          and report is not None and all(x['unchanged'] for x in unchanged))
write_json(OUT/'receipt.json', {'schema_version': 1, 'status': 'measured' if passed else 'failed',
    'source_unchanged': unchanged, 'job': job, 'report_sha256': sha256(OUT/'report.json') if report else None,
    'quality_approved': False, 'native_jobs': 0, 'provider_calls': 0, 'ai_role_calls': 0,
    'limits': 'Source capacity metrics are not a full-coat or visual-density verdict; no selector was implemented.'})
write_json(OUT/'manifest.json', {'schema_version': 1, 'kind': 'vertex_root_coverage_preflight',
    'status': 'measured' if passed else 'failed', 'artifacts': artifacts(OUT)})
print(json.dumps({'status': 'measured' if passed else 'failed', 'job': job,
                  'manifest_sha256': sha256(OUT/'manifest.json')}, sort_keys=True))
raise SystemExit(0 if passed else 1)
