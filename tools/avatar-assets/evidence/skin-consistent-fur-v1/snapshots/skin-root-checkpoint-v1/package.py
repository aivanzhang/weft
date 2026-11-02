"""Preserve exact root-placement evidence, including numeric binding and motion data."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
inputs = json.loads((HERE / 'inputs.json').read_text())
out = ROOT / inputs['output']
assert out.is_dir() and not list(out.iterdir())
assert (ROOT/'outputs/skin-root-card-pin-tests-v1/release.json').is_file()
started = time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


paths = []
for folder in inputs['folders']:
    origin = ROOT/'outputs'/folder
    assert origin.is_dir() and not origin.is_symlink()
    paths.extend(sorted(origin.rglob('*')))
paths.extend([HERE/name for name in ('package.py', 'inputs.json', 'package-preflight-v1.py', 'preflight-failure.json', 'request.json', 'request-v2.json')])
rows, references, seen = [], [], {}
selected_bytes = copied_bytes = 0
for path in paths:
    assert not path.is_symlink()
    if not path.is_file() or '__pycache__' in path.parts or '.cache' in path.parts:
        continue
    size = path.stat().st_size
    row = {'origin': str(path.relative_to(ROOT)), 'bytes': size, 'sha256': sha(path)}
    if path.suffix in ('.glb', '.blend'):
        references.append(row)
        continue
    assert path.suffix in ('.py', '.json', '.log', '.jsonl', '.png', '.md', '.diff', '.npz', '.lock')
    assert size <= inputs['limits']['single_file_bytes']
    selected_bytes += size
    assert len(rows) < inputs['limits']['selected_files']
    assert selected_bytes <= inputs['limits']['selected_bytes']
    key = (row['sha256'], size)
    if key not in seen:
        target = Path('snapshots')/path.relative_to(ROOT/'outputs')
        (out/target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out/target)
        assert sha(out/target) == row['sha256']
        seen[key] = str(target)
        copied_bytes += size
    row['snapshot'] = seen[key]
    rows.append(row)
    assert time.monotonic()-started < inputs['limits']['wall_seconds']
for ref in inputs['external_references']:
    path = Path(ref['path'])
    path = path if path.is_absolute() else ROOT/path
    assert sha(path) == ref['sha256']
    references.append({'origin': ref['path'], 'bytes': path.stat().st_size,
                       'sha256': ref['sha256'], 'unchanged': True})
write(out/'manifest.json', {
    'schema_version': 1,
    'recorded': datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z'),
    'status': inputs['status'], 'artifacts': rows, 'local_references_not_copied': references,
    'source_recipe': {'path': str((HERE/'package.py').relative_to(ROOT)), 'sha256': sha(HERE/'package.py')},
    'source_inputs_sha256': sha(HERE/'inputs.json'), 'source_receipts': len(rows),
    'unique_snapshots': len(seen), 'source_bytes': selected_bytes, 'copied_bytes': copied_bytes,
    'deduplication': 'Identical bytes reuse one exact snapshot; no normalization',
    'omitted_runtime_caches': ['__pycache__', '.cache'], 'verification': inputs['verification'],
    'binding_and_full_motion_npz_included': True,
    'new_provider_calls': 0, 'new_pipeline_role_calls': 0, 'new_rig_or_action_authored': False,
    'quality_approved': False, 'fresh_studio_acceptance': False, 'default_changed': False,
    'approved_assets_unchanged': True, 'weft_consumer_changed': False,
    'limits': inputs['limits'], 'elapsed_seconds': round(time.monotonic()-started, 3)})
with (out/'README.md').open('x') as stream:
    stream.write('''# Skin-consistent fur evidence

Exact source/test/configuration snapshots preserve the opt-in attachment policy,
stopped vertex-only draft, coverage screens, actual existing-fox finish, independent
body/color/all-root audits, public verification and matched static visual review.
The final prepared-job card-builder pin fix and its focused regression are separate
from the historical native build. All roots and all 108,000 sampled motion results
are retained in exact numeric archives; no threshold was relaxed.

The manifest maps origin paths to byte-identical snapshots, deduplicating only
identical content. GLB/Blend files and the original ZIP remain immutable local
references by hash. This compact bundle is not a complete runnable asset pack
or remote model backup. Analytic fixtures are not generated characters.

Attachment passes; appearance remains short plush, not dense Muse fur. One
coincident seam pair and source body/shading/deformation defects remain. No
continuous motion, loop, native strand sampling, fresh generation, automatic
Studio delivery or Weft compatibility is accepted. Defaults, approved models,
gallery and spending allowances are unchanged. No provider or generation-pipeline
role calls occurred. See ../../docs/skin-consistent-fur.md for exact commands,
resource bounds, IDs, limitations and the next safe action.
''')
print(json.dumps({'origins': len(rows), 'unique_snapshots': len(seen), 'bytes': copied_bytes,
                  'manifest_sha256': sha(out/'manifest.json'),
                  'seconds': round(time.monotonic()-started, 3)}))
