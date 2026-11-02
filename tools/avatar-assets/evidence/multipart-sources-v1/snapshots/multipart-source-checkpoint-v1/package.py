"""Preserve exact bounded multipart evidence; keep large local binaries by hash."""
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
started = time.monotonic()


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


rows, references, seen = [], [], {}
selected_bytes = copied_bytes = 0
paths = []
for folder in inputs['folders']:
    origin = ROOT / 'outputs' / folder
    assert origin.is_dir()
    paths.extend(sorted(origin.rglob('*')))
paths.extend([HERE / 'package.py', HERE / 'inputs.json'])
for path in paths:
    assert not path.is_symlink()
    if not path.is_file() or '__pycache__' in path.parts or '.cache' in path.parts:
        continue
    size = path.stat().st_size
    row = {'origin': str(path.relative_to(ROOT)), 'bytes': size, 'sha256': sha(path)}
    if path.suffix in ('.glb', '.blend', '.npz'):
        references.append(row)
        continue
    assert path.suffix in ('.py', '.json', '.log', '.jsonl', '.png', '.md', '.diff')
    assert size <= inputs['limits']['single_file_bytes']
    selected_bytes += size
    assert len(rows) < inputs['limits']['selected_files'] and selected_bytes <= inputs['limits']['selected_bytes']
    key = (row['sha256'], size)
    if key not in seen:
        target = Path('snapshots') / path.relative_to(ROOT / 'outputs')
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, out / target)
        assert sha(out / target) == row['sha256']
        seen[key] = str(target)
        copied_bytes += size
    row['snapshot'] = seen[key]
    rows.append(row)
    assert time.monotonic() - started < 30
for ref in inputs['external_references']:
    path = Path(ref['path'])
    path = path if path.is_absolute() else ROOT / path
    assert sha(path) == ref['sha256']
    references.append({'origin': ref['path'], 'bytes': path.stat().st_size,
                       'sha256': ref['sha256'], 'unchanged': True})
write(out / 'manifest.json', {
    'schema_version': 1, 'recorded': datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z'),
    'status': inputs['status'], 'artifacts': rows, 'local_binary_references_not_copied': references,
    'source_recipe': {'path': str((HERE / 'package.py').relative_to(ROOT)), 'sha256': sha(HERE / 'package.py')},
    'source_inputs_sha256': sha(HERE / 'inputs.json'), 'source_receipts': len(rows),
    'unique_snapshots': len(seen), 'source_bytes': selected_bytes, 'copied_bytes': copied_bytes,
    'deduplication': 'Identical bytes reuse one exact snapshot path; no normalization',
    'omitted_runtime_caches': ['__pycache__', '.cache'],
    'verified_focused_tests': inputs['verified_focused_tests'], 'actual_public_composition': True,
    'independent_actual_array_audit': True, 'native_four_view_load': True,
    'new_provider_calls': 0, 'new_pipeline_role_calls': 0, 'rig_created': False,
    'animation_verified': False, 'quality_approved': False, 'fresh_studio_acceptance': False,
    'approved_assets_unchanged': True, 'weft_consumer_changed': False,
    'limits': inputs['limits'], 'elapsed_seconds': round(time.monotonic() - started, 3)})
with (out / 'README.md').open('x') as stream:
    stream.write('# Multipart source composition evidence\n\n'
                 'Exact snapshots preserve the reusable local composer, analytic tests, source inventory failures, '
                 'declared layout, actual public CLI execution, independent geometry/material audit, native '
                 'four-view loading and limited visual review. The manifest maps each origin to byte-identical '
                 'snapshots; duplicate content is stored once. Large GLB/Blend files stay at immutable local '
                 'paths with hashes. Runtime caches are excluded. This compact bundle alone is not a runnable avatar.\n\n'
                 'The real robot/dragon case uses previously generated complete sources. It creates no skin, '
                 'skeleton, motion, shoulder attachment or new fur. Native preview success does not establish '
                 'an exporter round trip, prompt quality or Muse parity. Analytic fixtures are not generated '
                 'assets. No provider or generation-pipeline role calls occurred; pending allowance is unchanged.\n\n'
                 'See ../../docs/multipart-sources.md for exact commands, limits, source identities, failures, '
                 'ownership and the next safe action. Prior private checkpoint publication is also preserved.\n')
print(json.dumps({'origins': len(rows), 'unique_snapshots': len(seen), 'bytes': copied_bytes,
                  'manifest_sha256': sha(out / 'manifest.json'), 'seconds': round(time.monotonic() - started, 3)}))
