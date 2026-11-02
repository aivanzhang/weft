"""Preserve exact material-fur evidence; keep large local binaries by hash."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import shutil
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
inputs = json.loads((HERE / 'inputs-final.json').read_text())
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
paths.extend([HERE / 'package-final.py', HERE / 'inputs-final.json'])
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
    'source_recipe': {'path': str((HERE / 'package-final.py').relative_to(ROOT)), 'sha256': sha(HERE / 'package-final.py')},
    'source_inputs_sha256': sha(HERE / 'inputs-final.json'), 'source_receipts': len(rows),
    'unique_snapshots': len(seen), 'source_bytes': selected_bytes, 'copied_bytes': copied_bytes,
    'deduplication': 'Identical bytes reuse one exact snapshot path; no normalization',
    'omitted_runtime_caches': ['__pycache__', '.cache'],
    'verification': inputs['verification'],
    'new_provider_calls': 0, 'new_pipeline_role_calls': 0, 'rig_created': False,
    'new_motion_quality_accepted': False, 'quality_approved': False, 'fresh_studio_acceptance': False,
    'approved_assets_unchanged': True, 'weft_consumer_changed': False,
    'limits': inputs['limits'], 'elapsed_seconds': round(time.monotonic() - started, 3)})
with (out / 'README.md').open('x') as stream:
    stream.write('# Material-owned fur evidence\n\nExact snapshots preserve the opt-in color/native-groom/portable-tuft helpers,\nconfiguration and authoring contracts, focused tests, actual existing-fox finish,\nindependent body/color checks and rejected surface-attachment audit, four-view GLB loading and limited visual\nreview. The manifest maps each origin to byte-identical snapshots; identical\ncontent is stored once. All preparation/checker failures are retained. Large\nGLB/Blend/NumPy files remain at immutable local paths with hashes; this compact\nbundle is not a complete runnable model pack or remote binary backup.\n\nThe real example uses existing authored geometry and four existing actions, not\nfresh Meshy or AI-author generation. The portable coat remains short plush, not\ndense Muse fur; existing motion-stretch and surface-boundary limitations remain.\nThe declared exported skin passes but21/384sampled roots drift from the surface, up to2.495mm. Native strand sampling, all-root colors, continuous motion quality and an accepted\nStudio delivery are not established. Analytic custody fixtures are not native\nmodels/media. No provider or generation-pipeline role calls occurred, no pending\nallowance was granted, and no approved panda/dragon/gallery asset was replaced.\n\nSee ../../docs/material-owned-fur.md for inputs, exact commands, bounds, identities,\nresults, failures and continuation. The earlier private publication receipt is\npreserved; the new publication receipt will remain local after commit/push.\n')
print(json.dumps({'origins': len(rows), 'unique_snapshots': len(seen), 'bytes': copied_bytes,
                  'manifest_sha256': sha(out / 'manifest.json'), 'seconds': round(time.monotonic() - started, 3)}))
