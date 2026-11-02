"""Content-addressed compact evidence; original models/scenes remain local."""
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.safety import new_directory, require, sha256, write_json

cfg = json.loads((HERE/'sources.json').read_text())
entries, blobs = [], {}
for name in cfg['directories']:
    folder = ROOT/'outputs'/name
    require(folder.is_dir() and folder.parent == ROOT/'outputs', 'Invalid evidence source')
    for path in sorted(folder.rglob('*')):
        require(not path.is_symlink(), 'Do not package symlinks')
        if not path.is_file():
            continue
        require(len(entries) < cfg['max_entries'], 'Evidence entry budget exceeded')
        suffix = path.suffix.lower(); size = path.stat().st_size; digest = sha256(path)
        keep = (suffix in {'.json', '.jsonl', '.py', '.txt', '.log', '.diff', '.md', '.npz', '.png'}
                and size <= 4*1024**2 and '__pycache__' not in path.parts)
        row = {'origin': str(path.relative_to(ROOT)), 'bytes': size, 'sha256': digest,
               'retained_in_bundle': keep}
        if keep:
            target = 'files/'+digest+suffix
            row['bundle_path'] = target; blobs.setdefault(target, path)
        else:
            row['omission'] = 'Local-only model/scene/cache/large artifact; original retained unchanged'
        entries.append(row)
total = sum(path.stat().st_size for path in blobs.values())
require(total <= cfg['snapshot_byte_limit'], 'Compact evidence byte budget exceeded')
out = ROOT/cfg['bundle']
require(out.parent == ROOT/'evidence' and out.is_dir() and not out.is_symlink()
        and not any(out.iterdir()), 'Controller must supply a new empty evidence directory')
(out/'files').mkdir()
for name, path in sorted(blobs.items()):
    target = out/name; shutil.copyfile(path, target)
    require(sha256(target) == target.stem, 'Evidence copy changed')
shutil.copyfile(__file__, out/'package.py')
shutil.copyfile(HERE/'sources.json', out/'sources.json')
(out/'README.md').write_text(
    '# Avatar contour and contact quality checkpoint\n\n'
    'Normal-transfer rejection is verified in the planner, pipeline and actual Blender path. '
    'Regional fringe improves sampled contours; a UV-only candidate reduces but does not eliminate '
    'visible shoulder flecks. The curved cheek fit is rejected and produces no model. '
    'No new avatar, Muse-quality parity or fresh Studio delivery is accepted.\n\n'
    '`manifest.json` binds exact original paths and compact content-addressed copies. '
    'Large GLB/Blend files remain local with recorded hashes; this is not their remote backup. '
    'Failures and original bytes are retained. Read ../../TODO.md, '
    '../../docs/normal-transfer-guard.md and ../../docs/contour-contact-diagnosis.md.\n')
write_json(out/'manifest.json', {'schema_version': 1, 'status': 'development_quality_not_accepted',
    'origins': entries, 'origin_count': len(entries), 'unique_snapshots': len(blobs),
    'snapshot_bytes': total, 'package_sha256': sha256(out/'package.py'),
    'sources_sha256': sha256(out/'sources.json'), 'readme_sha256': sha256(out/'README.md'),
    'large_model_scene_remote_backup': False, 'quality_approved': False,
    'fresh_studio_accepted': False, 'new_paid_generation_calls': 0})
require(all(sha256(ROOT/r['origin']) == r['sha256'] for r in entries), 'Original evidence changed')
write_json(HERE/'receipt.json', {'bundle': str(out.relative_to(ROOT)),
    'manifest_sha256': sha256(out/'manifest.json'), 'origin_count': len(entries),
    'unique_snapshots': len(blobs), 'snapshot_bytes': total, 'all_origins_unchanged': True})
print((HERE/'receipt.json').read_text())
