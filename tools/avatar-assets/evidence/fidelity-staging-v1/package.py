"""Preserve compact experimental evidence; large models stay local with hashes."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from avatar_assets.safety import new_directory, sha256, write_json

NAMES = ['fringe-exporter-causes-v1', 'fringe-fidelity-fox-independent-audit-v1', 'fringe-fidelity-fox-producer-v1', 'fringe-fidelity-independent-audit-preparation-v1', 'fringe-fidelity-independent-preparation-v1', 'fringe-fidelity-independent-preparation-v2', 'fringe-fidelity-panda-independent-audit-v1', 'fringe-fidelity-panda-independent-audit-v2', 'fringe-fidelity-panda-producer-v1', 'fringe-fidelity-tests-v1', 'fringe-normal-storage-native-v1', 'fringe-source-id-review-v1', 'native-render-task-tests-v1', 'native-render-task-tests-v2', 'panda-staged-fringe-preparation-v1', 'panda-staged-fringe-v1', 'panda-staged-native-review-v1', 'panda-staged-portable-model-v1', 'panda-staged-portable-preparation-v1', 'panda-staged-portable-preview-preparation-v1', 'panda-staged-portable-preview-v1', 'panda-staged-skin-restore-v1', 'source-skin-fidelity-tests-v1', 'source-skin-snapshot-tests-v1', 'staged-finish-pipeline-contract-tests-v1', 'staged-fringe-id-audit-preparation-v2', 'staged-fringe-id-audit-v2', 'staged-native-render-review-v1']


def main():
    entries, blobs = [], {}
    for name in NAMES:
        folder = ROOT/'outputs'/name
        assert folder.is_dir(), name
        for path in sorted(folder.rglob('*')):
            assert not path.is_symlink(), path
            if not path.is_file():
                continue
            assert len(entries) < 10000
            suffix = path.suffix.lower()
            keep = suffix in {'.json', '.jsonl', '.py', '.txt', '.log', '.diff', '.md', '.npz'}
            if suffix == '.png':
                keep = True
            size, digest = path.stat().st_size, sha256(path)
            keep = keep and size <= 4*1024**2 and '__pycache__' not in path.parts
            entry = {'origin': str(path.relative_to(ROOT)), 'bytes': size, 'sha256': digest,
                     'retained_in_bundle': keep}
            if keep:
                target = 'files/'+digest+suffix
                entry['bundle_path'] = target
                blobs.setdefault(target, path)
            else:
                entry['omission'] = 'Local-only model/scene/cache/large or unselected image; preserve original output'
            entries.append(entry)
    total = sum(path.stat().st_size for path in blobs.values())
    assert total <= 32*1024**2, total
    out = new_directory(ROOT/'evidence/fidelity-staging-v1')
    (out/'files').mkdir()
    for name, path in sorted(blobs.items()):
        target = out/name
        shutil.copyfile(path, target)
        assert sha256(target) == target.stem
    shutil.copyfile(__file__, out/'package.py')
    (out/'README.md').write_text(
        '# Export fidelity and staged rendering checkpoint\n\n'
        'Actual panda normal/weight preservation and staged native memory checks pass. '
        'Native and portable appearance still fail overall acceptance; experimental finishing remains disabled. '
        'No model/default/Studio quality approval or new paid generation.\n\n'
        '`manifest.json` maps exact local origins to content-addressed snapshots. '
        'Large GLB/Blend files and omitted artifacts remain local; their hashes are recorded, '
        'but this bundle is not their remote backup. All failures are preserved. '
        'Read ../../TODO.md and ../../docs/export-fidelity-and-staging.md before continuing.\n')
    write_json(out/'manifest.json', {'schema_version': 1, 'status': 'experimental_execution_disabled',
        'origins': entries, 'origin_count': len(entries), 'unique_snapshots': len(blobs),
        'snapshot_bytes': total, 'package_source_sha256': sha256(out/'package.py'),
        'readme_sha256': sha256(out/'README.md'), 'large_model_scene_remote_backup': False,
        'quality_approved': False, 'fresh_studio_accepted': False, 'new_paid_generation_calls': 0})
    assert all(sha256(ROOT/row['origin']) == row['sha256'] for row in entries)
    write_json(HERE/'receipt.json', {'bundle': str(out.relative_to(ROOT)),
        'manifest_sha256': sha256(out/'manifest.json'), 'origin_count': len(entries),
        'unique_snapshots': len(blobs), 'snapshot_bytes': total, 'all_origins_unchanged': True})
    print(json.dumps(json.loads((HERE/'receipt.json').read_text())))


if __name__ == '__main__':
    main()
