"""Verify portable assets and existing consumer loading at the relocated root."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path('/Users/ivan/kintsugi/weft/tools/avatar-assets')
sys.path.insert(0, str(root))
sys.path.insert(0, '/Users/ivan/kintsugi/weft/src')
from avatar_assets.cli import verify_pack
from avatar_assets.gallery import _Catalog
from avatar_assets.safety import artifacts, new_directory, sha256, write_json
from weft.avatar_assets import import_avatar_model, load_imported_avatar

start = time.monotonic()
out = new_directory(root / 'outputs/weft-relocation-proof-v1')
selection = json.loads((root / 'scratch/weft-migration-v1/tracked-selection.json').read_text())
packages, other_folders = [], []
for name in selection['complete_packs']:
    path = root / 'outputs' / name
    manifest = path / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('artifacts'):
        result = verify_pack(path)
        packages.append({'pack': name, 'manifest_sha256': sha256(manifest), 'result': result})
    else:
        other_folders.append(name)
write_json(out / 'pack-verification.json', {'verified': packages, 'non_pack_evidence_folders': other_folders})

selected_files = set(subprocess.check_output(['git', '-C', str(root), 'ls-files', '--others', '--exclude-standard', '-z']).decode().split('\0'))
catalogs = []
for name in ('browser/gallery.json', 'browser/gallery-basic-v1.json'):
    catalog = _Catalog(root, name)
    missing = [resource.relative for resource in catalog.routes.values()
               if not resource.relative.startswith('browser/node_modules/') and resource.relative not in selected_files]
    assert not missing, missing
    catalogs.append({'catalog': name, 'sha256': sha256(root / name), 'assets': len(catalog.data['assets']),
                     'routes': len(catalog.routes), 'all_non_dependency_routes_in_git_selection': True})
write_json(out / 'catalog-portability.json', {'catalogs': catalogs, 'installed_dependency': 'browser/node_modules/@google/model-viewer'})

consumer = []
for name in ('weft-muse-panda-v1', 'weft-humanoid-robot-v1'):
    source = root / 'outputs' / name
    target = out / name
    sealed = import_avatar_model(source / 'model.glb', json.loads((source / 'mapping.json').read_text()), target)
    loaded = load_imported_avatar(target)
    consumer.append({'pack': name, 'model_sha256': sha256(source / 'model.glb'),
                     'mapping_sha256': sha256(source / 'mapping.json'), 'sealed': sealed,
                     'reloaded_source_manifest_sha256': loaded.source_manifest_sha256,
                     'target_runtime_alignment_verified': False})
write_json(out / 'consumer-loading.json', {'results': consumer,
    'implementation': {name: sha256(Path('/Users/ivan/kintsugi/weft/src/weft') / name)
                       for name in ('avatar_assets.py', 'avatar_gltf.py')},
    'physics_modified': False})

for name in ('python-tests.txt', 'session-move.json', 'copy-verification.json'):
    shutil.copy2(Path('/private/tmp/avatar-migration-v1') / name, out / name)
shutil.copy2('/Users/ivan/kintsugi/weft/.local/avatar-assets-migration-v1/runtime-relocation.json', out / 'runtime-relocation.json')
shutil.copy2(__file__, out / 'verify_relocation.py')
report = {'schema_version': 1, 'kind': 'weft_avatar_relocation_verification', 'status': 'passed',
          'elapsed_seconds': round(time.monotonic() - start, 3), 'packages_verified': len(packages),
          'catalogs_verified': 2, 'consumer_models_verified': 2, 'provider_calls': 0, 'provider_cost_usd': 0,
          'artifacts': artifacts(out)}
write_json(out / 'manifest.json', report)
print(json.dumps({key: value for key, value in report.items() if key != 'artifacts'}))
