"""Read-only, bounded UV/resource admission for a preserved native finishing pack."""

import argparse
import json
from pathlib import Path
import re
import shutil

from avatar_assets.provider_model import preflight
from avatar_assets.runner import run_bounded
from avatar_assets.safety import (artifacts, local_file, new_directory, preserve_failure,
                                 read_json, require, sha256, write_json)
from scripts.blender_fur_cards import tuft_export_budget
from scripts.blender_fur_undercoat import validate_undercoat


ROOT = Path(__file__).resolve().parents[1]
POLICY = 'saved-undercoat-admission-v1'
LIMITS = {'wall_seconds': 120, 'rss_mb': 2048, 'output_mb': 64, 'threads': 4}
NATIVE_SOURCES = ('blender_undercoat_admission.py', 'blender_fur_undercoat.py',
                  'undercoat_uv_atlas.py', 'groom_flow.py')


def _file(path, limit):
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), 'Symlink input is unsupported')
    return local_file(path, limit)


def _pin(path, limit):
    path = _file(path, limit)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}


def _count(value, name, maximum=100000):
    require(type(value) is int and 0 <= value <= maximum, 'Invalid '+name)
    return value


def _inputs(pack, settings):
    root = Path(pack).absolute()
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)), 'Invalid pack directory')
    root = root.resolve()
    manifest_pin = _pin(root/'manifest.json', 1024*1024)
    manifest = read_json(root/'manifest.json')
    require(manifest.get('schema_version') == 1 and manifest.get('status') == 'needs_visual_review',
            'Require a completed native finishing pack; this does not imply appearance approval')
    rows = manifest.get('artifacts')
    require(isinstance(rows, list) and 1 <= len(rows) <= 4096, 'Invalid manifest inventory')
    inventory = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'path', 'sha256', 'bytes'}, 'Invalid artifact record')
        name = row['path']
        require(isinstance(name, str) and name and not Path(name).is_absolute() and
                '..' not in Path(name).parts and name not in inventory and
                type(row['bytes']) is int and 0 <= row['bytes'] <= 512*1024*1024 and
                isinstance(row['sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['sha256']),
                'Invalid or duplicate artifact identity')
        inventory[name] = row
    pins = {'manifest': manifest_pin}

    def bound(name, limit=1024*1024):
        require(name in inventory, 'Missing manifest binding: '+name)
        pin = _pin(root/name, limit)
        require(pin['sha256'] == inventory[name]['sha256'] and pin['bytes'] == inventory[name]['bytes'],
                'Manifest artifact changed: '+name)
        pins[name] = pin
        return read_json(root/name) if name.endswith('.json') else pin

    cfg, report, job = (bound(name) for name in ('finish-config.json', 'finish-report.json', 'job.json'))
    blend = bound('source.blend', 512*1024*1024)
    require(manifest.get('finish') == report and manifest.get('job') == job,
            'Embedded finishing report or job differs from its artifact')
    require(job.get('exit_code') == 0 and job.get('failure') is None and
            job.get('process_exit_verified') is True and not job.get('manual_stop_required'),
            'Native finishing did not exit successfully')
    source = _pin(root/cfg['source'], 64*1024*1024)
    require(source['sha256'] == cfg.get('source_sha256') == report.get('source_sha256') ==
            manifest.get('source_sha256') and Path(manifest.get('source_path', '')).resolve() == Path(source['path']),
            'Finishing source binding differs')
    pins['source_model'] = source
    facts = preflight(Path(source['path']))
    require(facts['sha256'] == source['sha256'] and facts['skins'] == (0 if cfg.get('authored_rig') else 1),
            'Source preflight or rig route differs')
    groom = report.get('groom')
    require(isinstance(groom, dict) and groom.get('kind') == 'editable_native_blender_particle_hair' and
            groom.get('native_only') is True and not cfg.get('material_fur') and
            not cfg.get('fur_undercoat') and not cfg.get('portable_fringe'),
            'Admission currently requires one copied native groom without an existing undercoat or fringe')
    for key in ('object', 'source_object', 'armature'):
        require(isinstance(groom.get(key), str) and 0 < len(groom[key]) <= 128, 'Invalid recorded groom object')
    require(groom['object'] != groom['source_object'], 'Groom and original body must be distinct')
    rest = report.get('deformation_measurement')
    require(isinstance(rest, dict) and rest.get('schema_version') == 1 and rest.get('status') == 'measured' and
            rest.get('coordinate_space') == 'mesh_local_after_world_transform_applied' and rest.get('units') == 'meters' and
            rest.get('rest_array_encoding') == 'little-endian float64 XYZ' and
            rest.get('edge_array_encoding') == 'little-endian int32 pairs' and
            rest.get('unchanged_topology_required') is True, 'Missing bound rest geometry')
    vertices = _count(rest.get('vertices'), 'rest vertices')
    require(vertices > 0 and _count(rest.get('edges'), 'rest edges', 300000) > 0, 'Empty rest geometry')
    require(all(isinstance(rest.get(k), str) and re.fullmatch('[0-9a-f]{64}', rest[k])
                for k in ('rest_sha256', 'edges_sha256')), 'Invalid rest geometry hashes')
    selected = _count(groom.get('density_selected_vertices'), 'selected density vertices')
    excluded = _count(groom.get('density_excluded_vertices'), 'excluded density vertices')
    require(selected > 0 and selected+excluded == vertices, 'Recorded density counts differ from rest mesh')
    require(groom.get('mask_config') == cfg.get('groom_exclusions') and
            groom.get('fur_max_chroma') == cfg.get('fur_max_chroma'), 'Recorded groom mask design differs')
    cards, expected_faces = 0, None
    portable = cfg.get('portable_fur')
    if portable:
        require(portable.get('settings', {}).get('profile') == 'tuft-v1',
                'This diagnostic only admits the recorded twelve-corner tuft budget')
        bands = portable.get('bands')
        require(isinstance(bands, list) and 1 <= len(bands) <= 8, 'Invalid portable bands')
        cards = sum(_count(b.get('count'), 'portable card count', 5000) for b in bands)
        budget = tuft_export_budget(facts['vertices'], cards)
        require(bound('tuft-export-budget.json') == budget == report.get('portable_fur_export_budget'),
                'Recorded portable export budget differs from source/card counts')
        coats = report.get('portable_fur')
        require(isinstance(coats, list) and len(coats) == len(bands) and
                all(c.get('cards') == b['count'] for c, b in zip(coats, bands)), 'Portable counts differ')
        # Only a full brightness band is the same face selection as the saved density mask.
        if len(bands) == 1 and bands[0].get('brightness') == [0, 1.01]:
            selection = coats[0].get('root_summary', {}).get('binding_selection', {})
            if selection.get('excluded_degenerate_triangles') == 0:
                expected_faces = _count(selection.get('eligible_triangles'), 'recorded fur faces')
    budget = {'source_export_vertices': facts['vertices'], 'portable_export_vertices': cards*12,
              'reserve_vertices': 4096}
    require(sum(budget.values()) <= 100000, 'Unchanged source/coat/reserve exceeds export budget')
    return {'schema_version': 1, 'policy': POLICY, 'pack': str(root), 'inputs': pins, 'settings': settings,
            'source_preflight': facts, 'export_vertex_budget': budget,
            'expected': {'body': groom['source_object'], 'groom': groom['object'], 'rig': groom['armature'],
                         'density_group': 'AuthoredFurDensity', 'density_selected_vertices': selected,
                         'density_excluded_vertices': excluded, 'selected_triangles': expected_faces,
                         'triangles': facts['triangles'], 'rest': rest},
            'scope': 'UV, mask and export-budget admission only; no bake, tangent, shader, render or quality approval'}


def _unchanged(pins):
    for pin in pins.values():
        path = _file(pin['path'], max(pin['bytes'], 1))
        require(path.stat().st_size == pin['bytes'] and sha256(path) == pin['sha256'], 'Diagnostic input changed')


def inspect(pack, settings, out):
    """Plan undercoat on saved raw mesh data; a measured planner refusal is a successful diagnostic."""
    settings = validate_undercoat(settings)
    require(settings.get('uv_policy') == 'material-adaptive-v1', 'Admission requires explicit adaptive UV policy')
    require(not Path(out).resolve().is_relative_to(Path(pack).resolve()), 'Diagnostic output cannot modify its pack')
    output = new_directory(out)
    try:
        prepared = _inputs(pack, settings)
        sources = output/'sources'
        sources.mkdir()
        for name in NATIVE_SOURCES:
            shutil.copyfile(ROOT/'scripts'/name, sources/name)
        for path in (Path(__file__), ROOT/'avatar_assets/provider_model.py', ROOT/'avatar_assets/three_d.py',
                     ROOT/'avatar_assets/runner.py', ROOT/'avatar_assets/safety.py', ROOT/'scripts/blender_fur_cards.py'):
            shutil.copyfile(path, sources/path.name)
        write_json(output/'settings.json', settings)
        write_json(output/'admission-inputs.json', prepared)
        write_json(output/'source-preflight.json', prepared['source_preflight'])
        for name in ('runtime-home', 'runtime-tmp'):
            (output/name).mkdir(mode=0o700)
        exe = _file(ROOT/'.tools/Blender.app/Contents/MacOS/Blender', 512*1024*1024)
        # env exec preserves the monitored PID while excluding parent credentials and Python hooks.
        environment = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(output/'runtime-home'),
                       'TMPDIR': str(output/'runtime-tmp'), 'PYTHONDONTWRITEBYTECODE': '1',
                       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
                       'BLENDER_USER_CONFIG': str(output/'blender-config'),
                       'BLENDER_USER_SCRIPTS': str(output/'blender-scripts'),
                       'BLENDER_USER_EXTENSIONS': str(output/'blender-extensions')}
        command = ['/usr/bin/env', '-i', *[key+'='+value for key, value in environment.items()], str(exe),
                   '--background', '--factory-startup', '--disable-autoexec', '--python-exit-code', '1',
                   '--threads', '4', '--python', str(sources/'blender_undercoat_admission.py'), '--',
                   str(output/'admission-inputs.json'), str(output/'native-admission.json')]
        source_pins = {p.name: _pin(p, 1024*1024) for p in sources.iterdir()}
        write_json(output/'prepared.json', {'policy': POLICY, 'command': command, 'limits': LIMITS,
                   'sources': source_pins, 'settings_sha256': sha256(output/'settings.json'),
                   'inputs_sha256': sha256(output/'admission-inputs.json'),
                   'blender': _pin(exe, 512*1024*1024), 'provider_calls': 0})
        job = run_bounded(command, output/'admission.log', output, timeout=120, memory_mb=2048, output_mb=64)
        write_json(output/'job.json', job)
        _unchanged(prepared['inputs'])
        _unchanged(source_pins)
        require(job['exit_code'] == 0 and not job['failure'] and job.get('process_exit_verified') is True,
                'Native admission failed; inspect preserved diagnostic')
        native = read_json(output/'native-admission.json')
        require(native.get('policy') == POLICY and native.get('status') in ('supported', 'refused') and
                native.get('inputs_sha256') == sha256(output/'admission-inputs.json') and
                native.get('settings') == settings and native.get('input_writes') is False,
                'Native admission result binding differs')
        result = {**native, 'out': str(output), 'inputs': prepared['inputs'], 'source_preflight': prepared['source_preflight'],
                  'job': job, 'inputs_unchanged': True, 'provider_calls': 0, 'quality_approved': False}
        write_json(output/'result.json', result)
        manifest = {'schema_version': 1, 'kind': 'read_only_undercoat_admission',
                    'status': result['status'], 'artifacts': artifacts(output)}
        manifest_bytes = len((json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False)+'\n').encode())
        require(sum(row['bytes'] for row in manifest['artifacts'])+manifest_bytes <= 64*1024*1024,
                'Admission output including final receipts exceeds 64 MiB')
        write_json(output/'manifest.json', manifest)
        return result
    except Exception as error:
        preserve_failure(output, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pack', type=Path)
    parser.add_argument('settings', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    result = inspect(args.pack, read_json(args.settings), args.out)
    print(result['status']+': '+str(args.out))


if __name__ == '__main__':
    main()
