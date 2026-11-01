"""Bounded local finishing and explicit creature rigs for generated geometry."""

import json
import math
from pathlib import Path
import shutil

from avatar_assets.assemble_generated import rig_contract
from avatar_assets.provider_model import preflight
from avatar_assets.runner import run_bounded
from avatar_assets.safety import artifacts, local_file, new_directory, preserve_failure, read_json, require, sha256, slug, write_json
from avatar_assets.three_d import _GLB, validate_glb, validate_render_attributes

ROOT = Path(__file__).resolve().parents[1]


def number(value, low, high):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            f'Number outside {low}–{high}')


def region(value):
    require(isinstance(value, dict) and set(value) == {'center', 'radius'}, 'Invalid spatial region')
    for key in ('center', 'radius'):
        require(isinstance(value[key], list) and len(value[key]) == 3, 'Region needs three coordinates')
        for v in value[key]:
            number(v, .001 if key == 'radius' else -1, 3)


def validate(config):
    required = {'schema_version', 'id', 'subject', 'style', 'rights', 'prompt', 'source', 'source_sha256',
                'normal_strength', 'resolution', 'samples', 'states'}
    optional = {'groom', 'groom_exclusions', 'fur_max_chroma', 'fur_clump', 'attachment', 'smoothing',
                'weight_constraints', 'portable_fur', 'authored_actions', 'portrait_denoise', 'native_render_children',
                'authored_rig', 'camera_view', 'surface_profiles', 'surface_smoothing', 'groom_shader', 'groom_flow',
                'color_corrections'}
    require(isinstance(config, dict) and required <= set(config) <= required|optional
            and config['schema_version'] == 1, 'Invalid generated finishing config')
    slug(config['id'])
    for field in ('subject', 'style', 'rights', 'prompt', 'source', 'source_sha256'):
        require(isinstance(config[field], str) and 0 < len(config[field]) <= 4000, 'Invalid '+field)
    for field, low, high in (('normal_strength', 0, 1), ('resolution', 128, 1600), ('samples', 1, 64)):
        number(config[field], low, high)
    require(type(config['resolution']) is int and type(config['samples']) is int, 'Integer render settings required')
    require(type(config.get('portrait_denoise', False)) is bool, 'Invalid denoising flag')
    require(config.get('camera_view', 'front') in ('front', 'three_quarter', 'three_quarter_left'), 'Invalid camera view')
    if 'surface_profiles' in config:
        from avatar_assets.surface_profiles import validate_profiles
        validate_profiles(config['surface_profiles'])
    if 'color_corrections' in config:
        from avatar_assets.color_corrections import validate_corrections
        validate_corrections(config['color_corrections'])
    if 'surface_smoothing' in config:
        from avatar_assets.surface_smoothing_config import validate_surface_smoothing
        require('smoothing' not in config, 'Choose one smoothing method')
        validate_surface_smoothing(config['surface_smoothing'])
    states = config['states']
    require(isinstance(states, list) and 1 <= len(states) <= 8, 'Require 1–8 named states')
    names = []
    for state in states:
        require(set(state) == {'id', 'meaning'} and isinstance(state['meaning'], str)
                and 0 < len(state['meaning']) <= 4000, 'Invalid state')
        names.append(slug(state['id']))
    require(len(set(names)) == len(names), 'Duplicate states')
    if 'authored_rig' in config:
        from scripts.creature_rig_math import validate_rig
        validate_rig(config['authored_rig'])
        require(config.get('authored_actions'), 'A new creature rig requires explicit authored actions')
    boxes = config.get('groom_exclusions', [])
    require(isinstance(boxes, list) and len(boxes) <= 16, 'Too many spatial exclusions')
    for box in boxes:
        region(box)
    number(config.get('fur_max_chroma', 1), 0, 1)
    number(config.get('fur_clump', .06), 0, 1)
    groom = config.get('groom')
    if 'groom_flow' in config:
        from scripts.groom_flow import validate_flow
        require(groom and 'direction' in groom and 'comb' in groom,
                'Regional groom flow requires native groom direction and comb')
        validate_flow(config['groom_flow'])
    if 'groom_shader' in config:
        shader = config['groom_shader']
        require(groom and isinstance(shader, dict) and set(shader) == {'roughness', 'radial_roughness'},
                'Groom shader requires native hair and two explicit roughness settings')
        for value in shader.values():
            number(value, .04, 1)
    if groom:
        fields = {'parents', 'children', 'length', 'radius', 'roughness', 'seed'}
        require(set(groom) in (fields, fields|{'direction', 'comb'}), 'Invalid groom fields')
        for k, low, high in (('parents', 16, 10000), ('children', 0, 30), ('seed', 0, 2**31-1)):
            require(type(groom[k]) is int, 'Integer groom setting required')
            number(groom[k], low, high)
        for k, low, high in (('length', .001, .04), ('radius', .00001, .001), ('roughness', 0, .01)):
            number(groom[k], low, high)
        require(groom['parents']*(groom['children']+1) <= 240000, 'Groom exceeds fiber limit')
        if 'direction' in groom:
            require(len(groom['direction']) == 3 and math.hypot(*groom['direction']) > 1e-8, 'Invalid groom direction')
            for v in groom['direction']:
                number(v, -1, 1)
            number(groom['comb'], 0, .95)
    if 'native_render_children' in config:
        value = config['native_render_children']
        require(groom and type(value) is int and 0 <= value <= 90
                and groom['parents']*(value+1) <= 600000, 'Dense native profile exceeds 600000 configured fibers')
    if config.get('smoothing'):
        smooth = config['smoothing']
        require(set(smooth) == {'iterations', 'strength'} and type(smooth['iterations']) is int, 'Invalid smoothing')
        number(smooth['iterations'], 1, 30)
        number(smooth['strength'], .01, .8)
    if config.get('weight_constraints'):
        weights = config['weight_constraints']
        require(set(weights) == {'fallback_bone', 'regions'} and isinstance(weights['fallback_bone'], str)
                and isinstance(weights['regions'], dict) and len(weights['regions']) <= 64, 'Invalid weight constraints')
        for value in weights['regions'].values():
            region(value)
    if config.get('attachment'):
        attach = config['attachment']
        require(set(attach) == {'bone', 'parent_bone', 'anchor_from_bone', 'dominant_channel', 'dominant_ratio',
                               'min_vertices', 'max_vertices', 'regions', 'expand_meters'}, 'Invalid attachment')
        require(all(isinstance(attach[k], str) and 0 < len(attach[k]) <= 64
                    for k in ('bone', 'parent_bone', 'anchor_from_bone')), 'Invalid attachment bone')
        require(type(attach['dominant_channel']) is int and attach['dominant_channel'] in (0, 1, 2), 'Invalid color channel')
        number(attach['dominant_ratio'], 1, 5)
        number(attach['expand_meters'], 0, .05)
        require(type(attach['min_vertices']) is int and type(attach['max_vertices']) is int
                and 1 <= attach['min_vertices'] <= attach['max_vertices'] <= 30000, 'Invalid attachment bounds')
        require(1 <= len(attach['regions']) <= 8, 'Invalid attachment region count')
        for value in attach['regions']:
            region(value)
    if config.get('portable_fur'):
        fur = config['portable_fur']
        require(set(fur) == {'settings', 'bands'} and 1 <= len(fur['bands']) <= 4, 'Invalid portable fur')
        settings = fur['settings']
        fields = {'seed', 'length', 'width', 'direction'}
        require(fields <= set(settings) <= fields|{'profile', 'sample_root_color'}, 'Invalid card settings')
        require(settings.get('profile', 'legacy') in ('legacy', 'combed-v1'), 'Unknown card geometry profile')
        require(type(settings.get('sample_root_color', False)) is bool, 'Invalid card color flag')
        require(type(settings['seed']) is int and 0 <= settings['seed'] <= 2**31-5, 'Invalid card seed')
        number(settings['length'], .001, .04)
        number(settings['width'], .0003, .02)
        require(len(settings['direction']) == 3, 'Invalid card direction')
        for v in settings['direction']:
            number(v, -1, 1)
        total = 0
        for band in fur['bands']:
            require(set(band) == {'brightness', 'color', 'count'} and len(band['brightness']) == 2
                    and len(band['color']) == 4 and type(band['count']) is int, 'Invalid fur band')
            require(0 <= band['brightness'][0] < band['brightness'][1] <= 1.01, 'Invalid fur brightness')
            for v in band['color']:
                number(v, 0, 1)
            number(band['count'], 1, 5000)
            total += band['count']
        require(total <= 5000, 'Portable coat exceeds 30000 vertices')
    if config.get('authored_actions'):
        actions = config['authored_actions']
        require(isinstance(actions, list) and [a['id'] for a in actions] == names, 'Authored actions must match named states')
        for action in actions:
            require(set(action) == {'id', 'keys'} and 2 <= len(action['keys']) <= 32, 'Invalid authored action')
            previous = -1
            for key in action['keys']:
                require(set(key) == {'second', 'rotations'} and isinstance(key['rotations'], dict)
                        and len(key['rotations']) <= 64, 'Invalid action key')
                number(key['second'], 0, 10)
                require(key['second'] > previous, 'Action keys must increase')
                previous = key['second']
                for rotation in key['rotations'].values():
                    require(len(rotation) == 3, 'Rotation needs three angles')
                    for v in rotation:
                        number(v, -60, 60)
                if config.get('authored_rig'):
                    require(set(key['rotations']) <= {b['name'] for b in config['authored_rig']['bones']},
                            'Action names a bone absent from the authored rig')
    return config


def reuse_native(folder, out, cfg):
    root = Path(folder).resolve()
    manifest = read_json(root/'manifest.json')
    require(manifest.get('artifacts') == artifacts(root), 'Native attempt changed since its manifest')
    require(read_json(root/'finish-config.json') == cfg, 'Native config differs; rebuild required')
    job = read_json(root/'job.json')
    require(job['exit_code'] == 0 and not job['failure'], 'Cannot reuse a failed native process')
    for name in ('blender_finish_generated.py', 'blender_native_groom.py', 'blender_comb_groom.py',
                 'blender_verify.py', 'blender_fur_cards.py', 'blender_creature_rig.py', 'creature_rig_math.py',
                 'motion_samples.py', 'deformation_diagnostics.py', 'blender_surface_profiles.py', 'surface_smoothing.py', 'surface_integrity.py', 'groom_flow.py', 'blender_color_corrections.py'):
        require(sha256(root/'sources'/name) == sha256(ROOT/'scripts'/name), 'Native backend changed; rebuild required')
    require(sha256(root/'sources/surface_profiles.py') == sha256(ROOT/'avatar_assets/surface_profiles.py'),
            'Surface profile validator changed; rebuild required')
    require(sha256(root/'sources/surface_smoothing_config.py') == sha256(ROOT/'avatar_assets/surface_smoothing_config.py'),
            'Surface smoothing validator changed; rebuild required')
    require(sha256(root/'sources/color_corrections.py') == sha256(ROOT/'avatar_assets/color_corrections.py'),
            'Color correction validator changed; rebuild required')
    for name in ('source.blend', 'portrait.png', 'finish-report.json', 'job.json', 'finish.log', 'finish-monitor.json', 'finish-handle.json'):
        shutil.copyfile(root/name, out/name)
    if (root/'surface-integrity.json').exists():
        shutil.copyfile(root/'surface-integrity.json', out/'surface-integrity.json')
    for name in ('renders', 'textures'):
        if (root/name).exists():
            shutil.copytree(root/name, out/name)
    original = root/('model-native-export.glb' if (root/'model-native-export.glb').exists() else 'model.glb')
    shutil.copyfile(original, out/'model.glb')
    write_json(out/'native-reuse.json', {'source': str(root), 'source_manifest_sha256': sha256(root/'manifest.json'),
               'native_execution_repeated': False, 'source_export_sha256': sha256(original),
               'reason': 'Same hashed design/native backend; repeat only changed local postprocessing'})
    return job


def finish(config_path, output, reuse=None):
    path = local_file(config_path)
    cfg = validate(read_json(path))
    source = local_file(path.parent/cfg['source'], 64*1024*1024)
    require(sha256(source) == cfg['source_sha256'], 'Finishing source changed')
    facts = preflight(source)
    require(facts['skins'] == (0 if cfg.get('authored_rig') else 1),
            'Finishing requires one skin, or unrigged geometry with an explicit authored_rig')
    # Curved ribbons can split at tangent discontinuities: budget all twelve triangle corners.
    cards = sum(b['count']*12 for b in cfg.get('portable_fur', {}).get('bands', []))
    require(facts['vertices']+cards <= 100000, 'Base plus portable fur exceeds vertex budget')
    out = new_directory(output)
    try:
        (out/'sources').mkdir()
        shutil.copyfile(path, out/'sources/finish-design.json')
        cfg.update(source=str(source), export_model=True, sample_states=True)
        write_json(out/'finish-config.json', cfg)
        for name in ('blender_finish_generated.py', 'blender_native_groom.py', 'blender_comb_groom.py',
                     'blender_verify.py', 'blender_fur_cards.py', 'blender_creature_rig.py', 'creature_rig_math.py',
                     'motion_samples.py', 'deformation_diagnostics.py', 'blender_surface_profiles.py', 'surface_smoothing.py', 'surface_integrity.py', 'groom_flow.py', 'blender_color_corrections.py'):
            shutil.copyfile(ROOT/'scripts'/name, out/'sources'/name)
        shutil.copyfile(__file__, out/'sources/finish_generated.py')
        shutil.copyfile(ROOT/'avatar_assets/surface_profiles.py', out/'sources/surface_profiles.py')
        shutil.copyfile(ROOT/'avatar_assets/surface_smoothing_config.py', out/'sources/surface_smoothing_config.py')
        shutil.copyfile(ROOT/'avatar_assets/color_corrections.py', out/'sources/color_corrections.py')
        shutil.copyfile(ROOT/'avatar_assets/generated_tangents.py', out/'sources/generated_tangents.py')
        exe = ROOT/'.tools/Blender.app/Contents/MacOS/Blender'
        cmd = [str(exe), '--background', '--factory-startup', '--disable-autoexec', '--python-exit-code', '1',
               '--threads', '4', '--python', str(out/'sources/blender_finish_generated.py'), '--',
               str(out/'finish-config.json'), str(out)]
        if reuse:
            job = reuse_native(reuse, out, cfg)
        else:
            job = run_bounded(cmd, out/'finish.log', out, timeout=600, memory_mb=2048, output_mb=512)
            write_json(out/'job.json', job)
        require(job['exit_code'] == 0 and not job['failure'], 'Finishing failed; inspect preserved attempt')
        report = read_json(out/'finish-report.json')
        from avatar_assets.generated_tangents import repair
        native_export = out/'model-native-export.glb'
        (out/'model.glb').rename(native_export)
        tangent_report = repair(native_export, out/'model.glb')
        write_json(out/'tangent-repair.json', tangent_report)
        doc = _GLB(out/'model.glb').doc
        contract = rig_contract(doc, authored=bool(cfg.get('authored_rig')))
        write_json(out/'rig-contract.json', contract)
        rig = validate_glb(out/'model.glb', contract)
        attributes = validate_render_attributes(out/'model.glb')
        # Rejected motion still needs its exact per-state measurements for a bounded repair.
        write_json(out/'rig-verification.json', rig)
        write_json(out/'render-attributes.json', attributes)
        require(rig['motion']['animation_verified'], 'Finished source has no verified skin motion')
        durations = {s['id']: s['duration_seconds'] for s in report['states']}
        require(set(durations) == {s['id'] for s in cfg['states']}, 'Exported state names differ')
        parents = {child: i for i, n in enumerate(doc['nodes']) for child in n.get('children', [])}
        write_json(out/'exported-nodes.json', {'nodes': [{'id': i, 'parent_id': parents.get(i), **n} for i, n in enumerate(doc['nodes'])],
                   'skins': doc['skins'], 'local_transform_units': 'Respect complete hierarchy and inverse bind matrices; root scale may convert centimeters to meters',
                   'attachment': report.get('attachment'), 'weft_runtime_verified': False})
        result = {'schema_version': 1, 'kind': 'generated_3d_avatar', 'status': 'needs_visual_review',
                  'id': cfg['id'], 'subject': cfg['subject'], 'style': cfg['style'], 'rights': cfg['rights'],
                  'source_path': str(source), 'source_sha256': cfg['source_sha256'],
                  'states': [{**s, 'duration_seconds': durations[s['id']], 'loop_verified': False} for s in cfg['states']],
                  'rig_verification': rig, 'finish': report, 'job': job, 'has_transparency': False,
                  'reduced_motion': 'portrait.png and renders/STATE/0000.png; no autoplay',
                  'representations': {'native': 'Editable hair if configured', 'browser': 'Skinned alpha-masked cards if configured; not native strands'},
                  'quality_equivalence_verified': False, 'weft_runtime_verified': False, 'new_provider_calls': 0,
                  'artifacts': artifacts(out)}
        write_json(out/'manifest.json', result)
        return {'out': str(out), 'status': result['status'], 'vertices': rig['mesh_vertices'], 'states': len(durations)}
    except Exception as error:
        preserve_failure(out, error)
        raise


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('config')
    p.add_argument('--out', required=True)
    p.add_argument('--reuse-native', help='Reuse only an unchanged, hash-verified successful native pass; rerun postprocessing')
    args = p.parse_args()
    print(json.dumps(finish(args.config, args.out, args.reuse_native), indent=2))
