"""Prepare hash-bound exported previews; native execution is a separate explicit step."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import struct
import sys
import traceback


_FILE = Path(__file__).resolve()
ROOT = _FILE.parents[3] if _FILE.parent.name == 'sources' else _FILE.parents[2]
RESOLUTION, SAMPLES, LIMIT = 720, 16, 128*1024**2
PLANS = {
    'panda': {
        'output': 'outputs/panda-fringe-exported-720-v1',
        'model_expected': 'outputs/panda-portable-fringe-finish-v1/model.glb',
        'recipe': 'examples/meshy/panda-portable-fringe-v1.json',
        'recipe_sha256': '72f112ab66406b52a6ef8d39f2b625e7d20f1a52de3effb71d09b94da3de8da7',
        'reference_preview': 'outputs/panda-fringe-packed-eevee-preview-v1',
        'reference_model_name': 'original.glb',
        'origin': 'Previously generated panda geometry with authored skin, motion, companion separation and local fringe finishing; no new generation for this preview.',
        'states': [{'id': 'dance_flight', 'seconds': [.36, 1.36, 2.36, 3.36]}],
        'review_focus': ['Exposed shoulder/cheek continuity while companion departs', 'No fringe on dragon, face details or authored closure',
                         'Light and dark source markings retained', 'Root attachment during alternating hops and companion return',
                         'Finite geometry, no obvious spikes, full subject and airborne companion inside framing']},
    'fox': {
        'output': 'outputs/fox-fringe-exported-720-v1',
        'model_expected': None,
        'recipe': 'examples/scenes/fox-hd-v3.json',
        'recipe_sha256': None,
        'reference_preview': 'outputs/fox-hd-build-v2',
        'reference_model_name': 'model.glb',
        'origin': 'Existing agent-authored procedural fox character; local fringe component derivative only, not a fresh generated avatar or full heterogeneous native-groom finish.',
        'states': [{'id': 'idle', 'seconds': [0, 1, 2, 3]},
                   {'id': 'making_something', 'seconds': [.5, 1, 1.5, 2.5]},
                   {'id': 'milestone_level_up', 'seconds': [.75, 1.5, 2, 3.25]},
                   {'id': 'working', 'seconds': [0, 1, 2, 3]}],
        'review_focus': ['Orange, cream and dark materials preserve source colors', 'No fringe on eyes, nose, inner ears, scarf, book or pencil',
                         'No bald material seams or conspicuous layered strips', 'Roots follow body, tail and limbs through all four existing actions',
                         'Book/pencil and complete fox stay in frame; drawing, celebration and reading remain distinguishable']}}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def local(path):
    path = Path(path)
    path = path if path.is_absolute() else ROOT/path
    require(path.is_file() and not path.is_symlink() and path.absolute() == path.resolve()
            and path.is_relative_to(ROOT), 'Require an existing nonsymlink workspace file')
    return path


def inspect_glb(path):
    from PIL import Image
    require(28 <= path.stat().st_size <= 64*1024**2, 'Model exceeds 64MiB')
    data = path.read_bytes()
    magic, version, total, json_size, kind = struct.unpack_from('<4sIIII', data)
    require(magic == b'glTF' and version == 2 and total == len(data) and kind == 0x4E4F534A
            and json_size <= 2*1024**2, 'Invalid bounded GLB')
    doc = json.loads(data[20:20+json_size])
    length, kind = struct.unpack_from('<II', data, 20+json_size)
    require(kind == 0x004E4942 and len(data) == 28+json_size+length, 'Embedded BIN required')
    binary = memoryview(data)[28+json_size:]
    require(not any('uri' in row for row in doc.get('buffers', [])+doc.get('images', [])), 'Remote/data URLs are unsupported')
    ranges = {}
    for action in doc.get('animations', []):
        name = action.get('name')
        require(isinstance(name, str) and name not in ranges, 'Unique named actions required')
        tracks = []
        for sampler in action['samplers']:
            a = doc['accessors'][sampler['input']]
            require(a['componentType'] == 5126 and a['type'] == 'SCALAR' and not a.get('sparse')
                    and 1 <= a['count'] <= 10000, 'Unsupported bounded action times')
            v = doc['bufferViews'][a['bufferView']]
            offset, stride = v.get('byteOffset', 0)+a.get('byteOffset', 0), v.get('byteStride', 4)
            values = [struct.unpack_from('<f', binary, offset+i*stride)[0] for i in range(a['count'])]
            require(all(math.isfinite(t) and 0 <= t <= 30 for t in values) and values == sorted(set(values)), 'Invalid action time range')
            tracks.append({'first': values[0], 'last': values[-1], 'keys': len(values), 'interpolation': sampler.get('interpolation', 'LINEAR')})
        require(tracks, 'Action has no actual samplers')
        ranges[name] = {'first': min(t['first'] for t in tracks), 'last': max(t['last'] for t in tracks), 'tracks': tracks}
    textures = []
    for image in doc.get('images', []):
        v = doc['bufferViews'][image['bufferView']]
        blob = binary[v.get('byteOffset', 0):v.get('byteOffset', 0)+v['byteLength']]
        with Image.open(io.BytesIO(blob)) as im:
            require(im.width*im.height <= 16_777_216, 'Texture exceeds16MP')
            textures.append({'width': im.width, 'height': im.height, 'encoded_bytes': v['byteLength'], 'format': im.format})
    return {'sha256': digest(path), 'bytes': len(data), 'actions': ranges, 'textures': textures,
            'vertices': sum(doc['accessors'][p['attributes']['POSITION']]['count'] for m in doc['meshes'] for p in m['primitives']),
            'triangles': sum(doc['accessors'][p['indices']]['count']//3 for m in doc['meshes'] for p in m['primitives']),
            'skins': len(doc.get('skins', [])), 'image_header_inspection_only': True}


def prepare(key, model, expected_sha, evidence):
    plan = PLANS[key]
    model = local(model)
    require(digest(model) == expected_sha, 'Final model hash mismatch')
    recipe = local(plan['recipe'])
    if plan['recipe_sha256']:
        require(digest(recipe) == plan['recipe_sha256'], 'Prepared source recipe changed')
    facts = inspect_glb(model)
    require(set(facts['actions']) == {s['id'] for s in plan['states']}, 'Actual final action names differ from requested preview')
    for state in plan['states']:
        bounds = facts['actions'][state['id']]
        require(bounds['first'] <= min(state['seconds']) and max(state['seconds']) <= bounds['last'], 'Preview times outside actual exported action')
    reference = ROOT/plan['reference_preview']
    old_model, old_blend = reference/plan['reference_model_name'], reference/'source.blend'
    image_count = 4+sum(len(s['seconds']) for s in plan['states'])
    png_allowance = image_count*(math.ceil((720*3+1)*720*1.01)+65536)
    blend_estimate = math.ceil(old_blend.stat().st_size*1.25+max(0, facts['bytes']-old_model.stat().st_size)*1.5)
    overhead = 18*1024**2
    estimate = facts['bytes']+png_allowance+blend_estimate+overhead
    budget = {'limit_bytes': LIMIT, 'copied_model_bytes': facts['bytes'], 'image_count': image_count,
              'png_allowance_bytes': png_allowance, 'editable_blend_estimate_bytes': blend_estimate,
              'log_and_metadata_reserve_bytes': overhead, 'estimated_total_bytes': estimate,
              'estimated_headroom_bytes': LIMIT-estimate,
              'method': '25% over an existing editable file, plus1.5× positive model growth; PNG raw-scanline allowance and18MiB logs/metadata.',
              'reference_model': {'path': str(old_model.relative_to(ROOT)), 'bytes': old_model.stat().st_size, 'sha256': digest(old_model)},
              'reference_blend': {'path': str(old_blend.relative_to(ROOT)), 'bytes': old_blend.stat().st_size, 'sha256': digest(old_blend)},
              'not_a_rigorous_blend_upper_bound': True,
              'enforcement': 'Runtime monitor stops at128MiB; no bound increase or automatic retry.'}
    out = ROOT/plan['output']
    out.mkdir()
    write(out/'budget-estimate.json', budget)
    if estimate > LIMIT:
        write(out/'unsupported.json', {'status':'budget_rejected_not_executed','budget':budget,
            'reason':'Practical output estimate exceeds the fixed 128 MiB limit; no native work started.'})
    require(estimate <= LIMIT, 'Practical output estimate exceeds128MiB; preview not prepared for execution')
    (out/'sources').mkdir()
    shutil.copyfile(model, out/'original.glb')
    require(digest(out/'original.glb') == expected_sha, 'Copied model bytes changed')
    sources = []
    for path in [ROOT/'scripts/blender_provider_preview.py', ROOT/'scripts/blender_verify.py', ROOT/'avatar_assets/runner.py',
                 ROOT/'avatar_assets/safety.py', Path(__file__).resolve()]:
        target = out/'sources'/path.name
        shutil.copyfile(path, target)
        require(digest(path) == digest(target), 'Source changed while snapshotting')
        sources.append({'path': str(path.relative_to(ROOT)), 'snapshot': str(target.relative_to(out)), 'sha256': digest(target)})
    provenance = [{'path': str(recipe.relative_to(ROOT)), 'sha256': digest(recipe)}]
    for value in evidence:
        path = local(value)
        provenance.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path)})
    require(len(provenance) >= 2, 'Pin at least one final validation/provenance receipt')
    write(out/'model-inventory.json', facts)
    write(out/'motion-states.json', plan['states'])
    command = [str(ROOT/'.tools/Blender.app/Contents/MacOS/Blender'), '--background', '--factory-startup', '--disable-autoexec',
               '--python-exit-code', '1', '--threads', '4', '--python', str(out/'sources/blender_provider_preview.py'), '--',
               str(out/'original.glb'), str(out), '720', '16', 'false', str(out/'motion-states.json')]
    write(out/'prepared.json', {'schema_version': 1, 'status': 'prepared_execution_requires_separate_release',
        'source_model': {'path': str(model.relative_to(ROOT)), 'sha256': expected_sha}, 'origin': plan['origin'],
        'sources': sources, 'provenance': provenance, 'command': command, 'budget': budget,
        'motion_states_sha256': digest(out/'motion-states.json'), 'resolution': [720, 720], 'samples':16,
        'limits': {'wall_seconds':120, 'rss_mb':2048, 'threads':4, 'output_mb':128},
        'expected_image_count': image_count, 'provider_calls':0, 'quality_approved':False,
        'review_focus': plan['review_focus'], 'no_loop_claim':True})
    write(out/'review-template.json', {'status':'not_reviewed', 'quality_approved':False, 'muse_parity':False,
        'view_checks': {v: {'identity':'not_reviewed','coat_and_color':'not_reviewed','seams_and_accessories':'not_reviewed',
                            'framing':'not_reviewed','observations':[]} for v in ('front','three-quarter','back','side')},
        'state_checks': {s['id']: {'times_seconds':s['seconds'],'state_meaning':'not_reviewed','root_attachment':'not_reviewed',
                                  'deformation':'not_reviewed','framing':'not_reviewed','observations':[]} for s in plan['states']},
        'focus': plan['review_focus'], 'limits':['Four poses per clip are sampled evidence, not continuous motion/physics proof.',
            'EEVEE exported representation is not native hair or browser parity.']})
    print(json.dumps({'prepared':str((out/'prepared.json').relative_to(ROOT)), 'sha256':digest(out/'prepared.json'),
                      'status':'held_until_explicit_native_release','budget':budget}))


def execute(output, expected_prepared_sha):
    from PIL import Image
    sys.path.insert(0, str(ROOT))
    from avatar_assets.runner import run_bounded
    out = ROOT/output
    require(out.resolve() == out.absolute() and out.is_relative_to(ROOT/'outputs'), 'Invalid prepared output path')
    require(digest(out/'prepared.json') == expected_prepared_sha, 'Prepared execution hash changed')
    prepared = json.loads((out/'prepared.json').read_text())
    for item in prepared['sources']:
        require(digest(out/item['snapshot']) == item['sha256'], 'Prepared renderer snapshot changed')
    for item in prepared['provenance']:
        require(digest(ROOT/item['path']) == item['sha256'], 'Prepared provenance changed')
    for path in (ROOT/prepared['source_model']['path'], out/'original.glb'):
        require(digest(path) == prepared['source_model']['sha256'], 'Prepared source model changed')
    require(digest(out/'motion-states.json') == prepared['motion_states_sha256'], 'Motion request changed')
    job = run_bounded(prepared['command'], out/'preview.log', out, timeout=120, memory_mb=2048, output_mb=128)
    write(out/'job.json', job)
    passed = job['exit_code'] == 0 and not job['failure'] and job.get('process_exit_verified') is True
    images = []
    if passed:
        try:
            report = json.loads((out/'native-inspection.json').read_text())
            require(report['source_sha256'] == prepared['source_model']['sha256'] and report['source_unchanged'], 'Renderer source binding failed')
            states = json.loads((out/'motion-states.json').read_text())
            policy = report['render_policy']
            require(report['render_engine'] == 'BLENDER_EEVEE_NEXT' and policy['samples_actual'] == 16
                    and policy['engine_fallback'] is False, 'Export preview engine/sample policy changed')
            actual_states = [{'id':s['id'],'seconds':[p['second'] for p in s['poses']]} for s in report['motion_states']]
            require(actual_states == states, 'Actual rendered clip/times differ from prepared motion request')
            require([r['view'] for r in report['renders']] == ['front','three-quarter','back','side'], 'Missing rest-view evidence')
            expected = [out/(v+'.png') for v in ('front','three-quarter','back','side')]
            expected += [out/'renders'/s['id']/f'{i:04d}.png' for s in states for i in range(len(s['seconds']))]
            require(len(expected) == prepared['expected_image_count'], 'Image count differs')
            for path in expected:
                with Image.open(path) as image:
                    require(image.size == (720,720) and image.mode == 'RGB', 'Preview dimensions/opaque RGB changed')
                    image.verify()
                images.append({'path':str(path.relative_to(out)),'sha256':digest(path),'dimensions':[720,720]})
            require((out/'source.blend').is_file(), 'Editable preview missing')
        except Exception:
            passed = False
            (out/'verification-error.txt').write_text(traceback.format_exc())
    write(out/'manifest.json', {'schema_version':1,'status':'needs_visual_review' if passed else 'failed',
        'prepared_sha256':expected_prepared_sha,'source_model':prepared['source_model'],'images':images,
        'artifacts':[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':digest(p)}
                     for p in sorted(out.rglob('*')) if p.is_file()],
        'quality_approved':False,'provider_calls':0,'loop_claimed':False})
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('key',choices=PLANS);prep.add_argument('--model',required=True)
    prep.add_argument('--sha256',required=True);prep.add_argument('--evidence',action='append',required=True)
    run = sub.add_parser('execute');run.add_argument('output');run.add_argument('--prepared-sha256',required=True)
    args=parser.parse_args()
    if args.mode == 'prepare':prepare(args.key,args.model,args.sha256,args.evidence)
    else:raise SystemExit(execute(args.output,args.prepared_sha256))
