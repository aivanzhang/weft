"""Read-only custody of two independent Studio deliveries; no model import or dispatch."""

import argparse
from contextlib import closing
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from avatar_assets.safety import require, write_json
from avatar_assets.studio_jobs import _digest
from scripts.verify_studio_live import parse_json, regular

LIMITS = {'seconds': 60, 'files': 8192, 'bytes': 4*1024**3,
          'file_bytes': 512*1024**2, 'json_bytes': 1024**2, 'glb_bytes': 64*1024**2}
CLAIMS = {'separate_delivered_files': False, 'combined_timeline_verified': False,
          'attachment_relationship_verified': False, 'physics_verified': False,
          'loop_verified': False, 'automatic_mesh_split_verified': False,
          'human_approval': False}


class Reader:
    def __init__(self, root):
        self.root, self.started, self.total = Path(root), time.monotonic(), 0
        require(self.root == self.root.resolve(), 'Workspace must be canonical')
        self.pins = {}

    def check(self):
        require(time.monotonic()-self.started <= LIMITS['seconds'], 'Inspection deadline exceeded')

    def path(self, base, name):
        require(isinstance(name, str) and name and not Path(name).is_absolute()
                and all(p not in ('', '.', '..') for p in name.split('/')), 'Unsafe artifact path')
        path = base/name
        require(path == path.resolve() and path.is_relative_to(self.root), 'Linked or escaped artifact')
        return path

    def pin(self, path, maximum=None):
        self.check()
        maximum = LIMITS['file_bytes'] if maximum is None else maximum
        info = regular(path, maximum)
        key = str(path.relative_to(self.root))
        identity = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
        if key in self.pins:
            require(self.pins[key][1] == identity, 'Inspected artifact changed')
            return self.pins[key][0]
        require(len(self.pins) < LIMITS['files'] and self.total+info.st_size <= LIMITS['bytes'],
                'Aggregate inspection bound exceeded')
        digest, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            while block := stream.read(1024**2):
                self.check()
                count += len(block)
                require(count <= maximum, 'File grew beyond its inspection bound')
                digest.update(block)
        after = path.stat()
        require(count == info.st_size and identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
                'Artifact changed while hashing')
        row = {'path': key, 'bytes': info.st_size, 'sha256': digest.hexdigest()}
        self.total += info.st_size
        self.pins[key] = (row, identity)
        return row

    def load(self, path):
        self.pin(path, LIMITS['json_bytes'])
        value = parse_json(path.read_bytes())
        self.pin(path, LIMITS['json_bytes'])
        return value

    def expect(self, path, checksum, maximum=None):
        require(isinstance(checksum, str) and re.fullmatch(r'[a-f0-9]{64}', checksum), 'Invalid artifact hash')
        require(self.pin(path, maximum)['sha256'] == checksum, 'Artifact hash differs: '+path.name)

    def inventory(self, folder, rows):
        require(isinstance(rows, list) and 0 < len(rows) <= LIMITS['files'], 'Invalid artifact inventory')
        names = set()
        for row in rows:
            require(isinstance(row, dict) and set(row) == {'path', 'bytes', 'sha256'}, 'Invalid artifact pin')
            name = row['path']
            require(isinstance(name, str) and name not in names, 'Duplicate artifact pin')
            names.add(name)
            pin = self.pin(self.path(folder, name))
            require(type(row['bytes']) is int and row['bytes'] == pin['bytes']
                    and row['sha256'] == pin['sha256'], 'Artifact inventory differs: '+name)
        return names

    def pack(self, folder, checksum=None):
        manifest_path = folder/'manifest.json'
        if checksum is not None:
            self.expect(manifest_path, checksum, LIMITS['json_bytes'])
        manifest = self.load(manifest_path)
        require(manifest.get('schema_version') == 1, 'Unsupported pack schema')
        names = self.inventory(folder, manifest.get('artifacts')) | {'manifest.json'}
        actual, pending, entries = set(), [folder], 0
        while pending:
            self.check()
            with os.scandir(pending.pop()) as children:
                for entry in children:
                    entries += 1
                    require(entries <= 2*LIMITS['files'] and not entry.is_symlink(), 'Pack tree exceeds bound or contains links')
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
                    else:
                        require(entry.is_file(follow_symlinks=False), 'Pack contains a nonregular artifact')
                        actual.add(Path(entry.path).relative_to(folder).as_posix())
        require(actual == names, 'Pack has missing or unsealed artifacts')
        return manifest


def _registry(reader, scene_id):
    require(isinstance(scene_id, str) and re.fullmatch(r'[a-f0-9]{24}', scene_id), 'Invalid scene ID')
    binding = reader.load(reader.root/'.local/studio-registry.json')
    require(set(binding) == {'schema_version', 'state_dir'} and binding['schema_version'] == 1,
            'Invalid canonical registry binding')
    state = Path(binding['state_dir'])
    require(state == state.resolve() and state.is_relative_to(reader.root/'.local'), 'Registry escapes workspace')
    database = state/'jobs.sqlite3'
    regular(database, 64*1024**2)
    with closing(sqlite3.connect(database.as_uri()+'?mode=ro', uri=True, timeout=1)) as db:
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        db.execute('BEGIN')
        size = db.execute('SELECT length(CAST(request AS BLOB)) FROM scenes WHERE id=?', (scene_id,)).fetchone()
        require(size is not None and size[0] <= LIMITS['json_bytes'], 'Missing or oversized scene')
        scene = dict(db.execute('SELECT id,request,request_digest FROM scenes WHERE id=?', (scene_id,)).fetchone())
        bound = list(db.execute('SELECT asset,job FROM scene_assets WHERE scene=? LIMIT 4', (scene_id,)))
        require(len(bound) <= 2 and len({r['job'] for r in bound}) == len(bound), 'Scene children are not independent')
        rows = {}
        for item in bound:
            require(re.fullmatch(r'[a-f0-9]{24}', item['job']) is not None, 'Invalid child job ID')
            sizes = db.execute('SELECT length(CAST(request AS BLOB)),length(CAST(data AS BLOB)) FROM jobs WHERE id=?', (item['job'],)).fetchone()
            require(sizes is not None and all(type(v) is int and v <= LIMITS['json_bytes'] for v in sizes),
                    'Missing or oversized child row')
            row = dict(db.execute('SELECT id,request,request_digest,data,output,status FROM jobs WHERE id=?', (item['job'],)).fetchone())
            row['active'] = db.execute('SELECT 1 FROM active WHERE job=?', (item['job'],)).fetchone() is not None
            rows[item['asset']] = row
    return state, scene, rows


def _prepared(reader, state, row, request):
    from avatar_assets.studio_worker import _validate_prepared_request
    folder = reader.root/'outputs'/row['output']
    prepared = reader.load(folder/'request.json')
    _validate_prepared_request(prepared, request, folder, agent_backend=prepared.get('agent_backend'))
    request_sha = reader.pin(folder/'request.json')['sha256']
    require(reader.load(state/'jobs'/row['id']/'pipeline-request.json') == {
        'schema_version': 1, 'studio_request_digest': row['request_digest'], 'pipeline_request_sha256': request_sha},
        'Pipeline request is not bound to Studio')
    receipt = reader.load(folder/'input-receipt.json')
    require(set(receipt) == {'request_sha256', 'sources'} and receipt['request_sha256'] == request_sha,
            'Input receipt differs')
    reader.inventory(folder/'sources', receipt['sources'])
    if prepared['reference'] is not None:
        ref = prepared['reference']
        reader.expect(reader.path(folder, ref['path']), ref['sha256'], 8*1024**2)
    return folder, prepared, request_sha


def _source(reader, folder, prepared, plan, expected):
    # Scene creation currently admits fresh independent sources, never a claimed split of one GLB.
    require(prepared['reuse_model'] is None, 'Scene source reuse is not supported by this verifier')
    config = reader.load(folder/'generation-config.json')
    generation = reader.load(folder/'generation/plan.json')
    require(reader.load(folder/'generation/inputs/config.json') == config and generation.get('config') == config,
            'Generation configuration copies differ')
    reader.expect(folder/'generation/plan.json', reader.load(folder/'generation-receipt.json')['plan_sha256'])
    require(config.get('id') == prepared['id']+'-mesh' and config.get('prompt') == plan['generation_prompt']
            and config.get('subject') == plan['subject'] and config.get('style') == plan['style']
            and config.get('rig') == 'none' and config.get('states') == [], 'Source belongs to a different child plan')
    stage = 'mesh' if prepared['reference'] else 'texture'
    downloaded = folder/'generation/stages'/stage/'downloads'
    receipt = reader.pack(downloaded)
    require(receipt.get('kind') == 'meshy_original_downloads' and receipt.get('stage') == stage
            and receipt.get('plan_sha256') == reader.pin(folder/'generation/plan.json')['sha256'],
            'Downloaded source is not bound to its generation plan')
    reader.expect(downloaded/'model.glb', expected, LIMITS['glb_bytes'])
    return reader.pin(downloaded/'model.glb')


def _complete(reader, state, row, request):
    from avatar_assets.pipeline_prompts import validate_plan, validate_review, review_citations
    from avatar_assets.pipeline import Pipeline

    require(not row['active'], 'Completed child still owns the active worker lane')
    folder, prepared, request_sha = _prepared(reader, state, row, request)
    delivery = reader.load(folder/'delivery.json')
    attempt = delivery.get('selected_attempt')
    require(type(attempt) is int and 0 <= attempt <= 2, 'Invalid delivered attempt')
    expected_receipt = {'delivery.json', 'plan.json', 'gallery.json', 'quality-policy.json',
                        f'review-{attempt}.json', f'deformation-gate-{attempt}.json', f'portable-root-audit-{attempt}.json'}
    receipt = reader.load(folder/'delivery-receipt.json')
    require(set(receipt) == expected_receipt, 'Delivery receipt omits required evidence')
    for name, checksum in receipt.items():
        reader.expect(reader.path(folder, name), checksum, LIMITS['json_bytes'])
    require(delivery.get('status') == 'complete' and delivery.get('request_sha256') == request_sha
            and delivery.get('plan_sha256') == reader.pin(folder/'plan.json')['sha256'], 'Delivery request/plan differs')
    plan = validate_plan(reader.load(folder/'plan.json'), prepared)
    require(plan['status'] == 'ready', 'Delivered plan is not ready')
    state_ids = [s['id'] for s in plan['states']]
    pack_rows = delivery.get('packs')
    require(isinstance(pack_rows, list) and len(pack_rows) == 8, 'Expected eight sealed delivery packs')
    packs = {}
    for item in pack_rows:
        require(isinstance(item, dict) and set(item) == {'path', 'manifest_sha256'}
                and item['path'] not in packs, 'Duplicate or invalid delivery pack')
        path = reader.path(folder, item['path'])
        require(path.parent == folder, 'Delivery pack must be a direct child')
        packs[path.name] = reader.pack(path, item['manifest_sha256'])
    required = {f'{name}-{attempt}' for name in ('model', 'reload', 'media-portable', 'exported-preview',
                                               'action-audit', 'final-evidence', 'portable-root-audit')}
    media_names = set(packs)-required
    require(required <= set(packs) and len(media_names) == 1 and next(iter(media_names)) in
            (f'media-{attempt}', f'media-continuation-{attempt}'), 'Unexpected delivery pack roles')
    model_folder = folder/f'model-{attempt}'
    model = packs[model_folder.name]
    model_manifest = reader.pin(model_folder/'manifest.json')['sha256']
    require(delivery['model'] == f'model-{attempt}/model.glb' and delivery['editable_source'] == f'model-{attempt}/source.blend',
            'Delivery points outside selected model pack')
    model_pin = reader.pin(model_folder/'model.glb', LIMITS['glb_bytes'])
    source_pin = _source(reader, folder, prepared, plan, model['source_sha256'])
    require([s['id'] for s in model['states']] == state_ids, 'Model state IDs differ from the child plan')
    rig = reader.load(model_folder/'rig-verification.json')
    require(model.get('rig_verification') == rig and rig.get('sha256') == model_pin['sha256']
            and rig.get('status') == 'validated_local_structure' and rig.get('loading_verified') is True
            and type(rig.get('skin_count')) is int and rig['skin_count'] > 0
            and rig.get('motion', {}).get('animation_verified') is True
            and {a['name'] for a in rig['motion']['animations']} == set(state_ids), 'Rig/motion receipt differs or failed')
    config_pin = reader.pin(model_folder/'sources/finish-design.json')
    reader.expect(folder/f'finish-{attempt}.json', config_pin['sha256'], LIMITS['json_bytes'])
    config = reader.load(folder/f'finish-{attempt}.json')
    deformation = reader.load(folder/f'deformation-gate-{attempt}.json')
    binding = deformation.get('bindings', {})
    require(deformation.get('passed') is True and all(s.get('passed') is True for s in deformation.get('states', []))
            and [s['id'] for s in deformation['states']] == state_ids, 'Deformation gate did not pass all states')
    for key, value in {'source_sha256':source_pin['sha256'], 'exported_model_sha256':model_pin['sha256'],
                       'manifest_sha256':model_manifest, 'design_sha256':config_pin['sha256'], 'state_ids':state_ids,
                       'finish_report_sha256':reader.pin(model_folder/'finish-report.json')['sha256'],
                       'rig_verification_sha256':reader.pin(model_folder/'rig-verification.json')['sha256']}.items():
        require(binding.get(key) == value, 'Deformation binding differs: '+key)
    action_folder = folder/f'action-audit-{attempt}'
    action, intent = reader.load(action_folder/'report.json'), reader.load(action_folder/'intent.json')
    require(action.get('passed') is True and action.get('source_unchanged') is True
            and action.get('model_sha256') == model_pin['sha256']
            and action.get('intent_sha256') == reader.pin(action_folder/'intent.json')['sha256']
            and intent.get('source_sha256') == model_pin['sha256'] and intent.get('states') == plan['states']
            and intent.get('action_bindings') == config.get('action_bindings'), 'Action intent/gate differs')
    require([s['id'] for s in action['states']] == state_ids, 'Action gate omitted a state')
    for actual, planned in zip(action['states'], plan['states']):
        results = actual['requirement_results']
        require(results.get('passed') is True and all(r.get('status') == 'passed' for r in results['requirements'])
                and [r['id'] for r in results['requirements']] == [r['id'] for r in planned.get('requirements', [])],
                'Action gate omitted or failed a requirement')
    reload = packs[f'reload-{attempt}']
    require(reload.get('status') == 'passed' and reload.get('source_manifest_sha256') == model_manifest
            and reload.get('source_model_sha256') == model_pin['sha256'], 'Independent reload differs or failed')
    attachment_folder = folder/f'portable-root-audit-{attempt}'
    attachment = reader.load(attachment_folder/'report.json')
    attachment_intent = reader.load(attachment_folder/'intent.json')
    require(attachment.get('passed') is True and attachment.get('status') in ('measured', 'not_applicable')
            and attachment.get('intent') == attachment_intent
            and attachment_intent == {'schema_version':1, 'source_sha256':source_pin['sha256'],
                'model_sha256':model_pin['sha256'], 'design_sha256':config_pin['sha256'], 'state_ids':state_ids},
            'Portable attachment differs or failed')
    custody = reader.load(folder/f'portable-root-audit-{attempt}.json')
    for key, path in {'manifest_sha256':attachment_folder/'manifest.json', 'report_sha256':attachment_folder/'report.json',
                      'intent_sha256':attachment_folder/'intent.json', 'job_sha256':attachment_folder/'job.json',
                      'model_manifest_sha256':model_folder/'manifest.json'}.items():
        reader.expect(path, custody.get(key), LIMITS['json_bytes'])
    summary = delivery['portable_attachment']
    require(summary.get('passed') is True and summary.get('status') == attachment['status']
            and summary.get('evidence', {}).get('sha256') == custody['report_sha256']
            and summary['evidence'].get('manifest_sha256') == custody['manifest_sha256'], 'Delivered attachment summary differs')
    review_receipt = reader.load(folder/f'review-{attempt}.json')
    evidence_folder = folder/f'final-evidence-{attempt}'
    evidence = packs[evidence_folder.name]
    require(review_receipt.get('model_manifest_sha256') == model_manifest
            and review_receipt.get('portable_attachment_manifest_sha256') == custody['manifest_sha256']
            and review_receipt.get('review') == delivery['review'], 'Delivered review binding differs')
    for key, target in {'model_manifest_sha256':model_folder, 'exported_manifest_sha256':folder/f'exported-preview-{attempt}',
                        'media_manifest_sha256':folder/next(iter(media_names))}.items():
        reader.expect(target/'manifest.json', evidence.get(key), LIMITS['json_bytes'])
    require(packs[f'exported-preview-{attempt}'].get('source_glb_sha256') == model_pin['sha256'],
            'Exported previews belong to a different model')
    expected_images = [{**item, 'path':str(evidence_folder/item['path'])} for item in evidence['images']]
    require(review_receipt['evidence'] == expected_images, 'Review evidence list differs')
    for item in expected_images:
        reader.expect(Path(item['path']), item['sha256'], 8*1024**2)
    citations = review_citations(expected_images, action)
    review = validate_review(delivery['review'], plan, [e['id'] for e in citations])
    require(review['verdict'] == 'accept', 'Delivered critic did not accept')
    Pipeline._bind_review(review, expected_images, action)
    media_folder = folder/f'media-portable-{attempt}'
    media = packs[media_folder.name]
    gallery = reader.load(folder/'gallery.json')
    require(delivery.get('gallery_catalog') == str((folder/'gallery.json').relative_to(reader.root))
            and isinstance(gallery.get('assets'), list) and len(gallery['assets']) == 1, 'Invalid delivery gallery binding')
    gallery_asset = gallery['assets'][0]
    require(gallery_asset.get('id') == prepared['id'] and gallery_asset.get('kind') == 'generated_3d'
            and gallery_asset.get('pack') == str(model_folder.relative_to(reader.root))
            and gallery_asset.get('media_pack') == str(media_folder.relative_to(reader.root)),
            'Gallery points to a different child model or media')
    require(media.get('parent_3d_manifest_sha256') == model_manifest, 'Portable media belongs to a different model')
    clips = media.get('clips')
    require(isinstance(clips, list) and [c['id'] for c in clips] == state_ids, 'Portable media omitted requested states')
    available = []
    for clip in clips:
        video = reader.pin(reader.path(media_folder, clip['path']), 64*1024**2)
        fallback = reader.pin(reader.path(media_folder, clip['fallback']), 16*1024**2)
        require(Path(clip['path']).suffix == '.mp4' and Path(clip['fallback']).suffix in ('.png', '.webp')
                and clip.get('inspection', {}).get('sha256') == video['sha256']
                and clip.get('fallback_inspection', {}).get('sha256') == fallback['sha256'], 'Media inspection hash differs')
        available.append({'state_id':clip['id'], 'video_available':True, 'video':video,
                          'static_fallback_available':True, 'static_fallback':fallback, 'loop_verified':False})
    return {'status':'verified_complete', 'source':source_pin, 'model':model_pin,
            'delivery':reader.pin(folder/'delivery.json'), 'delivery_receipt':reader.pin(folder/'delivery-receipt.json'),
            'selected_attempt':attempt, 'state_ids':state_ids, 'media':available,
            'gates':{'rig':rig['status'], 'deformation':True, 'action':True, 'portable_attachment':attachment['status'],
                     'reload':True, 'critic':'accept'}, 'appearance_scope':delivery.get('appearance_acceptance_scope',
                     'The saved per-child critic criteria; not universal quality or human approval.'),
            'limitations':delivery.get('limitations', [])}


def inspect_scene(root, scene_id):
    """Authenticate existing artifacts only; malformed custody raises instead of becoming a partial pass."""
    from avatar_assets.scene_requests import validate_explicit_assets
    reader = Reader(Path(root))
    state, scene, rows = _registry(reader, scene_id)
    request = parse_json(scene['request'])
    require(set(request) == {'schema_version','title','assets','mode'} and request['schema_version'] == 1
            and request['mode'] == 'explicit_independent_assets' and _digest(request) == scene['request_digest']
            and isinstance(request['title'], str) and 1 <= len(request['title'].strip()) <= 120
            and isinstance(request['assets'], list) and len(request['assets']) == 2, 'Scene contract differs')
    validate_explicit_assets([{k:v for k,v in a.items() if k != 'reference'} for a in request['assets']])
    require(set(rows) <= {a['id'] for a in request['assets']}, 'Unknown scene member')
    results = []
    for asset in request['assets']:
        expected = {k:v for k,v in asset.items() if k not in ('id','name')}
        expected['reuse_model'] = None
        row = rows.get(asset['id'])
        base = {'asset_id':asset['id'], 'name':asset['name'],
                'request':{k:asset[k] for k in ('prompt','style','custom_actions','preset')}}
        if row is None:
            results.append({**base, 'status':'missing_child', 'canonical_status':None}); continue
        require(row['output'] == 'studio-'+row['id'] and parse_json(row['request']) == expected
                and row['request_digest'] == _digest(expected), 'Scene child request was swapped or changed')
        data = parse_json(row['data'])
        require(data.get('scene') == {'id':scene_id, 'title':request['title'], 'asset_id':asset['id'], 'asset_name':asset['name']},
                'Child scene membership differs')
        require(reader.load(state/'jobs'/row['id']/'request.json') == expected, 'Saved Studio request differs')
        base.update(job_id=row['id'], canonical_status=row['status'], request_digest=row['request_digest'])
        if row['status'] == 'complete':
            results.append({**base, **_complete(reader, state, row, expected)})
        else:
            detail = {'status':'not_complete', 'active_owner':row['active']}
            if row['status'] == 'unsupported':
                folder, _, digest = _prepared(reader, state, row, expected)
                unsupported = reader.load(folder/'unsupported.json')
                require(unsupported.get('status') == 'unsupported' and unsupported.get('request_sha256') == digest,
                        'Unsupported result request differs')
                detail.update(status='unsupported', stage=unsupported.get('stage'), reason=unsupported.get('reason'),
                              terminal_receipt=reader.pin(folder/'unsupported.json'))
            results.append({**base, **detail})
    complete = [r for r in results if r['status'] == 'verified_complete']
    if len(complete) == 2:
        require(len({r['source']['sha256'] for r in complete}) == 2, 'Both assets reuse the same source GLB')
        require(len({r['model']['sha256'] for r in complete}) == 2, 'Both assets deliver the same final GLB')
    require(_registry(reader, scene_id)[1:] == (scene, rows), 'Canonical scene changed during inspection')
    for pin, _identity in list(reader.pins.values()):
        reader.pin(reader.root/pin['path'])
    return {'schema_version':1, 'policy':'separate-studio-delivery-custody-v1', 'scene_id':scene_id,
            'scene_request_sha256':scene['request_digest'], 'title':request['title'],
            'status':'verified_complete' if len(complete) == 2 else 'partial' if complete else 'not_complete',
            'complete':len(complete) == 2, 'children':results,
            'claims':{**CLAIMS, 'separate_delivered_files':len(complete) == 2},
            'scope':'Saved independent child request/delivery custody and recorded gate outcomes; no GLB import, rerender, new critic, physical simulation or geometric unfusing proof.',
            'limits':LIMITS, 'elapsed_seconds':round(time.monotonic()-reader.started, 6),
            'verified_files':len(reader.pins), 'verified_bytes':reader.total,
            'artifacts':[pin[0] for _,pin in sorted(reader.pins.items())],
            'recorded_local':datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out if args.out.is_absolute() else ROOT/args.out
    require(out == out.resolve() and out.is_relative_to(ROOT/'outputs') and not out.exists(),
            'Use a fresh canonical output directory')
    out.mkdir(mode=0o700, parents=True)
    code = 0
    try:
        result = inspect_scene(ROOT, args.scene)
        code = 0 if result['complete'] else 2
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        result = {'schema_version':1, 'scene_id':args.scene, 'status':'integrity_failure', 'complete':False,
                  'error':str(error), 'claims':CLAIMS}
        code = 1
    write_json(out/'result.json', result)
    source = Path(__file__).resolve()
    (out/'verify_scene_delivery.py').write_bytes(source.read_bytes())
    pins = [{'path':p.name, 'bytes':p.stat().st_size, 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(out.iterdir())]
    write_json(out/'manifest.json', {'schema_version':1, 'status':result['status'], 'artifacts':pins})
    print(json.dumps({'status':result['status'], 'complete':result['complete'], 'out':str(out)}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
