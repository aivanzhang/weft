"""Optional bounded coat probe feedback; never change a recipe or interpret it as approval."""
from pathlib import Path
import json
import re

from avatar_assets import repair_visual_inputs, undercoat_admission
from avatar_assets.coat_representation import summarize_coat
from avatar_assets.safety import artifacts, local_file, read_json, require, sha256, write_json

POLICY = 'automatic-coat-admission-feedback-v1'
SETTINGS = {'profile': 'short-pile-v1', 'uv_policy': 'material-adaptive-v1',
            'resolution': 1024, 'pitch_m': .004, 'strength': .22, 'roughness': .92, 'seed': 4517}
MAX_PROBES = 3


def _unavailable(reason):
    return {'policy': POLICY, 'status': 'not_applicable', 'reason': reason,
            'scope': 'Optional capability data, not a required finish or an appearance verdict', 'quality_approved': False}


def _target(root, job, request, continuation, current_review, source_sha):
    if current_review is not None:
        require(current_review.get('source_sha256') == source_sha, 'Admission review source differs')
        pin = current_review.get('receipt', {})
        name = pin.get('path')
        require(isinstance(name, str) and re.fullmatch(r'review-[0-2]\.json', name), 'Invalid admission review path')
        path = local_file(job/name, 1024*1024)
        require(pin == {'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)}, 'Admission review receipt changed')
        parent, attempt, pins = job, int(name[7]), None
        expected_model = current_review['model_manifest_sha256']
    else:
        if not continuation or continuation.get('findings', {}).get('rejected_review') is not None:
            return None
        history = continuation.get('findings', {}).get('historical_visual_reviews', [])
        if not history:return None
        require(isinstance(history, list) and len(history) <= 8, 'Admission visual history exceeds bound')
        summary = history[-1]
        # Older summaries without model ancestry cannot authorize a native scene load.
        if not isinstance(summary, dict) or not {'parent_output','attempt','model_manifest_sha256','review_reference'} <= set(summary):
            return None
        from avatar_assets.pipeline_continuation import _read_context
        _, saved = _read_context(job, request)
        require(saved == continuation, 'Admission continuation context differs')
        parent, pins, ancestor_request = repair_visual_inputs._lineage_inventory(root, job, request, summary['parent_output'])
        require(ancestor_request['reuse_model']['sha256'] == source_sha, 'Admission ancestor source differs')
        attempt = summary['attempt'];expected_model = summary['model_manifest_sha256']
        require(type(attempt) is int and 0 <= attempt <= 2, 'Admission attempt exceeds bound')
        path = repair_visual_inputs._bound(parent, pins, f'review-{attempt}.json')
        require(sha256(path) == summary['review_reference']['sha256'], 'Admission ancestor review changed')
    receipt = read_json(path)
    pack = parent/f'model-{attempt}'
    model_path = (repair_visual_inputs._bound(parent, pins, f'model-{attempt}/manifest.json')
                  if pins is not None else local_file(pack/'manifest.json', 1024*1024))
    require(sha256(model_path) == expected_model == receipt['model_manifest_sha256'], 'Admission model receipt differs')
    manifest = read_json(model_path)
    require(manifest.get('source_sha256') == source_sha and manifest.get('status') == 'needs_visual_review',
            'Admission requires the exact completed source-bound model')
    repair_visual_inputs._review_evidence(parent,source_sha,attempt,receipt['review'],receipt['evidence'],pins)
    return pack, manifest


def _eligibility(pack, manifest, source_sha):
    facts = summarize_coat(pack,manifest,source_sha)
    if facts['status'] == 'unavailable':return None, 'Bound coat representation facts are unavailable'
    config = read_json(pack/'sources/finish-design.json');report = read_json(pack/'finish-report.json')
    groom = report.get('groom') or {};portable = config.get('portable_fur') or {}
    if (groom.get('kind') != 'editable_native_blender_particle_hair' or groom.get('native_only') is not True
            or config.get('material_fur') or config.get('fur_undercoat') or config.get('portable_fringe')
            or portable.get('settings', {}).get('profile') != 'tuft-v1'):
        return None, 'Probe supports copied native groom plus tufts without an existing undercoat, material groom or fringe'
    bands = portable.get('bands', []);coats = report.get('portable_fur', [])
    if not (len(bands) == len(coats) == 1 and bands[0].get('brightness') == [0, 1.01]):
        return None, 'Exact saved density-face count is unavailable for this band selection'
    selection = coats[0].get('root_summary', {}).get('binding_selection', {})
    count = selection.get('eligible_triangles')
    if not (selection.get('excluded_degenerate_triangles') == 0 and type(count) is int and 1 <= count <= 100000):
        return None, 'Verified nondegenerate selected-face count is unavailable'
    return {**SETTINGS, 'min_faces': count, 'max_faces': count}, None


def _summary(out, pack, manifest_sha, settings):
    manifest = read_json(out/'manifest.json')
    require(manifest.get('kind') == 'read_only_undercoat_admission' and manifest.get('status') in ('supported','refused')
            and manifest.get('artifacts') == artifacts(out), 'Admission output is incomplete, failed or changed; inspect custody')
    require(sum(row['bytes'] for row in manifest['artifacts'])+(out/'manifest.json').stat().st_size <= 64*1024*1024,
            'Admission output exceeds64MiB')
    result = read_json(out/'result.json');job = read_json(out/'job.json')
    require(result.get('policy') == undercoat_admission.POLICY and result.get('status') == manifest['status']
            and result.get('settings') == settings and read_json(out/'settings.json') == settings
            and result.get('inputs_unchanged') is True and result.get('input_writes') is False
            and result.get('provider_calls') == 0 and result.get('quality_approved') is False
            and result.get('job') == job and job.get('exit_code') == 0 and job.get('failure') is None
            and job.get('process_exit_verified') is True and not job.get('manual_stop_required'),
            'Admission result or process custody differs')
    native = read_json(out/'native-admission.json')
    require(all(result.get(key)==value for key,value in native.items()), 'Admission native/result fields differ')
    require(isinstance(result.get('counts'),dict) and result['counts'].get('selected_triangles')==settings['min_faces']==settings['max_faces'],
            'Admission measured selection differs from exact saved count')
    inputs = read_json(out/'admission-inputs.json')
    require(result.get('inputs_sha256') == sha256(out/'admission-inputs.json') and result.get('inputs') == inputs.get('inputs')
            and inputs.get('settings') == settings and inputs.get('pack') == str(pack)
            and inputs['inputs']['manifest']['sha256'] == manifest_sha == sha256(pack/'manifest.json'),
            'Admission result belongs to different inputs')
    undercoat_admission._unchanged(inputs['inputs'])
    prepared = read_json(out/'prepared.json')
    require(prepared.get('limits') == undercoat_admission.LIMITS and prepared.get('provider_calls') == 0
            and prepared.get('inputs_sha256') == sha256(out/'admission-inputs.json')
            and prepared.get('settings_sha256') == sha256(out/'settings.json'), 'Admission preparation differs')
    sources = prepared.get('sources')
    expected_names = set(undercoat_admission.NATIVE_SOURCES) | {'undercoat_admission.py','provider_model.py',
        'three_d.py','runner.py','safety.py','blender_fur_cards.py'}
    require(isinstance(sources,dict) and set(sources) == expected_names, 'Admission source inventory differs')
    undercoat_admission._unchanged(sources)
    for name,pin in sources.items():
        require(Path(pin['path']) == out/'sources'/name, 'Admission source snapshot escaped output')
        current = Path(undercoat_admission.__file__) if name == 'undercoat_admission.py' else (
            undercoat_admission.ROOT/('scripts' if name in undercoat_admission.NATIVE_SOURCES or name=='blender_fur_cards.py' else 'avatar_assets')/name)
        require(pin['sha256'] == sha256(current), 'Admission source version changed; no automatic rerun')
    reason = result.get('reason')
    require(isinstance(reason,str) and len(reason)<=2000,'Admission reason exceeds bound')
    summary = {'policy': POLICY, 'status': result['status'], 'reason': reason, 'settings': settings,
        'model_manifest_sha256': manifest_sha, 'probe_manifest_sha256': sha256(out/'manifest.json'),
        'result_reference': {'path': str((out/'result.json').relative_to(out.parent)), 'bytes': (out/'result.json').stat().st_size,
                             'sha256': sha256(out/'result.json')},
        'scope': 'Optional read-only UV/resource probe, not a requested recipe change, actual hair, visible coverage or quality approval. Do not automatically reduce counts.',
        'quality_approved': False}
    for key in ('counts','export_vertex_budget','material_seam_capacity'):
        value=result.get(key)
        if value is None:summary[key]={'status':'unavailable'}
        else:
            require(isinstance(value,dict) and len(json.dumps(value,allow_nan=False).encode())<=8*1024,
                    'Admission measurement exceeds bounded summary')
            summary[key]=value
    require(len(json.dumps(summary,allow_nan=False).encode()) <= 16*1024,'Admission feedback exceeds16KiB')
    return summary


def feedback(root, job, request, continuation, current_review, source_sha, *, before_probe=lambda: None):
    """Caller verifies continuation and current review first; no unbound pack path is accepted."""
    root, job = Path(root), Path(job)
    # Legacy/nonfur fixture summaries cannot authorize native work.
    if current_review is not None and current_review.get('coat_representation',{}).get('status') == 'unavailable':
        return _unavailable('Bound coat representation facts are unavailable')
    target = _target(root,job,request,continuation,current_review,source_sha)
    if target is None:return _unavailable('No authenticated valid visual repair target')
    pack,manifest = target
    settings, reason = _eligibility(pack,manifest,source_sha)
    if settings is None:return _unavailable(reason)
    checksum=sha256(pack/'manifest.json');out=job/('coat-admission-'+checksum)
    siblings=list(job.glob('coat-admission-*'))
    require(len(siblings)<=MAX_PROBES and all(p.is_dir() and not p.is_symlink() for p in siblings),
            'Admission job inventory exceeds three probes')
    receipt_path=job/('coat-probe-receipt-'+checksum+'.json')
    existing=out.exists()
    if not existing:
        require(not receipt_path.exists() and len(siblings)<MAX_PROBES,'Admission reached three probes or receipt lost its output')
        before_probe()
        undercoat_admission.inspect(pack,settings,out)
    else:
        receipt=read_json(receipt_path)
        require(receipt=={'policy':POLICY,'model_manifest_sha256':checksum,
                         'probe_manifest_sha256':sha256(out/'manifest.json'),
                         'request_sha256':sha256(job/'request.json')},
                'Admission cached receipt changed; no automatic replay')
    summary=_summary(out,pack,checksum,settings)
    if not existing:
        write_json(receipt_path,{'policy':POLICY,'model_manifest_sha256':checksum,
                               'probe_manifest_sha256':sha256(out/'manifest.json'),
                               'request_sha256':sha256(job/'request.json')})
    summary['receipt_reference']={'path':receipt_path.name,'bytes':receipt_path.stat().st_size,'sha256':sha256(receipt_path)}
    return summary
