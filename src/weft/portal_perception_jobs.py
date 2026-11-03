"""Two-phase equipment perception: local evidence first, explicit media consent before API work."""

from hashlib import sha256
import json
import math
from pathlib import Path
import re
import time

from weft.exercise_body_source import _json, _path
from weft.io import canonical_hash, digest, local_time
from weft.lab_worker import atomic_json
from weft.openai_access import MODEL, MAX_OUTPUT_TOKENS, PROJECT_ROOT, RESERVATION_MICROUSD, OpenAIBudget, credential_status
from weft.shared_world_packaging import read_package_file, write_package_files

PROFILE = 'portal-exercise-perception-v1'
ACTIVE = frozenset({'queued_preflight', 'preflighting', 'queued', 'analyzing', 'preparing_tracking',
                    'tracking', 'exporting_evidence', 'cancelling'})
RESTING = frozenset({'awaiting_consent', 'unsupported', 'evidence_ready', 'failed', 'cancelled', 'interrupted'})
SCOPE = 'Source observations and conditional worlds; inferred identity, geometry and contact are not physically verified.'


class UnsupportedPerception(ValueError):
    pass


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def _read(root, name, maximum=32 << 10):
    return _json(read_package_file(root, name, max_bytes=maximum))


def read_status(root):
    try:
        return _read(root, 'perception/status.json')
    except ValueError as error:
        if isinstance(error.__cause__, FileNotFoundError):
            return {}
        raise


def _context(root):
    root = _path(root)
    job = _read(root, 'job.json', 64 << 10)
    parent = _read(root, 'status.json')
    if (job.get('kind', 'video') != 'video' or parent.get('status') != 'needs_input'
            or parent.get('route') != 'exercise' or parent.get('lab')):
        raise ValueError('Perception applies only to unresolved original exercise uploads')
    if (root/'equipment').exists() or (root/'equipment').is_symlink():
        raise ValueError('Existing equipment candidates must be preserved; this project cannot start new paid perception')
    name = job.get('source_file')
    if (not isinstance(name, str) or Path(name).name != name or name in {'.', '..'}
            or not re.fullmatch(r'[a-f0-9]{64}', job.get('source_sha256', ''))):
        raise ValueError('Perception requires an exact safe uploaded source')
    return root, job, _path(root/name)


def _contract():
    from weft.portal_pipeline import INSTRUCTIONS, SCHEMA
    return canonical_hash({'model': MODEL, 'instructions': INSTRUCTIONS, 'schema': SCHEMA,
                           'max_output_tokens': MAX_OUTPUT_TOKENS})


def _quote(root):
    complete = _read(root, 'perception/preflight-complete.json')
    raw = read_package_file(root, 'perception/quote.json', max_bytes=8192,
                            expected_sha256=complete['quote_sha256'])
    quote = _json(raw)
    return quote, sha256(raw).hexdigest()


def summary(root, job, parent_status):
    eligible = (job.get('kind', 'video') == 'video' and parent_status.get('status') == 'needs_input'
                and parent_status.get('route') == 'exercise' and not parent_status.get('lab')
                and not (Path(root)/'equipment').exists() and not (Path(root)/'equipment').is_symlink())
    status = read_status(root)
    result = {'available': eligible, 'status': status.get('status', 'not_started'),
              'message': status.get('message', 'Check this video locally before choosing whether to send media to AI providers.'),
              'quote': None, 'quote_sha256': None, 'scope': SCOPE}
    artifacts = {'quote': 'quote.json', 'request': 'request.json', 'analysis_receipt': 'analysis/receipt.json',
                 'bridge_result': 'bridge/result.json', 'prepared_source': 'prepared-source.json', 'failure': 'failure.json',
                 'preflight_receipt': 'preflight-worker-receipt.json', 'preparation_receipt': 'preparation-worker-receipt.json',
                 'preflight_result': 'preflight/result.json', 'person_region': 'preflight/body-anchor.json',
                 'body_model': 'preflight/body_source/human.xml',
                 'body_evidence': 'preflight/body_source/evidence.json', 'body_clock': 'preflight/body_source/native-clock.json',
                 'body_preview': 'preflight/body_source/tracks.gif'}
    result['artifacts'] = {key: 'perception/'+name for key, name in artifacts.items()
                           if (Path(root)/'perception'/name).is_file() and not (Path(root)/'perception'/name).is_symlink()}
    if status:
        result.update({key: status[key] for key in ('stage', 'events', 'error', 'seconds') if key in status})
    if status.get('status') == 'awaiting_consent':
        result['quote'], result['quote_sha256'] = _quote(root)
    return result


def queue_preflight(root):
    root, job, _ = _context(root)
    if read_status(root):
        return
    work = root/'perception'
    work.mkdir(mode=0o700)
    write_package_files(work, {'parent-status.json': read_package_file(root, 'status.json', max_bytes=32 << 10),
        'preflight-request.json': _bytes({'profile': PROFILE, 'source_sha256': job['source_sha256'],
                                         'requested_at': local_time(), 'api_calls_authorized': 0})})
    atomic_json(work/'status.json', {'status': 'queued_preflight', 'stage': 'local_check',
        'message': 'Waiting to check video, native body evidence and the initial person region locally. No API call.'})


def _validate_request(payload, quote, pin):
    fields = {'quote_sha256', 'consent', 'openai_max_usd', 'meta_max_usd', 'request_id'}
    if (not isinstance(payload, dict) or set(payload) != fields or payload['quote_sha256'] != pin
            or not isinstance(payload['request_id'], str) or not re.fullmatch(r'[a-f0-9]{32}', payload['request_id'])
            or payload['consent'] != {'openai_sampled_images': True, 'meta_full_video_and_regions': True}
            or any(type(payload['consent'].get(k)) is not bool for k in ('openai_sampled_images', 'meta_full_video_and_regions'))):
        raise ValueError('Explicit separate OpenAI-image and Meta-full-video consent must match this quote')
    for key in ('openai_max_usd', 'meta_max_usd'):
        if type(payload[key]) not in (int, float) or not math.isfinite(payload[key]) or payload[key] != quote[key]:
            raise ValueError('Provider allowances must equal the displayed bounded quote')


def _validate_quote(root, quote, job):
    n = quote.get('frame_count')
    if (quote.get('profile') != 'portal-perception-quote-v1' or quote.get('schema_version') != 1
            or quote.get('source_sha256') != job['source_sha256'] or type(n) is not int or not 12 <= n <= 500
            or quote.get('openai_model') != MODEL or quote.get('meta_model') != 'sam-3.1'
            or quote.get('analysis_contract_sha256') != _contract()
            or quote.get('openai_max_usd') != RESERVATION_MICROUSD/1e6
            or quote.get('meta_max_usd') != max(.15, max(50, n)*4/5000)
            or quote.get('maximum_requests') != {'openai': 1, 'meta': 1}):
        raise ValueError('Source, model or cost contract changed; stale quote cannot authorize a request')
    read_package_file(root, 'perception/preflight/manifest.json', max_bytes=1 << 20,
                      expected_sha256=quote['preflight_manifest_sha256'])


def _budget_guard(project, quote):
    from weft.meta_access import access_status
    openai = OpenAIBudget(project).status()
    meta = access_status(project)
    if not credential_status(project)['configured'] or not meta['credential_present']:
        raise ValueError('Both server-side project keys must be configured before media preparation')
    if (not openai['configured'] or openai['remaining_usd'] < quote['openai_max_usd']
            or not meta['budget_configured'] or meta.get('budget_exceeded')
            or meta['remaining_usd'] < quote['meta_max_usd']):
        raise ValueError('Existing provider budgets cannot cover this quote; no media sent')


def queue_prepare(root, payload, *, project=PROJECT_ROOT):
    root, job, _ = _context(root)
    quote, pin = _quote(root)
    _validate_request(payload, quote, pin)
    _validate_quote(root, quote, job)
    work = root/'perception'
    if (work/'request.json').exists():
        if _read(work, 'request.json')['consented_action'] != payload:
            raise ValueError('This preparation already has a different immutable request; no retry')
        return
    if read_status(root).get('status') != 'awaiting_consent':
        raise ValueError('Complete local checks before requesting paid preparation')
    if read_package_file(root, 'status.json', max_bytes=32 << 10) != read_package_file(work, 'parent-status.json', max_bytes=32 << 10):
        raise ValueError('Parent upload changed after local preparation')
    _budget_guard(project, quote)
    request = {'profile': PROFILE, 'source_sha256': job['source_sha256'], 'consented_action': payload,
               'analysis_call_id': canonical_hash({'job': job['id'], 'request': payload})[:32], 'requested_at': local_time()}
    write_package_files(work, {'request.json': _bytes(request)})
    atomic_json(work/'status.json', {'status': 'queued', 'stage': 'consented_preparation',
        'message': 'Waiting for one bounded analysis/tracking preparation. No automatic retries.'})


def execute_perception_job(root, *, project=PROJECT_ROOT):
    from weft.exercise_perception_pipeline import (prepare_exercise_perception_preflight,
        validate_perception_preflight, prepare_exercise_perception, VISIBLE_ANCHOR_PROFILE)
    from weft.portal_pipeline import sample_video, analyze_frames

    root = _path(root); work = root/'perception'; started = time.monotonic(); deadline = started+420
    paid = (work/'request.json').exists()
    state = read_status(root)
    if state.get('status') in RESTING:
        return 0 if state['status'] in {'awaiting_consent', 'evidence_ready'} else 1
    events = []
    def check():
        if (work/'cancel-request.json').exists():
            raise InterruptedError('Cancelled; completed evidence and uncertain charges remain preserved')
        if time.monotonic() >= deadline:
            raise TimeoutError('Perception job exceeded its work deadline')
    def progress(stage, message):
        check()
        if len(events) < 40:
            events.append({'stage': stage, 'message': message[:500], 'time': local_time(),
                           'elapsed_seconds': round(time.monotonic()-started, 3)})
        atomic_json(work/'status.json', {'status': stage, 'stage': stage, 'message': message[:500], 'events': events})
    try:
        root, job, video = _context(root)
        if read_package_file(root, 'status.json', max_bytes=32 << 10) != read_package_file(work, 'parent-status.json', max_bytes=32 << 10):
            raise ValueError('Original upload changed during perception')
        pre_request = _read(work, 'preflight-request.json')
        if pre_request.get('profile') != PROFILE or pre_request.get('source_sha256') != job['source_sha256']:
            raise ValueError('Local preflight request differs from original upload')
        check()
        if video.suffix.lower() != '.mp4' or not 1 <= video.stat().st_size <= 37_000_000 or digest(video) != job['source_sha256']:
            raise UnsupportedPerception('Use the exact original MP4 within 37 MB; implicit conversion is unsupported')
        if not paid:
            write_package_files(work, {'preflight-start.json': _bytes({'started_at': local_time(), 'api_calls': 0})})
            progress('preflighting', 'Checking native video/body evidence and initial person visibility locally; no media sent.')
            prepared = prepare_exercise_perception_preflight(video, work/'preflight',
                pose_model=Path(project)/'.local/models/pose_landmarker_lite.task', max_seconds=210,
                anchor_profile=VISIBLE_ANCHOR_PROFILE)
            check()
            n = prepared['metadata']['frame_count']
            quote = {'schema_version': 1, 'profile': 'portal-perception-quote-v1', 'source_sha256': job['source_sha256'],
                'frame_count': n, 'preflight_manifest_sha256': prepared['manifest_sha256'],
                'analysis_contract_sha256': _contract(), 'openai_model': MODEL, 'meta_model': 'sam-3.1',
                'openai_max_usd': RESERVATION_MICROUSD/1e6, 'meta_max_usd': max(.15, max(50, n)*4/5000),
                'maximum_requests': {'openai': 1, 'meta': 1}, 'scope': SCOPE,
                'message': 'OpenAI receives up to six sampled images; Meta receives the original video and inferred tracking regions. Reservations are separate lifetime-budget bounds, not guaranteed provider hard caps.'}
            raw = _bytes(quote)
            write_package_files(work, {'quote.json': raw, 'preflight-complete.json': _bytes({
                'quote_sha256': sha256(raw).hexdigest(), 'preflight_manifest_sha256': prepared['manifest_sha256'],
                'source_sha256': job['source_sha256'], 'api_calls': 0})})
            atomic_json(work/'status.json', {'status': 'awaiting_consent', 'stage': 'local_ready', 'events': events,
                'message': 'Local checks passed. Choose separately whether to send sampled images to OpenAI and the full video to Meta.',
                'seconds': time.monotonic()-started, 'api_calls': 0})
            return 0
        quote, pin = _quote(root)
        _validate_quote(root, quote, job)
        request = _read(work, 'request.json')
        _validate_request(request['consented_action'], quote, pin)
        if (request.get('profile') != PROFILE or request.get('source_sha256') != job['source_sha256']
                or request.get('analysis_call_id') != canonical_hash({'job': job['id'], 'request': request['consented_action']})[:32]):
            raise ValueError('Paid action identity differs from its source-bound consent')
        _budget_guard(project, quote)
        prepared = validate_perception_preflight(video, work/'preflight', deadline=deadline)
        if (prepared['metadata']['frame_count'] != quote['frame_count']
                or prepared['manifest_sha256'] != quote['preflight_manifest_sha256']):
            raise ValueError('Quote frame count or source seal differs from the actual local preflight')
        check()
        write_package_files(work, {'preparation-start.json': _bytes({'started_at': local_time(),
            'request_sha256': digest(work/'request.json'), 'retry': False})})
        progress('analyzing', 'Preparing up to six source images for the consented OpenAI analysis.')
        frames = sample_video(video, work/'analysis')
        check()
        if deadline-time.monotonic() < 190:
            raise TimeoutError('Insufficient time for bounded analysis and tracking; analysis not sent')
        _budget_guard(project, quote)
        analyze_frames(frames, work/'analysis', identity=request['analysis_call_id'], project=project)
        check()
        progress('preparing_tracking', 'Validating the source-bound AI regions before any Meta video request.')
        def bridge_progress(row):
            stage = row.get('stage', 'preparing_tracking')
            stage = {'tracking_evidence': 'exporting_evidence'}.get(stage, stage)
            progress(stage if stage in ACTIVE else 'preparing_tracking', str(row.get('message', stage)))
        result = prepare_exercise_perception(video, work/'analysis', work/'bridge', preflight=work/'preflight',
            execute_tracking=True, max_meta_cost_usd=request['consented_action']['meta_max_usd'], project_root=project,
            max_seconds=min(420, deadline-time.monotonic()), progress=bridge_progress)
        check()
        if result.get('status') != 'evidence_ready' or result.get('assembly_inputs_ready') is not True:
            raise ValueError('Perception did not produce complete source-bound assembly inputs')
        from weft.portal_equipment_candidates import publish_perception_source, PROFILE as EQUIPMENT_PROFILE
        publish_perception_source(root, work/'bridge', project=project)
        check()
        equipment = root/'equipment'
        equipment.mkdir(mode=0o700)
        write_package_files(equipment, {'parent-status.json': read_package_file(root, 'status.json', max_bytes=32 << 10),
            'request.json': _bytes({'profile': EQUIPMENT_PROFILE, 'source_kind': 'perception',
                'source_sha256': job['source_sha256'], 'descriptor_sha256': digest(work/'prepared-source.json')})})
        atomic_json(equipment/'status.json', {'status': 'queued', 'message': 'New source evidence is ready; waiting for ordinary equipment construction. No further API call.'})
        atomic_json(work/'status.json', {'status': 'evidence_ready', 'stage': 'evidence_ready', 'events': events,
            'message': 'Source observations prepared; candidate construction is queued. No world is admitted yet.',
            'seconds': time.monotonic()-started, 'maximum_requests': quote['maximum_requests'], 'simulation_ready': False})
        return 0
    except Exception as error:
        status = 'cancelled' if isinstance(error, InterruptedError) else 'unsupported' if isinstance(error, UnsupportedPerception) else 'failed'
        failure = {'status': status, 'error_type': type(error).__name__, 'error': str(error)[:600], 'events': events,
            'message': 'Preparation stopped. Existing evidence and any unknown API charges are preserved; no automatic retry.',
            'seconds': time.monotonic()-started, 'retry': False, 'scope': SCOPE}
        atomic_json(work/'failure.json', failure)
        atomic_json(work/'status.json', failure)
        return 1
