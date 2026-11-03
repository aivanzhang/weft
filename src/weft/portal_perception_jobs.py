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
POSE_DIAGNOSTIC = 'portal-pose-failure-diagnostic-v1'


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


def _contract(profile=None):
    from weft.equipment_analysis_contract import LEGACY_PROFILE, SAMPLED_PROFILE, analysis_contract
    contract = analysis_contract(SAMPLED_PROFILE if profile is None else profile)
    value = {'model': MODEL, 'instructions': contract['instructions'], 'schema': contract['schema'],
             'max_output_tokens': MAX_OUTPUT_TOKENS}
    if contract['profile'] != LEGACY_PROFILE:
        value.update(profile=contract['profile'], format_name=contract['format_name'])
    return canonical_hash(value)


def _quote(root):
    complete = _read(root, 'perception/preflight-complete.json')
    raw = read_package_file(root, 'perception/quote.json', max_bytes=8192,
                            expected_sha256=complete['quote_sha256'])
    quote = _json(raw)
    return quote, sha256(raw).hexdigest()


def pose_failure_summary(root, source_sha256):
    """Read the small worker-produced snapshot, never parse poses on a request thread."""
    try:
        base = 'perception/pose-diagnostic/'
        manifest = _read(root, base+'manifest.json', 8192)
        failure = read_package_file(root, 'perception/failure.json', max_bytes=32 << 10,
                                    expected_sha256=manifest['failure_sha256'])
        result = _json(read_package_file(root, base+'summary.json', max_bytes=32 << 10,
                                        expected_sha256=manifest['summary_sha256']))
        if (manifest.get('profile') != POSE_DIAGNOSTIC or result.get('profile') != POSE_DIAGNOSTIC
                or result.get('source_sha256') != source_sha256 or manifest.get('source_sha256') != source_sha256
                or _json(failure).get('status') != 'failed' or result.get('simulation_ready') is not False):
            raise ValueError('Pose diagnostic differs from its failed source')
        counts = [result.get(k) for k in ('sampled_frames', 'usable_samples', 'identity_breaks', 'window_count')]
        if (any(type(n) is not int or not 0 <= n <= 1800 for n in counts)
                or not 1 <= counts[0] or counts[1] > counts[0] or not 1 <= counts[2] <= counts[0]
                or not isinstance(result.get('listed_barriers'), list) or len(result['listed_barriers']) != min(32, counts[2])
                or not isinstance(result.get('listed_windows'), list) or len(result['listed_windows']) != min(16, counts[3])):
            raise ValueError('Invalid pose diagnostic counts')
        for row in result['listed_barriers']:
            if (type(row.get('frame')) is not int or row['frame'] < 0 or row.get('hypothesis_count') != 2
                    or 'decoder_timestamp_s' not in row
                    or type(row.get('source_time_s')) not in (int, float) or not math.isfinite(row['source_time_s'])
                    or row.get('decoder_timestamp_s') is not None and (type(row['decoder_timestamp_s']) not in (int, float)
                        or not math.isfinite(row['decoder_timestamp_s']))):
                raise ValueError('Invalid pose diagnostic timestamp')
        for row in result['listed_windows']:
            interval = row.get('frame_interval')
            if (not isinstance(interval, list) or len(interval) != 2
                    or any(type(n) is not int or n < 0 for n in interval) or interval[1] < interval[0]
                    or type(row.get('screen_passed')) is not bool
                    or any(type(row.get(k)) is not int or not 0 <= row[k] <= 1800 for k in ('usable_samples', 'sampled_frames'))):
                raise ValueError('Invalid pose diagnostic window')
        return result
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        return None


def prepare_pose_failure_diagnostic(root, *, deadline):
    """Attach a bounded JSON-only explanation; never modify or retry the failed job."""
    from weft.pose_identity_windows import screen_identity_windows, MAX_INPUT_BYTES

    root = _path(root)
    target = root/'perception/pose-diagnostic'
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Pose diagnostic exceeded its deadline')
    check()
    job = _read(root, 'job.json', 64 << 10)
    failure = read_package_file(root, 'perception/failure.json', max_bytes=32 << 10)
    if _json(failure).get('status') != 'failed' or read_status(root).get('status') != 'failed':
        raise ValueError('Pose diagnostic requires a preserved failed preparation')
    if target.exists() or target.is_symlink():
        existing = pose_failure_summary(root, job.get('source_sha256'))
        if existing is None:
            raise ValueError('Existing pose diagnostic cannot be replaced')
        return existing
    evidence = read_package_file(root, 'perception/preflight/body_source/evidence.json', max_bytes=MAX_INPUT_BYTES)
    check()
    raw = read_package_file(root, 'perception/preflight/body_source/pose-landmarks.json',
                            max_bytes=MAX_INPUT_BYTES-len(evidence))
    check()
    report = screen_identity_windows(evidence, raw)
    check()
    if report['source_sha256'] != job.get('source_sha256'):
        raise ValueError('Pose diagnostic source differs from the original upload')
    if not report['identity_barriers']:
        return None
    slots = report['sample_slots']; barriers = report['identity_barriers']; windows = report['windows']
    summary = {'profile': POSE_DIAGNOSTIC, 'source_sha256': report['source_sha256'],
        'sampled_frames': len(slots), 'usable_samples': sum(s['screen_usable'] for s in slots),
        'identity_breaks': len(barriers), 'listed_barriers': [
            {'frame': b['frame'], 'source_time_s': slots[b['sample_index']]['source_time_s'],
             'decoder_timestamp_s': slots[b['sample_index']]['decoder_timestamp_s'],
             'hypothesis_count': len(b['hypothesis_indices'])} for b in barriers[:32]],
        'window_count': len(windows), 'listed_windows': [{k: w[k] for k in (
            'frame_interval', 'source_time_interval_s', 'usable_samples', 'sampled_frames',
            'screen_passed', 'screen_failures')} for w in windows[:16]],
        'source_time_basis': report['source_time_basis'], 'simulation_ready': False,
        'physical_identity_verified': False, 'native_clock_complete': False,
        'message': 'The pose detector returned competing body estimates at these samples. '
                   'We cannot safely choose one or join movement across them. Both alternatives remain saved; '
                   'this does not prove that multiple people were present.',
        'scope': 'Diagnostic only. Window checks cover usable-pose count, coverage and duration—not '
                 'continuous identity, motion, gaps, floor contact or physics. No body was completed and no retry was run.'}
    report_bytes, summary_bytes = _bytes(report), _bytes(summary)
    if len(report_bytes) > 16 << 20 or len(summary_bytes) > 32 << 10:
        raise ValueError('Pose diagnostic exceeds its output bound')
    manifest = {'profile': POSE_DIAGNOSTIC, 'source_sha256': report['source_sha256'],
        'failure_sha256': sha256(failure).hexdigest(), 'evidence_sha256': sha256(evidence).hexdigest(),
        'raw_pose_sha256': sha256(raw).hexdigest(), 'report_sha256': sha256(report_bytes).hexdigest(),
        'summary_sha256': sha256(summary_bytes).hexdigest(), 'api_calls': 0, 'inference_calls': 0,
        'decode_calls': 0, 'simulation_ready': False}
    check()
    # Recheck every input before publishing a snapshot beside immutable failed evidence.
    read_package_file(root, 'perception/failure.json', max_bytes=32 << 10, expected_sha256=manifest['failure_sha256'])
    read_package_file(root, 'perception/preflight/body_source/evidence.json', max_bytes=len(evidence),
                      expected_sha256=manifest['evidence_sha256'])
    read_package_file(root, 'perception/preflight/body_source/pose-landmarks.json', max_bytes=len(raw),
                      expected_sha256=manifest['raw_pose_sha256'])
    check()
    target.mkdir(mode=0o700)
    write_package_files(target, {'report.json': report_bytes, 'summary.json': summary_bytes,
                                 'manifest.json': _bytes(manifest)})
    return summary


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
                 'body_preview': 'preflight/body_source/tracks.gif',
                 'raw_poses': 'preflight/body_source/pose-landmarks.json',
                 'pose_diagnostic': 'pose-diagnostic/report.json', 'pose_diagnostic_summary': 'pose-diagnostic/summary.json',
                 'pose_diagnostic_manifest': 'pose-diagnostic/manifest.json', 'pose_diagnostic_error': 'pose-diagnostic-error.json'}
    result['artifacts'] = {key: 'perception/'+name for key, name in artifacts.items()
                           if (Path(root)/'perception'/name).is_file() and not (Path(root)/'perception'/name).is_symlink()}
    if status:
        result.update({key: status[key] for key in ('stage', 'events', 'error', 'seconds') if key in status})
    if status.get('status') == 'awaiting_consent':
        result['quote'], result['quote_sha256'] = _quote(root)
    if status.get('status') == 'failed':
        result['pose_diagnostic'] = pose_failure_summary(root, job.get('source_sha256'))
    return result


def queue_preflight(root):
    root, job, _ = _context(root)
    if read_status(root):
        return
    work = root/'perception'
    work.mkdir(mode=0o700)
    request = {'profile': PROFILE, 'source_sha256': job['source_sha256'],
               'requested_at': local_time(), 'api_calls_authorized': 0}
    if 'body_reuse' in job:
        request['body_reuse'] = job['body_reuse']
    write_package_files(work, {'parent-status.json': read_package_file(root, 'status.json', max_bytes=32 << 10),
                               'preflight-request.json': _bytes(request)})
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
    from weft.equipment_analysis_contract import LEGACY_PROFILE, SAMPLED_PROFILE
    profile = quote.get('analysis_profile', LEGACY_PROFILE)
    contract_hash = _contract() if profile == SAMPLED_PROFILE else _contract(profile)
    n = quote.get('frame_count')
    if (quote.get('profile') != 'portal-perception-quote-v1' or quote.get('schema_version') != 1
            or quote.get('source_sha256') != job['source_sha256'] or type(n) is not int or not 12 <= n <= 500
            or quote.get('openai_model') != MODEL or quote.get('meta_model') != 'sam-3.1'
            or quote.get('analysis_contract_sha256') != contract_hash
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
            options = {'pose_model': Path(project)/'.local/models/pose_landmarker_lite.task'}
            if ('body_reuse' in job or 'body_reuse' in pre_request
                    or (root/'body-reuse').exists() or (root/'body-reuse').is_symlink()):
                from weft.portal_body_reuse import validate_queued_body
                body = validate_queued_body(root, deadline=min(deadline, time.monotonic()+30))
                options = {'body_source': body, 'retain_body_source': True}
            progress('preflighting', 'Checking native video/body evidence and initial person visibility locally; no media sent.')
            prepared = prepare_exercise_perception_preflight(video, work/'preflight',
                max_seconds=210, anchor_profile=VISIBLE_ANCHOR_PROFILE, **options)
            check()
            n = prepared['metadata']['frame_count']
            from weft.equipment_analysis_contract import SAMPLED_PROFILE
            quote = {'schema_version': 1, 'profile': 'portal-perception-quote-v1', 'source_sha256': job['source_sha256'],
                'frame_count': n, 'preflight_manifest_sha256': prepared['manifest_sha256'],
                'analysis_profile': SAMPLED_PROFILE, 'analysis_contract_sha256': _contract(), 'openai_model': MODEL, 'meta_model': 'sam-3.1',
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
        from weft.equipment_analysis_contract import LEGACY_PROFILE
        analyze_frames(frames, work/'analysis', identity=request['analysis_call_id'], project=project,
                       analysis_profile=quote.get('analysis_profile', LEGACY_PROFILE))
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
        if not paid and status == 'failed' and time.monotonic() < deadline:
            try:
                prepare_pose_failure_diagnostic(root, deadline=min(deadline, time.monotonic()+5))
            except Exception as diagnostic_error:
                # An optional explanation cannot replace the original failure or trigger a retry.
                try:
                    write_package_files(work, {'pose-diagnostic-error.json': _bytes({
                        'profile': POSE_DIAGNOSTIC, 'error': str(diagnostic_error)[:400], 'simulation_ready': False})})
                except (ValueError, OSError):
                    pass
        return 1
