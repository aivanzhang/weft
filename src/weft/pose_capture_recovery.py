"""Recover complete pose measurements from a pinned interrupted study, never its body result."""
from pathlib import Path
import math
import re
import time

from weft.shared_world_packaging import read_package_file, validate_package_layout, write_package_files

PROFILE = 'interrupted-pose-capture-recovery-v1'
PRODUCERS = ('exercise_vision.py', 'source_clock.py', 'source_clock_capture.py', 'exercise_body_source.py')
EXACT_PRODUCERS = PRODUCERS[:3]
PROOF_FILES = {'pose-capture-recovery.json', 'capture-recovery/study-manifest.json',
               'capture-recovery/resource-failure.json', 'capture-recovery/implementation.json'} | {
                   'capture-recovery/implementation-source/' + name for name in PRODUCERS}
MAX_BYTES = 128 << 20
SCOPE = 'Complete source measurements recovered from a failed resource-bounded study; no body, floor, identity or physical qualification'


def _manifest(raw):
    from weft.exercise_body_source import _json
    value = _json(raw)
    declared = value.get('files')
    if (type(value.get('schema_version')) is not int or value['schema_version'] != 1
            or value.get('profile') != 'dense-pose-partial-study-seal-v1'
            or not isinstance(declared, dict) or not 1 <= len(declared) <= 512
            or any(not isinstance(pin, str) or not re.fullmatch('[0-9a-f]{64}', pin) for pin in declared.values())):
        raise ValueError('Recovery requires an exact bounded interrupted-study seal')
    validate_package_layout(declared)
    if any(len(name.split('/')) > 8 for name in declared):
        raise ValueError('Interrupted study exceeds the path-depth bound')
    return declared


def validate_capture_recovery_files(files, *, source_sha256, deadline=None):
    """Validate retained failed-study ancestry; callers still enforce ordinary pose/body gates."""
    from weft.exercise_body_source import POSE_FILES, CAPTURE_FILES, _json, _sha, _deadline
    if deadline is not None:
        _deadline(deadline)
    if not PROOF_FILES <= files.keys():
        raise ValueError('Capture recovery requires its complete failed-study ancestry')
    receipt = _json(files['pose-capture-recovery.json'])
    if (set(receipt) != {'schema_version', 'profile', 'status', 'source_sha256', 'source_artifact',
                        'study_manifest_sha256', 'input_files', 'producer_policy', 'original_outcome', 'work', 'scope'}
            or type(receipt.get('schema_version')) is not int or receipt['schema_version'] != 1 or receipt.get('profile') != PROFILE
            or receipt.get('status') != 'pose_evidence_only' or receipt.get('source_sha256') != source_sha256
            or receipt.get('original_outcome') != 'resource_interrupted_before_body_seal'
            or receipt.get('work') != {'video_decodes': 0, 'inference_calls': 0, 'physics_steps': 0, 'api_calls': 0}
            or any(type(value) is not int for value in receipt.get('work', {}).values()) or receipt.get('scope') != SCOPE
            or receipt.get('producer_policy') != 'Exact current vision/native-clock/capture; historical constructor is retained, never executed'):
        raise ValueError('Capture recovery cannot claim completed body construction or altered work')
    original = files['capture-recovery/study-manifest.json']
    if _sha(original) != receipt['study_manifest_sha256']:
        raise ValueError('Interrupted source seal changed')
    declared = _manifest(original)
    source_name = receipt['source_artifact']
    if source_name not in {'source.mp4', 'source.webm', 'source.mov', 'source.m4v', 'source.avi', 'source.mkv'}:
        raise ValueError('Recovery source artifact is invalid')
    required = POSE_FILES | CAPTURE_FILES | {'native-clock.json', source_name}
    if receipt['input_files'] != {name: 'body_source/' + name for name in sorted(required)}:
        raise ValueError('Recovery must retain exactly all original source/pose/capture inputs')
    for name, original_name in receipt['input_files'].items():
        if name not in files or _sha(files[name]) != declared.get(original_name):
            raise ValueError('Recovered source/pose/capture bytes differ from original study')
    if _sha(files[source_name]) != source_sha256:
        raise ValueError('Recovered original video identity changed')
    if any('body_source/' + name in declared for name in ('manifest.json', 'body-source.json', 'failure.json')):
        raise ValueError('Recovery is only for interrupted unsealed capture, not rejected/completed bodies')
    failure = files['capture-recovery/resource-failure.json']
    if _sha(failure) != declared.get('resource-failure.json'):
        raise ValueError('Original interruption receipt changed')
    event = _json(failure)
    if (set(event) != {'reason', 'stage', 'seconds', 'peak_rss_bytes'} or event.get('stage') != 'body'
            or not isinstance(event.get('reason'), str) or not 1 <= len(event['reason']) <= 500
            or type(event.get('seconds')) not in (int, float) or not math.isfinite(event['seconds']) or not 0 < event['seconds'] <= 210
            or type(event.get('peak_rss_bytes')) is not int or not 0 < event['peak_rss_bytes'] <= 2 << 30):
        raise ValueError('Recovery requires the retained bounded resource-interruption event')
    implementation = files['capture-recovery/implementation.json']
    if _sha(implementation) != declared.get('body_source/implementation.json'):
        raise ValueError('Original producer inventory changed')
    pins = _json(implementation).get('files', {})
    for name in PRODUCERS:
        raw = files['capture-recovery/implementation-source/' + name]
        if _sha(raw) != declared.get('body_source/implementation-source/' + name) or _sha(raw) != pins.get(name):
            raise ValueError('Original producer snapshot differs from both independent pins')
        if name in EXACT_PRODUCERS and raw != Path(__file__).with_name(name).read_bytes():
            raise ValueError('Original observation/capture producer requires explicit compatibility review')
    if deadline is not None:
        _deadline(deadline)
    return receipt


def prepare_pose_capture_recovery(study_root, out, *, expected_study_manifest_sha256, source_video, max_seconds=30):
    """Publish new pose-only evidence without decoding, inference, body fitting or executing snapshots."""
    from weft.exercise_body_source import (POSE_FILES, CAPTURE_FILES, _bytes, _json, _sha, _path, _deadline,
        _options, _validated_pose, _validate_capture_files)
    if (type(max_seconds) not in (int, float) or not math.isfinite(max_seconds) or not 1 <= max_seconds <= 30
            or not isinstance(expected_study_manifest_sha256, str)
            or not re.fullmatch('[0-9a-f]{64}', expected_study_manifest_sha256)):
        raise ValueError('Recovery requires a caller-pinned study seal and 1–30 second bound')
    started = time.monotonic(); deadline = started + max_seconds
    study, target, video = (_path(p) for p in (study_root, out, source_video))
    if target.exists() or any(target == p or target.is_relative_to(p) or p.is_relative_to(target) for p in (study, video)):
        raise ValueError('Recovery requires a new output outside every input')
    original = read_package_file(study, 'manifest-study.json', max_bytes=1 << 20,
                                 expected_sha256=expected_study_manifest_sha256)
    declared = _manifest(original)
    for name in ('manifest.json', 'body-source.json', 'failure.json'):
        if (study/'body_source'/name).exists() or (study/'body_source'/name).is_symlink():
            raise ValueError('Original body is not an interrupted unsealed capture')
    source_meta = _json(read_package_file(study, 'body_source/source.json', max_bytes=1 << 20,
                                        expected_sha256=declared.get('body_source/source.json')))
    source_name, source_sha = source_meta.get('artifact'), source_meta.get('sha256')
    if source_name not in {'source.mp4', 'source.webm', 'source.mov', 'source.m4v', 'source.avi', 'source.mkv'}:
        raise ValueError('Original source artifact is invalid')
    wanted = POSE_FILES | CAPTURE_FILES | {'native-clock.json', source_name}
    mapping = {name: 'body_source/' + name for name in sorted(wanted)}
    proof_mapping = {'capture-recovery/resource-failure.json': 'resource-failure.json',
                     'capture-recovery/implementation.json': 'body_source/implementation.json',
                     **{'capture-recovery/implementation-source/' + name: 'body_source/implementation-source/' + name for name in PRODUCERS}}
    reverse = {old: new for new, old in {**mapping, **proof_mapping}.items()}
    files = {'capture-recovery/study-manifest.json': original}; total = len(original)
    # Verify every source-study member, retaining only the complete pose/capture ancestry.
    for name, pin in sorted(declared.items()):
        _deadline(deadline)
        raw = read_package_file(study, name, max_bytes=min(64 << 20, max(1, MAX_BYTES-total)), expected_sha256=pin)
        total += len(raw)
        if total > MAX_BYTES:
            raise ValueError('Interrupted study exceeds the aggregate128MiB bound')
        if name in reverse:
            files[reverse[name]] = raw
    original_video = read_package_file(video.parent, video.name, max_bytes=64 << 20, expected_sha256=source_sha)
    if source_name not in files or files[source_name] != original_video:
        raise ValueError('Original external video differs from saved source bytes')
    del original_video, raw
    receipt = {'schema_version': 1, 'profile': PROFILE, 'status': 'pose_evidence_only',
        'source_sha256': source_sha, 'source_artifact': source_name,
        'study_manifest_sha256': expected_study_manifest_sha256, 'input_files': mapping,
        'producer_policy': 'Exact current vision/native-clock/capture; historical constructor is retained, never executed',
        'original_outcome': 'resource_interrupted_before_body_seal',
        'work': {'video_decodes': 0, 'inference_calls': 0, 'physics_steps': 0, 'api_calls': 0},
        'scope': SCOPE}
    files['pose-capture-recovery.json'] = _bytes(receipt)
    validate_capture_recovery_files(files, source_sha256=source_sha, deadline=deadline)
    _validate_capture_files(files, source_sha)
    config = _json(files['run-config.json'])
    options = _options(config.get('pose_mode', 'video'), config.get('pose_identity', 'raw'),
                       config.get('max_pose_hypotheses', 2), config.get('person_region'), config.get('pose_sampling', 'bounded'))
    _validated_pose(files, options, source_sha, deadline=deadline)
    if read_package_file(study, 'manifest-study.json', max_bytes=1 << 20) != original:
        raise ValueError('Interrupted study seal changed during recovery')
    # A separate manifest identifies recovered measurements, never a recovered successful body.
    manifest = _bytes({'schema_version': 1, 'profile': PROFILE, 'status': 'pose_evidence_only',
                      'simulation_ready': False, 'files': {name: _sha(raw) for name, raw in files.items()}})
    if sum(map(len, files.values())) + len(manifest) > MAX_BYTES:
        raise ValueError('Recovered measurements exceed aggregate128MiB')
    for name in ('manifest.json', 'body-source.json', 'failure.json'):
        if (study/'body_source'/name).exists() or (study/'body_source'/name).is_symlink():
            raise ValueError('Original body completed or failed during capture recovery')
    _deadline(deadline); target.mkdir(parents=True)
    write_package_files(target, files)
    _deadline(deadline); write_package_files(target, {'manifest.json': manifest})
    return target
