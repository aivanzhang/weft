"""Pure bridge from prepared source metadata and fixed-body geometry to a camera/root proposal.

The parent authenticates source custody and the extraction runtime. This module binds
provided bytes; it never reads files, replays crop proofs, extracts geometry or fits.
"""

from copy import deepcopy
from hashlib import sha256
import math
import time

import numpy as np

from weft import body_camera_root_refinement as proposal
from weft import whole_body_reference_inputs as saved
from weft import whole_body_reference_refinement as reference
from weft.io import canonical_hash


PROFILE = 'fixed-body-camera-root-inputs-v1'
GEOMETRY_PROFILE = 'fixed-original-body-geometry-v1'
CAMERA_PROFILE = 'fixed-body-camera-hypothesis-v1'
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_GEOMETRY_BYTES = 8 * 1024 * 1024
MAX_CAMERA_BYTES = 128 * 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
MAX_SECONDS = 10.
_BODY_BINDINGS = {'source_sha256', 'body_sha256', 'evidence_sha256', 'body_angles_sha256', 'body_shape_sha256'}
_GEOMETRY_BINDINGS = _BODY_BINDINGS | {'root_reference_sha256', 'floor_sha256'}


def _require(condition, message):
    if not condition:
        raise ValueError('Camera/root inputs: ' + message)


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Camera/root input preparation deadline exceeded')


def _hash(raw):
    return sha256(raw).hexdigest()


def _digest(value):
    return isinstance(value, str) and saved._SHA.fullmatch(value) is not None


def _fields(value, fields, label):
    _require(type(value) is dict and set(value) == set(fields), label + ' fields differ')


def _document(raw, maximum, label):
    _require(type(raw) is bytes and 0 < len(raw) <= maximum, label + ' requires bounded exact bytes')
    return saved._json(raw)


def _prepared(snapshot, deadline):
    _require(type(snapshot) is dict and isinstance(snapshot.get('files'), dict)
             and isinstance(snapshot.get('identity'), dict), 'prepared reference inputs required')
    files, identity = snapshot['files'], snapshot['identity']
    _require(identity.get('profile') == saved.PROFILE and identity.get('status') == 'metadata_verified'
             and 1 <= len(files) <= saved.MAX_MEMBERS + 2, 'prepared metadata status or member bound')
    for name, raw in files.items():
        saved._name(name)
        _require(name.startswith('inputs/') and type(raw) is bytes and len(raw) <= saved.MAX_MEDIA_BYTES,
                 'prepared member shape or byte bound')
    consumed, total = {}, 0

    def take(name, expected, maximum=saved.MAX_JSON_BYTES):
        nonlocal total
        _check(deadline)
        _require(name not in consumed and _digest(expected), 'missing or repeated prepared pin')
        raw = files.get(name)
        _require(type(raw) is bytes and 0 < len(raw) <= maximum, 'missing or oversized prepared member: ' + name)
        total += len(raw)
        _require(total <= MAX_METADATA_BYTES, 'selected metadata aggregate byte bound')
        digest = _hash(raw)
        _require(digest == expected, 'prepared member differs from its pin: ' + name)
        consumed[name] = digest
        return raw

    body_manifest = take('inputs/body-manifest.json', identity.get('body_manifest_sha256'), saved.MAX_MANIFEST_BYTES)
    manifest, declared = saved._manifest(body_manifest)
    _require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1
             and manifest.get('profile') == saved.BODY_PROFILE and manifest.get('status') == 'construction_only'
             and manifest.get('simulation_ready') is False and not saved._CONFLICTS & declared.keys(),
             'original construction-only body seal required')
    raw_files = {}
    for name in sorted(saved._REQUIRED | {'body-source.json', saved._PRODUCER}):
        member = 'inputs/' + ('evidence-original.json' if name == 'evidence.json' else name)
        raw_files[name] = take(member, declared.get(name))
    documents = {name: saved._json(raw) for name, raw in raw_files.items() if name.endswith('.json')}
    marker = documents['body-source.json']
    _require(type(marker.get('schema_version')) is int and marker['schema_version'] == 1
             and marker.get('profile') == saved.BODY_PROFILE and marker.get('status') == 'construction_only'
             and marker.get('simulation_ready') is False and marker.get('source_sha256') == identity.get('source_sha256')
             and marker.get('input_kind') == identity.get('input_kind'), 'body/source receipt differs')
    source_name = marker.get('source_artifact')
    _require(source_name in {'source' + suffix for suffix in ('.webm', '.mp4', '.mov', '.m4v', '.avi', '.mkv')}
             and declared.get(source_name) == marker['source_sha256'] == documents['source.json'].get('sha256')
             and documents['source.json'].get('artifact') == source_name, 'source media metadata differs')
    producer = declared[saved._PRODUCER]
    _require(producer in saved._DECODERS and documents['implementation.json'].get('files', {}).get('exercise_vision.py') == producer,
             'unsupported prepared decoder identity')
    recovering = 'pose-recovery.json' in declared
    required = saved._REQUIRED | (saved._RECOVERY_PROOF if recovering else set())
    _require(isinstance(marker.get('artifacts'), dict) and set(marker['artifacts']) == required
             and all(marker['artifacts'][name] == declared.get(name) for name in required), 'body member bindings differ')
    construction = marker.get('construction', {})
    _require(construction.get('body_model') == 'mobile' and construction.get('body_profile') == 'exercise-v3'
             and construction.get('original_pose_config') == 'run-config.json'
             and construction.get('original_pose_config_sha256') == declared['run-config.json'], 'body construction binding differs')
    if marker['input_kind'] == 'sealed_pose_evidence':
        parent_raw = take('inputs/source-manifest.json', identity.get('source_manifest_sha256'), saved.MAX_MANIFEST_BYTES)
        parent, parent_files = saved._manifest(parent_raw)
        _require(marker.get('input_manifest_sha256') == _hash(parent_raw)
                 and all(parent_files.get(name) == declared.get(name) for name in saved._POSE | {source_name}),
                 'original source manifest differs')
    else:
        _require(marker['input_kind'] == 'local_video' and not recovering and marker.get('input_manifest_sha256') is None
                 and identity.get('source_manifest_sha256') == _hash(body_manifest), 'unsupported prepared source route')
    if recovering:
        recovery = identity.get('recovery', {})
        _require(parent.get('profile') == 'exercise-pose-recovery-v1' and parent.get('status') == 'source_evidence_only'
                 and parent.get('simulation_ready') is False
                 and recovery.get('profile') == 'verified-saved-selective-crop-inputs-v1'
                 and recovery.get('status') == 'saved_derivation_verified', 'crop source needs its prepared proof receipt')
        for name in sorted(saved._RECOVERY_PROOF):
            _require(parent_files.get(name) == declared.get(name), 'crop proof parent pin differs')
            take('inputs/' + name, declared.get(name))
        proof_pins = {'marker_sha256': 'pose-recovery.json', 'selected_evidence_sha256': 'evidence.json',
                      'baseline_evidence_sha256': 'recovery/baseline-evidence.json', 'raw_pose_sha256': 'pose-landmarks.json',
                      'plans_sha256': 'recovery/plans.json', 'candidate_journal_sha256': 'recovery/candidates.jsonl',
                      'decisions_sha256': 'recovery/decisions.json', 'proposal_manifest_sha256': 'recovery/input-manifest.json'}
        _require(all(recovery.get(key) == declared[name] for key, name in proof_pins.items()), 'prepared crop proof identity differs')
    saved._pose_bindings(documents, raw_files, marker, recovery_verified=recovering)
    human_raw = raw_files['human.json']
    _require(identity.get('human_sha256') == _hash(human_raw)
             and identity.get('evidence_original_sha256') == _hash(raw_files['evidence.json']), 'prepared document identity differs')
    evidence_raw = take('inputs/evidence-derived.json', identity.get('evidence_derived_sha256'))
    evidence = saved._json(evidence_raw)
    _require({key: value for key, value in evidence.items() if key != 'image_dimension_provenance'} == documents['evidence.json'],
             'derived evidence changed original observations')
    dimensions = evidence.get('image_dimension_provenance', {})
    _require(dimensions.get('method') == 'sealed_decoder_metadata'
             and dimensions.get('source_evidence_sha256') == identity['evidence_original_sha256']
             and dimensions.get('source_manifest_sha256') == identity['source_manifest_sha256'], 'dimension provenance differs')
    return documents['human.json'], evidence, human_raw, evidence_raw, deepcopy(identity), consumed, total


def _floor(spec):
    mobility = spec.get('mobility', {})
    _require(mobility.get('floor_height') == 0, 'original model floor must remain at zero')
    normal = proposal._array(mobility.get('ground_plane', {}).get('normal', [0., 0., 1.]), (3,), 1.)
    norm = float(np.linalg.norm(normal))
    _require(norm > 0 and normal[2] > 0, 'original fixed floor normal is invalid')
    return {'normal': (normal / norm).tolist(), 'offset_model_units': 0.}


def _runtime(runtime):
    _fields(runtime, {'implementation_sha256', 'versions', 'model_xml_sha256'}, 'geometry runtime')
    versions, pins = runtime['versions'], runtime['implementation_sha256']
    _fields(versions, {'python', 'numpy', 'mujoco', 'gymnasium'}, 'runtime versions')
    _require(all(isinstance(value, str) and 1 <= len(value) <= 128 and all(ord(c) >= 32 for c in value)
                 for value in versions.values()), 'bounded runtime versions required')
    _require(type(pins) is dict and 1 <= len(pins) <= 64 and _digest(runtime['model_xml_sha256']), 'geometry runtime pins required')
    for name, digest in pins.items():
        saved._name(name)
        _require(name.endswith('.py') and len(name) <= 160 and _digest(digest), 'invalid geometry implementation pin')


def prepare_camera_root_document(prepared_reference_inputs, camera_hypothesis_bytes, geometry_snapshot_bytes, *, deadline):
    """Bind provided metadata and geometry; the parent retains source/runtime custody and rechecks."""
    started = time.monotonic()
    _require(type(deadline) in (int, float) and math.isfinite(deadline), 'finite monotonic deadline required')
    deadline = min(deadline, started + MAX_SECONDS)
    _check(deadline)
    spec, evidence, human_raw, evidence_raw, identity, consumed, metadata_bytes = _prepared(prepared_reference_inputs, deadline)
    times, _, roots, targets, root_audit = reference._prepare(spec, evidence)
    schedule, _, _ = reference._schedule(times)
    floor = _floor(spec)
    bindings = {'source_sha256': spec['source_sha256'], 'body_sha256': _hash(human_raw), 'evidence_sha256': _hash(evidence_raw),
                'body_angles_sha256': canonical_hash(spec['reference']),
                'body_shape_sha256': canonical_hash({k: v for k, v in spec.items() if k not in {'reference', 'root_reference'}}),
                'root_reference_sha256': canonical_hash(spec['root_reference']), 'floor_sha256': canonical_hash(floor)}
    camera = _document(camera_hypothesis_bytes, MAX_CAMERA_BYTES, 'camera hypothesis')
    _fields(camera, {'schema_version', 'profile', 'bindings', 'image', 'coordinate_frame', 'scale_model_units_per_unit',
                     'partition', 'selection', 'camera'}, 'camera hypothesis')
    _require(type(camera['schema_version']) is int and camera['schema_version'] == 1 and camera['profile'] == CAMERA_PROFILE
             and camera['bindings'] == {k: bindings[k] for k in ('source_sha256', 'body_sha256', 'evidence_sha256')}, 'camera identity differs')
    _require(camera['image'] == {k: evidence[k] for k in ('width', 'height', 'fps', 'frame_count')}
             and camera['coordinate_frame'] == 'original_body_model_units'
             and type(camera['scale_model_units_per_unit']) in (int, float) and camera['scale_model_units_per_unit'] == 1.
             and camera['partition'] == 'exposed_development' and camera['selection'] == 'explicit_before_proposal_no_search'
             and isinstance(camera['camera'], dict) and camera['camera'].get('quality') == 'authored', 'explicit fixed authored camera required')
    geometry = _document(geometry_snapshot_bytes, MAX_GEOMETRY_BYTES, 'geometry snapshot')
    _fields(geometry, {'schema_version', 'profile', 'status', 'bindings', 'floor', 'source_rows', 'support_names',
                       'floor_rows', 'runtime', 'physics_steps'}, 'geometry snapshot')
    _require(type(geometry['schema_version']) is int and geometry['schema_version'] == 1 and geometry['profile'] == GEOMETRY_PROFILE
             and geometry['status'] == 'completed' and geometry['bindings'] == bindings and geometry['floor'] == floor
             and type(geometry['physics_steps']) is int and geometry['physics_steps'] == 0, 'fixed original geometry identity differs')
    _runtime(geometry['runtime'])
    source_rows, support_names = geometry['source_rows'], geometry['support_names']
    _require(type(source_rows) is list and len(source_rows) == len(times), 'complete geometry source rows required')
    _require(type(support_names) is list and 1 <= len(support_names) <= 64
             and all(isinstance(v, str) and 1 <= len(v) <= 100 and all(ord(c) >= 32 for c in v) for v in support_names)
             and len(set(support_names)) == len(support_names), 'complete unique support names required')
    floor_rows = geometry['floor_rows']
    _require(type(floor_rows) is list and len(floor_rows) == len(schedule), 'complete source/grid floor rows required')
    repeated = {}
    for row, (kind, stamp, _) in zip(floor_rows, schedule, strict=True):
        _check(deadline)
        _fields(row, {'kind', 'time_s', 'gaps'}, 'floor row')
        _require(row['kind'] == kind and type(row['time_s']) in (int, float) and row['time_s'] == stamp
                 and type(row['gaps']) is dict and set(row['gaps']) == set(support_names), 'floor schedule or support inventory differs')
        _require(stamp not in repeated or row['gaps'] == repeated[stamp], 'duplicate-time floor values differ')
        repeated[stamp] = row['gaps']
    frames = {frame['id']: frame for frame in evidence['frames']}
    samples, provenance_rows, known = [], [], 0
    for i, (row, target) in enumerate(zip(source_rows, targets, strict=True)):
        _check(deadline)
        _fields(row, {'evidence_id', 'frame', 'time_s', 'decoder_timestamp_s', 'root', 'points'}, 'geometry source row')
        frame = frames[target['evidence_id']]
        _require(row['evidence_id'] == frame['id'] and type(row['frame']) is int and row['frame'] == frame['frame']
                 and type(row['time_s']) in (int, float) and row['time_s'] == frame['time_s']
                 and (row['decoder_timestamp_s'] is None or type(row['decoder_timestamp_s']) in (int, float))
                 and row['decoder_timestamp_s'] == frame['decoder_timestamp_s'] and row['root'] == roots[i].tolist(),
                 'geometry source identity, root or clock differs')
        _fields(row['points'], reference.POINTS, 'geometry endpoint inventory')
        points, details = [], []
        selected = frame.get('identity_raw', frame)
        source_path = 'identity_raw.landmarks' if 'identity_raw' in frame else 'landmarks'
        for j, (name, index) in enumerate(reference.POINTS.items()):
            weight, item = float(target['weights'][j]), target['points'][j]
            pixel = ([selected['landmarks'][index]['x'] * evidence['width'], selected['landmarks'][index]['y'] * evidence['height']]
                     if weight > 0 else None)
            known += weight > 0
            points.append({'name': name, 'world': row['points'][name], 'pixel': pixel, 'weight': weight})
            details.append({'name': name, 'status': item['status'], 'source_confidence': item['source_confidence'], 'weight': weight, 'pixel': pixel})
        record = {key: row[key] for key in ('evidence_id', 'frame', 'time_s', 'decoder_timestamp_s', 'root')}
        samples.append({**record, 'points': points})
        provenance_rows.append({key: row[key] for key in ('evidence_id', 'frame', 'time_s', 'decoder_timestamp_s')} | {
            'root_status': target['root_status'], 'target_origin': target.get('target_origin'), 'landmark_source': source_path, 'points': details})
    document = {'schema_version': 1, 'profile': proposal.INPUT_PROFILE, 'bindings': {k: bindings[k] for k in sorted(_BODY_BINDINGS)},
                'scale_model_units_per_unit': 1., 'camera': camera['camera'], 'floor': floor, 'samples': samples,
                'floor_checks': {'bindings': {k: bindings[k] for k in sorted(_GEOMETRY_BINDINGS - {'source_sha256', 'evidence_sha256'})}, 'rows': floor_rows}}
    input_bytes = saved._bytes(document)
    proposal._load(input_bytes, deadline)  # Reuse the frozen numeric schema; never invoke its solver.
    reference_ids = {row['evidence_id'] for row in source_rows}
    retained = [{key: frame[key] for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s', 'status')} | {
                    'has_body_reference': frame['id'] in reference_ids}
                for frame in evidence['frames']]
    retained_indices = {frame['frame'] for frame in evidence['frames']}
    slots = {'schema_version': 1, 'profile': 'camera-root-source-slots-v1', 'source_sha256': bindings['source_sha256'],
             'declared_frame_count': evidence['frame_count'], 'decoded_frames': evidence['decoded_frames'],
             'inference_stride': evidence['inference_stride'], 'retained_frames': retained,
             'unsampled_decoded_frame_indices': [i for i in range(evidence['decoded_frames']) if i not in retained_indices],
             'container_only_frame_indices': list(range(evidence['decoded_frames'], evidence['frame_count'])),
             'scope': 'Original retained pose slots and declared decoder coverage; unsampled slots have no invented pixels or native clock.'}
    pins = {'prepared_files': consumed, 'camera_hypothesis_sha256': _hash(camera_hypothesis_bytes),
            'geometry_snapshot_sha256': _hash(geometry_snapshot_bytes), 'input_sha256': _hash(input_bytes)}
    provenance = {'schema_version': 1, 'profile': PROFILE, 'status': 'prepared_metadata_bound', 'bindings': bindings,
                  **{k: v for k, v in pins.items() if k != 'prepared_files'}, 'body_manifest_sha256': identity['body_manifest_sha256'],
                  'image': {k: evidence[k] for k in ('width', 'height', 'fps', 'frame_count', 'decoded_frames', 'inference_stride')} | {
                      'dimension_provenance': evidence['image_dimension_provenance']},
                  'source_rows': provenance_rows, 'runtime': geometry['runtime'], 'prepared_file_pins': consumed,
                  'original_image_scale': root_audit['scale'], 'fixed_hash_recipes': {
                      'body_sha256': 'Exact original human bytes', 'evidence_sha256': 'Exact prepared derived-evidence bytes',
                      'body_angles_sha256': 'canonical_hash(human.reference)',
                      'body_shape_sha256': 'canonical_hash(all human fields except reference and root_reference)',
                      'root_reference_sha256': 'canonical_hash(human.root_reference)', 'floor_sha256': 'canonical_hash(floor)'},
                  'counts': {'source_rows': len(samples), 'control_grid_rows': len(schedule) - len(samples), 'floor_rows': len(schedule),
                             'retained_source_slots': len(retained), 'known_endpoints': known, 'unknown_endpoints': 12 * len(samples) - known,
                             'metadata_bytes_consumed': metadata_bytes, 'grid_image_targets': 0},
                  'camera_scope': 'Explicit authored hypothesis; not measured calibration',
                  'geometry_scope': 'Provided fixed-body geometry bindings checked; no geometry executed by this adapter',
                  'runtime_scope': 'Declared extraction identities; the trusted worker verifies installed bytes and versions',
                  'source_scope': 'Prepared selected-source metadata custody only; no external authenticity, body correctness or admission',
                  'source_qualified': False, 'admitted': False, 'promoted': False, 'physical_validation': 'unverified'}
    result = {'input_document_bytes': input_bytes, 'provenance_bytes': saved._bytes(provenance),
              'source_slots_bytes': saved._bytes(slots), 'pins': pins}
    _require(sum(len(value) for key, value in result.items() if key.endswith('_bytes')) <= MAX_OUTPUT_BYTES, 'combined output byte bound')
    _check(deadline)
    return result
