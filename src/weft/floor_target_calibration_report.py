"""Saved calibration inspection; no solver, source processing or installation."""

from collections import Counter
from hashlib import sha256
import html
import json
import math
from pathlib import Path, PurePosixPath
import re
import time

from weft.io import canonical_hash


PROFILE = 'floor-target-calibration-report-v1'
MAX_INPUT = MAX_OUTPUT = 4 << 20
MAX_RESULT, MAX_HTML, MAX_FILES = 1 << 20, 2 << 20, 16
MAX_FRAMES, PAGE_FRAMES = 1800, 200
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _require(condition, message):
    if not condition:
        raise ValueError('Floor calibration report: ' + message)


def _encode(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':')).encode()


def _document(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        _encode(value)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError('Floor calibration report: finite strict JSON required') from error
    _require(type(value) is dict, 'JSON object required')
    return value


def _pin(value):
    _require(type(value) is str and _SHA.fullmatch(value), 'exact SHA256 required')
    return value


def _text(value, maximum=1024, *, nullable=False):
    if value is None and nullable:
        return value
    _require(type(value) is str and 0 < len(value.strip()) <= maximum,
             'bounded text required')
    return value


def _integer(value, maximum=MAX_FRAMES, minimum=0):
    _require(type(value) is int and minimum <= value <= maximum, 'bounded integer required')
    return value


def _number(value, maximum=1e9, *, nullable=False):
    if value is None and nullable:
        return value
    _require(type(value) in (int, float) and math.isfinite(value) and abs(value) <= maximum,
             'bounded finite number required')
    return value


def _safe_path(value):
    _text(value, 256)
    path = PurePosixPath(value)
    _require(not path.is_absolute() and str(path) == value and '..' not in path.parts
             and '\\' not in value and not any(c in value for c in ('?', '#', ':', '%')),
             'safe fixed relative artifact path required')
    return value


def _pretty(value):
    return html.escape(json.dumps(value, allow_nan=False, sort_keys=True, indent=2))


def _inventory(document, request, request_raw):
    _require(set(document) == {'schema_version', 'profile', 'files'} and document['schema_version'] == 1
             and type(document['schema_version']) is int
             and document['profile'] == 'floor-target-calibration-input-files-v1', 'input inventory schema differs')
    rows = document['files']
    _require(type(rows) is list and 1 <= len(rows) <= 19, 'bounded input inventory required')
    paths, by_role = set(), {role:[] for role in ('observation', 'measurement', 'request', 'camera', 'ground')}
    for row in rows:
        _require(type(row) is dict and set(row) == {'role', 'path', 'sha256', 'bytes'}
                 and row['role'] in by_role, 'input inventory member differs')
        role, name, pin = row['role'], _safe_path(row['path']), _pin(row['sha256'])
        _integer(row['bytes'], MAX_INPUT, 1)
        expected = ('inputs/' + role + '.json' if role in ('request', 'camera', 'ground') else
                    'inputs/' + ('observations' if role == 'observation' else 'measurements') + '/' + pin + '.json')
        _require(name == expected and name not in paths, 'input path must retain its exact role and raw identity')
        paths.add(name); by_role[role].append(row)
    _require(len(by_role['request']) == 1 and by_role['request'][0]['sha256'] == sha256(request_raw).hexdigest()
             and by_role['request'][0]['bytes'] == len(request_raw), 'exact original request inventory differs')
    for role, field in (('observation', 'observation_sha256s'), ('measurement', 'measurement_sha256s')):
        pins = request[field]
        _require(type(pins) is list and len(pins) <= 8, 'bounded input identities required')
        for pin in pins:
            _pin(pin)
        _require(len(pins) == len(set(pins)), 'distinct input identities required')
        _require(sorted(r['sha256'] for r in by_role[role]) == sorted(pins), 'complete requested input inventory differs')
    for role in ('camera', 'ground'):
        pin = request['supplied_' + role + '_sha256']
        _require(len(by_role[role]) == (0 if pin is None else 1), 'supplied input inventory differs')
        if pin is not None:
            _require(by_role[role][0]['sha256'] == _pin(pin), 'supplied input identity differs')
    return rows


def _fields(value, names, label):
    _require(type(value) is dict and set(value) == set(names), label + ' fields differ')


def _reference(value):
    _fields(value, ('document_sha256', 'record_id'), 'measurement reference')
    return _pin(value['document_sha256']), _text(value['record_id'], 160)


def _vector(value, length, *, nullable=False, maximum=float('inf')):
    if value is None and nullable:
        return value
    _require(type(value) is list and len(value) == length, 'numeric vector shape differs')
    for number in value:
        _number(number, maximum)
    return value


def _same(actual, expected, label):
    if expected is None:
        _require(actual is None, label + ' must retain unknown')
    else:
        _number(actual, float('inf'))
        _require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9), label + ' arithmetic differs')


def _request(value):
    _fields(value, ('schema_version', 'profile', 'source_sha256', 'mode', 'observation_sha256s',
        'measurement_sha256s', 'supplied_camera_sha256', 'supplied_ground_sha256', 'world_frame',
        'anchor_frame', 'check_frames', 'evaluation_frames', 'fixed_interval', 'measurement_refs'), 'request')
    _require(type(value['schema_version']) is int and value['schema_version'] == 1
             and value['profile'] == 'floor-target-calibration-request-v1'
             and value['mode'] in ('supplied', 'planar_target'), 'request profile or mode differs')
    _pin(value['source_sha256'])
    _fields(value['world_frame'], ('id', 'length_unit', 'axes'), 'world frame')
    _text(value['world_frame']['id'], 160)
    _require(value['world_frame']['length_unit'] == 'm' and value['world_frame']['axes'] == 'x_right_y_away_z_up',
             'explicit metre world required')
    _fields(value['measurement_refs'], ('camera', 'target', 'target_to_world', 'floor'), 'selected measurements')
    for ref in value['measurement_refs'].values():
        if ref is not None:
            _reference(ref)
    selected = [] if value['anchor_frame'] is None else [_integer(value['anchor_frame'], MAX_FRAMES-1)]
    for name, cap in (('check_frames', 7), ('evaluation_frames', 8)):
        _require(type(value[name]) is list and len(value[name]) <= cap, 'bounded explicit frame selections required')
        selected.extend(_integer(frame, MAX_FRAMES-1) for frame in value[name])
    _require(len(selected) == len(set(selected)), 'construction and unused evaluation frames must be distinct')
    if value['mode'] == 'supplied':
        _require(not selected and value['fixed_interval'] is None and value['measurement_refs']['target'] is None
                 and value['measurement_refs']['target_to_world'] is None, 'supplied calibration does not fit a target')
    else:
        _require(value['supplied_camera_sha256'] is None and value['supplied_ground_sha256'] is None,
                 'target fitting cannot substitute supplied sidecars')
    return selected


_UNCERTAINTY = {
    'camera_calibration': {'intrinsics_px', 'radial_k1', 'translation_m', 'rotation_rad'},
    'planar_target': {'point_position', 'scale_relative'},
    'target_to_world': {'translation_m', 'rotation_rad'},
    'floor_plane': {'height_m', 'normal_rad'}, 'target_corners': {'pixel_error_note'},
}


def _records(rows, request, check):
    _require(type(rows) is list and len(rows) <= 1804, 'bounded measurement headers required')
    by_id = {}
    for row in rows:
        check()
        _fields(row, ('document_sha256', 'id', 'kind', 'quality', 'method', 'provenance', 'uncertainty', 'evidence_refs', 'target_units'), 'measurement header')
        key = _pin(row['document_sha256']), _text(row['id'], 160)
        _require(key not in by_id and key[0] in request['measurement_sha256s'] and row['kind'] in _UNCERTAINTY
                 and row['quality'] in ('measured', 'authored', 'unknown'), 'measurement identity/kind/quality differs')
        _text(row['method'], 1000); _text(row['provenance'], 2000)
        _require(row['target_units'] in ('m', 'mm', None) if row['kind'] == 'planar_target' else row['target_units'] is None,
                 'target uncertainty units differ')
        uncertainty = row['uncertainty']
        _fields(uncertainty, _UNCERTAINTY[row['kind']] | {'interpretation'}, 'declared uncertainty')
        _require(uncertainty['interpretation'] in ('declared_absolute_bound', 'standard_deviation', 'unknown'),
                 'uncertainty interpretation differs')
        for name in _UNCERTAINTY[row['kind']]:
            value = uncertainty[name]
            if value is not None:
                if name == 'pixel_error_note':
                    _text(value, 2000)
                else:
                    _require(_number(value, 1e6) >= 0, 'declared uncertainty cannot be negative')
        refs = row['evidence_refs']
        _require(type(refs) is list and len(refs) <= 32, 'bounded evidence references required')
        seen = set()
        for ref in refs:
            _fields(ref, ('document_sha256', 'evidence_id'), 'evidence reference')
            identity = _pin(ref['document_sha256']), _text(ref['evidence_id'], 240)
            _require(identity not in seen, 'duplicate evidence identity')
            seen.add(identity)
        by_id[key] = row
    return by_id


def _frames(rows, source, request, records, check):
    _require(type(rows) is list and len(rows) == source['retained_frames'], 'complete original frame coverage required')
    prepared, known = [], 0
    for index, row in enumerate(rows):
        check()
        _fields(row, ('source_frame', 'source_status', 'nominal_time_s', 'role', 'partition', 'observation_id',
                     'source_time_s', 'measurement_ref', 'status', 'reason', 'points'), 'frame')
        role = ('anchor' if index == request['anchor_frame'] else 'construction_check' if index in request['check_frames']
                else 'evaluation_not_used' if index in request['evaluation_frames'] else 'not_selected')
        _require(type(row['source_frame']) is int and row['source_frame'] == index and row['role'] == role
                 and row['source_status'] == ('decoded' if index < source['decoded_frames'] else 'metadata_only_not_decoded'),
                 'original frame identity or role differs')
        _same(row['nominal_time_s'], None if source['fps'] is None else index/source['fps'], 'nominal source clock')
        _require(row['partition'] is None or type(row['partition']) is str and len(row['partition']) <= 80,
                 'bounded original partition label required')
        _text(row['observation_id'], 240, nullable=True)
        _number(row['source_time_s'], 60, nullable=True); _text(row['reason'], nullable=True)
        _require(row['source_time_s'] is None or row['source_time_s'] >= 0, 'source time cannot be negative')
        _require(row['status'] in ('unavailable', 'observed', 'missing', 'occluded', 'prepared'), 'unknown frame status')
        if row['measurement_ref'] is not None:
            ref = _reference(row['measurement_ref'])
            _require(ref in records and records[ref]['kind'] == 'target_corners', 'frame measurement identity differs')
        points = row['points']
        _require(type(points) is list and len(points) <= 64, 'bounded point headers required')
        seen = set()
        for point in points:
            _fields(point, ('corner_id', 'status', 'reason'), 'point header')
            _text(point['corner_id'], 160); _text(point['reason'], nullable=True)
            _require(point['corner_id'] not in seen and point['status'] in ('observed', 'missing', 'occluded'),
                     'point identity/status differs')
            seen.add(point['corner_id'])
        if row['status'] == 'prepared':
            _require(role in ('anchor', 'construction_check') and row['reason'] is None
                     and row['source_status'] == 'decoded' and row['partition'] == 'construction'
                     and row['measurement_ref'] is not None and row['observation_id'] is not None
                     and row['source_time_s'] is not None, 'prepared construction frame lacks its source joins')
            count = sum(p['status'] == 'observed' for p in points)
            _require(count >= 4, 'prepared frame needs four known corners')
            prepared.append(index); known += count
        elif points:
            _require(role in ('anchor', 'construction_check') and row['status'] == 'observed'
                     and row['reason'] is None and row['source_status'] == 'decoded' and row['partition'] == 'construction'
                     and row['measurement_ref'] is not None and row['observation_id'] is not None
                     and row['source_time_s'] is not None, 'unprepared/excluded point headers lack eligible construction joins')
    selected = ([] if request['anchor_frame'] is None else [request['anchor_frame']]) + request['check_frames']
    prepared_set = set(prepared)
    return [frame for frame in selected if frame in prepared_set], known


def _alternatives(rows, frames, source, prepared, check):
    _require(type(rows) is list and len(rows) <= 2, 'at most two retained pose alternatives')
    projected = 0
    for index, row in enumerate(rows):
        check()
        _fields(row, ('index', 'R_ct', 't_ct', 'R_cw', 't_cw', 'duplicate_of', 'status', 'reasons', 'checks', 'projections'), 'pose alternative')
        _require(type(row['index']) is int and row['index'] == index and row['status'] in ('rejected', 'nominal_checks_passed'),
                 'alternative identity/status differs')
        for name in ('R_ct', 'R_cw'):
            matrix = row[name]
            if matrix is not None:
                _require(type(matrix) is list and len(matrix) == 3, 'rotation matrix shape differs')
                for vector in matrix:
                    _vector(vector, 3)
        for name in ('t_ct', 't_cw'):
            _vector(row[name], 3, nullable=True)
        reasons = row['reasons']
        _require(type(reasons) is list and len(reasons) <= 8, 'bounded alternative reasons required')
        for reason in reasons:
            _text(reason)
        checks = row['checks']
        _fields(checks, ('proper_rotation', 'depth', 'projection_domain', 'construction_residual', 'rms_px', 'maximum_px'), 'alternative checks')
        _require(all(type(checks[k]) is bool for k in ('proper_rotation', 'depth', 'projection_domain', 'construction_residual')),
                 'boolean saved checks required')
        projections = row['projections']
        _require(type(projections) is list and len(projections) <= 8, 'bounded construction comparisons required')
        if checks['proper_rotation']:
            _require([p['source_frame'] for p in projections] == prepared and all(row[k] is not None for k in ('R_ct', 't_ct', 'R_cw', 't_cw')),
                     'complete prepared-frame projections required')
        else:
            _require(not projections and row['status'] == 'rejected', 'invalid solver geometry cannot claim projections')
        errors, depth_ok, domain_ok, residual_ok = [], True, True, True
        for projection in projections:
            check()
            _fields(projection, ('source_frame', 'measurement_ref', 'points', 'rms_px', 'maximum_px'), 'saved projection')
            frame_id = _integer(projection['source_frame'], len(frames)-1)
            _require(frame_id in prepared and projection['measurement_ref'] == frames[frame_id]['measurement_ref'],
                     'projection must use its exact prepared construction measurement')
            points = projection['points']
            known = [p['corner_id'] for p in frames[frame_id]['points'] if p['status'] == 'observed']
            _require(type(points) is list and len(points) <= 64 and [p['corner_id'] for p in points] == known,
                     'all original known construction points must remain')
            frame_errors = []
            for point in points:
                _fields(point, ('corner_id', 'pixel', 'uncertainty_px', 'predicted_pixel', 'residual_px', 'error_px', 'depth_m', 'status'), 'projected point')
                pixel = _vector(point['pixel'], 2)
                _require(0 <= pixel[0] <= source['width'] and 0 <= pixel[1] <= source['height'], 'supplied point is outside image')
                _require(0 <= _number(point['uncertainty_px'], 64), 'declared point uncertainty differs')
                z = _number(point['depth_m'], float('inf')); valid_depth = .2 <= z <= 100
                predicted = _vector(point['predicted_pixel'], 2, nullable=True)
                valid_domain = predicted is not None
                if valid_domain:
                    residual = _vector(point['residual_px'], 2)
                    for a, b, d in zip(predicted, pixel, residual):
                        _same(d, a-b, 'pixel residual')
                    error = math.hypot(*residual); _same(point['error_px'], error, 'pixel error')
                    errors.append(point['error_px']); frame_errors.append(point['error_px'])
                else:
                    _require(point['residual_px'] is None and point['error_px'] is None, 'invalid projection must retain unknown residual')
                _require(point['status'] == ('projected' if valid_depth and valid_domain else 'invalid_depth_or_domain'),
                         'point status differs from saved depth/domain')
                depth_ok &= valid_depth; domain_ok &= valid_domain; projected += 1
            complete = len(frame_errors) == len(points)
            rms = math.sqrt(sum(e*e for e in frame_errors)/len(points)) if complete else None
            maximum = max(frame_errors) if complete else None
            _same(projection['rms_px'], rms, 'per-frame RMS'); _same(projection['maximum_px'], maximum, 'per-frame largest error')
            residual_ok &= complete and rms <= 1 and maximum <= 2
        if checks['proper_rotation']:
            _require((checks['depth'], checks['projection_domain'], checks['construction_residual']) == (depth_ok, domain_ok, residual_ok),
                     'saved depth/domain/residual checks disagree')
        _same(checks['rms_px'], math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None, 'alternative RMS')
        _same(checks['maximum_px'], max(errors) if errors else None, 'alternative largest error')
        passing = row['status'] == 'nominal_checks_passed'
        _require(not passing or not reasons and all(checks[k] for k in ('proper_rotation', 'depth', 'projection_domain', 'construction_residual')),
                 'passing alternative has a failed saved check')
        _require(passing or reasons, 'rejected alternative must retain its reason')
        if row['duplicate_of'] is not None:
            earlier = _integer(row['duplicate_of'], index-1)
            _require(passing and rows[earlier]['status'] == 'nominal_checks_passed' and rows[earlier]['duplicate_of'] is None,
                     'duplicate must point to an earlier distinct passing alternative')
            for name in ('R_ct', 't_ct'):
                values = row[name] if name == 't_ct' else sum(row[name], [])
                prior = rows[earlier][name] if name == 't_ct' else sum(rows[earlier][name], [])
                _require(max(abs(a-b) for a, b in zip(values, prior)) <= 1e-9, 'duplicate pose tolerance differs')
    _require(projected <= 1024, 'point comparison limit exceeded')
    return projected


def _artifacts(files, result, check):
    qualified = result['status'] == 'qualified_nominal'
    _require(('camera.json' in files) == ('ground.json' in files) == qualified,
             'camera and ground downloads require qualified nominal output')
    if not qualified:
        _require(result['artifact_bindings'] is None, 'unqualified result cannot advertise admitted artifacts')
        return None
    pins = result['artifact_bindings']
    _fields(pins, ('camera_raw_sha256', 'camera_canonical_sha256', 'ground_raw_sha256', 'qualities', 'generation'), 'artifact bindings')
    camera, ground = _document(files['camera.json']), _document(files['ground.json'])
    _require(pins['camera_raw_sha256'] == sha256(files['camera.json']).hexdigest()
             and pins['ground_raw_sha256'] == sha256(files['ground.json']).hexdigest()
             and pins['camera_canonical_sha256'] == canonical_hash(camera), 'exact admitted artifact identities differ')
    _fields(camera, ('schema_version', 'source_sha256', 'width', 'height', 'fps', 'frame_count', 'world_axes',
                     'units', 'provenance', 'quality', 'K', 'radial_k1', 'frames'), 'camera artifact')
    _fields(ground, ('schema_version', 'source_sha256', 'calibration_sha256', 'world_axes', 'units', 'height_m', 'quality', 'provenance'), 'ground artifact')
    source, request = result['source'], result['request']
    for document in (camera, ground):
        _require(type(document['schema_version']) is int and document['schema_version'] == 1
                 and document['source_sha256'] == source['sha256'] and document['world_axes'] == 'right_handed_z_up'
                 and document['units'] == 'm' and document['quality'] in ('measured', 'authored', 'reconstructed'),
                 'artifact source/world/quality differs')
        _text(document['provenance'], 2000)
    _require(pins['qualities'] == {'camera':camera['quality'], 'ground':ground['quality']}
             and ground['calibration_sha256'] == pins['camera_canonical_sha256'], 'ground must bind the canonical camera')
    _number(ground['height_m'], 1000)
    _integer(camera['width'], 1_000_000, 1); _integer(camera['height'], 1_000_000, 1)
    _integer(camera['frame_count'], MAX_FRAMES, 1); _number(camera['fps'], 240)
    _require(all(camera[k] == source[k] for k in ('width', 'height', 'fps'))
             and camera['frame_count'] == source['metadata_frames'], 'camera source metadata differs')
    _require(type(camera['frames']) is list and len(camera['frames']) == camera['frame_count'], 'complete camera clock required')
    _require(type(camera['K']) is list and len(camera['K']) == 3, 'camera matrix shape differs')
    for row in camera['K']:
        _vector(row, 3)
    _number(camera['radial_k1'], 1)
    for index, frame in enumerate(camera['frames']):
        check()
        _fields(frame, ('frame', 'time_s', 'R', 't'), 'camera frame')
        _require(type(frame['frame']) is int and frame['frame'] == index, 'camera frame identity differs')
        _number(frame['time_s'], 60)
        _require(abs(frame['time_s']-index/source['fps']) <= 1e-7, 'camera source clock differs')
        _require(type(frame['R']) is list and len(frame['R']) == 3, 'camera rotation shape differs')
        for row in frame['R']:
            _vector(row, 3)
        _vector(frame['t'], 3)
        if result['mode'] == 'planar_target':
            selected = result['alternatives'][result['selected_index']]
            _require(frame['R'] == selected['R_cw'] and frame['t'] == selected['t_cw'], 'nominal fixed pose differs from selected alternative')
    if result['mode'] == 'supplied':
        _require(pins['generation'] == 'supplied_exact_bytes' and pins['camera_raw_sha256'] == request['supplied_camera_sha256']
                 and pins['ground_raw_sha256'] == request['supplied_ground_sha256'], 'supplied sidecar bytes must remain unchanged')
    else:
        _require(pins['generation'] == 'planar_nominal' and set(pins['qualities'].values()) == {'reconstructed'},
                 'target-derived artifacts must remain labeled reconstructed')
    return {'camera_quality':camera['quality'], 'ground_quality':ground['quality'], 'height_m':ground['height_m']}


def _model(files, check):
    result, request = _document(files['calibration.json']), _document(files['inputs/request.json'])
    selected_frames = _request(request)
    inventory = _inventory(_document(files['input-files.json']), request, files['inputs/request.json'])
    _fields(result, ('schema_version', 'profile', 'status', 'mode', 'reason', 'input_bindings', 'source', 'request',
        'measurement_records', 'frames', 'alternatives', 'selected_index', 'artifact_bindings', 'counts', 'work', 'limits',
        'uncertainty', 'claims'), 'calibration result')
    _require(type(result['schema_version']) is int and result['schema_version'] == 1
             and result['profile'] == 'floor-target-calibration-result-v1'
             and result['status'] in ('qualified_nominal', 'unavailable', 'rejected', 'ambiguous')
             and result['mode'] == request['mode'] and result['request'] == request, 'result/request profile or status differs')
    _text(result['reason'])
    expected_bindings = {'source_sha256':request['source_sha256'], 'request_sha256':sha256(files['inputs/request.json']).hexdigest(),
        'observation_document_sha256s':sorted(request['observation_sha256s']), 'measurement_document_sha256s':sorted(request['measurement_sha256s']),
        'supplied_camera_sha256':request['supplied_camera_sha256'], 'supplied_ground_sha256':request['supplied_ground_sha256']}
    _require(result['input_bindings'] == expected_bindings, 'exact original input bindings differ')
    source = result['source']
    _fields(source, ('sha256', 'width', 'height', 'fps', 'native_frame_times_s', 'decoded_frames', 'metadata_frames', 'retained_frames', 'time'), 'source coverage')
    _require(source['sha256'] == request['source_sha256'], 'source identity differs')
    _integer(source['width'], 1_000_000, 1); _integer(source['height'], 1_000_000, 1)
    if source['fps'] is None:
        _require(result['status'] == 'unavailable' and result['reason'] == 'missing_source_nominal_clock',
                 'missing nominal clock cannot qualify calibration')
    else:
        _require(1 <= _number(source['fps'], 240), 'bounded source FPS required')
    for name in ('decoded_frames', 'metadata_frames', 'retained_frames'):
        _integer(source[name])
    count = source['retained_frames']
    _require(0 < count == max(source['decoded_frames'], source['metadata_frames']) and (source['fps'] is None or count/source['fps'] <= 60)
             and all(frame < count for frame in selected_frames), 'complete original source denominator differs')
    native_times = source['native_frame_times_s']
    if native_times is not None:
        _require(type(native_times) is list and len(native_times) == count, 'complete native source clock required')
        previous = -1
        for stamp in native_times:
            _require(0 <= _number(stamp, 60) and stamp > previous, 'native source timestamps must increase')
            previous = stamp
    _require(type(source['time']) is dict and set(source['time']) == set(request['observation_sha256s']), 'source clock inventory differs')
    _require(result['uncertainty'] == {'propagation':'not_propagated',
        'interpretation':'Nominal consistency only; declared bounds or standard deviations are not propagated or certified',
        'measurement_authenticity':'unverified'}, 'uncertainty cannot claim propagation or authentication')
    _require(result['claims'] == {'physical_validation':False, 'contact':'unknown', 'grasp':'unknown', 'load':'unknown',
        'geometry_modified':False, 'body_or_floor_installed':False, 'evaluation_used':False, 'source_measurement_authentication':False},
        'unsupported installation, contact or evaluation claim')
    records = _records(result['measurement_records'], request, check)
    prepared, known = _frames(result['frames'], source, request, records, check)
    projected = _alternatives(result['alternatives'], result['frames'], source, prepared, check)
    alternatives = result['alternatives']
    passing = [row for row in alternatives if row['status'] == 'nominal_checks_passed']
    distinct = [row for row in passing if row['duplicate_of'] is None]
    for index, row in enumerate(distinct):
        for earlier in distinct[:index]:
            same_r = max(abs(a-b) for a, b in zip(sum(row['R_ct'], []), sum(earlier['R_ct'], []))) <= 1e-9
            same_t = max(abs(a-b) for a, b in zip(row['t_ct'], earlier['t_ct'])) <= 1e-9
            _require(not (same_r and same_t), 'numerical duplicates cannot become distinct alternatives')
    counts = {'source_slots':count, 'measurement_records':len(records),
        'construction_frames_requested':(1 if request['anchor_frame'] is not None else 0)+len(request['check_frames']),
        'evaluation_frames_not_used':len(request['evaluation_frames']), 'prepared_construction_frames':len(prepared),
        'used_corner_observations':known, 'alternatives':len(alternatives), 'passing_alternatives':len(passing),
        'distinct_passing_alternatives':len(distinct),
        'frame_status_counts':dict(Counter(row['reason'] or row['status'] for row in result['frames']))}
    _require(result['counts'] == counts, 'complete result counts differ')
    for key, value in result['counts'].items():
        if key == 'frame_status_counts':
            for number in value.values():
                _integer(number)
        else:
            _integer(value, 1804)
    work = result['work']
    _fields(work, ('input_bytes', 'solver_calls', 'undistorted_points', 'point_reprojections', 'prepared_construction_frames'), 'saved work')
    for value in work.values():
        _integer(value, MAX_INPUT)
    _require(work['input_bytes'] == sum(row['bytes'] for row in inventory) and work['solver_calls'] in (0, 1)
             and work['undistorted_points'] == known and work['point_reprojections'] == projected
             and work['prepared_construction_frames'] == len(prepared), 'saved work/counts differ')
    limits = result['limits']
    _require(type(limits) is dict and all(type(value) in (int, float) and math.isfinite(value) for value in limits.values()),
             'finite numeric saved limits required')
    _require(type(limits) is dict and limits.get('maximum_source_frames') == 1800 and limits.get('maximum_corners') == 64
             and limits.get('maximum_alternatives') == 2 and limits.get('maximum_point_reprojections') == 1024
             and limits.get('maximum_rms_px') == 1 and limits.get('maximum_error_px') == 2
             and limits.get('minimum_depth_m') == .2 and limits.get('maximum_depth_m') == 100
             and limits.get('duplicate_rotation_max_abs') == 1e-9 and limits.get('duplicate_translation_max_abs_m') == 1e-9,
             'fixed saved calibration checks differ')
    if result['mode'] == 'supplied':
        _require(not alternatives and result['selected_index'] is None and not prepared and not work['solver_calls']
                 and result['status'] in ('qualified_nominal', 'unavailable'), 'supplied route cannot claim a target solve')
    elif result['status'] == 'qualified_nominal':
        _require(len(distinct) == 1 and type(result['selected_index']) is int and result['selected_index'] == distinct[0]['index']
                 and work['solver_calls'] == 1 and len(prepared) == counts['construction_frames_requested'],
                 'qualified target result needs one distinct passing alternative')
        fixed = request['fixed_interval']
        _fields(fixed, ('first_frame', 'last_frame', 'camera_static', 'target_static', 'method', 'uncertainty', 'evidence_refs'), 'fixed camera/target declaration')
        _require(type(fixed['first_frame']) is int and fixed['first_frame'] == 0
                 and type(fixed['last_frame']) is int and fixed['last_frame'] == count-1
                 and fixed['camera_static'] is True and fixed['target_static'] is True,
                 'qualified target result requires its complete fixed-camera/target declaration')
        _text(fixed['method'], 1000); _text(fixed['uncertainty'], 2000)
    else:
        _require(result['selected_index'] is None, 'unqualified alternatives cannot be selected')
        if result['status'] == 'unavailable':
            _require(not alternatives and work['solver_calls'] == 0, 'unavailable result cannot conceal a solver outcome')
        else:
            _require(work['solver_calls'] == 1 and (len(distinct) > 1 if result['status'] == 'ambiguous' else not distinct),
                     'rejected/ambiguous result differs from saved alternatives')
    artifact_summary = _artifacts(files, result, check)
    return result, inventory, artifact_summary


_STYLE = '''
body{max-width:1080px;margin:0 auto;padding:28px;font-family:system-ui,sans-serif;line-height:1.5}
h1{font-size:2rem;line-height:1.15}h2{margin-top:1.8rem}a{color:var(--weft-ink,#214F50)}
.summary{padding:18px;border:1px solid #c9bcae;border-radius:14px;background:#fff8ef}
.muted{color:#645d55}.controls{display:flex;flex-wrap:wrap;gap:18px;margin:16px 0}
label{display:grid;gap:5px}select{max-width:100%;padding:8px;font:inherit}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #d9ccbd}
.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem}details{margin:14px 0}
svg{display:block;width:100%;max-height:460px;background:#fff;border:1px solid #c9bcae;overflow:hidden}
.legend{display:flex;gap:20px;flex-wrap:wrap}.measured{color:#214F50}.projected{color:#965644}
nav{display:flex;gap:16px;flex-wrap:wrap}button,select,a:focus-visible{outline-offset:4px}
'''


def _page(title, content, data=None, script=''):
    payload = '' if data is None else _encode(data).decode().replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    raw = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'self\' \'unsafe-inline\'; script-src \'unsafe-inline\'; connect-src \'none\'; base-uri \'none\'; form-action \'none\'">'
        f'<title>{html.escape(title)} · Weft</title><link rel="stylesheet" href="brand.css"><style>{_STYLE}</style></head>'
        f'<body><p>Weft · calibration inspection</p><h1>{html.escape(title)}</h1>{content}'
        + (f'<script id="data" type="application/json">{payload}</script><script>{script}</script>' if data is not None else '')
        + '</body></html>').encode()
    _require(len(raw) <= MAX_HTML, 'HTML exceeds two MiB; no frame data was dropped')
    return raw


_FRAME_SCRIPT = r'''
'use strict';
const D=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id),NS='http://www.w3.org/2000/svg';
const text=(id,value)=>{value=String(value);if(el(id).textContent!==value)el(id).textContent=value;},show=v=>v===null||v===undefined?'Unavailable':String(v);
function option(select,value,label){const node=document.createElement('option');node.value=String(value);node.textContent=label;select.append(node);}
option(el('frame'),'','Choose a frame');D.rows.forEach((r,i)=>option(el('frame'),i,'Frame '+r.frame.source_frame+' · '+r.frame.role));
option(el('alternative'),'','Choose a pose alternative');D.alternatives.forEach((a,i)=>option(el('alternative'),i,'Alternative '+a.index+' · '+a.status));
function index(value,length){if(!/^(0|[1-9][0-9]*)$/.test(value))return null;const n=Number(value);return Number.isSafeInteger(n)&&n<length?n:null;}
function cell(row,value){const td=document.createElement('td');td.textContent=show(value);row.append(td);}
function circle(group,p,color,radius){const node=document.createElementNS(NS,'circle');node.setAttribute('cx',String(p[0]));node.setAttribute('cy',String(p[1]));node.setAttribute('r',String(radius));node.setAttribute('fill',color);group.append(node);}
let previous='';
function update(){
 const key=el('frame').value+'|'+el('alternative').value;if(key===previous)return;previous=key;
 const fi=index(el('frame').value,D.rows.length),ai=index(el('alternative').value,D.alternatives.length);
 if(fi===null){text('frame-state','Choose a source frame.');text('frame-clock','');text('frame-reason','');el('point-rows').replaceChildren();el('plot-points').replaceChildren();text('frame-record','');text('measurement-record','');text('projection-state','No frame selected.');return;}
 const entry=D.rows[fi],f=entry.frame;
 text('frame-state','Frame '+f.source_frame+' · '+f.source_status+' · '+f.role+' · '+f.status);
 text('frame-clock','Nominal source time: '+show(f.nominal_time_s)+' s · observation time: '+show(f.source_time_s)+' s.');
 text('frame-reason',f.reason===null?'No unavailable/rejection reason reported.':f.reason);
 text('frame-record',JSON.stringify(f,null,2));
 text('measurement-record',entry.measurement===null?'No measurement record for this original frame.':JSON.stringify(entry.measurement,null,2));
 const projection=ai===null?null:entry.projections[ai];el('point-rows').replaceChildren();el('plot-points').replaceChildren();
 if(projection===null){text('projection-state',ai===null?'Choose a pose alternative to inspect its saved pixel comparison.':'No projection was computed for this frame and alternative.');return;}
 text('projection-state','Saved comparison only · RMS '+show(projection.rms_px)+' px · largest error '+show(projection.maximum_px)+' px.');
 const radius=Math.max(D.width,D.height)/220;
 for(const p of projection.points){const row=document.createElement('tr');for(const value of [p.corner_id,p.status,JSON.stringify(p.pixel),JSON.stringify(p.predicted_pixel),p.uncertainty_px,p.error_px,p.depth_m])cell(row,value);el('point-rows').append(row);
  if(p.pixel!==null)circle(el('plot-points'),p.pixel,'#214F50',radius);
  if(p.predicted_pixel!==null)circle(el('plot-points'),p.predicted_pixel,'#965644',radius*.65);
 }
}
el('plot').setAttribute('viewBox','0 0 '+D.width+' '+D.height);el('frame').addEventListener('change',update);el('alternative').addEventListener('change',update);update();
'''


def _frame_page(rows, source, alternatives, index, total):
    navigation = '<nav><a href="index.html">Calibration summary</a>'
    if index:
        navigation += f'<a href="frames-{index:03d}.html">Previous frames</a>'
    if index + 1 < total:
        navigation += f'<a href="frames-{index+2:03d}.html">Next frames</a>'
    navigation += '</nav>'
    content = navigation + ('<p>Every original source slot remains available, including missing and unused frames. '
        'Only saved construction-frame comparisons are shown; these points do not establish physical contact.</p>'
        '<div class="controls"><label>Source frame<select id="frame"></select></label>'
        '<label>Pose alternative<select id="alternative"></select></label></div>'
        '<p id="frame-state" role="status"></p><p id="frame-clock"></p><p id="frame-reason"></p>'
        '<h2>Measured and projected image points</h2><p id="projection-state"></p>'
        '<p class="legend"><span class="measured">Large teal: supplied image point</span>'
        '<span class="projected">Small clay: projected point</span></p>'
        '<svg id="plot" role="img" aria-label="Saved image-point comparison, without a source image"><g id="plot-points"></g></svg>'
        '<div class="scroll"><table><thead><tr><th>Corner</th><th>Status</th><th>Supplied px</th><th>Projected px</th>'
        '<th>Declared uncertainty px</th><th>Error px</th><th>Depth m</th></tr></thead><tbody id="point-rows"></tbody></table></div>'
        '<details><summary>Exact frame and point availability</summary><pre id="frame-record"></pre></details>'
        '<details><summary>Supplied measurement and uncertainty</summary><pre id="measurement-record"></pre></details>')
    return _page('Inspect calibration frames', content,
                 {'rows':rows, 'width':source['width'], 'height':source['height'], 'alternatives':alternatives}, _FRAME_SCRIPT)


def _measurements(result):
    records = {(row['document_sha256'], row['id']):row for row in result['measurement_records']}
    content = '<h2>Supplied measurements</h2><p>“Measured” is the supplier’s description, not independent authentication. Uncertainty is reported as supplied and is not propagated into a certified camera or floor interval.</p>'
    for name, reference in result['request']['measurement_refs'].items():
        if reference is None:
            continue
        row = records.get((reference['document_sha256'], reference['record_id']))
        content += '<h3>' + html.escape({'camera':'Camera', 'target':'Target dimensions', 'target_to_world':'Target position in the world', 'floor':'Floor plane'}[name]) + '</h3>'
        if row is None:
            content += '<p>Selected measurement is unavailable. Its exact reference remains in the request.</p>'
            continue
        units = {'intrinsics_px':'px', 'radial_k1':'dimensionless', 'translation_m':'m', 'rotation_rad':'rad',
            'point_position':row['target_units'] or 'unknown target units', 'scale_relative':'relative / dimensionless',
            'height_m':'m', 'normal_rad':'rad', 'pixel_error_note':'supplied note'}
        values = ''.join('<tr><th>' + html.escape(key.replace('_', ' ')) + '</th><td>'
            + html.escape('Unknown' if value is None else str(value)) + '</td><td>' + html.escape(units[key]) + '</td></tr>'
            for key, value in row['uncertainty'].items() if key != 'interpretation')
        content += '<p>Quality: ' + html.escape(row['quality']) + ' · uncertainty interpretation: ' + html.escape(row['uncertainty']['interpretation'].replace('_', ' ')) + '.</p>'
        content += '<div class="scroll"><table><thead><tr><th>Quantity</th><th>Declared value</th><th>Units</th></tr></thead><tbody>' + values + '</tbody></table></div>'
        content += '<details><summary>Method and evidence identity</summary><pre>' + _pretty(row) + '</pre></details>'
    return content


def build_floor_target_calibration_report(files, *, max_seconds=5):
    """Return inspection/ bytes; the caller verifies originals and seals the package."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 5,
             'time bound must be within five seconds')
    deadline = time.monotonic() + max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Floor calibration report deadline exceeded')
    check()
    required = {'calibration.json', 'inputs/request.json', 'input-files.json'}
    _require(type(files) is dict and required <= set(files) <= required | {'camera.json', 'ground.json'}
             and all(type(value) is bytes for value in files.values()), 'fixed prepared input bytes required')
    _require(sum(map(len, files.values())) <= MAX_INPUT, 'prepared input exceeds four MiB')
    for name, cap in {'calibration.json':MAX_RESULT, 'inputs/request.json':64<<10, 'input-files.json':32<<10,
                      'camera.json':MAX_INPUT, 'ground.json':64<<10}.items():
        if name in files:
            _require(0 < len(files[name]) <= cap, name + ' exceeds its byte bound')
    result, inventory, artifact_summary = _model(files, check)
    source, alternatives = result['source'], result['alternatives']
    records = {(row['document_sha256'], row['id']):row for row in result['measurement_records']}
    projections = [{p['source_frame']:p for p in row['projections']} for row in alternatives]
    rows = []
    for frame in result['frames']:
        check(); reference = frame['measurement_ref']
        rows.append({'frame':frame, 'measurement':None if reference is None else records[_reference(reference)],
                     'projections':[values.get(frame['source_frame']) for values in projections]})
    pages = [rows[i:i+PAGE_FRAMES] for i in range(0, len(rows), PAGE_FRAMES)]
    css = (Path(__file__).parent/'assets/brand.css').read_bytes()
    _require(len(css) <= 32 << 10, 'stylesheet exceeds its byte bound')
    output, page_index = {'brand.css':css}, []
    brief_alternatives = [{key:row[key] for key in ('index', 'status')} for row in alternatives]
    for index, page in enumerate(pages):
        check(); name = f'frames-{index+1:03d}.html'
        output[name] = _frame_page(page, source, brief_alternatives, index, len(pages))
        page_index.append({'path':name, 'first_frame':page[0]['frame']['source_frame'], 'last_frame':page[-1]['frame']['source_frame'], 'frames':len(page)})
    title = {'qualified_nominal':'Nominal calibration available', 'unavailable':'Calibration needs more evidence',
             'rejected':'Calibration did not pass its checks', 'ambiguous':'More than one calibration fits'}[result['status']]
    route = 'Supplied camera and floor' if result['mode'] == 'supplied' else 'Target-based calibration'
    content = '<div class="summary"><p><strong>' + html.escape(route) + '</strong></p><p>' + html.escape(result['reason'].replace('_', ' ')) + '</p>'
    content += '<p>Nothing was installed or moved. This result does not establish physical contact, grasp or load.</p></div>'
    content += f'<h2>Source coverage</h2><p>{source["retained_frames"]} original frame slots · {source["decoded_frames"]} decoded · {source["retained_frames"]-source["decoded_frames"]} metadata-only slots. '
    content += f'{result["counts"]["prepared_construction_frames"]} construction frames used; {result["counts"]["evaluation_frames_not_used"]} evaluation frames retained but not used.</p><nav>'
    content += ''.join(f'<a href="{page["path"]}">Frames {page["first_frame"]}–{page["last_frame"]}</a>' for page in page_index) + '</nav>'
    content += '<h2>Downloads</h2><nav><a href="../calibration.json" download>Complete calibration result</a><a href="../inputs/request.json" download>Exact request</a></nav>'
    if artifact_summary is not None:
        content += '<p>These camera and floor files passed the core’s nominal checks; measurement uncertainty is still unpropagated. They are not automatically installed.</p>'
        content += '<nav><a href="../camera.json" download>Camera result</a><a href="../ground.json" download>Ground result</a></nav>'
        content += '<p>Camera quality: ' + html.escape(artifact_summary['camera_quality']) + ' · ground quality: ' + html.escape(artifact_summary['ground_quality']) + ' · retained ground height: ' + html.escape(str(artifact_summary['height_m'])) + ' m.</p>'
    else:
        content += '<p>No admitted camera or ground download is available for this result.</p>'
    content += _measurements(result) + '<h2>Pose alternatives</h2>'
    if not alternatives:
        content += '<p>' + ('No target pose was solved. The supplied camera and floor were checked directly.' if result['mode'] == 'supplied' else 'No pose alternative was produced; see the missing-evidence reason above.') + '</p>'
    for alternative in alternatives:
        check(); label = 'Alternative ' + str(alternative['index'])
        selected = result['selected_index'] == alternative['index']
        content += '<h3>' + label + (' · selected nominal result' if selected else '') + '</h3><p>' + html.escape(alternative['status'].replace('_', ' '))
        if alternative['duplicate_of'] is not None:
            content += ' · duplicate of alternative ' + str(alternative['duplicate_of'])
        content += '</p><p>' + html.escape('; '.join(alternative['reasons']) or 'All saved nominal checks passed.') + '</p>'
        content += '<p>Pixel error: RMS ' + ('unavailable' if alternative['checks']['rms_px'] is None else str(alternative['checks']['rms_px'])) + ' px · largest '
        content += ('unavailable' if alternative['checks']['maximum_px'] is None else str(alternative['checks']['maximum_px'])) + ' px.</p>'
        content += '<details><summary>Exact pose and check values</summary><pre>' + _pretty({key:value for key,value in alternative.items() if key != 'projections'}) + '</pre></details>'
    inputs = ''.join('<li><a href="../' + html.escape(row['path'], quote=True) + '" download>' + html.escape(row['role'] + ' · ' + row['path'])
                    + '</a> · ' + str(row['bytes']) + ' bytes · <code>' + row['sha256'] + '</code></li>' for row in inventory)
    technical = {key:result[key] for key in ('input_bindings', 'source', 'counts', 'work', 'limits', 'uncertainty', 'claims', 'artifact_bindings')}
    content += '<details><summary>Technical identities and original input files</summary><p>Original supplied camera/ground inputs are distinct from the admitted result downloads. The caller verifies these original file pins; this viewer does not read them.</p><ul>' + inputs + '</ul><pre>' + _pretty(technical) + '</pre></details>'
    output['index.html'] = _page(title, content)
    summary = {'schema_version':1, 'profile':PROFILE, 'status':result['status'], 'mode':result['mode'],
        'input_files':{name:sha256(raw).hexdigest() for name,raw in sorted(files.items())},
        'asset_files':{'brand.css':sha256(css).hexdigest()}, 'counts':result['counts'], 'pages':page_index,
        'admitted_downloads':['camera.json', 'ground.json'] if artifact_summary is not None else [],
        'original_input_verification':'caller_owned_not_performed_here', 'uncertainty_propagation':'not_propagated',
        'installed':False, 'physical_contact':'unknown'}
    output['report.json'] = _encode(summary)
    _require(len(output) <= MAX_FILES and sum(map(len, output.values())) <= MAX_OUTPUT,
             'complete offline output exceeds its bound; no slots were dropped')
    check()
    return output
