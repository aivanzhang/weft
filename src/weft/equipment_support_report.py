"""Offline inspection of saved equipment candidates and their image-only constraints."""

from collections import Counter
from fractions import Fraction
from hashlib import sha256
import html
import json
import math
from pathlib import Path
import re
import time


PROFILE = 'equipment-support-report-v1'
CONSTRAINT_PROFILE = 'equipment-silhouette-window-constraint-v1'
HAND_PROFILE = 'equipment-hand-support-report-v1'
HAND_CONSTRAINT_PROFILE = 'equipment-hand-window-constraint-v1'
MAX_INPUT = (8 << 20) + 1
MAX_HTML, MAX_OUTPUT, MAX_FILES = 2 << 20, 16 << 20, 32
MAX_FRAMES, MAX_POSES, PAGE_FRAMES, PAGE_BYTES = 1800, 900, 100, 1 << 20
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def _require(value, message):
    if not value:
        raise ValueError('Equipment support report: ' + message)


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _json(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    try:
        value = json.loads(raw, object_pairs_hook=unique)
        _encode(value)
    except (ValueError, RecursionError, UnicodeError) as error:
        raise ValueError('Equipment support report: finite strict JSON required') from error
    _require(type(value) is dict, 'JSON object required')
    return value


def _text(value, maximum=512, *, nullable=False):
    _require(nullable and value is None or type(value) is str and 0 < len(value) <= maximum
             and all(ord(c) >= 32 for c in value), 'bounded text required')
    return value


def _integer(value, maximum=MAX_FRAMES, minimum=0):
    _require(type(value) is int and minimum <= value <= maximum, 'bounded integer required')
    return value


def _number(value, maximum=86400., *, nullable=False):
    _require(nullable and value is None or type(value) in (int, float)
             and 0 <= value <= maximum and math.isfinite(value), 'finite nonnegative saved number required')
    return value


def _pin(value):
    _require(type(value) is str and _SHA.fullmatch(value), 'exact SHA-256 required')
    return value


def _canonical(value):
    # Match the existing producer's canonical JSON, including its default separators.
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _baseline(result, constraint, check, *, hand_mode=False):
    extension = 'equipment_hand_constraint' if hand_mode else 'equipment_silhouette_constraint'
    decision_name = 'hand_constraint' if hand_mode else 'silhouette_constraint'
    exclusion = 'image_hand_pointset_exclusion' if hand_mode else 'image_silhouette_exclusion'
    _require(result.get('profile') == 'exercise-support-evidence-v1'
             and type(result.get('schema_version')) is int and result['schema_version'] == 1,
             'existing equipment candidate profile required')
    _require(result.get('contact_state') == 'unknown' and result.get('ground_height') is None
             and result.get('support_force') is None and result.get('promoted') is False
             and result.get('physical_validation') == 'unverified', 'candidate evidence cannot claim physical support')
    source = _pin(result.get('source_sha256'))
    for key in ('pose_evidence_canonical_sha256', 'equipment_observations_sha256', 'equipment_source_sha256'):
        _pin(result.get(key))
    prior = result.get('camera_prior')
    _require(type(prior) is dict and set(prior) == {'mode', 'status', 'source_sha256', 'rationale'}
             and prior['mode'] == 'fixed_source_camera' and prior['status'] == 'authored_unverified'
             and prior['source_sha256'] == source, 'explicit unchanged fixed-camera assumption required')
    _text(prior['rationale'], 1000)
    rows, counts, coverage = result.get('rows'), result.get('counts'), result.get('coverage')
    _require(type(rows) is list and 1 <= len(rows) <= MAX_POSES and type(counts) is dict and type(coverage) is dict,
             'bounded complete supplied rows and coverage required')
    for name, value in counts.items():
        if name.endswith('_status_counts'):
            _require(type(value) is dict, 'saved status counts required')
            for status, number in value.items():
                _text(status, 160); _integer(number, MAX_POSES * 16)
        else:
            _integer(value, MAX_POSES * 16)
    decoded = _integer(coverage.get('decoded_frames'), MAX_FRAMES, 1)
    declared = _integer(coverage.get('declared_frame_count'), MAX_FRAMES, 1)
    construction = _integer(counts.get('construction_rows'), len(rows))
    _require(construction == math.floor(.6 * len(rows)) and counts.get('supplied_pose_rows') == len(rows)
             and counts.get('excluded_later_rows') == len(rows) - construction, 'original first-sixty-percent row boundary differs')
    equipment = result.get('equipment_object_ids')
    _require(type(equipment) is list and 1 <= len(equipment) <= 8 and all(type(v) is str for v in equipment)
             and len(equipment) == len(set(equipment)), 'explicit unique equipment IDs required')
    for identity in equipment: _text(identity, 80)
    keys = [(identity, hand) for identity in equipment for hand in ('left_wrist', 'right_wrist')]
    _require(counts.get('equipment_pair_slots') == construction * len(keys)
             and counts.get('retained_equipment_pair_slots') == len(rows) * len(keys)
             and counts.get('ground_point_slots') == construction * 4
             and counts.get('retained_ground_point_slots') == len(rows) * 4, 'complete original slot denominators differ')
    original, restored, active_counts, baseline_counts, ground_counts = {}, [], Counter(), Counter(), Counter()
    previous_frame, previous_time = -1, -1.
    allowed = {'projected_co_motion_candidate', 'image_separation', 'relative_image_motion', 'unavailable'}
    for i, row in enumerate(rows):
        check()
        _require(type(row) is dict and type(row.get('frame')) is int and previous_frame < row['frame'] < decoded
                 and row.get('id') == f"{source}:frame:{row['frame']}", 'original pose row identity/order differs')
        _number(row.get('time_s'), 60); _number(row.get('decoder_timestamp_s'), 61, nullable=True)
        _require(previous_time < row['time_s'] and row.get('partition') == ('construction' if i < construction else 'excluded_later'),
                 'original pose clock or partition boundary differs')
        previous_frame, previous_time = row['frame'], row['time_s']
        pairs = row.get('equipment_pairs')
        _require(type(pairs) is list and len(pairs) == len(keys) and all(type(p) is dict for p in pairs)
                 and [(p.get('object_id'), p.get('hand')) for p in pairs] == keys, 'complete ordered equipment pair slots required')
        originals = []
        for pair in pairs:
            _require(type(pair) is dict and pair.get('verified_contact') is False and pair.get('grasp') == 'unknown'
                     and 'co_motion_baseline_status' in pair and type(pair.get(decision_name)) is dict,
                     'saved pair decision and unknown contact scope required')
            before, after = pair['co_motion_baseline_status'], pair.get('status')
            _text(before, 160); _text(after, 160)
            if i < construction:
                _require(before in allowed and (after == before or before == 'projected_co_motion_candidate'
                         and after in {'unavailable', exclusion}), 'constraint cannot create or alter a noncandidate')
                baseline_counts[before] += 1; active_counts[after] += 1
            else:
                _require(before == after == 'excluded_later_partition'
                         and set(pair) == {'object_id', 'hand', 'status', 'verified_contact', 'grasp',
                                           'co_motion_baseline_status', decision_name}, 'later pair cannot expose construction payload')
            original_pair = {key: value for key, value in pair.items() if key not in {'co_motion_baseline_status', decision_name}}
            original_pair['status'] = before
            originals.append(original_pair)
        ground = row.get('ground_points')
        _require(type(ground) is list and len(ground) == 4
                 and all(type(point) is dict and point.get('verified_contact') is False for point in ground), 'original ground candidates remain unverified')
        if i < construction:
            for point in ground:
                ground_counts[_text(point.get('status'), 160)] += 1
        if i >= construction:
            _require('window' not in row and all(point.get('pixel') is None and point.get('confidence') is None
                     and point.get('status') == 'excluded_later_partition' for point in ground), 'later window/ground payload must remain excluded')
        restored.append({**row, 'equipment_pairs': originals})
        original[row['frame']] = i
    for key, expected in (
        ('unsampled_decoded_frame_indices', [i for i in range(decoded) if i not in original]),
        ('container_only_frame_indices', list(range(decoded, declared))),
        ('decoded_beyond_container_metadata_indices', list(range(declared, decoded)))):
        _require(coverage.get(key) == expected and all(type(i) is int for i in coverage[key]), 'complete original coverage list differs')
    _require(counts.get('equipment_status_counts') == dict(active_counts)
             and counts.get('ground_status_counts') == dict(ground_counts)
             and constraint.get('baseline_equipment_status_counts') == dict(baseline_counts), 'baseline/active equipment status denominators differ')
    baseline = {key: value for key, value in result.items() if key != extension}
    baseline['rows'] = restored
    baseline['counts'] = {**counts, 'equipment_status_counts': dict(baseline_counts)}
    _require(_canonical(baseline) == _pin(constraint.get('baseline_canonical_sha256')), 'saved original baseline cannot be reconstructed exactly')
    return original, keys, construction, max(decoded, declared)


_PARTS = ('wrist', 'pinky', 'index', 'thumb')
_NULL_HAND_FIELDS = tuple('point_channel_id human_entity_ref human_observation_id human_partition equipment_observation_id equipment_time_s equipment_partition mask_evidence_id mask_phase mask_status mask_time_s model_identity pixel source_confidence source_visibility_score source_presence_score representation authored_uncertainty_px intersects distance_infimum_squared_px2 raster_sha256 raster_dimensions image_box_px'.split())
_POINT_FIELDS = set(_NULL_HAND_FIELDS) | {'equipment_id', 'point_name', 'landmark_index', 'status', 'reason',
    'contact', 'touches_excluded_edge', 'producer_record_present', 'consumer_status', 'consumer_reason'}


def _hand_groups(groups):
    _require(type(groups) is list and 1 <= len(groups) <= 2, 'explicit selected hands required')
    result, channels = {}, set()
    for group in groups:
        _require(type(group) is dict and set(group) == {'hand_id', 'wrist_name', 'points'}
                 and group['hand_id'] in ('left_hand', 'right_hand') and group['hand_id'] not in result,
                 'exact unique hand descriptors required')
        side = group['hand_id'].removesuffix('_hand')
        _require(group['wrist_name'] == side + '_wrist' and type(group['points']) is list
                 and len(group['points']) == 4, 'all four ordered hand descriptors required')
        for part, index, point in zip(_PARTS, range(15 if side == 'left' else 16, 23, 2), group['points'], strict=True):
            _require(type(point) is dict and set(point) == {'part', 'point_name', 'landmark_index', 'channel_id'}
                     and point['part'] == part and point['point_name'] == side + '_' + part
                     and type(point['landmark_index']) is int and point['landmark_index'] == index,
                     'original hand landmark descriptor differs')
            _text(point['channel_id'], 80)
            _require(point['channel_id'] not in channels, 'hand channels must be unique')
            channels.add(point['channel_id'])
        result[group['hand_id']] = group
    _require(list(result) == [name for name in ('left_hand', 'right_hand') if name in result], 'hand descriptor order differs')
    return result


def _hand_point(point, descriptor, original, image, row, bindings, header_matches):
    _require(type(point) is dict and set(point) <= _POINT_FIELDS
             and point.get('producer_record_present') is True and point.get('contact') == 'unknown',
             'saved producer point and unknown contact required')
    status, reason = point.get('consumer_status'), point.get('consumer_reason')
    _require(status in ('intersection', 'exclusion', 'unavailable'), 'named point consumer status differs')
    _text(reason, 160, nullable=True)
    _require((reason is not None) == (status == 'unavailable'), 'point qualification reason differs')
    for key in ('point_channel_id', 'point_name', 'equipment_id', 'human_observation_id', 'human_partition',
                'equipment_observation_id', 'equipment_partition', 'mask_evidence_id', 'mask_phase', 'mask_status',
                'representation', 'status', 'reason'):
        _text(point.get(key), 256, nullable=True)
    for key in ('source_confidence', 'source_visibility_score', 'source_presence_score'):
        _number(point.get(key), 1., nullable=True)
    _number(point.get('authored_uncertainty_px'), 64., nullable=True)
    for key in ('equipment_time_s', 'mask_time_s'): _number(point.get(key), 61., nullable=True)
    pixel = point.get('pixel')
    _require(pixel is None or type(pixel) is list and len(pixel) == 2
             and all(type(v) in (int, float) and math.isfinite(v) for v in pixel), 'finite saved source pixel required')
    if status == 'unavailable':
        return status
    _require(header_matches and image['equipment_source_frame'] == original.get('object_source_frame'),
             'qualified hand point source frame or clock differs')
    ref = point.get('human_entity_ref')
    _require(type(ref) is dict and set(ref) == {'document_sha256', 'object_id'}
             and ref['document_sha256'] in bindings['observation_document_sha256s'], 'qualified human document differs')
    _text(ref['object_id'], 80)
    _require(point.get('point_channel_id') == descriptor['channel_id']
             and point.get('point_name') == descriptor['point_name']
             and type(point.get('landmark_index')) is int and point['landmark_index'] == descriptor['landmark_index']
             and point.get('equipment_id') == original['object_id']
             and point.get('human_observation_id') == f"{bindings['source_sha256']}:observation:{row['frame']}:{ref['object_id']}"
             and point.get('equipment_observation_id') == original.get('object_observation_id')
             and point.get('equipment_time_s') == original.get('object_time_s')
             and point.get('mask_time_s') == original.get('object_time_s')
             and point.get('human_partition') == point.get('equipment_partition') == 'construction',
             'qualified hand point identity or original clock differs')
    _require(point.get('representation') in {'supplied_unmarked_image_estimate_not_authenticated_raw',
             'declared_raw_model_estimate_not_authenticated_identity', 'matched_saved_precursor_model_estimate_identity_unverified'}
             and row.get('landmark_origin') == 'original' and pixel is not None
             and point.get('source_confidence') is not None and point.get('authored_uncertainty_px') is not None,
             'qualified original image point required')
    _require(point.get('image_box_px') == original.get('image_box_px')
             and point.get('mask_evidence_id') in original.get('object_evidence_ids', [])
             and point.get('status') == 'conditional_image_' + status and point.get('reason') is None
             and point.get('intersects') is (status == 'intersection'), 'qualified point relation or mask differs')
    if descriptor['part'] == 'wrist':
        _require(pixel == original.get('hand_pixel') and point['source_confidence'] == original.get('hand_image_confidence'),
                 'original wrist pixel/confidence cannot be replaced by hand evidence')
    distance = point.get('distance_infimum_squared_px2')
    _require(type(distance) is dict and set(distance) == {'numerator', 'denominator', 'outward_interval'}
             and type(distance['numerator']) is int and distance['numerator'] >= 0
             and type(distance['denominator']) is int and distance['denominator'] > 0,
             'saved exact image-distance interval required')
    interval = distance['outward_interval']
    _require(type(interval) is list and len(interval) == 2, 'saved distance interval shape differs')
    for value in interval: _number(value, 1e18)
    exact = Fraction(distance['numerator'], distance['denominator'])
    _require(Fraction(interval[0]) <= exact <= Fraction(interval[1])
             and (status != 'intersection' or exact == 0)
             and point.get('touches_excluded_edge') is (status == 'exclusion' and exact == 0),
             'zero image distance is not automatically intersection')
    _pin(point.get('raster_sha256'))
    dimensions = point.get('raster_dimensions')
    _require(type(dimensions) is list and len(dimensions) == 2, 'saved raster dimensions required')
    for value in dimensions: _integer(value, 4096, 1)
    identity = point.get('model_identity')
    _require(type(identity) is dict and set(identity) == {'status', 'gap_seen', 'unresolved_group_id', 'physical_identity_verified'}
             and identity['status'] in ('assumed_model_identity', 'unresolved_association')
             and type(identity['gap_seen']) is bool and identity['physical_identity_verified'] is False,
             'model association cannot claim physical identity')
    _text(identity['unresolved_group_id'], 160, nullable=True)
    return status


def _hand_row(image, row, keys, groups, bindings):
    fields = {'source_frame', 'source_status', 'pose_evidence_id', 'pose_time_s', 'pose_decoder_timestamp_s',
              'equipment_source_frame', 'source_mapping', 'hands'}
    _require(type(image) is dict and set(image) == fields and type(image['hands']) is list
             and len(image['hands']) == len(keys), 'complete saved hand row required')
    _integer(image['source_frame']); _text(image['source_status'], 80)
    _text(image['pose_evidence_id'], 256, nullable=True)
    for name in ('pose_time_s', 'pose_decoder_timestamp_s'): _number(image[name], 61., nullable=True)
    if image['equipment_source_frame'] is not None: _integer(image['equipment_source_frame'])
    _require(image['source_mapping'] is None or type(image['source_mapping']) is dict,
             'saved derivative mapping must remain explicit')
    header_matches = (image['source_frame'] == row['frame'] and image['source_status'] == 'decoded'
        and image['pose_evidence_id'] == row['id'] and image['pose_time_s'] == row['time_s']
        and image['pose_decoder_timestamp_s'] == row.get('decoder_timestamp_s')
        and image['source_mapping'] == row.get('equipment_source_mapping'))
    total = Counter()
    for original, key, hand in zip(row['equipment_pairs'], keys, image['hands'], strict=True):
        object_id, wrist = key
        hand_id = wrist.removesuffix('_wrist') + '_hand'
        _require(type(hand) is dict and set(hand) == {'object_id', 'hand_id', 'wrist_name', 'status', 'reason',
                 'point_counts', 'unavailable_points', 'points'} and hand['object_id'] == object_id
                 and hand['hand_id'] == hand_id and hand['wrist_name'] == wrist
                 and type(hand['points']) is list and len(hand['points']) == 4,
                 'complete ordered object/hand and four point slots required')
        group, counts = groups.get(hand_id), Counter()
        for part, index, point in zip(_PARTS, range(15 if hand_id == 'left_hand' else 16, 23, 2), hand['points'], strict=True):
            if group is None:
                expected = {**dict.fromkeys(_NULL_HAND_FIELDS), 'equipment_id': object_id,
                    'point_name': hand_id.removesuffix('_hand') + '_' + part, 'landmark_index': index,
                    'status': 'unavailable', 'reason': 'hand_group_not_supplied', 'contact': 'unknown',
                    'producer_record_present': False, 'consumer_status': 'unavailable', 'consumer_reason': 'hand_group_not_supplied'}
                _require(point == expected, 'unselected hand must keep exact empty named-point slots')
                status = 'unavailable'
            else:
                descriptor = group['points'][_PARTS.index(part)]
                status = _hand_point(point, descriptor, original, image, row, bindings, header_matches)
            counts[status] += 1
        expected_counts = {name: counts[name] for name in ('intersection', 'exclusion', 'unavailable')}
        expected_status = 'unavailable' if counts['unavailable'] else 'intersection' if counts['intersection'] else 'exclusion'
        reason = ('hand_group_not_supplied' if group is None else 'required_hand_point_unavailable') if counts['unavailable'] else None
        _require(hand['point_counts'] == expected_counts and all(type(v) is int for v in hand['point_counts'].values())
                 and hand['status'] == expected_status and hand['reason'] == reason
                 and hand['unavailable_points'] == [{'point_name': p.get('point_name'), 'reason': p['consumer_reason']}
                     for p in hand['points'] if p['consumer_status'] == 'unavailable'], 'saved four-point group decision differs')
        total.update(counts)
    return image['hands'], total


def _constraints(result, constraint, keys, construction, frames, check, *, hand_mode=False):
    row_name = 'hand_rows' if hand_mode else 'point_rows'
    prefix = 'hand' if hand_mode else 'silhouette'
    decision_name = 'hand_constraint' if hand_mode else 'silhouette_constraint'
    window_counts = 'window_hand_counts' if hand_mode else 'window_image_counts'
    fields = set('schema_version profile status baseline_profile baseline_canonical_sha256 baseline_equipment_status_counts input_bindings construction_frames counts work bounds contact physical_noncontact grasp load physical_validation body_or_floor_installed scope baseline_recovery'.split()) | {row_name}
    _require(set(constraint) == fields and type(constraint['schema_version']) is int and constraint['schema_version'] == 1
             and constraint['profile'] == (HAND_CONSTRAINT_PROFILE if hand_mode else CONSTRAINT_PROFILE) and constraint['status'] == 'applied_restrict_only'
             and constraint['baseline_profile'] == result['profile'], 'distinct restrict-only constraint contract required')
    _require(all(constraint[k] == 'unknown' for k in ('contact', 'physical_noncontact', 'grasp', 'load'))
             and constraint['physical_validation'] == 'unverified' and constraint['body_or_floor_installed'] is False,
             'silhouette constraints cannot claim physical admission')
    _text(constraint['scope'], 2048); _text(constraint['baseline_recovery'], 2048)
    _require(type(constraint['baseline_equipment_status_counts']) is dict, 'saved baseline counts required')
    for value in constraint['baseline_equipment_status_counts'].values(): _integer(value, MAX_POSES * 16)
    bindings = constraint['input_bindings']
    hashes = {'source_sha256', 'equipment_source_sha256', 'pose_evidence_canonical_sha256', 'pose_raw_sha256',
              'equipment_document_sha256', 'mask_document_sha256', 'annotation_document_sha256',
              prefix + '_metadata_sha256', prefix + '_rows_length_prefixed_sha256'}
    _require(type(bindings) is dict and set(bindings) == hashes | {'observation_document_sha256s',
             'rows_digest_encoding', 'source_equivalence_manifest_sha256'} | ({'hand_groups'} if hand_mode else set()), 'complete saved input bindings required')
    for name in hashes: _pin(bindings[name])
    for name in ('source_sha256', 'equipment_source_sha256', 'pose_evidence_canonical_sha256'):
        _require(bindings[name] == result[name], 'constraint source/pose identity differs')
    _require(bindings['equipment_document_sha256'] == result['equipment_observations_sha256'], 'constraint equipment bytes differ')
    documents = bindings['observation_document_sha256s']
    _require(type(documents) is list and 1 <= len(documents) <= 8 and all(type(v) is str for v in documents)
             and documents == sorted(set(documents)), 'bounded sorted observation identities required')
    for pin in documents: _pin(pin)
    _require(bindings['rows_digest_encoding'] == 'Each original cached row prefixed by its unsigned 8-byte big-endian byte length; not a source-file hash', 'cached-row digest encoding differs')
    proof = result['equipment_source_equivalence']
    if result['equipment_source_sha256'] == result['source_sha256']:
        _require(proof is None and bindings['source_equivalence_manifest_sha256'] is None, 'same-source input cannot attach derivative equivalence')
    else:
        _require(type(proof) is dict and proof.get('original_sha256') == result['source_sha256']
                 and proof.get('derivative_sha256') == result['equipment_source_sha256']
                 and proof.get('manifest_sha256') == _pin(bindings['source_equivalence_manifest_sha256']), 'original/derivative source identity differs')
    rows, points = result['rows'], constraint[row_name]
    selected = constraint['construction_frames']
    _require(type(selected) is list and all(type(v) is int for v in selected)
             and selected == [row['frame'] for row in rows[:construction]]
             and type(points) is list and len(points) == construction, 'every original construction point row is required')
    names = ('intersection', 'exclusion', 'unavailable')
    prefixes = [{name: [0] for name in names} for _ in keys]
    point_counts, named_point_counts, point_bytes = Counter(), Counter(), 0
    groups = _hand_groups(bindings['hand_groups']) if hand_mode else None
    for i, point in enumerate(points):
        check()
        if hand_mode:
            values, counts = _hand_row(point, rows[i], keys, groups, bindings)
            named_point_counts.update(counts)
        else:
            _require(type(point) is dict and set(point) == {'source_frame', 'pose_evidence_id', 'pairs'}
                     and type(point['source_frame']) is int and point['source_frame'] == rows[i]['frame']
                     and point['pose_evidence_id'] == rows[i]['id'] and type(point['pairs']) is list
                     and len(point['pairs']) == len(keys), 'original construction point identity/order differs')
            values = point['pairs']
        for j, (key, value) in enumerate(zip(keys, values, strict=True)):
            if not hand_mode:
                _require(type(value) is dict and set(value) == {'object_id', 'hand', 'status', 'reason', 'mask_evidence_id'}
                         and (value['object_id'], value['hand']) == key and value['status'] in names, 'complete named image-point qualifications required')
                _text(value['reason'], 160, nullable=True); _text(value['mask_evidence_id'], 256, nullable=True)
                _require((value['reason'] is not None) == (value['status'] == 'unavailable'), 'image-point availability reason differs')
            for name in names: prefixes[j][name].append(prefixes[j][name][-1] + int(value['status'] == name))
            point_counts[value['status']] += 1
        point_bytes += len(json.dumps(point, sort_keys=True, allow_nan=False).encode())
    index = {row['id']: i for i, row in enumerate(rows[:construction])}
    ranges, decisions = [], Counter()
    for i, row in enumerate(rows):
        check(); interval = None
        if i < construction:
            window = row.get('window')
            _require(type(window) is dict and set(window) == {'first_evidence_id', 'last_evidence_id', 'samples', 'span_seconds', 'reasons'}, 'original window metadata required')
            _text(window['first_evidence_id'], 96); _text(window['last_evidence_id'], 96)
            left, last = index.get(window['first_evidence_id']), index.get(window['last_evidence_id'])
            _require(type(left) is int and type(last) is int and 0 <= left <= i <= last < construction
                     and type(window['samples']) is int and window['samples'] == last - left + 1
                     and window['span_seconds'] == rows[last]['time_s'] - rows[left]['time_s'], 'original window identities/counts/clock differ')
            _require(type(window['reasons']) is list and len(window['reasons']) <= 16, 'bounded original window reasons required')
            for reason in window['reasons']: _text(reason, 160)
            interval = [left, last + 1]
        ranges.append(interval)
        for j, pair in enumerate(row['equipment_pairs']):
            decision = pair[decision_name]; before = pair['co_motion_baseline_status']
            _require(set(decision) == {'status', 'reason', 'window_point_range', window_counts}, 'exact saved selector decision required')
            if i >= construction or before != 'projected_co_motion_candidate':
                _require(decision == {'status': 'excluded_later' if i >= construction else 'not_candidate',
                         'reason': before, 'window_point_range': None, window_counts: None}, 'unselected pair cannot invent a constraint window')
            else:
                left, right = interval
                counts = decision[window_counts]
                _require(type(decision['window_point_range']) is list and all(type(v) is int for v in decision['window_point_range'])
                         and decision['window_point_range'] == interval and type(counts) is dict and set(counts) == set(names), 'constraint window differs from original window')
                for name in names:
                    _integer(counts[name], right - left)
                    _require(counts[name] == prefixes[j][name][right] - prefixes[j][name][left], 'saved window counts differ from every original image point')
                expected = 'unavailable' if counts['unavailable'] else 'excluded' if counts['exclusion'] else 'kept'
                reason = {'unavailable': 'required_window_hand_point_unavailable' if hand_mode else 'required_window_image_point_unavailable',
                          'excluded': 'conditional_hand_pointset_exclusion_in_original_window' if hand_mode else 'conditional_image_exclusion_in_original_window', 'kept': None}[expected]
                after = {'kept': before, 'excluded': 'image_hand_pointset_exclusion' if hand_mode else 'image_silhouette_exclusion', 'unavailable': 'unavailable'}[expected]
                _require(decision['status'] == expected and decision['reason'] == reason and pair['status'] == after,
                         'saved kept/excluded/unavailable decision contradicts its window')
            decisions[decision['status']] += 1
    candidates = sum(decisions[k] for k in ('kept', 'excluded', 'unavailable'))
    expected_counts = {'source_frames': frames, 'construction_pair_slots': construction * len(keys),
        'retained_pose_pair_slots': len(rows) * len(keys), 'baseline_candidates': candidates,
        'kept_candidates': decisions['kept'], 'excluded_candidates': decisions['excluded'],
        'unavailable_candidates': decisions['unavailable'], 'unchanged_construction_noncandidates': decisions['not_candidate'],
        'excluded_later_pose_pairs': decisions['excluded_later'], 'point_status_counts': dict(point_counts)}
    if hand_mode:
        expected_counts.pop('construction_pair_slots')
        expected_counts.update(construction_hand_slots=construction * len(keys),
            construction_point_slots=construction * len(keys) * 4,
            hand_status_counts=dict(point_counts), point_status_counts=dict(named_point_counts))
    counts = constraint['counts']
    _require(type(counts) is dict and counts == expected_counts
             and all(type(value) is int for key, value in counts.items() if key not in {'point_status_counts', 'hand_status_counts'}), 'complete selector decision denominators differ')
    for name in ('point_status_counts', 'hand_status_counts') if hand_mode else ('point_status_counts',):
        for value in counts[name].values(): _integer(value, 540 * 16 * (4 if hand_mode else 1))
    work = constraint['work']
    work_fields = {'selected_frame_queries', 'point_slots', 'window_queries', 'serialized_point_cache_bytes',
                   'prefix_counter_entries', 'prefix_counter_buffer_bytes', 'prepared_row_bytes', 'raster_calls', 'pose_annotation_queries'}
    if hand_mode:
        work_fields = work_fields - {'serialized_point_cache_bytes'} | {'serialized_hand_cache_bytes', 'hand_slots'}
    _require(type(work) is dict and set(work) == work_fields, 'bounded saved work counters required')
    for value in work.values(): _integer(value, 8 << 20)
    entries = (construction + 1) * len(keys) * 3
    _require(type(constraint.get('bounds')) is dict, 'saved selector bounds required')
    cache_limit = constraint['bounds'].get('additional_cache_bytes') if hand_mode else 2 << 20
    _require(type(cache_limit) is int and cache_limit in (2 << 20, 8 << 20), 'explicit supported hand cache bound required')
    _require(work['selected_frame_queries'] == construction and work['point_slots'] == construction * len(keys) * (4 if hand_mode else 1)
             and (not hand_mode or work['hand_slots'] == construction * len(keys))
             and work['window_queries'] == candidates and work['serialized_hand_cache_bytes' if hand_mode else 'serialized_point_cache_bytes'] == point_bytes
             and work['prefix_counter_entries'] == entries and work['prefix_counter_buffer_bytes'] in (entries * 2, entries * 4, entries * 8)
             and point_bytes + work['prefix_counter_buffer_bytes'] <= cache_limit
             and work['raster_calls'] == work['pose_annotation_queries'] == 0, 'saved point/prefix work arithmetic differs')
    _require(constraint['bounds'] == {'cooperative_seconds': 5., 'within_original_deadline': True,
             'additional_cache_bytes': cache_limit, 'complete_output_bytes': 8 << 20, 'construction_poses': 540,
             'equipment_ids': 8, **({'hands': 2, 'points_per_hand': 4} if hand_mode else {'wrists': 2})}, 'existing selector limits cannot be expanded')
    _require(constraint['bounds']['within_original_deadline'] is True, 'selector deadline claim differs')
    return ranges


_STYLE = '''body{margin:0;background:var(--weft-canvas);color:var(--weft-ink);font:16px/1.5 system-ui,sans-serif}main{max-width:1060px;margin:auto;padding:28px 20px}section{background:var(--weft-surface);border:1px solid var(--weft-border);border-radius:12px;padding:20px;margin:18px 0}a{color:var(--weft-primary)}label{display:block;margin:12px 0}select{font:inherit;padding:8px;max-width:100%;width:100%}pre,code{white-space:pre-wrap;overflow-wrap:anywhere}pre{max-height:24em;overflow:auto;background:var(--weft-canvas);padding:12px;font-size:.85em}nav{display:flex;flex-wrap:wrap;gap:16px}.notice{border-left:4px solid var(--weft-accent);padding-left:12px}.muted{color:var(--weft-muted)}:focus-visible{outline:3px solid var(--weft-ink)}li{margin:6px 0}'''


def _page(title, content, data=None, script=''):
    payload = '' if data is None else '<script id="data" type="application/json">' + _encode(data).decode().replace('<', '\\u003c') + '</script><script src="window-index.js"></script><script>' + script + '</script>'
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; style-src \'self\' \'unsafe-inline\'; script-src \'self\' \'unsafe-inline\'; connect-src \'none\'; base-uri \'none\'; form-action \'none\'">'
        f'<title>{html.escape(title)} · Weft</title><link rel="stylesheet" href="brand.css"><style>{_STYLE}</style></head>'
        f'<body><main><p class="muted">Weft · saved equipment candidate comparison</p><h1>{html.escape(title)}</h1>{content}'
        '<p class="muted">Offline saved-data inspection only. No source acquisition, image decode, fitting, simulation or network request.</p></main>' + payload + '</body></html>').encode()


_FRAME_SCRIPT = r'''(()=>{
'use strict';
const data=JSON.parse(document.getElementById('data').textContent), index=window.WeftWindowIndex;
const get=id=>document.getElementById(id), frame=get('frame'), pair=get('pair'), group=get('window-page');
const write=(id,value)=>{const node=get(id),text=typeof value==='string'?value:JSON.stringify(value,null,2);if(node.textContent!==text)node.textContent=text;};
const option=(select,value,text)=>{const node=document.createElement('option');node.value=String(value);node.textContent=text;select.appendChild(node);return node;};
data.rows.forEach((row,i)=>option(frame,i,'Frame '+row.source_frame+' · '+row.source_status));
data.pairs.forEach((item,i)=>option(pair,i,item.object_id+' · '+item.hand.replace('_',' ')));
const groups=[];for(let i=0;i<Math.ceil(index.length/20);i++)groups.push(option(group,i,'Window entries '+(i*20+1)+'–'+Math.min(index.length,(i+1)*20)));
const links=[];for(let i=0;i<20;i++){const item=document.createElement('li'),link=document.createElement('a');item.appendChild(link);get('window-links').appendChild(item);links.push({item,link});}
frame.value='0';pair.value='0';group.value='0';
function applyHash(){const hash=location.hash.match(/^#frame=(\d+)&pair=(\d+)$/);if(hash){const selected=data.rows.findIndex(row=>row.source_frame===Number(hash[1]));if(selected>=0)frame.value=String(selected);if(Number(hash[2])<data.pairs.length)pair.value=hash[2];group.value='0';}}
applyHash();
let previous='';
function render(){
const fi=Number(frame.value),pi=Number(pair.value),wi=Number(group.value)||0,key=fi+':'+pi+':'+wi;
if(key===previous)return;previous=key;
const row=data.rows[fi],pose=row.pose,range=row.window_range,active=pose?pose.equipment_pairs[pi]:null;
write('clock',{source_frame:row.source_frame,source_status:row.source_status,pose_evidence_id:pose?pose.id:null,
pose_time_s:pose?pose.time_s:null,pose_decoder_timestamp_s:pose?pose.decoder_timestamp_s:null,
equipment_source_mapping:pose?pose.equipment_source_mapping||null:null});
if(active){
const baseline={...active,status:active.co_motion_baseline_status};delete baseline.co_motion_baseline_status;delete baseline.silhouette_constraint;
write('decision',{baseline_status:active.co_motion_baseline_status,active_status:active.status,...active.silhouette_constraint});
write('baseline',baseline);write('point',row.image_points?row.image_points[pi]:'No image qualification: later pose partition.');
write('window',pose.window||'No construction window: later pose partition.');write('ground',pose.ground_points);
}else{
write('decision','Unavailable: no retained pose row for this original frame. No window or pair value was invented.');
write('baseline','Unavailable');write('point','Unavailable');write('window','Unavailable');write('ground','Unavailable');
}
const size=range?range[1]-range[0]:0,start=wi*20;
groups.forEach((node,i)=>{node.hidden=node.disabled=i*20>=size;});
group.disabled=!size;write('window-count',size?size+' original construction points; showing '+Math.min(start+1,size)+'–'+Math.min(start+20,size)+'. Follow a frame link to inspect this exact wrist/object point.':'No original construction window.');
links.forEach(({item,link},i)=>{const at=start+i,visible=Boolean(range&&at<size);item.hidden=!visible;if(visible){const target=index[range[0]+at];link.textContent='Frame '+target.source_frame+' · '+target.pose_evidence_id;link.href=target.page+'#frame='+target.source_frame+'&pair='+pi;}});
}
frame.addEventListener('change',()=>{group.value='0';render();});pair.addEventListener('change',render);group.addEventListener('change',render);window.addEventListener('hashchange',()=>{applyHash();render();});render();
})();'''


_HAND_FRAME_SCRIPT = _FRAME_SCRIPT.replace(
    "const groups=[];",
    """const handPoint=get('hand-point');['wrist','pinky','index','thumb'].forEach((name,i)=>option(handPoint,i,name));handPoint.value='0';
const handCells=[];for(let i=0;i<4;i++){const row=document.createElement('tr'),cells=[];for(let j=0;j<6;j++){const cell=document.createElement('td');row.appendChild(cell);cells.push(cell);}get('hand-points').appendChild(row);handCells.push(cells);}
const cellWrite=(node,value)=>{const text=value===null||value===undefined?'Unknown':typeof value==='string'?value:JSON.stringify(value);if(node.textContent!==text)node.textContent=text;};
function showHand(hand,reason){
write('hand-summary',hand?{status:hand.status,reason:hand.reason,point_counts:hand.point_counts,unavailable_points:hand.unavailable_points}:reason);
for(let i=0;i<4;i++){const p=hand?hand.points[i]:null,values=p?[p.point_name,p.consumer_status,p.source_confidence,p.pixel,p.authored_uncertainty_px,p.consumer_reason||p.reason||'None']: [['wrist','pinky','index','thumb'][i],'unavailable',null,null,null,reason];values.forEach((value,j)=>cellWrite(handCells[i][j],value));}
const at=Number(handPoint.value);write('point',hand&&Number.isInteger(at)&&at>=0&&at<4?hand.points[at]:reason);
}
const groups=[];""").replace(
    "key=fi+':'+pi+':'+wi;", "key=fi+':'+pi+':'+wi+':'+handPoint.value;").replace(
    'delete baseline.silhouette_constraint;', 'delete baseline.hand_constraint;').replace(
    '...active.silhouette_constraint', '...active.hand_constraint').replace(
    "write('point',row.image_points?row.image_points[pi]:'No image qualification: later pose partition.');",
    "showHand(row.image_points?row.image_points[pi]:null,'Unavailable: later pose partition; no construction hand points queried.');").replace(
    "write('point','Unavailable');", "showHand(null,'Unavailable: no retained pose row for this original frame.');").replace(
    "group.addEventListener('change',render);", "group.addEventListener('change',render);handPoint.addEventListener('change',render);").replace(
    'this exact wrist/object point.', 'this exact hand/equipment point set.').replace(
    'equipment_source_mapping:pose?pose.equipment_source_mapping||null:null});',
    'equipment_source_mapping:pose?pose.equipment_source_mapping||null:null,raw_hand_header:row.saved_hand_header||null});')


def _frame_page(rows, keys, number, total, *, hand_mode=False):
    nav = '<a href="index.html">Overview</a><a href="support-evidence.json">Complete saved result</a>'
    if number: nav += f'<a href="frames-{number:03d}.html">Previous frames</a>'
    if number + 1 < total: nav += f'<a href="frames-{number+2:03d}.html">Next frames</a>'
    content = f'<nav>{nav}</nav><p class="notice">These are image-only candidates. Kept does not mean contact; excluded does not mean verified physical separation. Support, load and grasp remain unknown.</p>'
    content += '<section><h2>Choose an original frame and pair</h2><label>Source frame<select id="frame"></select></label><label>Equipment and wrist<select id="pair"></select></label><p>Only this selected pair is rendered. Original source time, decoder time and derivative equipment mapping are separate saved values.</p><pre id="clock"></pre></section>'
    content += '<section><h2>Original candidate → image constraint</h2><pre id="decision"></pre><p>Kept: every point in the original window has conditional image intersection. Excluded: at least one point has image exclusion and none is unavailable. Unavailable: a required point is missing or unresolved. Noncandidates and later partitions remain unchanged.</p><details><summary>Original motion-based evidence</summary><pre id="baseline"></pre></details><h3>This frame’s image qualification</h3><pre id="point"></pre></section>'
    content += '<section><h2>The complete original window</h2><pre id="window"></pre><p id="window-count"></p><label>Window detail page<select id="window-page"></select></label><ol id="window-links"></ol><p>Window boundaries, sample counts and reasons come from the original motion selector. No later frame is used to repair an unavailable construction point.</p></section>'
    content += '<section><details><summary>Unchanged ground-image candidates</summary><pre id="ground"></pre></details><p>No ground height, physical surface, support force or installed body is established here.</p></section>'
    if hand_mode:
        content = content.replace('Equipment and wrist', 'Equipment and hand (original wrist label)')
        content = content.replace('Original candidate → image constraint', 'Original wrist candidate → hand-point constraint')
        content = content.replace('Kept: every point in the original window has conditional image intersection. Excluded: at least one point has image exclusion and none is unavailable. Unavailable: a required point is missing or unresolved. Noncandidates and later partitions remain unchanged.',
            'Every hand group needs all four named points. Any missing point keeps the group unknown. Otherwise any rectangle overlap gives point-set image consistency; four disjoint rectangles give point-set image exclusion. The original candidate is kept only when every group in its window is consistent. This is not a filled hand shape or verified contact.')
        content = content.replace('Original motion-based evidence', 'Original motion-based wrist evidence (unchanged)')
        content = content.replace('<h3>This frame’s image qualification</h3><pre id="point"></pre>',
            '<h3>This frame’s four named points</h3><pre id="hand-summary"></pre><div style="overflow:auto"><table><thead><tr><th>Point</th><th>Image check</th><th>Source confidence</th><th>Source pixel</th><th>Authored range (px)</th><th>Unavailable reason</th></tr></thead><tbody id="hand-points"></tbody></table></div><label>Exact saved point details<select id="hand-point"></select></label><pre id="point"></pre>')
    return _page(f'Original frames {rows[0]["source_frame"]}–{rows[-1]["source_frame"]}', content,
                 {'rows': rows, 'pairs': [{'object_id': obj, 'hand': hand} for obj, hand in keys]},
                 _HAND_FRAME_SCRIPT if hand_mode else _FRAME_SCRIPT)


def build_equipment_support_report(result_raw: bytes, *, max_seconds=5):
    """Inspect exact saved selector bytes; the caller authenticates selected input seals."""
    _require(type(max_seconds) in (int, float) and 0 < max_seconds <= 5, 'cooperative budget must be in (0, 5] seconds')
    deadline = time.monotonic() + max_seconds
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError('Equipment support report cooperative deadline exceeded')
    check()
    _require(type(result_raw) is bytes and 0 < len(result_raw) <= MAX_INPUT
             and len(result_raw.removesuffix(b'\n')) <= 8 << 20, 'complete saved result exceeds eight MiB')
    result = _json(result_raw); check()
    families = [name for name in ('equipment_silhouette_constraint', 'equipment_hand_constraint') if name in result]
    _require(len(families) == 1, 'exactly one explicit saved image-constraint family required')
    hand_mode = families[0] == 'equipment_hand_constraint'
    constraint = result[families[0]]
    _require(type(constraint) is dict, 'saved image-constrained result required')
    original, keys, construction, frames = _baseline(result, constraint, check, hand_mode=hand_mode)
    ranges = _constraints(result, constraint, keys, construction, frames, check, hand_mode=hand_mode)
    points, decoded = constraint['hand_rows' if hand_mode else 'point_rows'], result['coverage']['decoded_frames']
    pages, page, page_bytes = [], [], 0
    for frame in range(frames):
        check(); position = original.get(frame)
        row = {'source_frame': frame, 'source_status': ('metadata_only_not_decoded' if frame >= decoded else
               'unsampled_decoded_frame' if position is None else result['rows'][position]['partition']),
               'pose': None if position is None else result['rows'][position],
               'window_range': None if position is None else ranges[position],
               'image_points': None if position is None or position >= construction else points[position]['hands' if hand_mode else 'pairs']}
        if hand_mode:
            row['saved_hand_header'] = None if position is None or position >= construction else {
                name: value for name, value in points[position].items() if name != 'hands'}
        size = len(_encode(row))
        _require(size <= PAGE_BYTES, 'one complete frame exceeds the page bound; no evidence dropped')
        if page and (len(page) == PAGE_FRAMES or page_bytes + size > PAGE_BYTES):
            pages.append(page); page, page_bytes = [], 0
        page.append(row); page_bytes += size
    if page: pages.append(page)
    _require(len(pages) + 5 <= MAX_FILES, 'complete frame pagination exceeds file bound')
    frame_pages = {row['source_frame']: f'frames-{i+1:03d}.html' for i, page in enumerate(pages) for row in page}
    window_index = [{'source_frame': row['frame'], 'pose_evidence_id': row['id'], 'page': frame_pages[row['frame']]}
                    for row in result['rows'][:construction]]
    check()
    with (Path(__file__).parent / 'assets' / 'brand.css').open('rb') as stream:
        css = stream.read((32 << 10) + 1)
    _require(len(css) <= 32 << 10, 'brand asset exceeds its byte bound'); check()
    files = {'support-evidence.json': result_raw, 'brand.css': css,
             'window-index.js': b'window.WeftWindowIndex=' + _encode(window_index).replace(b'<', b'\\u003c') + b';\n'}
    navigation, page_index = [], []
    for i, page in enumerate(pages):
        check(); name = f'frames-{i+1:03d}.html'; files[name] = _frame_page(page, keys, i, len(pages), hand_mode=hand_mode)
        navigation.append(f'<li><a href="{name}">Original frames {page[0]["source_frame"]}–{page[-1]["source_frame"]}</a></li>')
        page_index.append({'path': name, 'first_source_frame': page[0]['source_frame'],
                           'last_source_frame': page[-1]['source_frame'], 'frames': len(page)})
    summary = {'schema_version': 1, 'profile': HAND_PROFILE if hand_mode else PROFILE, 'status': 'saved_candidate_comparison_inspected',
        'source_sha256': result['source_sha256'], 'equipment_source_sha256': result['equipment_source_sha256'],
        'baseline_canonical_sha256': constraint['baseline_canonical_sha256'], 'baseline_reconstruction_verified': True,
        'original_counts': {**result['counts'], 'equipment_status_counts': constraint['baseline_equipment_status_counts']},
        'active_counts': result['counts'], 'constraint_counts': constraint['counts'], 'coverage': result['coverage'],
        'complete_source_frames': frames, 'pages': page_index, 'window_index_rows': len(window_index),
        'input_files': {'support-evidence.json': sha256(result_raw).hexdigest()},
        'asset_files': {'brand.css': sha256(css).hexdigest()}, 'source_seal_verification': 'caller_owned_not_performed_here',
        'geometry_recomputed': False, 'physical_validation': 'unverified', 'contact': 'unknown', 'support': 'unknown',
        'physical_noncontact': 'unknown', 'load': 'unknown', 'grasp': 'unknown', 'body_or_floor_installed': False,
        'scope': 'Saved image-only candidate comparison; no source, mask, camera, contact, body or simulator admission.'}
    if hand_mode:
        summary['constraint_profile'] = HAND_CONSTRAINT_PROFILE
        summary['hand_groups'] = constraint['input_bindings']['hand_groups']
    files['report.json'] = _encode(summary) + b'\n'
    esc = lambda value: html.escape(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False))
    content = '<p class="notice">Compare the original motion-based candidates with image-silhouette constraints. This can rule out an image explanation or leave it unresolved; it cannot verify physical contact, support, grasp or load.</p>'
    content += f'<section><h2>What changed</h2><pre>{esc(constraint["counts"])}</pre><p>Only original motion candidates can be kept, excluded or marked unavailable. Noncandidates and later frames are unchanged. An unavailable point takes precedence over an exclusion.</p><details><summary>Original versus active status counts</summary><pre>{esc({"original": constraint["baseline_equipment_status_counts"], "active": result["counts"]["equipment_status_counts"]})}</pre></details></section>'
    content += f'<section><h2>Every original frame</h2><p>{frames} source-frame slots; {len(result["rows"])} retained poses; {construction} construction poses. Unsampled decoded slots and metadata-only tails remain explicit. The original first-sixty-percent pose split is unchanged.</p><ul>{"".join(navigation)}</ul><details><summary>Complete original coverage and denominators</summary><pre>{esc({"counts": summary["original_counts"], "coverage": result["coverage"]})}</pre></details></section>'
    content += f'<section><h2>Assumptions and units</h2><p>The fixed-camera assumption is authored and unverified. Original pixel positions and torso-normalized motion are not world distances. Silhouette intersection/exclusion is conditional on saved masks and authored wrist pixel ranges; zero image distance is not proof of intersection. Source time, decoder timestamps and derivative clocks are not interchangeable.</p><details><summary>Saved camera assumption, image scale and original contract</summary><pre>{esc({key: result[key] for key in ("camera_prior", "image_scale", "contract", "limits")})}</pre></details><p>No ground height or body is installed; physical noncontact also remains unknown.</p></section>'
    content += f'<section><h2>Evidence and reproducibility</h2><ul><li><a href="support-evidence.json">Complete exact saved result</a></li><li><a href="report.json">Inspector counts and byte identities</a></li></ul><p>The original baseline is reconstructed and its saved hash checked. This is a saved-data consistency check, not independent source authentication or a rerun of the geometry.</p><details><summary>Source and input identities</summary><pre>{esc({"input_bindings": constraint["input_bindings"], "source_equivalence": result["equipment_source_equivalence"], "baseline_canonical_sha256": constraint["baseline_canonical_sha256"]})}</pre></details><details><summary>Recorded work and bounds</summary><pre>{esc({"work": constraint["work"], "bounds": constraint["bounds"]})}</pre></details></section>'
    if hand_mode:
        content = content.replace('image-silhouette constraints', 'four-point hand image constraints')
        content = content.replace('An unavailable point takes precedence over an exclusion.',
            'All four named points must qualify. Any overlapping rectangle makes the group image-consistent; four disjoint rectangles exclude it. A missing point leaves it unknown, even if another point overlaps. The original wrist status stays separate.')
        content = content.replace('authored wrist pixel ranges', 'authored named-point pixel ranges')
        content += '<section><h2>Four points are not a hand surface</h2><p>Wrist, pinky, index and thumb are separate source estimates. No filled hand shape, hull, grasp or contact is recovered. Raw source headers remain visible even when an identity or clock mismatch makes the group unavailable.</p></section>'
    files['index.html'] = _page('Equipment hand-point candidates' if hand_mode else 'Equipment support candidates', content)
    check()
    _require(len(files) <= MAX_FILES and sum(map(len, files.values())) <= MAX_OUTPUT
             and all(len(raw) <= MAX_HTML for name, raw in files.items() if name.endswith('.html')),
             'complete offline output exceeds its bound; no slots dropped')
    return files
