"""Construction-only box-frame inference; image consistency is not physical stationarity."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
import re
import time

from weft.equipment_proposal_config import CONSENSUS_METHOD
from weft.io import canonical_hash


PROFILE = 'equipment-box-consensus-selection-v1'
REFERENCE_PROFILE = 'equipment-box-consensus-reference-v1'
MAX_REFERENCE_BYTES = 4096
MAX_INPUT_BYTES, MAX_OUTPUT_BYTES = 32 << 20, 4 << 20
MAX_SECONDS, MAX_OBJECTS, MAX_FRAMES, MAX_REFERENCES = 5., 12, 1800, 900
POLICY = {
    'method': CONSENSUS_METHOD, 'minimum_eligible_frames': 3, 'minimum_span_s': .2,
    'maximum_edge_range_fraction': .02, 'maximum_fitted_drift_fraction': .01,
    'maximum_cohort_gap_fraction': .01, 'minimum_cohort_size': 2,
    'coordinates': 'left/width, top/height, right/width, bottom/height',
    'selection': 'Exact rational L1 medoid of all eligible boxes; earliest original frame breaks exact ties',
    'drift': 'Unweighted least-squares edge slope times the complete eligible source-time span',
    'cohorts': 'Every adjacent sorted edge gap separating at least two samples on each side',
    'exclusions': 'Only unavailable source/evidence/box/body-reference slots; never statistical outliers',
    'partial_visibility': 'Included with the original visibility; apparent occlusion can cause abstention',
    'scope': 'Fixed authored image-consistency tolerances, not calibrated uncertainty, physical stationarity, floor, depth or contact',
}
SLOT_FIELDS = ['source_frame', 'observation_id', 'source_time_s', 'partition', 'admission', 'visibility',
               'body_reference_index', 'body_time_s', 'evidence_ids', 'reason', 'bounds_px',
               'edge_deviation_fraction', 'medoid_l1_cost']
REASONS = {'metadata_only_not_decoded', 'partition_withheld', 'not_original_body_reference',
           'source_visibility_unavailable', 'ambiguous_or_missing_evidence', 'inferred_image_box_unavailable'}
RECEIPT_FIELDS = set(('schema_version profile method construction_evidence_sha256 source_binding_sha256 '
    'body_references_sha256 source_sha256 world_object_id entity_ref observation_source_sha256 source_clock '
    'image decoded_frames original_source_slots body_reference_count policy slot_fields slots status reasons '
    'selected diagnostic_medoid statistics physical_stationarity physical_error_coverage contact source_custody').split())
UNAVAILABLE_REASONS = {'insufficient_eligible_frames', 'insufficient_time_span', 'edge_range_exceeds_image_policy',
                       'ordered_image_drift', 'competing_image_cohorts'}


def _require(ok, message):
    if not ok:
        raise ValueError('Equipment box consensus: ' + message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('Equipment box consensus exceeded its aggregate time bound')


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _document(raw):
    _require(type(raw) is bytes and 0 < len(raw) <= MAX_INPUT_BYTES, 'bounded exact construction bytes required')
    def unique(pairs):
        out = {}
        for key, value in pairs:
            _require(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def nonfinite(_):
        raise ValueError('Equipment box consensus requires finite JSON')
    try:
        out = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
        stack, nodes = [(out, 0)], 0
        while stack:
            value, depth = stack.pop(); nodes += 1
            _require(nodes <= 2_000_000 and depth <= 32, 'JSON tree exceeds its bound')
            if isinstance(value, (dict, list)):
                items = value.values() if isinstance(value, dict) else value
                _require(nodes + len(value) <= 2_000_000, 'JSON tree exceeds its bound')
                stack.extend((item, depth + 1) for item in items)
            elif isinstance(value, float):
                _require(math.isfinite(value), 'nonfinite JSON number')
    except (UnicodeError, RecursionError, json.JSONDecodeError) as error:
        raise ValueError('Equipment box consensus requires bounded finite JSON') from error
    _require(isinstance(out, dict), 'construction document must be an object')
    return out


def _clock(source, frame):
    contract = source['source']
    return contract['frame_times_s'][frame] if source['observation_schema_version'] == 3 else frame / contract['fps']


def _prepare(raw, binding, references, object_ids, deadline):
    document = _document(raw); _check(deadline)
    _require(type(document.get('schema_version')) is int and document['schema_version'] == 1
             and document.get('profile') == 'world-construction-evidence-v1'
             and document.get('allowed_partitions') == ['construction']
             and document.get('physical_validation') == 'unverified' and 'source_frame_selection' not in document,
             'complete construction-only evidence required')
    _require(isinstance(binding, dict) and binding.get('mode') == 'source_qualified'
             and _sha(binding.get('source_sha256')), 'source-qualified binding required')
    count = binding.get('total_source_frames')
    _require(type(count) is int and 1 <= count <= MAX_FRAMES, 'source frame bound')
    entities = binding.get('entity_bindings')
    _require(isinstance(entities, list) and 2 <= len(entities) <= MAX_OBJECTS + 1
             and all(isinstance(row, dict) and isinstance(row.get('world_entity_id'), str) for row in entities),
             'bounded complete entity bindings required')
    entities = {row['world_entity_id']: row for row in entities}
    _require(len(entities) == len(binding['entity_bindings']) and 'human' in entities, 'duplicate or missing entity binding')
    _require(isinstance(object_ids, list) and 1 <= len(object_ids) <= MAX_OBJECTS
             and all(isinstance(name, str) and name != 'human' and name in entities for name in object_ids)
             and len(set(object_ids)) == len(object_ids), 'unique bounded selected equipment IDs required')
    aliases = {binding['source_sha256']}
    mappings = binding.get('source_equivalences', [])
    _require(isinstance(mappings, list) and len(mappings) <= 7, 'bounded source equivalence bindings required')
    for mapping in mappings:
        _require(isinstance(mapping, dict) and mapping.get('original_source_sha256') == binding['source_sha256']
                 and _sha(mapping.get('derivative_source_sha256')) and _sha(mapping.get('bundle_manifest_sha256')),
                 'source equivalence identity differs')
        aliases.add(mapping['derivative_source_sha256'])
    sources = document.get('sources')
    _require(isinstance(sources, list) and 1 <= len(sources) <= 8
             and all(isinstance(item, dict) and _sha(item.get('document_sha256')) for item in sources),
             'bounded source inventory required')
    source_map = {item['document_sha256']: item for item in sources}
    _require(len(source_map) == len(sources) and set(source_map) == set(binding.get('evidence_documents', {})),
             'exact observation document inventory differs')
    rows, evidence = {}, {}
    for source in sources:
        _check(deadline)
        sha, contract = source['document_sha256'], source.get('source', {})
        image = source.get('image', {})
        _require(contract.get('sha256') in aliases and contract.get('decoded_frames') == count
                 and contract.get('decode_interval', [0, count - 1]) == [0, count - 1]
                 and contract.get('source_complete') is not False, 'source identity/coverage differs')
        _require(type(contract.get('metadata_frames')) is int and 1 <= contract['metadata_frames'] <= MAX_FRAMES,
                 'declared source slots exceed bound')
        _require(all(type(image.get(key)) is int and 1 <= image[key] <= 4096 for key in ('width_px', 'height_px'))
                 and all(image[key] == contract.get(key) for key in ('width_px', 'height_px')), 'source image dimensions differ')
        if source.get('observation_schema_version') == 3:
            stamps = contract.get('frame_times_s')
            _require(contract['metadata_frames'] == count and isinstance(stamps, list) and len(stamps) == count
                     and all(_number(t) and 0 <= t <= 3600 for t in stamps)
                     and all(a < b for a, b in zip(stamps, stamps[1:])), 'source timestamp table differs')
        else:
            _require(source.get('observation_schema_version') in (1, 2)
                     and _number(contract.get('fps')) and 0 < contract['fps'] <= 1000, 'source frame clock differs')
        objects = source.get('objects')
        _require(isinstance(objects, list) and 1 <= len(objects) <= MAX_OBJECTS
                 and all(isinstance(obj, dict) and isinstance(obj.get('id'), str) for obj in objects), 'source entity bound')
        ids = {obj['id'] for obj in objects}
        _require(len(ids) == len(objects), 'duplicate source entity')
        records = source.get('evidence')
        _require(isinstance(records, list) and len(records) <= count * len(ids) * 4, 'evidence record bound')
        for item in records:
            _require(isinstance(item, dict) and isinstance(item.get('id'), str) and len(item['id']) <= 512
                     and (sha, item['id']) not in evidence and item.get('document_sha256') == sha, 'evidence identity differs')
            evidence[(sha, item['id'])] = item
        observed = source.get('observations')
        _require(isinstance(observed, list) and len(observed) == count * len(ids), 'complete source slots required')
        for row in observed:
            _check(deadline)
            _require(isinstance(row, dict) and type(row.get('frame')) is int and 0 <= row['frame'] < count
                     and row.get('object_id') in ids and isinstance(row.get('id'), str) and len(row['id']) <= 512,
                     'source slot identity differs')
            key = sha, row['object_id'], row['frame']
            _require(key not in rows and row.get('entity_ref') == {'document_sha256': sha, 'object_id': row['object_id']}
                     and row.get('time_s') == _clock(source, row['frame']), 'source slot entity/clock differs')
            included = row.get('partition') == 'construction'
            _require(row.get('admission') == ('included' if included else 'withheld'), 'source partition admission differs')
            _require(isinstance(row.get('measurements'), dict) and isinstance(row.get('evidence_refs'), list)
                     and len(row['evidence_refs']) <= 4, 'source measurements/evidence differ')
            if not included:
                _require(not row['measurements'] and not row['evidence_refs'] and row.get('visibility') is None,
                         'withheld payload must remain absent')
            for ref in row['evidence_refs']:
                _require(isinstance(ref, dict) and ref.get('document_sha256') == sha, 'document-qualified evidence required')
                item = evidence.get((sha, ref.get('evidence_id')), {})
                _require(all(item.get(k) == row[k] for k in ('frame', 'time_s', 'object_id')), 'evidence source/clock differs')
            rows[key] = row
    for entity in entities.values():
        key = entity.get('observation_document_sha256'), entity.get('observation_entity_id')
        _require(key[0] in source_map and all((*key, f) in rows for f in range(count)), 'bound source entity is missing')
    human = entities['human']; human_source = source_map[human['observation_document_sha256']]
    _require(human_source['source']['sha256'] == binding['source_sha256'], 'body must retain original source identity')
    _require(isinstance(references, list) and 3 <= len(references) <= MAX_REFERENCES, 'bounded complete body references required')
    by_frame, previous, previous_time = {}, -1, -1.
    for index, ref in enumerate(references):
        _require(isinstance(ref, dict) and set(ref) == {'index', 'source_frame', 'evidence_id', 'time_s'}
                 and type(ref['index']) is int and ref['index'] == index and type(ref['source_frame']) is int
                 and previous < ref['source_frame'] < count and _number(ref['time_s']) and previous_time < ref['time_s'] <= 3600,
                 'ordered body reference identity/clock differs')
        frame = ref['source_frame']; row = rows[(human['observation_document_sha256'], human['observation_entity_id'], frame)]
        _require(ref['evidence_id'] == f"{binding['source_sha256']}:frame:{frame}" and row['time_s'] == ref['time_s']
                 and row['admission'] == 'included' and row['visibility'] in ('detected', 'partial')
                 and row['evidence_refs'] == [{'document_sha256': human['observation_document_sha256'], 'evidence_id': ref['evidence_id']}],
                 'body reference lacks its exact original construction evidence')
        by_frame[frame] = ref; previous, previous_time = frame, ref['time_s']
    _require(len(raw) + len(_bytes(binding)) + len(_bytes(references)) <= MAX_INPUT_BYTES, 'aggregate input byte bound')
    return entities, source_map, rows, by_frame, max(count, human_source['source']['metadata_frames'])


def _statistics(values, times, deadline):
    n = len(values); costs = [Fraction(0)] * n
    ranges, gaps, drifts = [], [], []
    exact_times = list(map(Fraction, times)); mean_time = sum(exact_times) / n
    time_deltas = [t - mean_time for t in exact_times]
    variance = sum(t * t for t in time_deltas)
    span = exact_times[-1] - exact_times[0]
    for dimension in range(4):
        _check(deadline)
        ordered = sorted((point[dimension], index) for index, point in enumerate(values))
        total, prefix = sum(point for point, _ in ordered), Fraction(0)
        for rank, (point, index) in enumerate(ordered):
            costs[index] += point * rank - prefix + (total - prefix - point) - point * (n - rank - 1)
            prefix += point
        ranges.append(ordered[-1][0] - ordered[0][0])
        gaps.append(max((ordered[i][0] - ordered[i - 1][0] for i in range(2, n - 1)), default=Fraction(0)))
        mean = sum(point[dimension] for point in values) / n
        drifts.append(sum(t * (point[dimension] - mean) for t, point in zip(time_deltas, values)) * span / variance
                      if variance else Fraction(0))
    selected = min(range(n), key=lambda index: (costs[index], index))
    return selected, costs, ranges, drifts, gaps, span


def _select_equipment_box_frames(construction_evidence, *, source_binding, body_references, object_ids, deadline=None):
    """Infer one observed frame per requested object from already source-qualified construction bytes."""
    started = time.monotonic()
    _require(deadline is None or _number(deadline), 'finite caller deadline required')
    deadline = min(started + MAX_SECONDS, deadline) if deadline is not None else started + MAX_SECONDS
    _check(deadline)
    entities, sources, indexed, body, total = _prepare(construction_evidence, source_binding, body_references, object_ids, deadline)
    common = {'construction_evidence_sha256': hashlib.sha256(construction_evidence).hexdigest(),
              'source_binding_sha256': canonical_hash(source_binding), 'body_references_sha256': canonical_hash(body_references)}
    result = {}
    for name in object_ids:
        _check(deadline)
        entity = entities[name]; document_sha = entity['observation_document_sha256']; oid = entity['observation_entity_id']
        source = sources[document_sha]; width, height = (source['image'][key] for key in ('width_px', 'height_px'))
        slots, eligible, values, times = [], [], [], []
        for frame in range(total):
            _check(deadline)
            row = indexed.get((document_sha, oid, frame)); ref = body.get(frame)
            reason = ('metadata_only_not_decoded' if row is None else 'partition_withheld' if row['admission'] != 'included' else
                      'not_original_body_reference' if ref is None else 'source_visibility_unavailable' if row['visibility'] not in ('detected', 'partial') else
                      'ambiguous_or_missing_evidence' if len(row['evidence_refs']) != 1 else None)
            bounds = None
            if reason is None:
                box = row['measurements'].get('box', {}); value = box.get('value') if isinstance(box, dict) else None
                good = (isinstance(box, dict) and (box.get('coordinate_system'), box.get('unit'), box.get('status')) == ('image', 'pixels', 'inferred')
                        and isinstance(value, list) and len(value) == 4 and all(_number(v) for v in value))
                if good:
                    x, y, w, h = value
                    good = 0 <= x < x + w <= width and 0 <= y < y + h <= height
                if good:
                    bounds = [x, y, x + w, y + h]
                else:
                    reason = 'inferred_image_box_unavailable'
            slots.append([frame, row['id'] if row else None, row['time_s'] if row else None,
                row['partition'] if row else None, row['admission'] if row else None, row['visibility'] if row else None,
                ref['index'] if ref else None, ref['time_s'] if ref else None,
                [item['evidence_id'] for item in row['evidence_refs']] if row else [], reason, bounds, None, None])
            if reason is None:
                eligible.append(frame); times.append(row['time_s'])
                values.append([Fraction(v) / size for v, size in zip(bounds, (width, height, width, height))])
        reasons, candidate, selected, stats = [], None, None, None
        if len(eligible) < POLICY['minimum_eligible_frames']:
            reasons.append('insufficient_eligible_frames')
        if eligible:
            index, costs, ranges, drifts, gaps, span = _statistics(values, times, deadline)
            frame = eligible[index]; slot = slots[frame]
            candidate = {'source_frame': frame, 'observation_id': slot[1], 'time_s': slot[2], 'evidence_id': slot[8][0],
                         'body_reference_index': slot[6], 'body_time_s': slot[7], 'bounds_px': slot[10]}
            for i, frame in enumerate(eligible):
                slots[frame][11] = [float(value - center) for value, center in zip(values[i], values[index])]
                slots[frame][12] = float(costs[i])
            stats = {'eligible_count': len(eligible), 'source_time_span_s': float(span),
                     'edge_range_fraction': list(map(float, ranges)), 'fitted_drift_fraction': list(map(float, drifts)),
                     'maximum_cohort_gap_fraction': list(map(float, gaps)), 'diagnostic_medoid_l1_cost': float(costs[index])}
            for failed, reason in (
                (span < Fraction('0.2'), 'insufficient_time_span'),
                (max(ranges) > Fraction('0.02'), 'edge_range_exceeds_image_policy'),
                (max(map(abs, drifts)) > Fraction('0.01'), 'ordered_image_drift'),
                (max(gaps) > Fraction('0.01'), 'competing_image_cohorts')):
                if failed: reasons.append(reason)
            if not reasons: selected = deepcopy(candidate)
        receipt = {'schema_version': 1, 'profile': PROFILE, 'method': CONSENSUS_METHOD, **common,
            'source_sha256': source_binding['source_sha256'], 'world_object_id': name,
            'entity_ref': {'document_sha256': document_sha, 'object_id': oid},
            'observation_source_sha256': source['source']['sha256'], 'source_clock': deepcopy(source['time']),
            'image': deepcopy(source['image']), 'decoded_frames': source_binding['total_source_frames'],
            'original_source_slots': total, 'body_reference_count': len(body_references),
            'policy': deepcopy(POLICY), 'slot_fields': list(SLOT_FIELDS), 'slots': slots,
            'status': 'selected' if selected is not None else 'unavailable', 'reasons': reasons,
            'selected': selected, 'diagnostic_medoid': candidate, 'statistics': stats,
            'physical_stationarity': 'unknown', 'physical_error_coverage': 'unknown', 'contact': 'unknown',
            'source_custody': 'Caller-qualified complete construction binding; source proof is not rerun by this pure estimator'}
        result[name] = receipt
    _require(len(_bytes(result)) <= MAX_OUTPUT_BYTES, 'complete selection receipts exceed output bound; none were trimmed')
    _check(deadline)
    return result


def select_equipment_box_frames(construction_evidence, *, source_binding, body_references, object_ids, deadline=None):
    """Use the caller-qualified complete binding; this pure step never reopens source media or proofs."""
    try:
        return _select_equipment_box_frames(construction_evidence, source_binding=source_binding,
            body_references=body_references, object_ids=object_ids, deadline=deadline)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise ValueError('Equipment box consensus: malformed bounded source or selection input') from error


def validate_selection_receipt(value):
    """Check saved receipt shape only; construction separately recomputes every inferred value."""
    _require(isinstance(value, dict) and set(value) == RECEIPT_FIELDS and len(_bytes(value)) <= MAX_OUTPUT_BYTES
             and type(value.get('schema_version')) is int and value['schema_version'] == 1
             and value.get('profile') == PROFILE and value.get('method') == CONSENSUS_METHOD
             and value.get('policy') == POLICY and value.get('slot_fields') == SLOT_FIELDS, 'unsupported saved selection receipt')
    for key in ('source_sha256', 'observation_source_sha256', 'construction_evidence_sha256', 'source_binding_sha256', 'body_references_sha256'):
        _require(_sha(value.get(key)), 'missing saved receipt identity: ' + key)
    entity = value.get('entity_ref')
    _require(isinstance(entity, dict) and set(entity) == {'document_sha256', 'object_id'} and _sha(entity['document_sha256'])
             and isinstance(entity['object_id'], str) and isinstance(value.get('world_object_id'), str), 'saved entity reference differs')
    count, rows = value.get('original_source_slots'), value.get('slots')
    _require(type(count) is int and 1 <= count <= MAX_FRAMES and isinstance(rows, list) and len(rows) == count,
             'saved complete slot inventory differs')
    _require(type(value['decoded_frames']) is int and 1 <= value['decoded_frames'] <= count
             and type(value['body_reference_count']) is int and 3 <= value['body_reference_count'] <= MAX_REFERENCES
             and isinstance(value['image'], dict)
             and all(type(value['image'].get(key)) is int and 1 <= value['image'][key] <= 4096 for key in ('width_px', 'height_px'))
             and isinstance(value['source_clock'], dict)
             and all(value[key] == 'unknown' for key in ('physical_stationarity', 'physical_error_coverage', 'contact')),
             'saved coverage/physical scope differs')
    eligible = []
    for frame, row in enumerate(rows):
        _require(isinstance(row, list) and len(row) == len(SLOT_FIELDS) and type(row[0]) is int and row[0] == frame,
                 'saved slot shape/order differs')
        _require(row[9] is None or isinstance(row[9], str) and row[9] in REASONS, 'saved slot reason differs')
        _require(isinstance(row[8], list) and len(row[8]) <= 4 and all(isinstance(v, str) and len(v) <= 512 for v in row[8])
                 and (row[6] is None and row[7] is None or type(row[6]) is int
                      and 0 <= row[6] < value['body_reference_count'] and _number(row[7])), 'saved slot evidence/body identity differs')
        if row[9] is None:
            _require(isinstance(row[10], list) and len(row[10]) == 4 and all(_number(v) for v in row[10])
                     and isinstance(row[11], list) and len(row[11]) == 4 and all(_number(v) for v in row[11])
                     and _number(row[12]) and row[12] >= 0 and row[3:5] == ['construction', 'included']
                     and type(row[6]) is int and isinstance(row[8], list) and len(row[8]) == 1,
                     'saved eligible evidence/diagnostics differ')
            eligible.append(frame)
        else:
            _require(row[10:] == [None, None, None], 'unavailable slot cannot expose selected coordinates')
    status, chosen = value.get('status'), value.get('selected')
    _require(status in ('selected', 'unavailable') and isinstance(value.get('reasons'), list)
             and all(isinstance(reason, str) and reason in UNAVAILABLE_REASONS for reason in value['reasons'])
             and len(set(value['reasons'])) == len(value['reasons']), 'saved selection status differs')
    _require(status == 'selected' and chosen == value.get('diagnostic_medoid') and not value['reasons']
             or status == 'unavailable' and chosen is None and bool(value['reasons']), 'saved selection/rejection contradicts status')
    candidate = value.get('diagnostic_medoid')
    stats = value.get('statistics')
    _require(stats is None if not eligible else isinstance(stats, dict) and set(stats) == {
        'eligible_count', 'source_time_span_s', 'edge_range_fraction', 'fitted_drift_fraction',
        'maximum_cohort_gap_fraction', 'diagnostic_medoid_l1_cost'}, 'saved diagnostic fields differ')
    if eligible:
        _require(stats['eligible_count'] == len(eligible) and type(stats['eligible_count']) is int
                 and _number(stats['source_time_span_s']) and stats['source_time_span_s'] >= 0
                 and _number(stats['diagnostic_medoid_l1_cost']) and stats['diagnostic_medoid_l1_cost'] >= 0
                 and all(isinstance(stats[key], list) and len(stats[key]) == 4 and all(_number(v) for v in stats[key])
                         for key in ('edge_range_fraction', 'fitted_drift_fraction', 'maximum_cohort_gap_fraction')),
                 'saved diagnostic dimensions/counts differ')
    _require(candidate is not None if eligible else candidate is None, 'saved medoid availability differs')
    if candidate is not None:
        _require(isinstance(candidate, dict) and candidate.get('source_frame') in eligible, 'saved medoid is not an eligible observed frame')
        row = rows[candidate['source_frame']]
        _require(candidate == {'source_frame': row[0], 'observation_id': row[1], 'time_s': row[2], 'evidence_id': row[8][0],
                               'body_reference_index': row[6], 'body_time_s': row[7], 'bounds_px': row[10]}, 'saved selected observation differs')
    return deepcopy(value)


def selection_reference(receipt):
    """Bind a compact scene reference to the complete retained receipt, without reselection."""
    receipt = validate_selection_receipt(receipt)
    _require(receipt['status'] == 'selected', 'only selected receipts can enter scene provenance')
    result = {'schema_version': 1, 'profile': REFERENCE_PROFILE, 'method': CONSENSUS_METHOD,
        'status': 'selected', 'receipt_location': 'source_candidate_package',
        'receipt_file': 'frame-selection.json', 'receipt_key': receipt['world_object_id'],
        'receipt_sha256': canonical_hash(receipt), 'receipt_hash_kind': 'canonical_json_sha256',
        'selected': receipt['selected'], 'policy_sha256': canonical_hash(receipt['policy']),
        'image_policy': {key: receipt['policy'][key] for key in ('minimum_eligible_frames', 'minimum_span_s',
            'maximum_edge_range_fraction', 'maximum_fitted_drift_fraction', 'maximum_cohort_gap_fraction',
            'minimum_cohort_size')},
        'eligible_count': receipt['statistics']['eligible_count'],
        **{key: receipt[key] for key in ('construction_evidence_sha256', 'source_binding_sha256',
            'body_references_sha256', 'source_sha256', 'observation_source_sha256', 'entity_ref', 'source_clock',
            'original_source_slots', 'decoded_frames', 'body_reference_count',
            'physical_stationarity', 'physical_error_coverage', 'contact')}}
    _require(len(_bytes(result)) <= MAX_REFERENCE_BYTES, 'scene receipt reference exceeds its 4 KiB bound')
    return result


def validate_selection_reference(value, receipt):
    """Check the exact reference-to-receipt join; source rederivation remains the constructor's job."""
    _require(isinstance(value, dict) and len(_bytes(value)) <= MAX_REFERENCE_BYTES,
             'bounded scene receipt reference required')
    expected = selection_reference(receipt)
    _require(canonical_hash(value) == canonical_hash(expected), 'scene receipt reference differs from complete receipt')
    return expected
