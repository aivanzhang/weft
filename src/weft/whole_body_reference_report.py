"""Offline inspection of sealed reference proposals; no geometry is recomputed."""

from bisect import bisect_left
from collections import Counter
from datetime import datetime
from hashlib import sha256
import base64
import html
import json
import math
import os
from pathlib import Path
import re
import time
from zoneinfo import ZoneInfo

from weft.shared_world_packaging import read_package_file, write_package_files


PROFILE = "whole-body-reference-report-v1"
SURFACE_METHOD = "blocked-whole-body-surface-interval-reference-v1"
SURFACE_PROFILE = "whole-body-surface-interval-fit-v1"
SURFACE_OBJECTIVE = "source-surface-interval-objective-v1"
METHODS = {"blocked-whole-body-floor-reference-v1", "blocked-whole-body-floor-first-reference-v1", SURFACE_METHOD}
LIMIT = .025
MEMBERS = {"candidate-human.json": "candidate-human.json.txt", "report.json": "proposal-report.json.txt",
           "inputs/human.json": "source-human.json.txt", "inputs/evidence-derived.json": "evidence-derived.json.txt"}
MAX_BYTES, MAX_HTML = 16 * 1024 * 1024, 2 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_TARGET_HASHES = ('original_observation_sha256', 'proposal_sha256', 'decision_sha256')
_CROP_METHOD = 'single-view-selective-crop-v1'
_COUNTERS = {'floor_qp_calls', 'image_qp_calls', 'floor_descent_accepts', 'image_tiebreak_accepts',
             'invalid_directions', 'zero_directions', 'backtrack_trials', 'rejected_backtracks', 'qp_status_counts'}
_QP_STATUSES = {'optimal', 'iteration_limit', 'time_limit', 'unsupported_infeasible_initial', 'dependent_active_set',
                'kkt_verification_failed', 'solution_bound', 'numerical_feasibility_loss', 'dependent_blocker', 'other'}
_PRIMITIVES = {'torso_geom': 'Torso', 'head_geom': 'Head', 'shoulder_frame': 'Shoulder frame', 'pelvis_frame': 'Pelvis frame',
               **{f'{side}_{joint}_geom': f'{side.capitalize()} {segment} capsule' for side in ('left', 'right')
                  for joint, segment in (('knee', 'shin'), ('hip', 'thigh'), ('shoulder', 'upper arm'), ('elbow', 'forearm'))}
              }


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _canonical(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, "Duplicate report JSON key")
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=pairs)
        json.dumps(result, allow_nan=False)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError('Report evidence must be bounded strict JSON') from error
    _require(isinstance(result, dict), "Report evidence must be a JSON object")
    return result


def _text(value, maximum=512):
    _require(isinstance(value, str) and len(value) <= maximum, "Unbounded report text")
    return value


def _number(value, *, low=-10000, high=10000):
    if value is not None:
        _require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
                 "Invalid bounded report number")
    return value


def _count(value, maximum=10_000_000):
    if value is not None:
        _require(type(value) is int and 0 <= value <= maximum, "Invalid report count")
    return value


def _support_label(name, primitive):
    label = _PRIMITIVES.get(primitive, 'Unknown saved primitive')
    endpoint = name[len(primitive) + 1:] if name.startswith(primitive + ':') else None
    suffix = {'-1.0': 'endpoint −1', '1.0': 'endpoint +1', '0.0': 'center sample'}.get(endpoint, 'unmapped sample')
    return f'{label} · {suffix}'


def _target_info(frame, reference):
    descriptor = reference.get('target_origin')
    selected = 'image_refinement' in frame or 'image_refinement_raw' in frame
    path = 'identity_raw.landmarks' if 'identity_raw' in frame else 'landmarks'
    if selected:
        meta, raw = frame.get('image_refinement'), frame.get('image_refinement_raw')
        _require('identity_raw' not in frame and isinstance(meta, dict) and isinstance(raw, dict)
                 and not any(key in raw for key in ('identity_raw', 'image_refinement', 'image_refinement_raw')),
                 'Selected crop targets cannot mix identity repair or omit preserved raw evidence')
        _require(meta.get('method') == _CROP_METHOD and meta.get('anatomical_identity_verified') is False,
                 'Unsupported selected crop provenance')
        _require(all(isinstance(meta.get(key), str) and _SHA.fullmatch(meta[key]) for key in _TARGET_HASHES),
                 'Crop provenance requires exact SHA256 identifiers')
        _require(all(key in frame and key in raw and frame[key] == raw[key]
                     for key in ('id', 'frame', 'time_s', 'decoder_timestamp_s')), 'Crop source identities or clocks differ')
        _require(_canonical(raw) == meta['original_observation_sha256']
                 and _canonical({k: v for k, v in frame.items() if k not in {'image_refinement', 'image_refinement_raw'}})
                 == meta['proposal_sha256'], 'Crop preserved/raw proposal identity differs')
        expected = {'kind': 'selected_crop_estimate_mapped_to_source', 'method': _CROP_METHOD,
                    **{key: meta[key] for key in _TARGET_HASHES}, 'anatomical_identity_verified': False}
        _require(type(descriptor) is dict and descriptor.get('anatomical_identity_verified') is False and descriptor == expected,
                 'Selected crop source requires its exact target_origin descriptor')
        label = 'Selected crop estimate, mapped to source image'
    else:
        _require(descriptor is None, 'Target origin claims a crop absent from its source frame')
        if 'identity_raw' in frame:
            raw = frame['identity_raw']
            _require(isinstance(raw, dict) and not any(key in raw for key in ('image_refinement', 'image_refinement_raw'))
                     and all(raw.get(key) == frame.get(key) for key in ('id', 'frame', 'time_s')),
                     'Preserved whole-frame target identity differs or mixes crop provenance')
        label = ('Inferred whole-frame estimate (preserved before identity processing)' if 'identity_raw' in frame
                 else 'Inferred whole-frame estimate')
    for point in reference['points']:
        recorded = point.get('landmark_source')
        _require(recorded == path if selected else recorded is None or recorded == path,
                 'Endpoint landmark_source must name its actual selected JSON path')
    return {'target_origin': descriptor, 'target_label': label, 'target_landmark_source': path}


def _surface_number(value, *, nonnegative=False):
    _require(type(value) in (int, float) and math.isfinite(value)
             and (not nonnegative or value >= 0), 'Surface values must be finite, with explicit unknowns')
    return value


def _surface_equal(value, expected, label):
    _require(value is None if expected is None else value is not None and
             math.isclose(_surface_number(value), expected, rel_tol=1e-9, abs_tol=1e-12), label)


def _surface_vector(value, size=3):
    _require(type(value) is list and len(value) == size, 'Surface vector shape differs')
    return [_surface_number(v) for v in value]


def _surface_result(result, slot, body_hash):
    _require(type(result) is dict and type(result.get('schema_version')) is int and result['schema_version'] == 1
             and result.get('profile') == SURFACE_OBJECTIVE and result.get('source_frame') == slot['source_frame']
             and result.get('body_sha256') == body_hash and result.get('body_reference') == slot['body_reference']
             and result.get('geometry_modified') is False and result.get('contact') == 'unknown'
             and result.get('physical_validation') == 'unverified', 'Surface evaluation identity or scope differs')
    channels = result.get('channels')
    _require(type(channels) is list and len(channels) == len(slot['channels']), 'Surface evaluation channel count differs')
    losses, distances = [], []
    for channel, joined in zip(channels, slot['channels']):
        _require(type(channel) is dict and all(channel.get(k) == v for k, v in joined.items()
                 if k not in {'world_envelope_m', 'status', 'reason'}), 'Surface evaluation source join differs')
        if channel.get('status') == 'conditional_interval_residual':
            _require(joined['status'] == 'conditional_interval_ready' and channel.get('reason') is None,
                     'Unavailable source channel cannot become an interval target')
            position, residual = _surface_vector(channel.get('world_position_m')), _surface_vector(channel.get('signed_outside_m'))
            bounds = joined['world_envelope_m']
            expected = [p - min(max(p, lo), hi) for p, lo, hi in zip(position, bounds['minimum'], bounds['maximum'])]
            _require(all(math.isclose(v, e, rel_tol=1e-9, abs_tol=1e-12) for v, e in zip(residual, expected)),
                     'Saved outside-box residual differs from supplied world bounds')
            active = [p < lo or p > hi for p, lo, hi in zip(position, bounds['minimum'], bounds['maximum'])]
            _require(type(channel.get('active_axes')) is list and all(type(v) is bool for v in channel['active_axes'])
                     and channel['active_axes'] == active, 'Saved interval active axes differ')
            distance, loss = math.hypot(*expected), .5 * sum(v*v for v in expected)
            _surface_equal(channel.get('distance_to_outer_envelope_m'), distance, 'Saved outside-box distance differs')
            _surface_equal(channel.get('objective_m2'), loss, 'Saved squared interval loss differs')
            losses.append(loss); distances.append(distance)
        else:
            _require(channel.get('status') == 'unavailable' and
                     (channel.get('reason') == joined['reason'] if joined['status'] == 'unavailable'
                      else channel.get('reason') == 'missing_candidate_body_point'), 'Unavailable interval reason changed')
            _require(all(channel.get(k) is None for k in ('world_position_m', 'signed_outside_m',
                     'distance_to_outer_envelope_m', 'objective_m2', 'active_axes', 'jacobian_m_per_rad')),
                     'Unavailable interval channel cannot have a residual or position')
    _require(result.get('counts') == {'retained_channels': len(channels), 'evaluated_channels': len(losses),
             'unavailable_channels': len(channels) - len(losses)}, 'Saved interval evaluation counts differ')
    _surface_equal(result.get('objective_m2'), sum(losses) if losses else None, 'Saved interval total differs')
    return losses, distances


def _surface_model(report, candidate, source, documents, hashes, check):
    surface = report.get('surface_intervals')
    _require(type(surface) is dict and type(surface.get('schema_version')) is int and surface['schema_version'] == 1
             and surface.get('profile') == SURFACE_PROFILE and surface.get('weight_status') == 'authored'
             and surface.get('weight_units') == 'existing_model_objective_units_per_supplied_world_metre_squared'
             and surface.get('contact') == 'unknown' and surface.get('physical_validation') == 'unverified'
             and surface.get('optimizer_installation') == 'installed_in_existing_whole_body_fitter',
             'Unsupported surface fit metadata or physical claim')
    config, plan = documents['inputs/surface-config.json'], documents['plan.json']
    weight = _surface_number(surface.get('weight'))
    _require(0 < weight <= 1000 and weight == config['weight'], 'Surface authored weight differs')
    inputs = surface.get('inputs')
    _require(type(inputs) is dict and inputs.get('profile') == SURFACE_OBJECTIVE
             and inputs.get('source_sha256') == source['source_sha256'] and inputs.get('body_sha256') == hashes['inputs/human.json']
             and inputs.get('body_model_sha256') == _canonical(source), 'Surface input body/source identities differ')
    _require(inputs.get('contact') == inputs.get('load') == inputs.get('grasp') == 'unknown'
             and inputs.get('physical_validation') == 'unverified' and inputs.get('geometry_modified') is False
             and inputs.get('body_or_floor_installed') is False and inputs.get('confidence') is None,
             'Surface inputs cannot claim physical qualification')
    expected_bindings = {'observation_document_sha256s': sorted(row['sha256'] for row in config['observations']),
        **{key + '_sha256': config[key]['sha256'] if config[key] is not None else None
           for key in ('camera', 'surface_intervals', 'annotations')}}
    binding_hash = hashes.get('inputs/surface-body-binding.json')
    _require(inputs.get('envelope_bindings') == expected_bindings and inputs.get('binding_sha256') == binding_hash
             and inputs.get('binding') == documents.get('inputs/surface-body-binding.json')
             and type(inputs.get('envelope_cache_sha256')) is str and _SHA.fullmatch(inputs['envelope_cache_sha256']),
             'Surface saved input bindings differ from exact copied inputs')
    work = surface.get('work_limits')
    _require(type(work) is dict and set(work) == {'maximum_evaluations', 'maximum_journal_bytes'}
             and type(work['maximum_evaluations']) is int and 1 <= work['maximum_evaluations'] <= 20000
             and type(work['maximum_journal_bytes']) is int and 1 <= work['maximum_journal_bytes'] <= 4 << 20,
             'Surface work bounds differ')
    _require(type(surface.get('evaluations')) is int and 0 <= surface['evaluations'] <= work['maximum_evaluations'],
             'Surface evaluation count exceeds its bound')
    marker = candidate['whole_body_reference_refinement'].get('surface_intervals')
    _require(marker == {'profile': SURFACE_PROFILE, 'input_bindings': {
        'objective_profile': SURFACE_OBJECTIVE, **{key: inputs[key] for key in
            ('source_sha256', 'body_sha256', 'binding_sha256', 'envelope_cache_sha256', 'envelope_bindings')}},
        'weight': weight, 'weight_status': 'authored', 'work_limits': work, 'summary': surface.get('summary')},
        'Candidate surface marker differs from saved report')
    _require(source['reference'][0] == candidate['reference'][0], 'Surface fitting changed the frozen first reference')

    observations, clocks, signatures = {}, {}, set()
    for i, entry in enumerate(config['observations']):
        document = documents[f'inputs/surface-observation-{i:02d}.json']; check()
        metadata = document.get('source', {})
        _require(metadata.get('sha256') == source['source_sha256'], 'Surface observation source differs')
        decoded, declared = metadata.get('decoded_frames'), metadata.get('metadata_frames')
        _require(type(decoded) is int and type(declared) is int and 0 <= decoded <= 1800 and 0 <= declared <= 1800,
                 'Surface original frame counts exceed bounds')
        signatures.add((decoded, declared)); clocks[entry['sha256']] = document.get('time')
        rows = document.get('observations')
        _require(type(rows) is list and len(rows) <= 100000, 'Surface observation slots exceed bounds')
        for row in rows:
            _require(type(row) is dict and type(row.get('frame')) is int and 0 <= row['frame'] < max(decoded, declared),
                     'Surface observation frame differs')
            key = (entry['sha256'], _text(row.get('object_id'), 120), row['frame'])
            _require(key not in observations, 'Duplicate source observation slot')
            observations[key] = row
    _require(len(signatures) == 1 and inputs.get('source_clocks') == clocks, 'Surface original coverage or clocks differ')
    decoded, declared = signatures.pop(); frame_count = max(decoded, declared)
    annotations = documents.get('inputs/surface-annotations.json', {})
    declarations = annotations.get('channels', [])
    _require(type(declarations) is list and len(declarations) <= 12
             and len({row['id'] for row in declarations}) == len(declarations), 'Surface channel declarations differ')
    annotation_rows = annotations.get('annotations', [])
    _require(type(annotation_rows) is list and len(annotation_rows) <= 1800 * 12, 'Supplied annotation count exceeds bounds')
    annotation_index = {}
    for row in annotation_rows:
        check(); key = (row.get('frame'), row.get('channel_id'))
        _require(key not in annotation_index and type(key[0]) is int and 0 <= key[0] < frame_count,
                 'Duplicate or foreign supplied annotation')
        annotation_index[key] = row
    surface_rows = documents.get('inputs/surface-intervals.json', {}).get('surfaces', [])
    _require(type(surface_rows) is list and len(surface_rows) <= 12, 'Supplied surface count exceeds bounds')
    surface_index = {row['id']: row for row in surface_rows}
    _require(len(surface_index) == len(surface_rows), 'Duplicate supplied surface')
    for row in surface_rows:
        bounds = row.get('height_interval_m')
        if bounds is not None:
            lo, hi = _surface_vector(bounds, 2)
            _require(lo <= hi and row.get('quality') == 'authored', 'Surface heights must be ordered authored metre ranges')
    source_slots = surface.get('source_slots')
    _require(type(source_slots) is list and len(source_slots) == frame_count and 1 <= frame_count <= 1800,
             'Complete original surface source slots required')
    references = {row['frame']: {'reference_index': i, 'evidence_id': row['evidence_id'], 'body_time_s': row['time_s']}
                  for i, row in enumerate(report['references'])}
    status_counts, display = Counter(), []
    for frame, slot in enumerate(source_slots):
        check()
        _require(type(slot) is dict and slot.get('source_frame') == frame and slot.get('body_reference') == references.get(frame)
                 and slot.get('source_status') == ('decoded' if frame < decoded else 'metadata_only_not_decoded'),
                 'Surface source frame/body reference join differs')
        channels = slot.get('channels')
        _require(type(channels) is list and len(channels) == len(declarations), 'Complete surface channel slots required')
        shown = []
        for channel, declaration in zip(channels, declarations):
            _require(type(channel) is dict and channel.get('channel_id') == declaration['id']
                     and channel.get('entity_ref') == declaration['entity_ref'], 'Surface channel/entity order differs')
            entity = declaration['entity_ref']
            observed = observations.get((entity['document_sha256'], entity['object_id'], frame))
            _require(all(channel.get(a) == (observed.get(b) if observed is not None else None) for a, b in
                     (('source_observation_id', 'id'), ('source_time_s', 'time_s'), ('source_partition', 'partition'))),
                     'Surface source observation identity, partition or clock differs')
            bounds = channel.get('world_envelope_m')
            if bounds is not None:
                _require(type(bounds) is dict and bounds.get('unit') == 'm' and bounds.get('kind') == 'outer_axis_aligned_enclosure'
                         and channel.get('envelope_status') == 'conditional_surface_envelope', 'Saved world enclosure units or status differ')
                lo, hi = _surface_vector(bounds.get('minimum')), _surface_vector(bounds.get('maximum'))
                _require(all(a <= b for a, b in zip(lo, hi)), 'Saved world enclosure bounds are reversed')
            else:
                _require(channel.get('envelope_status') == 'unavailable', 'Available envelope lacks saved world bounds')
            if channel.get('status') == 'conditional_interval_ready':
                _require(bounds is not None and channel.get('reason') is None and observed is not None
                         and channel['source_partition'] == 'construction' and frame in references
                         and channel['source_time_s'] == references[frame]['body_time_s']
                         and references[frame]['evidence_id'] in observed.get('evidence_ids', [])
                         and inputs.get('binding_unavailable_reason') is None and binding_hash is not None,
                         'Excluded or unbound source cannot supply a surface fitting target')
            else:
                _require(channel.get('status') == 'unavailable' and type(channel.get('reason')) is str
                         and 0 < len(channel['reason']) <= 200, 'Unavailable surface slot needs its saved reason')
            status_counts[channel.get('reason') or channel['status']] += 1
            annotation = annotation_index.get((frame, declaration['id']))
            pixel, uncertainty = None, None
            if annotation is not None and annotation.get('status') == 'annotated':
                pixel, uncertainty = _surface_vector(annotation.get('pixel'), 2), _surface_number(annotation.get('uncertainty_px'), nonnegative=True)
                _require(uncertainty <= 64 and observed is not None and annotation.get('observation_id') == observed['id']
                         and math.isclose(annotation.get('time_s'), observed['time_s'], rel_tol=0, abs_tol=1e-9),
                         'Supplied pixel range source join differs')
            shown.append({'channel': declaration['id'], 'point': channel.get('body_point'), 'status': channel['status'],
                'reason': channel['reason'], 'pixel': pixel, 'uncertainty_px': uncertainty,
                'height_interval_m': surface_index.get(declaration['surface_id'], {}).get('height_interval_m'),
                'world_bounds_m': None if bounds is None else [bounds['minimum'], bounds['maximum']],
                'before_m': None, 'after_m': None})
        display.append({'frame': frame, 'status': slot['source_status'], 'reference_index':
                        references[frame]['reference_index'] if frame in references else None, 'channels': shown})
    _require(inputs.get('counts') == {'retained_source_frames': frame_count, 'retained_channel_slots': frame_count * len(declarations),
             'original_body_references': len(references), 'status_counts': dict(status_counts)}, 'Surface complete source counts differ')
    start = min(references)
    _require(inputs.get('original_start_frame') == start and inputs.get('original_start_statuses') ==
             [{k: v for k, v in row.items() if k != 'world_envelope_m'} for row in source_slots[start]['channels']],
             'Surface original-start source statuses differ')
    saved_references = surface.get('references')
    _require(type(saved_references) is list and len(saved_references) == len(references), 'Surface reference count differs')
    all_losses, all_distances, available = [[], []], [[], []], 0
    for index, (reference, original) in enumerate(zip(saved_references, report['references'])):
        check(); frame = original['frame']
        _require(type(reference) is dict and reference.get('reference_index') == index and reference.get('source_frame') == frame
                 and reference.get('evidence_id') == original['evidence_id'] and reference.get('time_s') == original['time_s'],
                 'Surface reference identity or original clock differs')
        for side, name in enumerate(('before', 'after')):
            losses, distances = _surface_result(reference.get(name), source_slots[frame], hashes['inputs/human.json'])
            all_losses[side].extend(losses); all_distances[side].extend(distances)
            if side == 0: available += bool(losses)
            for target, channel in zip(display[frame]['channels'], reference[name]['channels']):
                target[name + '_m'] = channel['distance_to_outer_envelope_m']
    expected = {'source_frame_slots': frame_count, 'channel_slots': frame_count * len(declarations),
        'available_reference_terms': available, 'unavailable_reference_terms': len(references) - available,
        'evaluated_channels_before': len(all_losses[0]), 'evaluated_channels_after': len(all_losses[1])}
    summary = surface.get('summary')
    _require(type(summary) is dict, 'Missing surface summary')
    for key, value in expected.items():
        _require(type(summary.get(key)) is int and summary[key] == value, 'Surface summary counts differ')
    for side, name in enumerate(('before', 'after')):
        loss = sum(all_losses[side]) if all_losses[side] else None
        for key, value in (('sum_objective_' + name + '_m2', loss), ('weighted_sum_' + name, None if loss is None else weight * loss),
                           ('maximum_channel_outside_' + name + '_m', max(all_distances[side]) if all_distances[side] else None)):
            _surface_equal(summary.get(key), value, 'Surface summary arithmetic differs')
    gate = report['gates'].get('surface_interval_nonregression')
    _require(type(gate) is bool, 'Saved surface gate must be explicitly true or false')
    if gate:
        for reference in saved_references:
            before, after = reference['before'], reference['after']
            _require(before['counts'] == after['counts'], 'Saved surface gate contradicts channel counts')
            for a, b in zip(before['channels'], after['channels']):
                _require(all(a.get(k) == b.get(k) for k in ('channel_id', 'status', 'reason')),
                         'Saved surface gate contradicts channel availability')
                if a['status'] == 'conditional_interval_residual':
                    _require(b['distance_to_outer_envelope_m'] <= a['distance_to_outer_envelope_m'],
                             'Saved surface gate contradicts channel deviation')
    journal = surface.get('trial_journal')
    _require(type(journal) is list and len(journal) <= 20000, 'Surface trial journal exceeds its bound')
    journal_bytes = 0
    for index, row in enumerate(journal):
        check()
        size = len(json.dumps(row, allow_nan=False, separators=(',', ':')).encode())
        _require(type(row) is dict and row.get('sequence') == index and row.get('kind') in {'qp', 'backtrack'}
                 and size <= 16 << 10 and type(row.get('reference_index')) is int
                 and 0 < row['reference_index'] < len(saved_references)
                 and row.get('source_frame') == saved_references[row['reference_index']]['source_frame'],
                 'Surface journal record exceeds its bound or kind')
        journal_bytes += size
    _require(type(surface.get('journal_bytes')) is int and surface['journal_bytes'] == journal_bytes
             and journal_bytes <= work['maximum_journal_bytes'],
             'Surface journal byte count exceeds its bound')
    return {'profile': SURFACE_PROFILE, 'weight': weight, 'summary': summary, 'source_status_counts': dict(status_counts),
        'source_slots': display, 'work_limits': work, 'evaluations': surface['evaluations'], 'journal_records': len(journal),
        'journal_bytes': surface['journal_bytes'], 'config_sha256': plan['surface_constraints']['config_sha256'],
        'body_binding': inputs['binding'], 'world_frame': inputs.get('world_frame'), 'contact': 'unknown',
        'physical_validation': 'unverified', 'saved_geometry_recomputed': False}


def _model(documents, hashes, check):
    report, candidate = documents["report.json"], documents["candidate-human.json"]
    source, evidence = documents["inputs/human.json"], documents["inputs/evidence-derived.json"]
    method = report.get("profile")
    _require(type(report.get("schema_version")) is int and report['schema_version'] == 1 and method in METHODS,
             "Unsupported reference proposal profile")
    _require(report.get("status") in {"rejected_construction_proposal", "unadmitted_construction_proposal", "partial_budget_exhausted"}
             and report.get("promoted") is False and report.get("physical_validation") == "unverified",
             "Only explicitly unpromoted, physically unverified proposals are supported")
    _require(report.get("candidate_sha256") == _canonical(candidate), "Canonical candidate identity differs")
    for key, name in (("input_human_sha256", "inputs/human.json"), ("input_evidence_sha256", "inputs/evidence-derived.json")):
        _require(report.get(key) == hashes[name], "Proposal input byte identity differs")
    marker = candidate.get("whole_body_reference_refinement")
    _require(isinstance(marker, dict) and marker.get("method") == method and marker.get("status") == report["status"]
             and marker.get("promoted") is False and all(marker.get(k) == report[k]
             for k in ("input_human_sha256", "input_evidence_sha256")), "Candidate provenance differs from its report")
    source_id = report.get("source_sha256")
    _require(isinstance(source_id, str) and _SHA.fullmatch(source_id)
             and all(value.get("source_sha256") == source_id for value in (source, candidate, evidence)),
             "Source identifiers disagree across selected evidence")
    _require({k: v for k, v in candidate.items() if k not in {"reference", "whole_body_reference_refinement"}}
             == {k: v for k, v in source.items() if k != "reference"}, "Nonreference body, root, shape, floor or clocks changed")
    contract = report.get("contract")
    _require(isinstance(contract, dict) and contract.get("method") == method
             and contract.get("floor_penetration_limit_model_units") == LIMIT, "The fixed 0.025 floor limit changed")
    tolerance = _number(contract.get("numerical_tolerance"), low=0, high=.000001)
    _require(tolerance is not None, "Missing numerical tolerance")
    gates = report.get("gates")
    _require(isinstance(gates, dict) and len(gates) <= 32 and all(v is None or type(v) is bool for v in gates.values()),
             "Invalid saved gate outcomes")
    for name in gates: _text(name, 100)
    _require(not (report["status"] == "unadmitted_construction_proposal" and any(v is False for v in gates.values())),
             "Proposal status contradicts its rejected gates")
    failures = report.get('failures')
    _require(failures is None or isinstance(failures, list) and len(failures) <= 32 and all(isinstance(name, str) for name in failures)
             and set(failures) == {name for name, value in gates.items() if value is False}, 'Saved failed gate names differ')
    _require(report['status'] != 'rejected_construction_proposal' or any(v is False for v in gates.values()),
             'Rejected proposal is missing its failed gate')
    references, old_refs, new_refs = report.get("references"), source.get("reference"), candidate.get("reference")
    _require(isinstance(references, list) and isinstance(old_refs, list) and isinstance(new_refs, list)
             and 1 <= len(references) == len(old_refs) == len(new_refs) <= 900, "Reference counts differ or exceed bounds")
    times = [r.get("time_s") for r in old_refs]
    _require(all(_number(t, low=0) is not None for t in times) and all(a < b for a, b in zip(times, times[1:])),
             "Reference clocks must increase")
    _require([r.get("time_s") for r in new_refs] == times == [r.get("time_s") for r in references],
             "Source/candidate/report clocks differ")
    origins = source.get("reference_evidence")
    _require(isinstance(origins, list) and len(origins) == len(references), "Missing source reference identities")
    slots, frames = report.get("retained_pose_slots"), evidence.get("frames")
    _require(isinstance(slots, list) and isinstance(frames, list) and 1 <= len(slots) == len(frames) <= 900,
             "Retained source-slot counts differ")
    _require(slots == [{k: row.get(k) for k in ("id", "frame", "time_s", "decoder_timestamp_s", "status")} for row in frames],
             "Retained source slots differ from the exact input evidence")
    _require(len({row.get("id") for row in slots}) == len(slots), "Duplicate retained source identity")
    by_id, target_info = {}, {}
    frames_by_id = {row['id']: row for row in frames}
    for index, ref in enumerate(references):
        check()
        identity = _text(ref.get("evidence_id"))
        _require(identity == origins[index].get("evidence_id") and origins[index].get('time_s') == times[index]
                 and identity not in by_id, "Reference evidence identities differ")
        _text(ref.get('status'), 100)
        points = ref.get("points")
        _require(isinstance(points, list) and len(points) <= 32, "Endpoint details exceed their bound")
        names = set()
        for point in points:
            joint = _text(point.get("joint"), 100); _require(joint not in names, "Duplicate endpoint"); names.add(joint)
            _text(point.get("status"), 100)
            for field in ("source_confidence", "weight"): _number(point.get(field), low=0, high=1)
            for field in ("error_before_px", "error_after_px"): _number(point.get(field), low=0, high=1_000_000)
            if point.get("weight") == 0:
                _require(point.get("error_before_px") is None and point.get("error_after_px") is None,
                         "Missing or low-confidence endpoint errors must stay unknown")
        by_id[identity] = (index, ref)
        _require(identity in frames_by_id, 'A target reference has no source frame')
        target_info[identity] = _target_info(frames_by_id[identity], ref)
    support_names, primitive_names = report.get("support_names"), report.get("support_primitive_names")
    _require(isinstance(support_names, list) and isinstance(primitive_names, list)
             and 1 <= len(support_names) == len(primitive_names) <= 64 and len(set(support_names)) == len(support_names),
             "Invalid bounded support identities")
    for name in support_names + primitive_names: _text(name, 100)
    checks = report.get("geometry_checks")
    _require(isinstance(checks, list) and len(checks) <= 2701, "Geometry sample count exceeds its bound")
    source_checks, grid, unique = {}, [], {}
    for row in checks:
        check()
        kind, stamp = row.get("kind"), _number(row.get("time_s"), low=0)
        _require(kind in {"source_pose", "control_grid"} and stamp is not None, "Invalid saved sample clock or kind")
        for side in ("before", "after"):
            gaps = row.get(side + "_floor_support_gaps")
            _require(gaps is None or isinstance(gaps, list) and len(gaps) == len(support_names), "Support gap count differs")
            if gaps is not None:
                for value in gaps: _number(value)
            peak = _number(row.get(side + "_max_penetration_model_units"), low=0)
            if gaps is not None and all(v is not None for v in gaps) and peak is not None:
                _require(math.isclose(peak, max(0., -min(gaps)), abs_tol=1e-10), "Reported penetration differs from saved gaps")
        _number(row.get("maximum_depth_change_model_units"), low=0)
        _require(row.get("no_worse_floor") is None or type(row["no_worse_floor"]) is bool, "Invalid floor comparison flag")
        pair = [row.get(side + "_floor_support_gaps") for side in ("before", "after")]
        if stamp in unique:
            _require(unique[stamp] == pair, "Coincident source/grid support records disagree")
        unique[stamp] = pair
        if kind == "source_pose":
            index = row.get("source_reference_index")
            _require(type(index) is int and 0 <= index < len(references) and index not in source_checks
                     and stamp == times[index], "Source geometry sample identity differs")
            source_checks[index] = row
        else:
            _require(row.get("source_reference_index") is None, "Grid samples cannot claim a source-reference index")
            _require(times[0] <= stamp <= times[-1], 'Control-grid clock is outside the source reference span')
            grid.append(row)
    _require(len(grid) <= 1801 and all(a['time_s'] < b['time_s'] for a, b in zip(grid, grid[1:])), "Grid clocks must increase")

    def sample(slot, geometry, ref=None):
        return {"time_s": slot.get("time_s"), "decoder_timestamp_s": slot.get("decoder_timestamp_s"),
                "frame": slot.get("frame"), "evidence_id": slot.get("id"), "source_status": slot.get("status"),
                "geometry": geometry, "points": ref.get("points") if ref else None,
                "reference_status": ref.get("status") if ref else "No body reference for this retained source slot",
                **(target_info[ref['evidence_id']] if ref else {'target_origin': None, 'target_landmark_source': None,
                    'target_label': 'Unavailable — no saved source endpoint targets for this sample'})}

    source_rows = []
    for slot in slots:
        _text(slot.get("id")); _text(slot.get("status"), 100)
        _count(slot.get("frame")); _number(slot.get("time_s"), low=0); _number(slot.get("decoder_timestamp_s"), low=0)
        found = by_id.get(slot["id"])
        if found:
            index, ref = found
            _require(slot['frame'] == ref.get('frame') and slot['time_s'] == ref['time_s'], "Retained slot/reference identity differs")
            source_rows.append(sample(slot, source_checks.get(index), ref))
        else: source_rows.append(sample(slot, None))
    slot_times = [row['time_s'] for row in slots]
    _require(all(t is not None for t in slot_times) and all(a < b for a, b in zip(slot_times, slot_times[1:])),
             'Retained nominal clocks must increase; missing decoder clocks remain unknown')
    _require(set(by_id) <= {s['id'] for s in slots}, "A body reference has no retained source slot")
    grid_rows = []
    for row in grid:
        index = bisect_left(times, row['time_s'])
        neighbors = ([index] if index < len(times) and times[index] == row['time_s'] else
                     sorted({max(0, index - 1), min(len(times) - 1, index)}))
        value = sample({"time_s": row['time_s'], "status": "Authored control-grid sample"}, row)
        value['bracketing_references'] = [{"evidence_id": references[i]['evidence_id'], "time_s": times[i]} for i in neighbors]
        value['reference_status'] = "No source-image endpoint target is authored for control-grid samples"
        grid_rows.append(value)
    counts = report.get("source_counts", {})
    _require(isinstance(counts, dict), "Invalid source counts")
    for key, actual in (("retained_pose_slots", len(slots)), ("body_reference_knots", len(references))):
        _require(counts.get(key) == actual, "Saved source counts differ")
    for key in ("declared_decoded_frames", "declared_frame_count"): _count(counts.get(key))
    _require(counts.get('declared_decoded_frames') == evidence.get('decoded_frames')
             and counts.get('declared_frame_count') == evidence.get('frame_count'), "Declared source frame counts differ")
    floor = report.get('floor_summary', {})
    _require(isinstance(floor, dict), "Invalid floor summary")
    for key, actual in (('checks', len(checks)), ('source_pose_checks', len(source_checks)), ('control_grid_checks', len(grid))):
        _require(floor.get(key) is None or _count(floor[key]) == actual, "Saved geometry counts differ")
    displayed_counts = {}
    for kind, rows in (('source_pose', list(source_checks.values())), ('control_grid', grid)):
        value = {'checks': len(rows)}
        for side in ('before', 'after'):
            values = [row.get(side + '_max_penetration_model_units') for row in rows]
            value[side + '_failures'] = sum(v is not None and v > LIMIT for v in values)
            value[side + '_unknown'] = sum(v is None for v in values)
        displayed_counts[kind] = value
    for side in ('before', 'after'):
        values = [row.get(side + '_max_penetration_model_units') for row in checks]
        reported = _count(floor.get(side + '_failures'))
        maximum = _number(floor.get(side + '_maximum_penetration_model_units'), low=0)
        if values and all(v is not None for v in values) and reported is not None:
            _require(reported == sum(v > LIMIT for v in values), "Saved floor failure total differs")
        if values and all(v is not None for v in values) and maximum is not None:
            _require(math.isclose(maximum, max(values), abs_tol=1e-10), 'Saved maximum penetration differs')
    _number(floor.get('maximum_penetration_reduction_model_units'))
    _count(floor.get('samples_with_reduced_penetration'))
    after = [row.get('after_max_penetration_model_units') for row in checks]
    gate = gates.get('floor_feasibility')
    if gate is True:
        _require(not any(value is not None and value > LIMIT for value in after), 'Saved floor gate contradicts known penetration')
    if (gate is False and after and all(value is not None for value in after) and len(source_checks) == len(references)
            and grid and floor.get('checks') == len(checks) and floor.get('control_grid_checks') == len(grid)):
        _require(any(value > LIMIT for value in after), 'Saved floor gate contradicts complete saved penetration values')
    objective = report.get('objective')
    if method.endswith('floor-first-reference-v1'):
        _require(isinstance(objective, dict) and objective.get('profile') == 'floor_first_v1'
                 and objective.get('units') == 'model_units_squared' and objective.get('proposal_target_gap_model_units') == 0.,
                 "Missing supported floor-first objective")
        _require(_count(objective.get('unique_samples')) == len(unique), "Floor-first unique time count differs")
        for side, name in enumerate(('primary_before', 'primary_after')):
            value = _number(objective.get(name), low=0, high=1_000_000)
            if all(pair[side] is not None and all(v is not None for v in pair[side]) for pair in unique.values()):
                expected = .5 * sum(min(0., gap + LIMIT)**2 for pair in unique.values() for gap in pair[side])
                _require(value is not None and math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12),
                         "Saved squared excess objective differs from unique-time gaps")
        counter = objective.get('counters')
        _require(isinstance(counter, dict) and set(counter) == _COUNTERS, "Missing supported floor-first counters")
        for key, value in counter.items():
            _text(key, 100)
            if key == 'qp_status_counts':
                _require(isinstance(value, dict) and set(value) == _QP_STATUSES, "Invalid QP status counts")
                for name, count in value.items(): _text(name, 100); _require(_count(count) is not None, "Missing QP count")
            else: _require(_count(value) is not None, "Missing objective counter")
        _require(counter.get('floor_qp_calls', -1) + counter.get('image_qp_calls', -1) == report.get('qp_calls')
                 == sum(counter.get('qp_status_counts', {}).values()), "QP counters differ")
        _require(counter.get('floor_descent_accepts', -1) + counter.get('image_tiebreak_accepts', -1) == report.get('accepted_updates')
                 and counter.get('backtrack_trials') == report.get('accepted_updates', -1) + counter.get('rejected_backtracks', -1),
                 "Completed-trial counters differ")
    elif method == SURFACE_METHOD:
        _require(type(objective) is dict and objective.get('profile') == 'surface_interval_v1'
                 and objective.get('interval_units') == 'supplied_world_metre_squared'
                 and objective.get('weight_status') == 'authored' and objective.get('grid_interval_targets') == 0
                 and objective.get('weight') == report.get('surface_intervals', {}).get('weight'),
                 'Missing supported source-knot surface objective')
        for key in ('primary', 'acceptance'): _text(objective.get(key), 512)
    else: _require(objective is None, "Image-first profile cannot silently claim a floor-first objective")
    surface = _surface_model(report, candidate, source, documents, hashes, check) if method == SURFACE_METHOD else None
    if surface is None:
        _require('surface_intervals' not in report and 'surface_intervals' not in marker,
                 'Legacy profile cannot silently claim surface constraints')
    image = report.get('image_summary', {})
    _require(isinstance(image, dict) and image.get('grid_image_targets') in (None, 0), "Grid image targets must remain unavailable")
    for name in ('mean_error_before_px', 'mean_error_after_px'): _number(image.get(name), low=0, high=1_000_000)
    _count(image.get('confident_source_endpoint_pairs'))
    points = [p for ref in references for p in ref['points']]
    if all(p.get('weight') is not None for p in points):
        pairs = [p for p in points if p['weight'] > 0]
        _require(image.get('confident_source_endpoint_pairs') in (None, len(pairs)), 'Confident endpoint counts differ')
        for side in ('before', 'after'):
            errors = [p.get(f'error_{side}_px') for p in pairs]
            average = image.get(f'mean_error_{side}_px')
            if errors and all(e is not None for e in errors) and average is not None:
                _require(math.isclose(average, sum(errors) / len(errors), rel_tol=1e-9, abs_tol=1e-10), 'Saved mean endpoint error differs')
    return {'method': method, 'status': report['status'], 'promoted': False, 'physical_validation': 'unverified',
            'geometry_admission': report.get('geometry_admission'), 'gates': gates, 'failures': report.get('failures'),
            'source_sha256': source_id, 'candidate_canonical_sha256': report['candidate_sha256'], 'floor_limit': LIMIT,
            'source_counts': counts, 'source_status_counts': dict(Counter(slot['status'] for slot in slots)),
            'saved_floor_summary': floor, 'sample_counts': displayed_counts, 'unique_sample_times': len(unique),
            'source_slots_without_body_reference': len(slots) - len(references), 'image_summary': image, 'objective': objective,
            'source_references_without_geometry': len(references) - len(source_checks),
            'support_names': support_names, 'support_primitive_names': primitive_names,
            'support_labels': [_support_label(name, primitive) for name, primitive in zip(support_names, primitive_names)],
            'source': source_rows, 'grid': grid_rows, **({'surface_intervals': surface} if surface is not None else {}),
            'scope': 'Saved source image estimates and authored-floor sample checks, not measured truth, physical poses, contact, anatomy or task/policy proof.'}


_SCRIPT = r"""'use strict';
const data=JSON.parse(document.getElementById('data').textContent),mode=document.getElementById('mode'),select=document.getElementById('sample');
const el=id=>document.getElementById(id),num=v=>v===null||v===undefined?'Unknown':Number(v).toPrecision(6),ns='http://www.w3.org/2000/svg';
let rows=[];
function cell(tr,value,head=false){const node=document.createElement(head?'th':'td');if(head)node.scope='row';node.textContent=value;tr.append(node);}
function textRow(table,values){const tr=document.createElement('tr');values.forEach((v,i)=>cell(tr,v,i===0));table.append(tr);return tr;}
function chart(){const svg=el('chart'),layer=el('lines');layer.replaceChildren();const values=rows.flatMap(r=>['before','after'].map(k=>r.geometry?.[k+'_max_penetration_model_units'])).filter(v=>v!==null&&v!==undefined);
 const ymax=Math.max(.03,...values)*1.1,first=rows[0]?.time_s??0,last=rows.at(-1)?.time_s??first;
 const x=t=>50+850*(t-first)/Math.max(.001,last-first),y=v=>250-210*v/ymax;
 svg.dataset.ymax=ymax;svg.dataset.first=first;svg.dataset.last=last;
 function line(d,klass){const p=document.createElementNS(ns,'path');p.setAttribute('d',d);p.setAttribute('class',klass);layer.append(p);}
 line(`M50 ${y(data.floor_limit)}H900`,'limit');
 for(const side of ['before','after']){let path='',open=false;for(const r of rows){const v=r.geometry?.[side+'_max_penetration_model_units'];if(v===null||v===undefined||r.time_s===null){open=false;continue;}path+=(open?' L':' M')+x(r.time_s)+' '+y(v);open=true;}line(path,side);}
 el('ymax').textContent=num(ymax);el('tfirst').textContent=num(first)+' s';el('tlast').textContent=num(last)+' s';
}
function detail(){const row=rows[Number(select.value)],geometry=row?.geometry;el('gaps').replaceChildren();el('points').replaceChildren();el('target-metadata').replaceChildren();el('target-details').hidden=true;el('target-label').textContent='Unavailable — no saved source endpoint targets for this sample';
 if(!row){el('identity').textContent='No saved samples available.';return;}
 el('identity').textContent=`${mode.value==='source'?'Retained source slot':'Control-grid check'} ${Number(select.value)+1}/${rows.length} · nominal time ${num(row.time_s)} s · decoder time ${num(row.decoder_timestamp_s)} s · frame ${row.frame??'unavailable'}`;
 el('source-id').textContent=row.evidence_id||'No direct source observation ID for this interpolated control sample';
 el('context').textContent=[row.source_status,row.reference_status,...(row.bracketing_references||[]).map(v=>`Body reference ${v.evidence_id} at ${num(v.time_s)} s`)].join(' · ');
 el('penetration').textContent=`Input body: ${num(geometry?.before_max_penetration_model_units)}; Proposed body: ${num(geometry?.after_max_penetration_model_units)} model units. Fixed limit: 0.025. Unknown is not zero.`;
 el('target-label').textContent=row.target_label;
 if(row.target_origin){el('target-details').hidden=false;for(const [label,value]of[['Method',row.target_origin.method],['Preserved whole-frame observation SHA256',row.target_origin.original_observation_sha256],['Selected crop proposal SHA256',row.target_origin.proposal_sha256],['Saved decision SHA256',row.target_origin.decision_sha256],['Landmark JSON path',row.target_landmark_source],['Anatomical identity','Not independently verified']]){const term=document.createElement('dt'),detail=document.createElement('dd');term.textContent=label;detail.textContent=value;el('target-metadata').append(term,detail);}}
 let shown=0;for(let i=0;i<data.support_names.length;i++){const a=geometry?.before_floor_support_gaps?.[i],b=geometry?.after_floor_support_gaps?.[i],known=a!==null&&a!==undefined&&b!==null&&b!==undefined;
 if(el('affected').checked&&known&&a===b&&a>=0&&b>=0)continue;
 const tr=textRow(el('gaps'),[data.support_labels[i],'',num(a),num(b),!known?'Unknown':b>a?'Gap increased':b<a?'Gap decreased':'Unchanged']);
 const details=document.createElement('details'),label=document.createElement('summary'),ids=document.createElement('span');label.textContent='Exact saved IDs';ids.textContent=data.support_names[i]+' · '+data.support_primitive_names[i];details.append(label,ids);tr.children[1].append(details);shown++;}
 if(!shown)textRow(el('gaps'),['No changed or below-floor support gaps in this sample.','','','','']);
 if(mode.value==='grid'||!row.points)textRow(el('points'),['Unavailable — no source endpoint targets for this sample.','','','','']);
 else for(const p of row.points)textRow(el('points'),[p.joint,p.status,num(p.source_confidence),num(p.error_before_px),num(p.error_after_px)]);
 const svg=el('chart'),ymax=Number(svg.dataset.ymax),first=Number(svg.dataset.first),last=Number(svg.dataset.last);for(const side of ['before','after']){const dot=el(side+'-dot'),v=geometry?.[side+'_max_penetration_model_units'];dot.style.display=v===null||v===undefined?'none':'';if(v!==null&&v!==undefined){dot.setAttribute('cx',50+850*(row.time_s-first)/Math.max(.001,last-first));dot.setAttribute('cy',250-210*v/ymax);}}
 el('previous').disabled=Number(select.value)<=0;el('next').disabled=Number(select.value)>=rows.length-1;
}
function changeMode(){rows=data[mode.value];select.replaceChildren();rows.forEach((row,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`${i+1} · ${num(row.time_s)} s${row.frame===null||row.frame===undefined?'':' · frame '+row.frame}`;select.append(option);});select.value='0';chart();detail();}
mode.addEventListener('change',changeMode);select.addEventListener('change',detail);el('affected').addEventListener('change',detail);
for(const [id,d]of[['previous',-1],['next',1]])el(id).addEventListener('click',()=>{select.value=Math.max(0,Math.min(rows.length-1,Number(select.value)+d));detail();});changeMode();
if(data.surface_intervals){const surface=data.surface_intervals,selector=el('interval-sample');let shownFrame=null;
 const vector=v=>v===null||v===undefined?'Unknown':v.map(num).join(', ');
 function intervalDetail(){const index=Number(selector.value);if(index===shownFrame)return;shownFrame=index;
  const row=surface.source_slots[index],table=el('interval-channels');table.replaceChildren();
  el('interval-context').textContent=`Original source frame ${row.frame} · ${row.status} · ${row.reference_index===null?'No body reference; no fitted term':'Body reference '+row.reference_index}`;
  for(const c of row.channels){const pixels=c.pixel===null?'Unknown':c.pixel.map((v,i)=>`${i?'y':'x'} [${num(v-c.uncertainty_px)}, ${num(v+c.uncertainty_px)}]`).join('; ');
   textRow(table,[c.channel+(c.point?' → '+c.point:''),c.reason||'Conditional range available',pixels,
    c.height_interval_m===null?'Unknown':'['+vector(c.height_interval_m)+']',
    c.world_bounds_m===null?'Unknown':'min ['+vector(c.world_bounds_m[0])+']; max ['+vector(c.world_bounds_m[1])+']',num(c.before_m),num(c.after_m)]);}
  if(!row.channels.length)textRow(table,['No supplied channels','','','','','','']);
 }
 surface.source_slots.forEach((row,i)=>{const option=document.createElement('option');option.value=i;option.textContent=`Frame ${row.frame} · ${row.status}${row.reference_index===null?' · no body reference':''}`;selector.append(option);});
 selector.value='0';selector.addEventListener('change',intervalDetail);intervalDetail();
}
"""


def _surface_html(surface, esc, fmt):
    if surface is None:
        return ''
    summary = surface['summary']
    return f'''<section><h2>Supplied surface ranges</h2>
<p class="notice">These are conditional ranges from supplied camera, pixel, surface-height and body-alignment assumptions. <strong>Zero outside-box error does not verify contact or an attainable source point.</strong> Calibration, body scale, anatomy, support and physical accuracy remain unverified. No midpoint is selected as a target.</p>
<p>Original source frames: {summary['source_frame_slots']}; channel slots: {summary['channel_slots']}. Body-reference terms available: {summary['available_reference_terms']}; unavailable: {summary['unavailable_reference_terms']}. Evaluated channels input → proposed: {summary['evaluated_channels_before']} → {summary['evaluated_channels_after']}. Missing and excluded slots remain unavailable, not zero-error samples.</p>
<p>Maximum distance outside supplied world range: {fmt(summary['maximum_channel_outside_before_m'])} → {fmt(summary['maximum_channel_outside_after_m'])} <strong>metres</strong>. Sum of half-squared distances: {fmt(summary['sum_objective_before_m2'])} → {fmt(summary['sum_objective_after_m2'])} <strong>m²</strong>.</p>
<p>Authored weight: {fmt(surface['weight'])} existing-model objective units per m². Weighted term: {fmt(summary['weighted_sum_before'])} → {fmt(summary['weighted_sum_after'])}. This is an authored fitting tradeoff, <strong>not calibrated physical energy</strong>. It is separate from image errors in pixels and floor gaps in model units; the floor limit stays 0.025 model units.</p>
<label>Surface source slot <select id="interval-sample"></select></label><p id="interval-context" aria-live="polite"></p>
<div class="scroll"><table><thead><tr><th>Channel / assumed body point</th><th>Saved availability</th><th>Supplied pixel range (px)</th><th>Supplied height range (m)</th><th>Saved outer world bounds (m)</th><th>Input outside (m)</th><th>Proposed outside (m)</th></tr></thead><tbody id="interval-channels"></tbody></table></div>
<p>The pixel and height ranges above come from exact copied inputs. World bounds and positions are saved producer results; this report checks joins and residual arithmetic but does not recompute geometry or validate the camera. All original slots remain selectable, including slots without body references. No control-grid interval targets are created.</p>
<details><summary>Fixed assumptions and saved work</summary><pre>{esc(json.dumps({'world_frame': surface['world_frame'], 'body_binding': surface['body_binding'], 'source_status_counts': surface['source_status_counts']}, indent=2))}</pre>
<p>Saved interval evaluations: {surface['evaluations']} / {surface['work_limits']['maximum_evaluations']}. Saved trial records: {surface['journal_records']}; journal bytes: {surface['journal_bytes']} / {surface['work_limits']['maximum_journal_bytes']}. Interrupted and rejected attempts remain in <a href="artifacts/proposal-report.json.txt">the exact proposal report</a>. Counts and saved outcomes are not a replay or a new optimization run.</p></details></section>'''


def _render(summary, palette, members=None):
    esc = lambda value: html.escape(str(value), quote=True)
    fmt = lambda value: "Unknown" if value is None else esc(value)
    labels = {'rejected_construction_proposal': 'Rejected proposal — not promoted',
              'unadmitted_construction_proposal': 'Unadmitted proposal — not promoted',
              'partial_budget_exhausted': 'Incomplete proposal — not promoted'}
    counts = ''.join(f'<tr><th scope="row">{label}</th><td>{row["checks"]}</td><td>{row["before_failures"]}</td><td>{row["after_failures"]}</td><td>{row["before_unknown"]} / {row["after_unknown"]}</td></tr>'
                     for key, label in [('source_pose', 'Source-reference checks'), ('control_grid', 'Control-grid checks')]
                     for row in [summary['sample_counts'][key]])
    gates = ''.join(f'<tr><th scope="row">{esc(name)}</th><td>{"Unknown" if value is None else "Reported satisfied" if value else "Reported failure"}</td></tr>'
                    for name, value in summary['gates'].items())
    objective = summary['objective']
    objective_html = ('<p>Image-first proposal. No floor-first objective was reported.</p>' if objective is None else
        '<p>The existing image and regularization objective includes an authored surface-range term at original source knots only. No surface targets are assigned to the control grid. The unchanged floor and source-image gates still apply.</p>'
        if objective.get('profile') == 'surface_interval_v1' else
        f'<p>Saved squared <strong>excess above 0.025</strong>: {fmt(objective.get("primary_before"))} → {fmt(objective.get("primary_after"))} model units² over {fmt(objective.get("unique_samples"))} unique sampled times. Lower is not a pass; zero would mean no saved excess above the limit, not zero penetration or contact.</p><details><summary>Saved floor-first counters</summary><pre>{esc(json.dumps(objective.get("counters"),indent=2))}</pre><p>Backtrack counts cover completed trials only. Interrupted trials may still add geometry calls. No per-update ledger was saved; these aggregates do not establish each optimization step.</p></details>')
    data = json.dumps(summary, separators=(',', ':'), allow_nan=False).replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
    script_hash = base64.b64encode(sha256(_SCRIPT.encode()).digest()).decode()
    members = MEMBERS if members is None else members
    links = ''.join(f'<li><a href="artifacts/{name}">{esc(source)}</a></li>' for source, name in members.items())
    if 'surface_input_validator' in summary:
        links += '<li><a href="artifacts/source_surface_fit_inputs.py.txt">Current surface-input validation helper</a> (inspection code, not a study input)</li>'
    css = """body{margin:0;background:var(--weft-canvas);color:var(--weft-ink);font:16px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:30px 20px}h1{font:700 2.4rem/1.1 Georgia,serif}section{background:var(--weft-surface);border:1px solid var(--weft-border);border-radius:14px;padding:22px;margin:20px 0}a{color:var(--weft-primary)}.notice{border-left:4px solid var(--weft-primary);padding:10px 16px}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:8px;border-bottom:1px solid var(--weft-border)}.scroll{overflow-x:auto}code,pre,.identity{overflow-wrap:anywhere}pre{white-space:pre-wrap}label{display:inline-block;margin:8px 14px 8px 0}select,button{font:inherit;padding:7px;max-width:100%}button:focus-visible,select:focus-visible{outline:3px solid var(--weft-primary)}svg{width:100%;height:auto}.before{fill:none;stroke:var(--weft-ink);stroke-width:2}.after{fill:none;stroke:var(--weft-primary);stroke-width:2.5}.limit{fill:none;stroke:var(--weft-muted);stroke-dasharray:7 5;stroke-width:1.5}svg text{fill:var(--weft-ink);font-size:13px}.dot-before{fill:var(--weft-ink)}.dot-after{fill:var(--weft-primary)}small{color:var(--weft-muted)}@media(max-width:600px){main{padding:16px 10px}section{padding:14px}td,th{padding:6px;font-size:.85rem}}"""
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; base-uri 'none'; form-action 'none'; object-src 'none'">
<title>Weft · Reference proposal</title><style>{palette}{css}</style></head><body><main>
<header><p>weft · saved reference proposal</p><h1>{labels[summary['status']]}</h1><p class="notice">The floor penetration limit remains <strong>0.025 model units</strong>. Fewer failures or smaller image error do not make a proposal pass. Source image estimates are <strong>not measured truth</strong>. No geometry, physics, policy, model or media processing runs in this report.</p><p>Method: <code>{esc(summary['method'])}</code></p></header>
<section><h2>What changed, and what did not</h2><p>Source slots: {fmt(summary['source_counts'].get('retained_pose_slots'))}; body references: {fmt(summary['source_counts'].get('body_reference_knots'))}; retained slots without a body reference: {summary['source_slots_without_body_reference']}. Declared decoded frames: {fmt(summary['source_counts'].get('declared_decoded_frames'))}; container frame count: {fmt(summary['source_counts'].get('declared_frame_count'))}.</p>
<div class="scroll"><table><thead><tr><th>Sampling view</th><th>Checks</th><th>Input body failures</th><th>Proposed body failures</th><th>Unknown input / proposed</th></tr></thead><tbody>{counts}</tbody></table></div><p>Counts cover saved checks only. Body references without saved geometry: {summary['source_references_without_geometry']}. The grid schedule is not re-created here; missing checks do not imply success. Source and grid checks overlap in time: {summary['unique_sample_times']} unique nominal sampled times. They are <strong>not distinct physical poses</strong> and not a continuous-time collision check. Root, shape, floor, clocks and nonreference fields match the supplied input body (after any pose recovery), not a body from before crop selection.</p>
<p>Input body → Proposed body. Reported worst floor penetration: {fmt(summary['saved_floor_summary'].get('before_maximum_penetration_model_units'))} → {fmt(summary['saved_floor_summary'].get('after_maximum_penetration_model_units'))} model units. Reported confident endpoint error: {fmt(summary['image_summary'].get('mean_error_before_px'))} → {fmt(summary['image_summary'].get('mean_error_after_px'))} pixels; reported pairs: {fmt(summary['image_summary'].get('confident_source_endpoint_pairs'))}. Both bodies are compared against the <strong>same selected source-image targets</strong>. These are body-fitting errors, not before/after crop-detector accuracy. Missing targets remain unknown.</p>{objective_html}<details><summary>Saved gate outcomes — not a new evaluation</summary><table><tbody>{gates}</tbody></table></details></section>
<section><h2>Floor penetration by saved sample</h2><label>Sampling view <select id="mode"><option value="source">Retained source slots</option><option value="grid">Control grid</option></select></label>
<p>Input body: teal line. Proposed body: clay line. Dashed line: unchanged 0.025 limit. Missing values break the line; they are not plotted as zero.</p>
<svg id="chart" viewBox="0 0 950 290" role="img" aria-labelledby="chart-title chart-description"><title id="chart-title">Input body and proposed body maximum floor penetration</title><desc id="chart-description">Model-unit penetration by nominal time. Use the sample selector for exact values and unknowns.</desc><path d="M50 35V250H900" fill="none" stroke="currentColor"/><text x="5" y="20">model units</text><text id="ymax" x="5" y="43"></text><text x="28" y="254">0</text><text id="tfirst" x="50" y="278"></text><text id="tlast" x="800" y="278"></text><g id="lines"></g><circle id="before-dot" r="4" class="dot-before"/><circle id="after-dot" r="4" class="dot-after"/></svg>
<label>Sample <select id="sample"></select></label><button id="previous" type="button">Previous</button> <button id="next" type="button">Next</button>
<p id="identity" aria-live="polite"></p><p id="source-id" class="identity"></p><p id="context" class="identity"></p><p id="penetration"></p>
<h3>Signed support gaps</h3><p>Negative means below the authored floor; positive means above it. A larger signed gap alone does not establish support or contact. Capsule endpoints −1 and +1 and the center sample 0 are points on saved shapes, not extra body parts. Labels describe authored collision shapes, not recovered anatomy.</p><label><input id="affected" type="checkbox" checked> Show changed, below-floor or unknown gaps</label>
<div class="scroll"><table><thead><tr><th>Authored shape / support point</th><th>Saved identity</th><th>Input body gap</th><th>Proposed body gap</th><th>Change</th></tr></thead><tbody id="gaps"></tbody></table></div>
<h3>Source endpoint agreement</h3><p>Only confident source-knot estimates supply targets. Control-grid samples have no source-image targets, including times that coincide with a source slot.</p><p id="target-label"></p><details id="target-details" hidden><summary>Selected target provenance</summary><dl id="target-metadata" class="identity"></dl><p>Preserved observation and selected proposal hashes match the copied input. The decision hash is a saved identifier; decision bytes and image pixels are not read or independently verified here. Original whole-frame estimates remain provenance, not alternative fitting targets.</p></details><div class="scroll"><table><thead><tr><th>Endpoint</th><th>Saved status</th><th>Source confidence</th><th>Input body error (px)</th><th>Proposed body error (px)</th></tr></thead><tbody id="points"></tbody></table></div><noscript>Enable local JavaScript for sample selection, or inspect the exact supporting JSON below. No backend is used.</noscript></section>
{_surface_html(summary.get('surface_intervals'), esc, fmt)}
<section><h2>Exact copied evidence</h2><p>Only {'four' if len(members) == 4 else len(members)} selected members were hash-checked and copied. Other manifest members were not read. External manifest hash supplied: {str(summary['external_manifest_pin_supplied']).lower()}. Without one, the local manifest identity is frozen for consistency, not externally authenticated. The stated video source ID is not a new hash of video bytes.</p><ul>{links}<li><a href="source-manifest.json.txt">Original manifest</a></li><li><a href="report.json">Report data and provenance</a></li></ul><p class="identity">Source manifest SHA256: <code>{summary['source_manifest_sha256']}</code></p><p>{esc(summary['scope'])}</p><small>Generated {esc(summary['created_at'])} (America/Los_Angeles).</small></section>
</main><script id="data" type="application/json">{data}</script><script>{_SCRIPT}</script></body></html>'''.encode()


def prepare_whole_body_reference_report(source, out, *, expected_manifest_sha256=None, max_seconds=10,
                                        max_input_bytes=MAX_BYTES, max_output_bytes=MAX_BYTES):
    """Copy selected sealed JSON into a new bounded, offline inspection report."""
    _require(type(max_seconds) in (int, float) and math.isfinite(max_seconds) and 0 < max_seconds <= 30, "Report time bound must be 0–30 seconds")
    for value in (max_input_bytes, max_output_bytes):
        _require(type(value) is int and 1024 <= value <= MAX_BYTES, "Report byte bounds must be 1 KiB–16 MiB")
    if expected_manifest_sha256 is not None:
        _require(isinstance(expected_manifest_sha256, str) and _SHA.fullmatch(expected_manifest_sha256), "Expected manifest must be an exact SHA256")
    source, out = Path(source).absolute(), Path(out).absolute()
    _require('..' not in source.parts + out.parts and not source.is_symlink() and not out.is_symlink(), "Report paths must not traverse parents or links")
    source, out = source.resolve(strict=True), out.resolve(strict=False)
    _require(len(source.parts) <= 64 and len(out.parts) <= 64, 'Report paths exceed the depth bound')
    _require(not out.exists() and out != source and not out.is_relative_to(source) and not source.is_relative_to(out),
             "Report output must be new and outside its source")
    started = time.monotonic(); deadline = started + max_seconds

    def check():
        if time.monotonic() > deadline:
            raise TimeoutError("Offline reference report exceeded its cooperative time bound")

    check()
    manifest_raw = read_package_file(source, 'manifest.json', max_bytes=min(128 * 1024, max_input_bytes), expected_sha256=expected_manifest_sha256)
    manifest_sha = sha256(manifest_raw).hexdigest(); manifest = _json(manifest_raw); files = manifest.get('files')
    _require(type(manifest.get('schema_version')) is int and manifest['schema_version'] == 1
             and manifest.get('profile') == 'whole-body-reference-candidate-study-v1'
             and manifest.get('status') == 'completed' and isinstance(files, dict) and 1 <= len(files) <= 1024,
             "Expected a bounded completed study seal")
    for name, identity in files.items():
        _require(isinstance(name, str) and 0 < len(name) <= 512 and '\\' not in name and ':' not in name and all(ord(c) >= 32 for c in name)
                 and not Path(name).is_absolute() and all(part not in ('', '.', '..') for part in name.split('/'))
                 and name != 'manifest.json' and isinstance(identity, str) and _SHA.fullmatch(identity), "Unsafe manifest member or identity")
    _require(set(MEMBERS) <= set(files), "Study lacks required sealed report inputs")
    _require(not any(parent.as_posix() in files for name in files for parent in Path(name).parents), 'Manifest file/directory paths overlap')
    raw, total, members = {}, len(manifest_raw), dict(MEMBERS)
    def read_selected(name, bound=MAX_BYTES):
        nonlocal total
        check(); remaining = max_input_bytes - total
        _require(remaining > 0, "Selected report input byte bound exhausted")
        _require(name in files, 'Study lacks required sealed surface input: ' + name)
        raw[name] = read_package_file(source, name, max_bytes=min(bound, remaining), expected_sha256=files[name]); total += len(raw[name])
    for name in members: read_selected(name)
    documents = {name: _json(value) for name, value in raw.items()}
    if documents['report.json'].get('profile') == SURFACE_METHOD:
        from weft.source_surface_fit_inputs import surface_fit_member_names, validate_surface_fit_snapshot

        read_selected('plan.json', 128 << 10); plan = _json(raw['plan.json'])
        _require(type(plan.get('schema_version')) is int and plan['schema_version'] == 1
                 and plan.get('profile') == 'whole-body-reference-candidate-plan-v1' and plan.get('status') == 'frozen'
                 and plan.get('proposal_objective') == 'surface_interval_v1', 'Surface report requires its frozen input plan')
        descriptor = plan.get('surface_constraints'); names = surface_fit_member_names(descriptor)
        _require(type(plan.get('input_files')) is dict and all(plan['input_files'].get(name) == files.get(name)
                 and name in files for name in names | {'inputs/human.json', 'inputs/evidence-derived.json'}),
                 'Frozen surface input pins differ from the study seal')
        for name in sorted(names): read_selected(name, 64 << 10 if name.endswith('surface-config.json') else 8 << 20)
        check(); validate_surface_fit_snapshot(raw, descriptor, source_sha256=documents['inputs/human.json'].get('source_sha256')); check()
        members.update({'plan.json': 'frozen-plan.json.txt', **{name: Path(name).name + '.txt' for name in sorted(names)}})
        documents.update({name: _json(raw[name]) for name in {'plan.json', *names}})
    model = _model(documents, {name: files[name] for name in members}, check)
    check()
    palette_path = Path(__file__).parent / 'assets/brand.css'
    with palette_path.open('rb') as stream: palette_raw = stream.read(16385)
    _require(len(palette_raw) <= 16384 and b'</' not in palette_raw, "Invalid report palette")
    implementation = read_package_file(Path(__file__).parent, Path(__file__).name, max_bytes=256 * 1024)
    helper = None
    if 'surface_intervals' in model:
        helper = read_package_file(Path(__file__).parent, 'source_surface_fit_inputs.py', max_bytes=128 << 10)
        model['surface_input_validator'] = {'path': 'artifacts/source_surface_fit_inputs.py.txt',
            'sha256': sha256(helper).hexdigest(), 'bytes': len(helper)}
    model.update(schema_version=1, profile=PROFILE, created_at=datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z'),
        source_manifest_sha256=manifest_sha, external_manifest_pin_supplied=expected_manifest_sha256 is not None,
        selected_members={name: {'sha256': files[name], 'bytes': len(value)} for name, value in raw.items()},
        selected_input_bytes=total, source_read_byte_bound=2 * max_input_bytes,
        local_asset_read_byte_bound=16384 + 256 * 1024 + (128 << 10 if helper is not None else 0),
        implementation_sha256=sha256(implementation).hexdigest(), palette_sha256=sha256(palette_raw).hexdigest())
    page = _render(model, palette_raw.decode(), members)
    _require(len(page) <= MAX_HTML, "Offline report HTML exceeds its 2 MiB bound")
    output = {'index.html': page, 'report.json': _bytes(model), 'source-manifest.json.txt': manifest_raw,
              **{'artifacts/' + members[name]: value for name, value in raw.items()}}
    if helper is not None:
        output['artifacts/source_surface_fit_inputs.py.txt'] = helper
    _require(sum(map(len, output.values())) <= max_output_bytes - 65536, "Offline report output exceeds its byte bound")
    check(); out.mkdir(parents=True, exist_ok=False)
    for name, value in output.items():
        check(); write_package_files(out, {name: value})
    for name in members:
        check(); read_package_file(source, name, max_bytes=len(raw[name]), expected_sha256=files[name])
    check(); read_package_file(source, 'manifest.json', max_bytes=128 * 1024, expected_sha256=manifest_sha)
    for directory, expected in ((out, {'index.html', 'report.json', 'source-manifest.json.txt', 'artifacts'}),
                                (out / 'artifacts', set(members.values()) | ({'source_surface_fit_inputs.py.txt'} if helper is not None else set()))):
        with os.scandir(directory) as entries:
            actual = set()
            for entry in entries:
                check()
                _require(entry.name in expected and not entry.is_symlink(), 'Report output inventory changed')
                actual.add(entry.name)
        _require(actual == expected, 'Report output inventory changed')
    seal = _bytes({'schema_version': 1, 'profile': PROFILE, 'status': 'report_generated', 'source_manifest_sha256': manifest_sha,
                   'proposal_status': model['status'], 'promoted': False, 'files': {name: sha256(value).hexdigest() for name, value in output.items()}})
    _require(sum(map(len, output.values())) + len(seal) <= max_output_bytes, "Report seal exceeds its output byte bound")
    check(); write_package_files(out, {'manifest.json': seal})
    return {'directory': str(out), 'manifest_sha256': sha256(seal).hexdigest(), 'source_manifest_sha256': manifest_sha,
            'status': 'report_generated', 'proposal_status': model['status'], 'promoted': False, 'files': len(output) + 1,
            'bytes': sum(map(len, output.values())) + len(seal), 'seconds': time.monotonic() - started}
