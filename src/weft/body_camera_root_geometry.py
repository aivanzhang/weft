"""Read original body geometry without dynamics; never install proposed roots."""

from hashlib import sha256
import json
import math
import time

import numpy as np

from weft.io import canonical_hash
from weft import whole_body_reference_refinement as reference


PROFILE = 'fixed-original-body-geometry-v1'
CHECK_PROFILE = 'fixed-original-body-geometry-check-v1'
MAX_BYTES = 8 << 20
MAX_UNIQUE_SAMPLES = 2701
TOLERANCE = 1e-8


def _require(value, message):
    if not value:
        raise ValueError('Camera/root geometry: ' + message)


def _check(deadline):
    _require(type(deadline) in (int, float) and math.isfinite(deadline), 'finite deadline required')
    if time.monotonic() >= deadline:
        raise TimeoutError('Camera/root geometry exceeded its deadline')


def _encode(value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    _require(len(raw) <= MAX_BYTES, 'geometry exceeds 8 MiB')
    return raw


def _array(value, shape):
    original = np.asarray(value, dtype=object)
    _require(original.shape == shape and all(isinstance(item, (int, float, np.integer, np.floating))
             and not isinstance(item, (bool, np.bool_)) for item in original.flat), 'numeric geometry values required')
    result = np.asarray(value, dtype=float)
    _require(result.shape == shape and np.isfinite(result).all() and np.max(abs(result), initial=0.) <= 1e4,
             'invalid native geometry values')
    return result


def _prepare(human_document, evidence_document, deadline):
    _check(deadline)
    spec, evidence = reference._document(human_document), reference._document(evidence_document)
    times, angles, roots, prepared, _ = reference._prepare(spec, evidence)
    rows, samples, lookup = reference._schedule(times)
    _require(len(samples) <= MAX_UNIQUE_SAMPLES, 'unique sample bound')
    _check(deadline)
    return spec, evidence, times, angles, roots, prepared, rows, samples, lookup


def _make_geometry(spec):
    return reference._NativeGeometry(spec)


def _xml(spec):
    from weft.human import human_xml
    return human_xml(spec).encode()


def _plane(geometry):
    normal = _array(geometry.data.geom_xmat[geometry.floor].reshape(3, 3)[:, 2], (3,))
    origin = _array(geometry.data.geom_xpos[geometry.floor], (3,))
    _require(abs(np.linalg.norm(normal) - 1.) <= 1e-10, 'unit fixed floor normal required')
    return {'normal': normal.tolist(), 'offset_model_units': float(normal @ origin)}


def _declared_floor(spec):
    normal = _array(spec['mobility'].get('ground_plane', {}).get('normal', [0., 0., 1.]), (3,))
    length = np.linalg.norm(normal)
    _require(length > 0, 'nonzero declared floor normal required')
    return {'normal': (normal / length).tolist(), 'offset_model_units': 0.}


def _same_plane(actual, expected):
    return (np.max(abs(np.asarray(actual['normal']) - expected['normal'])) <= TOLERANCE
            and abs(actual['offset_model_units'] - expected['offset_model_units']) <= TOLERANCE)


def _state(geometry, angles, root, deadline):
    _check(deadline)
    value = geometry.evaluate(angles, root, derivatives=False)
    _check(deadline)
    # The reused kinematics returns pelvis-relative sites, not absolute model coordinates.
    pelvis = geometry.data.site_xpos[geometry.env._site_ids['pelvis']]
    points = _array(value['points'], (12, 3)) + _array(pelvis, (3,))
    points = _array(points, (12, 3))
    gaps = _array(value['gaps'], (len(geometry.support_names),))
    return {'points': points, 'gaps': gaps, 'floor': _plane(geometry)}


def _supports(geometry):
    names = geometry.support_names
    _require(1 <= len(names) <= 64 and len(set(names)) == len(names)
             and all(type(name) is str and 1 <= len(name) <= 100 for name in names), 'complete distinct support IDs required')
    return list(names)


def _runtime(runtime, model_xml):
    _require(type(runtime) is dict and set(runtime) == {'implementation_sha256', 'versions'}, 'exact runtime identity required')
    pins = runtime['implementation_sha256']
    _require(type(pins) is dict and 1 <= len(pins) <= 32 and all(
        type(name) is str and name.endswith('.py') and '/' not in name and '\\' not in name
        and type(value) is str and len(value) == 64 and set(value) <= set('0123456789abcdef')
        for name, value in pins.items()), 'invalid runtime pins')
    versions = runtime['versions']
    _require(type(versions) is dict and set(versions) == {'python', 'numpy', 'mujoco', 'gymnasium'}
             and all(type(value) is str and 1 <= len(value) <= 100 for value in versions.values()), 'invalid dependency versions')
    return {**runtime, 'model_xml_sha256': sha256(model_xml).hexdigest()}


def extract_original_geometry(human_document, evidence_document, *, runtime, deadline):
    """Extract source endpoints and all source/30-Hz floor gaps from one unchanged model."""
    spec, evidence, times, angles, roots, prepared, rows, samples, lookup = _prepare(
        human_document, evidence_document, deadline)
    runtime = _runtime(runtime, _xml(spec))
    floor = _declared_floor(spec)
    _check(deadline)
    geometry = _make_geometry(spec)
    values = []
    try:
        supports = _supports(geometry)
        for sample in samples:
            q = sum(weight * angles[index] for index, weight in sample['weights'])
            root = sum(weight * roots[index] for index, weight in sample['weights'])
            value = _state(geometry, q, root, deadline)
            _require(_same_plane(value['floor'], floor), 'native floor differs from the fixed declared plane')
            values.append(value)
        frames = {row['id']: row for row in evidence['frames']}
        source_rows = []
        for index, prepared_row in enumerate(prepared):
            value = values[lookup[float(times[index])]]
            frame = frames[prepared_row['evidence_id']]
            source_rows.append({'evidence_id': prepared_row['evidence_id'], 'frame': prepared_row['frame'],
                'time_s': prepared_row['time_s'], 'decoder_timestamp_s': frame.get('decoder_timestamp_s'),
                'root': roots[index].tolist(),
                'points': {name: point.tolist() for name, point in zip(reference.POINTS, value['points'], strict=True)}})
        bindings = {'source_sha256': spec['source_sha256'], 'body_sha256': sha256(human_document).hexdigest(),
            'evidence_sha256': sha256(evidence_document).hexdigest(), 'body_angles_sha256': canonical_hash(spec['reference']),
            'body_shape_sha256': canonical_hash({key: value for key, value in spec.items() if key not in {'reference', 'root_reference'}}),
            'root_reference_sha256': canonical_hash(spec['root_reference']), 'floor_sha256': canonical_hash(floor)}
        result = {'schema_version': 1, 'profile': PROFILE, 'status': 'completed', 'bindings': bindings,
            'floor': floor, 'source_rows': source_rows, 'support_names': supports,
            'floor_rows': [{'kind': kind, 'time_s': stamp,
                'gaps': dict(zip(supports, values[lookup[stamp]]['gaps'].tolist(), strict=True))} for kind, stamp, _ in rows],
            'runtime': runtime, 'physics_steps': 0}
        _check(deadline)
        return _encode(result)
    finally:
        geometry.close()


def check_proposed_geometry(human_document, evidence_document, geometry_document, proposal_document, *, deadline):
    """Re-evaluate changed roots in a fresh original model; this is not body admission."""
    spec, _, times, angles, roots, _, rows, samples, lookup = _prepare(human_document, evidence_document, deadline)
    original = reference._document(geometry_document)
    _require(type(proposal_document) is bytes and 0 < len(proposal_document) <= 16 << 20, 'bounded proposal bytes required')
    def unique(rows):
        result = {}
        for key, value in rows:
            _require(key not in result, 'duplicate proposal key')
            result[key] = value
        return result
    proposal = json.loads(proposal_document, object_pairs_hook=unique)
    json.dumps(proposal, allow_nan=False)
    _require(type(proposal) is dict, 'proposal object required')
    _require(original.get('profile') == PROFILE and original.get('status') == 'completed'
             and original['bindings']['body_sha256'] == sha256(human_document).hexdigest()
             and original['bindings']['evidence_sha256'] == sha256(evidence_document).hexdigest(), 'original geometry identity differs')
    _require(original['runtime']['model_xml_sha256'] == sha256(_xml(spec)).hexdigest(), 'original model XML differs')
    _require(original['floor'] == _declared_floor(spec), 'original declared plane differs')
    candidate = proposal.get('root_reference_candidate')
    _require(type(candidate) is list and len(candidate) == len(times), 'complete proposed roots required')
    proposed_roots = []
    for row, stamp in zip(candidate, times, strict=True):
        _require(type(row) is dict and set(row) == {'time_s', 'translation'} and row['time_s'] == stamp,
                 'proposed root clocks differ')
        proposed_roots.append(_array(row['translation'], (3,)))
    proposed_roots = np.array(proposed_roots)
    _require(np.array_equal(proposed_roots[0], roots[0]), 'first root changed')
    delta = proposed_roots - roots
    _require(np.max(np.linalg.norm(delta, axis=1)) <= .08 + TOLERANCE
             and np.max(np.linalg.norm(np.diff(delta, axis=0) / np.diff(times)[:, None], axis=1)) <= .3 + TOLERANCE,
             'proposed roots exceed unchanged bounds')
    result = {'schema_version': 1, 'profile': CHECK_PROFILE, 'status': 'not_required', 'reason': 'original_roots_unchanged',
        'bindings': {'body_sha256': sha256(human_document).hexdigest(), 'evidence_sha256': sha256(evidence_document).hexdigest(),
                     'geometry_sha256': sha256(geometry_document).hexdigest(), 'proposal_sha256': sha256(proposal_document).hexdigest()},
        'source_pose_checks': 0, 'control_grid_checks': 0, 'unique_evaluations': 0, 'rows': [],
        'maximum_point_translation_error_model_units': None, 'maximum_gap_translation_error_model_units': None,
        'gates': {name: None for name in ('fixed_plane', 'floor_no_worse', 'floor_feasible', 'translation_consistency')},
        'physics_steps': 0, 'admitted': False, 'promoted': False, 'physical_validation': 'unverified',
        'scope': 'Fresh original-model sampled floor/translation diagnostic only; no equipment, self-collision, continuous-time or physical qualification'}
    if np.array_equal(proposed_roots, roots):
        return _encode(result)
    supports = original['support_names']
    original_rows = original['floor_rows']
    _require(len(original_rows) == len(rows) and len(original['source_rows']) == len(times), 'incomplete original geometry')
    baseline = {}
    for saved, (kind, stamp, _) in zip(original_rows, rows, strict=True):
        _require(saved['kind'] == kind and saved['time_s'] == stamp and set(saved['gaps']) == set(supports),
                 'original floor schedule/supports differ')
        gaps = _array([saved['gaps'][name] for name in supports], (len(supports),))
        _require(stamp not in baseline or np.array_equal(baseline[stamp], gaps), 'duplicate-time original gaps differ')
        baseline[stamp] = gaps
    _check(deadline)
    geometry = _make_geometry(spec)
    values, max_point_error, max_gap_error = [], 0., 0.
    try:
        _require(_supports(geometry) == supports, 'fresh model support inventory differs')
        normal = _array(original['floor']['normal'], (3,))
        for sample in samples:
            q = sum(weight * angles[index] for index, weight in sample['weights'])
            root = sum(weight * proposed_roots[index] for index, weight in sample['weights'])
            shift = sum(weight * delta[index] for index, weight in sample['weights'])
            value = _state(geometry, q, root, deadline)
            _require(_same_plane(value['floor'], original['floor']), 'fresh original floor differs')
            error = float(np.max(abs(value['gaps'] - baseline[sample['time_s']] - normal @ shift)))
            max_gap_error = max(max_gap_error, error)
            values.append({**value, 'error': error})
        for index, row in enumerate(original['source_rows']):
            _require(row['time_s'] == times[index], 'original source clock differs')
            expected = _array([row['points'][name] for name in reference.POINTS], (12, 3)) + delta[index]
            actual = values[lookup[float(times[index])]]['points']
            max_point_error = max(max_point_error, float(np.max(abs(actual - expected))))
        after_rows = [{'kind': kind, 'time_s': stamp,
            'gaps': dict(zip(supports, values[lookup[stamp]]['gaps'].tolist(), strict=True)),
            'maximum_gap_translation_error_model_units': values[lookup[stamp]]['error']} for kind, stamp, _ in rows]
        no_worse = all(np.all(values[index]['gaps'] >= np.minimum(0., baseline[sample['time_s']]) - TOLERANCE)
                       for index, sample in enumerate(samples))
        feasible = all(np.all(value['gaps'] >= -.025) for value in values)
        result.update(status='completed', reason=None, rows=after_rows, source_pose_checks=len(times),
            control_grid_checks=len(rows) - len(times), unique_evaluations=len(samples),
            maximum_point_translation_error_model_units=max_point_error,
            maximum_gap_translation_error_model_units=max_gap_error,
            gates={'fixed_plane': True, 'floor_no_worse': bool(no_worse), 'floor_feasible': bool(feasible),
                   'translation_consistency': max(max_point_error, max_gap_error) <= TOLERANCE})
        _check(deadline)
        return _encode(result)
    finally:
        geometry.close()
