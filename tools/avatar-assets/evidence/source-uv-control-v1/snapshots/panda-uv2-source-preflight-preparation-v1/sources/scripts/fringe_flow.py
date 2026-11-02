"""Opt-in regional tip reshaping; selected roots, source shading and skin stay exact."""

from dataclasses import asdict
import hashlib
import json
import math
import time

import numpy as np

if __package__:
    from .edge_fringe import export_budget
    from .groom_flow import prepare_flow, evaluate_prepared_flow
else:
    from edge_fringe import export_budget
    from groom_flow import prepare_flow, evaluate_prepared_flow


POLICY = 'shared-regional-fin-response-v2'
ARRAYS = {'positions', 'normals', 'uv', 'triangles', 'joints', 'weights'}
MAX_FINS = 4800
MAX_SECONDS = 10


def _require(condition, message):
    if not condition:
        raise ValueError('Fringe flow: '+message)


def _array(value, shape, kinds, name):
    _require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype.kind in kinds,
             'invalid '+name+' array')
    _require(value.dtype.kind != 'f' or value.dtype.itemsize in (4, 8) and np.isfinite(value).all(),
             'nonfinite or unsupported '+name)
    return value


def _hash(value):
    return hashlib.sha256(value.tobytes(order='C')).hexdigest()


def _unit(value, label):
    length = np.linalg.norm(value, axis=-1, keepdims=True)
    _require(np.isfinite(length).all() and np.all(length > 1e-10), 'degenerate '+label)
    return value/length


def _range(value):
    return {'min': float(value.min()), 'median': float(np.median(value)), 'max': float(value.max())}


def reshape_fringe(fins, *, low, span, base_direction, base_comb, groom_flow,
                   base_length_m, source_export_vertices, source_export_triangles):
    """Return six detached arrays plus report; input uses rootA/rootB/tipA/tipB order.

    The immutable selection is not reranked. Each midpoint samples the shared field
    once; its length scale and tangent-projected comb affect only the two tips.
    Caller retains root-ID metadata and must independently verify exported motion.
    """
    started = time.monotonic()

    def deadline():
        _require(time.monotonic()-started < MAX_SECONDS, 'ten-second deadline exceeded')

    _require(isinstance(fins, dict) and set(fins) == ARRAYS, 'expected exactly six selected-fin arrays')
    points = fins['positions']
    _require(isinstance(points, np.ndarray) and points.ndim == 2 and len(points)%4 == 0
             and 1 <= len(points)//4 <= MAX_FINS, 'requires1..4800 four-vertex fins')
    count, vertices = len(points)//4, len(points)
    budget = export_budget(source_export_vertices, source_export_triangles, count)
    _require(type(base_length_m) in (int, float) and math.isfinite(base_length_m)
             and .001 <= base_length_m <= .04, 'base length must be .001..04 meters')
    points = _array(points, (vertices, 3), 'f', 'positions')
    normal = _array(fins['normals'], (vertices, 3), 'f', 'normals')
    uv = _array(fins['uv'], (vertices, 2), 'f', 'atlas UV')
    triangles = _array(fins['triangles'], (count*4, 3), 'iu', 'triangles')
    joints = _array(fins['joints'], (vertices, 4), 'iu', 'joints')
    weights = _array(fins['weights'], (vertices, 4), 'f', 'weights')
    _require(np.abs(points).max() <= 10000 and np.all((uv >= 0)&(uv <= 1)), 'position/atlas UV bounds exceeded')
    _require(np.max(np.abs(np.linalg.norm(normal.astype(float), axis=1)-1)) < 1e-5, 'normals must be unit')
    _require(joints.min() >= 0 and joints.max() < 64 and np.all((weights >= 0)&(weights <= 1))
             and np.max(np.abs(weights.astype(float).sum(axis=1)-1)) <= 1e-6, 'invalid four-weight skin')
    front = (np.arange(count)[:, None, None]*4+np.array([[0, 1, 3], [0, 3, 2]])).reshape(-1, 3)
    expected = np.concatenate((front, front[:, ::-1]))
    _require(np.array_equal(triangles, expected), 'requires exact front and reversed fin winding/multiplicity')
    quads, shading = points.astype(float).reshape(count, 4, 3), normal.astype(float).reshape(count, 4, 3)
    for name, value in (('normals', normal), ('joints', joints), ('weights', weights)):
        rows = value.reshape(count, 4, -1)
        _require(np.array_equal(rows[:, :2], rows[:, 2:]), 'tips must inherit exact root '+name)
    _require(np.all(np.einsum('ij,ij->i', shading[:, 0], shading[:, 1]) >= .5), 'incoherent endpoint normals')
    outward = _unit(shading[:, :2].sum(axis=1), 'outward normal')
    edge = quads[:, 1]-quads[:, 0]
    edge_length = np.linalg.norm(edge, axis=1)
    _require(np.all((edge_length >= .0085-1e-6)&(edge_length <= .020+1e-6)), 'source edge span differs from8.5..20mm policy')
    old_lifts = quads[:, 2:]-quads[:, :2]
    _require(np.all(np.einsum('fvc,fvc->fv', old_lifts, shading[:, :2]) > 1e-8), 'input tips are inward or collapsed')
    old_faces = quads.reshape(-1, 3)[front]
    old_area = np.linalg.norm(np.cross(old_faces[:, 1]-old_faces[:, 0], old_faces[:, 2]-old_faces[:, 0]), axis=1)
    _require(np.all(old_area > 1e-12), 'degenerate input fin triangles')
    field = prepare_flow(low, span, base_direction, base_comb, groom_flow)
    field_json = json.dumps(asdict(field), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    roots = quads[:, :2].mean(axis=1)
    directions, combs, scales = np.empty((count, 3)), np.empty(count), np.empty(count)
    for index, root in enumerate(roots):
        if index%128 == 0:
            deadline()
        directions[index], combs[index], scales[index] = evaluate_prepared_flow(root, field)
    lengths = float(base_length_m)*scales
    _require(np.isfinite(lengths).all() and np.all((lengths >= .001)&(lengths <= .04)),
             'regional length outside .001..04 meters; no silent clamp')
    tangent = directions-outward*np.einsum('fc,fc->f', directions, outward)[:, None]
    tangent_lengths = np.linalg.norm(tangent, axis=1)
    # A parallel field has no tangent to invent; its displacement remains outward.
    parallel = tangent_lengths <= 1e-12
    tangent[~parallel] /= tangent_lengths[~parallel, None]
    tangent[parallel] = 0
    displacement_direction = _unit(outward*(1-combs[:, None])+tangent*combs[:, None], 'regional displacement')
    result = {name: np.array(value, copy=True, order='C') for name, value in fins.items()}
    reshaped = result['positions'].reshape(count, 4, 3)
    reshaped[:, 2:] = quads[:, :2]+displacement_direction[:, None]*lengths[:, None, None]
    actual = reshaped.astype(float)
    actual_lifts = actual[:, 2:]-actual[:, :2]
    length_error = np.abs(np.linalg.norm(actual_lifts, axis=2)-lengths[:, None])
    outward_distance = np.einsum('fvc,fvc->fv', actual_lifts, shading[:, :2])
    _require(np.isfinite(actual).all() and np.abs(actual).max() <= 10000, 'invalid or oversized output positions')
    _require(np.all(outward_distance > 1e-8), 'regional response points inward at a root endpoint')
    _require(length_error.max() <= 1e-6, 'stored tips cannot preserve requested length within one micrometer')
    faces = actual.reshape(-1, 3)[front]
    area = np.linalg.norm(np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0]), axis=1)
    _require(np.all(area > 1e-12), 'regional response creates degenerate triangles')
    _require(reshaped[:, :2].tobytes() == points.reshape(count, 4, 3)[:, :2].tobytes(), 'root bytes changed')
    hashes = {name: {'sha256': _hash(value), 'dtype': value.dtype.str, 'shape': list(value.shape)} for name, value in fins.items()}
    unchanged = sorted(ARRAYS-{'positions'})
    _require(all(_hash(result[name]) == hashes[name]['sha256'] for name in unchanged), 'protected fin arrays changed')
    deadline()
    result['report'] = {'schema_version': 1, 'policy': POLICY, 'fins': count, 'vertices': vertices, 'triangles': len(triangles),
        'base_length_m': float(base_length_m), 'length_m': _range(lengths), 'regional_comb': _range(combs),
        'regional_length_scale': _range(scales), 'outward_angle_degrees': _range(np.degrees(np.arccos(np.clip(
            np.einsum('fc,fc->f', displacement_direction, outward), -1, 1)))),
        'prepared_field_sha256': hashlib.sha256(field_json).hexdigest(), 'root_midpoints_sha256': _hash(roots),
        'input_arrays': hashes, 'output_positions_sha256': _hash(result['positions']),
        'root_positions_byte_exact': True, 'unchanged_arrays': unchanged, 'selection_changed': False,
        'parallel_field_roots': int(parallel.sum()), 'maximum_stored_length_error_m': float(length_error.max()),
        'minimum_endpoint_outward_displacement_m': float(outward_distance.min()), 'export_budget': budget,
        'elapsed_seconds': time.monotonic()-started, 'limits': {'fins': MAX_FINS, 'seconds': MAX_SECONDS,
            'export_vertices': 100000, 'export_triangles': 100000, 'regional_length_m': [.001, .04]},
        'quality_approved': False, 'motion_verified': False, 'input_modified': False,
        'limitations': ['Authored fixed-root tip reshaping, not denser fur or recovered/simulated hair.',
            'Positive displacement against endpoint normals does not prove clearance from curved or other surfaces.',
            'Caller preserves root metadata and independently verifies the actual exported roots, skin, motions and appearance.',
            'This helper does not change v1 selection, public schemas, defaults, materials, atlas alpha or source geometry.']}
    return result
