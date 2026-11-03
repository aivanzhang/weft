"""Reject gross sampled skin spikes; absence is not visual approval."""

import json
import math
from pathlib import Path
import re

from avatar_assets.safety import local_file, read_json, require, sha256, slug
from scripts import motion_samples


VERSION = 'sampled-deformation-gate-v2'
MAX_STATES = 8
MAX_ROWS = 4
MAX_ARTIFACTS = 4096
MAX_OUTPUT_BYTES = 64 * 1024
SEVERE_RATIO = 8.0
SEVERE_HEIGHT_FRACTION = .03


def _hash(value, label):
    require(isinstance(value, str) and re.fullmatch(r'[a-f0-9]{64}', value), 'Invalid '+label)
    return value


def _number(value, low, high, label):
    require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high,
            'Invalid '+label)
    return float(value)


def _ids(rows, expected, label):
    require(isinstance(rows, list) and len(rows) == len(expected), 'Invalid '+label)
    require(all(isinstance(row, dict) for row in rows) and [row.get('id') for row in rows] == expected,
            label+' differ from the requested states')


def _artifact(manifest, name, path):
    rows = manifest.get('artifacts')
    require(isinstance(rows, list) and 1 <= len(rows) <= MAX_ARTIFACTS, 'Invalid bounded artifact inventory')
    require(all(isinstance(row, dict) for row in rows), 'Invalid artifact entry')
    matches = [row for row in rows if row.get('path') == name]
    require(len(matches) == 1, 'Missing or duplicate artifact: '+name)
    row = matches[0]
    require(type(row.get('bytes')) is int and row['bytes'] == path.stat().st_size
            and _hash(row.get('sha256'), 'artifact hash') == sha256(path), 'Artifact changed: '+name)
    return row['sha256']


def _names(value, label):
    require(isinstance(value, list) and 1 <= len(value) <= 64
            and all(isinstance(name, str) and 0 < len(name) <= 128
                    and all(ord(char) >= 32 for char in name) for name in value), 'Invalid '+label)
    require(len(set(value)) == len(value), 'Duplicate '+label)
    return set(value)


def _measurement(value, route):
    fields = {'schema_version', 'status', 'rig_route', 'coordinate_space', 'units', 'rest_sha256',
              'edges_sha256', 'rest_array_encoding', 'edge_array_encoding', 'vertices', 'edges',
              'bone_names', 'rest_bounds', 'pose_samples_measured', 'unchanged_topology_required'}
    require(isinstance(value, dict) and set(value) == fields, 'Missing or invalid native deformation measurement')
    require(type(value['schema_version']) is int and value['schema_version'] == 1
            and value['status'] == 'measured' and value['rig_route'] == route
            and value['coordinate_space'] == 'mesh_local_after_world_transform_applied'
            and value['units'] == 'meters' and value['unchanged_topology_required'] is True
            and value['rest_array_encoding'] == 'little-endian float64 XYZ'
            and value['edge_array_encoding'] == 'little-endian int32 pairs',
            'Unsupported native deformation measurement contract')
    for key in ('rest_sha256', 'edges_sha256'):
        _hash(value[key], 'measured '+key)
    for key, maximum in (('vertices', 100000), ('edges', 300000), ('pose_samples_measured', 64)):
        require(type(value[key]) is int and 1 <= value[key] <= maximum, 'Invalid measured '+key)
    bones = _names(value['bone_names'], 'measured bone names')
    bounds = value['rest_bounds']
    require(isinstance(bounds, dict) and set(bounds) == {'min', 'max', 'height', 'height_basis'},
            'Invalid measured rest bounds')
    for name in ('min', 'max'):
        require(isinstance(bounds[name], list) and len(bounds[name]) == 3, 'Invalid measured bounds vector')
        for coordinate in bounds[name]:
            _number(coordinate, -1e6, 1e6, 'measured bounds coordinate')
    span = [high-low for low, high in zip(bounds['min'], bounds['max'])]
    require(min(span) >= 0 and max(span) > 0, 'Degenerate measured rest bounds')
    basis = 'z_span' if span[2] > max(span)*1e-6 else 'maximum_span'
    height = span[2] if basis == 'z_span' else max(span)
    require(bounds['height_basis'] == basis
            and math.isclose(_number(bounds['height'], 1e-12, 1e6, 'measured height'), height,
                             rel_tol=1e-9, abs_tol=1e-12), 'Measured height disagrees with rest bounds')
    return json.loads(json.dumps(value, allow_nan=False)), bones, height


def _exported_bones(root, manifest, measured_bones):
    path = local_file(root/'rig-verification.json', 1024*1024)
    require(path.is_relative_to(root), 'Rig verification escaped its pack')
    verification_hash = _artifact(manifest, 'rig-verification.json', path)
    proof = read_json(path)
    require(isinstance(proof, dict) and type(proof.get('schema_version')) is int and proof['schema_version'] == 1
            and proof.get('status') == 'validated_local_structure'
            and proof.get('loading_verified') is True and proof.get('units') == 'meters'
            and manifest.get('rig_verification') == proof, 'Invalid bound rig verification')
    require(_names(proof.get('bones'), 'exported bone names') == measured_bones,
            'Measured bones differ from the verified exported skin')
    models = [row for row in manifest['artifacts'] if row.get('path') == 'model.glb']
    require(len(models) == 1 and type(models[0].get('bytes')) is int
            and 1 <= models[0]['bytes'] <= 64*1024*1024
            and _hash(models[0].get('sha256'), 'exported model hash') == proof.get('sha256'),
            'Rig verification belongs to a different exported model')
    return {'rig_verification_sha256': verification_hash, 'exported_model_sha256': proof['sha256']}


def _outlier(row, seconds, bones, vertex_limit):
    fields = {'endpoint_indices', 'endpoint_influences', 'excess_length', 'posed_length',
              'rest_length', 'rest_normalized_bounds', 'second', 'stretch_ratio'}
    require(isinstance(row, dict) and set(row) == fields, 'Invalid deformation outlier fields')
    endpoints = row['endpoint_indices']
    require(isinstance(endpoints, list) and len(endpoints) == 2
            and all(type(i) is int and 0 <= i < vertex_limit for i in endpoints)
            and endpoints[0] != endpoints[1], 'Invalid edge endpoints')
    second = _number(row['second'], 0, 4, 'outlier second')
    require(any(math.isclose(second, s, abs_tol=1e-6) for s in seconds), 'Outlier is not a sampled pose')
    rest = _number(row['rest_length'], 1e-12, 1e6, 'rest length')
    posed = _number(row['posed_length'], 1e-12, 1e6, 'posed length')
    excess = _number(row['excess_length'], 0, 1e6, 'excess length')
    ratio = _number(row['stretch_ratio'], 1, 1e12, 'stretch ratio')
    require(posed > rest and ratio > 1 and excess > 0
            and math.isclose(posed-rest, excess, rel_tol=1e-6, abs_tol=1e-9)
            and math.isclose(posed/rest, ratio, rel_tol=1e-6, abs_tol=1e-9),
            'Inconsistent deformation lengths or ratio')
    coordinates = row['rest_normalized_bounds']
    require(isinstance(coordinates, list) and len(coordinates) == 2, 'Invalid endpoint locations')
    for point in coordinates:
        require(isinstance(point, list) and len(point) == 3, 'Invalid endpoint coordinate')
        for value in point:
            _number(value, -1e-6, 1+1e-6, 'normalized endpoint coordinate')
    influences = row['endpoint_influences']
    require(isinstance(influences, list) and len(influences) == 2, 'Invalid endpoint weights')
    for weights in influences:
        require(isinstance(weights, dict) and 1 <= len(weights) <= 4 and set(weights) <= bones,
                'Invalid endpoint bones')
        values = [_number(value, 1e-6, 1+1e-6, 'endpoint weight') for value in weights.values()]
        require(math.isclose(sum(values), 1, abs_tol=1e-5), 'Endpoint weights are not normalized')
    return json.loads(json.dumps(row, allow_nan=False))


def audit_finish_pack(pack, *, expected_source_sha256, expected_design_sha256,
                      expected_state_ids, authored_height):
    """Read hash-bound samples; imported skins use their measured scale, not planned height."""
    source_hash = _hash(expected_source_sha256, 'expected source hash')
    design_hash = _hash(expected_design_sha256, 'expected design hash')
    height = _number(authored_height, .05, 10, 'authored height')
    require(isinstance(expected_state_ids, list) and 1 <= len(expected_state_ids) <= MAX_STATES,
            'Expected one to eight state IDs')
    expected = [slug(value) for value in expected_state_ids]
    require(len(set(expected)) == len(expected), 'Duplicate expected states')
    root = Path(pack)
    require(not root.is_symlink() and root.is_dir(), 'Invalid finish pack directory')
    root = root.resolve()
    paths = {}
    for key, name in (('manifest', 'manifest.json'), ('report', 'finish-report.json'),
                      ('design', 'sources/finish-design.json')):
        paths[key] = local_file(root/name, 1024*1024)
        require(paths[key].is_relative_to(root), 'Finish evidence escaped its pack')
    manifest, report, design = (read_json(paths[key]) for key in ('manifest', 'report', 'design'))
    require(all(isinstance(item, dict) and type(item.get('schema_version')) is int and item['schema_version'] == 1
                for item in (manifest, report, design)), 'Invalid finish evidence schema')
    require(manifest.get('status') == 'needs_visual_review', 'Native finish did not complete')
    require(all(item.get('source_sha256') == source_hash for item in (manifest, report, design))
            and report.get('source_unchanged') is True, 'Finish source binding changed')
    require(manifest.get('id') == design.get('id') and isinstance(design.get('id'), str), 'Finish design ID differs')
    report_hash = _artifact(manifest, 'finish-report.json', paths['report'])
    require(_artifact(manifest, 'sources/finish-design.json', paths['design']) == design_hash,
            'Finish design binding changed')
    require(manifest.get('finish') == report, 'Embedded finish report differs')
    rig, authored = design.get('authored_rig'), report.get('authored_rig')
    route = 'authored' if rig is not None else 'source_skin'
    measurement, extra_bindings, vertex_limit, threshold_height = None, {}, 100000, height
    if route == 'authored':
        require(isinstance(rig, dict) and isinstance(authored, dict)
                and authored.get('design') == rig and authored.get('authored_not_recovered') is True,
                'Measured rig differs from the bound design')
        require(_number(rig.get('height_meters'), .05, 10, 'design height') == height
                and _number(authored.get('height_meters'), .05, 10, 'measured authored height') == height,
                'Authored height differs from the expected scale')
        bone_rows = rig.get('bones')
        require(isinstance(bone_rows, list) and 1 <= len(bone_rows) <= 64
                and all(isinstance(bone, dict) and isinstance(bone.get('name'), str)
                        and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', bone['name']) for bone in bone_rows),
                'Invalid bounded authored bones')
        bones = _names([bone['name'] for bone in bone_rows], 'authored bones')
    else:
        require(authored is None, 'Source skin unexpectedly contains an authored rig report')
    if 'deformation_measurement' in report or route == 'source_skin':
        measurement, measured_bones, measured_height = _measurement(report.get('deformation_measurement'), route)
        extra_bindings = _exported_bones(root, manifest, measured_bones)
        vertex_limit = measurement['vertices']
        if route == 'authored':
            require(measured_bones == bones and math.isclose(measured_height, height, rel_tol=1e-4, abs_tol=1e-7),
                    'Native measurement differs from the authored bones or height')
        bones, threshold_height = measured_bones, measured_height
    for label, rows in (('Design states', design.get('states')), ('Manifest states', manifest.get('states')),
                        ('Reported states', report.get('states')), ('Authored actions', design.get('authored_actions'))):
        _ids(rows, expected, label)
    results = []
    for state, action in zip(report['states'], design['authored_actions']):
        require(_number(state.get('duration_seconds'), 4, 4, 'sampled duration') == 4,
                'Expected four-second authored actions')
        sampling = motion_samples.sample_action(action, (0, 96), 24)
        for key in ('frames', 'seconds'):
            values = state.get(key)
            require(isinstance(values, list) and len(values) == len(sampling[key]), 'Invalid sampled '+key)
            for actual, planned in zip(values, sampling[key]):
                require(math.isclose(_number(actual, 0, 96, 'sampled '+key), planned, abs_tol=1e-6),
                        'Sample times differ from the bound authored action')
        require(state.get('sampling_method') == sampling['sampling_method'], 'Sampling method differs')
        if measurement is not None:
            require(type(state.get('deformation_samples_measured')) is int
                    and state['deformation_samples_measured'] == len(sampling['frames'])
                    and state.get('deformation_rest_sha256') == measurement['rest_sha256'],
                    'State lacks bound native deformation samples')
        else:
            require('deformation_samples_measured' not in state and 'deformation_rest_sha256' not in state,
                    'Native measurement marker is missing')
        rows = state.get('deformation_outliers')
        require(isinstance(rows, list) and len(rows) <= MAX_ROWS, 'Missing or oversized deformation evidence')
        checked = [_outlier(row, sampling['seconds'], bones, vertex_limit) for row in rows]
        require(len({tuple(sorted(row['endpoint_indices'])) for row in checked}) == len(checked),
                'Duplicate deformation edge evidence')
        offenders = [row for row in checked if row['stretch_ratio'] > SEVERE_RATIO
                     and row['excess_length'] > SEVERE_HEIGHT_FRACTION*threshold_height]
        results.append({'id': state['id'], 'passed': not offenders, 'sample_seconds': sampling['seconds'],
                        'reported_outliers': len(checked), 'offending_outliers': offenders,
                        'max_reported_excess_m': max((row['excess_length'] for row in checked), default=0),
                        'max_reported_stretch_ratio': max((row['stretch_ratio'] for row in checked), default=1)})
    if measurement is not None:
        require(measurement['pose_samples_measured'] == sum(len(state['sample_seconds']) for state in results),
                'Native deformation sample total differs')
    passed = all(state['passed'] for state in results)
    result = {'schema_version': 1, 'version': VERSION, 'passed': passed,
              'status': 'no_reported_severe_spike' if passed else 'severe_sampled_deformation',
              'bindings': {'source_sha256': source_hash, 'design_sha256': design_hash,
                           'finish_report_sha256': report_hash, 'manifest_sha256': sha256(paths['manifest']),
                           'state_ids': expected, 'requested_height_m': height,
                           'authored_height_m': height if route == 'authored' else None, **extra_bindings},
              'rig_route': route, 'measurement': measurement,
              'measurement_scope': 'bound_native_rest_and_pose_samples' if measurement else 'legacy_authored_samples',
              'implementation_sha256': sha256(Path(__file__)),
              'sampling_implementation_sha256': sha256(Path(motion_samples.__file__)),
              'thresholds': {'stretch_ratio_strictly_greater_than': SEVERE_RATIO,
                             'excess_height_fraction_strictly_greater_than': SEVERE_HEIGHT_FRACTION,
                             'height_m': threshold_height,
                             'height_source': 'native_measured_rest' if measurement else 'bound_authored_design',
                             'excess_m_strictly_greater_than': SEVERE_HEIGHT_FRACTION*threshold_height},
              'states': results, 'visual_approval': False, 'physics_verified': False,
              'scope': 'Conservative gross-spike rejection of at most four retained edges per sampled pose set; '
                       'not exhaustive edge/frame coverage, skin correctness or visual acceptance.'}
    require(len(json.dumps(result, allow_nan=False).encode()) <= MAX_OUTPUT_BYTES, 'Deformation report exceeds bound')
    return result
