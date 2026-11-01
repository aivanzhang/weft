"""Reject gross sampled skin spikes; absence is not visual approval."""

import json
import math
from pathlib import Path
import re

from avatar_assets.safety import local_file, read_json, require, sha256, slug
from scripts import motion_samples


VERSION = 'sampled-deformation-gate-v1'
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


def _outlier(row, seconds, bones):
    fields = {'endpoint_indices', 'endpoint_influences', 'excess_length', 'posed_length',
              'rest_length', 'rest_normalized_bounds', 'second', 'stretch_ratio'}
    require(isinstance(row, dict) and set(row) == fields, 'Invalid deformation outlier fields')
    endpoints = row['endpoint_indices']
    require(isinstance(endpoints, list) and len(endpoints) == 2
            and all(type(i) is int and 0 <= i < 100000 for i in endpoints)
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
    """Read hash-bound authored-rig samples; do not scan or render the model."""
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
    rig, measured = design.get('authored_rig'), report.get('authored_rig')
    require(isinstance(rig, dict) and isinstance(measured, dict),
            'Authored-rig sampled evidence required; pre-rigged diagnostics are unsupported')
    require(measured.get('design') == rig and measured.get('authored_not_recovered') is True,
            'Measured rig differs from the bound design')
    require(_number(rig.get('height_meters'), .05, 10, 'design height') == height
            and _number(measured.get('height_meters'), .05, 10, 'measured height') == height,
            'Authored height differs from the expected scale')
    bone_rows = rig.get('bones')
    require(isinstance(bone_rows, list) and 1 <= len(bone_rows) <= 64
            and all(isinstance(bone, dict) and isinstance(bone.get('name'), str)
                    and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', bone['name']) for bone in bone_rows),
            'Invalid bounded authored bones')
    bones = {bone['name'] for bone in bone_rows}
    require(len(bones) == len(bone_rows), 'Duplicate authored bone')
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
        rows = state.get('deformation_outliers')
        require(isinstance(rows, list) and len(rows) <= MAX_ROWS, 'Missing or oversized deformation evidence')
        checked = [_outlier(row, sampling['seconds'], bones) for row in rows]
        require(len({tuple(sorted(row['endpoint_indices'])) for row in checked}) == len(checked),
                'Duplicate deformation edge evidence')
        offenders = [row for row in checked if row['stretch_ratio'] > SEVERE_RATIO
                     and row['excess_length'] > SEVERE_HEIGHT_FRACTION*height]
        results.append({'id': state['id'], 'passed': not offenders, 'sample_seconds': sampling['seconds'],
                        'reported_outliers': len(checked), 'offending_outliers': offenders,
                        'max_reported_excess_m': max((row['excess_length'] for row in checked), default=0),
                        'max_reported_stretch_ratio': max((row['stretch_ratio'] for row in checked), default=1)})
    passed = all(state['passed'] for state in results)
    result = {'schema_version': 1, 'version': VERSION, 'passed': passed,
              'status': 'no_reported_severe_spike' if passed else 'severe_sampled_deformation',
              'bindings': {'source_sha256': source_hash, 'design_sha256': design_hash,
                           'finish_report_sha256': report_hash, 'manifest_sha256': sha256(paths['manifest']),
                           'state_ids': expected, 'authored_height_m': height},
              'implementation_sha256': sha256(Path(__file__)),
              'sampling_implementation_sha256': sha256(Path(motion_samples.__file__)),
              'thresholds': {'stretch_ratio_strictly_greater_than': SEVERE_RATIO,
                             'excess_height_fraction_strictly_greater_than': SEVERE_HEIGHT_FRACTION,
                             'excess_m_strictly_greater_than': SEVERE_HEIGHT_FRACTION*height},
              'states': results, 'visual_approval': False, 'physics_verified': False,
              'scope': 'Conservative gross-spike rejection of at most four retained edges per sampled pose set; '
                       'not exhaustive edge/frame coverage, skin correctness or visual acceptance.'}
    require(len(json.dumps(result, allow_nan=False).encode()) <= MAX_OUTPUT_BYTES, 'Deformation report exceeds bound')
    return result
