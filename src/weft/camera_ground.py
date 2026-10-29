"""Supplied ground coordinates are additional evidence, not a fitted contact repair."""

from copy import deepcopy
import math

import numpy as np


METHOD = 'supplied-horizontal-world-ground-v1'


def validate_ground(ground, binding):
    fields = {'schema_version', 'source_sha256', 'calibration_sha256', 'world_axes',
              'units', 'height_m', 'quality', 'provenance'}
    if (not isinstance(ground, dict) or set(ground) != fields
            or type(ground['schema_version']) is not int or ground['schema_version'] != 1
            or ground['source_sha256'] != binding['source_sha256']
            or ground['calibration_sha256'] != binding['calibration_sha256']
            or ground['world_axes'] != 'right_handed_z_up' or ground['units'] != 'm'
            or ground['quality'] not in ('reconstructed', 'measured', 'authored')):
        raise ValueError('Ground must match the source, calibration and horizontal world coordinate contract')
    height = ground['height_m']
    if type(height) not in (int, float) or not math.isfinite(height) or abs(height) > 1000:
        raise ValueError('Supplied ground height must be finite within the camera-world envelope')
    if not isinstance(ground['provenance'], str) or not 1 <= len(ground['provenance'].strip()) <= 2000:
        raise ValueError('Supplied ground requires explicit provenance')
    return ground


def validate_ground_binding(spec):
    binding = spec['camera_reference']
    ground = validate_ground(binding['ground'], binding)
    mobility = spec.get('mobility')
    expected = binding['origin_world_m'][2]-ground['height_m']
    if (not isinstance(mobility, dict) or 'floor_fit' in mobility or 'ground_plane' in mobility
            or type(mobility.get('base_height')) not in (int, float)
            or not math.isfinite(mobility['base_height'])
            or not math.isclose(mobility['base_height'], expected, abs_tol=1e-10, rel_tol=0)):
        raise ValueError('Supplied ground must preserve its exact camera-world height without pose fitting')


def apply_camera_ground(spec, ground):
    from weft.human import validate_human

    original = validate_human(spec)
    binding = original.get('camera_reference')
    if binding is None or 'ground' in binding:
        raise ValueError('Ground placement requires an original camera-body candidate without supplied ground')
    validate_ground(ground, binding)
    height = binding['origin_world_m'][2]-ground['height_m']
    if not .01 <= height <= 5:
        raise ValueError('First body root must remain 0.01–5 model units above the supplied ground')
    result = deepcopy(original)
    result['camera_reference']['ground'] = deepcopy(ground)
    result['mobility']['base_height'] = float(height)
    result['mobility'].pop('floor_fit', None)
    result['assumptions'] = [value for value in result['assumptions'] if not any(
        text in value.lower() for text in ('floor is aligned only', 'floor offset uses only', 'absolute world origin is removed'))]
    result['assumptions'].extend([
        'Supplied horizontal ground fixes the world-to-simulation vertical offset; no pose, floor score or clearance is fitted',
        'Root displacement starts at zero; body x/y are relative to the first root and height is relative to supplied ground',
        'Calibrated ground placement does not establish body scale, foot geometry or contact timing; gaps and intersections remain visible',
    ])
    return validate_human(result)


def audit_ground_clearance(spec):
    import mujoco
    from weft.human import HumanEnv

    if 'ground' not in spec.get('camera_reference', {}):
        raise ValueError('Ground clearance inspection requires explicit supplied camera-world ground')
    env = HumanEnv(spec)
    rows = []
    try:
        data = mujoco.MjData(env.model)
        for index, reference in enumerate(spec['reference']):
            data.qpos[env._q_array] = [reference['joints'][name] for name in env.joint_names]
            data.qpos[env.mobility.q] = spec['root_reference'][index]['translation']
            mujoco.mj_forward(env.model, data)
            surfaces = []
            for geom in range(env.model.ngeom):
                if not env.model.geom_contype[geom] & 2:
                    continue
                height = data.geom_xpos[geom, 2]-env.model.geom_size[geom, 0]
                kind = env.model.geom_type[geom]
                if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    height -= abs(data.geom_xmat[geom].reshape(3, 3)[2, 2])*env.model.geom_size[geom, 1]
                elif kind != mujoco.mjtGeom.mjGEOM_SPHERE:
                    raise ValueError('Ground gap audit supports only the current native capsule/sphere body')
                surfaces.append({'geom': mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom),
                                 'signed_clearance': float(height)})
            if not surfaces:
                raise ValueError('Ground gap audit requires collision-enabled body geometry')
            lowest = min(surfaces, key=lambda row: row['signed_clearance'])
            rows.append({'time_s': reference['time_s'], 'evidence_id': spec['reference_evidence'][index]['evidence_id'],
                         'minimum_signed_clearance': lowest['signed_clearance'], 'lowest_geom': lowest['geom'],
                         'surfaces': surfaces})
    finally:
        env.close()
    values = np.array([row['minimum_signed_clearance'] for row in rows])
    return {'method': 'native-body-ground-clearance-v1', 'placement_method': METHOD, 'status': 'descriptive_no_contact_evidence',
            'scope': 'Signed native collision-surface height above supplied ground, not contact timing or a physical pass',
            'ground': deepcopy(spec['camera_reference']['ground']), 'checks': rows,
            'summary': {'reference_poses': len(rows), 'minimum': float(values.min()),
                        'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)),
                        'maximum': float(values.max()), 'positive_gap_poses': int(np.sum(values > 0)),
                        'penetrating_poses': int(np.sum(values < 0)), 'physics_steps': 0}}
