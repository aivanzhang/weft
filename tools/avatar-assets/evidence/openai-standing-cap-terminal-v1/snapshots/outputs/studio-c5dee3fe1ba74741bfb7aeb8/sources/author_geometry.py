"""Read-only proposal diagnostics; native finishing remains authoritative."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from avatar_assets import three_d
from avatar_assets.safety import require
from scripts.creature_rig_math import validate_rig

VERSION = 'author-geometry-diagnostics-v1'
MAX_VERTICES = 100000
MAX_LAYERS = 32


def _unknown_roles(design, names):
    result = []
    bindings = design.get('action_bindings', [])
    require(isinstance(bindings, list) and len(bindings) <= 8, 'Unbounded action bindings')
    for item in bindings:
        require(isinstance(item, dict), 'Invalid action binding')
        state, roles = item.get('state_id'), item.get('roles')
        require(isinstance(state, str) and len(state) <= 64 and isinstance(roles, list)
                and len(roles) <= 4, 'Invalid bounded state roles')
        for role in roles:
            require(isinstance(role, dict), 'Invalid action role')
            label, root, bones = role.get('id'), role.get('root_bone'), role.get('bones')
            require(isinstance(label, str) and len(label) <= 64 and isinstance(root, str)
                    and len(root) <= 64 and isinstance(bones, list) and len(bones) <= 64
                    and all(isinstance(name, str) and len(name) <= 64 for name in bones),
                    'Invalid bounded role bone references')
            for field, values in [('root_bone', [root]), ('bones', bones)]:
                missing = sorted(set(values)-names)
                if missing:
                    result.append({'state_id': state, 'role_id': label, 'field': field,
                                   'unknown_names': missing})
    return result


def _geometry(source, expected_hash, native):
    glb = three_d._GLB(source)
    require(glb.hash == expected_hash, 'Source hash differs from expected source')
    require(isinstance(native, dict) and native.get('source_sha256') == glb.hash
            and native.get('source_unchanged') is True, 'Native inspection source binding mismatch')
    guide = native.get('rig_guides', {})
    require(isinstance(guide, dict) and guide.get('source_sha256') == glb.hash
            and guide.get('space') == 'normalized_bounds_blender_z_up',
            'Native normalized guide binding is unavailable')
    doc = glb.doc
    require(not doc.get('skins') and not doc.get('animations') and not native.get('rigs'),
            'Only an unrigged static source has supported geometry correspondence')
    nodes, meshes, scenes = doc.get('nodes', []), doc.get('meshes', []), doc.get('scenes', [])
    require(len(nodes) == len(meshes) == len(scenes) == 1 and doc.get('scene', 0) == 0
            and scenes[0].get('nodes') == [0] and nodes[0].get('mesh') == 0
            and not nodes[0].get('children') and 'skin' not in nodes[0],
            'Geometry diagnostics require one mesh instance in one root scene')
    matrix = three_d._world_matrices(nodes, {})[0]
    require(tuple(matrix) == three_d._IDENTITY, 'Nonidentity source transforms have unverified native correspondence')
    primitives = meshes[0].get('primitives', [])
    require(len(primitives) == 1 and not meshes[0].get('weights'),
            'Multiple primitives or morph weights have ambiguous vertex correspondence')
    primitive = primitives[0]
    require(isinstance(primitive, dict) and primitive.get('mode', 4) == 4
            and not primitive.get('targets'), 'Only unmorphed triangle geometry is supported')
    attrs = primitive.get('attributes', {})
    require(isinstance(attrs, dict) and not any(name.startswith(('JOINTS_', 'WEIGHTS_')) for name in attrs),
            'Source skin attributes are unsupported')
    points = np.asarray(glb.accessor(attrs.get('POSITION'), 'VEC3', (5126,)), dtype=float)
    require(1 <= len(points) <= MAX_VERTICES, 'Source vertex count exceeds bound')
    if 'indices' in primitive:
        indices = glb.accessor(primitive['indices'], 'SCALAR', (5121, 5123, 5125),
                               count_limit=three_d.MAX_TRIANGLE_INDICES)
        require(len(indices) % 3 == 0 and all(type(row[0]) is int and 0 <= row[0] < len(points) for row in indices),
                'Invalid source triangle indices')
        require(len({row[0] for row in indices}) == len(points), 'Unreferenced vertices have ambiguous native correspondence')
        faces = len(indices)//3
    else:
        require(len(points) % 3 == 0, 'Nonindexed triangles require complete triples')
        faces = len(points)//3
    observed = native.get('meshes', [])
    require(isinstance(observed, list) and len(observed) == 1 and isinstance(observed[0], dict) and observed[0].get('vertices') == len(points)
            and observed[0].get('faces') == faces, 'Source/native geometry counts differ')
    points = points[:, [0, 2, 1]] * np.array([1, -1, 1])
    low, high = points.min(axis=0), points.max(axis=0)
    require(np.isfinite(points).all() and np.all(high-low > 1e-8), 'Degenerate source bounds')
    bounds = {'min': low.tolist(), 'max': high.tolist()}
    require(native.get('bounds_blender_z_up') == bounds and guide.get('bounds_blender_z_up') == bounds,
            'Source/native exact bounds differ; correspondence unavailable')
    return points, low, high, {'source_sha256': glb.hash, 'vertices': len(points), 'faces': faces,
        'bounds_blender_z_up': bounds, 'profile': 'single-identity-node-unrigged-triangles',
        'axis_conversion': 'glTF (x,y,z) -> Blender (x,-z,y)',
        'agreement': 'Exact native source hash, vertex/face counts and world bounds'}


def _coverage(points, low, high, rig):
    # Match native calibration before normalization, including floating-point boundaries.
    factor = rig['height_meters']/(high[2]-low[2])
    offset = np.array([(low[0]+high[0])/2, (low[1]+high[1])/2, low[2]])
    points = (points-offset)*factor
    low, high = points.min(axis=0), points.max(axis=0)
    normalized = (points-low)/(high-low)
    rows = []
    for index, layer in enumerate(rig['layers']):
        edge = np.minimum(normalized-layer['min'], np.asarray(layer['max'])-normalized).min(axis=1)
        # Subnormal smoothstep values can round to zero; do not infer their membership.
        if layer['feather']:
            require(not np.any((edge > 0) & (edge < np.sqrt(np.finfo(float).tiny)*layer['feather'])),
                    'Region boundary is too close to numerical underflow for exact diagnostics')
        # Away from underflow, smoothstep is positive exactly when its input is positive.
        count = int(np.count_nonzero(edge > 0 if layer['feather'] else edge >= 0))
        rows.append({'layer_index': index, 'bones': list(layer['bones']), 'min': list(layer['min']),
                     'max': list(layer['max']), 'feather': layer['feather'], 'selected_vertices': count,
                     'min_vertices': layer['min_vertices'], 'passes': count >= layer['min_vertices']})
    return rows


def author_diagnostics(design: dict, source_path: Path, source_sha256: str, native_inspection: dict) -> dict:
    """Return bounded findings without changing a proposal, source, or native threshold."""
    result = {'schema_version': 1, 'version': VERSION, 'status': 'unavailable',
              'unknown_bone_references': [], 'layers': [], 'failures': [],
              'limits': {'vertices': MAX_VERTICES, 'layers': MAX_LAYERS},
              'scope': 'Authored role references and region occupancy only; not skin quality or animation acceptance'}
    try:
        require(isinstance(design, dict), 'Decoded author design must be an object')
        rig = design.get('authored_rig')
        if rig is None:
            result.update(status='not_applicable', reason='No newly authored rig')
            return result
        validate_rig(rig)
        names = {bone['name'] for bone in rig['bones']}
        result['available_bones'] = sorted(names)
        result['unknown_bone_references'] = _unknown_roles(design, names)
        for row in result['unknown_bone_references']:
            result['failures'].append({'kind': 'unknown_role_bones', **row})
        points, low, high, proof = _geometry(source_path, source_sha256, native_inspection)
        result['geometry'] = proof
        result['layers'] = _coverage(points, low, high, rig)
        result['failures'].extend({'kind': 'region_undercoverage', **row} for row in result['layers'] if not row['passes'])
        result['status'] = 'available'
    except (ValueError, TypeError, KeyError, IndexError, OSError, OverflowError) as error:
        result['reason'] = str(error)[:512]
    return result
